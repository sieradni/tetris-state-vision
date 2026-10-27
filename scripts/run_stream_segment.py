"""One finite larger-model segment followed by explicit-status evidence replay.

Parent:1.25GiB,2100s,at most512 optimizer updates. Serialized children:
training1GiB/1860s (1800s inner), curation128MiB/120s. No automatic restart.
This starts immediately; it is not a scheduled/recurring task or notification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from atomic_write import replace_retry
from probe_model import require_bounded_job
from probe_stream_recovery import verify_dead
from source_snapshot import verify_source

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    if path.stat().st_size > 1024**2:
        raise ValueError("segment JSON budget exceeded")
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    replace_retry(temporary, path)


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--target-steps", type=int, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"m2-stream-[a-z0-9-]{1,40}", args.name) or not 1 <= args.updates <= 512 or not 1 <= args.target_steps <= 4096:
        parser.error("owned name, target1..4096 and update cap1..512 required")
    pipeline = ROOT/"runs"/(args.name+"-pipeline")
    pipeline.mkdir(exist_ok=False)
    run = ROOT/"runs"/args.name
    if run.exists():
        raise FileExistsError("training output must be new")
    start = 0
    trainer = ROOT/"scripts/train_m2_stream.py"
    if args.resume_from is not None:
        parent = args.resume_from.resolve()
        archive = verify_source(parent/"code")
        pointer = read(parent/"latest-training.json")
        if not re.fullmatch(r"step-[0-9]{6}-[0-9a-f]{32}", pointer["generation"]):
            raise ValueError("resume generation invalid")
        manifest_path = parent/"training-snapshots"/pointer["generation"]/"manifest.json"
        if not manifest_path.resolve().is_relative_to(parent) or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != pointer["manifest_sha256"]:
            raise ValueError("resume manifest identity differs")
        manifest = read(manifest_path)
        if manifest["contract"]["steps"] != args.target_steps or manifest["contract"]["catalogue_sha256"] != archive["catalogue_sha256"]:
            raise ValueError("resume target/source differs; finished diagnostics cannot silently extend")
        start = manifest["step"]
        trainer = parent/"code/scripts/train_m2_stream.py"
    budget = min(args.updates, args.target_steps-start)
    if not 0 < budget <= 512:
        raise ValueError("target already complete or resume cursor invalid")
    worker = [sys.executable, str(trainer), "--output", str(run), "--model", "accuracy", "--steps", str(args.target_steps),
              "--max-new-steps", str(budget), "--wall-seconds", "1800", "--development-groups", "6",
              "--checkpoint-every", "64", "--admission", str(args.admission.resolve())]
    if args.resume_from is not None:
        worker += ["--resume-from", str(args.resume_from.resolve())]
    training_receipt = ROOT/"runs"/(args.name+"-job.json")
    training = [sys.executable, str(ROOT/"scripts/run_bounded.py"), "--memory-mib", "1024", "--wall-seconds", "1860",
                "--work-items", str(budget), "--receipt", str(training_receipt), "--", *worker]
    with (ROOT/"runs"/(args.name+".log")).open("xb") as log, (ROOT/"runs"/(args.name+".stderr.log")).open("xb") as stderr:
        process = subprocess.Popen(training, stdout=log, stderr=stderr, cwd=ROOT)
        write(pipeline/"progress.json", {"phase": "training", "supervisor_pid": process.pid, "command": training,
                                        "start_step": start, "target_steps": args.target_steps, "new_update_cap": budget})
        print(json.dumps({"phase": "training", "supervisor_pid": process.pid, "command": training}), flush=True)
        code = process.wait(timeout=1890)
    job = read(training_receipt)
    if code not in (0, 2) or job["status"] != ("passed" if code == 0 else "failed") or job["exit_code"] != code:
        raise RuntimeError(f"training segment did not save a controlled result: {job}")
    summary, progress = read(run/"summary.json"), read(run/"progress.json")
    for pid in (process.pid, job["supervisor_pid"], job["child_pid"], progress["pid"]):
        verify_dead(pid)
    if (summary["status"] not in {"complete", "step_partial", "wall_partial"} or
            summary["requested_steps"] != args.target_steps or summary["steps"] != start+summary["updates_this_run"] or
            not 0 < summary["updates_this_run"] <= budget or not summary["checkpoint_round_trip_exact"]):
        raise ValueError("saved segment cursor/completion contract differs")
    curation = [sys.executable, str(ROOT/"scripts/run_bounded.py"), "--memory-mib", "128", "--wall-seconds", "120",
                "--work-items", str(summary["steps"]+27), "--receipt", str(pipeline/"curation-job.json"), "--",
                sys.executable, str(ROOT/"scripts/record_stream_run.py"), "--name", args.name, "--allow-partial"]
    with (pipeline/"curation.log").open("xb") as log, (pipeline/"curation.stderr.log").open("xb") as stderr:
        process = subprocess.Popen(curation, stdout=log, stderr=stderr, cwd=ROOT)
        write(pipeline/"progress.json", {"phase": "curation", "supervisor_pid": process.pid, "command": curation})
        if process.wait(timeout=135) != 0:
            raise RuntimeError("segment evidence replay failed; preserve logs and state")
    verify_dead(process.pid)
    evidence = read(ROOT/"docs/evidence"/args.name/"scope.json")
    expected = "passed" if summary["status"] == "complete" else "verified_partial"
    if evidence["status"] != expected or evidence["steps"] != summary["steps"]:
        raise ValueError("curated status/step differs")
    result = {"status": "segment_verified", "training_status": summary["status"], "global_steps": summary["steps"],
              "target_steps": args.target_steps, "updates_this_segment": summary["updates_this_run"],
              "target_complete": summary["steps"] == args.target_steps, "weights_sha256": summary["checkpoint_sha256"],
              "source_catalogue_sha256": summary["catalogue_sha256"], "evidence_status": evidence["status"],
              "next_resume_from": str(run) if summary["steps"] < args.target_steps else None,
              "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "One bounded training segment and replayed explicit-status evidence; not full goal, adequate convergence or model quality."}
    write(pipeline/"result.json", result)
    write(pipeline/"progress.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
