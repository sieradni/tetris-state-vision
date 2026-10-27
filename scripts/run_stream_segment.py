"""One admitted finite CPU segment followed by explicit-status evidence replay.

Parent:1.25GiB,2100s,at most512 optimizer updates. Serialized children:
training640MiB or1GiB from the verified profile/1860s (1800s inner),
curation128MiB/120s. No automatic restart.
This starts immediately; it is not a scheduled/recurring task or notification.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

from atomic_write import replace_retry
from probe_model import require_bounded_job
from probe_stream_recovery import verify_dead
from record_stream_run import checkpoint_architecture, validate_completion
from source_snapshot import verify_source
from train_m2_stream import NATIVE_CACHE_LIMITS, validate_admission

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    if path.stat().st_size > 1024**2:
        raise ValueError("segment JSON budget exceeded")
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    replace_retry(temporary, path)


def require_controller_source(expected):
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != expected:
        raise ValueError("controller source changed during execution; no verified result")


def validate_recipe(recipe):
    for key, low, high in (("seed", 0, 2**31-1), ("evaluate_every", 1, 1024),
                           ("checkpoint_every", 1, 1024), ("development_groups", 3, 12)):
        if type(recipe[key]) is not int or not low <= recipe[key] <= high:
            raise ValueError("resume recipe outside supported finite bounds")
    rate = recipe["learning_rate"]
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 < rate <= .01:
        raise ValueError("resume learning rate outside finite bounds")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--target-steps", type=int, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--component-capacity", type=int, help="optional assertion; default derives actual profile/resume capacity")
    parser.add_argument("--plan-only", action="store_true", help="verify and print the finite plan without output creation or child launch")
    args = parser.parse_args(argv)
    if (not re.fullmatch(r"m[23]-stream-[a-z0-9-]{1,40}", args.name) or
            not 1 <= args.updates <= 512 or not 1 <= args.target_steps <= 4096 or
            args.component_capacity is not None and args.component_capacity != 0 and not 10 <= args.component_capacity <= 64):
        parser.error("owned name, target1..4096 and update cap1..512 required")
    return args


def segment_plan(args):
    """Read verified identities/recipe before creating output or launching work."""
    admission = args.admission.resolve()
    if not admission.is_relative_to(ROOT/"runs"):
        raise ValueError("owned local resource profile required")
    profile = read(admission/"summary.json")
    profile_job = read(admission.parent/(admission.name+"-job.json"))
    archive = verify_source(admission/"code", profile["catalogue_sha256"])
    profile_manifest = read(admission/"checkpoint/manifest.json")
    architecture = checkpoint_architecture(profile, profile_manifest)
    capacity = profile["config"].get("component_capacity", 0)
    if (capacity != 0 and not 10 <= capacity <= 64 or
            not args.name.startswith("m3-stream-" if capacity else "m2-stream-") or
            args.component_capacity is not None and args.component_capacity != capacity):
        raise ValueError("run name/request differs from actual component capacity")
    ceiling = profile["job_ceiling_bytes"]
    if type(ceiling) is not int or ceiling not in (640*1024**2, 1024*1024**2):
        raise ValueError("profile must fit a supported640MiB/1GiB child envelope")
    if profile["model_kind"] not in {"pilot", "accuracy"}:
        raise ValueError("supported profile model configuration required")
    start, code = 0, admission/"code"
    recipe = {"seed": 20261026, "learning_rate": .001, "evaluate_every": 128,
              "checkpoint_every": 64, "development_groups": profile["development_groups"]}
    run = ROOT/"runs"/args.name
    pipeline = ROOT/"runs"/(args.name+"-pipeline")
    if run.exists() or pipeline.exists():
        raise FileExistsError("training and pipeline outputs must be new")
    if args.resume_from is not None:
        parent = args.resume_from.resolve()
        if not parent.is_relative_to(ROOT/"runs"):
            raise ValueError("owned local resume source required")
        verify_source(parent/"code", archive["catalogue_sha256"])
        pointer = read(parent/"latest-training.json")
        if not re.fullmatch(r"step-[0-9]{6}-[0-9a-f]{32}", pointer["generation"]):
            raise ValueError("resume generation invalid")
        manifest_path = parent/"training-snapshots"/pointer["generation"]/"manifest.json"
        if not manifest_path.resolve().is_relative_to(parent) or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != pointer["manifest_sha256"]:
            raise ValueError("resume manifest identity differs")
        manifest = read(manifest_path)
        contract = manifest["contract"]
        if (contract["steps"] != args.target_steps or contract["catalogue_sha256"] != archive["catalogue_sha256"] or
                contract["config"] != profile["config"] or contract["profile_square"] is not False or
                contract["weight_decay"] != .01 or contract["max_side"] != 512 or contract["threads"] != 1 or
                contract["rng"] != "torch_cpu_only" or contract["native_cache_limits"] != NATIVE_CACHE_LIMITS or
                capacity and contract.get("input_policy") != "M3-final-capture-v1" or
                not capacity and contract.get("input_policy", "M2-prepared-resize-v1") != "M2-prepared-resize-v1"):
            raise ValueError("resume target/source differs; finished diagnostics cannot silently extend")
        start = manifest["step"]
        code = parent/"code"
        recipe = {key: contract[key] for key in recipe}
    if type(start) is not int or not 0 <= start < args.target_steps:
        raise ValueError("target already complete or resume cursor invalid")
    validate_recipe(recipe)
    supervisor_sha = hashlib.sha256((ROOT/"scripts/run_bounded.py").read_bytes()).hexdigest()
    if archive["files"]["scripts/run_bounded.py"] != supervisor_sha:
        raise ValueError("archived supervisor differs from live guard")
    validate_admission(profile, profile_job, catalogue=archive["catalogue_sha256"], model=profile["model_kind"],
                       config=profile["config"], ceiling=ceiling, development_groups=recipe["development_groups"],
                       supervisor_sha=supervisor_sha)
    budget = min(args.updates, args.target_steps-start)
    trainer = code/"scripts/train_m2_stream.py"
    if not trainer.is_file():
        raise ValueError("verified archived trainer missing")
    worker = [sys.executable, str(trainer), "--output", str(run), "--model", profile["model_kind"], "--steps", str(args.target_steps),
              "--max-new-steps", str(budget), "--wall-seconds", "1800", "--admission", str(admission)]
    for key, flag in (("seed", "--seed"), ("learning_rate", "--learning-rate"), ("evaluate_every", "--evaluate-every"),
                      ("checkpoint_every", "--checkpoint-every"), ("development_groups", "--development-groups")):
        worker += [flag, str(recipe[key])]
    if capacity:
        worker += ["--component-capacity", str(capacity)]
    if args.resume_from is not None:
        worker += ["--resume-from", str(args.resume_from.resolve())]
    return {"run": run, "pipeline": pipeline, "start": start, "budget": budget, "worker": worker,
            "memory_mib": ceiling//1024**2, "component_capacity": capacity, "config": profile["config"],
            "model_kind": profile["model_kind"], "catalogue_sha256": archive["catalogue_sha256"],
            "architecture": architecture, "development_groups": recipe["development_groups"], "target_steps": args.target_steps}


def validate_segment(summary, job, manifest, plan):
    validate_completion(summary, job, allow_partial=True)
    architecture = checkpoint_architecture(summary, manifest)
    if (summary["requested_steps"] != plan["target_steps"] or summary["steps"] != plan["start"]+summary["updates_this_run"] or
            not 0 < summary["updates_this_run"] <= plan["budget"] or job["work_items"] != plan["budget"] or
            job["memory_bytes"] != plan["memory_mib"]*1024**2 or summary["config"] != plan["config"] or
            summary["catalogue_sha256"] != plan["catalogue_sha256"] or summary["model_kind"] != plan["model_kind"] or
            architecture != plan["architecture"] or summary.get("component_capacity", 0) != plan["component_capacity"]):
        raise ValueError("saved segment cursor/configuration/resource identity differs")
    probes = 0
    for field, expected in (("final_training_probe", 9), ("development", 3*plan["development_groups"])):
        if (summary[field]["images"] != expected or len(summary[field]["cases"]) != expected or
                len({row["sample_id"] for row in summary[field]["cases"]}) != expected):
            raise ValueError("saved probe count differs from declared population")
        probes += expected
    return probes


def main(argv=None):
    require_bounded_job()
    controller_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args = arguments(argv)
    plan = segment_plan(args)
    if args.plan_only:
        require_controller_source(controller_sha)
        print(json.dumps({"status": "plan_verified", "controller_sha256": controller_sha, **plan}, default=str), flush=True)
        return 0
    pipeline, run = plan["pipeline"], plan["run"]
    pipeline.mkdir(exist_ok=False)
    training_receipt = ROOT/"runs"/(args.name+"-job.json")
    training = [sys.executable, str(ROOT/"scripts/run_bounded.py"), "--memory-mib", str(plan["memory_mib"]), "--wall-seconds", "1860",
                "--work-items", str(plan["budget"]), "--receipt", str(training_receipt), "--", *plan["worker"]]
    with (ROOT/"runs"/(args.name+".log")).open("xb") as log, (ROOT/"runs"/(args.name+".stderr.log")).open("xb") as stderr:
        process = subprocess.Popen(training, stdout=log, stderr=stderr, cwd=ROOT)
        write(pipeline/"progress.json", {"phase": "training", "supervisor_pid": process.pid, "command": training,
                                        "start_step": plan["start"], "target_steps": args.target_steps, "new_update_cap": plan["budget"],
                                        "component_capacity": plan["component_capacity"], "model_kind": plan["model_kind"]})
        print(json.dumps({"phase": "training", "supervisor_pid": process.pid, "command": training}), flush=True)
        code = process.wait(timeout=1890)
    job = read(training_receipt)
    if code not in (0, 2) or job["status"] != ("passed" if code == 0 else "failed") or job["exit_code"] != code:
        raise RuntimeError(f"training segment did not save a controlled result: {job}")
    summary, progress = read(run/"summary.json"), read(run/"progress.json")
    for pid in (process.pid, job["supervisor_pid"], job["child_pid"], progress["pid"]):
        verify_dead(pid)
    manifest = read(run/"checkpoint/manifest.json")
    probes = validate_segment(summary, job, manifest, plan)
    curation = [sys.executable, str(ROOT/"scripts/run_bounded.py"), "--memory-mib", "128", "--wall-seconds", "120",
                "--work-items", str(summary["steps"]+probes), "--receipt", str(pipeline/"curation-job.json"), "--",
                sys.executable, str(ROOT/"scripts/record_stream_run.py"), "--name", args.name, "--allow-partial"]
    with (pipeline/"curation.log").open("xb") as log, (pipeline/"curation.stderr.log").open("xb") as stderr:
        process = subprocess.Popen(curation, stdout=log, stderr=stderr, cwd=ROOT)
        write(pipeline/"progress.json", {"phase": "curation", "supervisor_pid": process.pid, "command": curation})
        if process.wait(timeout=135) != 0:
            raise RuntimeError("segment evidence replay failed; preserve logs and state")
    verify_dead(process.pid)
    evidence = read(ROOT/"docs/evidence"/args.name/"scope.json")
    expected = "passed" if summary["status"] == "complete" else "verified_partial"
    if (evidence["status"] != expected or evidence["steps"] != summary["steps"] or
            evidence["source_catalogue_sha256"] != plan["catalogue_sha256"] or evidence["weights_sha256"] != summary["checkpoint_sha256"] or
            evidence["core_architecture"] != manifest["architecture"] or evidence["component_capacity"] != plan["component_capacity"]):
        raise ValueError("curated status/step differs")
    require_controller_source(controller_sha)
    result = {"status": "segment_verified", "training_status": summary["status"], "global_steps": summary["steps"],
              "target_steps": args.target_steps, "updates_this_segment": summary["updates_this_run"],
              "target_complete": summary["steps"] == args.target_steps, "weights_sha256": summary["checkpoint_sha256"],
              "source_catalogue_sha256": summary["catalogue_sha256"], "evidence_status": evidence["status"],
              "core_architecture": manifest["architecture"], "component_capacity": plan["component_capacity"],
              "model_kind": summary["model_kind"], "verified_probe_images": probes,
              "next_resume_from": str(run) if summary["steps"] < args.target_steps else None,
              "controller_sha256": controller_sha,
              "scope": "One bounded training segment and replayed explicit-status evidence; not full goal, adequate convergence or model quality."}
    write(pipeline/"result.json", result)
    write(pipeline/"progress.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
