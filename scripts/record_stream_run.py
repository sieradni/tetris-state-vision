"""Verify/replay finite stream ledgers from their archived source and curate evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from probe_model import require_bounded_job
from source_snapshot import verify_source

ROOT = Path(__file__).resolve().parents[1]


def validate_completion(summary, job, *, allow_partial=False):
    partial = summary.get("status") in {"wall_partial", "step_partial"}
    steps, target = summary.get("steps"), summary.get("requested_steps")
    if (type(steps) is not int or type(target) is not int or not 1 <= steps <= target <= 4096 or
            not summary.get("checkpoint_round_trip_exact")):
        raise ValueError("finite saved checkpoint and exact reload required")
    if partial:
        if not allow_partial or steps >= target or job.get("status") != "failed" or job.get("exit_code") != 2:
            raise ValueError("explicit controlled partial with saved state required")
    elif summary.get("status") != "complete" or steps != target or job.get("status") != "passed" or job.get("exit_code") != 0:
        raise ValueError("complete native run required")
    updates = summary.get("updates_this_run", steps)
    if type(updates) is not int or not 0 < updates <= job.get("work_items", 0):
        raise ValueError("new-update count exceeds declared cap")
    return partial


def digest(path, maximum=64*1024**2):
    value, consumed = hashlib.sha256(), 0
    with path.open("rb") as data:
        for chunk in iter(lambda: data.read(65536), b""):
            consumed += len(chunk)
            if consumed > maximum:
                raise ValueError("evidence file byte budget exceeded")
            value.update(chunk)
    return value.hexdigest()


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--allow-partial", action="store_true", help="preserve controlled wall/step partial status; never a completion claim")
    args = parser.parse_args()
    if not re.fullmatch(r"m[23]-stream-[a-z0-9-]{1,40}", args.name):
        raise ValueError("owned stream run name required")
    run = ROOT/"runs"/args.name
    summary = json.loads((run/"summary.json").read_text())
    job = json.loads((ROOT/"runs"/(args.name+"-job.json")).read_text())
    partial = validate_completion(summary, job, allow_partial=args.allow_partial)
    code = run/"code"
    archive = verify_source(code, summary["catalogue_sha256"])
    sys.path[:0] = [str(code/"scripts"), str(code)]
    sys.dont_write_bytecode = True
    if summary.get("resource_profile_version", 0) >= 2 and summary["profile_square"]:
        from train_m2_stream import validate_admission
        validate_admission(summary, job, catalogue=archive["catalogue_sha256"],
                           model=summary["model_kind"], config=summary["config"],
                           ceiling=summary["job_ceiling_bytes"], development_groups=summary["development_groups"],
                           supervisor_sha=archive["files"]["scripts/run_bounded.py"])
    from PIL import Image
    from m2_stream_data import sample_case
    if not Path(sample_case.__code__.co_filename).resolve().is_relative_to(code):
        raise ValueError("sampler did not import from archived source")
    manifest = json.loads((run/"checkpoint/manifest.json").read_text())
    if digest(run/"checkpoint/weights.pt") != summary["checkpoint_sha256"] or manifest["weights_sha256"] != summary["checkpoint_sha256"]:
        raise ValueError("weight checksum mismatch")
    recovery_manifest = None
    resume_parent = None
    resume_files = []
    if summary.get("resource_profile_version", 0) >= 3:
        pointer = json.loads((run/"latest-training.json").read_text())
        generation = pointer.get("generation", "")
        if (not re.fullmatch(r"step-[0-9]{6}-[0-9a-f]{32}", generation) or
                pointer != summary["latest_training_snapshot"] or pointer != manifest["metadata"]["latest_training_snapshot"]):
            raise ValueError("training recovery pointer identity differs")
        folder = run/"training-snapshots"/generation
        if not folder.resolve().is_relative_to(run) or digest(folder/"manifest.json", 64*1024) != pointer["manifest_sha256"]:
            raise ValueError("training recovery manifest identity differs")
        recovery_manifest = json.loads((folder/"manifest.json").read_text())
        if (recovery_manifest["step"] != summary["steps"] or recovery_manifest["contract"] != summary["recovery_contract"] or
                digest(folder/"tensors.pt", 256*1024**2) != recovery_manifest["tensors_sha256"]):
            raise ValueError("training recovery artifact/recipe identity differs")
        if set(recovery_manifest["ledgers"]) != {"curve.jsonl", "training.jsonl", "evaluations.jsonl"}:
            raise ValueError("training recovery ledger set differs")
        for name, cursor in recovery_manifest["ledgers"].items():
            if (run/name).stat().st_size != cursor["bytes"] or digest(run/name, 16*1024**2) != cursor["sha256"]:
                raise ValueError("complete-run recovery ledger prefix differs")
        if not 0 < summary["updates_this_run"] <= job["work_items"]:
            raise ValueError("native new-update count exceeds declared segment cap")
        if summary["resume"] is not None:
            resume = summary["resume"]
            parent = Path(resume["source_run"]).resolve()
            if not parent.is_relative_to(ROOT/"runs"):
                raise ValueError("resume evidence parent escaped owned runs")
            verify_source(parent/"code", summary["catalogue_sha256"])
            parent_pointer = resume["pointer"]
            if not re.fullmatch(r"step-[0-9]{6}-[0-9a-f]{32}", parent_pointer["generation"]):
                raise ValueError("resume parent generation invalid")
            parent_folder = parent/"training-snapshots"/parent_pointer["generation"]
            if not parent_folder.resolve().is_relative_to(parent) or digest(parent_folder/"manifest.json", 64*1024) != parent_pointer["manifest_sha256"]:
                raise ValueError("resume parent manifest identity differs")
            parent_manifest = json.loads((parent_folder/"manifest.json").read_text())
            if (parent_manifest["step"] != resume["step"] or parent_manifest["contract"] != summary["recovery_contract"] or
                    resume["step"]+summary["updates_this_run"] != summary["steps"] or
                    digest(parent_folder/"tensors.pt", 256*1024**2) != parent_manifest["tensors_sha256"] or
                    set(parent_manifest["ledgers"]) != {"curve.jsonl", "training.jsonl", "evaluations.jsonl"}):
                raise ValueError("resume parent tensor/step/recipe identity differs")
            for name, cursor in parent_manifest["ledgers"].items():
                size = cursor["bytes"]
                if type(size) is not int or not 0 <= size <= 16*1024**2 or cursor["sha256"] != resume["ledger_prefix_sha256"][name]:
                    raise ValueError("resume prefix cursor differs")
                for file in (parent/name, run/name):
                    with file.open("rb") as stream:
                        prefix = stream.read(size)
                    if len(prefix) != size or hashlib.sha256(prefix).hexdigest() != cursor["sha256"]:
                        raise ValueError("parent/resumed committed ledger prefix differs")
            parent_job = json.loads((parent.parent/(parent.name+"-job.json")).read_text())
            resume_parent = {"step": resume["step"], "run": str(parent), "native_status": parent_job["status"],
                             "native_exit_code": parent_job["exit_code"], "manifest_sha256": parent_pointer["manifest_sha256"],
                             "tensors_sha256": parent_manifest["tensors_sha256"]}
            resume_files = [(parent_folder/"manifest.json", "resume-parent-snapshot-manifest.json"),
                            (parent/"code/source.json", "resume-parent-source-manifest.json")]
            for suffix in ("-job.json", ".log", ".stderr.log"):
                resume_files.append((parent.parent/(parent.name+suffix), "resume-parent"+suffix))
            enclosing = parent.parent/(parent.name+"-pipeline-job.json")
            if enclosing.exists():
                if enclosing.stat().st_size > 64*1024:
                    raise ValueError("resume enclosing receipt budget exceeded")
                enclosing_job = json.loads(enclosing.read_text())
                resume_parent.update(enclosing_native_status=enclosing_job["status"],
                                     enclosing_native_exit_code=enclosing_job["exit_code"],
                                     child_receipt_stale_after_ancestor_stop=parent_job["exit_code"] is None and enclosing_job["exit_code"] is not None)
                for suffix in ("-job.json", ".log", ".stderr.log"):
                    resume_files.append((parent.parent/(parent.name+"-pipeline"+suffix), "resume-parent-enclosing"+suffix))
            if (parent/"summary.json").exists():
                resume_files.append((parent/"summary.json", "resume-parent-summary.json"))
    ledger = run/"training.jsonl"
    if digest(ledger, 16*1024**2) != summary["stream_ledger_sha256"] or not 1 <= summary["steps"] <= 4096:
        raise ValueError("stream ledger budget/identity mismatch")
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    if [row["step"] for row in rows] != list(range(1, summary["steps"]+1)):
        raise ValueError("stream ledger incomplete")
    for row in rows:
        if bool(row.get("final_capture", False)) != bool(summary.get("component_capacity", 0)):
            raise ValueError("stream capture policy differs from joint model contract")
        extra = {"final_capture": True} if row.get("final_capture") else {}
        replay = sample_case(row["group_index"], row["view"], row["split"], row["base_seed"], profile_square=row["profile_square"], **extra)
        if {key: value for key, value in row.items() if key != "step"} != json.loads(json.dumps(replay.record)):
            raise ValueError("stream source does not reproduce the recorded pixels/labels/recipe")
    curve = [json.loads(line) for line in (run/"curve.jsonl").read_text().splitlines()]
    if [row["sample_id"] for row in curve] != [row["sample_id"] for row in rows]:
        raise ValueError("optimizer steps and source ledger differ")
    probe_images = 0
    for split in ("training", "development"):
        for line in (run/(split+"-probe.jsonl")).read_text().splitlines():
            row = json.loads(line)
            if bool(row.get("final_capture", False)) != bool(summary.get("component_capacity", 0)):
                raise ValueError("probe capture policy differs from joint model contract")
            extra = {"final_capture": True} if row.get("final_capture") else {}
            replay = sample_case(row["group_index"], row["view"], row["split"], row["base_seed"], **extra)
            if json.loads(json.dumps(replay.record)) != row:
                raise ValueError("probe metadata/pixels changed")
            folder = run/(split+"-probe")
            with Image.open(folder/(row["sample_id"]+".png")) as stored:
                if stored.mode != replay.frame.image.mode or stored.size != replay.frame.image.size or stored.tobytes() != replay.frame.image.tobytes():
                    raise ValueError("saved probe raster changed")
            if digest(folder/(row["sample_id"]+".json")) != row["state_sha256"]:
                raise ValueError("saved probe annotation changed")
            probe_images += 1
    evidence = ROOT/"docs/evidence"/args.name
    evidence.mkdir(exist_ok=False)
    for name in ("summary.json", "progress.json", "training.jsonl", "curve.jsonl", "evaluations.jsonl", "training-probe.jsonl", "development-probe.jsonl"):
        shutil.copyfile(run/name, evidence/name)
    if summary.get("resource_profile_version", 0) >= 3:
        if (run/"memory.jsonl").stat().st_size > 1024**2:
            raise ValueError("memory trajectory byte budget exceeded")
        shutil.copyfile(run/"memory.jsonl", evidence/"memory.jsonl")
    shutil.copyfile(code/"source.json", evidence/"source-manifest.json")
    shutil.copyfile(run/"checkpoint/manifest.json", evidence/"checkpoint-manifest.json")
    if recovery_manifest is not None:
        shutil.copyfile(run/"latest-training.json", evidence/"latest-training.json")
        (evidence/"training-snapshot-manifest.json").write_text(json.dumps(recovery_manifest, indent=2)+"\n")
    for source, name in resume_files:
        shutil.copyfile(source, evidence/name)
    for suffix in ("-job.json", ".log", ".stderr.log"):
        source = ROOT/"runs"/(args.name+suffix)
        shutil.copyfile(source, evidence/source.name)
    comparison_status = None
    if (run/"analysis.json").exists():
        if (run/"analysis.json").stat().st_size > 64*1024:
            raise ValueError("comparison analysis byte budget exceeded")
        comparison = json.loads((run/"analysis.json").read_text())
        if comparison["weights_sha256"] != summary["checkpoint_sha256"] or comparison["source_catalogue_sha256"] != summary["catalogue_sha256"]:
            raise ValueError("comparison is for another artifact/source")
        shutil.copyfile(run/"analysis.json", evidence/"analysis.json")
        comparison_status = comparison["status"]
    scope = {"status": "verified_partial" if partial else "passed", "run_status": summary["status"],
             "steps": summary["steps"], "requested_steps": summary["requested_steps"],
             "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "source_catalogue_sha256": archive["catalogue_sha256"], "weights_sha256": summary["checkpoint_sha256"],
             "replayed_training_samples": len(rows), "verified_saved_probe_images": probe_images,
             "native_peak_job_commit_bytes": job["peak_job_commit_bytes"], "all_final_tensors_reload_exact": True,
             "source_archive": str(code), "scope": "Bounded curriculum plumbing/source binding/resource admission, not model accuracy.",
             "test_gate_scope": "Separate named, source-pinned fixture receipts; this curator does not attach historical fixture logs to new runs.",
             "training_snapshot_identity_verified": recovery_manifest is not None,
             "comparison_status": comparison_status,
             "updates_this_run": summary.get("updates_this_run", summary["steps"]), "resume_parent": resume_parent,
             "curator_sha256": digest(Path(__file__)), "core_architecture": "Unchanged M2; pilot and accuracy configurations explicit."}
    (evidence/"scope.json").write_text(json.dumps(scope, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": scope["status"], "run_status": summary["status"], "steps": summary["steps"],
                      "requested_steps": summary["requested_steps"], "replayed_samples": len(rows),
                      "verified_probe_images": probe_images, "scope": str(evidence/"scope.json")}))


if __name__ == "__main__":
    main()
