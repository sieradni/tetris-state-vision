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
    args = parser.parse_args()
    if not re.fullmatch(r"m2-stream-[a-z0-9-]{1,40}", args.name):
        raise ValueError("owned stream run name required")
    run = ROOT/"runs"/args.name
    summary = json.loads((run/"summary.json").read_text())
    job = json.loads((ROOT/"runs"/(args.name+"-job.json")).read_text())
    if summary["status"] != "complete" or job["status"] != "passed" or not summary["checkpoint_round_trip_exact"]:
        raise ValueError("complete native run and all-head checkpoint parity required")
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
    ledger = run/"training.jsonl"
    if digest(ledger, 16*1024**2) != summary["stream_ledger_sha256"] or not 1 <= summary["steps"] <= 4096:
        raise ValueError("stream ledger budget/identity mismatch")
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    if [row["step"] for row in rows] != list(range(1, summary["steps"]+1)):
        raise ValueError("stream ledger incomplete")
    for row in rows:
        replay = sample_case(row["group_index"], row["view"], row["split"], row["base_seed"], profile_square=row["profile_square"])
        if {key: value for key, value in row.items() if key != "step"} != json.loads(json.dumps(replay.record)):
            raise ValueError("stream source does not reproduce the recorded pixels/labels/recipe")
    curve = [json.loads(line) for line in (run/"curve.jsonl").read_text().splitlines()]
    if [row["sample_id"] for row in curve] != [row["sample_id"] for row in rows]:
        raise ValueError("optimizer steps and source ledger differ")
    probe_images = 0
    for split in ("training", "development"):
        for line in (run/(split+"-probe.jsonl")).read_text().splitlines():
            row = json.loads(line)
            replay = sample_case(row["group_index"], row["view"], row["split"], row["base_seed"])
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
    for suffix in ("-job.json", ".log", ".stderr.log"):
        source = ROOT/"runs"/(args.name+suffix)
        shutil.copyfile(source, evidence/source.name)
    scope = {"status": "passed", "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "source_catalogue_sha256": archive["catalogue_sha256"], "weights_sha256": summary["checkpoint_sha256"],
             "replayed_training_samples": len(rows), "verified_saved_probe_images": probe_images,
             "native_peak_job_commit_bytes": job["peak_job_commit_bytes"], "all_final_tensors_reload_exact": True,
             "source_archive": str(code), "scope": "Bounded curriculum plumbing/source binding/resource admission, not model accuracy.",
             "test_gate_scope": "Separate named, source-pinned fixture receipts; this curator does not attach historical fixture logs to new runs.",
             "training_snapshot_identity_verified": recovery_manifest is not None,
             "curator_sha256": digest(Path(__file__)), "core_architecture": "Unchanged M2; pilot and accuracy configurations explicit."}
    (evidence/"scope.json").write_text(json.dumps(scope, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "replayed_samples": len(rows), "verified_probe_images": probe_images, "scope": str(evidence/"scope.json")}))


if __name__ == "__main__":
    main()
