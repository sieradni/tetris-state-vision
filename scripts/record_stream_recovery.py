"""Curate the finite recovery gate, including archived recipe/pixel identities."""
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
from record_stream_run import checkpoint_architecture

ROOT = Path(__file__).resolve().parents[1]


def digest(path, limit=16*1024**2):
    result, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            size += len(chunk)
            if size > limit:
                raise ValueError("recovery evidence byte budget exceeded")
            result.update(chunk)
    return result.hexdigest()


def read(path):
    if path.stat().st_size > 1024**2:
        raise ValueError("gate JSON budget exceeded")
    return json.loads(path.read_text())


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"m[23]-stream-recovery-gate(?:-r[0-9]{1,2})?", args.name):
        raise ValueError("owned finite recovery gate name required")
    run = ROOT/"runs"/args.name
    proof = read(run/"proof.json")
    root_job = read(ROOT/"runs"/(args.name+"-job.json"))
    expected = {"profile": 2, "continuous": 6, "partial": 3, "resumed": 3}
    if (root_job["status"] != "passed" or root_job["work_items"] != 14 or root_job["memory_bytes"] != 1024**3 or
            proof["status"] != "passed" or proof["updates"] != expected or
            proof["orchestrator_sha256"] != digest(ROOT/"scripts/probe_stream_recovery.py")):
        raise ValueError("complete exact-source native gate required")
    code = run/"continuous/code"
    catalogue = proof["source_catalogue_sha256"]
    archive = verify_source(code, catalogue)
    sys.path[:0] = [str(code/"scripts"), str(code)]
    sys.dont_write_bytecode = True
    from m2_stream_data import sample_case
    from PIL import Image
    if not Path(sample_case.__code__.co_filename).resolve().is_relative_to(code):
        raise ValueError("recovery evidence sampler is not frozen")
    selected, summary_rows, training_count, probe_count = [], {}, 0, 0
    for name, updates in expected.items():
        child = run/name
        summary, job = read(child/"summary.json"), read(run/(name+"-job.json"))
        verify_source(child/"code", catalogue)
        partial = name == "partial"
        if (job["exit_code"] != (2 if partial else 0) or job["status"] != ("failed" if partial else "passed") or
                job["memory_bytes"] != 640*1024**2 or job["work_items"] != updates or
                job["supervisor_script_sha256"] != archive["files"]["scripts/run_bounded.py"] or
                summary["updates_this_run"] != updates or summary["status"] != ("step_partial" if partial else "complete") or
                summary["catalogue_sha256"] != catalogue or
                not summary["checkpoint_round_trip_exact"] or
                not Path(summary["driver_source_path"]).resolve().is_relative_to(child/"code")):
            raise ValueError("child status/resource/archive contract differs")
        inference = read(child/"checkpoint/manifest.json")
        architecture = checkpoint_architecture(summary, inference)
        if (proof.get("architecture", architecture) != architecture or
                proof.get("component_capacity", summary.get("component_capacity", 0)) != summary.get("component_capacity", 0) or
                proof.get("config", summary["config"]) != summary["config"]):
            raise ValueError("recovery proof architecture/configuration differs")
        if digest(child/"checkpoint/weights.pt") != summary["checkpoint_sha256"] or inference["weights_sha256"] != summary["checkpoint_sha256"]:
            raise ValueError("recovery-gate inference weight identity differs")
        pointer = read(child/"latest-training.json")
        if pointer != summary["latest_training_snapshot"] or not re.fullmatch(r"step-[0-9]{6}-[0-9a-f]{32}", pointer["generation"]):
            raise ValueError("recovery pointer differs")
        folder = child/"training-snapshots"/pointer["generation"]
        snapshot = read(folder/"manifest.json")
        if (digest(folder/"manifest.json") != pointer["manifest_sha256"] or
                digest(folder/"tensors.pt", 256*1024**2) != snapshot["tensors_sha256"] or
                snapshot["step"] != summary["steps"] or snapshot["contract"] != summary["recovery_contract"]):
            raise ValueError("recovery-gate snapshot identity differs")
        if set(snapshot["ledgers"]) != {"curve.jsonl", "training.jsonl", "evaluations.jsonl"}:
            raise ValueError("recovery snapshot ledger set differs")
        for ledger, cursor in snapshot["ledgers"].items():
            if ledger not in {"curve.jsonl", "training.jsonl", "evaluations.jsonl"} or (child/ledger).stat().st_size != cursor["bytes"] or digest(child/ledger) != cursor["sha256"]:
                raise ValueError("committed ledger prefix differs")
        rows = [json.loads(line) for line in (child/"training.jsonl").read_text().splitlines()]
        if len(rows) != summary["steps"] or not 1 <= len(rows) <= 6:
            raise ValueError("finite recovery ledger count differs")
        for row in rows:
            if bool(row.get("final_capture", False)) != bool(summary.get("component_capacity", 0)):
                raise ValueError("recovery capture policy differs from model")
            extra = {"final_capture": True} if row.get("final_capture") else {}
            replay = sample_case(row["group_index"], row["view"], row["split"], row["base_seed"], profile_square=row["profile_square"], **extra)
            if {key: value for key, value in row.items() if key != "step"} != json.loads(json.dumps(replay.record)):
                raise ValueError("recovery training recipe/pixels differ")
            training_count += 1
        for split in ("training", "development"):
            for line in (child/(split+"-probe.jsonl")).read_text().splitlines():
                row = json.loads(line)
                if bool(row.get("final_capture", False)) != bool(summary.get("component_capacity", 0)):
                    raise ValueError("recovery probe capture policy differs from model")
                extra = {"final_capture": True} if row.get("final_capture") else {}
                replay = sample_case(row["group_index"], row["view"], split, row["base_seed"], **extra)
                path = child/(split+"-probe")/row["sample_id"]
                with Image.open(path.with_suffix(".png")) as stored:
                    if stored.mode != replay.frame.image.mode or stored.size != replay.frame.image.size or stored.tobytes() != replay.frame.image.tobytes():
                        raise ValueError("recovery saved probe pixels differ")
                if digest(path.with_suffix(".json")) != row["state_sha256"] or json.loads(json.dumps(replay.record)) != row:
                    raise ValueError("recovery saved probe labels/recipe differ")
                probe_count += 1
        summary_rows[name] = summary
        for filename in ("summary.json", "progress.json", "curve.jsonl", "training.jsonl", "evaluations.jsonl", "memory.jsonl",
                         "training-probe.jsonl", "development-probe.jsonl", "latest-training.json"):
            selected.append((child/filename, name+"/"+filename))
        selected += [(child/"code/source.json", name+"/source-manifest.json"),
                     (child/"checkpoint/manifest.json", name+"/checkpoint-manifest.json"),
                     (folder/"manifest.json", name+"/training-snapshot-manifest.json")]
        selected += [(run/(name+suffix), name+suffix) for suffix in ("-job.json", ".log", ".stderr.log")]
    def curves(name):
        return [{key: value for key, value in json.loads(line).items() if key != "elapsed_seconds"}
                for line in (run/name/"curve.jsonl").read_text().splitlines()]
    if (curves("continuous") != curves("resumed") or curves("partial") != curves("continuous")[:3] or
            (run/"continuous/training.jsonl").read_bytes() != (run/"resumed/training.jsonl").read_bytes() or
            (run/"continuous/evaluations.jsonl").read_bytes() != (run/"resumed/evaluations.jsonl").read_bytes()):
        raise ValueError("exact recovery trajectory differs")
    for field in ("exact_data_ledger", "exact_loss_terms_gradient_curve", "exact_evaluations", "exact_model_optimizer_rng", "all_final_tensors_reload_exact"):
        if proof[field] is not True:
            raise ValueError("native recovery equivalence incomplete")
    evidence = ROOT/"docs/evidence"/args.name
    evidence.mkdir(exist_ok=False)
    selected.append((run/"proof.json", "proof.json"))
    selected += [(ROOT/"runs"/(args.name+suffix), args.name+suffix) for suffix in ("-job.json", ".log", ".stderr.log")]
    hashes = {}
    for source, relative in selected:
        hashes[relative] = digest(source)
        target = evidence/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    scope = {"status": "passed", "source_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "architecture": architecture, "component_capacity": summary_rows["continuous"].get("component_capacity", 0),
             "source_catalogue_sha256": catalogue, "replayed_training_records_including_resumed_prefix": training_count,
             "verified_saved_probe_images": probe_count, "artifact_sha256": hashes,
             "native_peak_job_commit_bytes": root_job["peak_job_commit_bytes"],
             "curator_sha256": digest(Path(__file__)),
             "identity_validator_sha256": digest(ROOT/"scripts/record_stream_run.py"),
             "claim": "Native2+6+3+3 recovery/archived-source equivalence; not long-shape memory or model accuracy acceptance."}
    (evidence/"scope.json").write_text(json.dumps(scope, indent=2)+"\n")
    print(json.dumps({"status": "passed", "replayed_training_records": training_count, "saved_probe_images": probe_count, "scope": str(evidence/"scope.json")}))


if __name__ == "__main__":
    main()
