"""Exact two-update default-path comparison to the actual prior pilot profile."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
from probe_model import require_bounded_job
from record_stream_run import checkpoint_architecture
from source_snapshot import verify_source


def read(path):
    if path.stat().st_size > 1024**2:
        raise ValueError("metadata budget")
    return json.loads(path.read_text())


def sha(path):
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError("file budget")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    require_bounded_job()
    names = ("m2-stream-layer-control-profile-r1", "m2-stream-default-policy-profile-r1")
    runs, summaries, archives = [], [], []
    hashes = {}
    for name in names:
        run = ROOT / "runs" / name
        summary = read(run / "summary.json")
        job = read(ROOT / "runs" / (name + "-job.json"))
        if (job["status"] != "passed" or job["exit_code"] != 0 or job["work_items"] != 2 or
                job["memory_bytes"] != 640 * 1024**2 or summary["status"] != "complete" or summary["steps"] != 2 or
                not summary["training_recovery_profile_exact"] or not summary["checkpoint_round_trip_exact"]):
            raise ValueError("actual complete640MiB two-update profiles required")
        archive = verify_source(run / "code", summary["catalogue_sha256"])
        manifest = read(run / "checkpoint/manifest.json")
        checkpoint_architecture(summary, manifest)
        if sha(run / "checkpoint/weights.pt") != summary["checkpoint_sha256"] or manifest["weights_sha256"] != summary["checkpoint_sha256"]:
            raise ValueError("actual weight identity differs")
        runs.append(run)
        summaries.append(summary)
        archives.append(archive)
        hashes[name] = {file: sha(run / file) for file in ("summary.json", "curve.jsonl", "training.jsonl", "checkpoint/weights.pt")}
    for field in ("config", "checkpoint_sha256", "initial_training_probe", "initial_development", "final_training_probe", "development"):
        if summaries[0][field] != summaries[1][field]:
            raise ValueError("default numerical field differs: " + field)
    for file in ("training.jsonl", "training-probe.jsonl", "development-probe.jsonl"):
        if (runs[0] / file).read_bytes() != (runs[1] / file).read_bytes():
            raise ValueError("default data bytes differ")
    curves = []
    for run in runs:
        curve = [json.loads(line) for line in (run / "curve.jsonl").read_text().splitlines()]
        if [row["step"] for row in curve] != [1, 2]:
            raise ValueError("two-update cursor differs")
        curves.append([{key: value for key, value in row.items() if key != "elapsed_seconds"} for row in curve])
    if curves[0] != curves[1]:
        raise ValueError("default loss/terms/gradients/samples differ")
    packages = [{path: value for path, value in archive["files"].items() if path.startswith("tetris_vision/")} for archive in archives]
    if packages[0] != packages[1]:
        raise ValueError("neural package sources differ")
    out = ROOT / "docs/evidence/default-curriculum-numerics-r1"
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / "verify.py")
    proof = {"status": "passed", "compared_optimizer_updates": 2, "runs": names,
             "source_catalogues": [summary["catalogue_sha256"] for summary in summaries],
             "identical_actual_weight_bytes": True, "identical_data_bytes": True, "identical_all_four_probe_metric_objects": True,
             "exact_loss_terms_gradients_samples_excluding_elapsed": True, "identical_neural_package_sources": True,
             "weights_sha256": summaries[0]["checkpoint_sha256"], "input_sha256": hashes,
             "script_sha256": sha(Path(__file__)),
             "scope": "Actual default two-step warm-profile numerical compatibility only. Not long training, all-class learning, convergence or real accuracy."}
    (out / "scope.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({"status": "passed", "updates": 2}), flush=True)


if __name__ == "__main__":
    main()
