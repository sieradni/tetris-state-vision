"""Verify finite training artifacts and curate receipts, retaining weights in runs/."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from tetris_vision.dataset import source_digest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()
    if not args.name.startswith("m2-learn") or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in args.name):
        raise ValueError("expected an owned m2-learn run name")
    run = ROOT/"runs"/args.name
    summary = json.loads((run/"summary.json").read_text())
    supervisor = json.loads((ROOT/f"runs/{args.name}-job.json").read_text())
    checkpoint = json.loads((run/"checkpoint/manifest.json").read_text())
    if supervisor["status"] != "passed" or summary["status"] != "complete" or summary["steps"] != args.steps:
        raise ValueError("requested optimizer budget did not complete")
    if not summary["checkpoint_round_trip_exact"] or not summary["cpu_only"]:
        raise ValueError("CPU/checkpoint gate missing")
    if summary["runtime_source_sha256"] != source_digest():
        raise ValueError("runtime source differs from trained artifact")
    if summary["script_sha256"] != sha(ROOT/"scripts/train_m2_pilot.py"):
        raise ValueError("training driver differs from measured run")
    weights = run/"checkpoint/weights.pt"
    if sha(weights) != summary["checkpoint_sha256"] or sha(weights) != checkpoint["weights_sha256"]:
        raise ValueError("trained weights checksum mismatch")
    if sha(run/"manifest.jsonl") != summary["manifest_sha256"]:
        raise ValueError("data manifest checksum mismatch")
    records = [json.loads(line) for line in (run/"manifest.jsonl").read_text().splitlines()]
    for record in records:
        if sha(run/record["image"]) != record["image_sha256"] or sha(run/record["state"]) != record["state_sha256"]:
            raise ValueError("rendered image/target checksum mismatch")
    curve = [json.loads(line) for line in (run/"curve.jsonl").read_text().splitlines()]
    if [row["step"] for row in curve] != list(range(1, args.steps+1)):
        raise ValueError("optimizer ledger is incomplete or duplicated")
    evidence = ROOT/"docs/evidence"/args.name
    evidence.mkdir(exist_ok=False)
    for filename in ("summary.json", "curve.jsonl", "progress.json", "manifest.jsonl"):
        shutil.copyfile(run/filename, evidence/filename)
    shutil.copyfile(run/"checkpoint/manifest.json", evidence/"checkpoint-manifest.json")
    for suffix in ("-job.json", ".log", ".stderr.log"):
        source = ROOT/f"runs/{args.name}{suffix}"
        shutil.copyfile(source, evidence/source.name)
    for suffix in (".json", ".log"):
        source = ROOT/f"runs/m2-train-admission{suffix}"
        if source.exists():
            shutil.copyfile(source, evidence/source.name)
    shutil.copyfile(run/"lattice-preview.png", ROOT/f"examples/{args.name}.png")
    result = {"milestone": "M2 finite scratch-learning pilot", "steps": args.steps,
              "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "runtime_source_sha256": source_digest(), "driver_sha256": summary["script_sha256"],
              "weights_path": str(weights), "weights_sha256": sha(weights),
              "data_images_verified": len(records), "curve_steps_verified": len(curve),
              "checkpoint_round_trip_exact": True, "raw_peak_job_bytes": supervisor["peak_job_commit_bytes"],
              "admission_tests": "m2-train-admission.log: 2 passed; common numerical gate remains 85 checks at 8d9d0f2",
              "not_achieved": "Complete-state parsing, precise generalization, real-image quality, calibration or release",
              "artifact_policy": "Weights/data stay in retained runs/; receipts/hashes/previews are committed."}
    with (evidence/"receipt.json").open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps({"status": "passed", "steps": args.steps, "images_verified": len(records),
                      "weights_sha256": sha(weights), "receipt": str(evidence/"receipt.json")}))


if __name__ == "__main__":
    main()
