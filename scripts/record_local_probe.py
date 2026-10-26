"""Curate actual same-layout intervention evidence without an accuracy claim."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from tetris_vision.dataset import source_digest
from probe_model import require_bounded_job


def main():
    require_bounded_job()
    def read(name):
        return json.loads((ROOT/"runs"/name).read_text())
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    result, job, admission = (read(name) for name in ("m2-local512-result.json", "m2-local512-job.json", "m2-local512-admission.json"))
    if result["status"] != "passed" or job["status"] != "passed" or not 0 < result["aggregate"]["forwards"] <= 54:
        raise ValueError("finite native intervention gate missing")
    if result["script_sha256"] != admission["script_sha256"] or sha(ROOT/"scripts/probe_local_reading.py") != result["script_sha256"]:
        raise ValueError("probe source changed")
    if result["runtime_source_sha256"] != source_digest() or source_digest() != admission["runtime_source_sha256"]:
        raise ValueError("runtime source changed")
    actual = {"targeted_correct": sum(row["targeted_correct"] for row in result["trials"]),
              "exact_counterfactual_boards": sum(row["exact_counterfactual_board"] for row in result["trials"]),
              "unrelated_changed_cells": sum(row["unrelated_prediction_changes"] for row in result["trials"]),
              "correct_direction_probability_changes": sum(row["signed_probability_delta"] > 0 for row in result["trials"])}
    if any(result["aggregate"][key] != value for key, value in actual.items()):
        raise ValueError("intervention aggregation differs")
    weights = ROOT/"runs/m2-learn512/checkpoint/weights.pt"
    if sha(weights) != result["checkpoint_sha256"]:
        raise ValueError("weight artifact changed")
    evidence = ROOT/"docs/evidence/m2-local-reading"
    evidence.mkdir(exist_ok=False)
    for name in ("m2-local512-result.json", "m2-local512-job.json", "m2-local512-admission.json", "m2-local512.log", "m2-local512.stderr.log"):
        shutil.copyfile(ROOT/"runs"/name, evidence/name)
    scope = {"status": "passed", "source_base_commit": admission["base_commit"],
             "curation_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "runtime_source_sha256": source_digest(), "probe_script_sha256": result["script_sha256"],
             "weight_sha256": result["checkpoint_sha256"], "result_sha256": sha(evidence/"m2-local512-result.json"),
             "aggregate": result["aggregate"], "native_peak_job_commit_bytes": job["peak_job_commit_bytes"],
             "scope": result["scope"], "not_claimed": "Independent real examples, complete-state accuracy, a completed512-step run or adequate learning.",
             "next_direction": "Broader state/position training with the same visibility contracts; preserve layer/geometry failures as open acceptance."}
    (evidence/"scope.json").write_text(json.dumps(scope, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "aggregate": result["aggregate"]}))


if __name__ == "__main__":
    main()
