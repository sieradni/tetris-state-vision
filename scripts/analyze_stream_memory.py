"""Predeclared finite768-step memory gate and exact565-record failure prefix."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from probe_model import require_bounded_job
from source_snapshot import verify_source

ROOT = Path(__file__).resolve().parents[1]


def digest(path, limit=256*1024**2):
    result, consumed = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            consumed += len(chunk)
            if consumed > limit:
                raise ValueError("stream analysis artifact byte budget exceeded")
            result.update(chunk)
    return result.hexdigest()


def read(path):
    if path.stat().st_size > 1024**2:
        raise ValueError("analysis JSON byte budget exceeded")
    return json.loads(path.read_text())


def rows(path):
    if path.stat().st_size > 16*1024**2:
        raise ValueError("analysis ledger byte budget exceeded")
    values = [json.loads(line) for line in path.read_text().splitlines()]
    if len(values) > 4096:
        raise ValueError("analysis row count exceeded")
    return values


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", default="m2-stream-learn768-r2")
    args = parser.parse_args()
    for name in (args.candidate, args.baseline):
        if not re.fullmatch(r"m2-stream-[a-z0-9-]{1,40}", name):
            raise ValueError("owned stream name required")
    candidate, baseline = (ROOT/"runs"/name for name in (args.candidate, args.baseline))
    summary = read(candidate/"summary.json")
    job, old_job = (read(ROOT/"runs"/(name+"-job.json")) for name in (args.candidate, args.baseline))
    if (summary["status"] != "complete" or summary["steps"] != 768 or summary["resume"] is not None or
            job["status"] != "passed" or job["exit_code"] != 0 or job["memory_bytes"] != 640*1024**2 or
            old_job["status"] != "failed" or old_job["exit_code"] != 1 or old_job["memory_bytes"] != job["memory_bytes"]):
        raise ValueError("predeclared complete768 vs failed640MiB baseline envelope differs")
    archive = verify_source(candidate/"code", summary["catalogue_sha256"])
    old_archive = verify_source(baseline/"code")
    package = {name for name in archive["files"] if name.startswith("tetris_vision/")}
    old_package = {name for name in old_archive["files"] if name.startswith("tetris_vision/")}
    fixed = package | {"scripts/m2_stream_data.py", "pyproject.toml", "uv.lock"}
    exact_neural_data_sources = package == old_package and all(archive["files"].get(name) == old_archive["files"].get(name) for name in fixed)
    curves, old_curves = rows(candidate/"curve.jsonl"), rows(baseline/"curve.jsonl")
    ledger, old_ledger = rows(candidate/"training.jsonl"), rows(baseline/"training.jsonl")
    if ([row["step"] for row in curves] != list(range(1, 769)) or
            [row["step"] for row in ledger] != list(range(1, 769)) or
            [row["step"] for row in old_curves] != list(range(1, 566)) or len(old_ledger) != 565):
        raise ValueError("finite complete/failure ledgers differ from predeclared768/565")
    normalized = lambda values: [{key: value for key, value in row.items() if key != "elapsed_seconds"} for row in values]
    first_curve_difference = next((index+1 for index, (a, b) in enumerate(zip(normalized(curves), normalized(old_curves))) if a != b), None)
    first_data_difference = next((index+1 for index, (a, b) in enumerate(zip(ledger, old_ledger)) if a != b), None)
    memory = rows(candidate/"memory.jsonl")
    if [row["step"] for row in memory] != [1, *range(16, 769, 16)]:
        raise ValueError("predeclared every16-step memory trajectory incomplete")
    half_peak = next(row["peak_job_commit_bytes"] for row in memory if row["step"] == 384)
    final_peak = job["peak_job_commit_bytes"]
    growth = final_peak-half_peak
    pointer = read(candidate/"latest-training.json")
    if pointer != summary["latest_training_snapshot"] or not re.fullmatch(r"step-000768-[0-9a-f]{32}", pointer["generation"]):
        raise ValueError("final recovery pointer differs")
    folder = candidate/"training-snapshots"/pointer["generation"]
    snapshot = read(folder/"manifest.json")
    if digest(folder/"manifest.json") != pointer["manifest_sha256"] or digest(folder/"tensors.pt") != snapshot["tensors_sha256"]:
        raise ValueError("final recovery tensor/manifest identity differs")
    inference = read(candidate/"checkpoint/manifest.json")
    if digest(candidate/"checkpoint/weights.pt") != summary["checkpoint_sha256"] or inference["weights_sha256"] != summary["checkpoint_sha256"]:
        raise ValueError("candidate inference weights differ")
    checks = {"native_complete768_same640MiB": True, "unchanged_neural_render_data_and_lock": exact_neural_data_sources,
              "exact565_loss_terms_gradient_norm_prefix": first_curve_difference is None,
              "exact565_data_recipe_pixel_label_prefix": first_data_difference is None,
              "recovery_snapshot768": snapshot["step"] == 768 and snapshot["contract"] == summary["recovery_contract"],
              "all_final_tensors_reload_exact": summary["checkpoint_round_trip_exact"],
              "second_half_raw_peak_growth_at_most32MiB": 0 <= growth <= 32*1024**2}
    selected_metrics = ("images", "ancestry_groups", "loss", "exact_locked_boards", "cell_accuracy",
                        "mean_geometry_error_cells", "exact_known_layers", "known_layers")
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
              "source_catalogue_sha256": archive["catalogue_sha256"], "baseline_catalogue_sha256": old_archive["catalogue_sha256"],
              "weights_sha256": summary["checkpoint_sha256"], "steps": 768, "prefix_records": 565,
              "first_curve_difference": first_curve_difference, "first_data_difference": first_data_difference,
              "baseline_weights_available": False, "baseline_raw_peak_job_commit_bytes": old_job["peak_job_commit_bytes"],
              "candidate_raw_peak_job_commit_bytes": final_peak, "raw_peak_at384": half_peak,
              "second_half_raw_peak_growth_bytes": growth, "growth_threshold_bytes": 32*1024**2,
              "memory_samples": len(memory), "elapsed_seconds": summary["elapsed_seconds"],
              "metrics": {field: {key: summary[field][key] for key in selected_metrics} for field in
                          ("initial_training_probe", "initial_development", "final_training_probe", "development")},
              "analyzer_sha256": digest(Path(__file__)),
              "scope": "Finite768-step native memory/recovery gate and exact565 recorded prefix. No old model tensors survived; no real-image, whole-state, release or untested-long-run claim."}
    with (candidate/"analysis.json").open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
