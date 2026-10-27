"""Replay192 recipes and rederive every aggregate from384 saved raw cases."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
from probe_model import require_bounded_job
from source_snapshot import verify_source
from m2_error_analysis import analyze_case


def read(path):
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("JSON budget")
    return json.loads(path.read_text())


def sha(path):
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError("hash budget")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    require_bounded_job()
    out = ROOT / "docs/evidence/layer-generalization768-r1"
    result = read(out / "result.json")
    job = read(ROOT / "runs/layer-generalization768-r1-job.json")
    if (job["status"] != "passed" or job["exit_code"] != 0 or job["memory_bytes"] != 640 * 1024**2 or
            job["work_items"] != 384 or job["wall_seconds"] != 180 or result["status"] != "passed" or
            result["forwards"] != 384 or result["all_model_tensors_unchanged"] is not True):
        raise ValueError("complete finite native inference required")
    for name, expected in result["artifact_sha256"].items():
        if sha(out / name) != expected:
            raise ValueError("raw artifact changed")
    for name, expected in result["sources"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("diagnostic dependency changed")
    if sha(out / "analysis-source.py") != sha(ROOT / "docs/experiments/probe_layer_generalization.py"):
        raise ValueError("executed analysis source differs")
    for model in result["models"].values():
        run = ROOT / "runs" / model["run"]
        verify_source(run / "code", result["source_catalogue_sha256"])
        manifest = read(run / "checkpoint/manifest.json")
        if manifest["config"] != model["config"] or sha(run / "checkpoint/weights.pt") != model["weights_sha256"]:
            raise ValueError("actual model artifact differs")
    code = ROOT / "runs" / result["models"]["control"]["run"] / "code"
    sys.path[:0] = [str(code / "scripts"), str(code)]
    sys.dont_write_bytecode = True
    from m2_stream_data import sample_case
    recipes = [json.loads(line) for line in (out / "recipes.jsonl").read_text().splitlines()]
    if [(row["group_index"], row["view"]) for row in recipes] != [(group, view) for group in range(100, 164) for view in range(3)]:
        raise ValueError("predeclared population differs")
    for recipe in recipes:
        replay = sample_case(recipe["group_index"], recipe["view"], "development", 20261026)
        if json.loads(json.dumps(replay.record)) != recipe:
            raise ValueError("pixel/state/recipe replay differs")
    additions = {}
    for label in ("control", "candidate"):
        rows = [json.loads(line) for line in (out / (label + ".jsonl")).read_text().splitlines()]
        if [row["sample_id"] for row in rows] != [row["sample_id"] for row in recipes]:
            raise ValueError("raw prediction population differs")
        for row in rows:
            actual = analyze_case(row["prediction"], row["target"])
            if actual != row["diagnostics"] or row["metrics"]["exact_locked_board"] != actual["locked"]["exact_board"]:
                raise ValueError("raw independent decomposition differs")
            if not math.isclose(row["metrics"]["mean_geometry_error_cells"], actual["geometry"]["mean_cells"], rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError("raw geometry differs")
        totals = {"images": len(rows), "ancestry_groups": len({row["group_id"] for row in rows}),
                  "populations": {}, "roles_by_target_class": {}, "pose_rows_by_role": {},
                  "mean_geometry_error_cells": sum(row["metrics"]["mean_geometry_error_cells"] for row in rows) / len(rows)}
        for population in ("empty", "nonempty"):
            cells = [row["diagnostics"]["locked"] for row in rows if row["diagnostics"]["locked"]["population"] == population]
            totals["populations"][population] = {"images": len(cells), "exact_boards": sum(item["exact_board"] for item in cells),
                **{key: sum(item[key] for item in cells) for key in ("known_cells", "true_positive", "true_negative", "false_positive", "false_negative")}}
        for index, role in enumerate(("active", "ghost")):
            classes, poses = {}, Counter()
            for row in rows:
                item = row["diagnostics"]["layers"][role]
                key = str(item["target_type"]) if item["target_known"] else "unknown"
                counts = classes.setdefault(key, Counter())
                counts.update(images=1, exact=int(item["exact"] is True), type_errors=int(item["type_correct"] is False), pose_known=int(item["target_pose_known"]))
                for dimension in ("column", "row", "rotation"):
                    counts[dimension + "_errors"] += int(item[dimension + "_correct"] is False)
                if item["target_pose_known"]:
                    poses[str(row["target"]["layers"][index]["row"])] += 1
            totals["roles_by_target_class"][role], totals["pose_rows_by_role"][role] = classes, poses
        totals["missing_observed_piece_classes_by_role"] = {
            role: ["IOTSZJL"[i-1] for i in range(1, 8) if str(i) not in totals["roles_by_target_class"][role]] for role in ("active", "ghost")}
        for key in rows[0]["metrics"]:
            if key.startswith("dense_"):
                totals[key] = sum(row["metrics"][key] for row in rows)
        if totals != result["results"][label]:
            raise ValueError("aggregate differs from raw cases")
        additions[label] = {"exact_observed_active_ancestries": sorted({row["group_id"] for row in rows if row["diagnostics"]["layers"]["active"]["target_pose_known"] and row["diagnostics"]["layers"]["active"]["exact"]}),
                            "joint_exact_locked_and_both_supported_layers": sum(row["diagnostics"]["locked"]["exact_board"] and all(item["target_known"] and item["exact"] for item in row["diagnostics"]["layers"].values()) for row in rows),
                            "predicted_types_by_role": {role: dict(Counter(str(row["prediction"]["layers"][i]["type"]) for row in rows)) for i, role in enumerate(("active", "ghost"))}}
    scope = {"status": "passed", "replayed_recipes": 192, "raw_cases_recomputed": 384,
             "all_reported_aggregates_recomputed": True, "result_sha256": sha(out / "result.json"),
             "curator_sha256": sha(Path(__file__)), "additional_raw_diagnostics": additions,
             "scope": "Synthetic source/artifact/recipe/raw-diagnostic/aggregate verification only. No real or complete-state quality acceptance."}
    if (out / "scope.json").exists():
        raise FileExistsError("scope exists")
    (out / "scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    print(json.dumps({"status": "passed", "replayed_recipes": 192, "recomputed_cases": 384}), flush=True)


if __name__ == "__main__":
    main()
