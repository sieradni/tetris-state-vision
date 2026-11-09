"""Record the per-board conditioning decomposition of the occupancy failure.

`docs/M3_OCCUPANCY_CONDITIONING.md` predeclares the rule; this recorder applies it to the
committed widened sweep cases and writes one evidence directory. Read-only: no model, no training,
no new threshold, and every published point estimate is re-derived and asserted equal first.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from board_conditioning import DECIDING_SPLIT, measure
from occupancy_sweep import pooled_auc
from probe_model import require_bounded_job

ROOT = Path(__file__).resolve().parents[1]
WIDENED = ROOT/"docs/evidence/m3-widened-deciding-population-r1"
INPUTS = {"scope": WIDENED/"scope.json", "sweep": WIDENED/"sweep-result.json",
          "replay": WIDENED/"occupancy-result.json"}


def read(path, maximum=8*1024**2):
    if path.stat().st_size > maximum:
        raise ValueError("conditioning evidence JSON byte budget exceeded")
    return json.loads(path.read_text())


def digest(path):
    value, consumed = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            consumed += len(block)
            if consumed > 64*1024**2:
                raise ValueError("conditioning evidence artifact byte budget exceeded")
            value.update(block)
    return value.hexdigest()


def density_of(replay):
    """Density label per sample id when the committed replay rows carry one; never inferred."""
    labels = {}
    for row in replay.get("cases", []):
        recipe = row.get("density_recipe")
        if isinstance(recipe, str):
            labels[row["sample_id"]] = recipe
        elif isinstance(recipe, dict) and isinstance(recipe.get("density"), str):
            labels[row["sample_id"]] = recipe["density"]
    return labels


def density_summary(boards, labels):
    if not labels:
        return {"available": False, "note": "committed replay rows carry no density recipe label"}
    grouped = {}
    for board in boards:
        label = labels.get(board["board"])
        if label is None:
            continue
        grouped.setdefault(label, []).append(board["oracle_threshold"])
    return {"available": True, "labels": {label: {"boards": len(values),
                                                  "median_oracle_threshold": round(statistics.median(values), 6),
                                                  "spread": round(max(values)-min(values), 6)}
                                          for label, values in sorted(grouped.items())}}


def main():
    bounded = require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("evidence directory already exists")
    inputs = {name: read(path) for name, path in INPUTS.items()}
    scope, sweep, replay = inputs["scope"], inputs["sweep"], inputs["replay"]
    for name, expected in scope["artifact_sha256"].items():
        if digest(WIDENED/name) != expected:
            raise ValueError("widened-population evidence artifact changed")
    if sweep["weights_sha256"] != scope["weights_sha256"]["occupancy"]:
        raise ValueError("conditioning decomposition would measure different weights")

    result = measure(sweep)
    published = scope["occupancy"]
    if (len(result["boards"]), sum(row["occupied_cells"] for row in result["boards"])) != (
            scope["populations"]["after"]["nonempty_images"],
            published["ranking"]["positives"]):
        raise ValueError("conditioning population differs from the published one")
    pooled_totals = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
    for row in result["boards"]:
        for key in pooled_totals:
            pooled_totals[key] += row["pooled_counts"][key]
    best = published["ranking"]["best"]
    if pooled_totals != {key: best[key] for key in pooled_totals}:
        raise ValueError("per-board counts at the standing threshold do not join the published sweep")

    cells = [(logit, label) for case in sweep["cases"] if case["split"] == DECIDING_SPLIT
             and any(case["occupied"]) for logit, label in zip(case["logits"], case["occupied"])]
    auc = pooled_auc([value for value, label in cells if label],
                     [value for value, label in cells if not label])
    if abs(auc-scope["occupancy"]["ranking"]["auc"]) > 1e-12:
        raise ValueError("AUC does not reproduce the published point estimate")

    labels = density_of(replay)
    payload = {
        "status": "passed",
        "instrument": "m3-occupancy-conditioning-r1",
        "predeclaration": "docs/M3_OCCUPANCY_CONDITIONING.md",
        "scope": ("Read-only decomposition of the closed occupancy failure into per-board ranking "
                  "versus operating-point conditioning on the widened deciding population. No "
                  "training, model, loss, threshold, gate or prior evidence byte changes, and the "
                  "ranking verdict is untouched by construction."),
        "inputs": {name: path.relative_to(ROOT).as_posix() for name, path in INPUTS.items()},
        "input_sha256": {name: digest(path) for name, path in INPUTS.items()},
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "population": {"split": DECIDING_SPLIT, "boards": len(result["boards"]),
                       "occupied_cells": sum(row["occupied_cells"] for row in result["boards"]),
                       "known_cells": sum(row["known_cells"] for row in result["boards"]),
                       "ancestry_groups": len({row["group"] for row in result["boards"]})},
        "classification": result["classification"],
        "auc": {"pooled": auc, "per_board": result["auc_distribution"]},
        "oracle": {"pooled": result["oracle_pooled"],
                   "pooled_threshold": result["pooled_threshold"],
                   "pooled_f1_median": result["pooled_f1_median"]},
        "boards": result["boards"],
        "density": density_summary(result["boards"], labels),
        "unchanged": {"point_estimates": True, "thresholds": scope["unchanged_thresholds"],
                      "ranking_verdict": published["ranking"]["verdict"],
                      "frozen_manifest": "not touched"},
    }
    args.output.mkdir(parents=True)
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    print(json.dumps({key: payload[key] for key in
                      ("classification", "auc", "oracle", "density", "population")}, indent=2))


if __name__ == "__main__":
    main()
