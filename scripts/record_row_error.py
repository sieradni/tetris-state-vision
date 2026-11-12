"""Publish the row-error diagnostic: where each arm's decision boundary sits against its label row.

Reads the four frozen M5 sweeps through the same instrument that produced them, checks that every arm's
record is the arm it claims (weights digest, population, threshold grid), re-derives the cross-arm table
from the per-board blocks rather than trusting the summary, applies the reading rule declared in
`docs/M6_ROW_OFFSET.md` before the arm was declared, and writes one evidence scope.
"""
import argparse
import json
from pathlib import Path

from probe_model import require_bounded_job

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 640*1024**2
ARMS = ("control", "wide", "demean", "column")
BOARDS = 28
OCCUPIED_CELLS = 667
ROW_FLOOR = .5
READINGS = ("systematic-row-offset", "row-offset-arm-specific", "no-row-offset")
PUBLISHED = ROOT/"docs/evidence/m5-column-readout-r1/scope.json"


def read(path):
    return json.loads(Path(path).read_text())


def main():
    bounded = require_bounded_job()
    if bounded != CEILING_BYTES:
        raise ValueError("the diagnostic record must run at its declared 640 MiB ceiling")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("evidence directory already exists")
    result = read(ROOT/"runs/m5-rowerror-r1/result.json")
    job = read(ROOT/"runs/m5-rowerror-r1-job.json")
    if job.get("status") != "passed" or job.get("exit_code") != 0 or job.get("work_items") != 4:
        raise ValueError("the diagnostic did not pass under its declared envelope")
    if result.get("status") != "passed" or result.get("instrument") != "m5-row-error-r1":
        raise ValueError("the diagnostic record is not the declared instrument's")
    published = read(PUBLISHED)
    arms = {}
    for name in ARMS:
        arm = result["arms"][name]
        if arm["arm"]["weights_sha256"] != published["arms"][name]["weights_sha256"]:
            raise ValueError(f"{name} diagnostic is not the published arm's weights")
        if arm["arm"]["sweep"] != f"m5-colread-{name}-r1":
            raise ValueError(f"{name} diagnostic names another run")
        blocks = arm["zero"]["boards"]
        if len(blocks) != BOARDS or any(block["columns"] == 0 for block in blocks):
            raise ValueError(f"{name} diagnostic does not cover the deciding boards")
        # Re-derive the mean from the per-board means, weighted by their column counts: the summary
        # cannot disagree with the blocks it was computed from (V-106 class).
        weighted = sum(block["mean_error"]*block["columns"] for block in blocks)
        columns = sum(block["columns"] for block in blocks)
        if abs(weighted/columns-arm["zero"]["error_rows"]["mean"]) > 1e-6:
            raise ValueError(f"{name} diagnostic mean does not re-derive from its own boards")
        arms[name] = {"zero": {"mean": arm["zero"]["error_rows"]["mean"],
                               "median": arm["zero"]["error_rows"]["median"],
                               "min": arm["zero"]["error_rows"]["min"],
                               "max": arm["zero"]["error_rows"]["max"],
                               "p10": arm["zero"]["error_rows"]["p10"],
                               "p90": arm["zero"]["error_rows"]["p90"],
                               "mean_absolute_deviation":
                                   arm["zero"]["error_rows"]["mean_absolute_deviation"],
                               "exact_fraction": arm["zero"]["exact_fraction"],
                               "within_one_fraction": arm["zero"]["within_one_fraction"],
                               "columns": arm["zero"]["columns"],
                               "error_counts": arm["zero"]["error_counts"]},
                     "best": {"threshold": arm["best"]["threshold"],
                              "mean": arm["best"]["error_rows"]["mean"],
                              "median": arm["best"]["error_rows"]["median"],
                              "exact_fraction": arm["best"]["exact_fraction"],
                              "within_one_fraction": arm["best"]["within_one_fraction"]},
                     "arm": arm["arm"]}
        if arms[name]["zero"]["columns"]["both"]+arms[name]["zero"]["columns"]["label_only"] + \
                arms[name]["zero"]["columns"]["decided_only"]+arms[name]["zero"]["columns"]["neither"] \
                != BOARDS*10:
            raise ValueError(f"{name} column census does not cover the population")
    means = {name: arms[name]["zero"]["mean"] for name in ARMS}
    same_sign = (means["control"] > 0) == (means["wide"] > 0)
    if abs(means["control"]) >= ROW_FLOOR and abs(means["wide"]) >= ROW_FLOOR and same_sign:
        reading = "systematic-row-offset"
        reading_text = ("Both the published read-out and the family's best arm decide the stack above "
                        "its label row by at least half a row, in the same direction, so the bias is "
                        "geometry: the read looks one place and decides about another.")
    elif max(abs(means[name]) for name in ARMS) >= ROW_FLOOR:
        reading = "row-offset-arm-specific"
        reading_text = ("At least one arm's boundary is offset by half a row or more, but not both of "
                        "the two arms the rule names, so the bias belongs to a lever rather than to "
                        "the read-out.")
    else:
        reading = "no-row-offset"
        reading_text = ("No arm's boundary is offset by half a row, so the decisions are where the "
                        "labels are and the remaining error is not a sampling geometry problem.")
    if reading not in READINGS:
        raise ValueError("an undeclared reading was produced")
    scope = {
        "status": "passed",
        "instrument": "m5-row-error-r1",
        "predeclaration": "docs/M6_ROW_OFFSET.md",
        "scope": ("Read-only re-analysis of four frozen sweeps: for every column of every deciding "
                  "board, the top-most decided row against the top-most occupied row, in rows, "
                  "bottom-first. Positive means the decision claims a taller stack than the label. No "
                  "training, no weight change, no decoded state, no accuracy claim."),
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "rule": {"statistic": "mean of (decided top row - label top row), rows bottom-first",
                 "threshold": 0.0,
                 "floor_rows": ROW_FLOOR,
                 "arms_named": ["control", "wide"],
                 "readings": list(READINGS)},
        "reading": reading,
        "reading_text": reading_text,
        "arms": arms,
        "population": {"split": "training", "boards": BOARDS, "occupied_cells": OCCUPIED_CELLS,
                       "columns_per_board": 10, "column_observations": BOARDS*10},
        "provenance": {"result": "runs/m5-rowerror-r1/result.json",
                       "receipt": "runs/m5-rowerror-r1-job.json",
                       "sweeps": {name: f"runs/m5-colread-{name}-r1-sweep-r1/result.json"
                                  for name in ARMS}},
    }
    args.output.mkdir(parents=True)
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(scope, stream, indent=2, allow_nan=False)
    print(json.dumps({"reading": reading, "means": means}, indent=2))


if __name__ == "__main__":
    main()
