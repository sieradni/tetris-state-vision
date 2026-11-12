"""Measure how far each occupancy decision boundary sits from the label boundary, in rows.

Read-only re-analysis of the four frozen M5 sweeps: every sweep records one logit and one label per
visible-board cell, so the top-most decided row of each column can be compared with the top-most
occupied row of the same column without loading a checkpoint or running a forward. Rows are bottom-first
everywhere, so a positive error means the decision sits above the label - the head claims the stack is
taller than it is - and a negative error means it sits below.

No training, no weight change, no decoded state, no accuracy claim. This measures where the boundary is,
not why.
"""
import argparse
import json
from pathlib import Path
import sys

from probe_model import require_bounded_job

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 640*1024**2
ARMS = ("control", "wide", "demean", "column")
BOARDS = 28
OCCUPIED_CELLS = 667
CLIP = 3
GRID_POINTS = 81


def read(path):
    return json.loads(Path(path).read_text())


def boundaries(logits, labels, rows, columns, threshold):
    """Top-most decided and top-most occupied row of every column, and whether either is empty."""
    decided, occupied = [], []
    for column in range(columns):
        cells = [logits[row*columns+column] for row in range(rows)]
        truth = [labels[row*columns+column] for row in range(rows)]
        above = [row for row, value in enumerate(cells) if value > threshold]
        filled = [row for row, value in enumerate(truth) if value]
        decided.append(max(above) if above else None)
        occupied.append(max(filled) if filled else None)
    return decided, occupied


def summarise(values):
    if not values:
        raise ValueError("summary of an empty sample")
    ordered = sorted(values)
    count = len(ordered)
    mean = sum(ordered)/count
    absolute = sorted(abs(value-mean) for value in values)
    return {"mean": round(mean, 6), "median": ordered[count//2], "min": ordered[0],
            "max": ordered[-1], "p10": ordered[int(.1*(count-1))],
            "p90": ordered[int(round(.9*(count-1)))],
            "mean_absolute_deviation": round(sum(absolute)/count, 6),
            "cells": count}


def arm_block(result, name):
    deciding = result["deciding"]
    config = result["config"]
    rows, columns = config["visible_rows"], config["columns"]
    cases = [case for case in result["cases"] if case["split"] == "training"]
    if len(cases) != 36 or sum(1 for case in cases if any(case["occupied"])) != BOARDS:
        raise ValueError(f"{name} sweep does not carry the deciding training population")
    blocks = {}
    for key, threshold in (("zero", 0.0), ("best", deciding["best"]["threshold"])):
        errors = []
        census = {"both": 0, "label_only": 0, "decided_only": 0, "neither": 0}
        counts = {str(value): 0 for value in range(-CLIP, CLIP+1)}
        clipped = 0
        boards = []
        conditional = []
        for case in cases:
            if not any(case["occupied"]):
                continue
            decided, occupied = boundaries(case["logits"], case["occupied"], rows, columns, threshold)
            board_errors = []
            for column in range(columns):
                got, want = decided[column], occupied[column]
                if got is not None and want is not None:
                    census["both"] += 1
                    error = got-want
                    errors.append(error)
                    board_errors.append(error)
                    if want >= 3:
                        conditional.append(error)
                    if error < -CLIP:
                        clipped += 1
                        counts[str(-CLIP)] += 1
                    elif error > CLIP:
                        clipped += 1
                        counts[str(CLIP)] += 1
                    else:
                        counts[str(error)] += 1
                elif want is not None:
                    census["label_only"] += 1
                elif got is not None:
                    census["decided_only"] += 1
                else:
                    census["neither"] += 1
            boards.append({"sample_id": case["sample_id"], "group_id": case["group_id"],
                           "mean_error": (round(sum(board_errors)/len(board_errors), 6)
                                          if board_errors else None),
                           "columns": len(board_errors),
                           "exact": (round(sum(1 for value in board_errors if value == 0)
                                           /len(board_errors), 6) if board_errors else None)})
        if not errors:
            raise ValueError(f"{name} {key} boundary has no comparable columns")
        blocks[key] = {
            "threshold": threshold,
            "columns": census,
            "error_rows": summarise(errors),
            "error_counts": counts,
            "clipped": clipped,
            "exact_fraction": round(sum(1 for value in errors if value == 0)/len(errors), 6),
            "within_one_fraction": round(sum(1 for value in errors if abs(value) <= 1)/len(errors), 6),
            "conditional_on_label_top_at_least_3": summarise(conditional),
            "boards": boards,
        }
    blocks["arm"] = {"occupancy_deform": config.get("occupancy_deform", 0),
                     "occupancy_column": config.get("occupancy_column", 0),
                     "occupancy_detail": config.get("occupancy_detail", 0),
                     "weights_sha256": result["weights_sha256"],
                     "sweep": result["run"],
                     "catalogue_sha256": result.get("catalogue_sha256")}
    return blocks


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    job = read(ROOT/"runs"/(args.output.name+"-job.json"))
    if job["memory_bytes"] != CEILING_BYTES:
        raise ValueError("row-error ceiling differs from the predeclared 640 MiB")
    if args.output.exists():
        raise ValueError("an evidence directory already exists")
    arms = {}
    for name in ARMS:
        sweep = f"m5-colread-{name}-r1-sweep-r1"
        result = read(ROOT/"runs"/sweep/"result.json")
        if result.get("status") != "passed" or result.get("run") != f"m5-colread-{name}-r1":
            raise ValueError(f"{name} sweep is not a passed measurement of its own run")
        if len(result["deciding"]["thresholds"]) != GRID_POINTS or 0.0 not in result["deciding"]["thresholds"]:
            raise ValueError(f"{name} sweep did not use the standing grid")
        if result["deciding"]["occupied_cells"] != OCCUPIED_CELLS:
            raise ValueError(f"{name} sweep decided another population")
        arms[name] = arm_block(result, name)
    grid = {"zero": {}, "best": {}}
    for key in ("zero", "best"):
        grid[key] = {name: {"mean": arms[name][key]["error_rows"]["mean"],
                            "median": arms[name][key]["error_rows"]["median"],
                            "exact_fraction": arms[name][key]["exact_fraction"],
                            "within_one_fraction": arms[name][key]["within_one_fraction"],
                            "columns": arms[name][key]["error_rows"]["cells"],
                            "label_only": arms[name][key]["columns"]["label_only"],
                            "decided_only": arms[name][key]["columns"]["decided_only"]}
                        for name in ARMS}
    result = {
        "status": "passed",
        "instrument": "m5-row-error-r1",
        "scope": ("Read-only re-analysis of four frozen sweeps: for every column of every deciding board, "
                  "the top-most decided row against the top-most occupied row, in rows. Rows are "
                  "bottom-first, so a positive error means the decision claims a taller stack than the "
                  "label. No training, no weight change, no decoded state, no accuracy claim."),
        "job_ceiling_bytes": job["memory_bytes"],
        "grid": grid,
        "arms": {name: {key: arms[name][key] for key in ("zero", "best", "arm")} for name in ARMS},
        "population": {"split": "training", "boards": BOARDS, "occupied_cells": OCCUPIED_CELLS,
                       "columns_per_board": 10, "arms": list(ARMS)},
        "sources": {name: f"runs/m5-colread-{name}-r1-sweep-r1/result.json" for name in ARMS},
    }
    args.output.mkdir(parents=True)
    with (args.output/"result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(grid, indent=2))


if __name__ == "__main__":
    main()
