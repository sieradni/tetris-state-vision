"""Decompose why the record's best arms produce no strict exact board.

Read-only: every sweep records one logit and one label per visible-board cell, so exactness can be taken apart without a
checkpoint or a forward. For each arm this records, per board: how many cells miss, in which rows, in which direction,
whether any per-board level would have made the board exact (the oracle-threshold test - if a board is rescued by a level
the miss is a level, if it is not the miss is not one), whether a per-row level would (a row-structure test), the
within-board separation of the read (AUC, which says noise) against the level it sits at (which says bias), and the board
features a mechanism could be told.

No training, no weight change, no threshold tuned into the record: the arm's own `0.0` boundary stays the one that is
scored, and the oracle columns are diagnostics that say what *would* have been exact, never what was.
"""
import argparse
import json
import statistics

import torch
from pathlib import Path

from probe_model import require_bounded_job

from occupancy_sweep import threshold_grid
from tetris_vision.model import otsu_board, stack_contiguous

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 640*1024**2
CATALOGUES = {"m22-stencil": ROOT/"docs/evidence/m22-stencil-r1/scope.json",
              "m23-contrast": ROOT/"docs/evidence/m23-contrast-r1/scope.json"}
ARMS = ("stencil", "centred", "norm", "norm-cross", "envelope", "control")


def read(path):
    return json.loads(Path(path).read_text())


def cells_of(case, rows, columns):
    logits = case["logits"]
    labels = case["occupied"]
    return [[float(logits[row*columns+column]) for column in range(columns)] for row in range(rows)], \
           [[bool(labels[row*columns+column]) for column in range(columns)] for row in range(rows)]


def exact_at(logits, labels, threshold):
    """Strict (V-139): every visible cell right, and every column's top right with an empty column compared."""
    wrong = 0
    for row in range(len(logits)):
        for column in range(len(logits[0])):
            wrong += int((logits[row][column] > threshold) != labels[row][column])
    return wrong == 0, wrong


def boundary_exact(logits, labels, threshold):
    """The boundary rule the record also reports: every column's top decided row equals its top occupied row."""
    columns = len(logits[0])
    for column in range(columns):
        decided = [row for row in range(len(logits)) if logits[row][column] > threshold]
        filled = [row for row in range(len(logits)) if labels[row][column]]
        if (max(decided) if decided else None) != (max(filled) if filled else None):
            return False
    return True


def exact_level_search(logits, labels):
    """Every distinct level the read can be cut at, exactly - not the 81-point grid.

    A level can make a board exact only if every occupied cell outranks every empty one. Searching the read's own values
    as cut points answers that question without a grid, so "no level works" cannot be an artefact of the grid.
    """
    flat = [(logits[row][column], labels[row][column])
            for row in range(len(logits)) for column in range(len(logits[0]))]
    values = sorted({value for value, _ in flat})
    cuts = [values[0]-1e-6]+[(low+high)/2 for low, high in zip(values, values[1:])]+[values[-1]+1e-6]
    wrong = [sum(1 for value, label in flat if (value > cut) != label) for cut in cuts]
    return {"exact_at_any_level": min(wrong) == 0, "fewest_wrong_cells": min(wrong),
            "levels_searched": len(cuts)}


def inversions(logits, labels):
    """How many (occupied, empty) pairs are ranked the wrong way round on this board - the ordering deficit."""
    occupied = [logits[row][column] for row in range(len(logits)) for column in range(len(logits[0]))
                if labels[row][column]]
    empty = [logits[row][column] for row in range(len(logits)) for column in range(len(logits[0]))
             if not labels[row][column]]
    if not occupied or not empty:
        return 0
    return sum(1 for a in occupied for b in empty if a <= b)


def projected_exactness(logits, labels, rows, columns):
    """Can any declared consistency projection, at its best level, make the board exact?

    The record already contains the projections - the monotone envelope and the per-board crossing rule - so this asks
    whether the ordering deficit survives them. A projection can only move cells the read already orders; it cannot
    invent separation that is not there.
    """
    tensor = torch.tensor([[logits[row][column] for column in range(columns)] for row in range(rows)],
                          dtype=torch.float32)[None]
    shapes = {"raw": tensor,
              "monotone": stack_contiguous(tensor, rows),
              "monotone_then_crossing": otsu_board(stack_contiguous(tensor, rows), rows)}
    out = {}
    flat_labels = [label for row in labels for label in row]
    for name, value in shapes.items():
        flat = [float(v) for v in value.flatten()]
        rows_of = [flat[index*columns:(index+1)*columns] for index in range(rows)]
        labels_of = [flat_labels[index*columns:(index+1)*columns] for index in range(rows)]
        out[name] = exact_level_search(rows_of, labels_of)
    return out


def board_auc(logits, labels):
    """Within-board separation: the probability an occupied cell outranks an empty one, on this board alone."""
    flat = [(value, label) for row in range(len(logits))
            for value, label in zip(logits[row], labels[row])]
    occupied = [value for value, label in flat if label]
    empty = [value for value, label in flat if not label]
    if not occupied or not empty:
        return None
    wins = sum(1 for a in occupied for b in empty if a > b)
    ties = sum(1 for a in occupied for b in empty if a == b)
    return (wins+0.5*ties)/(len(occupied)*len(empty))


def board_block(case, rows, columns, grid):
    logits, labels = cells_of(case, rows, columns)
    exact, wrong = exact_at(logits, labels, 0.0)
    false_positive = sum(1 for row in range(rows) for column in range(columns)
                         if logits[row][column] > 0.0 and not labels[row][column])
    false_negative = sum(1 for row in range(rows) for column in range(columns)
                         if logits[row][column] <= 0.0 and labels[row][column])
    missed_rows = sorted({row for row in range(rows) for column in range(columns)
                          if (logits[row][column] > 0.0) != labels[row][column]})
    row_cells = [sum(labels[row]) for row in range(rows)]
    row_signed = [round(statistics.fmean([logits[row][column]-(1.0 if labels[row][column] else 0.0)
                                          for column in range(columns)]), 6) for row in range(rows)]
    # oracle levels: is there any per-board level that would have been exact? any per-row one?
    best = min((wrong_at for wrong_at in (exact_at(logits, labels, level)[1] for level in grid)))
    with_level = [level for level in grid if exact_at(logits, labels, level)[0]]
    rows_with_own_level = 0
    rows_rescued = 0
    for row in range(rows):
        row_ok = any(all((logits[row][column] > level) == labels[row][column] for column in range(columns))
                     for level in grid)
        rows_rescued += int(row_ok)
        rows_with_own_level += 1
    margins = {"occupied": [logits[row][column] for row in range(rows) for column in range(columns)
                            if labels[row][column]],
               "empty": [logits[row][column] for row in range(rows) for column in range(columns)
                         if not labels[row][column]]}
    return {
        "sample_id": case["sample_id"], "board": case.get("group_id"),
        "exact": bool(exact), "boundary_exact": bool(boundary_exact(logits, labels, 0.0)),
        "missed_cells": wrong, "false_positive": false_positive, "false_negative": false_negative,
        "missed_rows": missed_rows,
        "occupied_cells": sum(row_cells), "occupancy_rate": round(sum(row_cells)/(rows*columns), 6),
        "row_totals": row_cells,
        "row_signed_margin": row_signed,
        "auc_within_board": (round(board_auc(logits, labels), 6) if board_auc(logits, labels) is not None else None),
        "mean_margin_occupied": round(statistics.fmean(margins["occupied"]), 6) if margins["occupied"] else None,
        "mean_margin_empty": round(statistics.fmean(margins["empty"]), 6) if margins["empty"] else None,
        "margin_sd_occupied": round(statistics.pstdev(margins["occupied"]), 6) if len(margins["occupied"]) > 1 else None,
        "margin_sd_empty": round(statistics.pstdev(margins["empty"]), 6) if len(margins["empty"]) > 1 else None,
        "exact_level_search": exact_level_search(logits, labels),
        "projected_exactness": projected_exactness(logits, labels, rows, columns),
        "inversions": inversions(logits, labels),
        "pairs": (sum(1 for row in range(rows) for column in range(columns) if labels[row][column])
                  * sum(1 for row in range(rows) for column in range(columns) if not labels[row][column])),
        "oracle_level_exact": len(with_level) > 0,
        "oracle_levels_that_work": len(with_level),
        "oracle_best_level": (min(with_level) if with_level else None),
        "fewest_wrong_cells_at_any_level": best,
        "rows_would_be_exact_at_a_row_level": rows_rescued,
        "rows": rows_with_own_level,
    }


def arm_block(scope, name):
    arm = scope["arms"][name]
    sweep = read(ROOT/"runs"/(arm["sweep"]+"/result.json"))
    config = sweep["config"]
    rows, columns = config["visible_rows"], config["columns"]
    grid = tuple(threshold_grid())
    cases = [case for case in sweep["cases"] if case["split"] == "training" and any(case["occupied"])]
    boards = [board_block(case, rows, columns, grid) for case in cases]
    total = len(boards)
    return {
        "arm": name, "study": arm["study"],
        "auc": arm["auc"], "f1_at_zero": arm["f1_at_zero"],
        "row_error_at_zero": arm["row_error_at_zero"]["mean"],
        "boards": total,
        "exact_boards_at_zero": sum(1 for board in boards if board["exact"]),
        "boundary_exact_boards_at_zero": sum(1 for board in boards if board["boundary_exact"]),
        "boards_rescued_by_a_per_board_level": sum(1 for board in boards if board["oracle_level_exact"]),
        "boards_exact_at_some_exact_level": sum(1 for board in boards
                                                if board["exact_level_search"]["exact_at_any_level"]),
        "fewest_wrong_cells_at_any_exact_level": min(board["exact_level_search"]["fewest_wrong_cells"]
                                                    for board in boards),
        "mean_fewest_wrong_cells_at_any_exact_level": round(statistics.fmean(
            board["exact_level_search"]["fewest_wrong_cells"] for board in boards), 6),
        "median_inversions": statistics.median(board["inversions"] for board in boards),
        "max_inversions": max(board["inversions"] for board in boards),
        "mean_inversion_rate": round(statistics.fmean(board["inversions"]/board["pairs"] for board in boards), 6),
        "boards_with_a_wrong_cell_at_every_level": sum(1 for board in boards if not board["oracle_level_exact"]),
        "fewest_wrong_cells_at_any_level": min(board["fewest_wrong_cells_at_any_level"] for board in boards),
        "mean_missed_cells": round(statistics.fmean(board["missed_cells"] for board in boards), 6),
        "mean_false_positive": round(statistics.fmean(board["false_positive"] for board in boards), 6),
        "mean_false_negative": round(statistics.fmean(board["false_negative"] for board in boards), 6),
        "boards_exact_after_a_projection": {name: sum(1 for board in boards
                                                       if board["projected_exactness"][name]["exact_at_any_level"])
                                             for name in ("raw", "monotone", "monotone_then_crossing")},
        "boards_with_misses_only_above_the_stack": sum(1 for board in boards
                                                       if board["missed_cells"] and board["false_negative"] == 0),
        "rows_would_be_exact_at_a_row_level": sum(board["rows_would_be_exact_at_a_row_level"] for board in boards),
        "row_slots": sum(board["rows"] for board in boards),
        "mean_auc_within_board": round(statistics.fmean(board["auc_within_board"] for board in boards
                                                        if board["auc_within_board"] is not None), 6),
        "per_board": boards,
    }


def correlation(boards, key):
    pairs = [(board[key], board["missed_cells"]) for board in boards if board[key] is not None]
    if len(pairs) < 3:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sx = statistics.pstdev(xs)
    sy = statistics.pstdev(ys)
    if not sx or not sy:
        return None
    return round(sum((x-mx)*(y-my) for x, y in pairs)/(len(pairs)*sx*sy), 6)


def main():
    bounded = require_bounded_job()
    if bounded != CEILING_BYTES:
        raise ValueError("this decomposition must run at its declared 640 MiB ceiling")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("an evidence directory already exists")
    scope = read(CATALOGUES["m23-contrast"])
    arms = {}
    for name in ARMS:
        block = arm_block(scope, name)
        block["correlation_of_miss_with"] = {
            key: correlation(block["per_board"], key)
            for key in ("occupancy_rate", "occupied_cells", "mean_margin_occupied", "auc_within_board")}
        arms[name] = block
    scope_out = {
        "status": "passed",
        "instrument": "m24-exactness-r1",
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "source_catalogue": str(CATALOGUES["m23-contrast"].relative_to(ROOT)),
        "reading": ("why each board of the record's best arms is not exact, per board and per row: missed cells and "
                    "their direction, the rows they sit in, whether any per-board level would have been exact, whether "
                    "any per-row level would have been, and the within-board separation of the read against the level it "
                    "sits at"),
        "rule": {"exact": "every visible cell correct at 0.0 (V-139)",
                 "oracle_level": "a level from the standing 81-point grid at which that board is exact",
                 "oracle_level_does_not_change_the_scored_number": True},
        "arms": arms,
    }
    args.output.mkdir(parents=True)
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(scope_out, stream, indent=2, allow_nan=False)
    print(json.dumps({name: {key: arms[name][key] for key in
                             ("boards", "exact_boards_at_zero", "boards_rescued_by_a_per_board_level",
                              "boards_with_a_wrong_cell_at_every_level", "mean_missed_cells", "mean_false_positive",
                              "mean_false_negative", "mean_auc_within_board",
                              "rows_would_be_exact_at_a_row_level", "row_slots")}
                      for name in ARMS}, indent=2))


if __name__ == "__main__":
    main()
