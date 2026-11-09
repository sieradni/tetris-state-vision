"""Does the occupancy failure live in the boards or in the operating point?

`occupancy_sweep` closes the family of **unconditioned, global** thresholds. This module measures
the other half of that question from committed per-board logits: how well each nonempty deciding
board ranks on its own, and whether the boards that rank well agree on an operating point. It
declares no new threshold - the grid and the 0.70 bar are the standing ones - and no gate imports
it.
"""
from occupancy_sweep import F1_THRESHOLD, threshold_grid

DECIDING_SPLIT = "training"
ORACLE_BOARDS_MIN = 4
THRESHOLD_SPREAD_BAR = 1.5
TIE_EPSILON = 1e-12

VERDICTS = ("representation-limited", "conditioned-boundary", "inconsistent")


def board_rows(sweep, split=DECIDING_SPLIT):
    """Nonempty deciding boards, sorted by sample id so every number is order-stable."""
    boards = []
    for case in sweep["cases"]:
        if case["split"] != split or not any(case["occupied"]):
            continue
        boards.append({"board": case["sample_id"], "group": case["group_id"], "view": case["view"],
                       "cells": [(float(logit), int(label))
                                 for logit, label in zip(case["logits"], case["occupied"])]})
    if not boards:
        raise ValueError("nonempty deciding boards required")
    return sorted(boards, key=lambda board: board["board"])


def board_counts(board, threshold):
    counts = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
    for logit, label in board["cells"]:
        predicted = logit > threshold
        if label and predicted:
            counts["true_positive"] += 1
        elif label:
            counts["false_negative"] += 1
        elif predicted:
            counts["false_positive"] += 1
    return counts


def board_f1(board, threshold):
    """2TP/(2TP+FP+FN) on this board's own known cells; no occupied cell means no F1."""
    counts = board_counts(board, threshold)
    denominator = 2*counts["true_positive"]+counts["false_positive"]+counts["false_negative"]
    return 0.0 if not denominator else 2*counts["true_positive"]/denominator


def oracle_board(board, grid=None):
    """The board's best standing-grid point; strict improvement keeps the lowest threshold on ties."""
    grid = threshold_grid() if grid is None else tuple(grid)
    if not grid:
        raise ValueError("at least one grid threshold required")
    best = None
    for threshold in grid:
        value = board_f1(board, threshold)
        if best is None or value > best["f1"]+TIE_EPSILON:
            best = {"threshold": threshold, "f1": value,
                    "counts": board_counts(board, threshold)}
    best["f1"] = round(best["f1"], 6)
    return best


def board_auc(board):
    positive = [logit for logit, label in board["cells"] if label]
    negative = [logit for logit, label in board["cells"] if not label]
    if not positive or not negative:
        return None
    from occupancy_sweep import pooled_auc
    return pooled_auc(positive, negative)


def classify(oracle, *, bar=F1_THRESHOLD, minimum=ORACLE_BOARDS_MIN, spread_bar=THRESHOLD_SPREAD_BAR):
    """The predeclared classification; the count and the spread decide, nothing else."""
    if not oracle:
        raise ValueError("at least one board required")
    reaching = [entry for entry in oracle if entry["f1"] >= bar-TIE_EPSILON]
    thresholds = sorted(entry["threshold"] for entry in reaching)
    spread = 0.0 if len(thresholds) < 2 else round(thresholds[-1]-thresholds[0], 6)
    if len(reaching) < minimum:
        verdict = "representation-limited"
    elif spread > spread_bar:
        verdict = "conditioned-boundary"
    else:
        verdict = "inconsistent"
    return {"verdict": verdict, "reaching": len(reaching), "boards": len(oracle),
            "minimum_reaching": minimum, "oracle_threshold_spread": spread,
            "spread_bar": spread_bar, "bar": bar,
            "reaching_thresholds": thresholds}


def measure(sweep, pooled_threshold=-0.90):
    """Per-board oracle table plus the pooled and oracle-pooled counts the record quotes."""
    boards = board_rows(sweep)
    rows = []
    for board in boards:
        oracle = oracle_board(board)
        rows.append({"board": board["board"], "group": board["group"], "view": board["view"],
                     "known_cells": len(board["cells"]),
                     "occupied_cells": sum(label for _, label in board["cells"]),
                     "auc": board_auc(board),
                     "oracle_threshold": oracle["threshold"], "oracle_f1": oracle["f1"],
                     "oracle_counts": oracle["counts"],
                     "pooled_counts": board_counts(board, pooled_threshold),
                     "pooled_f1": round(board_f1(board, pooled_threshold), 6)})
    verdict = classify([{"f1": row["oracle_f1"], "threshold": row["oracle_threshold"]} for row in rows])
    totals = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
    for row in rows:
        for key in totals:
            totals[key] += row["oracle_counts"][key]
    oracle_pooled = (2*totals["true_positive"] /
                     (2*totals["true_positive"]+totals["false_positive"]+totals["false_negative"]))
    aucs = sorted(row["auc"] for row in rows if row["auc"] is not None)
    return {"boards": rows, "classification": verdict,
            "oracle_pooled": {**totals, "f1": round(oracle_pooled, 6)},
            "auc_distribution": {"count": len(aucs), "min": aucs[0] if aucs else None,
                                 "median": aucs[len(aucs)//2] if aucs else None,
                                 "max": aucs[-1] if aucs else None},
            "pooled_threshold": pooled_threshold,
            "pooled_f1_median": sorted(row["pooled_f1"] for row in rows)[len(rows)//2]}
