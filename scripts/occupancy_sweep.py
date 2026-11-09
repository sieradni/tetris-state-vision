"""Predeclared read-only rule for the locked occupancy logit ranking.

`docs/M3_OCCUPANCY_RANKING.md` states the rule; this module is its single implementation:
the fixed threshold grid, the pooled AUC with its chance reference, the swept
precision/recall/F1 curve, the named best operating point and the verdict. The grid is fixed
before any receipt exists and never read from a run's current operating point.

Decision convention: a known cell is predicted occupied when ``logit > threshold``, the
same strict comparison the replay's ordinary metrics use, so the grid's zero row is a
reproducible join with earlier receipts. A logit exactly at a threshold counts as empty.
"""
import bisect

GRID_LOW, GRID_HIGH, GRID_STEP = -6.0, 6.0, 0.15
AUC_THRESHOLD = 0.90
CHANCE_AUC = 0.5
F1_THRESHOLD = 0.70
DECIDING_SPLIT = "training"


def replay_keywords(recipe, capacity):
    """The declared replay policy, held here because the checkpoint archive predates it.

    The sweep re-executes against a run's frozen archive, which was captured at training
    time and contains only the training sources; a test binds this copy to
    ``m3_error_analysis.replay_keywords`` so the two cannot drift (V-052 pattern).
    """
    if type(capacity) is not int or not 0 <= capacity <= 64:
        raise ValueError("bounded component capacity required")
    capture = recipe.get("final_capture", False)
    if type(capture) is not bool or capture != bool(capacity):
        raise ValueError("probe capture policy differs from component configuration")
    return (({"final_capture": True}, {"component_capacity": capacity}) if capacity else ({}, {}))


def threshold_grid():
    """The whole predeclared grid: 81 thresholds on [-6, 6] with step 0.15."""
    count = int(round((GRID_HIGH-GRID_LOW)/GRID_STEP))
    return tuple(round(GRID_LOW+index*GRID_STEP, 2) for index in range(count+1))


def require_cells(cells):
    """Both occupied and empty known-cell logits are required; unknown is not empty."""
    if type(cells) is not dict:
        raise ValueError("cell population required")
    positive, negative = cells.get("positive"), cells.get("negative")
    if not isinstance(positive, list) or not isinstance(negative, list) or not positive or not negative:
        raise ValueError("occupied and empty known-cell logits required")


def pooled_auc(positive, negative):
    """P(occupied logit > empty logit) with ties counted one half (Mann-Whitney)."""
    require_cells({"positive": positive, "negative": negative})
    ordered = sorted(negative)
    total = 0.0
    for value in positive:
        left, right = bisect.bisect_left(ordered, value), bisect.bisect_right(ordered, value)
        total += left+0.5*(right-left)
    return total/(len(positive)*len(negative))


def _counts(case, threshold):
    true_positive = false_positive = false_negative = true_negative = 0
    right = True
    for value, label in zip(case["logits"], case["occupied"]):
        predicted = value > threshold
        if label and predicted:
            true_positive += 1
        elif label:
            false_negative += 1
            right = False
        elif predicted:
            false_positive += 1
            right = False
        else:
            true_negative += 1
    return true_positive, false_positive, false_negative, true_negative, right


def evaluate(cases, thresholds=None):
    """Sweep one split's saved boards; each case carries known-cell logits and labels."""
    thresholds = threshold_grid() if thresholds is None else tuple(thresholds)
    if type(cases) is not list or not cases:
        raise ValueError("at least one saved board is required")
    for case in cases:
        if (type(case) is not dict or case.get("split") not in ("training", "development") or
                not isinstance(case.get("logits"), list) or not isinstance(case.get("occupied"), list) or
                len(case["logits"]) != len(case["occupied"]) or not case["logits"] or
                any(type(value) is not int or value not in (0, 1) for value in case["occupied"])):
            raise ValueError("saved board logits and labels required")
    split = cases[0]["split"]
    if any(case["split"] != split for case in cases):
        raise ValueError("a sweep is one split at a time")
    nonempty = [case for case in cases if any(case["occupied"])]
    empty = [case for case in cases if not any(case["occupied"])]
    positive = [value for case in nonempty for value, label in zip(case["logits"], case["occupied"]) if label]
    negative = [value for case in nonempty for value, label in zip(case["logits"], case["occupied"]) if not label]
    auc = pooled_auc(positive, negative)
    rows = []
    for threshold in thresholds:
        totals = [0, 0, 0, 0]
        exact_nonempty = exact_empty = 0
        for case in nonempty:
            counts = _counts(case, threshold)
            totals = [left+right for left, right in zip(totals, counts[:4])]
            exact_nonempty += int(counts[4])
        # Counts are the deciding nonempty population only; the empty population's false
        # positives are reported at the best threshold instead of pooled into this curve.
        for case in empty:
            exact_empty += int(_counts(case, threshold)[4])
        rows.append({"threshold": threshold, "true_positive": totals[0], "false_positive": totals[1],
                     "false_negative": totals[2], "true_negative": totals[3],
                     "exact_nonempty_boards": exact_nonempty, "exact_empty_boards": exact_empty})
    for row in rows:
        predicted = row["true_positive"]+row["false_positive"]
        observed = row["true_positive"]+row["false_negative"]
        row["precision"] = row["true_positive"]/predicted if predicted else 0.0
        row["recall"] = row["true_positive"]/observed if observed else 0.0
        row["f1"] = (2*row["precision"]*row["recall"]/(row["precision"]+row["recall"])
                     if row["precision"]+row["recall"] else 0.0)
    # Highest F1; ties to the higher precision, then to the lower threshold.
    best = max(rows, key=lambda row: (row["f1"], row["precision"], -row["threshold"]))
    empty_known = sum(len(case["logits"]) for case in empty)
    empty_at_best = [0, 0, 0, 0]
    for case in empty:
        counts = _counts(case, best["threshold"])
        empty_at_best = [left+right for left, right in zip(empty_at_best, counts[:4])]
    return {"split": split, "nonempty_images": len(nonempty), "empty_images": len(empty),
            "occupied_cells": len(positive), "empty_cells_on_nonempty_boards": len(negative),
            "auc": auc, "chance_auc": CHANCE_AUC,
            "thresholds": list(thresholds), "rows": rows, "best": dict(best),
            "empty_population": {"images": len(empty), "known_cells": empty_known,
                                 "false_positive": empty_at_best[1],
                                 "false_positive_rate": empty_at_best[1]/empty_known if empty_known else 0.0},
            "exact_boards_at_best": {"nonempty": best["exact_nonempty_boards"],
                                     "empty": best["exact_empty_boards"]}}


def sweep_verdict(summary, *, auc_threshold=AUC_THRESHOLD, f1_threshold=F1_THRESHOLD):
    """``informative`` iff the ranking clears the AUC bar and some swept point clears F1."""
    if type(summary) is not dict:
        raise ValueError("sweep summary required")
    auc, best = summary.get("auc"), summary.get("best")
    if (not isinstance(auc, (int, float)) or type(best) is not dict or
            not isinstance(best.get("f1"), (int, float))):
        raise ValueError("sweep AUC and best operating point required")
    return ("informative" if auc >= auc_threshold and best["f1"] >= f1_threshold
            else "uninformative")
