"""The predeclared per-cell decision rule (`docs/M3_PERCELL_DECISION.md`) as code.

The rule is stated once: the lever OPENS iff the independently replayed run reports an
occupied-cell F1 of at least ``CELL_F1_THRESHOLD`` on the deciding split's nonempty locked
population. Exact-board counts, empty boards, development, roles, ghost, geometry,
components and queue side results are costs and can never open it.

``decision_mechanism`` names which error mass pins a standing decision, from the same
receipt counts, so a negative names its mechanism instead of being asserted.
"""
CELL_F1_THRESHOLD = 0.70
DECIDING_SPLIT = "training"


def _cells(locked_populations):
    if type(locked_populations) is not dict or type(locked_populations.get("nonempty")) is not dict:
        raise ValueError("nonempty locked population required")
    nonempty = locked_populations["nonempty"]
    for key in ("true_positive", "false_positive", "false_negative"):
        if type(nonempty.get(key)) is not int:
            raise ValueError("nonempty cell counts required")
    return nonempty


def occupied_cell_f1(locked_populations):
    """2TP/(2TP+FP+FN) over known cells; the predeclared per-cell quantity."""
    nonempty = _cells(locked_populations)
    true_positive = nonempty["true_positive"]
    denominator = 2*true_positive+nonempty["false_positive"]+nonempty["false_negative"]
    if denominator == 0:
        raise ValueError("occupied-cell F1 needs occupied or predicted cells")
    return 2*true_positive/denominator


def percell_gate_opens(locked_populations, threshold=CELL_F1_THRESHOLD):
    """True only at or above the threshold; an exact tie opens, below stands."""
    return occupied_cell_f1(locked_populations) >= threshold


def decision_mechanism(locked_populations):
    """Which error mass pins the decision: misses, inventions, or both.

    Ties between false positives and false negatives classify as ``both-limited`` so a
    balanced trade is never labelled as a one-sided defect.
    """
    nonempty = _cells(locked_populations)
    true_positive = nonempty["true_positive"]
    false_positive, false_negative = nonempty["false_positive"], nonempty["false_negative"]
    if percell_gate_opens(locked_populations):
        return "discriminating"
    if true_positive == 0 and not false_positive:
        return "constant-empty"
    if false_positive and false_negative:
        if false_positive == false_negative:
            return "both-limited"
        return "coverage-limited" if false_negative > false_positive else "specificity-limited"
    return "coverage-limited" if false_negative else "specificity-limited"
