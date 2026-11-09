"""The predeclared identity discrimination rule as executable code.

Every M3 identity lever was judged on what the heads emit, not only on how often they are
right (`docs/M3_IDENTITY_DISCRIMINATION.md`, V-048): on the deciding split the queue-token
head must emit at least ``IDENTITY_CLASS_THRESHOLD`` distinct piece classes and must not be a
single constant, and the hold head must emit at least the same number. Both conditions are
required.

Exact identity counts are costs and can never open this rule, because a constant head's exact
count is only its class frequency. The rule is the one every identity document stated; this
module is its single implementation so a re-derivation cannot restate it by hand.
"""
IDENTITY_CLASS_THRESHOLD = 3
DECIDING_SPLIT = "training"


def _head(values, name):
    if type(values) is not dict:
        raise ValueError(f"{name} identity summary required")
    distinct = values.get("distinct_predicted_classes")
    constant = values.get("constant_predictor")
    if type(distinct) is not int or isinstance(distinct, bool) or not 0 <= distinct <= 8:
        raise ValueError(f"{name} distinct predicted class count required")
    if type(constant) is not bool:
        raise ValueError(f"{name} constant-predictor flag required")
    if constant and distinct != 1:
        raise ValueError(f"{name} constant flag disagrees with the distinct class count")
    return distinct, constant


def identity_head_opens(values, threshold=IDENTITY_CLASS_THRESHOLD):
    """One head clears the bar when it emits enough classes and is not a constant."""
    if type(threshold) is not int or isinstance(threshold, bool) or not 2 <= threshold <= 8:
        raise ValueError("identity class bar2..8 required")
    distinct, constant = _head(values, "identity")
    return distinct >= threshold and not constant


def identity_gate_opens(queue, hold, threshold=IDENTITY_CLASS_THRESHOLD):
    """Both heads must clear the class bar; one head carrying the other is not reading."""
    return identity_head_opens(queue, threshold) and identity_head_opens(hold, threshold)


def identity_verdict(queue, hold, threshold=IDENTITY_CLASS_THRESHOLD):
    return "opens" if identity_gate_opens(queue, hold, threshold) else "stands"
