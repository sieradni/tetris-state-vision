"""The predeclared fit-gate rule (`docs/M3_FIT_GATE.md`) as executable code.

The rule is stated once: the gate OPENS iff the independently replayed run reports at
least ``FIT_GATE_THRESHOLD`` exact **nonempty** training boards. Empty boards, development
spilts, roles, geometry and component counts are costs and can never open it.

The mechanism label keeps a negative honest: "constant-empty" is the read-out that never
predicts an occupied cell (so its walk over the empty boards is free), "per-cell-error" is
a read-out that does predict occupied cells and still reproduces no board. A negative
names which of the two it is instead of assuming starvation.
"""
from m3_error_analysis import constant_empty_readout

FIT_GATE_THRESHOLD = 3
DECIDING_SPLIT = "training"


def _nonempty(locked_populations):
    if type(locked_populations) is not dict or type(locked_populations.get("nonempty")) is not dict:
        raise ValueError("nonempty locked population required")
    nonempty = locked_populations["nonempty"]
    if type(nonempty.get("exact_boards")) is not int or type(nonempty.get("images")) is not int:
        raise ValueError("nonempty exact-board counts required")
    return nonempty


def fit_gate_opens(locked_populations, threshold=FIT_GATE_THRESHOLD):
    """True only for >= threshold exact nonempty boards on the deciding split."""
    return _nonempty(locked_populations)["exact_boards"] >= threshold


def occupancy_mechanism(locked_populations):
    """Name what pins the locked read-out: no occupied cell, or occupied cells but no board."""
    nonempty = _nonempty(locked_populations)
    for key in ("true_positive", "false_positive", "false_negative"):
        if type(nonempty.get(key)) is not int:
            raise ValueError("nonempty cell counts required")
    return "constant-empty" if constant_empty_readout({"nonempty": nonempty}) else "per-cell-error"
