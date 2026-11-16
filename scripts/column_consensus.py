"""M55's column-consensus lattice: the blocks can *certify* the box's columns, but they cannot place them.

M54's receipt named this axis, and its per-board table answers it in two lines:

- `training-20261027-v0` and `-v1` have **two components** each (four edge positions, six pitch pairs). There the fitted
  pitch drifted to `13.666667` px against the box's `14.0` / `13.9`, and M53's sub-pixel phase snap moved the frame by
  `2.833333` / `1.833333` px: displacement went from the box's `0.101000` / `0.054600` to `0.654800`.
- Boards with many components (`20261031-v0`, 54 components, 218 pitch pairs) have edge fractions clustered at
  `0.05`-`0.14` of a cell - the drawing inset - and M54's snap moved them by `-1.0` px, which is that inset measured from
  both sides and cancelled. They stayed inside the gate.

The mechanism behind both is the same and is a property of the axis, not of the estimator: a stack occupies *some*
columns of the well, and which columns those are is unknown, so its left edge carries no absolute information the way the
floor does. Every observable - the pitch, the inset, the phase - is invariant under moving the whole stack sideways. The
horizontal frame therefore cannot be *placed* from the blocks; it can only be *certified*: when at least
`MIN_COLUMN_BLOCKS` components put their left and right edges within `COLUMN_TOLERANCE` cells of a single phase, and that
phase sits inside the drawing band `COLUMN_INSET_MAX`, the box's columns are the blocks' columns and the frame stays
exactly where M51 put it. A phase outside that band is a column disagreement this axis cannot resolve in either
direction, and the declared fallback keeps the box unchanged.

So this instrument applies **no horizontal shift at all** and says why on every board; the vertical axis is M54's
consensus floor and its one-sided whole-cell row correction, unchanged. That is the honest answer to M54's named
question: the two boards it blamed on the snap are repaired by *not* snapping, and the board that is genuinely
misplaced horizontally (`training-20261035-v2`, `1.93` cells) is not reachable from block edges at any phase - it needs
the well's walls, which is M56.

No label, prevalence, scored cell, model lattice or target lattice is read: the input is the raster and M51's label-free
box, and the output is a lattice.
"""
import numpy as np

from floor_consensus import floor_consensus, row_correction
from stack_anchor import (
    COLUMNS, MIN_ANCHORS, RATIO_TOLERANCE, ROWS, SATURATION, anchors, blocks, pitch_from,
)

MIN_COLUMN_BLOCKS = 2
COLUMN_TOLERANCE = 0.15
COLUMN_INSET_MAX = 0.20
MIN_PITCH_SPAN_CELLS = 3.0
CONSTANTS = {"saturation": SATURATION, "min_block_cells": 0.25, "max_block_cells": 5.0,
             "ratio_tolerance": RATIO_TOLERANCE, "min_pairs": 3, "min_anchors": MIN_ANCHORS,
             "min_floor_blocks": 2, "floor_agreement_cells": 0.15,
             "min_column_blocks": MIN_COLUMN_BLOCKS, "column_tolerance_cells": COLUMN_TOLERANCE,
             "column_inset_max_cells": COLUMN_INSET_MAX, "min_pitch_span_cells": MIN_PITCH_SPAN_CELLS}


def _fraction(edge, box_left, pitch):
    return ((float(edge)-float(box_left))/pitch) % 1.0


def _spread(values):
    """The largest deviation from the circular median, on a unit period."""
    angles = np.asarray(values, dtype=float)*2.0*np.pi
    median = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())/(2.0*np.pi) % 1.0
    deviations = [min(abs(v-median), 1.0-abs(v-median)) for v in values]
    return float(median), float(max(deviations))


def column_consensus(found, pitch_x, box_left, tolerance=COLUMN_TOLERANCE, inset_max=COLUMN_INSET_MAX,
                     minimum=MIN_COLUMN_BLOCKS):
    """Whether the blocks certify the box's columns, and never a shift of them.

    The stack cannot pin an absolute column, so the only decision this instrument can make honestly is between
    "certified" (`kept`, shift `0`) and "not resolvable" (the declared fallback, shift `0`). The receipt records the
    measured phases either way.
    """
    if not found:
        return {"kept": False, "shift": 0, "reason": "no visible blocks", "blocks": 0}
    lefts = [_fraction(component["left"], box_left, pitch_x) for component in found]
    rights = [_fraction(component["right"], box_left, pitch_x) for component in found]
    left_median, left_spread = _spread(lefts)
    right_median, right_spread = _spread(rights)
    report = {"kept": False, "shift": 0, "blocks": len(found),
              "left_phase": round(left_median, 6), "right_phase": round(right_median, 6),
              "left_spread": round(left_spread, 6), "right_spread": round(right_spread, 6),
              "lefts": [round(v, 4) for v in lefts], "rights": [round(v, 4) for v in rights]}
    if len(found) < minimum:
        report["reason"] = "too few components to certify the columns"
        return report
    if left_spread > tolerance or right_spread > tolerance:
        report["reason"] = "the components' edge phases disagree beyond tolerance"
        return report
    inside_band = left_median <= inset_max and right_median >= 1.0-inset_max
    if not inside_band:
        report["reason"] = "the edge phase sits outside the drawing band: not resolvable from block edges"
        return report
    report.update({"kept": True, "reason": None,
                   "reading": "the blocks pin the box's columns inside the drawing inset"})
    return report


def anchor_lattice(pixels, box):
    """M54's vertical consensus plus the column certification: the horizontal frame is M51's, by measurement."""
    frame = box["frame"]
    pitch_x = float(box["u"][0])
    pitch_y = float(-box["v"][1])
    found = blocks(pixels, pitch_x, pitch_y, frame)
    left_edges, top_edges = anchors(found)
    report = {"blocks": len(found), "box": {"left": float(frame["left"]), "top": float(frame["top"]),
                                            "pitch_x": pitch_x, "pitch_y": pitch_y}}

    fitted_x, pairs_x = pitch_from(left_edges, pitch_x)
    fitted_y, pairs_y = pitch_from(top_edges, pitch_y)
    report["pairs"] = {"x": pairs_x, "y": pairs_y}

    # a pitch fitted from edges spanning less than MIN_PITCH_SPAN_CELLS cells is not pinned: the quantisation of a
    # couple of edges is a larger error than the pitch itself, and over twenty rows it drags the whole frame
    span_x = (max(left_edges)-min(left_edges))/pitch_x if len(left_edges) >= 2 else 0.0
    span_y = (max(top_edges)-min(top_edges))/pitch_y if len(top_edges) >= 2 else 0.0
    x_pinned = fitted_x is not None and span_x >= MIN_PITCH_SPAN_CELLS
    y_pinned = fitted_y is not None and span_y >= MIN_PITCH_SPAN_CELLS
    report["span_cells"] = {"x": round(span_x, 4), "y": round(span_y, 4)}
    report["pitch_pinned"] = {"x": bool(x_pinned), "y": bool(y_pinned)}
    if not x_pinned:
        fitted_x = None
    if not y_pinned:
        fitted_y = None

    if len(left_edges) >= MIN_ANCHORS:
        columns = column_consensus(found, pitch_x, frame["left"])
        left = float(frame["left"])
        report["x"] = {"kept": True, "pitch": round(pitch_x, 6), "origin": round(left, 6),
                       "anchors": len(left_edges),
                       "anchored_pitch": None if fitted_x is None else round(fitted_x, 6),
                       "columns": columns,
                       "correction": {"columns": 0, "applied": False,
                                      "reason": columns.get("reason") or columns.get("reading")}}
    else:
        left = float(frame["left"])
        report["x"] = {"kept": False, "reason": "too few horizontal anchors", "anchors": len(left_edges),
                       "anchored_pitch": None if fitted_x is None else round(fitted_x, 6),
                       "columns": {"kept": False, "reason": "too few horizontal anchors", "shift": 0},
                       "correction": {"columns": 0, "applied": False, "reason": "too few horizontal anchors"}}

    if y_pinned and fitted_y is not None and len(top_edges) >= MIN_ANCHORS:
        floor = floor_consensus(found, fitted_y)
        rows, why = row_correction(floor, frame["bottom"], fitted_y)
        bottom = float(frame["bottom"])+rows*fitted_y
        report["y"] = {"kept": True, "pitch": round(fitted_y, 6), "origin": round(bottom, 6),
                       "anchors": len(top_edges), "rows_corrected": rows, "floor": floor,
                       "correction": {"rows": rows, "applied": rows != 0, "reason": why}}
        top = bottom-ROWS*fitted_y
        step_y = fitted_y
    else:
        top, step_y = float(frame["top"]), pitch_y
        report["y"] = {"kept": False, "reason": "too few vertical anchors", "anchors": len(top_edges),
                       "rows_corrected": 0, "floor": {"kept": False, "reason": "too few vertical anchors"},
                       "correction": {"rows": 0, "applied": False, "reason": "too few vertical anchors"}}

    # the pitch is used only where the anchors span enough cells to pin it; the origin never moves in x
    step_x = fitted_x if x_pinned else pitch_x
    report["x"]["pitch"] = round(step_x, 6)
    return {"origin": [left, top+ROWS*step_y], "u": [step_x, 0.0], "v": [0.0, -step_y],
            "frame": {"left": left, "right": left+COLUMNS*step_x, "top": top, "bottom": top+ROWS*step_y},
            "fit": report, "source": "column-consensus"}
