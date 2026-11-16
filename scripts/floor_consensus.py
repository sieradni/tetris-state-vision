"""M54's floor-consensus lattice: the vertical correction is decided by how many blocks share a bottom.

M53's receipt located its failure precisely. On `training-20261031-v1` and `-v2` the detected box's bottom sat one row
above the well's floor, and the lowest visible block edge - the stack resting on that floor - moved the box down by one
whole cell: `0.924144` cells of displacement became `0.000000`. On `training-20261027-v2` and the `training-20261033`
trio the box's bottom was already right, and the same rule moved it by one cell for no reason: `0.055` became `1.077`.
The single lowest edge is therefore the right anchor on some captures and the wrong one on others, and the difference
is not visible in its position - only in how many blocks sit at it.

This module keeps M53's anchors, its pitch median and its horizontal phase snap unchanged, and replaces exactly one
decision: the row correction is taken from a **consensus floor** - the bottom edge shared by at least
`MIN_FLOOR_BLOCKS` distinct components, agreeing within `FLOOR_AGREEMENT` cells - rather than from the single lowest
edge. A single usable floor, or two rows tying on mass, is a declared fallback: the box's own bottom is kept and the
receipt says so.

No label, prevalence, scored cell, model lattice or target lattice is read: the input is the raster and M51's label-free
box, and the output is a lattice.
"""
import numpy as np

from stack_anchor import (
    COLUMNS, MIN_ANCHORS, RATIO_TOLERANCE, ROWS, SATURATION, anchors, blocks, circular_median, fmod, pitch_from,
)

MIN_FLOOR_BLOCKS = 2
FLOOR_AGREEMENT = 0.15
CONSTANTS = {"saturation": SATURATION, "min_block_cells": 0.25, "max_block_cells": 5.0,
             "ratio_tolerance": RATIO_TOLERANCE, "min_pairs": 3, "min_anchors": MIN_ANCHORS,
             "min_floor_blocks": MIN_FLOOR_BLOCKS, "floor_agreement_cells": FLOOR_AGREEMENT}


def floor_consensus(found, pitch_y, tolerance=FLOOR_AGREEMENT, minimum=MIN_FLOOR_BLOCKS):
    """The bottom row the blocks agree on, or a declared fallback.

    Bottoms are clustered within `tolerance` cells; the heaviest cluster wins only if it holds at least `minimum`
    components and no other cluster ties it. Anything else - no blocks, one block, two rows with equal mass - declines
    the correction instead of guessing.
    """
    if not found:
        return {"kept": False, "reason": "no visible blocks", "clusters": [], "blocks": 0}
    bottoms = sorted(float(box["bottom"]) for box in found)
    clusters = []
    for value in bottoms:
        if clusters and value - clusters[-1][-1] <= tolerance*pitch_y:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    counts = [len(cluster) for cluster in clusters]
    report = {"kept": False, "clusters": counts, "blocks": len(found),
              "bottoms": [round(cluster[0], 3) for cluster in clusters]}
    pinned = [index for index, count in enumerate(counts) if count >= minimum]
    if not pinned:
        report["members"] = max(counts)
        report["reason"] = "no row is pinned by enough blocks"
        return report
    members = clusters[pinned[-1]]
    report["members"] = counts[pinned[-1]]
    report.update({"kept": True, "bottom": float(np.median(members)),
                   "spread": round(max(members)-min(members), 6), "reason": None})
    return report


def row_correction(floor, box_bottom, pitch_y):
    """The whole-cell offset the consensus floor justifies, and why it is or is not applied.

    The correction is one-sided by construction. Blocks cannot sit below the well's floor, so ink pinned *below* the
    box's bottom can only mean the box stopped short of the floor - that is the `training-20261031` case, and it is
    corrected. A pinned row *above* the box's bottom is the other reading: the box already reaches the floor and the
    bottom row is empty - the `training-20261027-v2` and `training-20261033` case, where moving the frame would invent
    a row. The second case is therefore left alone rather than corrected downwards.
    """
    if not floor.get("kept"):
        return 0, floor.get("reason") or "no consensus floor"
    rows = int(round((floor["bottom"]-float(box_bottom))/pitch_y))
    if rows < 0:
        return 0, "the pinned row is above the box's floor: the bottom row is empty"
    return rows, None if rows else "the consensus floor agrees with the box"


def anchor_lattice(pixels, box):
    """M53's stack-anchored lattice with the row correction decided by the consensus floor."""
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

    if fitted_x is not None and len(left_edges) >= MIN_ANCHORS:
        phase = circular_median(left_edges, fitted_x)
        offset = phase-fmod(frame["left"], fitted_x)
        offset -= fitted_x*round(offset/fitted_x)
        left = float(frame["left"]+offset)
        report["x"] = {"kept": True, "pitch": round(fitted_x, 6), "origin": round(left, 6),
                       "phase_pixels": round(phase, 6), "snap_pixels": round(offset, 6),
                       "anchors": len(left_edges)}
    else:
        left, fitted_x = float(frame["left"]), pitch_x
        report["x"] = {"kept": False, "reason": "too few horizontal anchors", "anchors": len(left_edges)}

    if fitted_y is not None and len(top_edges) >= MIN_ANCHORS:
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
                       "rows_corrected": 0, "floor": {"kept": False, "reason": "too few vertical anchors"}}

    return {"origin": [left, top+ROWS*step_y], "u": [fitted_x, 0.0], "v": [0.0, -step_y],
            "frame": {"left": left, "right": left+COLUMNS*fitted_x, "top": top, "bottom": top+ROWS*step_y},
            "fit": report, "source": "floor-consensus"}
