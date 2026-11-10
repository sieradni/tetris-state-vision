"""Placement measurement for the declared geometry lever.

`docs/M3_OCCUPANCY_GEOMETRY.md` fixes this instrument. It is read-only and works on stored lattices:
the model's predicted (origin, u, v) basis is mapped into the *target* cell frame, so every model
cell has a continuous (column, row) offset in target cells. The occupancy score is taken on the
full deciding population - every known cell of every nonempty deciding board - never on the
oracle pairing's co-scoreable subset, because that subset is exactly where alignment is granted.
"""
import math

from gate_power import f1_from_counts
from occupancy_sweep import AUC_THRESHOLD, F1_THRESHOLD, pooled_auc, sweep_verdict
from percell_gate import CELL_F1_THRESHOLD
from registration_attribution import COLUMNS, ROWS, counts_of, lattice_of

ATTRIBUTION_FLOOR = 0.05
PLACEMENT_IMPROVEMENT = 0.25
TRANSLATION_FLOOR = 1.0
SCALE_FLOOR = 0.10
DISPERSION_FLOOR = 1.5
BOOTSTRAP_DRAWS = 400
BOOTSTRAP_SEED = 20261026


def cell_offsets(predicted, target):
    """Continuous (column, row) offset of each model cell centre in target cells, bottom-first."""
    t_origin, t_u, t_v, determinant = lattice_of(target, "target")
    p_origin, p_u, p_v, _ = lattice_of(predicted, "predicted")
    offsets = []
    for r in range(ROWS):
        for c in range(COLUMNS):
            x = p_origin[0]+(c+.5)*p_u[0]+(r+.5)*p_v[0]
            y = p_origin[1]+(c+.5)*p_u[1]+(r+.5)*p_v[1]
            dx, dy = x-t_origin[0], y-t_origin[1]
            column = (dx*t_v[1]-dy*t_v[0])/determinant
            row = (t_u[0]*dy-t_u[1]*dx)/determinant
            offsets.append((column-c-.5, row-r-.5))
    return offsets


def score(cases):
    """Occupancy at 0.0 on every known cell of the nonempty deciding boards."""
    nonempty = [case for case in cases if any(case["occupied"])]
    if not nonempty:
        raise ValueError("the deciding population requires at least one nonempty board")
    pairs = [(value, label) for case in nonempty
             for value, label, known in zip(case["logits"], case["occupied"], case["known"]) if known]
    counts = counts_of(pairs)
    positive = [value for value, label in pairs if label]
    negative = [value for value, label in pairs if not label]
    if not positive or not negative:
        raise ValueError("the deciding population needs both classes")
    f1 = f1_from_counts(counts["true_positive"], counts["false_positive"], counts["false_negative"])
    return {"images": len(nonempty), "empty_images": len(cases)-len(nonempty),
            "known_cells": len(pairs), "occupied_cells": len(positive),
            "empty_cells": len(negative), "auc": pooled_auc(positive, negative),
            "f1": f1, "counts": counts}


def stats(cases):
    """Mean, signed and scale placement error over the same boards, all cells."""
    nonempty = [case for case in cases if any(case["occupied"])]
    if not nonempty:
        raise ValueError("placement requires the nonempty deciding boards")
    per_board, signed_column, signed_row, absolute = [], [], [], []
    scale_u, scale_v, shifted, mapped = [], [], 0, 0
    for case in nonempty:
        board = case["offsets"]
        if len(board) != COLUMNS*ROWS:
            raise ValueError("each board must carry one offset per cell")
        per_board.append(sum(abs(value) for pair in board for value in pair)/(2*len(board)))
        signed_column.append(sum(pair[0] for pair in board)/len(board))
        signed_row.append(sum(pair[1] for pair in board)/len(board))
        absolute.extend(abs(value) for pair in board for value in pair)
        p_origin, p_u, p_v, _ = lattice_of(case["predicted"], "predicted")
        t_origin, t_u, t_v, _ = lattice_of(case["target"], "target")
        scale_u.append(math.hypot(*p_u)/math.hypot(*t_u))
        scale_v.append(math.hypot(*p_v)/math.hypot(*t_v))
        shifted += sum(1 for index, target in enumerate(case["mapping"])
                       if target is not None and target != index)
        mapped += sum(1 for target in case["mapping"] if target is not None)
    total = COLUMNS*ROWS*len(nonempty)
    ordered = sorted(per_board)
    return {
        "images": len(nonempty), "cells": total,
        "mean_abs_offset_cells": sum(absolute)/len(absolute),
        "mean_board_error_cells": sum(per_board)/len(per_board),
        "signed_column_offset_cells": sum(signed_column)/len(signed_column),
        "signed_row_offset_cells": sum(signed_row)/len(signed_row),
        "mean_scale_ratio_u": sum(scale_u)/len(scale_u),
        "mean_scale_ratio_v": sum(scale_v)/len(scale_v),
        "board_error_spread_cells": ordered[-1]-ordered[0],
        "board_error_min_cells": ordered[0], "board_error_max_cells": ordered[-1],
        "shifted_share": shifted/total, "mapped_share": mapped/total,
        "per_board_error_cells": [round(value, 6) for value in per_board],
    }


def relative_improvement(baseline, lever):
    """Declared relative placement improvement; the baseline error must be positive."""
    if baseline <= 0:
        raise ValueError("baseline placement error must be positive")
    return (baseline-lever)/baseline


def attribution_delta(lever_best, baseline_best):
    return lever_best-baseline_best


SIGNED_REDUCTION = 0.5


def signed_reduction(baseline, lever):
    """Per-axis fractional reduction of the signed offset magnitude."""
    reduction = {}
    for axis in ("signed_column_offset_cells", "signed_row_offset_cells"):
        if abs(baseline[axis]) < 1e-12:
            raise ValueError("baseline signed offset must be nonzero")
        reduction[axis] = 1-abs(lever[axis])/abs(baseline[axis])
    return reduction


def signed_offsets_moved(baseline, lever, *, floor=SIGNED_REDUCTION):
    """Declared condition: both signed components fall to at most half their baseline."""
    reduction = signed_reduction(baseline, lever)
    return all(value >= floor for value in reduction.values()), reduction


def translation_verdict(occupancy, ranking, attribution, baseline_placement, lever_placement, *,
                        cell_bar=CELL_F1_THRESHOLD, attribution_floor=ATTRIBUTION_FLOOR,
                        signed_floor=SIGNED_REDUCTION):
    """Three standing gates plus the declared signed-offset condition."""
    moved, reduction = signed_offsets_moved(baseline_placement, lever_placement, floor=signed_floor)
    conditions = {"per_cell_f1": occupancy["f1"] >= cell_bar,
                  "ranking": ranking == "informative",
                  "attribution": attribution >= attribution_floor,
                  "signed_offsets": moved}
    return {"verdict": "cleared" if all(conditions.values()) else "not-cleared",
            "conditions": conditions,
            "failing": [name for name, met in conditions.items() if not met],
            "cell_bar": cell_bar, "attribution_floor": attribution_floor,
            "signed_reduction_floor": signed_floor, "signed_reduction": reduction,
            "signed_column_offset_cells": lever_placement["signed_column_offset_cells"],
            "signed_row_offset_cells": lever_placement["signed_row_offset_cells"],
            "ranking_auc_bar": AUC_THRESHOLD, "ranking_f1_bar": F1_THRESHOLD}


def translation_decomposition(baseline_placement, lever_placement, decision):
    """The declared non-circular rule that names the next suspect after this lever."""
    reduction = decision["signed_reduction"]
    smallest = min(reduction.values())
    occupancy_met = decision["conditions"]["per_cell_f1"] and decision["conditions"]["ranking"]
    if smallest < 0.25:
        rule, suspect = "extent", "box-extent supervision of the lattice box"
    elif decision["conditions"]["signed_offsets"] and not occupancy_met:
        rule, suspect = "read-out_inheritance", "a lattice-agnostic occupancy read-out"
    elif lever_placement["board_error_spread_cells"] >= DISPERSION_FLOOR:
        rule, suspect = "dispersion", "per-board lattice refinement capacity"
    else:
        rule, suspect = "extent", "box-extent supervision of the lattice box"
    return {"rule": rule, "next_suspect": suspect, "signed_reduction": reduction,
            "smallest_signed_reduction": smallest,
            "occupancy_gates_met": occupancy_met,
            "lever_extent_ratio_u": lever_placement["mean_scale_ratio_u"],
            "lever_extent_ratio_v": lever_placement["mean_scale_ratio_v"],
            "lever_board_error_spread_cells": lever_placement["board_error_spread_cells"],
            "floors": {"signed_reduction": decision["signed_reduction_floor"],
                       "extent_unmoved_below": 0.25,
                       "dispersion_cells": DISPERSION_FLOOR}}


EXTENT_REDUCTION = 0.5


def extent_reductions(baseline, lever):
    """Per-axis reduction of |log extent ratio|; a ratio crossing 1.0 counts as a reduction."""
    reduction = {}
    for axis in ("mean_scale_ratio_u", "mean_scale_ratio_v"):
        reference, value = abs(math.log(baseline[axis])), abs(math.log(lever[axis]))
        if reference < 1e-12:
            raise ValueError("baseline extent ratio must differ from 1.0")
        reduction[axis] = 1-value/reference
    return reduction


def extent_offsets_moved(baseline, lever, *, floor=EXTENT_REDUCTION):
    """Declared condition: both extent log-ratios fall to at most half their baseline magnitude."""
    reduction = extent_reductions(baseline, lever)
    return all(value >= floor for value in reduction.values()), reduction


def extent_verdict(occupancy, ranking, attribution, baseline_placement, lever_placement, *,
                   cell_bar=CELL_F1_THRESHOLD, attribution_floor=ATTRIBUTION_FLOOR,
                   extent_floor=EXTENT_REDUCTION):
    """Three standing gates plus the declared extent condition."""
    moved, reduction = extent_offsets_moved(baseline_placement, lever_placement, floor=extent_floor)
    conditions = {"per_cell_f1": occupancy["f1"] >= cell_bar,
                  "ranking": ranking == "informative",
                  "attribution": attribution >= attribution_floor,
                  "extent": moved}
    return {"verdict": "cleared" if all(conditions.values()) else "not-cleared",
            "conditions": conditions,
            "failing": [name for name, met in conditions.items() if not met],
            "cell_bar": cell_bar, "attribution_floor": attribution_floor,
            "extent_reduction_floor": extent_floor, "extent_reduction": reduction,
            "extent_ratio_u": lever_placement["mean_scale_ratio_u"],
            "extent_ratio_v": lever_placement["mean_scale_ratio_v"],
            "ranking_auc_bar": AUC_THRESHOLD, "ranking_f1_bar": F1_THRESHOLD}


def extent_decomposition(baseline_placement, lever_placement, decision):
    """The declared non-circular rule that names the next suspect after this lever."""
    reduction = decision["extent_reduction"]
    smallest = min(reduction.values())
    occupancy_met = decision["conditions"]["per_cell_f1"] and decision["conditions"]["ranking"]
    if smallest < 0.25:
        rule, suspect = "geometry_capacity", "geometric supervision capacity (support anchors)"
    elif decision["conditions"]["extent"] and not occupancy_met:
        rule, suspect = "read_out_inheritance", "a lattice-agnostic occupancy read-out"
    elif lever_placement["board_error_spread_cells"] >= DISPERSION_FLOOR:
        rule, suspect = "dispersion", "per-board lattice refinement capacity"
    else:
        rule, suspect = "scaling", "occupancy read-out at a second scale"
    return {"rule": rule, "next_suspect": suspect, "extent_reduction": reduction,
            "smallest_extent_reduction": smallest, "occupancy_gates_met": occupancy_met,
            "lever_signed_column_offset_cells": lever_placement["signed_column_offset_cells"],
            "lever_signed_row_offset_cells": lever_placement["signed_row_offset_cells"],
            "lever_extent_ratio_u": lever_placement["mean_scale_ratio_u"],
            "lever_extent_ratio_v": lever_placement["mean_scale_ratio_v"],
            "lever_board_error_spread_cells": lever_placement["board_error_spread_cells"],
            "floors": {"extent_reduction": decision["extent_reduction_floor"],
                       "extent_unmoved_below": 0.25, "dispersion_cells": DISPERSION_FLOOR}}


CAPACITY_MOVED_FLOOR = 0.05


def placement_reduction(baseline, lever):
    """Declared placement reduction: 1 - lever error / baseline error, both in target cells."""
    base = baseline["mean_abs_offset_cells"]
    if not base > 0:
        raise ValueError("baseline placement error must be positive")
    return 1-lever["mean_abs_offset_cells"]/base


def capacity_decomposition(baseline_placement, lever_placement, decision):
    """The declared non-circular rule that names the next suspect after the capacity lever.

    No branch can return `translation`, `scale` or `geometry_capacity`: those three levers have been
    run and refused, so they are inputs to this rule, never outputs.
    """
    improvement = decision["placement_improvement"]
    occupancy_met = decision["conditions"]["per_cell_f1"] and decision["conditions"]["ranking"]
    if improvement < CAPACITY_MOVED_FLOOR:
        rule, suspect = "placement_capacity", "per-stage lattice residual capacity"
    elif decision["conditions"]["placement"] and not occupancy_met:
        rule, suspect = "read_out_inheritance", "a lattice-agnostic occupancy read-out"
    elif lever_placement["board_error_spread_cells"] >= DISPERSION_FLOOR:
        rule, suspect = "dispersion", "per-board lattice refinement capacity"
    else:
        rule, suspect = "scaling", "occupancy read-out at a second scale"
    return {"rule": rule, "next_suspect": suspect, "placement_improvement": improvement,
            "improvement_floor": decision["placement_floor"],
            "unmoved_below": CAPACITY_MOVED_FLOOR, "occupancy_gates_met": occupancy_met,
            "lever_mean_abs_offset_cells": lever_placement["mean_abs_offset_cells"],
            "baseline_mean_abs_offset_cells": baseline_placement["mean_abs_offset_cells"],
            "lever_signed_column_offset_cells": lever_placement["signed_column_offset_cells"],
            "lever_signed_row_offset_cells": lever_placement["signed_row_offset_cells"],
            "lever_extent_ratio_u": lever_placement["mean_scale_ratio_u"],
            "lever_extent_ratio_v": lever_placement["mean_scale_ratio_v"],
            "lever_board_error_spread_cells": lever_placement["board_error_spread_cells"],
            "floors": {"placement_improvement": decision["placement_floor"],
                       "unmoved_below": CAPACITY_MOVED_FLOOR, "dispersion_cells": DISPERSION_FLOOR}}


def decomposition(placement, placement_delta):
    """The declared precedence that names the next measurable suspect from a refusal."""
    if abs(placement["signed_column_offset_cells"]) >= TRANSLATION_FLOOR or \
            abs(placement["signed_row_offset_cells"]) >= TRANSLATION_FLOOR:
        rule, suspect = "translation", "box-origin supervision of the lattice box"
    elif abs(math.log(placement["mean_scale_ratio_u"])) >= SCALE_FLOOR or \
            abs(math.log(placement["mean_scale_ratio_v"])) >= SCALE_FLOOR:
        rule, suspect = "scale", "box-extent supervision of the lattice box"
    elif placement["board_error_spread_cells"] >= DISPERSION_FLOOR:
        rule, suspect = "dispersion", "per-board lattice refinement capacity"
    elif placement_delta >= PLACEMENT_IMPROVEMENT:
        rule, suspect = "read-out_inheritance", "a lattice-agnostic occupancy read-out"
    else:
        rule, suspect = "geometry_capacity", "geometric supervision capacity (support anchors)"
    return {"rule": rule, "next_suspect": suspect,
            "signed_column_offset_cells": placement["signed_column_offset_cells"],
            "signed_row_offset_cells": placement["signed_row_offset_cells"],
            "mean_scale_ratio_u": placement["mean_scale_ratio_u"],
            "mean_scale_ratio_v": placement["mean_scale_ratio_v"],
            "board_error_spread_cells": placement["board_error_spread_cells"],
            "placement_improvement": placement_delta,
            "floors": {"translation_cells": TRANSLATION_FLOOR, "scale_log_ratio": SCALE_FLOOR,
                       "dispersion_cells": DISPERSION_FLOOR,
                       "placement_improvement": PLACEMENT_IMPROVEMENT}}


def verdict(occupancy, ranking, attribution, placement_improvement, *,
            cell_bar=CELL_F1_THRESHOLD, attribution_floor=ATTRIBUTION_FLOOR,
            placement_floor=PLACEMENT_IMPROVEMENT):
    """All four declared conditions; a tie at any boundary counts as met."""
    conditions = {"per_cell_f1": occupancy["f1"] >= cell_bar,
                  "ranking": ranking == "informative",
                  "attribution": attribution >= attribution_floor,
                  "placement": placement_improvement >= placement_floor}
    return {"verdict": "cleared" if all(conditions.values()) else "not-cleared",
            "conditions": conditions,
            "failing": [name for name, met in conditions.items() if not met],
            "cell_bar": cell_bar, "attribution_floor": attribution_floor,
            "placement_floor": placement_floor,
            "ranking_auc_bar": AUC_THRESHOLD, "ranking_f1_bar": F1_THRESHOLD}


def summarize(probe_cases):
    """Everything a curator needs, re-derivable from the probe's own stored cells."""
    occupancy = score(probe_cases)
    placement = stats(probe_cases)
    return {"occupancy": occupancy, "placement": placement,
            "ranking": sweep_verdict({"auc": occupancy["auc"],
                                      "best": {"f1": occupancy["f1"]}})}
