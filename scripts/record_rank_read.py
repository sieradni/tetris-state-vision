"""Curate an ordering-lever block (M25 one-sided pair order, M26 symmetric pair margin) under one protocol.

`docs/M25_PAIR_ORDER.md` and `docs/M26_GAP_MARGIN.md` each declare their arms before any receipt. This recorder is shared
by both so the two blocks cannot drift: it asserts each arm's declared fields against its own run summary and nothing
else, checks that the reproduction arm reproduces the published control exactly, scores every arm at the standing boundary
over the standing 81-point grid, takes the decomposition quantities M24 defined (strict exact boards, exact-level search,
inversions, within-board AUC), splits every free-channel arm into its lattice, pooled and local parts, and prints the
reading the declaration assigns - it selects no threshold and edits no guard.
"""
import argparse
import json
from pathlib import Path

from decompose_exactness import board_auc, exact_level_search, inversions
from measure_row_error import arm_block
from occupancy_sweep import evaluate, threshold_grid
from probe_model import require_bounded_job

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 1900*1024**2
STEPS = 64
FORWARDS = 54
BOARDS = 28
PUBLISHED_CONTROL_AUC = 0.8226433610683003
PUBLISHED_CONTROL_F1 = 0.38164
FROZEN_FROM = ROOT/"docs/evidence/m23-contrast-r1/scope.json"
FROZEN_ARMS = ("control", "stencil", "norm", "norm-cross", "envelope", "centred")
AUC_TOLERANCE = .02
INVERSION_FALL = .01
READINGS = ("pair-order-clears-exactness", "pair-order-moves-without-clearing", "pair-order-fails")
BASE_DECLARED = ("occupancy_lattice_free", "occupancy_stack_monotone", "occupancy_crossing_threshold",
                 "occupancy_free_stencil")
SPECS = {
    "m25-rank": {
        "instrument": "m25-rank-read-r1",
        "predeclaration": "docs/M25_PAIR_ORDER.md",
        "declared": BASE_DECLARED+("occupancy_pair_rank",),
        # (lattice_free, stack_monotone, crossing_threshold, free_stencil, pair_rank)
        "arms": {"repro": (0, 0, 0, 0, 0), "rank": (2, 0, 0, 0, 1), "rank-stencil": (2, 1, 0, 1, 1)},
        "references": {"repro": "control", "rank": "control", "rank-stencil": "stencil"},
        "constants": {"pair_rank_margin": 1.0, "pair_rank_weight": .5},
    },
    "m26-gap": {
        "instrument": "m26-gap-read-r1",
        "predeclaration": "docs/M26_GAP_MARGIN.md",
        "declared": BASE_DECLARED+("occupancy_pair_rank", "occupancy_pair_gap"),
        # (lattice_free, stack_monotone, crossing_threshold, free_stencil, pair_rank, pair_gap)
        "arms": {"repro": (0, 0, 0, 0, 0, 0), "gap": (2, 0, 0, 0, 0, 1), "gap-stencil": (2, 1, 0, 1, 0, 1)},
        "references": {"repro": "control", "gap": "control", "gap-stencil": "stencil"},
        "constants": {"pair_gap_margin": 1.0, "pair_gap_weight": .5},
    },
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_base(family, arm):
    return f"{family}-{arm}-r1"


def declared_fields(spec, arm):
    return dict(zip(spec["declared"], spec["arms"][arm]))


def part_auc(cases, name):
    """AUC of one part of the read on the same boards, so a channel's contribution is measured on its own.

    The published read has no parts - it is a single channel - so an arm without them reports `None` rather than a
    number invented from a channel it does not have.
    """
    if not all(name in case.get("logits_parts", {}) for case in cases):
        return None
    positive = [value for case in cases for value, label in zip(case["logits_parts"][name], case["occupied"])
                if label]
    negative = [value for case in cases for value, label in zip(case["logits_parts"][name], case["occupied"])
                if not label]
    if not positive or not negative:
        return None
    wins = sum(1 for high in positive for low in negative if high > low)
    ties = sum(1 for high in positive for low in negative if high == low)
    return round((wins+.5*ties)/(len(positive)*len(negative)), 6)


def mean_inversion_rate(cases, rows, columns):
    rates = []
    for case in cases:
        logits = [[float(case["logits"][row*columns+column]) for column in range(columns)]
                  for row in range(rows)]
        labels = [[bool(case["occupied"][row*columns+column]) for column in range(columns)]
                  for row in range(rows)]
        pairs = (sum(1 for row in range(rows) for column in range(columns) if labels[row][column]) *
                 sum(1 for row in range(rows) for column in range(columns) if not labels[row][column]))
        rates.append(inversions(logits, labels)/pairs if pairs else 0.0)
    return sum(rates)/len(rates)


def arm_record(family, arm, spec):
    base = run_base(family, arm)
    summary = read(ROOT/"runs"/base/"summary.json")
    sweep = read(ROOT/"runs"/(base+"-sweep-r1")/"result.json")
    arguments = summary["arguments"]
    contract = summary["recovery_contract"]
    for field, value in declared_fields(spec, arm).items():
        if arguments.get(field, 0) != value or contract.get(field, 0) != value:
            raise ValueError("%s: %s differs from the arm's declaration" % (arm, field))
    if summary.get("steps") != STEPS:
        raise ValueError("%s: the study did not run its declared 64 steps" % arm)
    config = sweep["config"]
    rows, columns = config["visible_rows"], config["columns"]
    scored = evaluate([case for case in sweep["cases"] if case["split"] == "training"])
    zero = [row for row in scored["rows"] if row["threshold"] == 0.0]
    if len(zero) != 1:
        raise ValueError("%s: the grid must carry exactly one 0.0 boundary" % arm)
    zero = zero[0]
    rows_at_zero = arm_block(sweep, arm)["zero"]["error_rows"]
    cases = [case for case in sweep["cases"] if case["split"] == "training" and any(case["occupied"])]
    if len(cases) != BOARDS:
        raise ValueError("%s: %d boards, expected %d" % (arm, len(cases), BOARDS))
    exact = boundary = 0
    inverted = []
    within = []
    for case in cases:
        logits = [[float(case["logits"][row*columns+column]) for column in range(columns)]
                  for row in range(rows)]
        labels = [[bool(case["occupied"][row*columns+column]) for column in range(columns)]
                  for row in range(rows)]
        search = exact_level_search(logits, labels)
        wrong = sum(1 for row in range(rows) for column in range(columns)
                    if (logits[row][column] > 0.0) != labels[row][column])
        exact += int(wrong == 0)
        boundary += int(search["exact_at_any_level"])
        inverted.append(inversions(logits, labels))
        separation = board_auc(logits, labels)
        if separation is not None:
            within.append(separation)
    return {
        "arm": arm, "study": base, "sweep": base+"-sweep-r1",
        "declared": {field: arguments.get(field, 0) for field in spec["declared"]},
        "auc": scored["auc"], "f1_at_zero": zero["f1"],
        "best": {"threshold": scored["best"]["threshold"], "f1": round(scored["best"]["f1"], 6)},
        "row_error_at_zero": {"mean": rows_at_zero["mean"], "median": rows_at_zero["median"],
                              "cells": rows_at_zero["cells"]},
        "positive_share_at_zero": round((zero["true_positive"]+zero["false_positive"]) /
                                        (scored["occupied_cells"]+scored["empty_cells_on_nonempty_boards"]), 6),
        "precision_at_zero": round(zero["precision"], 6), "recall_at_zero": round(zero["recall"], 6),
        "exact_boards_at_zero": {"boards": len(cases), "exact": exact,
                                 "grid_exact": zero["exact_nonempty_boards"], "exact_at_some_level": boundary},
        "inversions": {"median": sorted(inverted)[len(inverted)//2], "max": max(inverted),
                       "mean_inversion_rate": round(mean_inversion_rate(cases, rows, columns), 6)},
        "mean_auc_within_board": round(sum(within)/len(within), 6),
        "channel_split": {name: part_auc(cases, name) for name in ("lattice", "free", "local")},
        "steps": summary["steps"],
    }


def reference(arm, spec):
    """The frozen arm whose read this one inherits, with M24's own decomposition numbers for it."""
    scope = read(ROOT/"docs/evidence/m24-exactness-r1/scope.json")
    name = spec["references"][arm]
    return name, scope["arms"][name]["auc"], scope["arms"][name]["mean_inversion_rate"]


def verdict(arm, record, spec):
    """The record's own vocabulary: did the term clear exactness, move without clearing, or fail?"""
    if record["exact_boards_at_zero"]["exact"] > 0:
        return "pair-order-clears-exactness"
    if arm == "repro":
        # The reproduction arm declares no lever: it exists so the harness can be shown to be unchanged rather than so
        # it can be judged as a mechanism.
        return "pair-order-fails"
    name, base_auc, base_inversion = reference(arm, spec)
    improved_ordering = record["inversions"]["mean_inversion_rate"] <= base_inversion-INVERSION_FALL
    improved_auc = record["auc"] >= base_auc+AUC_TOLERANCE
    if improved_ordering or improved_auc:
        return "pair-order-moves-without-clearing"
    return "pair-order-fails"


def main():
    bounded = require_bounded_job()
    if bounded != CEILING_BYTES:
        raise ValueError("this curation must run at its declared 1900 MiB ceiling")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(SPECS), default="m25-rank")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("an evidence directory already exists")
    spec = SPECS[args.family]
    frozen_scope = read(FROZEN_FROM)
    frozen = {name: {"auc": frozen_scope["arms"][name]["auc"],
                     "f1_at_zero": frozen_scope["arms"][name]["f1_at_zero"],
                     "exact": frozen_scope["arms"][name]["exact_boards_at_zero"]["exact"],
                     "row_error": frozen_scope["arms"][name]["row_error_at_zero"]["mean"]}
              for name in FROZEN_ARMS}
    arms = {arm: arm_record(args.family, arm, spec) for arm in spec["arms"]}
    if abs(arms["repro"]["auc"]-PUBLISHED_CONTROL_AUC) > 1e-9:
        raise ValueError("the reproduction arm does not reproduce the published control AUC")
    if abs(arms["repro"]["f1_at_zero"]-PUBLISHED_CONTROL_F1) > 1e-6:
        raise ValueError("the reproduction arm does not reproduce the published control F1")
    for arm, record in arms.items():
        record["reading"] = verdict(arm, record, spec)
        if record["reading"] not in READINGS:
            raise ValueError("unrecorded reading")
    scope = {
        "status": "passed", "instrument": spec["instrument"], "family": args.family,
        "declared_receipt": args.receipt, "declared_memory_bytes": bounded,
        "predeclaration": spec["predeclaration"], "steps": STEPS, "forwards": FORWARDS,
        "boards": BOARDS, "grid_points": len(threshold_grid()),
        "arms_differ_in": list(spec["declared"]),
        "frozen_from": str(FROZEN_FROM.relative_to(ROOT)),
        "frozen": frozen, "arms": arms,
        "reproduced": {"arm": "repro", "auc": PUBLISHED_CONTROL_AUC, "f1_at_zero": PUBLISHED_CONTROL_F1,
                       "exact": True},
        "declared_constants": spec["constants"],
        "verdicts": {arm: arms[arm]["reading"] for arm in spec["arms"]},
    }
    args.output.mkdir(parents=True)
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(scope, stream, indent=2, allow_nan=False)
    print(json.dumps({arm: {key: arms[arm][key] for key in
                            ("auc", "f1_at_zero", "exact_boards_at_zero", "row_error_at_zero",
                             "positive_share_at_zero", "mean_auc_within_board", "inversions",
                             "channel_split", "reading")}
                      for arm in spec["arms"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
