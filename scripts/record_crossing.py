"""Curate the label-free per-board crossing rule against the reproduced control.

`docs/M16_CROSSING_THRESHOLD.md` declares the mechanism, the arms, the readings and the falsification before any receipt.
This recorder asserts that the treated arm carries the three declared fields and nothing else, that each sweep carries
its own study's checkpoint while the neighbour's (M15's combined arm) is excluded by name, that the control reproduces
the published numbers exactly, and it reports the plain-binary F1 at the rule's own crossing - which, because the rule
subtracts its level inside the head, is simply the F1 at `0.0` - beside the fixed-threshold ceiling of the same read.
"""
import argparse
import json
from pathlib import Path
import sys

from measure_row_error import boundaries
from occupancy_sweep import AUC_THRESHOLD, F1_THRESHOLD, sweep_verdict, threshold_grid
from percell_gate import CELL_F1_THRESHOLD
from probe_model import require_bounded_job
from record_demean_lever import arm_summary, components_of, lever_verdict
from record_representation_lever import read

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 640*1024**2
CURATION_CEILING_BYTES = 1900*1024**2
STEPS = 64
FORWARDS = 54
BOARDS = 28
OCCUPIED_CELLS = 667
KNOWN_CELLS = BOARDS*200
DECIDING_IMAGES = 36
ATTRIBUTION_DELTA = .05
AUC_TOLERANCE = .02
FAMILY = "m16-crossing"
INSTRUMENT = "m16-crossing-threshold-r1"
PREDECLARATION = "docs/M16_CROSSING_THRESHOLD.md"
PREVIOUS = ROOT/"docs/evidence/m15-stack-combined-r1/scope.json"
ENVELOPE = ROOT/"docs/evidence/m15-stack-contiguity-r1/scope.json"
CENTRING = ROOT/"docs/evidence/m14-read-centring-r1/scope.json"
MIDPOINT = ROOT/"docs/evidence/m13-midpoint-r1/scope.json"
DENSE = ROOT/"docs/evidence/m12-dense-hinge-r1/scope.json"
QUANTILE = ROOT/"docs/evidence/m11-quantile-anchor-r1/scope.json"
PRIOR = ROOT/"docs/evidence/m10-supervised-anchor-r1/scope.json"
HINGE = ROOT/"docs/evidence/m9-supervised-margin-r1/scope.json"
CONTROL = "control"
TREATED = "ruled"
RUNS = {CONTROL: "control", TREATED: "ruled"}
READINGS = ("crossing-clears-the-ceiling", "crossing-moves-without-clearing", "crossing-fails")
SECONDARY = ("ranking-preserved", "ranking-degraded")
SUPERVISION = ("occupancy_midpoint", "occupancy_dense_hinge", "occupancy_quantile_anchor", "occupancy_anchor",
               "occupancy_margin_hinge")
READ_LEVERS = ("occupancy_normalise", "occupancy_offset_rows", "occupancy_deform", "occupancy_context",
               "occupancy_detail_stencil", "occupancy_resolution", "occupancy_column", "occupancy_lattice_free",
               "occupancy_placed")
DECLARED = ("occupancy_centre", "occupancy_crossing_threshold", "occupancy_stack_monotone")


def require_sweep(name):
    """Both arms must be the declared measurement, of their own run, under the declared envelope."""
    study = f"{FAMILY}-{RUNS[name]}-r1"
    sweep = f"{study}-sweep-r1"
    result = read(ROOT/"runs"/sweep/"result.json")
    job = read(ROOT/"runs"/(sweep+"-job.json"))
    summary = read(ROOT/"runs"/study/"summary.json")
    study_job = read(ROOT/"runs"/(study+"-job.json"))
    if (job.get("status") != "passed" or job.get("exit_code") != 0 or
            job.get("memory_bytes") != CEILING_BYTES or job.get("work_items") != FORWARDS):
        raise ValueError(f"{name} sweep did not pass under the declared 640 MiB / 54 work items")
    if (result.get("status") != "passed" or result.get("run") != study or
            result.get("forwards") != FORWARDS or
            result.get("weights_sha256") != summary.get("checkpoint_sha256")):
        raise ValueError(f"{name} sweep is not a complete measurement of its own run")
    if sweep_verdict(result["deciding"]) != result["verdict"]:
        raise ValueError(f"{name} sweep verdict does not re-derive")
    if (summary.get("steps") != STEPS or summary.get("status") != "complete" or
            study_job.get("status") != "passed" or study_job.get("work_items") != STEPS):
        raise ValueError(f"{name} study is not the declared 64-step budget")
    if (result["deciding"]["occupied_cells"] != OCCUPIED_CELLS or
            len(result["deciding"]["thresholds"]) != 81 or 0.0 not in result["deciding"]["thresholds"]):
        raise ValueError(f"{name} sweep did not decide the standing population on the standing grid")
    declared = 1 if name == TREATED else 0
    # Both declared fields, in the config, the arguments and the recovery contract: the treated arm is the
    # combination, and the control must carry neither.
    for field in DECLARED:
        if (summary.get("config", {}).get(field) != declared or
                summary.get("arguments", {}).get(field) != declared or
                summary.get("recovery_contract", {}).get(field) != declared):
            raise ValueError(f"{name} run does not carry the declared {field}={declared}")
    # No added term may sit beside the structural option, and no lever may move what is read or where zero sits.
    for key in SUPERVISION+READ_LEVERS:
        if summary.get("config", {}).get(key, 0) or summary.get("arguments", {}).get(key, 0):
            raise ValueError(f"{name} carries an undeclared lever beside the monotone read: {key}")
    return study, summary, result


def training_occupancy(study):
    """The occupancy loss the arm paid, since a structural option adds no term of its own."""
    log = ROOT/"runs"/(study+"-study.log")
    if not log.exists():
        raise ValueError(f"{study} step log is missing")
    values = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if '"terms":' not in line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        terms = record.get("terms") or {}
        if "occupancy" in terms and type(record.get("step")) is int:
            values.append((record["step"], float(terms["occupancy"])))
    if len(values) != STEPS:
        raise ValueError(f"{study} does not log the occupancy term at every declared step")
    return {"steps": len(values), "first": round(values[0][1], 6), "last_step": values[-1][0],
            "last": round(values[-1][1], 6), "mean": round(sum(v for _, v in values)/len(values), 6),
            "source": str(log.relative_to(ROOT)), "in_sample": True,
            "reading": "the occupancy loss the arm paid; this arm adds no term of its own, so there is no separate "
                       "charge to report - the structural option removes the level instead of charging it"}


def row_error_at_zero(result):
    config = result["config"]
    rows, columns = config["visible_rows"], config["columns"]
    errors = []
    for case in result["cases"]:
        if case["split"] != "training" or not any(case["occupied"]):
            continue
        decided, occupied = boundaries(case["logits"], case["occupied"], rows, columns, 0.0)
        errors.extend(got-want for got, want in zip(decided, occupied) if got is not None and want is not None)
    if not errors:
        raise ValueError("row diagnostic carried no columns")
    return {"mean": round(sum(errors)/len(errors), 6), "columns": len(errors)}


def exact_boards_at_zero(result):
    config = result["config"]
    rows, columns = config["visible_rows"], config["columns"]
    exact = total = 0
    for case in result["cases"]:
        if case["split"] != "training" or not any(case["occupied"]):
            continue
        decided, occupied = boundaries(case["logits"], case["occupied"], rows, columns, 0.0)
        pairs = [(got, want) for got, want in zip(decided, occupied) if got is not None and want is not None]
        if not pairs:
            continue
        total += 1
        exact += all(got == want for got, want in pairs)
    if not total:
        raise ValueError("the fit gate carried no board")
    return {"boards": total, "exact": exact}


def reference(path, name):
    scope = read(path)
    arm = scope["arms"][name]
    return {"auc": arm["auc"], "f1_at_zero": arm["f1_at_zero"], "best_f1": arm["best"]["f1"],
            "out_of_fold_f1": arm["out_of_fold_f1"], "scope": str(path.relative_to(ROOT))}


def main():
    bounded = require_bounded_job()
    if bounded != CURATION_CEILING_BYTES:
        raise ValueError("curation must run at its declared 1,900 MiB ceiling")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("an evidence directory already exists")
    prior = read(PRIOR)
    previous_scope = read(PREVIOUS)
    if prior.get("instrument") != "m10-supervised-anchor-r1":
        raise ValueError("the reproduction reference is not the committed extremal-anchor block")
    expected_auc, expected_f1 = prior["arms"][CONTROL]["auc"], prior["arms"][CONTROL]["f1_at_zero"]

    grid = tuple(threshold_grid())
    arms, reproduction = {}, {}
    for name in (CONTROL, TREATED):
        study, summary, result = require_sweep(name)
        scored = arm_summary(result["deciding"])
        counts = scored["operating_counts"]
        positives = counts["true_positive"]+counts["false_positive"]
        arms[name] = {
            "study": study, "sweep": f"{study}-sweep-r1",
            "occupancy_stack_monotone": summary["config"]["occupancy_stack_monotone"],
            "occupancy_crossing_threshold": summary["config"]["occupancy_crossing_threshold"],
            "occupancy_centre": summary["config"]["occupancy_centre"],
            "weights_sha256": result["weights_sha256"],
            "config": {key: summary["config"][key] for key in ("occupancy_stack_monotone", "occupancy_centre", "occupancy_crossing_threshold",
                                                               "occupancy_normalise", "occupancy_detail", "occupancy_offset_rows",
                                                               "dimension", "component_capacity")},
            "catalogue_sha256": summary["catalogue_sha256"],
            "study_elapsed_seconds": summary["elapsed_seconds"],
            "auc": result["deciding"]["auc"],
            "f1_at_zero": scored["per_cell_f1"],
            "operating_counts": counts,
            "positives_at_zero": positives,
            "positive_share_at_zero": round(positives/KNOWN_CELLS, 6),
            "precision": round(counts["true_positive"]
                               /max(1, counts["true_positive"]+counts["false_positive"]), 6),
            "recall": round(counts["true_positive"]
                            /max(1, counts["true_positive"]+counts["false_negative"]), 6),
            "best": {"threshold": result["deciding"]["best"]["threshold"],
                     "f1": result["deciding"]["best"]["f1"],
                     "precision": result["deciding"]["best"]["precision"]},
            "out_of_fold_f1": components_of([case for case in result["cases"]
                                             if case["split"] == "training"], grid)["out_of_fold_f1"],
            "exact_boards_at_zero": exact_boards_at_zero(result),
            "row_error_at_zero": row_error_at_zero(result),
            "training_occupancy": training_occupancy(study),
            "verdict": result["verdict"],
        }
        reproduction[name] = {"auc": result["deciding"]["auc"], "f1_at_zero": scored["per_cell_f1"],
                              "prior_scope": str(PRIOR.relative_to(ROOT))}
    if len({arms[name]["catalogue_sha256"] for name in arms}) != 1:
        raise ValueError("the arms were captured under different catalogues")
    control, treated = arms[CONTROL], arms[TREATED]
    if control["auc"] != expected_auc or control["f1_at_zero"] != expected_f1:
        raise ValueError(f"control does not reproduce the recorded {expected_auc!r} / {expected_f1!r}")

    # The aliasing guard a structural lever needs: the two arms must be two measurements, not one measurement twice,
    # and the only difference between them must be the declared field.
    differing = sorted(key for key in control["config"] if control["config"][key] != treated["config"][key])
    if differing != sorted(DECLARED):
        raise ValueError(f"the arms differ in more than the declared field: {differing}")
    if treated["weights_sha256"] == control["weights_sha256"]:
        raise ValueError("the two arms scored identical weights, so one of them did not measure its own run")
    left = next(name for name in previous_scope["arms"] if name != "control")
    if treated["weights_sha256"] == previous_scope["arms"][left]["weights_sha256"]:
        raise ValueError("the ruled arm scored the neighbour block's checkpoint")

    delta = round(treated["f1_at_zero"]-control["f1_at_zero"], 6)
    auc_delta = round(treated["auc"]-control["auc"], 6)
    overshot = treated["best"]["threshold"] < 0.0
    ceiling = previous_scope["arms"][left]["f1_at_zero"]
    if delta >= ATTRIBUTION_DELTA and treated["f1_at_zero"] > ceiling:
        reading = "crossing-clears-the-ceiling"
        reading_text = ("Deciding each board at its own crossing clears the fixed-threshold ceiling of the same read "
                        "and the standing attribution floor. The level was the obstacle all along, and the read "
                        "states the level itself without ever seeing a label.")
    elif treated["f1_at_zero"] > ceiling:
        reading = "crossing-moves-without-clearing"
        reading_text = ("Deciding at the read's own crossing beats the same read at the fixed `0.0` threshold, but "
                        "short of the attribution floor: the rule recovers part of the gap between the fixed "
                        "boundary and the read's own optimum, not all of it.")
    else:
        reading = "crossing-fails"
        reading_text = ("Deciding at the read's own crossing does not beat the same read at the fixed `0.0` threshold. "
                        "The decision-rule hypothesis is falsified on this population: a per-board rule that cannot "
                        "see prevalence is not entitled to the population-level optimum, and no threshold grid or "
                        "swept number is offered in its place.")
    if reading not in READINGS:
        raise ValueError("an undeclared reading was produced")
    secondary = "ranking-preserved" if auc_delta >= -AUC_TOLERANCE else "ranking-degraded"
    scope = {
        "status": "passed",
        "instrument": INSTRUMENT,
        "predeclaration": PREDECLARATION,
        "scope": ("The M15 block's own 64-step control against a 64-step arm carrying the per-column monotone "
                  "monotone envelope inside the head before the objective sees it - captured in one catalogue and "
                  "scored read-only at the standing 0.0 boundary and over the standing 81-point grid. The option is "
                  "structural: no label, no threshold, no fitted parameter, no prevalence, no statistic of the "
                  "population. 0 is exactly the published function. No threshold change, no population change and "
                  "no real-image, complete-state or quality claim."),
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "mechanism": {
            "name": "label-free-crossing-decision",
            "flag": "--occupancy-crossing-threshold 1 (on the envelope and centred read)",
            "definition": "per column, the suffix maximum of the visible logits in bottom-first row order "
                          "(equivalently a cumulative maximum from the top of the window down to the floor), "
                          "applied inside the head before the objective or any scorer sees the tensor",
            "where": "model head, before the loss and before every consumer",
            "parameters_added": 0, "tensors_added": 0, "loss_terms_added": 0,
            "weight": None, "margin": None,
            "statistic": "none: monotonicity is the domain's own constraint on a locked stack, so nothing about "
                         "the population is read",
            "orientation": "non-increasing upward, because material rests on the floor",
            "crossings_per_column": "at most one, and the crossing is the column's height",
            "invariance": "equivariant to a board-wide shift (envelope(L + c) == envelope(L) + c), so unlike "
                          "centring this transform leaves the level free and places the crossing instead",
            "declared_before_receipts": True,
            "unknown_cells": "spawn rows are outside the visible window and pass through untouched",
            "identity_at_zero": "occupancy_stack_monotone 0 is exactly the published function",
            "composition": "the envelope places the crossing, then centring sets its level to the board's own "
                           "visible mean; the declared order is fixed in the head and asserted in the tests",
            "supersedes": "nothing: two structural options on the read, still refused beside every supervision "
                          "term, the normalised output and every read-moving lever",
            "expected_reading": "crossing-moves-without-clearing",
            "label_free": True,
            "label_free_proof": "the rule is a function of (logits, visible_rows) only, applied before any loss "
                                 "is evaluated; tests permute, invert and zero the labels and require "
                                 "bit-identical output, and scan the executable body for any label reference",
            "selection": "exact Otsu split over every gap between consecutive sorted visible logits, level = the "
                          "midpoint of the selected gap",
            "falsification": "falsified if the primary at the rule's crossing does not exceed the "
                              "fixed-threshold ceiling of the same read",
            "placement_ceiling": .412034,
            "placement_ceiling_note": "control's own swept maximum over the standing 81-point grid is 0.412034, "
                                      "0.0196 below the 0.431640 bar, so no placement-only transform can clear "
                                      "the primary; the informative readings are placement, row error and exact "
                                      "boards",
            "failure_mode": "the rule cannot see prevalence, so on a 12%-occupied population its split can sit "
                            "above or below the F1-optimal level; it is declared falsified if it fails to beat "
                            "the same read at the fixed boundary",
        },
        "envelope": {"steps": STEPS, "ceiling_bytes": CEILING_BYTES,
                     "work_items": {"sweep": FORWARDS, "curation": 9},
                     "attribution_floor": ATTRIBUTION_DELTA, "auc_tolerance": AUC_TOLERANCE},
        "arms": arms,
        "contrasts": {
            "centre_control": {"f1_at_zero_delta": delta, "auc_delta": auc_delta,
                               "best_f1_delta": round(treated["best"]["f1"]-control["best"]["f1"], 6),
                               "best_threshold_delta": round(treated["best"]["threshold"]
                                                             - control["best"]["threshold"], 6),
                               "out_of_fold_delta": round(treated["out_of_fold_f1"]
                                                          - control["out_of_fold_f1"], 6),
                               "positive_share_delta": round(treated["positive_share_at_zero"]
                                                             - control["positive_share_at_zero"], 6),
                               "row_error_delta": round(treated["row_error_at_zero"]["mean"]
                                                        - control["row_error_at_zero"]["mean"], 6),
                               "occupancy_loss_delta": round(treated["training_occupancy"]["mean"]
                                                             - control["training_occupancy"]["mean"], 6)},
        },
        "aliasing_guard": {
            "arms_differ_in": differing,
            "weights_differ": treated["weights_sha256"] != control["weights_sha256"],
            "not_the_previous_block": treated["weights_sha256"] != previous_scope["arms"][left]["weights_sha256"],
            "rule": "a structural option changes the read globally, so the block proves it measured its own "
                    "checkpoint: each sweep's weights equal its own study's checkpoint, the arms' weights differ, "
                    "the previous block's checkpoint is excluded by name, and the configs differ in exactly the "
                    "declared field",
        },
        "reference": {
            "combined_read": reference(PREVIOUS, left),
            "envelope": reference(ENVELOPE, "contiguous"),
            "centring": reference(CENTRING, "centred"),
            "midpoint_anchor": reference(MIDPOINT, "level"),
            "dense_hinge": reference(DENSE, "banded"),
            "quantile_anchor": reference(QUANTILE, "bulk"),
            "extremal_anchor": reference(PRIOR, "anchor"),
            "hinge": reference(HINGE, "hinge"),
            "role": "reference-only: the M15 combined read, the M15 envelope arm, the M14 centred read, the M13 midpoint anchor, the M12 dense hinge, the M11 "
                    "quantile anchor, the M10 extremal anchor and the M9 hinge, each measured in its own catalogue "
                    "and none re-measured or re-verdicted here",
        },
        "reading": reading,
        "reading_text": reading_text,
        "secondary_reading": secondary,
        "secondary_text": ("Deciding AUC with the standing -0.02 tolerance. A maximum can only raise cells, so a "
                           "ranking change here is the training's response to a constrained read, not the transform's "
                           "own doing, and it is reported as the "
                           "measured price."),
        "fixed_threshold_ceiling": {"arm": left, "scope": str(PREVIOUS.relative_to(ROOT)),
                                    "f1_at_zero": ceiling,
                                    "rule_primary": treated["f1_at_zero"]},
        "ceiling_broken": treated["f1_at_zero"] > ceiling,
        "verdict": lever_verdict(treated["f1_at_zero"], treated["verdict"], delta),
        "gates": {"per_cell_f1": CELL_F1_THRESHOLD, "ranking_auc": AUC_THRESHOLD,
                  "ranking_f1": F1_THRESHOLD, "attribution_floor": ATTRIBUTION_DELTA, "grid_points": 81},
        "fit_gate": {"exact_nonempty_boards": treated["exact_boards_at_zero"]["exact"],
                     "boards": treated["exact_boards_at_zero"]["boards"],
                     "cleared": (treated["exact_boards_at_zero"]["exact"]
                                 == treated["exact_boards_at_zero"]["boards"] == BOARDS),
                     "boundary": 0.0, "measured_on": TREATED},
        "reproduction": reproduction,
        "population": {"split": "training", "images": DECIDING_IMAGES, "boards": BOARDS,
                       "occupied_cells": OCCUPIED_CELLS, "known_cells": KNOWN_CELLS,
                       "prevalence": round(OCCUPIED_CELLS/KNOWN_CELLS, 6)},
    }
    args.output.mkdir(parents=True)
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(scope, stream, indent=2, allow_nan=False)
    print(json.dumps({key: scope[key] for key in ("reading", "secondary_reading", "contrasts",
                                                  "aliasing_guard", "verdict", "fit_gate")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
