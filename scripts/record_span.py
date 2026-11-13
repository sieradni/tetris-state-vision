"""Curate the declared row-span tap against the reproduced published read on the corrected strict metric.

`docs/M18_TAP_SPAN.md` declares the geometry, the falsification and the readings before any receipt. This recorder
asserts the one declared field and nothing else, that each sweep carries its own study's checkpoint while the neighbour
block's is excluded by name, that the control reproduces the published numbers exactly, and it reports the fit gate under
**both** strict definitions V-139 left behind: every visible cell right, and every column's decided top right with an empty
column compared as a value.
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
FAMILY = "m18-span"
INSTRUMENT = "m18-tap-span-r1"
PREDECLARATION = "docs/M18_TAP_SPAN.md"
PREVIOUS = ROOT/"docs/evidence/m15-stack-combined-r1/scope.json"
SKYLINE = ROOT/"docs/evidence/m17-skyline-recovery-r1/scope.json"
RULED = ROOT/"docs/evidence/m16-crossing-threshold-r1/scope.json"
ENVELOPE = ROOT/"docs/evidence/m15-stack-contiguity-r1/scope.json"
CENTRING = ROOT/"docs/evidence/m14-read-centring-r1/scope.json"
MIDPOINT = ROOT/"docs/evidence/m13-midpoint-r1/scope.json"
DENSE = ROOT/"docs/evidence/m12-dense-hinge-r1/scope.json"
QUANTILE = ROOT/"docs/evidence/m11-quantile-anchor-r1/scope.json"
PRIOR = ROOT/"docs/evidence/m10-supervised-anchor-r1/scope.json"
HINGE = ROOT/"docs/evidence/m9-supervised-margin-r1/scope.json"
CONTROL = "control"
TREATED = "tap"
RUNS = {CONTROL: "control", TREATED: "tap"}
READINGS = ("span-clears-the-ceiling", "span-moves-without-clearing", "span-fails")
SECONDARY = ("ranking-preserved", "ranking-degraded")
SUPERVISION = ("occupancy_midpoint", "occupancy_dense_hinge", "occupancy_quantile_anchor", "occupancy_anchor",
               "occupancy_margin_hinge")
READ_LEVERS = ("occupancy_normalise", "occupancy_offset_rows", "occupancy_deform", "occupancy_context",
               "occupancy_detail_stencil", "occupancy_resolution", "occupancy_column", "occupancy_lattice_free",
               "occupancy_placed", "occupancy_centre", "occupancy_stack_monotone", "occupancy_crossing_threshold",
               "occupancy_skyline_recovery")
DECLARED = ("occupancy_tap_span",)
RECORDED_CONFIG = ("occupancy_tap_span", "occupancy_detail", "occupancy_normalise", "occupancy_centre",
                   "occupancy_stack_monotone", "occupancy_crossing_threshold", "occupancy_skyline_recovery",
                   "dimension", "component_capacity")


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
    for field in DECLARED:
        if (summary.get("config", {}).get(field) != declared or
                summary.get("arguments", {}).get(field) != declared or
                summary.get("recovery_contract", {}).get(field) != declared):
            raise ValueError(f"{name} run does not carry the declared {field}={declared}")
    # Both arms carry the published detail tap; no other term or lever may sit beside the span.
    if summary.get("config", {}).get("occupancy_detail") != 1:
        raise ValueError(f"{name} run does not carry the published detail tap")
    for key in SUPERVISION+READ_LEVERS:
        if summary.get("config", {}).get(key, 0) or summary.get("arguments", {}).get(key, 0):
            raise ValueError(f"{name} carries an undeclared lever beside the span: {key}")
    return study, summary, result


def training_occupancy(study):
    """The occupancy loss the arm paid, since a tap geometry adds no term of its own."""
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
                       "charge to report - the tap geometry changes what one projection reads"}


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
    """A board is exact only when every visible cell is decided right, and its skyline is right column by column.

    Two definitions, both strict (V-139). The cell definition requires every known visible cell to agree with its label
    at `0.0`. The boundary definition compares every column's top-most decided row with its top-most occupied row **with
    an empty column compared as a value** - the earlier version dropped those columns and produced the phantom `1 of 28`.
    """
    config = result["config"]
    rows, columns = config["visible_rows"], config["columns"]
    exact = boundary = total = 0
    for case in result["cases"]:
        if case["split"] != "training" or not any(case["occupied"]):
            continue
        decided, occupied = boundaries(case["logits"], case["occupied"], rows, columns, 0.0)
        total += 1
        cells = [(value > 0.0) == bool(label) for value, label in zip(case["logits"], case["occupied"])]
        exact += all(cells)
        boundary += all(got == want for got, want in zip(decided, occupied))
    if not total:
        raise ValueError("the fit gate carried no board")
    return {"boards": total, "exact": exact, "boundary_exact": boundary}


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
        width = result["config"]["widths"][0]
        arms[name] = {
            "study": study, "sweep": f"{study}-sweep-r1",
            "occupancy_tap_span": summary["config"]["occupancy_tap_span"],
            "weights_sha256": result["weights_sha256"],
            "config": {key: summary["config"][key] for key in RECORDED_CONFIG},
            "tap_in_features": width*(3 if name == TREATED else 1),
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

    differing = sorted(key for key in control["config"] if control["config"][key] != treated["config"][key])
    if differing != sorted(DECLARED):
        raise ValueError(f"the arms differ in more than the declared field: {differing}")
    if treated["weights_sha256"] == control["weights_sha256"]:
        raise ValueError("the two arms scored identical weights, so one of them did not measure its own run")
    left = next(name for name in previous_scope["arms"] if name != "control")
    if treated["weights_sha256"] == previous_scope["arms"][left]["weights_sha256"]:
        raise ValueError("the span arm scored the neighbour block's checkpoint")

    delta = round(treated["f1_at_zero"]-control["f1_at_zero"], 6)
    auc_delta = round(treated["auc"]-control["auc"], 6)
    precision_delta = round(treated["precision"]-control["precision"], 6)
    ceiling = control["f1_at_zero"]
    if delta >= ATTRIBUTION_DELTA:
        reading = "span-clears-the-ceiling"
        reading_text = ("Reading each cell's row neighbourhood as well as the cell itself clears the published read's "
                        "own primary by the standing attribution floor. The boundary evidence the point sample could "
                        "not express - the row-to-row change that marks a step - is what the remaining errors were "
                        "missing, and no label was consulted to read it.")
    elif delta > 0:
        reading = "span-moves-without-clearing"
        reading_text = ("The row-span tap beats the published read it widens, but short of the attribution floor: the "
                        "neighbouring rows carry some of the boundary evidence, not enough of it to move the primary "
                        "by the declared minimum.")
    else:
        reading = "span-fails"
        reading_text = ("Reading the row neighbourhood does not beat the published point read at the same budget. The "
                        "row-span hypothesis is falsified on this population: the remaining errors are not a missing "
                        "neighbour comparison in the tap, and no span width, sample count or projection size is offered "
                        "in its place.")
    if reading not in READINGS:
        raise ValueError("an undeclared reading was produced")
    secondary = "ranking-preserved" if auc_delta >= -AUC_TOLERANCE else "ranking-degraded"
    scope = {
        "status": "passed",
        "instrument": INSTRUMENT,
        "predeclaration": PREDECLARATION,
        "scope": ("The published read's own 64-step control against a 64-step arm whose only declared difference is "
                  "the tap geometry - three row samples instead of one, through the same zero-initialized projection "
                  "- captured in one catalogue and scored read-only at the standing 0.0 boundary and over the "
                  "standing 81-point grid. The option is structural geometry: no label, no threshold, no fitted "
                  "parameter, no prevalence, no statistic of the population. 0 is exactly the published function. No "
                  "threshold change, no population change and no real-image, complete-state or quality claim."),
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "mechanism": {
            "name": "row-span-tap",
            "flag": "--occupancy-tap-span 1 (a declared form of --occupancy-detail 1)",
            "definition": "the detail tap samples three points along the board's row axis - the cell's own row, the "
                          "row immediately above and the row immediately below, in cell units, bottom-first - and "
                          "feeds all three to the same zero-initialized projection",
            "where": "model head, in the occupancy tap, before the read-out and before every consumer",
            "sample_count": 3,
            "offsets": [[0.0, 1.0], [0.0, 0.0], [0.0, -1.0]],
            "parameters_added": 0, "tensors_added": 0, "loss_terms_added": 0,
            "projection": "one Linear(widths[0]*3 -> dimension), zero-initialized, replacing the point tap's "
                          "Linear(widths[0] -> dimension) under the same name",
            "weight": None, "margin": None,
            "statistic": "none: the three offsets are the board's own row geometry, so nothing about the population "
                         "is read",
            "justification": "the measured error density is 2.2x higher in the four rows at the top of each stack "
                             "(0.419, 0.407 and 0.428 of cells wrong at distance 0, 1 and 2-3 from the stack's top "
                             "row) than far below it (0.188), over the 3,522 retained nonempty boards of the 84 "
                             "retained sweeps, and every cell's logit was a function of one interior sample, so a "
                             "cell could not express that its own row differs from the row below it",
            "declared_before_receipts": True,
            "unknown_cells": "spawn rows are outside the visible window and pass through untouched",
            "identity_at_zero": "occupancy_tap_span 0 is exactly the published function",
            "composition": "refused beside every other tap geometry, every lever that moves the sample or changes "
                           "where zero sits, and every supervised read-out, so this arm's only declared difference "
                           "is the geometry itself",
            "expected_reading": "span-moves-without-clearing",
            "label_free": True,
            "label_free_proof": "the geometry is a constant, the projection is a function of the sampled features, "
                                "and the option is applied before any loss is evaluated; tests permute the labels "
                                "and require bit-identical output and scan the executable body for any label, "
                                "target, occupancy-mask, prevalence or threshold reference",
            "falsification": "falsified if the primary at 0.0 does not exceed the published read's own primary at "
                             "the same step budget, and separately recorded if the strict exact-board gate stays "
                             "at zero",
            "failure_mode": "three samples a full cell apart cannot see a step that falls between two cell centres, "
                            "so the arm buys boundary evidence only where the misalignment is within half a cell",
        },
        "envelope": {"steps": STEPS, "ceiling_bytes": CEILING_BYTES,
                     "work_items": {"sweep": FORWARDS, "curation": 9},
                     "attribution_floor": ATTRIBUTION_DELTA, "auc_tolerance": AUC_TOLERANCE},
        "arms": arms,
        "contrasts": {
            "span_control": {"f1_at_zero_delta": delta, "auc_delta": auc_delta,
                             "precision_delta": precision_delta,
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
            "rule": "a tap geometry changes what each cell reads, so the block proves it measured its own "
                    "checkpoint: each sweep's weights equal its own study's checkpoint, the arms' weights differ, "
                    "the previous block's checkpoint is excluded by name, and the configs differ in exactly the "
                    "declared field",
        },
        "reference": {
            "skyline_recovery": reference(SKYLINE, "recovered"),
            "crossing_rule": reference(RULED, "ruled"),
            "combined_read": reference(PREVIOUS, left),
            "envelope": reference(ENVELOPE, "contiguous"),
            "centring": reference(CENTRING, "centred"),
            "midpoint_anchor": reference(MIDPOINT, "level"),
            "dense_hinge": reference(DENSE, "banded"),
            "quantile_anchor": reference(QUANTILE, "bulk"),
            "extremal_anchor": reference(PRIOR, "anchor"),
            "hinge": reference(HINGE, "hinge"),
            "role": "reference-only: the M17 skyline arm, the M16 crossing rule, the M15 combined read, the M15 "
                    "envelope arm, the M14 centred read, the M13 midpoint anchor, the M12 dense hinge, the M11 "
                    "quantile anchor, the M10 extremal anchor and the M9 hinge, each measured in its own catalogue "
                    "and none re-measured or re-verdicted here",
        },
        "reading": reading,
        "reading_text": reading_text,
        "secondary_reading": secondary,
        "secondary_text": ("Deciding AUC with the standing -0.02 tolerance. A tap that reads its row neighbourhood "
                           "can only change the ranking by changing what it sees, so a ranking change here is the "
                           "read's own doing and is reported as the measured price or gain."),
        "fixed_threshold_ceiling": {"arm": CONTROL, "scope": "this block's reproduced published read",
                                    "prior_scope": str(PRIOR.relative_to(ROOT)),
                                    "f1_at_zero": ceiling,
                                    "span_primary": treated["f1_at_zero"]},
        "ceiling_broken": treated["f1_at_zero"] > ceiling,
        "gate_moved": treated["exact_boards_at_zero"]["exact"] > 0,
        "neighbour_excluded": {"arm": left, "scope": str(PREVIOUS.relative_to(ROOT))},
        "verdict": lever_verdict(treated["f1_at_zero"], treated["verdict"], delta),
        "gates": {"per_cell_f1": CELL_F1_THRESHOLD, "ranking_auc": AUC_THRESHOLD,
                  "ranking_f1": F1_THRESHOLD, "attribution_floor": ATTRIBUTION_DELTA, "grid_points": 81},
        "fit_gate": {"exact_nonempty_boards": treated["exact_boards_at_zero"]["exact"],
                     "boundary_exact_nonempty_boards": treated["exact_boards_at_zero"]["boundary_exact"],
                     "boards": treated["exact_boards_at_zero"]["boards"],
                     "cleared": (treated["exact_boards_at_zero"]["exact"] == BOARDS),
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
