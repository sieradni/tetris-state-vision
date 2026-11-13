"""Curate the span-under-level composition against the reproduced published read on the corrected strict metric.

`docs/M19_SPAN_LEVEL.md` declares the hypothesis, the falsification and the reading vocabulary before any receipt. This
recorder asserts the three declared fields and nothing else, that each sweep carries its own study's checkpoint while the
neighbour block's is excluded by name, that the control reproduces the published numbers exactly, that the composition's
level actually moved toward the control's (read from the M18 record, not restated), and it reports the fit gate under
**both** strict definitions V-139 left behind.
"""
import argparse
import json
from pathlib import Path

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
FAMILY = "m19-span-level"
INSTRUMENT = "m19-span-level-r1"
PREDECLARATION = "docs/M19_SPAN_LEVEL.md"
PREVIOUS = ROOT/"docs/evidence/m15-stack-combined-r1/scope.json"
SPAN = ROOT/"docs/evidence/m18-tap-span-r1/scope.json"
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
TREATED = "composed"
RUNS = {CONTROL: "control", TREATED: "composed"}
READINGS = ("span-level-passes", "span-level-moves-without-clearing", "span-level-fails")
SECONDARY = ("ranking-preserved", "ranking-degraded")
SUPERVISION = ("occupancy_midpoint", "occupancy_dense_hinge", "occupancy_quantile_anchor", "occupancy_anchor",
               "occupancy_margin_hinge")
READ_LEVERS = ("occupancy_normalise", "occupancy_offset_rows", "occupancy_deform", "occupancy_context",
               "occupancy_detail_stencil", "occupancy_resolution", "occupancy_column", "occupancy_lattice_free",
               "occupancy_placed", "occupancy_stack_monotone", "occupancy_skyline_recovery")
DECLARED = ("occupancy_centre", "occupancy_crossing_threshold", "occupancy_tap_span")
RECORDED_CONFIG = ("occupancy_tap_span", "occupancy_centre", "occupancy_crossing_threshold", "occupancy_detail",
                   "occupancy_normalise", "occupancy_stack_monotone", "occupancy_skyline_recovery", "dimension",
                   "component_capacity")


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
    # Both arms carry the published detail tap; nothing else may sit beside the composition.
    if summary.get("config", {}).get("occupancy_detail") != 1:
        raise ValueError(f"{name} run does not carry the published detail tap")
    for key in SUPERVISION+READ_LEVERS:
        if summary.get("config", {}).get(key, 0) or summary.get("arguments", {}).get(key, 0):
            raise ValueError(f"{name} carries an undeclared lever beside the composition: {key}")
    return study, summary, result


def training_occupancy(study):
    """The occupancy loss the arm paid, since these structural options add no term of their own."""
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
            "reading": "the occupancy loss the arm paid; the arm adds no term of its own - the geometry changes what "
                       "one projection reads and the transforms only move where zero sits"}


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
    """Strict (V-139): every visible cell right, and every column's top right with an empty column compared."""
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
    span_scope = read(SPAN)
    if prior.get("instrument") != "m10-supervised-anchor-r1":
        raise ValueError("the reproduction reference is not the committed extremal-anchor block")
    if span_scope.get("instrument") != "m18-tap-span-r1":
        raise ValueError("the level diagnostic's reference is not the committed span block")
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
            "occupancy_centre": summary["config"]["occupancy_centre"],
            "occupancy_crossing_threshold": summary["config"]["occupancy_crossing_threshold"],
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
        raise ValueError(f"the arms differ in more than the declared fields: {differing}")
    if treated["weights_sha256"] == control["weights_sha256"]:
        raise ValueError("the two arms scored identical weights, so one of them did not measure its own run")
    left = next(name for name in previous_scope["arms"] if name != "control")
    for excluded, scope in (("neighbour block", previous_scope), ("span-only block", span_scope)):
        for arm_name, arm in scope["arms"].items():
            if treated["weights_sha256"] == arm["weights_sha256"]:
                raise ValueError(f"the composed arm scored the {excluded}'s checkpoint ({arm_name})")

    delta = round(treated["f1_at_zero"]-control["f1_at_zero"], 6)
    auc_delta = round(treated["auc"]-control["auc"], 6)
    span_only = span_scope["arms"]["tap"]
    span_share, span_row = span_only["positive_share_at_zero"], span_only["row_error_at_zero"]["mean"]
    gate_moved = treated["exact_boards_at_zero"]["exact"] > 0
    level_repaired = (abs(treated["positive_share_at_zero"]-control["positive_share_at_zero"])
                      < abs(span_share-control["positive_share_at_zero"]) and
                      abs(treated["row_error_at_zero"]["mean"]-control["row_error_at_zero"]["mean"])
                      < abs(span_row-control["row_error_at_zero"]["mean"]))
    ceiling = control["f1_at_zero"]
    if delta >= ATTRIBUTION_DELTA and gate_moved:
        reading = "span-level-passes"
        reading_text = ("The span under the level transforms clears the published read's primary by the standing "
                        "attribution floor and puts a genuinely exact board on the gate: the span's separation was "
                        "real, its failure was placement, and two label-free transforms were enough to convert it.")
    elif delta > 0:
        reading = "span-level-moves-without-clearing"
        reading_text = ("The composed arm beats the published read at the declared boundary but does not clear the "
                        "attribution floor together with the gate: part of the span's separation survives the level "
                        "transforms, not enough of it to count as a win.")
    else:
        reading = "span-level-fails"
        reading_text = ("The span under the level transforms does not beat the published read at the declared "
                        "boundary. The composition hypothesis is falsified on this population: the span's gain was "
                        "not recoverable by moving where zero sits, and no transform weight, penalty or threshold is "
                        "offered in its place.")
    if reading not in READINGS:
        raise ValueError("an undeclared reading was produced")
    secondary = "ranking-preserved" if auc_delta >= -AUC_TOLERANCE else "ranking-degraded"
    scope = {
        "status": "passed",
        "instrument": INSTRUMENT,
        "predeclaration": PREDECLARATION,
        "scope": ("The published read's own 64-step control against a 64-step arm that composes the M18 row-span tap "
                  "with the M14 centring and the M16 crossing rule - captured in one catalogue and scored read-only at "
                  "the standing 0.0 boundary and over the standing 81-point grid. Every part is structural and "
                  "label-free: no threshold, no fitted parameter, no prevalence, no statistic of the population, and "
                  "no part of the arm is a swept constant. With all three fields at 0 the model is exactly the "
                  "published function. No population change and no real-image, complete-state or quality claim."),
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "mechanism": {
            "name": "row-span-tap-under-level-transforms",
            "flag": ("--occupancy-detail 1 --occupancy-tap-span 1 --occupancy-centre 1 "
                     "--occupancy-crossing-threshold 1"),
            "definition": ("the detail tap reads three row-aligned samples (the cell and its two row neighbours) "
                           "through one zero-initialized projection, then the head removes the board's additive mode "
                           "and re-places zero at the read's own Otsu split, in that fixed order"),
            "where": "model head, before the loss and before every consumer",
            "sample_count": 3,
            "offsets": [[0.0, 1.0], [0.0, 0.0], [0.0, -1.0]],
            "parameters_added": 0, "tensors_added": 0, "loss_terms_added": 0,
            "weight": None, "margin": None,
            "statistic": "none: the geometry is the board's own row axes and both transforms are statistics of the "
                         "board's own visible logits, so nothing about the population is read",
            "justification": ("M18 measured the row span's swept optimum 0.443255 and out-of-fold 0.429607 against "
                              "the point tap's 0.412034 and 0.400399, with its failure a level: 80.55% of cells "
                              "positive at 0.0 and +1.59 rows of placement error"),
            "declared_before_receipts": True,
            "unknown_cells": "spawn rows are outside the visible window and pass through untouched",
            "identity_at_zero": "all three fields 0 is exactly the published function",
            "composition": "refused beside every other tap geometry, every sample-moving lever, both decision-side "
                           "structural options and every supervised read-out; centring and the crossing rule are "
                           "admitted together and only together",
            "non_degeneracy": "tests assert that the transforms perturb the span's logits from the same trained tap "
                              "weights, and that the arm's tap reads 3*widths[0] values per cell while the control's "
                              "reads widths[0], so no reading of this arm can be produced by the control's geometry",
            "expected_reading": "span-level-moves-without-clearing",
            "label_free": True,
            "label_free_proof": "the geometry is a constant and both transforms are functions of the logits and the "
                                "visible rows; tests permute the labels and require bit-identical output and scan the "
                                "executable body for any label, target, occupancy-mask, prevalence or threshold "
                                "reference",
            "falsification": ("falsified if the primary does not exceed the published read's own primary; separately "
                              "falsified if the positive share and the row error are not both closer to the "
                              "control's than the span-only arm's; separately recorded if the strict exact-board gate "
                              "stays at zero"),
            "failure_mode": ("if it fails while the level is repaired, the span's gain was noise that the transforms "
                             "merely re-levelled; if it fails with the level unrepaired, the transforms did not reach "
                             "the arm's placement"),
        },
        "envelope": {"steps": STEPS, "ceiling_bytes": CEILING_BYTES,
                     "work_items": {"sweep": FORWARDS, "curation": 9},
                     "attribution_floor": ATTRIBUTION_DELTA, "auc_tolerance": AUC_TOLERANCE},
        "arms": arms,
        "contrasts": {
            "composed_control": {"f1_at_zero_delta": delta, "auc_delta": auc_delta,
                                 "precision_delta": round(treated["precision"]-control["precision"], 6),
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
            "not_the_previous_block": all(treated["weights_sha256"] != arm["weights_sha256"]
                                          for arm in previous_scope["arms"].values()),
            "not_the_span_only_arm": all(treated["weights_sha256"] != arm["weights_sha256"]
                                         for arm in span_scope["arms"].values()),
            "rule": "a composition changes what each cell reads and where zero sits, so the block proves it measured "
                    "its own checkpoint: each sweep's weights equal its own study's checkpoint, the arms' weights "
                    "differ, the neighbour block's and the span-only block's checkpoints are excluded by name, and "
                    "the configs differ in exactly the declared fields",
        },
        "reference": {
            "span_only": reference(SPAN, "tap"),
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
            "role": "reference-only: the M18 span-only arm, the M17 skyline arm, the M16 crossing rule, the M15 "
                    "combined read, the M15 envelope arm, the M14 centred read, the M13 midpoint anchor, the M12 dense "
                    "hinge, the M11 quantile anchor, the M10 extremal anchor and the M9 hinge, each measured in its own "
                    "catalogue and none re-measured or re-verdicted here",
        },
        "reading": reading,
        "reading_text": reading_text,
        "secondary_reading": secondary,
        "secondary_text": ("Deciding AUC with the standing -0.02 tolerance. A tap that reads its row neighbours and two "
                           "transforms that move the level can change the ranking, so a ranking change here is the "
                           "arm's own doing and is reported as the measured price or gain."),
        "fixed_threshold_ceiling": {"arm": CONTROL, "scope": "this block's reproduced published read",
                                    "prior_scope": str(PRIOR.relative_to(ROOT)),
                                    "f1_at_zero": ceiling, "composed_primary": treated["f1_at_zero"]},
        "ceiling_broken": treated["f1_at_zero"] > ceiling,
        "gate_moved": gate_moved,
        "level_repaired": level_repaired,
        "level_diagnostic": {"positive_share": treated["positive_share_at_zero"],
                             "control_positive_share": control["positive_share_at_zero"],
                             "span_only_positive_share": span_share,
                             "row_error": treated["row_error_at_zero"]["mean"],
                             "control_row_error": control["row_error_at_zero"]["mean"],
                             "span_only_row_error": span_row},
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
                                                  "aliasing_guard", "verdict", "fit_gate",
                                                  "level_diagnostic")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
