"""Curate the de-meaned registration arm against the free field, the one-cell field and control.

`docs/M3_OCCUPANCY_REGISTRATION.md` fixes the rule; this recorder applies it to four admitted
continuations and their widened sweeps, joins the measured offset field of each deformation arm, and
checks that the three arms shared with the previous block reproduce its published numbers exactly.
The de-meaned arm's constant mode is zero by construction, so its field decides whether per-cell
displacement becomes the dominant mode once the translation is removed.
"""
import argparse
import bisect
import json
from pathlib import Path

from cluster_interval import cluster_bootstrap
from gate_power import f1_from_counts
from occupancy_sweep import AUC_THRESHOLD, F1_THRESHOLD, pooled_auc, sweep_verdict, threshold_grid
from percell_gate import CELL_F1_THRESHOLD
from probe_model import require_bounded_job
from record_precision_lever import precision_of
from record_representation_lever import (ancestry_of, arm_summary, cells_of, f1_at_threshold,
                                         lever_verdict, read)

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 1792*1024**2
SWEEP_CEILING_BYTES = 640*1024**2
FIELD_CEILING_BYTES = 640*1024**2
STEPS = 64
DECIDING_IMAGES = 36
DEVELOPMENT_IMAGES = 18
FORWARDS = DECIDING_IMAGES+DEVELOPMENT_IMAGES
FIELD_FORWARDS = DECIDING_IMAGES
ATTRIBUTION_DELTA = 0.05
PRECISION_DELTA = 0.05
CONSTANT_DOMINANT = .5
STRUCTURE_CELLS = .25
BOOTSTRAP_DRAWS = 400
BOOTSTRAP_SEED = 20261026
Z_POWER_80 = 2.802
SOURCE_RUN = "m3-percell-balance-r1"
FAMILY = "m3-occdeform3"
INSTRUMENT = "m3-occupancy-deformation-r3"
PREDECLARATION = "docs/M3_OCCUPANCY_REGISTRATION.md"
REPRODUCTION = ROOT/"docs/evidence/m3-occupancy-deformation-r2/scope.json"
ARMS = {"demean": 3, "wide": 2, "lever": 1, "control": 0}
REACH = {"demean": 8.0, "wide": 4.0, "lever": 1.0, "control": 0.0}
PRIMARY = ("demean", "control")
CONTRASTS = (("demean", "control"), ("demean", "wide"), ("wide", "control"), ("lever", "control"))
METRICS = ("auc", "f1", "precision", "cv_f1")
DISPERSION = {"mean_vector_cells", "spread_cells", "mean_square_cells", "constant_share",
              "coherence"}
DEFORM_ARMS = ("demean", "wide", "lever")


def require_grid(grid):
    if type(grid) not in (tuple, list) or len(grid) != 81 or 0.0 not in grid:
        raise ValueError("the standing 81-point grid on [-6, 6] step 0.15 is required")
    return tuple(grid)


def curves_of(rows):
    grouped = {}
    for row in rows:
        positive, negative = grouped.setdefault(ancestry_of(row["sample_id"]), ([], []))
        for logit, label in zip(row["logits"], row["occupied"]):
            (positive if label else negative).append(logit)
    return tuple((name, sorted(positive), sorted(negative))
                 for name, (positive, negative) in sorted(grouped.items()))


def counts_at_curves(curves, threshold):
    true_positive = false_positive = false_negative = 0
    for _, positive, negative in curves:
        hit = len(positive)-bisect.bisect_right(positive, threshold)
        true_positive += hit
        false_negative += len(positive)-hit
        false_positive += len(negative)-bisect.bisect_right(negative, threshold)
    return true_positive, false_positive, false_negative


def best_threshold(curves, grid):
    best = None
    for threshold in grid:
        true_positive, false_positive, false_negative = counts_at_curves(curves, threshold)
        f1 = f1_from_counts(true_positive, false_positive, false_negative)
        precision = (0.0 if not true_positive+false_positive
                     else true_positive/(true_positive+false_positive))
        if best is None or (f1, precision) > best[0]:
            best = ((f1, precision), threshold)
    return best[1]


def out_of_fold(curves, grid):
    require_grid(grid)
    curves = tuple(curves)
    if len(curves) < 2:
        raise ValueError("at least two ancestry clusters required")
    true_positive = false_positive = false_negative = 0
    chosen = []
    for index, (_, positive, negative) in enumerate(curves):
        others = tuple(entry for position, entry in enumerate(curves) if position != index)
        threshold = best_threshold(others, grid)
        chosen.append(threshold)
        hit = len(positive)-bisect.bisect_right(positive, threshold)
        true_positive += hit
        false_negative += len(positive)-hit
        false_positive += len(negative)-bisect.bisect_right(negative, threshold)
    return {"f1": round(f1_from_counts(true_positive, false_positive, false_negative), 6),
            "thresholds": chosen, "groups": len(curves),
            "counts": {"true_positive": true_positive, "false_positive": false_positive,
                       "false_negative": false_negative}}


def deciding_rows(rows):
    trials = [row for row in rows if any(row["occupied"])]
    if not trials:
        raise ValueError("the deciding population is empty")
    return trials


def components_of(rows, grid):
    curves = curves_of(deciding_rows(rows))
    best = best_threshold(curves, grid)
    best_counts = counts_at_curves(curves, best)
    zero_counts = counts_at_curves(curves, 0.0)
    folded = out_of_fold(curves, grid)
    best_f1 = round(f1_from_counts(*best_counts), 6)
    zero_f1 = round(f1_from_counts(*zero_counts), 6)
    best_true_positive, best_false_positive, _ = best_counts
    return {"best_f1": best_f1, "best_threshold": best,
            "zero_f1": zero_f1, "zero_cost": round(zero_f1-best_f1, 6),
            "best_precision": round(0.0 if not best_true_positive+best_false_positive
                                    else best_true_positive/(best_true_positive+best_false_positive), 6),
            "out_of_fold_f1": folded["f1"],
            "transfer_cost": round(folded["f1"]-zero_f1, 6),
            "out_of_fold_thresholds": folded["thresholds"],
            "out_of_fold_counts": folded["counts"], "groups": folded["groups"]}


def detectable(interval):
    return round(Z_POWER_80*interval["standard_error"], 6)


def mechanism_of(measured, bootstrap):
    best = measured["demean"]["best"]["precision"]-measured["control"]["best"]["precision"]
    zero = measured["demean"]["precision"]-measured["control"]["precision"]
    interval = bootstrap["precision_delta_demean_control"]
    if abs(zero-interval["median"]) > 0.05:
        raise ValueError("the precision interval does not belong to the fixed-boundary delta it is "
                         "reported with")
    return {"field_active": True,
            "read": "precision at each arm's own best boundary",
            "best_precision_delta": round(best, 6),
            "best_precision_improved": bool(best >= PRECISION_DELTA),
            "floor": PRECISION_DELTA,
            "zero_boundary_precision_delta": {
                "delta": round(zero, 6), "interval": interval,
                "minimum_detectable_effect": detectable(interval),
                "note": ("fixed 0.0 boundary, which a re-scaled head moves; a replay cost, not a "
                         "mechanism verdict")}}


def require_arm(name):
    deform = ARMS[name]
    study = f"{FAMILY}-{name}-r1"
    summary = read(ROOT/"runs"/study/"summary.json")
    job = read(ROOT/"runs"/(study+"-job.json"))
    if job.get("status") != "passed" or job.get("exit_code") != 0:
        raise ValueError(f"{name} study did not pass inside its declared envelope")
    if (summary.get("status") != "complete" or summary.get("steps") != STEPS or
            summary.get("requested_steps") != STEPS or
            not summary.get("checkpoint_round_trip_exact") or
            job.get("work_items") != STEPS or job.get("memory_bytes") != CEILING_BYTES):
        raise ValueError(f"{name} study is not a complete 64-step run inside 1,792 MiB")
    arguments = summary["arguments"]
    if (arguments.get("occupancy_detail") != 1 or arguments.get("occupancy_detail_stencil") != 0 or
            arguments.get("occupancy_resolution") != 0 or arguments.get("occupancy_context") != 0 or
            arguments.get("occupancy_deform") != deform or arguments.get("steps") != STEPS or
            arguments.get("seed") != 20261026 or arguments.get("capture_max_side") != 512 or
            summary["recovery_contract"].get("occupancy_deform") != deform):
        raise ValueError(f"{name} study did not run the declared lever at the declared budget")
    origin = summary.get("initial_weights_from") or {}
    if not str(origin.get("run", "")).endswith(SOURCE_RUN) or not origin.get("weights_sha256"):
        raise ValueError(f"{name} study was not initialized from the published checkpoint")
    return study, summary, job, origin


def require_sweep(name):
    study = f"{FAMILY}-{name}-r1"
    sweep = f"{study}-sweep-r1"
    result = read(ROOT/"runs"/sweep/"result.json")
    job = read(ROOT/"runs"/(sweep+"-job.json"))
    summary = read(ROOT/"runs"/study/"summary.json")
    profile = read(ROOT/"runs"/(sweep+"-profile-r1")/"summary.json")
    profile_job = read(ROOT/"runs"/(sweep+"-profile-r1-job.json"))
    if (job.get("status") != "passed" or job.get("exit_code") != 0 or
            job.get("memory_bytes") != SWEEP_CEILING_BYTES or
            job.get("work_items") != result.get("forwards")):
        raise ValueError(f"{name} sweep did not pass under the declared 640 MiB / 54 work items")
    if (result.get("status") != "passed" or result.get("run") != study or
            result.get("forwards") != FORWARDS or len(result.get("cases", [])) != FORWARDS):
        raise ValueError(f"{name} sweep is not a complete 54-forward widened measurement")
    if result.get("weights_sha256") != summary.get("checkpoint_sha256"):
        raise ValueError(f"{name} sweep measured different weights from its study")
    if sweep_verdict(result["deciding"]) != result["verdict"]:
        raise ValueError(f"{name} sweep verdict does not re-derive")
    if len(result["deciding"]["thresholds"]) != 81 or 0.0 not in result["deciding"]["thresholds"]:
        raise ValueError(f"{name} sweep did not use the standing 81-point grid")
    declared = result.get("training_population") or {}
    if (declared.get("evaluated_groups") != 12 or declared.get("evaluated_cases") != DECIDING_IMAGES):
        raise ValueError(f"{name} sweep did not score the widened deciding population")
    if (profile_job.get("status") != "passed" or profile.get("status") != "complete" or
            profile.get("weights_sha256") != result.get("weights_sha256")):
        raise ValueError(f"{name} sweep admission profile differs")
    return sweep, result, job


def require_field(name):
    """The applied offset field of one deformation arm, joined to the study it belongs to."""
    study = f"{FAMILY}-{name}-r1"
    instrument = f"{study}-field-r1"
    summary = read(ROOT/"runs"/study/"summary.json")
    result = read(ROOT/"runs"/instrument/"result.json")
    job = read(ROOT/"runs"/(instrument+"-job.json"))
    if (job.get("status") != "passed" or job.get("exit_code") != 0 or
            job.get("memory_bytes") != FIELD_CEILING_BYTES or
            job.get("work_items") != FIELD_FORWARDS):
        raise ValueError(f"{name} field measurement did not pass under its declared envelope")
    if (result.get("status") != "passed" or result.get("forwards") != FIELD_FORWARDS or
            result.get("weights_sha256") != summary.get("checkpoint_sha256")):
        raise ValueError(f"{name} field measurement is not the study's own weights")
    if (result["population"]["images"] != DECIDING_IMAGES or
            result["population"]["known_cells"] != DECIDING_IMAGES*200 or
            result["population"]["occupied_cells"] != 667):
        raise ValueError(f"{name} field measurement decided another population")
    if (result["arm"]["occupancy_deform"] != ARMS[name] or
            result["arm"]["bound_cells"] != REACH[name]):
        raise ValueError(f"{name} field measurement reports another reach")
    field = result["field"]
    if not DISPERSION <= set(field):
        raise ValueError(f"{name} field measurement lacks the dispersion block")
    # Re-derive the constant share from stored numbers: a formula that reads zero regardless of the
    # data cannot survive this (V-106).
    mean_x, mean_y = field["mean_vector_cells"]
    derived = ((mean_x*mean_x+mean_y*mean_y)/field["mean_square_cells"]
               if field["mean_square_cells"] else 1.0)
    if abs(derived-field["constant_share"]) > 1e-5:
        raise ValueError(f"{name} constant share does not re-derive from its own mean vector and "
                         "mean square")
    if ARMS[name] == 3 and field["constant_share"] >= CONSTANT_DOMINANT:
        raise ValueError("the de-meaned arm still carries a dominant constant mode, which its "
                         "construction forbids")
    return instrument, result


def reproduction_of(measured):
    """Control, wide and one-cell must return the previous block's own published numbers."""
    prior = read(REPRODUCTION)
    if prior.get("instrument") != "m3-occupancy-deformation-r2":
        raise ValueError("the reproduction reference is not the committed widened block")
    checked = {}
    for name in ("control", "wide", "lever"):
        published = prior["arms"][name]
        current = measured[name]
        for key in ("auc", "per_cell_f1"):
            if current[key] != published[key]:
                raise ValueError(f"{name} {key} does not reproduce the published {published[key]!r}")
        if current["best"]["f1"] != published["best"]["f1"]:
            raise ValueError(f"{name} best F1 does not reproduce the published value")
        checked[name] = {"auc": current["auc"], "per_cell_f1": current["per_cell_f1"],
                         "best_f1": current["best"]["f1"],
                         "prior_scope": str(REPRODUCTION.relative_to(ROOT))}
    return checked


def main():
    bounded = require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("evidence directory already exists")

    source = read(ROOT/"runs"/SOURCE_RUN/"summary.json")
    arms = {}
    for name in ARMS:
        study, summary, job, origin = require_arm(name)
        if origin["weights_sha256"] != source["checkpoint_sha256"]:
            raise ValueError(f"{name} study did not start from the published checkpoint bytes")
        sweep, result, sweep_job = require_sweep(name)
        arms[name] = {"study": study, "summary": summary, "job": job, "sweep": sweep,
                      "result": result, "sweep_job": sweep_job}
    if len({arms[name]["summary"]["catalogue_sha256"] for name in ARMS}) != 1:
        raise ValueError("the arms were captured under different catalogues")
    initial = {name: arms[name]["summary"]["initial_weights_sha256"] for name in ARMS}
    # The de-meaned arm adds no tensors to the widened one, so their saved initial weights are the
    # same bytes; the other three parameter sets must each save their own.
    if initial["demean"] != initial["wide"]:
        raise ValueError("the de-meaned arm must start from exactly the widened arm's tensors, since "
                         "de-meaning adds no parameters")
    if len({initial["control"], initial["lever"], initial["wide"]}) != 3:
        raise ValueError("each distinct parameter set must save its own initial tensor file")

    fields = {name: require_field(name) for name in DEFORM_ARMS}

    cases = {name: [row for row in arms[name]["result"]["cases"] if row["split"] == "training"]
             for name in ARMS}
    for name in ARMS:
        if len(cases[name]) != DECIDING_IMAGES:
            raise ValueError("the curation requires exactly the 36 deciding training images")
    if len({tuple(row["sample_id"] for row in cases[name]) for name in ARMS}) != 1:
        raise ValueError("the four sweeps must share one deciding population")
    for first in ARMS:
        for second in ARMS:
            if first < second and [row["logits"] for row in cases[first]] == \
                    [row["logits"] for row in cases[second]]:
                raise ValueError(f"{first} and {second} produced identical logits; the lever did not run")
    nonempty = sum(1 for row in cases["demean"] if any(row["occupied"]))
    contexts = len({ancestry_of(row["sample_id"]) for row in cases["demean"]})
    if (nonempty, contexts) != (28, 12):
        raise ValueError("deciding arithmetic differs from the predeclaration (28 boards, 12 clusters)")

    grid = require_grid(threshold_grid())
    measured = {}
    for name in ARMS:
        block = arm_summary(arms[name]["result"]["deciding"])
        block["precision"] = round(precision_of(block["operating_counts"]), 6)
        block["components"] = components_of(cases[name], grid)
        measured[name] = block
    for name in ARMS:
        components = measured[name]["components"]
        if (abs(components["zero_f1"]-measured[name]["per_cell_f1"]) > 1e-6 or
                abs(components["best_f1"]-measured[name]["best"]["f1"]) > 1e-6):
            raise ValueError(f"{name} components were recomputed on a different population from the "
                             "deciding rows the verdict reads")

    reproduction = reproduction_of(measured)
    bootstrap = component_bootstrap(cases, grid)
    if bootstrap["clusters"] != 12:
        raise ValueError("the paired bootstrap must resample the twelve ancestry clusters")
    for first, second in CONTRASTS:
        printed = measured[first]["per_cell_f1"]-measured[second]["per_cell_f1"]
        if abs(bootstrap["f1_delta_"+first+"_"+second]["median"]-printed) > 0.05:
            raise ValueError("an F1 interval does not belong to the fixed-boundary delta it is "
                             "reported with")

    deltas = {}
    for first, second in CONTRASTS:
        key = first+"_"+second
        deltas[key] = {
            "best_f1_delta": round(measured[first]["best"]["f1"]
                                   - measured[second]["best"]["f1"], 6),
            "auc_delta": round(measured[first]["auc"]-measured[second]["auc"], 6),
            "per_cell_f1_delta": round(measured[first]["per_cell_f1"]
                                       - measured[second]["per_cell_f1"], 6),
            "precision_delta": round(measured[first]["precision"]
                                     - measured[second]["precision"], 6),
            "out_of_fold_f1_delta": round(measured[first]["components"]["out_of_fold_f1"]
                                          - measured[second]["components"]["out_of_fold_f1"], 6),
            "attribution_met": bool(measured[first]["best"]["f1"]
                                    - measured[second]["best"]["f1"] >= ATTRIBUTION_DELTA),
            "zero_boundary_interval": bootstrap["f1_delta_"+key],
            "zero_boundary_minimum_detectable_effect": detectable(bootstrap["f1_delta_"+key]),
            "out_of_fold_interval": bootstrap["cv_f1_delta_"+key],
            "out_of_fold_minimum_detectable_effect": detectable(bootstrap["cv_f1_delta_"+key]),
            "auc_interval": bootstrap["auc_delta_"+key],
            "auc_minimum_detectable_effect": detectable(bootstrap["auc_delta_"+key]),
            "precision_interval": bootstrap["precision_delta_"+key],
            "precision_minimum_detectable_effect": detectable(bootstrap["precision_delta_"+key]),
        }

    demean_field = fields["demean"][1]["field"]
    wide_field = fields["wide"][1]["field"]
    structure = demean_field["magnitude_cells"]["mean"] >= STRUCTURE_CELLS
    moves = deltas["demean_control"]["best_f1_delta"] >= ATTRIBUTION_DELTA
    if demean_field["constant_share"] >= CONSTANT_DOMINANT:
        raise ValueError("the de-meaned field carries a dominant constant mode; its construction "
                         "forbids one, so this is a defect, not a reading")
    if not structure:
        reading = "translation-was-the-signal"
        reading_text = ("The de-meaned field finds almost no structure once the constant mode is "
                        "removed, so the translation was the only signal the occupancy loss could use.")
    elif moves:
        reading = "per-cell-learnable"
        reading_text = ("With the constant mode structurally removed the field becomes per-cell "
                        "dominant and the head still moves ranking, so per-cell registration is "
                        "learnable in this read-out.")
    else:
        reading = "registration-without-signal"
        reading_text = ("The field is per-cell dominant but ranking does not move, so registration is "
                        "learnable and is not the signal this objective can use.")

    verdict = lever_verdict(measured["demean"]["per_cell_f1"], measured["demean"]["verdict"],
                            deltas["demean_control"]["best_f1_delta"])

    scope = {
        "status": "passed",
        "instrument": INSTRUMENT,
        "predeclaration": PREDECLARATION,
        "scope": ("Matched 64-step continuation of one published checkpoint by four arms differing "
                  "only in the registration lever: detail control, the one-cell field, the four-cell "
                  "coarse-to-fine field, and the same four-cell field de-meaned across the board so "
                  "its constant mode is exactly zero by construction. Scored read-only on the widened "
                  "deciding population with the standing gates and grid, with the applied offset field "
                  "measured in cells from the frozen archive; no threshold, population or gate "
                  "changes, and no real-image, complete-state, calibrated or quality claim."),
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "envelope": {"steps": STEPS, "ceiling_bytes": CEILING_BYTES,
                     "sweep_ceiling_bytes": SWEEP_CEILING_BYTES,
                     "field_ceiling_bytes": FIELD_CEILING_BYTES,
                     "work_items": {"profile": 2, "study": STEPS, "sweep_profile": 2,
                                    "sweep": FORWARDS, "field": FIELD_FORWARDS, "curation": 7},
                     "attribution_floor": ATTRIBUTION_DELTA, "precision_floor": PRECISION_DELTA,
                     "constant_dominant_below": CONSTANT_DOMINANT,
                     "structure_cells": STRUCTURE_CELLS,
                     "reach_cells": REACH},
        "source": {"run": SOURCE_RUN, "weights_sha256": source["checkpoint_sha256"],
                   "steps": source["steps"]},
        "matched": {"seed": 20261026, "learning_rate": .001, "capture_max_side": 512,
                    "curriculum": "stream-v1", "config_difference": ["occupancy_deform"],
                    "shared": {"occupancy_detail": 1, "occupancy_detail_stencil": 0,
                               "occupancy_resolution": 0, "occupancy_context": 0},
                    "catalogue_sha256": arms["demean"]["summary"]["catalogue_sha256"],
                    "initial_weights": arms["demean"]["summary"]["initial_weights_from"]},
        "arms": {name: {"study": arms[name]["study"], "sweep": arms[name]["sweep"],
                        "occupancy_deform": ARMS[name], "declared_bound_cells": REACH[name],
                        "weights_sha256": arms[name]["result"]["weights_sha256"],
                        "catalogue_sha256": arms[name]["summary"]["catalogue_sha256"],
                        "study_elapsed_seconds": arms[name]["summary"]["elapsed_seconds"],
                        "study_peak_commit_bytes": arms[name]["job"].get("peak_job_commit_bytes"),
                        "sweep_peak_commit_bytes": arms[name]["sweep_job"].get("peak_job_commit_bytes"),
                        "field": (None if name == "control" else {
                            "instrument": fields[name][0],
                            "peak_commit_bytes": read(ROOT/"runs"/(fields[name][0]+"-job.json")).get(
                                "peak_job_commit_bytes"),
                            **fields[name][1]["field"],
                            "arm": fields[name][1]["arm"],
                            "reference": fields[name][1]["reference"]}),
                        **measured[name]} for name in ARMS},
        "gates": {"per_cell_f1": CELL_F1_THRESHOLD, "ranking_auc": AUC_THRESHOLD,
                  "ranking_f1": F1_THRESHOLD, "attribution_floor": ATTRIBUTION_DELTA,
                  "grid_points": 81},
        "contrasts": deltas,
        "reproduction": reproduction,
        "components": {
            "note": ("Reported, never read by the verdict: ranking at each arm's own best boundary, "
                     "the cost of the fixed 0.0 boundary, and the cost of transferring a boundary "
                     "chosen on the other eleven ancestry groups. Each delta carries the interval that "
                     "resamples exactly it and a minimum detectable effect at 80% power."),
            "arms": {name: {key: measured[name]["components"][key] for key in
                            ("best_f1", "best_threshold", "best_precision", "zero_f1", "zero_cost",
                             "out_of_fold_f1", "transfer_cost", "out_of_fold_thresholds")}
                     for name in ARMS},
            "power": {"z": Z_POWER_80, "rule": "minimum detectable effect = 2.802 x cluster SE"}},
        "constant_mode": {
            "question": ("Is per-cell registration learnable once the constant mode is removed from "
                         "the space of functions? The free field put 97.8% of its displaced energy "
                         "into one vector, so the objective only had to find a translation."),
            "reading": reading,
            "reading_text": reading_text,
            "structure_found": structure,
            "ranking_moves": moves,
            "thresholds": {"constant_dominant_below": CONSTANT_DOMINANT,
                           "structure_cells": STRUCTURE_CELLS,
                           "attribution_floor": ATTRIBUTION_DELTA},
            "comparison": {
                "constant_share": {"demean": demean_field["constant_share"],
                                   "wide": wide_field["constant_share"]},
                "mean_abs_offset_cells": {"demean": demean_field["magnitude_cells"]["mean"],
                                          "wide": wide_field["magnitude_cells"]["mean"]},
                "occupied_mean_abs_offset_cells": {
                    "demean": demean_field["occupied_magnitude_cells"]["mean"],
                    "wide": wide_field["occupied_magnitude_cells"]["mean"]},
                "outside_own_cell_fraction": {
                    "demean": demean_field["outside_own_cell_fraction"],
                    "wide": wide_field["outside_own_cell_fraction"]},
                "at_bound_fraction": {"demean": demean_field["at_bound_fraction"],
                                      "wide": wide_field["at_bound_fraction"]},
                "board_mean_of_applied_field": {
                    "demean": demean_field["mean_vector_cells"],
                    "wide": wide_field["mean_vector_cells"]},
                "ranking": {"demean_best_f1": measured["demean"]["best"]["f1"],
                            "wide_best_f1": measured["wide"]["best"]["f1"],
                            "demean_auc": measured["demean"]["auc"],
                            "wide_auc": measured["wide"]["auc"]}},
            "field": {name: {"instrument": fields[name][0], **fields[name][1]["field"],
                             "arm": fields[name][1]["arm"],
                             "reference": fields[name][1]["reference"]} for name in DEFORM_ARMS},
            "oracle_statistic": {"mean_placement_offset_cells": 2.57,
                                 "wrong_target_cell_fraction": .483},
        },
        "mechanism": mechanism_of(measured, bootstrap),
        "verdict": verdict,
        "population": {"split": "training", "images": DECIDING_IMAGES,
                       "nonempty_boards": nonempty,
                       "occupied_cells": arms["demean"]["result"]["deciding"]["occupied_cells"],
                       "empty_cells": arms["demean"]["result"]["deciding"]["empty_cells_on_nonempty_boards"],
                       "contexts": contexts},
        "unchanged": {"thresholds": {"per_cell_f1": CELL_F1_THRESHOLD, "ranking_auc": AUC_THRESHOLD,
                                     "ranking_f1": F1_THRESHOLD},
                      "population": "widened deciding population, twelve groups",
                      "source_checkpoint": source["checkpoint_sha256"],
                      "prior_evidence": "reproduced, not touched", "frozen_manifest": "not touched"},
    }
    args.output.mkdir(parents=True)
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(scope, stream, indent=2, allow_nan=False)
    print(json.dumps({key: scope[key] for key in ("verdict", "constant_mode", "reproduction")},
                     indent=2, default=str))


def component_bootstrap(cases, grid):
    require_grid(grid)
    grouped = {}
    for name in ARMS:
        for row in cases[name]:
            grouped.setdefault(ancestry_of(row["sample_id"]), {}).setdefault(name, []).append(row)
    if len(grouped) < 2:
        raise ValueError("at least two ancestry clusters required")
    for cluster, arms in grouped.items():
        if set(arms) != set(ARMS):
            raise ValueError("every arm must cover every ancestry cluster")
    ordered = tuple((name, arms) for name, arms in sorted(grouped.items()))

    def statistic(sample):
        values = {}
        for name in ARMS:
            rows = deciding_rows([row for entry in sample for row in entry[name]])
            positive = [logit for logit, label in cells_of(rows) if label]
            negative = [logit for logit, label in cells_of(rows) if not label]
            if not positive or not negative:
                return None
            counts = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
            for logit, label in cells_of(rows):
                predicted = logit > 0.0
                if label and predicted:
                    counts["true_positive"] += 1
                elif label:
                    counts["false_negative"] += 1
                elif predicted:
                    counts["false_positive"] += 1
            values["auc_"+name] = pooled_auc(positive, negative)
            values["f1_"+name] = f1_at_threshold(rows)
            values["precision_"+name] = precision_of(counts)
            values["cv_f1_"+name] = out_of_fold(curves_of(rows), grid)["f1"]
        for first, second in CONTRASTS:
            for metric in METRICS:
                values[f"{metric}_delta_{first}_{second}"] = (values[f"{metric}_{first}"]
                                                              - values[f"{metric}_{second}"])
        return values

    names = tuple([f"{metric}_{name}" for metric in METRICS for name in ARMS] +
                  [f"{metric}_delta_{first}_{second}" for metric in METRICS
                   for first, second in CONTRASTS])
    return cluster_bootstrap(statistic, ordered, names, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)


if __name__ == "__main__":
    main()
