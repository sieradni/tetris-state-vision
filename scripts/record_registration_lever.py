"""Curate the declared per-cell registration lever against its matched control.

`docs/M3_OCCUPANCY_REGISTRATION.md` fixes the rule; this recorder applies it to the two admitted
continuations and their two widened sweeps. Both arms carry the declared detail tap with a single
center sample and differ only in whether a bounded per-cell field moves that sample. The three
standing gates are the verdict; the ranking components, the out-of-fold boundary and its cluster
interval are reported beside them and read by nothing.
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
STEPS = 64
DECIDING_IMAGES = 36
DEVELOPMENT_IMAGES = 18
FORWARDS = DECIDING_IMAGES+DEVELOPMENT_IMAGES
ATTRIBUTION_DELTA = 0.05
PRECISION_DELTA = 0.05
BOOTSTRAP_DRAWS = 400
BOOTSTRAP_SEED = 20261026
Z_POWER_80 = 2.802
SOURCE_RUN = "m3-percell-balance-r1"
FAMILY = "m3-occdeform"
INSTRUMENT = "m3-occupancy-deformation-r1"
PREDECLARATION = "docs/M3_OCCUPANCY_REGISTRATION.md"
REFERENCE = ROOT/"docs/evidence/m3-occupancy-resolution-r2/scope.json"
ARMS = {"lever": 1, "control": 0}


def require_grid(grid):
    """The standing 81-point grid, refused if a caller substitutes another one."""
    if type(grid) not in (tuple, list) or len(grid) != 81 or 0.0 not in grid:
        raise ValueError("the standing 81-point grid on [-6, 6] step 0.15 is required")
    return tuple(grid)


def curves_of(rows):
    """Per-ancestry-cluster sorted logits, so every boundary read is a pair of bisections."""
    grouped = {}
    for row in rows:
        positive, negative = grouped.setdefault(ancestry_of(row["sample_id"]), ([], []))
        for logit, label in zip(row["logits"], row["occupied"]):
            (positive if label else negative).append(logit)
    return tuple((name, sorted(positive), sorted(negative))
                 for name, (positive, negative) in sorted(grouped.items()))


def counts_at_curves(curves, threshold):
    """Occupied/empty counts at one threshold, pooled over the given clusters."""
    true_positive = false_positive = false_negative = 0
    for _, positive, negative in curves:
        hit = len(positive)-bisect.bisect_right(positive, threshold)
        true_positive += hit
        false_negative += len(positive)-hit
        false_positive += len(negative)-bisect.bisect_right(negative, threshold)
    return true_positive, false_positive, false_negative


def best_threshold(curves, grid):
    """The sweep's own rule: highest F1, then higher precision, then lower threshold."""
    best = None
    for threshold in grid:
        true_positive, false_positive, false_negative = counts_at_curves(curves, threshold)
        f1 = f1_from_counts(true_positive, false_positive, false_negative)
        precision = 0.0 if not true_positive+false_positive else true_positive/(true_positive+false_positive)
        if best is None or (f1, precision) > best[0]:
            best = ((f1, precision), threshold)
    return best[1]


def out_of_fold(curves, grid):
    """Leave-one-ancestry-group-out boundary, then pooled counts from the twelve held-out groups.

    The boundary applied to a group is chosen on the other eleven, so it is invariant to any
    monotone shift of an arm's logits and cannot be fitted to the cells it scores.
    """
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
    """The sweep's own deciding population: training boards with at least one occupied cell.

    The eight empty training boards carry 1,600 empty known cells that the sweep's `deciding` block
    excludes, so a component computed over all 36 images would describe a different population from
    the one the verdict reads. One helper feeds both (V-097).
    """
    trials = [row for row in rows if any(row["occupied"])]
    if not trials:
        raise ValueError("the deciding population is empty")
    return trials


def components_of(rows, grid):
    """The declared three-part primary: ranking at the best boundary, zero cost, transfer cost."""
    curves = curves_of(deciding_rows(rows))
    best = best_threshold(curves, grid)
    best_counts = counts_at_curves(curves, best)
    zero_counts = counts_at_curves(curves, 0.0)
    folded = out_of_fold(curves, grid)
    best_f1 = round(f1_from_counts(*best_counts), 6)
    zero_f1 = round(f1_from_counts(*zero_counts), 6)
    return {"best_f1": best_f1, "best_threshold": best,
            "zero_f1": zero_f1, "zero_cost": round(zero_f1-best_f1, 6),
            "out_of_fold_f1": folded["f1"],
            "transfer_cost": round(folded["f1"]-zero_f1, 6),
            "out_of_fold_thresholds": folded["thresholds"],
            "out_of_fold_counts": folded["counts"], "groups": folded["groups"]}


def component_bootstrap(lever_cases, control_cases, grid):
    """Whole-cluster resampling of the operating-point and out-of-fold components together."""
    require_grid(grid)
    grouped = {}
    for name, rows in (("lever", lever_cases), ("control", control_cases)):
        for row in rows:
            grouped.setdefault(ancestry_of(row["sample_id"]), {}).setdefault(name, []).append(row)
    if len(grouped) < 2:
        raise ValueError("at least two ancestry clusters required")
    for cluster, arms in grouped.items():
        if set(arms) != set(ARMS):
            raise ValueError("both arms must cover every ancestry cluster")
    ordered = tuple((name, arms) for name, arms in sorted(grouped.items()))

    def statistic(sample):
        values = {}
        for name in ("lever", "control"):
            rows = deciding_rows([row for entry in sample for row in entry[name]])
            positive = [logit for logit, label in cells_of(rows) if label]
            negative = [logit for logit, label in cells_of(rows) if not label]
            if not positive or not negative:
                return None
            values["auc_"+name] = pooled_auc(positive, negative)
            values["f1_"+name] = f1_at_threshold(rows)
            values["cv_f1_"+name] = out_of_fold(curves_of(rows), grid)["f1"]
            counts = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
            for logit, label in cells_of(rows):
                predicted = logit > 0.0
                if label and predicted:
                    counts["true_positive"] += 1
                elif label:
                    counts["false_negative"] += 1
                elif predicted:
                    counts["false_positive"] += 1
            values["precision_"+name] = precision_of(counts)
        values["f1_delta"] = values["f1_lever"]-values["f1_control"]
        values["cv_f1_delta"] = values["cv_f1_lever"]-values["cv_f1_control"]
        values["auc_delta"] = values["auc_lever"]-values["auc_control"]
        values["precision_delta"] = values["precision_lever"]-values["precision_control"]
        return values

    return cluster_bootstrap(statistic, ordered,
                             ("auc_lever", "f1_lever", "cv_f1_lever", "precision_lever",
                              "auc_control", "f1_control", "cv_f1_control", "precision_control",
                              "f1_delta", "cv_f1_delta", "auc_delta", "precision_delta"),
                             draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)


def detectable(interval):
    """Minimum detectable effect at 80% power from the resampled standard error."""
    return round(Z_POWER_80*interval["standard_error"], 6)


def mechanism_of(measured, bootstrap):
    """Does the field improve what it decides, read at each arm's own boundary?

    The 0.0 boundary is not calibration-neutral here: the field re-scales the logits it feeds, which is
    why this pass's best threshold moved from 0.9 to 3.15, so a fixed-boundary precision delta measures
    the replay point rather than the mechanism (V-103). The mechanism is therefore read at the arm's own
    best boundary - no interval, because that boundary is fitted on the cells it scores - and the
    fixed-boundary delta is reported beside it with the interval that resamples exactly that statistic,
    as a replay cost and never as a verdict.
    """
    best = measured["lever"]["best"]["precision"]-measured["control"]["best"]["precision"]
    zero = measured["lever"]["precision"]-measured["control"]["precision"]
    interval = bootstrap["precision_delta"]
    if abs(zero-interval["median"]) > 0.05:
        raise ValueError("the precision interval does not belong to the fixed-boundary delta it is "
                         "reported with")
    return {"field_active": True,
            "read": "precision at the arm's own best boundary",
            "best_precision_delta": round(best, 6),
            "best_precision_improved": bool(best >= PRECISION_DELTA),
            "floor": PRECISION_DELTA,
            "zero_boundary_precision_delta": {
                "delta": round(zero, 6), "interval": interval,
                "minimum_detectable_effect": detectable(interval),
                "note": ("fixed 0.0 boundary, which this lever's own logit scale moves; a replay cost, "
                         "not a mechanism verdict")}}


def unmatched_reference():
    """The resolution pass's arm, under another catalogue: context, never a comparison."""
    record = read(REFERENCE)
    if record.get("instrument") != "m3-occupancy-resolution-r2":
        raise ValueError("the reference record is not the committed resolution pass")
    arm = record["arms"]["lever"]
    return {"instrument": record["instrument"], "predeclaration": record["predeclaration"],
            "matched_to_this_pass": False,
            "arm": {key: arm[key] for key in ("auc", "per_cell_f1", "best", "components")}}


def require_arm(name):
    deform = ARMS[name]
    study = f"{FAMILY}-{name}-r1"
    summary = read(ROOT/"runs"/study/"summary.json")
    job = read(ROOT/"runs"/(study+"-job.json"))
    if job.get("status") != "passed" or job.get("exit_code") != 0:
        raise ValueError(f"{name} study did not pass inside its declared envelope")
    if (summary.get("status") != "complete" or summary.get("steps") != STEPS or
            summary.get("requested_steps") != STEPS or not summary.get("checkpoint_round_trip_exact") or
            job.get("work_items") != STEPS or job.get("memory_bytes") != CEILING_BYTES):
        raise ValueError(f"{name} study is not a complete 64-step run inside 1,792 MiB")
    arguments = summary["arguments"]
    if (arguments.get("occupancy_detail") != 1 or arguments.get("occupancy_detail_stencil") != 0 or
            arguments.get("occupancy_resolution") != 0 or arguments.get("occupancy_context") != 0 or
            arguments.get("occupancy_deform") != deform or arguments.get("steps") != STEPS or
            arguments.get("seed") != 20261026 or arguments.get("capture_max_side") != 512 or
            summary["recovery_contract"].get("occupancy_deform") != deform):
        raise ValueError(f"{name} study did not run the declared single lever at the declared budget")
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
    if arms["lever"]["summary"]["catalogue_sha256"] != arms["control"]["summary"]["catalogue_sha256"]:
        raise ValueError("the arms were captured under different catalogues")
    if arms["lever"]["summary"]["initial_weights_sha256"] == arms["control"]["summary"]["initial_weights_sha256"]:
        raise ValueError("the lever and control must not share one saved initial tensor file")

    lever_cases = [row for row in arms["lever"]["result"]["cases"] if row["split"] == "training"]
    control_cases = [row for row in arms["control"]["result"]["cases"] if row["split"] == "training"]
    if len(lever_cases) != DECIDING_IMAGES or len(control_cases) != DECIDING_IMAGES:
        raise ValueError("the curation requires exactly the 36 deciding training images")
    if [row["sample_id"] for row in lever_cases] != [row["sample_id"] for row in control_cases]:
        raise ValueError("the two sweeps must share one deciding population")
    if [row["logits"] for row in lever_cases] == [row["logits"] for row in control_cases]:
        raise ValueError("the two arms produced identical logits; the field did not run")
    nonempty = sum(1 for row in lever_cases if any(row["occupied"]))
    contexts = len({ancestry_of(row["sample_id"]) for row in lever_cases})
    if (nonempty, contexts) != (28, 12):
        raise ValueError("deciding arithmetic differs from the predeclaration (28 boards, 12 clusters)")

    grid = require_grid(threshold_grid())
    measured = {}
    for name in ARMS:
        block = arm_summary(arms[name]["result"]["deciding"])
        block["precision"] = round(precision_of(block["operating_counts"]), 6)
        block["components"] = components_of(
            [row for row in arms[name]["result"]["cases"] if row["split"] == "training"], grid)
        measured[name] = block
    for name in ARMS:
        components = measured[name]["components"]
        if (abs(components["zero_f1"]-measured[name]["per_cell_f1"]) > 1e-6 or
                abs(components["best_f1"]-measured[name]["best"]["f1"]) > 1e-6):
            raise ValueError(f"{name} components were recomputed on a different population from the "
                             "deciding rows the verdict reads")
    bootstrap = component_bootstrap(lever_cases, control_cases, grid)
    if bootstrap["clusters"] != 12:
        raise ValueError("the paired bootstrap must resample the twelve ancestry clusters")
    if abs(bootstrap["f1_delta"]["median"]
           - (measured["lever"]["per_cell_f1"]-measured["control"]["per_cell_f1"])) > 0.05:
        raise ValueError("the F1 interval does not belong to the fixed-boundary delta it is reported with")
    delta_best = measured["lever"]["best"]["f1"]-measured["control"]["best"]["f1"]
    delta_precision = measured["lever"]["precision"]-measured["control"]["precision"]
    verdict = lever_verdict(measured["lever"]["per_cell_f1"], measured["lever"]["verdict"], delta_best)
    reference = unmatched_reference()
    lever_components, control_components = measured["lever"]["components"], measured["control"]["components"]
    if lever_components["out_of_fold_f1"] < 0 or control_components["out_of_fold_f1"] < 0:
        raise ValueError("an out-of-fold component is not a proportion")

    scope = {
        "status": "passed",
        "instrument": INSTRUMENT,
        "predeclaration": PREDECLARATION,
        "scope": ("Matched 64-step continuation of one published checkpoint, both arms carrying the "
                  "declared zero-initialized detail tap with a single center sample and differing only "
                  "in whether a bounded per-cell field moves that sample: the lever samples at the "
                  "decoded cell centre plus a field of at most one cell, the control at the centre. "
                  "Scored read-only on the widened deciding population with the standing gates and "
                  "grid; no threshold, population or gate changes, and no real-image, complete-state, "
                  "calibrated or quality claim."),
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "envelope": {"steps": STEPS, "ceiling_bytes": CEILING_BYTES,
                     "sweep_ceiling_bytes": SWEEP_CEILING_BYTES,
                     "work_items": {"profile": 2, "study": STEPS, "sweep_profile": 2,
                                    "sweep": FORWARDS, "curation": 4},
                     "attribution_floor": ATTRIBUTION_DELTA, "precision_floor": PRECISION_DELTA,
                     "deform_limit_cells": 1.0},
        "source": {"run": SOURCE_RUN, "weights_sha256": source["checkpoint_sha256"],
                   "steps": source["steps"]},
        "matched": {"seed": 20261026, "learning_rate": .001, "capture_max_side": 512,
                    "curriculum": "stream-v1", "config_difference": ["occupancy_deform"],
                    "shared": {"occupancy_detail": 1, "occupancy_detail_stencil": 0,
                               "occupancy_resolution": 0, "occupancy_context": 0},
                    "catalogue_sha256": arms["lever"]["summary"]["catalogue_sha256"],
                    "initial_weights": arms["lever"]["summary"]["initial_weights_from"]},
        "arms": {name: {"study": arms[name]["study"], "sweep": arms[name]["sweep"],
                        "occupancy_deform": ARMS[name],
                        "weights_sha256": arms[name]["result"]["weights_sha256"],
                        "catalogue_sha256": arms[name]["summary"]["catalogue_sha256"],
                        "study_elapsed_seconds": arms[name]["summary"]["elapsed_seconds"],
                        "study_peak_commit_bytes": arms[name]["job"].get("peak_job_commit_bytes"),
                        "sweep_peak_commit_bytes": arms[name]["sweep_job"].get("peak_job_commit_bytes"),
                        **measured[name]} for name in ARMS},
        "gates": {"per_cell_f1": CELL_F1_THRESHOLD, "ranking_auc": AUC_THRESHOLD,
                  "ranking_f1": F1_THRESHOLD, "attribution_floor": ATTRIBUTION_DELTA,
                  "grid_points": 81},
        "attribution": {"read": "best swept F1 delta at the arm's own boundary",
                        "best_f1_delta": round(delta_best, 6),
                        "auc_delta": round(measured["lever"]["auc"]-measured["control"]["auc"], 6),
                        "per_cell_f1_delta": round(
                            measured["lever"]["per_cell_f1"]-measured["control"]["per_cell_f1"], 6),
                        "precision_delta": round(delta_precision, 6),
                        "recall_delta": round(
                            measured["lever"]["operating_intervals"]["recall"]["estimate"]
                            - measured["control"]["operating_intervals"]["recall"]["estimate"], 6),
                        "cluster_bootstrap": bootstrap},
        "components": {
            "note": ("Reported, never read: ranking at the arm's own best boundary, the cost of the "
                     "fixed 0.0 boundary, and the cost of transferring a boundary chosen on the other "
                     "eleven ancestry groups. Each delta carries the interval that resamples it and a "
                     "minimum detectable effect at 80% power."),
            "lever": {key: lever_components[key] for key in
                      ("best_f1", "best_threshold", "zero_f1", "zero_cost", "out_of_fold_f1",
                       "transfer_cost", "out_of_fold_thresholds")},
            "control": {key: control_components[key] for key in
                        ("best_f1", "best_threshold", "zero_f1", "zero_cost", "out_of_fold_f1",
                         "transfer_cost", "out_of_fold_thresholds")},
            "deltas": {
                "best_f1": {"delta": round(delta_best, 6), "interval": None,
                            "note": "boundary fitted on the scored cells; reported, never read"},
                "zero_boundary_f1": {
                    "delta": round(measured["lever"]["per_cell_f1"]
                                   - measured["control"]["per_cell_f1"], 6),
                    "interval": bootstrap["f1_delta"],
                    "minimum_detectable_effect": detectable(bootstrap["f1_delta"])},
                "out_of_fold_f1": {
                    "delta": round(lever_components["out_of_fold_f1"]
                                   - control_components["out_of_fold_f1"], 6),
                    "interval": bootstrap["cv_f1_delta"],
                    "minimum_detectable_effect": detectable(bootstrap["cv_f1_delta"])}},
            "power": {"z": Z_POWER_80, "rule": "minimum detectable effect = 2.802 x cluster SE"}},
        "mechanism": mechanism_of(measured, bootstrap),
        "verdict": verdict,
        "unmatched_reference": reference,
        "population": {"split": "training", "images": DECIDING_IMAGES,
                       "nonempty_boards": nonempty,
                       "occupied_cells": arms["lever"]["result"]["deciding"]["occupied_cells"],
                       "empty_cells": arms["lever"]["result"]["deciding"]["empty_cells_on_nonempty_boards"],
                       "contexts": contexts},
        "unchanged": {"thresholds": {"per_cell_f1": CELL_F1_THRESHOLD, "ranking_auc": AUC_THRESHOLD,
                                     "ranking_f1": F1_THRESHOLD},
                      "population": "widened deciding population, twelve groups",
                      "source_checkpoint": source["checkpoint_sha256"],
                      "prior_evidence": "not touched", "frozen_manifest": "not touched"},
    }
    args.output.mkdir(parents=True)
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(scope, stream, indent=2, allow_nan=False)
    print(json.dumps({key: scope[key] for key in ("verdict", "components", "mechanism")}, indent=2))


if __name__ == "__main__":
    main()
