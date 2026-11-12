"""Curate the placed-target lever against control, the widened field and the de-meaned field.

`docs/M4_OCCUPANCY_PLACED.md` fixes the rule; this recorder applies it to four admitted continuations,
their widened sweeps, the measured offset field of each deformation arm and the image-space landing
points of every arm. Control, wide and de-meaned must reproduce the previous block's published numbers
exactly before anything is published.
"""
import argparse
import json
from pathlib import Path

from cluster_interval import cluster_bootstrap
from gate_power import f1_from_counts
from occupancy_sweep import AUC_THRESHOLD, F1_THRESHOLD, pooled_auc, sweep_verdict, threshold_grid
from percell_gate import CELL_F1_THRESHOLD
from probe_model import require_bounded_job
from record_demean_lever import (ancestry_of, arm_summary, best_threshold, cells_of,
                                 components_of, counts_at_curves, curves_of, deciding_rows,
                                 detectable, f1_at_threshold, lever_verdict, out_of_fold,
                                 precision_of)
from record_representation_lever import read

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 1792*1024**2
SWEEP_CEILING_BYTES = 640*1024**2
INSTRUMENT_CEILING_BYTES = 640*1024**2
CURATION_CEILING_BYTES = 1900*1024**2
STEPS = 64
DECIDING_IMAGES = 36
FORWARDS = 54
INSTRUMENT_FORWARDS = 36
ATTRIBUTION_DELTA = .05
PRECISION_DELTA = .05
BOOTSTRAP_DRAWS = 400
BOOTSTRAP_SEED = 20261026
Z_POWER_80 = 2.802
SOURCE_RUN = "m3-percell-balance-r1"
FAMILY = "m4-occplaced"
INSTRUMENT = "m4-placed-supervision-r1"
PREDECLARATION = "docs/M4_OCCUPANCY_PLACED.md"
REPRODUCTION = ROOT/"docs/evidence/m3-occupancy-deformation-r3/scope.json"
# name -> (occupancy_deform, occupancy_placed)
ARMS = {"placed": (2, 1), "wide": (2, 0), "demean": (3, 0), "control": (0, 0)}
REACH = {"placed": 4.0, "wide": 4.0, "demean": 8.0, "control": 0.0}
REPRODUCED = ("control", "wide", "demean")
PRIMARY = ("placed", "wide")
CONTRASTS = (("placed", "control"), ("placed", "wide"), ("placed", "demean"),
             ("wide", "control"))
METRICS = ("auc", "f1", "precision", "cv_f1")
DISPERSION = {"mean_vector_cells", "spread_cells", "mean_square_cells", "constant_share",
              "coherence"}
POSITION = {"in_frame_fraction", "in_board_fraction", "out_of_board_or_frame_fraction",
            "mean_landing_cells", "mean_shift_cells", "mean_applied_vector_cells",
            "mean_square_shift_cells", "shift_coherence", "magnitude_cells"}
READINGS = ("supervision-is-the-lever", "supervision-moves-without-ranking",
            "objective-class-fails")


def require_grid(grid):
    if type(grid) not in (tuple, list) or len(grid) != 81 or 0.0 not in grid:
        raise ValueError("the standing 81-point grid on [-6, 6] step 0.15 is required")
    return tuple(grid)


def require_arm(name):
    deform, placed = ARMS[name]
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
            arguments.get("occupancy_deform") != deform or
            arguments.get("occupancy_placed") != placed or arguments.get("steps") != STEPS or
            arguments.get("seed") != 20261026 or arguments.get("capture_max_side") != 512 or
            summary["recovery_contract"].get("occupancy_deform") != deform or
            summary["recovery_contract"].get("occupancy_placed") != placed):
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
    study = f"{FAMILY}-{name}-r1"
    instrument = f"{study}-field-r1"
    summary = read(ROOT/"runs"/study/"summary.json")
    result = read(ROOT/"runs"/instrument/"result.json")
    job = read(ROOT/"runs"/(instrument+"-job.json"))
    if (job.get("status") != "passed" or job.get("exit_code") != 0 or
            job.get("memory_bytes") != INSTRUMENT_CEILING_BYTES or
            job.get("work_items") != INSTRUMENT_FORWARDS):
        raise ValueError(f"{name} field measurement did not pass under its declared envelope")
    if (result.get("status") != "passed" or result.get("forwards") != INSTRUMENT_FORWARDS or
            result.get("weights_sha256") != summary.get("checkpoint_sha256")):
        raise ValueError(f"{name} field measurement is not the study's own weights")
    if (result["population"]["images"] != DECIDING_IMAGES or
            result["population"]["known_cells"] != DECIDING_IMAGES*200 or
            result["population"]["occupied_cells"] != 667):
        raise ValueError(f"{name} field measurement decided another population")
    if (result["arm"]["occupancy_deform"] != ARMS[name][0] or
            result["arm"]["bound_cells"] != REACH[name]):
        raise ValueError(f"{name} field measurement reports another reach")
    field = result["field"]
    if not DISPERSION <= set(field):
        raise ValueError(f"{name} field measurement lacks the dispersion block")
    mean_x, mean_y = field["mean_vector_cells"]
    derived = ((mean_x*mean_x+mean_y*mean_y)/field["mean_square_cells"]
               if field["mean_square_cells"] else 1.0)
    if abs(derived-field["constant_share"]) > 1e-5:
        raise ValueError(f"{name} constant share does not re-derive from its own mean vector and "
                         "mean square")
    return instrument, result


def require_position(name):
    study = f"{FAMILY}-{name}-r1"
    instrument = f"{study}-position-r1"
    summary = read(ROOT/"runs"/study/"summary.json")
    result = read(ROOT/"runs"/instrument/"result.json")
    job = read(ROOT/"runs"/(instrument+"-job.json"))
    if (job.get("status") != "passed" or job.get("exit_code") != 0 or
            job.get("memory_bytes") != INSTRUMENT_CEILING_BYTES or
            job.get("work_items") != INSTRUMENT_FORWARDS):
        raise ValueError(f"{name} position measurement did not pass under its declared envelope")
    if (result.get("status") != "passed" or result.get("forwards") != INSTRUMENT_FORWARDS or
            result.get("weights_sha256") != summary.get("checkpoint_sha256")):
        raise ValueError(f"{name} position measurement is not the study's own weights")
    if (result["population"]["images"] != DECIDING_IMAGES or
            result["population"]["known_cells"] != DECIDING_IMAGES*200 or
            result["population"]["occupied_cells"] != 667 or len(result["cases"]) != DECIDING_IMAGES):
        raise ValueError(f"{name} position measurement decided another population")
    # The placed lever is a training-objective flag, not a model field, so it cannot appear in a
    # checkpoint's config; the measurement's arm identity is its deformation lever plus the weight
    # digest asserted above, and the declared placed flag is carried from the verified study.
    if result["arm"]["occupancy_deform"] != ARMS[name][0]:
        raise ValueError(f"{name} position measurement reports another arm")
    positions = result["positions"]
    if not POSITION <= set(positions):
        raise ValueError(f"{name} position measurement lacks the declared statistics")
    samples = positions["samples"]
    if samples != result["population"]["known_cells"]:
        raise ValueError(f"{name} position measurement scored a different sample count")
    # Re-derive the coherence and the mean magnitude from the stored numbers, the V-106 class guard.
    mean_x, mean_y = positions["mean_applied_vector_cells"]
    square = positions["mean_square_shift_cells"]
    derived = ((mean_x*mean_x+mean_y*mean_y)/square) if square else 1.0
    if abs(derived-positions["shift_coherence"]) > 1e-5:
        raise ValueError(f"{name} shift coherence does not re-derive from its own moments")
    weighted = sum(case["mean_magnitude_cells"]*case["known_cells"] for case in result["cases"])
    if abs(weighted/positions["samples"]-positions["mean_shift_cells"]) > 1e-6:
        raise ValueError(f"{name} mean shift does not re-derive from its own per-image means")
    for key in ("in_frame_fraction", "in_board_fraction", "out_of_board_or_frame_fraction"):
        if not 0.0 <= positions[key] <= 1.0:
            raise ValueError(f"{name} {key} is not a fraction")
    return instrument, result


def reproduction_of(measured):
    prior = read(REPRODUCTION)
    if prior.get("instrument") != "m3-occupancy-deformation-r3":
        raise ValueError("the reproduction reference is not the committed de-meaned block")
    checked = {}
    for name in REPRODUCED:
        published, current = prior["arms"][name], measured[name]
        for key in ("auc", "per_cell_f1"):
            if current[key] != published[key]:
                raise ValueError(f"{name} {key} does not reproduce the published {published[key]!r}")
        if current["best"]["f1"] != published["best"]["f1"]:
            raise ValueError(f"{name} best F1 does not reproduce the published value")
        checked[name] = {"auc": current["auc"], "per_cell_f1": current["per_cell_f1"],
                         "best_f1": current["best"]["f1"],
                         "prior_scope": str(REPRODUCTION.relative_to(ROOT))}
    return checked


def paired_bootstrap(cases, grid):
    require_grid(grid)
    grouped = {}
    for name in ARMS:
        for row in cases[name]:
            grouped.setdefault(ancestry_of(row["sample_id"]), {}).setdefault(name, []).append(row)
    if len(grouped) < 2:
        raise ValueError("at least two ancestry clusters required")
    for arms in grouped.values():
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


def main():
    bounded = require_bounded_job()
    if bounded != CURATION_CEILING_BYTES:
        raise ValueError("curation must run at its declared 1,900 MiB ceiling")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("evidence directory already exists")

    source = read(ROOT/"runs"/SOURCE_RUN/"summary.json")
    arms, fields, positions = {}, {}, {}
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
    if not (initial["placed"] == initial["wide"] == initial["demean"]):
        raise ValueError("the field arms add no tensors, so they must save the same initial weights")
    if initial["control"] == initial["wide"]:
        raise ValueError("the control and the widened arm must save different parameter sets")

    for name in ("placed", "wide", "demean"):
        fields[name] = require_field(name)
    for name in ARMS:
        positions[name] = require_position(name)

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
    nonempty = sum(1 for row in cases["placed"] if any(row["occupied"]))
    contexts = len({ancestry_of(row["sample_id"]) for row in cases["placed"]})
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
            raise ValueError(f"{name} components were recomputed on a different population")

    reproduction = reproduction_of(measured)
    bootstrap = paired_bootstrap(cases, grid)
    if bootstrap["clusters"] != 12:
        raise ValueError("the paired bootstrap must resample the twelve ancestry clusters")
    for first, second in CONTRASTS:
        printed = measured[first]["per_cell_f1"]-measured[second]["per_cell_f1"]
        if abs(bootstrap["f1_delta_"+first+"_"+second]["median"]-printed) > 0.05:
            raise ValueError("an F1 interval does not belong to the delta printed beside it")

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

    over_wide = deltas["placed_wide"]
    if over_wide["best_f1_delta"] >= ATTRIBUTION_DELTA and over_wide["auc_interval"]["low"] > 0:
        reading = "supervision-is-the-lever"
        reading_text = ("A supervised displacement beats the free field on ranking with an interval "
                        "clear of zero and clears the attribution floor, so per-cell registration is a "
                        "lever this architecture carries once it is told where to sample.")
    elif over_wide["best_f1_delta"] >= ATTRIBUTION_DELTA:
        reading = "supervision-moves-without-ranking"
        reading_text = ("The placed target clears the attribution floor over the free field but the "
                        "ranking interval still spans zero at twelve clusters, so registration moves "
                        "and does not yet rank.")
    else:
        reading = "objective-class-fails"
        reading_text = ("Supervising the displacement does not clear the attribution floor over the "
                        "free field, so neither occupancy-only nor placed-target supervision of this "
                        "field makes per-cell registration pay. The objective class is what fails, "
                        "not the reach, the bound or the field.")
    if reading not in READINGS:
        raise ValueError("an undeclared reading was produced")

    verdict = lever_verdict(measured["placed"]["per_cell_f1"], measured["placed"]["verdict"],
                            deltas["placed_control"]["best_f1_delta"])

    def position_block(name):
        result = positions[name][1]
        return {"instrument": positions[name][0], **result["positions"],
                "arm": result["arm"],
                "declared_occupancy_placed": ARMS[name][1],
                "declared_from": "the study's admission contract"}

    scope = {
        "status": "passed",
        "instrument": INSTRUMENT,
        "predeclaration": PREDECLARATION,
        "scope": ("Matched 64-step continuation of one published checkpoint by four arms differing "
                  "only in the registration lever: detail control, the four-cell coarse-to-fine field, "
                  "the same field de-meaned across the board, and the same field under a placed-target "
                  "term that supervises the displacement against the cell each sample should have "
                  "read. Scored read-only on the widened deciding population with the standing gates "
                  "and grid, with the applied field and its image-space landing points measured from "
                  "each frozen archive; no threshold, population or gate changes, and no real-image, "
                  "complete-state, calibrated or quality claim."),
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "envelope": {"steps": STEPS, "ceiling_bytes": CEILING_BYTES,
                     "sweep_ceiling_bytes": SWEEP_CEILING_BYTES,
                     "instrument_ceiling_bytes": INSTRUMENT_CEILING_BYTES,
                     "work_items": {"profile": 2, "study": STEPS, "sweep_profile": 2,
                                    "sweep": FORWARDS, "field": INSTRUMENT_FORWARDS,
                                    "position": INSTRUMENT_FORWARDS, "curation": 9},
                     "attribution_floor": ATTRIBUTION_DELTA, "precision_floor": PRECISION_DELTA,
                     "reach_cells": REACH, "placed_target_weight": .25},
        "source": {"run": SOURCE_RUN, "weights_sha256": source["checkpoint_sha256"],
                   "steps": source["steps"]},
        "matched": {"seed": 20261026, "learning_rate": .001, "capture_max_side": 512,
                    "curriculum": "stream-v1",
                    "config_difference": ["occupancy_deform", "occupancy_placed"],
                    "shared": {"occupancy_detail": 1, "occupancy_detail_stencil": 0,
                               "occupancy_resolution": 0, "occupancy_context": 0},
                    "catalogue_sha256": arms["placed"]["summary"]["catalogue_sha256"],
                    "initial_weights": arms["placed"]["summary"]["initial_weights_from"]},
        "arms": {name: {"study": arms[name]["study"], "sweep": arms[name]["sweep"],
                        "occupancy_deform": ARMS[name][0], "occupancy_placed": ARMS[name][1],
                        "declared_bound_cells": REACH[name],
                        "weights_sha256": arms[name]["result"]["weights_sha256"],
                        "catalogue_sha256": arms[name]["summary"]["catalogue_sha256"],
                        "study_elapsed_seconds": arms[name]["summary"]["elapsed_seconds"],
                        "study_peak_commit_bytes": arms[name]["job"].get("peak_job_commit_bytes"),
                        "sweep_peak_commit_bytes": arms[name]["sweep_job"].get("peak_job_commit_bytes"),
                        "field": (None if name == "control" else {
                            "instrument": fields[name][0],
                            **fields[name][1]["field"], "arm": fields[name][1]["arm"],
                            "reference": fields[name][1]["reference"]}),
                        "positions": position_block(name),
                        **measured[name]} for name in ARMS},
        "gates": {"per_cell_f1": CELL_F1_THRESHOLD, "ranking_auc": AUC_THRESHOLD,
                  "ranking_f1": F1_THRESHOLD, "attribution_floor": ATTRIBUTION_DELTA,
                  "grid_points": 81},
        "contrasts": deltas,
        "reproduction": reproduction,
        "components": {
            "note": ("Reported, never read by the verdict: ranking at each arm's own best boundary, "
                     "the cost of the fixed 0.0 boundary, and the cost of transferring a boundary "
                     "chosen on the other eleven ancestry groups, each with the interval that "
                     "resamples exactly it and a minimum detectable effect at 80% power."),
            "arms": {name: {key: measured[name]["components"][key] for key in
                            ("best_f1", "best_threshold", "best_precision", "zero_f1", "zero_cost",
                             "out_of_fold_f1", "transfer_cost", "out_of_fold_thresholds")}
                     for name in ARMS},
            "power": {"z": Z_POWER_80, "rule": "minimum detectable effect = 2.802 x cluster SE"}},
        "placed": {
            "question": ("Does supervising the displacement - telling the field which cell each sample "
                         "should have read - make per-cell registration pay where occupancy-only "
                         "supervision of the same field did not?"),
            "reading": reading,
            "reading_text": reading_text,
            "thresholds": {"attribution_floor": ATTRIBUTION_DELTA,
                           "auc_interval_rule": "clear of zero"},
            "comparison": {
                "best_f1": {name: measured[name]["best"]["f1"] for name in ARMS},
                "auc": {name: measured[name]["auc"] for name in ARMS},
                "per_cell_f1": {name: measured[name]["per_cell_f1"] for name in ARMS},
                "out_of_fold_f1": {name: measured[name]["components"]["out_of_fold_f1"]
                                   for name in ARMS},
                "constant_share": {name: fields[name][1]["field"]["constant_share"]
                                   for name in fields},
                "mean_abs_offset_cells": {name: fields[name][1]["field"]["magnitude_cells"]["mean"]
                                          for name in fields},
            },
            "field": {name: {"instrument": fields[name][0], **fields[name][1]["field"],
                             "arm": fields[name][1]["arm"],
                             "reference": fields[name][1]["reference"]} for name in fields},
        },
        "positions": {
            "question": ("Where did the displaced samples actually land? The field instrument says how "
                         "far they moved; this says whether a shifted-window exploit at the frame edge "
                         "or a decode offset explains the wide arm's 5.128-cell translation."),
            "arms": {name: position_block(name) for name in ARMS},
            "reading": ("Each arm's in-frame and in-board fractions are reported beside its mean "
                        "landing point, with shift coherence re-derived from the stored moments."),
        },
        "mechanism": {"field_active": True,
                      "read": "precision at each arm's own best boundary",
                      "placed_best_precision_delta_over_wide": round(
                          measured["placed"]["best"]["precision"]
                          - measured["wide"]["best"]["precision"], 6),
                      "floor": PRECISION_DELTA},
        "verdict": verdict,
        "population": {"split": "training", "images": DECIDING_IMAGES,
                       "nonempty_boards": nonempty,
                       "occupied_cells": arms["placed"]["result"]["deciding"]["occupied_cells"],
                       "empty_cells": arms["placed"]["result"]["deciding"]["empty_cells_on_nonempty_boards"],
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
    print(json.dumps({key: scope[key] for key in ("verdict", "placed", "reproduction")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
