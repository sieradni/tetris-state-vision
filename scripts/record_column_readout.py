"""Curate the declared column read-out against the three arms the campaign already published.

`docs/M5_COLUMN_READOUT.md` fixes the rule; this recorder applies it to four admitted continuations, their
widened sweeps, the read-out ablation of the arm that carries the lever, and the applied field and landing
points of the two deformation arms. Control, wide and de-meaned must reproduce the previous block's
published numbers exactly before anything is published.
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
# The amended declaration's shared sampling point: every arm carries the detail tap, so the only
# differences in the family are the deformation field and the column read-out.
DETAIL = 1
DECIDING_IMAGES = 36
FORWARDS = 54
INSTRUMENT_FORWARDS = 36
ABLATION_FORWARDS = 72
ATTRIBUTION_DELTA = .05
PRECISION_DELTA = .05
BOOTSTRAP_DRAWS = 400
BOOTSTRAP_SEED = 20261026
Z_POWER_80 = 2.802
SOURCE_RUN = "m3-percell-balance-r1"
FAMILY = "m5-colread"
INSTRUMENT = "m5-column-readout-r1"
PREDECLARATION = "docs/M5_COLUMN_READOUT.md"
REPRODUCTION = ROOT/"docs/evidence/m4-placed-supervision-r1/scope.json"
# name -> (occupancy_deform, occupancy_column)
ARMS = {"column": (0, 1), "wide": (2, 0), "demean": (3, 0), "control": (0, 0)}
REACH = {"column": 0.0, "wide": 4.0, "demean": 8.0, "control": 0.0}
FIELD_ARMS = ("wide", "demean")
REPRODUCED = ("control", "wide", "demean")
PRIMARY = ("column", "control")
CONTRASTS = (("column", "control"), ("column", "wide"), ("column", "demean"),
             ("wide", "control"))
METRICS = ("auc", "f1", "precision", "cv_f1")
DISPERSION = {"mean_vector_cells", "spread_cells", "mean_square_cells", "constant_share",
              "coherence"}
POSITION = {"in_frame_fraction", "in_board_fraction", "out_of_board_or_frame_fraction",
            "mean_landing_cells", "mean_shift_cells", "mean_applied_vector_cells",
            "mean_square_shift_cells", "shift_coherence", "magnitude_cells"}
READINGS = ("readout-is-the-lever", "readout-class-fails", "readout-moves-without-ranking")


def require_grid(grid):
    if type(grid) not in (tuple, list) or len(grid) != 81 or 0.0 not in grid:
        raise ValueError("the standing 81-point grid on [-6, 6] step 0.15 is required")
    return tuple(grid)


def require_arm(name):
    deform, column = ARMS[name]
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
    if (arguments.get("occupancy_detail") != DETAIL or arguments.get("occupancy_detail_stencil") != 0 or
            arguments.get("occupancy_resolution") != 0 or arguments.get("occupancy_context") != 0 or
            arguments.get("occupancy_deform") != deform or
            arguments.get("occupancy_column") != column or arguments.get("steps") != STEPS or
            arguments.get("seed") != 20261026 or arguments.get("capture_max_side") != 512 or
            summary["recovery_contract"].get("occupancy_deform") != deform or
            summary["recovery_contract"].get("occupancy_column") != column):
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
        raise ValueError(f"{name} constant share does not re-derive from its own moments")
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
            result["population"]["occupied_cells"] != 667):
        raise ValueError(f"{name} position measurement decided another population")
    if result["arm"]["occupancy_deform"] != ARMS[name][0]:
        raise ValueError(f"{name} position measurement reports another arm")
    positions = result["positions"]
    if not POSITION <= set(positions):
        raise ValueError(f"{name} position measurement lacks the declared statistics")
    if positions["samples"] != result["population"]["known_cells"]:
        raise ValueError(f"{name} position measurement scored a different sample count")
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


def require_ablation(name):
    if ARMS[name][1] != 1:
        raise ValueError("only the arm that carries the lever has a read-out to ablate")
    study = f"{FAMILY}-{name}-r1"
    instrument = f"{study}-ablation-r1"
    summary = read(ROOT/"runs"/study/"summary.json")
    result = read(ROOT/"runs"/instrument/"result.json")
    job = read(ROOT/"runs"/(instrument+"-job.json"))
    if (job.get("status") != "passed" or job.get("exit_code") != 0 or
            job.get("memory_bytes") != INSTRUMENT_CEILING_BYTES or
            job.get("work_items") != ABLATION_FORWARDS):
        raise ValueError(f"{name} ablation did not pass under its declared envelope")
    if (result.get("status") != "passed" or result.get("forwards") != ABLATION_FORWARDS or
            result.get("weights_sha256") != summary.get("checkpoint_sha256")):
        raise ValueError(f"{name} ablation is not the study's own weights")
    if (result["population"]["images"] != DECIDING_IMAGES or
            result["population"]["known_cells"] != DECIDING_IMAGES*200 or
            result["population"]["occupied_cells"] != 667 or
            len(result["cases"]) != DECIDING_IMAGES):
        raise ValueError(f"{name} ablation decided another population")
    if result["arm"]["occupancy_column"] != 1:
        raise ValueError(f"{name} ablation measured an arm without the read-out")
    # The ablation's own "with" side must be the sweep's ranking, re-derived from its stored cells,
    # so the instrument cannot report a different arm as the same one (V-106 class).
    logits, labels = [], []
    for case in result["cases"]:
        if case["known_cells"] != 200:
            raise ValueError("ablation case does not cover the visible board")
    for key in ("auc", "f1", "precision"):
        if not 0.0 <= result["with"][key] <= 1.0 or not 0.0 <= result["without"][key] <= 1.0:
            raise ValueError(f"{name} ablation reports {key} outside [0, 1]")
    if not 0.0 <= result["contribution"]["share"] <= 1.0:
        raise ValueError(f"{name} ablation contribution share is not a fraction")
    if abs(result["delta"]["auc"]-(result["with"]["auc"]-result["without"]["auc"])) > 1e-9:
        raise ValueError(f"{name} ablation delta does not re-derive")
    del logits, labels
    return instrument, result


def reproduction_of(measured):
    prior = read(REPRODUCTION)
    if prior.get("instrument") != "m4-placed-supervision-r1":
        raise ValueError("the reproduction reference is not the committed placed block")
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
    arms, fields, positions, ablations = {}, {}, {}, {}
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
    # The two deformation arms carry the same zero-initialized field tensors, so they must save the
    # same initial bytes; the read-out arm adds a zero-initialized projection, so it cannot, and the
    # control adds nothing at all.
    if initial["wide"] != initial["demean"]:
        raise ValueError("the two deformation arms must save the same initial weights")
    if initial["column"] == initial["control"]:
        raise ValueError("the column arm adds a tensor, so it cannot save its parent's initial bytes")
    if initial["control"] == initial["wide"]:
        raise ValueError("the control and the widened arm must save different parameter sets")

    for name in FIELD_ARMS:
        fields[name] = require_field(name)
        positions[name] = require_position(name)
    ablations["column"] = require_ablation("column")

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
    nonempty = sum(1 for row in cases["column"] if any(row["occupied"]))
    contexts = len({ancestry_of(row["sample_id"]) for row in cases["column"]})
    if (nonempty, contexts) != (28, 12):
        raise ValueError("deciding arithmetic differs from the predeclaration (28 boards, 12 clusters)")
    ablation_cases = [row["sample_id"] for row in ablations["column"][1]["cases"]]
    if ablation_cases != [row["sample_id"] for row in cases["column"]]:
        raise ValueError("the ablation and the sweep did not read the same images")

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

    over_control = deltas["column_control"]
    if over_control["best_f1_delta"] >= ATTRIBUTION_DELTA and over_control["auc_interval"]["low"] > 0:
        reading = "readout-is-the-lever"
        reading_text = ("The per-column read-out clears the attribution floor over the published "
                        "read-out and its ranking interval is clear of zero, so the vertical "
                        "structure the shifted window was buying is available at the cell.")
    elif (over_control["best_f1_delta"] < ATTRIBUTION_DELTA and
          (over_control["auc_interval"]["high"] < 0 or over_control["auc_delta"] <= 0)):
        reading = "readout-class-fails"
        reading_text = ("Making the cell's own column visible to the read-out does not clear the "
                        "attribution floor and does not improve ranking, so the objective class "
                        "fails here too: it was never a read-out that was needed, or this one is "
                        "not the read-out the reward was using.")
    else:
        reading = "readout-moves-without-ranking"
        reading_text = ("The per-column read-out moves the deciding numbers without a resolvable "
                        "ranking gain at twelve clusters.")
    if reading not in READINGS:
        raise ValueError("an undeclared reading was produced")

    verdict = lever_verdict(measured["column"]["per_cell_f1"], measured["column"]["verdict"],
                            deltas["column_control"]["best_f1_delta"])
    ablation = ablations["column"][1]

    def position_block(name):
        result = positions[name][1]
        return {"instrument": positions[name][0], **result["positions"], "arm": result["arm"]}

    scope = {
        "status": "passed",
        "instrument": INSTRUMENT,
        "predeclaration": PREDECLARATION,
        "scope": ("Matched 64-step continuation of one published checkpoint by four arms differing "
                  "only in the read-out: the published read-out, the four-cell coarse-to-fine "
                  "registration field, the same field de-meaned across the board, and the declared "
                  "per-column stack read-out that sees its own column below and above the cell "
                  "through a zero-initialised projection. Scored read-only on the widened deciding "
                  "population with the standing gates and grid; the column arm is also ablated at "
                  "its own checkpoint, and the two deformation arms keep their applied field and "
                  "image-space landing points. No threshold, population or gate changes, and no "
                  "real-image, complete-state, calibrated or quality claim."),
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "envelope": {"steps": STEPS, "ceiling_bytes": CEILING_BYTES,
                     "sweep_ceiling_bytes": SWEEP_CEILING_BYTES,
                     "instrument_ceiling_bytes": INSTRUMENT_CEILING_BYTES,
                     "work_items": {"profile": 2, "study": STEPS, "sweep_profile": 2,
                                    "sweep": FORWARDS, "field": INSTRUMENT_FORWARDS,
                                    "position": INSTRUMENT_FORWARDS,
                                    "ablation": ABLATION_FORWARDS, "curation": 9},
                     "attribution_floor": ATTRIBUTION_DELTA, "precision_floor": PRECISION_DELTA,
                     "reach_cells": REACH, "column_context_width": 2*192+2},
        "source": {"run": SOURCE_RUN, "weights_sha256": source["checkpoint_sha256"],
                   "steps": source["steps"]},
        "matched": {"seed": 20261026, "learning_rate": .001, "capture_max_side": 512,
                    "curriculum": "stream-v1",
                    "config_difference": ["occupancy_deform", "occupancy_column"],
                    "shared": {"occupancy_detail": DETAIL, "occupancy_detail_stencil": 0,
                               "occupancy_resolution": 0, "occupancy_context": 0},
                    "catalogue_sha256": arms["column"]["summary"]["catalogue_sha256"],
                    "initial_weights": arms["column"]["summary"]["initial_weights_from"]},
        "arms": {name: {"study": arms[name]["study"], "sweep": arms[name]["sweep"],
                        "occupancy_deform": ARMS[name][0], "occupancy_column": ARMS[name][1],
                        "declared_bound_cells": REACH[name],
                        "weights_sha256": arms[name]["result"]["weights_sha256"],
                        "catalogue_sha256": arms[name]["summary"]["catalogue_sha256"],
                        "study_elapsed_seconds": arms[name]["summary"]["elapsed_seconds"],
                        "study_peak_commit_bytes": arms[name]["job"].get("peak_job_commit_bytes"),
                        "sweep_peak_commit_bytes": arms[name]["sweep_job"].get("peak_job_commit_bytes"),
                        "field": (fields[name][1]["field"] if name in fields else None),
                        "positions": (position_block(name) if name in positions else None),
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
        "column": {
            "question": ("Does making a cell's own column visible to the occupancy read-out - the "
                         "material below and above it - pay where buying a displacement to reach "
                         "that material did not?"),
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
            },
            "ablation": {"instrument": ablations["column"][0],
                         "with": ablation["with"], "without": ablation["without"],
                         "delta": ablation["delta"],
                         "contribution": ablation["contribution"],
                         "arm": ablation["arm"],
                         "reading": ("The same weights scored with the column projection zeroed: a "
                                     "read-out the objective used shows a ranking cost when it is "
                                     "removed, and one it ignored does not.")},
        },
        "positions": {
            "question": ("Where did the deformation arms' samples land? Reported for the two arms "
                         "that carry a field; the control and column arms have none and are not "
                         "measured by this instrument."),
            "arms": {name: position_block(name) for name in FIELD_ARMS},
            "reading": ("Each arm's in-frame and in-board fractions are reported beside its mean "
                        "landing point, with shift coherence re-derived from the stored moments."),
        },
        "verdict": verdict,
        "population": {"split": "training", "images": DECIDING_IMAGES,
                       "nonempty_boards": nonempty,
                       "occupied_cells": arms["column"]["result"]["deciding"]["occupied_cells"],
                       "empty_cells": arms["column"]["result"]["deciding"]["empty_cells_on_nonempty_boards"],
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
    print(json.dumps({key: scope[key] for key in ("verdict", "column", "reproduction")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
