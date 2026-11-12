"""Curate the declared row sampling offset against control, wide and the opposite sign.

`docs/M6_ROW_OFFSET.md` fixes the rule; this recorder applies it to four admitted continuations and their
widened sweeps, re-derives the row-error diagnostic on every arm from the same frozen sweep records, and
requires the published diagnostic's numbers to come back exactly. Control and wide must reproduce the last
block's published numbers before anything is published.
"""
import argparse
import json
from pathlib import Path

from cluster_interval import cluster_bootstrap
from gate_power import f1_from_counts
from measure_row_error import arm_block
from occupancy_sweep import AUC_THRESHOLD, F1_THRESHOLD, pooled_auc, sweep_verdict, threshold_grid
from percell_gate import CELL_F1_THRESHOLD
from probe_model import require_bounded_job
from record_demean_lever import (ancestry_of, arm_summary, components_of, cells_of, curves_of,
                                 deciding_rows, detectable, f1_at_threshold, lever_verdict,
                                 out_of_fold, precision_of)
from record_representation_lever import read

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 1792*1024**2
SWEEP_CEILING_BYTES = 640*1024**2
CURATION_CEILING_BYTES = 1900*1024**2
STEPS = 64
DETAIL = 1
DECIDING_IMAGES = 36
FORWARDS = 54
ATTRIBUTION_DELTA = .05
PRECISION_DELTA = .05
BOOTSTRAP_DRAWS = 400
BOOTSTRAP_SEED = 20261026
Z_POWER_80 = 2.802
ROW_SHIFT_FLOOR = .5
SOURCE_RUN = "m3-percell-balance-r1"
FAMILY = "m6-rowoffset"
INSTRUMENT = "m6-row-offset-r1"
PREDECLARATION = "docs/M6_ROW_OFFSET.md"
REPRODUCTION = ROOT/"docs/evidence/m5-column-readout-r1/scope.json"
DIAGNOSTIC = ROOT/"runs/m5-rowerror-r1/result.json"
# name -> (occupancy_deform, occupancy_offset_rows)
ARMS = {"up": (0, 1), "down": (0, -1), "wide": (2, 0), "control": (0, 0)}
REPRODUCED = ("control", "wide")
DECLARED = "up"
CONTRASTS = (("up", "control"), ("up", "wide"), ("down", "control"), ("wide", "control"))
METRICS = ("auc", "f1", "precision", "cv_f1")
READINGS = ("offset-is-the-lever", "offset-moves-without-ranking", "offset-class-fails")


def require_grid(grid):
    if type(grid) not in (tuple, list) or len(grid) != 81 or 0.0 not in grid:
        raise ValueError("the standing 81-point grid on [-6, 6] step 0.15 is required")
    return tuple(grid)


def require_arm(name):
    deform, offset = ARMS[name]
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
    if (arguments.get("occupancy_detail") != DETAIL or
            arguments.get("occupancy_detail_stencil") != 0 or
            arguments.get("occupancy_resolution") != 0 or arguments.get("occupancy_context") != 0 or
            arguments.get("occupancy_column") != 0 or arguments.get("occupancy_placed") != 0 or
            arguments.get("occupancy_deform") != deform or
            arguments.get("occupancy_offset_rows") != offset or arguments.get("steps") != STEPS or
            arguments.get("seed") != 20261026 or arguments.get("capture_max_side") != 512 or
            summary["recovery_contract"].get("occupancy_deform") != deform or
            summary["recovery_contract"].get("occupancy_offset_rows") != offset):
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
    if declared.get("evaluated_groups") != 12 or declared.get("evaluated_cases") != DECIDING_IMAGES:
        raise ValueError(f"{name} sweep did not score the widened deciding population")
    if (profile_job.get("status") != "passed" or profile.get("status") != "complete" or
            profile.get("weights_sha256") != result.get("weights_sha256")):
        raise ValueError(f"{name} sweep admission profile differs")
    return sweep, result, job


def reproduction_of(measured):
    prior = read(REPRODUCTION)
    if prior.get("instrument") != "m5-column-readout-r1":
        raise ValueError("the reproduction reference is not the committed column block")
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
    if not (initial["control"] == initial["up"] == initial["down"]):
        raise ValueError("a declared offset adds no tensor, so it must save its parent's initial bytes")
    if initial["control"] == initial["wide"]:
        raise ValueError("the widened arm adds a field, so it must save different initial bytes")

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
    nonempty = sum(1 for row in cases["up"] if any(row["occupied"]))
    contexts = len({ancestry_of(row["sample_id"]) for row in cases["up"]})
    if (nonempty, contexts) != (28, 12):
        raise ValueError("deciding arithmetic differs from the predeclaration (28 boards, 12 clusters)")

    # The same diagnostic, recomputed on all four sweeps of this block. The two arms that also exist in
    # the previous block must return exactly the published diagnostic's numbers, or these are not the
    # same records (V-106 class).
    row_error = {}
    for name in ARMS:
        block = arm_block(arms[name]["result"], name)
        row_error[name] = {"threshold": 0.0,
                           "mean": block["zero"]["error_rows"]["mean"],
                           "median": block["zero"]["error_rows"]["median"],
                           "exact_fraction": block["zero"]["exact_fraction"],
                           "within_one_fraction": block["zero"]["within_one_fraction"],
                           "columns": block["zero"]["columns"],
                           "error_counts": block["zero"]["error_counts"],
                           "at_best_threshold": {
                               "threshold": block["best"]["threshold"],
                               "mean": block["best"]["error_rows"]["mean"],
                               "median": block["best"]["error_rows"]["median"],
                               "exact_fraction": block["best"]["exact_fraction"],
                               "within_one_fraction": block["best"]["within_one_fraction"]}}
    published = read(DIAGNOSTIC)
    for name in ("control", "wide"):
        if abs(published["grid"]["zero"][name]["mean"]-row_error[name]["mean"]) > 1e-9:
            raise ValueError(f"{name} row error does not reproduce the published diagnostic")
    moved = round(row_error["control"]["mean"]-row_error[DECLARED]["mean"], 6)

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
            "row_error_shift_toward_zero": (round(row_error[second]["mean"]-row_error[first]["mean"], 6)
                                            if second in row_error and first in row_error else None),
            "zero_boundary_interval": bootstrap["f1_delta_"+key],
            "zero_boundary_minimum_detectable_effect": detectable(bootstrap["f1_delta_"+key]),
            "out_of_fold_interval": bootstrap["cv_f1_delta_"+key],
            "out_of_fold_minimum_detectable_effect": detectable(bootstrap["cv_f1_delta_"+key]),
            "auc_interval": bootstrap["auc_delta_"+key],
            "auc_minimum_detectable_effect": detectable(bootstrap["auc_delta_"+key]),
            "precision_interval": bootstrap["precision_delta_"+key],
            "precision_minimum_detectable_effect": detectable(bootstrap["precision_delta_"+key]),
        }

    over_control = deltas[DECLARED+"_control"]
    if (over_control["best_f1_delta"] >= ATTRIBUTION_DELTA and
            over_control["auc_interval"]["low"] > 0 and moved >= ROW_SHIFT_FLOOR):
        reading = "offset-is-the-lever"
        reading_text = ("The declared one-row sampling offset clears the attribution floor, its ranking "
                        "interval is clear of zero and its boundary moves at least half a row toward "
                        "the labels, so the offset the diagnostic predicted is the mechanism.")
    elif moved >= ROW_SHIFT_FLOOR:
        reading = "offset-moves-without-ranking"
        reading_text = ("The declared sampling offset moves the decision boundary toward the labels by "
                        "at least half a row without clearing the attribution floor or a ranking "
                        "interval, so the geometry moves and the ranking does not.")
    else:
        reading = "offset-class-fails"
        reading_text = ("The declared sampling offset moves neither the boundary far enough to count "
                        "nor the ranking, so the measured row bias is not corrected by moving where "
                        "this read looks.")
    if reading not in READINGS:
        raise ValueError("an undeclared reading was produced")

    verdict = lever_verdict(measured[DECLARED]["per_cell_f1"], measured[DECLARED]["verdict"],
                            over_control["best_f1_delta"])

    scope = {
        "status": "passed",
        "instrument": INSTRUMENT,
        "predeclaration": PREDECLARATION,
        "scope": ("Matched 64-step continuation of one published checkpoint by four arms differing "
                  "only in the declared sampling geometry: the published sampling point, the "
                  "four-cell registration field, and a constant one-row offset above and below the "
                  "cell. Scored read-only on the widened deciding population with the standing gates "
                  "and grid, with the row-error diagnostic recomputed on every arm from the same "
                  "frozen sweep records. No threshold, population or gate changes, and no real-image, "
                  "complete-state, calibrated or quality claim."),
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "envelope": {"steps": STEPS, "ceiling_bytes": CEILING_BYTES,
                     "sweep_ceiling_bytes": SWEEP_CEILING_BYTES,
                     "work_items": {"profile": 2, "study": STEPS, "sweep_profile": 2,
                                    "sweep": FORWARDS, "curation": 9},
                     "attribution_floor": ATTRIBUTION_DELTA, "precision_floor": PRECISION_DELTA,
                     "row_shift_floor_cells": ROW_SHIFT_FLOOR},
        "source": {"run": SOURCE_RUN, "weights_sha256": source["checkpoint_sha256"],
                   "steps": source["steps"]},
        "matched": {"seed": 20261026, "learning_rate": .001, "capture_max_side": 512,
                    "curriculum": "stream-v1",
                    "config_difference": ["occupancy_deform", "occupancy_offset_rows"],
                    "shared": {"occupancy_detail": DETAIL, "occupancy_detail_stencil": 0,
                               "occupancy_resolution": 0, "occupancy_context": 0,
                               "occupancy_column": 0, "occupancy_placed": 0},
                    "catalogue_sha256": arms["up"]["summary"]["catalogue_sha256"],
                    "initial_weights": arms["up"]["summary"]["initial_weights_from"]},
        "arms": {name: {"study": arms[name]["study"], "sweep": arms[name]["sweep"],
                        "occupancy_deform": ARMS[name][0],
                        "occupancy_offset_rows": ARMS[name][1],
                        "weights_sha256": arms[name]["result"]["weights_sha256"],
                        "catalogue_sha256": arms[name]["summary"]["catalogue_sha256"],
                        "study_elapsed_seconds": arms[name]["summary"]["elapsed_seconds"],
                        "study_peak_commit_bytes": arms[name]["job"].get("peak_job_commit_bytes"),
                        "sweep_peak_commit_bytes": arms[name]["sweep_job"].get("peak_job_commit_bytes"),
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
        "offset": {
            "question": ("The diagnostic measured every arm's decision boundary above its label row. "
                         "Does moving the sample by the declared one row in that direction correct "
                         "the boundary and pay for it?"),
            "declared_direction": "up",
            "declared_from": "the diagnostic's positive mean row error",
            "reading": reading,
            "reading_text": reading_text,
            "boundary_shift_toward_labels_cells": moved,
            "thresholds": {"attribution_floor": ATTRIBUTION_DELTA,
                           "row_shift_floor_cells": ROW_SHIFT_FLOOR,
                           "auc_interval_rule": "clear of zero"}},
        "row_error": {
            "question": ("Where does each arm's top-most decided row sit relative to the top-most "
                         "occupied row, in rows, bottom-first?"),
            "arms": row_error,
            "published_diagnostic": str(DIAGNOSTIC.relative_to(ROOT)),
            "reading": ("Positive means the decision claims a taller stack than the label. The two "
                        "arms shared with the previous block reproduce its published means exactly."),
        },
        "verdict": verdict,
        "population": {"split": "training", "images": DECIDING_IMAGES,
                       "nonempty_boards": nonempty,
                       "occupied_cells": arms["up"]["result"]["deciding"]["occupied_cells"],
                       "empty_cells": arms["up"]["result"]["deciding"]["empty_cells_on_nonempty_boards"],
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
    print(json.dumps({key: scope[key] for key in ("verdict", "offset", "reproduction")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
