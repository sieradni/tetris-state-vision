"""Curate the declared box-extent lever against its matched-budget baseline.

`docs/M3_OCCUPANCY_EXTENT.md` fixes the rule: one arm adds a relative log-scale term on the final
lattice (weight 1.0), the other keeps the published objective. Both run 256 continuation steps from the
same published checkpoint under one catalogue. Occupancy is scored on the **full** deciding population;
the extent log-ratios and their per-axis reduction are the declared extent quantities. Every published
number is re-derived from the probes' own cells, and each probe must reproduce its own sweep before a
verdict. The matched start goes through the one shared contract (V-071), which adds no parameter here.
"""
import argparse
import json
from pathlib import Path

import torch

from cluster_interval import cluster_bootstrap
from matched_start import matched_start, step_zero_identical
from occupancy_sweep import AUC_THRESHOLD, F1_THRESHOLD, sweep_verdict
from percell_gate import CELL_F1_THRESHOLD
from placement_error import (ATTRIBUTION_FLOOR, EXTENT_REDUCTION, extent_decomposition,
                             extent_verdict, summarize)
from probe_model import require_bounded_job
from record_representation_lever import ancestry_of, cells_of, f1_at_threshold, read

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 1152*1024**2
SWEEP_CEILING_BYTES = 640*1024**2
PROBE_CEILING_BYTES = 640*1024**2
PROBE_WALL_SECONDS = 300
STEPS = 256
DECIDING_IMAGES = 36
NONEMPTY_BOARDS = 28
OCCUPIED_CELLS = 667
DECIDING_CELLS = NONEMPTY_BOARDS*200
FORWARDS = DECIDING_IMAGES+18
PROBE_FORWARDS = 36
BOOTSTRAP_DRAWS = 400
BOOTSTRAP_SEED = 20261026
SOURCE_RUN = "m3-percell-balance-r1"
PREVIOUS = {"geometry": ROOT/"docs/evidence/m3-occupancy-geometry-r1/scope.json",
            "origin": ROOT/"docs/evidence/m3-occupancy-translation-r1/scope.json"}
FAMILY = "m3-occextent"
INSTRUMENT = "m3-occupancy-extent-r1"
ARMS = {"lever": 1.0, "baseline": 0.0}


def arm_matches(summary, job, box_extent_weight, *, steps=STEPS, ceiling=CEILING_BYTES):
    """The admission join: one declaration of the box-extent lever, bound by a test."""
    if job.get("status") != "passed" or job.get("exit_code") != 0:
        raise ValueError("study did not pass inside its declared envelope")
    if job.get("work_items") != steps or job.get("memory_bytes") != ceiling:
        raise ValueError("study receipt does not certify the declared work and ceiling")
    if (summary.get("status") != "complete" or summary.get("steps") != steps or
            summary.get("requested_steps") != steps or not summary.get("checkpoint_round_trip_exact")):
        raise ValueError("study is not a complete run at the declared budget")
    arguments = summary.get("arguments", {})
    contract = summary.get("recovery_contract", {})
    if (arguments.get("box_extent_weight") != box_extent_weight or
            contract.get("box_extent_weight") != box_extent_weight):
        raise ValueError("study did not run the declared box-extent lever")
    if arguments.get("lattice_support") != 0 or contract.get("lattice_support") != 0:
        raise ValueError("this lever is declared on the published box head")
    if arguments.get("geometry_final_only") != 0 or contract.get("geometry_final_only") != 0:
        raise ValueError("this lever is declared on the published geometry stage objective")
    if (arguments.get("occupancy_detail") != 1 or arguments.get("occupancy_detail_stencil") != 0 or
            arguments.get("occupancy_context") != 0 or arguments.get("steps") != steps or
            arguments.get("seed") != 20261026 or arguments.get("capture_max_side") != 512 or
            arguments.get("occupancy_factor") != 4 or arguments.get("occupancy_positive_weight") != 4):
        raise ValueError("study did not run the declared read-out configuration")
    return True


def require_arm(name):
    study = f"{FAMILY}-{name}-r1"
    summary = read(ROOT/"runs"/study/"summary.json")
    job = read(ROOT/"runs"/(study+"-job.json"))
    arm_matches(summary, job, ARMS[name])
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


def require_probe(name):
    study = f"{FAMILY}-{name}-r1"
    probe = f"{study}-probe-r1"
    result = read(ROOT/"runs"/probe/"result.json")
    job = read(ROOT/"runs"/(probe+"-job.json"))
    summary = read(ROOT/"runs"/study/"summary.json")
    if (job.get("status") != "passed" or job.get("exit_code") != 0 or
            job.get("work_items") != PROBE_FORWARDS or
            job.get("memory_bytes") != PROBE_CEILING_BYTES or
            job.get("wall_seconds") != float(PROBE_WALL_SECONDS)):
        raise ValueError(f"{name} probe did not pass as the declared 36-forward 640 MiB / 300 s job")
    if (result.get("status") != "passed" or result.get("forwards") != PROBE_FORWARDS or
            result.get("run") != study or result.get("threshold") != 0.0 or
            len(result.get("cases", [])) != PROBE_FORWARDS):
        raise ValueError(f"{name} probe is not a complete 36-board deciding-population measurement")
    if result.get("weights_sha256") != summary.get("checkpoint_sha256"):
        raise ValueError(f"{name} probe measured different weights from its study")
    if result.get("geometry_final_only") != 0:
        raise ValueError(f"{name} probe did not score the published geometry stage objective")
    population = result.get("population") or {}
    if (population.get("nonempty_boards"), population.get("clusters"),
            population.get("occupied_cells")) != (NONEMPTY_BOARDS, 12, OCCUPIED_CELLS):
        raise ValueError(f"{name} probe population differs from the predeclaration")
    return probe, result, job


def paired_bootstrap(lever_cases, baseline_cases):
    """Whole-ancestry-cluster resampling of best swept F1 on the same sampled clusters."""
    grouped = {}
    for name, rows in (("lever", lever_cases), ("baseline", baseline_cases)):
        for row in rows:
            grouped.setdefault(ancestry_of(row["sample_id"]), {}).setdefault(name, []).append(row)
    if len(grouped) < 2:
        raise ValueError("at least two ancestry clusters required")
    for cluster, arms in grouped.items():
        if set(arms) != set(ARMS):
            raise ValueError("both arms must cover every ancestry cluster")
    ordered = tuple(sorted(grouped.items()))

    def statistic(sample):
        values = {}
        for name in ("lever", "baseline"):
            rows = [row for entry in sample for row in entry[name]]
            positive = [value for row in rows for value, label in cells_of([row]) if label]
            negative = [value for row in rows for value, label in cells_of([row]) if not label]
            if not positive or not negative:
                return None
            values["f1_"+name] = f1_at_threshold(rows)
        values["f1_delta"] = values["f1_lever"]-values["f1_baseline"]
        return values

    return cluster_bootstrap(statistic, ordered, ("f1_lever", "f1_baseline", "f1_delta"),
                             draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)


def reconciliation():
    """What the two prior levers bought and cost, read from their own curated receipts."""
    if not all(path.exists() for path in PREVIOUS.values()):
        return {"available": False}
    geometry, origin = read(PREVIOUS["geometry"]), read(PREVIOUS["origin"])
    origin_baseline = origin["arms"]["baseline"]["placement"]["mean_abs_offset_cells"]
    origin_lever = origin["arms"]["lever"]["placement"]["mean_abs_offset_cells"]
    return {"available": True, "note": "reference only, never verdict input",
            "geometry": {"instrument": geometry["instrument"],
                         "occupancy_best_f1_delta": geometry["attribution"]["best_f1_delta"],
                         "occupancy_interval": geometry["attribution"]["cluster_bootstrap"]["f1_delta"],
                         "placement_improvement": geometry["attribution"]["placement_improvement"],
                         "extent_ratio_u": geometry["arms"]["lever"]["placement"]["mean_scale_ratio_u"],
                         "extent_ratio_v": geometry["arms"]["lever"]["placement"]["mean_scale_ratio_v"]},
            "origin": {"instrument": origin["instrument"],
                       "occupancy_best_f1_delta": origin["attribution"]["best_f1_delta"],
                       "placement_improvement": (origin_baseline-origin_lever)/origin_baseline,
                       "extent_ratio_u": origin["arms"]["lever"]["placement"]["mean_scale_ratio_u"],
                       "extent_ratio_v": origin["arms"]["lever"]["placement"]["mean_scale_ratio_v"]}}


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
        probe, probe_result, probe_job = require_probe(name)
        arms[name] = {"study": study, "summary": summary, "job": job, "sweep": sweep,
                      "result": result, "sweep_job": sweep_job, "probe": probe,
                      "probe_result": probe_result, "probe_job": probe_job}
    if arms["lever"]["summary"]["catalogue_sha256"] != arms["baseline"]["summary"]["catalogue_sha256"]:
        raise ValueError("the arms were captured under different catalogues")
    mappable = {"map_location": "cpu", "weights_only": True, "mmap": True}
    # This lever changes only the objective and adds no parameter, so the shared contract's
    # no-addition branch applies: the two saved states must be bitwise identical.
    matched_start(
        torch.load(ROOT/"runs"/arms["lever"]["study"]/"initial-weights.pt", **mappable),
        torch.load(ROOT/"runs"/arms["baseline"]["study"]/"initial-weights.pt", **mappable))
    step_zero_identical(arms["lever"]["summary"], arms["baseline"]["summary"])

    measured = {}
    for name in ARMS:
        probe_result = arms[name]["probe_result"]
        derived = summarize(probe_result["cases"])
        if derived != probe_result["measured"]:
            raise ValueError(f"{name} probe summary does not re-derive from its own cells")
        occupancy = derived["occupancy"]
        swept = arms[name]["result"]["deciding"]
        index = swept["thresholds"].index(0.0)
        if abs(occupancy["f1"]-swept["rows"][index]["f1"]) > 1e-6:
            raise ValueError(f"{name} probe does not reproduce its sweep's per-cell F1")
        if occupancy["known_cells"] != DECIDING_CELLS or occupancy["occupied_cells"] != OCCUPIED_CELLS:
            raise ValueError(f"{name} probe did not score the full 28-board deciding population")
        if any(occupancy["counts"][key] != swept["rows"][index][key]
               for key in ("true_positive", "false_positive", "false_negative")):
            raise ValueError(f"{name} probe does not reproduce its sweep's operating counts")
        measured[name] = {"occupancy": {key: occupancy[key] for key in
                                        ("images", "known_cells", "occupied_cells", "empty_cells",
                                         "auc", "f1", "counts")},
                          "placement": {key: derived["placement"][key] for key in
                                        ("images", "cells", "mean_abs_offset_cells",
                                         "mean_board_error_cells", "signed_column_offset_cells",
                                         "signed_row_offset_cells", "mean_scale_ratio_u",
                                         "mean_scale_ratio_v", "board_error_spread_cells",
                                         "shifted_share", "mapped_share")},
                          "best": {"threshold": swept["best"]["threshold"], "f1": swept["best"]["f1"],
                                   "precision": swept["best"]["precision"],
                                   "recall": swept["best"]["recall"]},
                          "verdict": arms[name]["result"]["verdict"]}

    delta_best = measured["lever"]["best"]["f1"]-measured["baseline"]["best"]["f1"]
    bootstrap = paired_bootstrap(
        [row for row in arms["lever"]["result"]["cases"] if row["split"] == "training"],
        [row for row in arms["baseline"]["result"]["cases"] if row["split"] == "training"])
    if bootstrap["clusters"] != 12:
        raise ValueError("the paired bootstrap must resample the twelve ancestry clusters")
    decision = extent_verdict(measured["lever"]["occupancy"], measured["lever"]["verdict"], delta_best,
                              measured["baseline"]["placement"], measured["lever"]["placement"])
    if decision["verdict"] == "not-cleared":
        next_step = extent_decomposition(measured["baseline"]["placement"],
                                         measured["lever"]["placement"], decision)
    else:
        next_step = {"rule": "cleared", "next_suspect": None,
                     "statement": "the declared box-extent lever cleared every condition"}

    scope = {
        "status": "passed",
        "instrument": INSTRUMENT,
        "predeclaration": "docs/M3_OCCUPANCY_EXTENT.md",
        "scope": ("Matched 256-step continuation of one published checkpoint. The only difference is "
                  "the declared box-extent lever: a relative log-scale term on the final lattice, so "
                  "the four box parameters' extent is supervised directly while the origin and the "
                  "occupancy read-out are untouched. Scored read-only on the full widened deciding "
                  "population (28 nonempty boards, 200 cells each) with the standing grid plus the "
                  "extent log-ratios; no threshold, population, ceiling, label or prior-evidence "
                  "change, and no real-image, complete-state or calibrated claim."),
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "envelope": {"steps": STEPS, "ceiling_bytes": CEILING_BYTES,
                     "sweep_ceiling_bytes": SWEEP_CEILING_BYTES,
                     "probe_ceiling_bytes": PROBE_CEILING_BYTES,
                     "work_items": {"profile": 2, "study": STEPS, "sweep_profile": 2,
                                    "sweep": FORWARDS, "probe": PROBE_FORWARDS, "curation": 8}},
        "source": {"run": SOURCE_RUN, "weights_sha256": source["checkpoint_sha256"],
                   "steps": source["steps"]},
        "matched": {"seed": 20261026, "learning_rate": .001, "capture_max_side": 512,
                    "curriculum": "stream-v1", "config_difference": [],
                    "objective_difference": ["box_extent_weight"],
                    "shared": {"geometry_final_only": 0, "lattice_support": 0,
                               "occupancy_detail": 1, "occupancy_detail_stencil": 0,
                               "occupancy_context": 0, "occupancy_factor": 4,
                               "occupancy_positive_weight": 4},
                    "catalogue_sha256": arms["lever"]["summary"]["catalogue_sha256"],
                    "initial_weights": arms["lever"]["summary"]["initial_weights_from"],
                    "initial_weights_sha256": arms["lever"]["summary"]["initial_weights_sha256"]},
        "arms": {name: {"study": arms[name]["study"], "sweep": arms[name]["sweep"],
                        "probe": arms[name]["probe"], "box_extent_weight": ARMS[name],
                        "weights_sha256": arms[name]["result"]["weights_sha256"],
                        "study_elapsed_seconds": arms[name]["summary"]["elapsed_seconds"],
                        "study_peak_commit_bytes": arms[name]["job"].get("peak_job_commit_bytes"),
                        "probe_peak_commit_bytes": arms[name]["probe_job"].get("peak_job_commit_bytes"),
                        **measured[name]} for name in ARMS},
        "population": {"split": "training", "images": DECIDING_IMAGES,
                       "nonempty_boards": NONEMPTY_BOARDS, "known_cells": DECIDING_CELLS,
                       "occupied_cells": OCCUPIED_CELLS, "clusters": 12,
                       "note": ("Full deciding population; the oracle pairing's co-scoreable subset is "
                                "never the judgement population.")},
        "gates": {"per_cell_f1": CELL_F1_THRESHOLD, "ranking_auc": AUC_THRESHOLD,
                  "ranking_f1": F1_THRESHOLD, "attribution_floor": ATTRIBUTION_FLOOR,
                  "extent_reduction_floor": EXTENT_REDUCTION,
                  "grid_points": 81, "decision_threshold": 0.0},
        "attribution": {"best_f1_delta": round(delta_best, 6),
                        "auc_delta": round(measured["lever"]["occupancy"]["auc"]
                                           - measured["baseline"]["occupancy"]["auc"], 6),
                        "per_cell_f1_delta": round(measured["lever"]["occupancy"]["f1"]
                                                   - measured["baseline"]["occupancy"]["f1"], 6),
                        "extent_reduction": decision["extent_reduction"],
                        "cluster_bootstrap": bootstrap},
        "verdict": decision,
        "next_step": next_step,
        "reconciliation": reconciliation(),
        "unchanged": {"thresholds": {"per_cell_f1": CELL_F1_THRESHOLD, "ranking_auc": AUC_THRESHOLD,
                                     "ranking_f1": F1_THRESHOLD},
                      "population": "full widened deciding population, twelve groups",
                      "source_checkpoint": source["checkpoint_sha256"],
                      "prior_evidence": "not touched", "frozen_manifest": "not touched"},
    }
    args.output.mkdir(parents=True)
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(scope, stream, indent=2, allow_nan=False)
    print(json.dumps({key: scope[key] for key in ("verdict", "attribution", "next_step")}, indent=2))


if __name__ == "__main__":
    main()
