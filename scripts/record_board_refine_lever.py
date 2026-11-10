"""Curate the declared per-board lattice refinement capacity lever against its matched baseline.

`docs/M3_OCCUPANCY_BOARD_REFINEMENT.md` fixes the rule: one arm adds a board-conditioned four-vector that
corrects the decoded lattice in cell space, applied identically at every stage (12,612 declared
zero-initialized parameters), the other keeps the published shared decode. Both run 256 continuation
steps from the same published checkpoint under one catalogue. Occupancy is scored on the **full**
deciding population. This pass also carries a declared **activation condition**: the probe's measured
applied correction must reach 0.10 cells and 2% axis scale, or the verdict records `not-engaged` and no
claim about the channel is allowed.
"""
import argparse
import json
import subprocess
from pathlib import Path

import torch

from cluster_interval import cluster_bootstrap
from matched_start import matched_start, step_zero_identical
from occupancy_sweep import AUC_THRESHOLD, F1_THRESHOLD, sweep_verdict
from percell_gate import CELL_F1_THRESHOLD
from placement_error import (ATTRIBUTION_FLOOR, BOARD_REFINE_ACTIVATION_FLOOR, BOARD_REFINE_SCALE_FLOOR,
                             PLACEMENT_IMPROVEMENT, board_refinement_decomposition, placement_reduction,
                             summarize, verdict)
from probe_model import require_bounded_job
from record_representation_lever import ancestry_of, cells_of, f1_at_threshold, read

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 1152*1024**2
SWEEP_CEILING_BYTES = 640*1024**2
PROBE_CEILING_BYTES = 640*1024**2
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
PREDECLARATION = "docs/M3_OCCUPANCY_BOARD_REFINEMENT.md"
PREVIOUS = {"geometry": ROOT/"docs/evidence/m3-occupancy-geometry-r1/scope.json",
            "origin": ROOT/"docs/evidence/m3-occupancy-translation-r1/scope.json",
            "extent": ROOT/"docs/evidence/m3-occupancy-extent-r1/scope.json",
            "capacity": ROOT/"docs/evidence/m3-occupancy-geometry-capacity-r1/scope.json",
            "residual": ROOT/"docs/evidence/m3-occupancy-lattice-residual-r1/scope.json",
            "readout": ROOT/"docs/evidence/m3-occupancy-lattice-free-readout-r1/scope.json"}
FAMILY = "m3-occbref"
INSTRUMENT = "m3-occupancy-board-refinement-r1"
ARMS = {"lever": 1, "baseline": 0}
# The tensors this lever adds: a two-layer board-conditioned refinement head, all exactly zero at
# initialization. Declared once and bound to the model's own state-dict difference by a test.
ADDED_TENSORS = frozenset(("lattice_board_refine.0.weight", "lattice_board_refine.0.bias",
                           "lattice_board_refine.2.weight", "lattice_board_refine.2.bias"))
ADDED_PARAMETERS = 192*64+64+64*4+4
CURATED_CONTROL = {"f1": 0.4146485528647372, "auc": 0.8255386800822172,
                   "best_f1": 0.4237196765498652, "best_threshold": -0.3,
                   "mean_abs_offset_cells": 1.9812198625,
                   "mean_scale_ratio_u": 1.2256880473012681,
                   "mean_scale_ratio_v": 1.18035288054444,
                   "board_error_spread_cells": 4.604344325}
CONTROL_TOLERANCE = 1e-6


def activation_engaged(activation):
    """The declared activation condition, implemented once and bound by a test."""
    if not isinstance(activation, dict) or not activation.get("available"):
        raise ValueError("the lever probe carries no activation diagnostic")
    return (activation["mean_abs_origin_cells"] >= BOARD_REFINE_ACTIVATION_FLOOR and
            activation["mean_abs_log_scale"] >= BOARD_REFINE_SCALE_FLOOR)


def arm_matches(summary, job, lattice_board_refine, *, steps=STEPS, ceiling=CEILING_BYTES):
    """The admission join: one declaration of the board-refinement lever, bound by a test."""
    if job.get("status") != "passed" or job.get("exit_code") != 0:
        raise ValueError("study did not pass inside its declared envelope")
    if job.get("work_items") != steps or job.get("memory_bytes") != ceiling:
        raise ValueError("study receipt does not certify the declared work and ceiling")
    if (summary.get("status") != "complete" or summary.get("steps") != steps or
            summary.get("requested_steps") != steps or not summary.get("checkpoint_round_trip_exact")):
        raise ValueError("study is not a complete run at the declared budget")
    arguments = summary.get("arguments", {})
    contract = summary.get("recovery_contract", {})
    if (arguments.get("lattice_board_refine") != lattice_board_refine or
            contract.get("lattice_board_refine") != lattice_board_refine):
        raise ValueError("study did not run the declared board-refinement lever")
    for key in ("lattice_residual", "lattice_support", "geometry_final_only", "occupancy_lattice_free"):
        if arguments.get(key, 0) != 0 or contract.get(key, 0) != 0:
            raise ValueError(f"the board-refinement lever is declared on the published {key}")
    for key in ("support_anchor_weight", "box_extent_weight"):
        if arguments.get(key, 0.0) not in (0, 0.0) or contract.get(key, 0.0) not in (0, 0.0):
            raise ValueError(f"the board-refinement lever is declared on the published {key}")
    if (arguments.get("occupancy_detail") != 1 or arguments.get("occupancy_detail_stencil") != 0 or
            arguments.get("occupancy_context") != 0 or arguments.get("steps") != steps or
            arguments.get("seed") != 20261026 or arguments.get("capture_max_side") != 512 or
            arguments.get("occupancy_factor") != 4 or arguments.get("occupancy_positive_weight") != 4):
        raise ValueError("study did not run the declared shared configuration")
    return True


def receipts_after(commit_time, paths):
    """A receipt must postdate the predeclaration commit; time is the only proof an untracked file has."""
    for path in paths:
        if not Path(path).exists():
            raise ValueError(f"declared receipt {path} is missing")
        if Path(path).stat().st_mtime < commit_time:
            raise ValueError(f"declared receipt {path} predates the predeclaration commit")
    return True


def predeclaration_commit():
    """Resolve the commit that first added the predeclaration and prove it is an ancestor of HEAD."""
    try:
        added = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H", "--", PREDECLARATION],
                               cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        if not added:
            raise ValueError("the predeclaration has no commit")
        commit = added[-1]
        if subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT).returncode:
            raise ValueError("the predeclaration commit is not an ancestor of HEAD")
        stamp = int(subprocess.run(["git", "show", "-s", "--format=%ct", commit],
                                   cwd=ROOT, capture_output=True, text=True, check=True).stdout)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot prove the predeclaration ancestry: {error}") from error
    return commit, stamp


def require_initial_state(lever_state, baseline_state):
    """The parameter-addition branch of the shared guard, with the declared added tensors only."""
    return matched_start(lever_state, baseline_state, added=ADDED_TENSORS)


def require_arm(name):
    study = f"{FAMILY}-{name}-r1"
    summary = read(ROOT/"runs"/study/"summary.json")
    job = read(ROOT/"runs"/(study+"-job.json"))
    arm_matches(summary, job, ARMS[name])
    config = summary.get("config") or {}
    if config.get("lattice_board_refine") != ARMS[name]:
        raise ValueError(f"{name} study recorded a different lattice configuration")
    origin = summary.get("initial_weights_from") or {}
    if not str(origin.get("run", "")).endswith(SOURCE_RUN) or not origin.get("weights_sha256"):
        raise ValueError(f"{name} study was not initialized from the published checkpoint")
    return study, summary, job, origin


def require_sweep(name):
    study = f"{FAMILY}-{name}-r1"
    sweep = f"{study}-sweep-r1"
    result = read(ROOT/"runs"/(sweep+"/result.json"))
    job = read(ROOT/"runs"/(sweep+"-job.json"))
    summary = read(ROOT/"runs"/study/"summary.json")
    admission = result.get("admission") or {}
    declared_profile = str(admission.get("run", ""))
    if not declared_profile.startswith(sweep+"-profile"):
        raise ValueError(f"{name} sweep was not admitted by its own profile")
    profile_dir = ROOT/"runs"/declared_profile
    profile = read(profile_dir/"summary.json")
    profile_job = read(profile_dir.parent/(profile_dir.name+"-job.json"))
    if (job.get("status") != "passed" or job.get("exit_code") != 0 or
            job.get("memory_bytes") != SWEEP_CEILING_BYTES or
            job.get("work_items") != result.get("forwards")):
        raise ValueError(f"{name} sweep did not pass under the declared 640 MiB work items")
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
    if (profile_job.get("status") != "passed" or profile_job.get("exit_code") != 0 or
            profile_job.get("work_items") != 2 or
            profile_job.get("memory_bytes") != SWEEP_CEILING_BYTES or
            profile.get("status") != "complete" or
            profile.get("weights_sha256") != result.get("weights_sha256") or
            admission.get("weights_sha256") != result.get("weights_sha256") or
            admission.get("ceiling_bytes") != SWEEP_CEILING_BYTES):
        raise ValueError(f"{name} sweep admission profile differs")
    return sweep, result, job


def require_probe(name):
    study = f"{FAMILY}-{name}-r1"
    probe = f"{study}-probe-r1"
    result = read(ROOT/"runs"/(probe+"/result.json"))
    job = read(ROOT/"runs"/(probe+"-job.json"))
    summary = read(ROOT/"runs"/study/"summary.json")
    if (job.get("status") != "passed" or job.get("exit_code") != 0 or
            job.get("work_items") != PROBE_FORWARDS or
            job.get("memory_bytes") != PROBE_CEILING_BYTES or
            job.get("wall_seconds") != 300.0):
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


def require_control(placement, occupancy, best):
    """The baseline must be the curated control; drift is a defect, never a lever result."""
    observed = {"f1": occupancy["f1"], "auc": occupancy["auc"], "best_f1": best["f1"],
                "best_threshold": best["threshold"],
                "mean_abs_offset_cells": placement["mean_abs_offset_cells"],
                "mean_scale_ratio_u": placement["mean_scale_ratio_u"],
                "mean_scale_ratio_v": placement["mean_scale_ratio_v"],
                "board_error_spread_cells": placement["board_error_spread_cells"]}
    drift = {key: (value, CURATED_CONTROL[key]) for key, value in observed.items()
             if abs(value-CURATED_CONTROL[key]) > CONTROL_TOLERANCE}
    if drift:
        raise ValueError(f"the baseline drifted from the curated control: {drift}")
    return True


def reconciliation():
    """What the six closed levers bought and cost, read from their own curated receipts."""
    if not all(path.exists() for path in PREVIOUS.values()):
        return {"available": False}
    records = {name: read(path) for name, path in PREVIOUS.items()}

    def improvement(record):
        baseline = record["arms"]["baseline"]["placement"]["mean_abs_offset_cells"]
        lever = record["arms"]["lever"]["placement"]["mean_abs_offset_cells"]
        return (baseline-lever)/baseline

    return {"available": True, "note": "reference only, never verdict input",
            "geometry": {"instrument": records["geometry"]["instrument"],
                         "occupancy_best_f1_delta": records["geometry"]["attribution"]["best_f1_delta"],
                         "placement_improvement": records["geometry"]["attribution"]["placement_improvement"]},
            "origin": {"instrument": records["origin"]["instrument"],
                       "occupancy_best_f1_delta": records["origin"]["attribution"]["best_f1_delta"],
                       "placement_improvement": improvement(records["origin"])},
            "extent": {"instrument": records["extent"]["instrument"],
                       "occupancy_best_f1_delta": records["extent"]["attribution"]["best_f1_delta"],
                       "placement_improvement": improvement(records["extent"])},
            "capacity": {"instrument": records["capacity"]["instrument"],
                         "occupancy_best_f1_delta": records["capacity"]["attribution"]["best_f1_delta"],
                         "placement_improvement": records["capacity"]["attribution"]["placement_improvement"]},
            "residual": {"instrument": records["residual"]["instrument"],
                         "occupancy_best_f1_delta": records["residual"]["attribution"]["best_f1_delta"],
                         "placement_improvement": records["residual"]["attribution"]["placement_improvement"]},
            "readout": {"instrument": records["readout"]["instrument"],
                        "occupancy_best_f1_delta": records["readout"]["attribution"]["best_f1_delta"],
                        "placement_improvement": records["readout"]["attribution"]["placement_improvement"],
                        "activation": records["readout"]["readout"]["gate_reading"]}}


def main():
    bounded = require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("evidence directory already exists")
    commit, stamp = predeclaration_commit()

    source = read(ROOT/"runs"/SOURCE_RUN/"summary.json")
    arms, receipts = {}, []
    for name in ARMS:
        study, summary, job, origin = require_arm(name)
        if origin["weights_sha256"] != source["checkpoint_sha256"]:
            raise ValueError(f"{name} study did not start from the published checkpoint bytes")
        sweep, result, sweep_job = require_sweep(name)
        probe, probe_result, probe_job = require_probe(name)
        receipts += [ROOT/"runs"/study/"summary.json", ROOT/"runs"/(study+"-job.json"),
                     ROOT/"runs"/(sweep+"/result.json"), ROOT/"runs"/(sweep+"-job.json"),
                     ROOT/"runs"/study/"summary.json",
                     ROOT/"runs"/(sweep+"-profile-r1/summary.json"),
                     ROOT/"runs"/(sweep+"-profile-r1-job.json"),
                     ROOT/"runs"/(probe+"/result.json"), ROOT/"runs"/(probe+"-job.json")]
        arms[name] = {"study": study, "summary": summary, "job": job, "sweep": sweep,
                      "result": result, "sweep_job": sweep_job, "probe": probe,
                      "probe_result": probe_result, "probe_job": probe_job}
    receipts_after(stamp, receipts)
    if arms["lever"]["summary"]["catalogue_sha256"] != arms["baseline"]["summary"]["catalogue_sha256"]:
        raise ValueError("the arms were captured under different catalogues")
    mappable = {"map_location": "cpu", "weights_only": True, "mmap": True}
    lever_state = torch.load(ROOT/"runs"/arms["lever"]["study"]/"initial-weights.pt", **mappable)
    baseline_state = torch.load(ROOT/"runs"/arms["baseline"]["study"]/"initial-weights.pt", **mappable)
    require_initial_state(lever_state, baseline_state)
    if set(lever_state)-set(baseline_state) != set(ADDED_TENSORS):
        raise ValueError("the lever's added tensors are not the declared set")
    step_zero_identical(arms["lever"]["summary"], arms["baseline"]["summary"])

    trained = torch.load(ROOT/"runs"/arms["lever"]["study"]/"checkpoint"/"weights.pt", **mappable)
    if not all(name in trained for name in ADDED_TENSORS):
        raise ValueError("the trained checkpoint does not carry the declared tensors")
    activation = arms["lever"]["probe_result"].get("activation") or {}
    engaged = activation_engaged(activation)

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
    require_control(measured["baseline"]["placement"], measured["baseline"]["occupancy"],
                    measured["baseline"]["best"])
    if arms["baseline"]["probe_result"].get("activation", {}).get("available"):
        raise ValueError("the baseline must not carry the declared refinement channel")

    delta_best = measured["lever"]["best"]["f1"]-measured["baseline"]["best"]["f1"]
    bootstrap = paired_bootstrap(
        [row for row in arms["lever"]["result"]["cases"] if row["split"] == "training"],
        [row for row in arms["baseline"]["result"]["cases"] if row["split"] == "training"])
    if bootstrap["clusters"] != 12:
        raise ValueError("the paired bootstrap must resample the twelve ancestry clusters")
    improvement = placement_reduction(measured["baseline"]["placement"], measured["lever"]["placement"])
    decision = verdict(measured["lever"]["occupancy"], measured["lever"]["verdict"], delta_best,
                       improvement)
    decision["placement_improvement"] = improvement
    decision["conditions"]["activation"] = engaged
    decision["failing"] = [name for name, met in decision["conditions"].items() if not met]
    decision["verdict"] = "cleared" if all(decision["conditions"].values()) else "not-cleared"
    decision["activation"] = activation
    decision["activation_floor"] = {"origin_cells": BOARD_REFINE_ACTIVATION_FLOOR,
                                    "log_scale": BOARD_REFINE_SCALE_FLOOR}
    if decision["conditions"]["activation"]:
        next_step = board_refinement_decomposition(measured["baseline"]["placement"],
                                                   measured["lever"]["placement"], decision)
    else:
        next_step = board_refinement_decomposition(measured["baseline"]["placement"],
                                                   measured["lever"]["placement"], decision)

    scope = {
        "status": "passed",
        "instrument": INSTRUMENT,
        "predeclaration": PREDECLARATION,
        "predeclaration_commit": commit,
        "scope": ("Matched 256-step continuation of one published checkpoint. The only difference is "
                  "the declared capacity lever: one board-conditioned four-vector that corrects the "
                  "decoded lattice in cell space (origin offset plus log-scale per axis), applied "
                  "identically at every stage. No new input, box head, stage policy, objective term or "
                  "read-out. Scored read-only on the full widened deciding population (28 nonempty "
                  "boards, 200 cells each) with the standing grid and placement measurement, plus the "
                  "declared activation diagnostic; no real-image, complete-state or calibrated claim."),
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
                    "curriculum": "stream-v1", "config_difference": ["lattice_board_refine"],
                    "objective_difference": [],
                    "shared": {"lattice_residual": 0, "geometry_final_only": 0, "lattice_support": 0,
                               "occupancy_lattice_free": 0, "box_extent_weight": 0.0,
                               "support_anchor_weight": 0.0, "occupancy_detail": 1,
                               "occupancy_detail_stencil": 0, "occupancy_context": 0,
                               "occupancy_factor": 4, "occupancy_positive_weight": 4},
                    "catalogue_sha256": arms["lever"]["summary"]["catalogue_sha256"],
                    "initial_weights": arms["lever"]["summary"]["initial_weights_from"],
                    "initial_weights_sha256": arms["lever"]["summary"]["initial_weights_sha256"],
                    "added_tensors": sorted(ADDED_TENSORS),
                    "added_parameters": ADDED_PARAMETERS,
                    "added_initialization": "exactly zero in the declared tensor set",
                    "baseline_control": CURATED_CONTROL,
                    "control_tolerance": CONTROL_TOLERANCE},
        "refinement": {"form": "0.25*tanh(raw) in decoded cell space: origin offset in cells along the "
                               "decoded axes plus a log-scale per axis",
                       "conditioning": "the board token only; one four-vector per board image",
                       "stage_policy": "the same vector at every decoder stage (board-conditioned, not "
                                       "per-stage)",
                       "supervision": "the published geometry term, unchanged",
                       "parameters": ADDED_PARAMETERS,
                       "activation": activation,
                       "activation_floor": {"origin_cells": BOARD_REFINE_ACTIVATION_FLOOR,
                                            "log_scale": BOARD_REFINE_SCALE_FLOOR},
                       "engaged": engaged},
        "arms": {name: {"study": arms[name]["study"], "sweep": arms[name]["sweep"],
                        "probe": arms[name]["probe"], "lattice_board_refine": ARMS[name],
                        "activation": arms[name]["probe_result"].get("activation"),
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
                  "placement_improvement_floor": PLACEMENT_IMPROVEMENT,
                  "activation_origin_cells": BOARD_REFINE_ACTIVATION_FLOOR,
                  "activation_log_scale": BOARD_REFINE_SCALE_FLOOR,
                  "placement_error": "mean |offset| in target cells over 200 cells x 2 axes",
                  "grid_points": 81, "decision_threshold": 0.0},
        "attribution": {"best_f1_delta": round(delta_best, 6),
                        "auc_delta": round(measured["lever"]["occupancy"]["auc"]
                                           - measured["baseline"]["occupancy"]["auc"], 6),
                        "per_cell_f1_delta": round(measured["lever"]["occupancy"]["f1"]
                                                   - measured["baseline"]["occupancy"]["f1"], 6),
                        "placement_improvement": improvement,
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
    print(json.dumps({key: scope[key] for key in ("verdict", "attribution", "next_step",
                                                  "refinement")}, indent=2))


if __name__ == "__main__":
    main()
