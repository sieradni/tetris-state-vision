"""Curate the declared 2x2 in-cell detail stencil against its matched single-sample control.

`docs/M3_OCCUPANCY_STENCIL.md` fixes the rule; this recorder applies it to the two admitted
continuations and their two widened sweeps. Both arms carry the declared detail tap and differ only
in its stencil form, so the contrast is the increment of the extra in-cell capacity. Every published
quantity is re-derived and the deciding arithmetic asserted before writing.
"""
import argparse
import json
from pathlib import Path

from occupancy_sweep import AUC_THRESHOLD, F1_THRESHOLD, sweep_verdict
from percell_gate import CELL_F1_THRESHOLD
from probe_model import require_bounded_job
from record_precision_lever import mechanism, paired_bootstrap, precision_of
from record_representation_lever import arm_summary, lever_verdict, read

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 1152*1024**2
SWEEP_CEILING_BYTES = 640*1024**2
STEPS = 64
DECIDING_IMAGES = 36
DEVELOPMENT_IMAGES = 18
FORWARDS = DECIDING_IMAGES+DEVELOPMENT_IMAGES
ATTRIBUTION_DELTA = 0.05
PRECISION_DELTA = 0.05
SOURCE_RUN = "m3-percell-balance-r1"
FAMILY = "m3-occdstencil"
INSTRUMENT = "m3-occupancy-stencil-r1"
REFERENCE = ROOT/"docs/evidence/m3-occupancy-precision-r1/scope.json"
ARMS = {"lever": 1, "control": 0}


def require_arm(name):
    stencil = ARMS[name]
    study = f"{FAMILY}-{name}-r1"
    summary = read(ROOT/"runs"/study/"summary.json")
    job = read(ROOT/"runs"/(study+"-job.json"))
    if job.get("status") != "passed" or job.get("exit_code") != 0:
        raise ValueError(f"{name} study did not pass inside its declared envelope")
    if (summary.get("status") != "complete" or summary.get("steps") != STEPS or
            summary.get("requested_steps") != STEPS or not summary.get("checkpoint_round_trip_exact") or
            job.get("work_items") != STEPS or job.get("memory_bytes") != CEILING_BYTES):
        raise ValueError(f"{name} study is not a complete 64-step run inside 1,152 MiB")
    arguments = summary["arguments"]
    if (arguments.get("occupancy_detail") != 1 or arguments.get("occupancy_detail_stencil") != stencil or
            arguments.get("occupancy_context") != 0 or arguments.get("steps") != STEPS or
            arguments.get("seed") != 20261026 or arguments.get("capture_max_side") != 512 or
            summary["recovery_contract"].get("occupancy_detail") != 1 or
            summary["recovery_contract"].get("occupancy_detail_stencil") != stencil):
        raise ValueError(f"{name} study did not run the declared stencil form at the declared budget")
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


def require_reference():
    """The prior single-sample arms: reported for comparison, never read by the verdict."""
    record = read(REFERENCE)
    if record.get("status") != "passed" or record.get("instrument") != "m3-occupancy-precision-r1":
        raise ValueError("the prior precision record is not the committed reference")
    arm = record["arms"]["lever"]
    return {"instrument": record["instrument"],
            "predeclaration": record["predeclaration"],
            "scope": record.get("scope"),
            "matched_to_this_pass": False,
            "catalogue_sha256": record["matched"]["catalogue_sha256"],
            "arm": {key: arm[key] for key in ("auc", "precision", "per_cell_f1", "best", "verdict")}}


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

    lever_cases = [row for row in arms["lever"]["result"]["cases"] if row["split"] == "training"]
    control_cases = [row for row in arms["control"]["result"]["cases"] if row["split"] == "training"]
    if len(lever_cases) != DECIDING_IMAGES or len(control_cases) != DECIDING_IMAGES:
        raise ValueError("the curation requires exactly the 36 deciding training images")
    if [row["sample_id"] for row in lever_cases] != [row["sample_id"] for row in control_cases]:
        raise ValueError("the two sweeps must share one deciding population")
    if [row["logits"] for row in lever_cases] == [row["logits"] for row in control_cases]:
        raise ValueError("the two arms produced identical logits; the stencil did not run")
    nonempty = sum(1 for row in lever_cases if any(row["occupied"]))
    contexts = len({"-".join(row["sample_id"].split("-")[:2]) for row in lever_cases})
    if (nonempty, contexts) != (28, 12):
        raise ValueError("deciding arithmetic differs from the predeclaration (28 boards, 12 clusters)")

    measured = {}
    for name in ARMS:
        block = arm_summary(arms[name]["result"]["deciding"])
        block["precision"] = round(precision_of(block["operating_counts"]), 6)
        measured[name] = block
    bootstrap = paired_bootstrap(lever_cases, control_cases)
    if bootstrap["clusters"] != 12:
        raise ValueError("the paired bootstrap must resample the twelve ancestry clusters")
    delta_best = measured["lever"]["best"]["f1"]-measured["control"]["best"]["f1"]
    delta_precision = measured["lever"]["precision"]-measured["control"]["precision"]
    verdict = lever_verdict(measured["lever"]["per_cell_f1"], measured["lever"]["verdict"], delta_best)
    reference = require_reference()

    scope = {
        "status": "passed",
        "instrument": INSTRUMENT,
        "predeclaration": "docs/M3_OCCUPANCY_STENCIL.md",
        "scope": ("Matched 64-step continuation of one published checkpoint, both arms carrying the "
                  "declared zero-initialized detail tap and differing only in its form: lever = the "
                  "2x2 in-cell stencil (four quarter-cell samples instead of one), control = the "
                  "single center sample. Scored read-only on the widened deciding population with "
                  "the standing gates and grid; no threshold, population, ceiling or gate changes, "
                  "and no real-image, complete-state, calibrated or quality claim."),
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "envelope": {"steps": STEPS, "ceiling_bytes": CEILING_BYTES,
                     "sweep_ceiling_bytes": SWEEP_CEILING_BYTES,
                     "work_items": {"profile": 2, "study": STEPS, "sweep_profile": 2,
                                    "sweep": FORWARDS, "curation": 4},
                     "attribution_floor": ATTRIBUTION_DELTA, "precision_floor": PRECISION_DELTA},
        "source": {"run": SOURCE_RUN, "weights_sha256": source["checkpoint_sha256"],
                   "steps": source["steps"]},
        "matched": {"seed": 20261026, "learning_rate": .001, "capture_max_side": 512,
                    "curriculum": "stream-v1",
                    "config_difference": ["occupancy_detail_stencil"],
                    "shared": {"occupancy_detail": 1, "occupancy_context": 0},
                    "catalogue_sha256": arms["lever"]["summary"]["catalogue_sha256"],
                    "initial_weights": arms["lever"]["summary"]["initial_weights_from"]},
        "arms": {name: {"study": arms[name]["study"], "sweep": arms[name]["sweep"],
                        "occupancy_detail_stencil": ARMS[name],
                        "weights_sha256": arms[name]["result"]["weights_sha256"],
                        "catalogue_sha256": arms[name]["summary"]["catalogue_sha256"],
                        "study_elapsed_seconds": arms[name]["summary"]["elapsed_seconds"],
                        "study_peak_commit_bytes": arms[name]["job"].get("peak_job_commit_bytes"),
                        "sweep_peak_commit_bytes": arms[name]["sweep_job"].get("peak_job_commit_bytes"),
                        **measured[name]} for name in ARMS},
        "gates": {"per_cell_f1": CELL_F1_THRESHOLD, "ranking_auc": AUC_THRESHOLD,
                  "ranking_f1": F1_THRESHOLD, "attribution_floor": ATTRIBUTION_DELTA,
                  "grid_points": 81},
        "attribution": {"best_f1_delta": round(delta_best, 6),
                        "auc_delta": round(measured["lever"]["auc"]-measured["control"]["auc"], 6),
                        "per_cell_f1_delta": round(measured["lever"]["per_cell_f1"]-measured["control"]["per_cell_f1"], 6),
                        "precision_delta": round(delta_precision, 6),
                        "recall_delta": round(
                            measured["lever"]["operating_intervals"]["recall"]["estimate"]
                            - measured["control"]["operating_intervals"]["recall"]["estimate"], 6),
                        "cluster_bootstrap": bootstrap},
        "mechanism": mechanism(delta_precision, bootstrap),
        "verdict": verdict,
        "reference": reference,
        "reference_comparison": {
            "note": ("Unmatched cross-pass comparison: the prior arms ran under another catalogue, "
                     "so nothing here enters the verdict."),
            "precision_vs_prior_single_tap": round(
                measured["lever"]["precision"]-reference["arm"]["precision"], 6),
            "per_cell_f1_vs_prior_single_tap": round(
                measured["lever"]["per_cell_f1"]-reference["arm"]["per_cell_f1"], 6),
            "auc_vs_prior_single_tap": round(measured["lever"]["auc"]-reference["arm"]["auc"], 6),
            "best_f1_vs_prior_single_tap": round(
                measured["lever"]["best"]["f1"]-reference["arm"]["best"]["f1"], 6)},
        "population": {"split": "training", "images": DECIDING_IMAGES,
                       "nonempty_boards": nonempty,
                       "occupied_cells": arms["lever"]["result"]["deciding"]["occupied_cells"],
                       "empty_cells": arms["lever"]["result"]["deciding"]["empty_cells_on_nonempty_boards"],
                       "contexts": contexts},
        "open": ["stacking the occupancy_context and occupancy_detail taps remains untested"],
        "unchanged": {"thresholds": {"per_cell_f1": CELL_F1_THRESHOLD, "ranking_auc": AUC_THRESHOLD,
                                     "ranking_f1": F1_THRESHOLD},
                      "population": "widened deciding population, twelve groups",
                      "source_checkpoint": source["checkpoint_sha256"],
                      "prior_evidence": "not touched", "frozen_manifest": "not touched"},
    }
    args.output.mkdir(parents=True)
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(scope, stream, indent=2, allow_nan=False)
    print(json.dumps({key: scope[key] for key in ("verdict", "mechanism", "reference_comparison")},
                     indent=2))


if __name__ == "__main__":
    main()
