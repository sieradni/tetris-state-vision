"""Curate the declared output parameterisation against the published control arm.

`docs/M8_NORMALISED_BOUNDARY.md` fixes the rule; this recorder checks that the control arm reproduces the
recorded numbers exactly in this catalogue, that both arms were measured under one catalogue and the same
envelope, re-derives every scored number from the sweeps' own counters, and applies the declared primary and
secondary readings. The primary metric is the plain-binary F1 at the standing `0.0` boundary - no calibration,
no sweep, no post-processing - because that is the boundary the parameterisation claims to place.
"""
import argparse
import json
from pathlib import Path

from gate_power import f1_from_counts
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
DECIDING_IMAGES = 36
ATTRIBUTION_DELTA = .05
AUC_TOLERANCE = .02
FAMILY = "m8-normbound"
PREDECLARATION = "docs/M8_NORMALISED_BOUNDARY.md"
PRIOR = ROOT/"docs/evidence/m7-calibration-r1/scope.json"
CONTROL = "control"
TREATED = "norm"
READINGS = ("normalisation-clears-the-boundary", "normalisation-moves-without-clearing",
            "normalisation-fails")
SECONDARY = ("ranking-preserved", "ranking-degraded")


def require_sweep(name):
    """Both arms must be the declared measurement, of their own run, under the declared envelope."""
    study = f"{FAMILY}-{name}-r1"
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
    if summary.get("config", {}).get("occupancy_normalise") != (1 if name == TREATED else 0):
        raise ValueError(f"{name} checkpoint does not carry the declared parameterisation")
    if summary.get("arguments", {}).get("occupancy_normalise") != (1 if name == TREATED else 0):
        raise ValueError(f"{name} run arguments do not carry the declared parameterisation")
    return study, summary, result


def row_error_at_zero(result):
    """The published row diagnostic at the standing boundary, recomputed from the sweep's own logits."""
    config = result["config"]
    rows, columns = config["visible_rows"], config["columns"]
    errors = []
    for case in result["cases"]:
        if case["split"] != "training" or not any(case["occupied"]):
            continue
        decided, occupied = boundaries(case["logits"], case["occupied"], rows, columns, 0.0)
        errors.extend(got-want for got, want in zip(decided, occupied)
                      if got is not None and want is not None)
    if not errors:
        raise ValueError("row diagnostic carried no columns")
    return {"mean": round(sum(errors)/len(errors), 6), "columns": len(errors)}


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
    if prior.get("instrument") != "m7-calibration-r1":
        raise ValueError("the reproduction reference is not the committed calibration block")
    expected_auc, expected_f1 = prior["arms"][CONTROL]["auc"], prior["arms"][CONTROL]["f1_at_zero"]

    grid = tuple(threshold_grid())
    arms, reproduction = {}, {}
    for name in (CONTROL, TREATED):
        study, summary, result = require_sweep(name)
        scored = arm_summary(result["deciding"])
        arms[name] = {
            "study": study, "sweep": f"{study}-sweep-r1",
            "occupancy_normalise": summary["config"]["occupancy_normalise"],
            "occupancy_detail": summary["config"].get("occupancy_detail", 0),
            "occupancy_offset_rows": summary["config"].get("occupancy_offset_rows", 0),
            "weights_sha256": result["weights_sha256"],
            "catalogue_sha256": summary["catalogue_sha256"],
            "study_elapsed_seconds": summary["elapsed_seconds"],
            "auc": result["deciding"]["auc"],
            "f1_at_zero": scored["per_cell_f1"],
            "operating_counts": scored["operating_counts"],
            "precision": round(scored["operating_counts"]["true_positive"]/
                                max(1, scored["operating_counts"]["true_positive"]
                                    + scored["operating_counts"]["false_positive"]), 6),
            "recall": round(scored["operating_counts"]["true_positive"]/
                             max(1, scored["operating_counts"]["true_positive"]
                                 + scored["operating_counts"]["false_negative"]), 6),
            "best": {"threshold": result["deciding"]["best"]["threshold"],
                     "f1": result["deciding"]["best"]["f1"],
                     "precision": result["deciding"]["best"]["precision"]},
            "out_of_fold_f1": components_of([case for case in result["cases"]
                                             if case["split"] == "training"], grid)["out_of_fold_f1"],
            "exact_boards": result["deciding"]["best"]["exact_nonempty_boards"],
            "row_error_at_zero": row_error_at_zero(result),
            "verdict": result["verdict"],
        }
        reproduction[name] = {"auc": result["deciding"]["auc"],
                              "f1_at_zero": scored["per_cell_f1"],
                              "prior_scope": str(PRIOR.relative_to(ROOT))}
    if len({arms[name]["catalogue_sha256"] for name in arms}) != 1:
        raise ValueError("the arms were captured under different catalogues")
    control, treated = arms[CONTROL], arms[TREATED]
    if control["auc"] != expected_auc or control["f1_at_zero"] != expected_f1:
        raise ValueError(f"control does not reproduce the recorded {expected_auc!r} / {expected_f1!r}")

    delta = round(treated["f1_at_zero"]-control["f1_at_zero"], 6)
    auc_delta = round(treated["auc"]-control["auc"], 6)
    if delta >= ATTRIBUTION_DELTA:
        reading = "normalisation-clears-the-boundary"
        reading_text = ("Training the read in normalised space moves the plain-binary operating point at "
                        "`0.0` past control's by the attribution floor, so the boundary the post-hoc rule "
                        "could not find is placed by the objective itself.")
    elif delta > 0:
        reading = "normalisation-moves-without-clearing"
        reading_text = ("The parameterisation moves the plain-binary boundary in the right direction but "
                        "short of the attribution floor: training can place the boundary only weakly, and "
                        "the drift is partly outside what the parameterisation controls.")
    else:
        reading = "normalisation-fails"
        reading_text = ("Training the read in normalised space does not put the plain-binary boundary "
                        "above control's, so the offset's operating point is not reachable by changing "
                        "where zero sits; the boundary has to be supervised directly.")
    if reading not in READINGS:
        raise ValueError("an undeclared reading was produced")
    secondary = "ranking-preserved" if auc_delta >= -AUC_TOLERANCE else "ranking-degraded"
    scope = {
        "status": "passed",
        "instrument": "m8-normalised-boundary-r1",
        "predeclaration": PREDECLARATION,
        "scope": ("Two 64-step arms - the published read-out and the same read-out trained with a "
                  "per-board normalised occupancy output - captured in one catalogue and scored read-only "
                  "at the standing 0.0 boundary and over the standing 81-point grid. The parameterisation "
                  "is inside the model, so the tensor the objective differentiates is the tensor scored; "
                  "it adds no parameter and no tensor, and 0 is exactly the published function. No "
                  "threshold change, no population change and no real-image, complete-state or quality "
                  "claim."),
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "mechanism": {
            "name": "per-board-normalised-occupancy-output",
            "flag": "--occupancy-normalise 1",
            "definition": "z = (logit - mean_visible) / sd_visible, computed inside the forward pass",
            "statistics": "the board's own visible cells; no label, no prevalence, no cross-board term",
            "parameters_added": 0, "tensors_added": 0,
            "scale_invariance": "positive rescalings cancel; a negative one inverts every decision",
            "identity_at_zero": "occupancy_normalise 0 is exactly the published function",
        },
        "envelope": {"steps": STEPS, "ceiling_bytes": CEILING_BYTES, "work_items":
                     {"sweep": FORWARDS, "curation": 9},
                     "attribution_floor": ATTRIBUTION_DELTA, "auc_tolerance": AUC_TOLERANCE},
        "arms": arms,
        "contrasts": {
            "norm_control": {"f1_at_zero_delta": delta, "auc_delta": auc_delta,
                             "best_f1_delta": round(treated["best"]["f1"]-control["best"]["f1"], 6),
                             "out_of_fold_delta": round(treated["out_of_fold_f1"]
                                                        - control["out_of_fold_f1"], 6),
                             "row_error_delta": round(treated["row_error_at_zero"]["mean"]
                                                      - control["row_error_at_zero"]["mean"], 6)},
        },
        "reading": reading,
        "reading_text": reading_text,
        "secondary_reading": secondary,
        "secondary_text": ("Deciding AUC is reported beside the primary metric with a declared tolerance "
                           "of -0.02: the parameterisation cannot add information, so a gain is a change "
                           "in what was learned and a loss is a real cost."),
        "verdict": lever_verdict(treated["f1_at_zero"], treated["verdict"], delta),
        "gates": {"per_cell_f1": CELL_F1_THRESHOLD, "ranking_auc": AUC_THRESHOLD,
                  "ranking_f1": F1_THRESHOLD, "attribution_floor": ATTRIBUTION_DELTA,
                  "grid_points": 81},
        "fit_gate": {"exact_nonempty_boards": control["exact_boards"], "boards": BOARDS,
                     "cleared": control["exact_boards"] == BOARDS},
        "reproduction": reproduction,
        "population": {"split": "training", "images": DECIDING_IMAGES, "boards": BOARDS,
                       "occupied_cells": OCCUPIED_CELLS, "prevalence": round(OCCUPIED_CELLS/(BOARDS*200), 6)},
    }
    args.output.mkdir(parents=True)
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(scope, stream, indent=2, allow_nan=False)
    print(json.dumps({key: scope[key] for key in ("reading", "secondary_reading", "contrasts",
                                                  "verdict", "fit_gate")}, indent=2, default=str))


if __name__ == "__main__":
    main()
