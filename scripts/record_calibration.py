"""Curate the label-free operating-point calibration against control, wide and the -1 offset arm.

`docs/M7_CALIBRATION.md` fixes the rule; this recorder checks that the three re-captured sweeps reproduce the
published record exactly, that the instrument's boundary at `0.0` re-derives from the sweep's own counters,
applies the two declared readings to the printed numbers, and writes one evidence scope. No threshold,
population or gate may move after this receipt exists.
"""
import argparse
import json
from pathlib import Path

from gate_power import f1_from_counts
from measure_calibrated_operating_point import ARMS, RECOVERY_FRACTION
from occupancy_sweep import AUC_THRESHOLD, F1_THRESHOLD, sweep_verdict, threshold_grid
from percell_gate import CELL_F1_THRESHOLD
from probe_model import require_bounded_job
from record_demean_lever import arm_summary, components_of, lever_verdict
from record_representation_lever import read

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 640*1024**2
CURATION_CEILING_BYTES = 1900*1024**2
ATTRIBUTION_DELTA = .05
STEPS = 64
DECIDING_IMAGES = 36
FORWARDS = 54
BOARDS = 28
OCCUPIED_CELLS = 667
FAMILY = "m7-calibration"
INSTRUMENT = "m7-calibration-r1"
PREDECLARATION = "docs/M7_CALIBRATION.md"
PRIOR = ROOT/"docs/evidence/m6-row-offset-r1/scope.json"
CONTROL = "control"
WIDE = "wide"
TREATED = "down"
# The reference points the declaration fixed before the receipts existed, from the M6 record.
CONTROL_F1_AT_ZERO = .381640
LABEL_CHOSEN_BEST = .469259
CONTROL_AUC = 0.8226433610683003
WIDE_AUC = 0.8822623454135491
READINGS = ("calibration-recovers-the-operating-point", "calibration-moves-without-recovering",
            "calibration-fails")
SECONDARY = ("offset-advantage-survives-calibration", "offset-advantage-was-a-threshold-effect")


def require_sweep(name, published_auc):
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
            result.get("forwards") != FORWARDS or result.get("weights_sha256") != summary.get("checkpoint_sha256")):
        raise ValueError(f"{name} sweep is not a complete measurement of its own run")
    if sweep_verdict(result["deciding"]) != result["verdict"]:
        raise ValueError(f"{name} sweep verdict does not re-derive")
    if (summary.get("steps") != STEPS or summary.get("status") != "complete" or
            study_job.get("status") != "passed" or study_job.get("work_items") != STEPS):
        raise ValueError(f"{name} study is not the declared 64-step budget")
    if result["deciding"]["occupied_cells"] != OCCUPIED_CELLS:
        raise ValueError(f"{name} sweep decided another population")
    if result["deciding"]["auc"] != published_auc:
        raise ValueError(f"{name} does not reproduce the published {published_auc!r}")
    return study, summary, result


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
    prior = read(PRIOR)
    if prior.get("instrument") != "m6-row-offset-r1":
        raise ValueError("the reproduction reference is not the committed row-offset block")
    published_auc = {CONTROL: CONTROL_AUC, WIDE: WIDE_AUC,
                     TREATED: prior["arms"][TREATED]["auc"]}
    measured = read(ROOT/"runs"/INSTRUMENT/"result.json")
    job = read(ROOT/"runs"/(INSTRUMENT+"-job.json"))
    if (job.get("status") != "passed" or job.get("exit_code") != 0 or
            job.get("memory_bytes") != CEILING_BYTES or job.get("work_items") != len(ARMS)):
        raise ValueError("the calibration instrument did not pass under its declared envelope")
    if measured.get("status") != "passed" or measured.get("instrument") != INSTRUMENT:
        raise ValueError("the calibration record is not the declared instrument's")

    grid = tuple(threshold_grid())
    arms, reproduction, summaries = {}, {}, {}
    for name in ARMS:
        study, summary, result = require_sweep(name, published_auc[name])
        block = measured["arms"][name]
        if block["arm"]["sweep"] != f"{FAMILY}-{name}-r1":
            raise ValueError(f"{name} calibration is not the sweep's own run")
        rederived = arm_summary(result["deciding"])
        if abs(block["zero"]["f1"]-rederived["per_cell_f1"]) > 1e-6:
            raise ValueError(f"{name} calibration does not re-derive the sweep's own 0.0 boundary")
        if abs(block["zero"]["precision"]-(block["zero"]["true_positive"]
                                           /max(1, block["zero"]["true_positive"]
                                                + block["zero"]["false_positive"]))) > 1e-6:
            raise ValueError(f"{name} calibration precision does not re-derive from its counts")
        if (result["deciding"]["best"]["f1"] != prior["arms"][name]["best"]["f1"] or
                rederived["per_cell_f1"] != prior["arms"][name]["per_cell_f1"]):
            raise ValueError(f"{name} does not reproduce the published record")
        arms[name] = {
            "study": study, "sweep": f"{study}-sweep-r1",
            "occupancy_deform": summary["arguments"].get("occupancy_deform", 0),
            "occupancy_offset_rows": summary["arguments"].get("occupancy_offset_rows", 0),
            "weights_sha256": result["weights_sha256"],
            "catalogue_sha256": summary["catalogue_sha256"],
            "study_elapsed_seconds": summary["elapsed_seconds"],
            "study_peak_commit_bytes": json.loads((ROOT/"runs"/(study+"-job.json")).read_text())
            .get("peak_job_commit_bytes"),
            "auc": result["deciding"]["auc"],
            "f1_at_zero": rederived["per_cell_f1"],
            "best": {"threshold": result["deciding"]["best"]["threshold"],
                     "f1": result["deciding"]["best"]["f1"],
                     "precision": result["deciding"]["best"]["precision"]},
            "exact_boards": result["deciding"]["best"]["exact_nonempty_boards"],
            "out_of_fold_f1": components_of([case for case in result["cases"]
                                              if case["split"] == "training"], grid)
            ["out_of_fold_f1"],
            "calibrated": {**block["calibrated"], "threshold_mean": block["thresholds"]["mean"],
                           "threshold_sd": block["thresholds"]["sd"],
                           "threshold_min": block["thresholds"]["min"],
                           "threshold_max": block["thresholds"]["max"],
                           "thresholds": block["thresholds"],
                           "distance_to_label_best": block["distance_to_label_best"],
                           "f1_gain_over_zero": block["f1_gain_of_calibration_over_zero"],
                           "recovery_of_label_chosen_gain":
                               block["recovery_of_the_label_chosen_gain"],
                           "label_free": True},
            "row_error": block["row_error"],
            "boards": block["boards"],
        }
        summaries[name] = rederived
        reproduction[name] = {"auc": result["deciding"]["auc"],
                              "f1_at_zero": rederived["per_cell_f1"],
                              "best_f1": result["deciding"]["best"]["f1"],
                              "prior_scope": str(PRIOR.relative_to(ROOT))}
    if len({arms[name]["catalogue_sha256"] for name in ARMS}) != 1:
        raise ValueError("the arms were captured under different catalogues")
    treated = arms[TREATED]
    if (treated["calibrated"]["f1"] >= CONTROL_F1_AT_ZERO+ATTRIBUTION_DELTA and
            treated["calibrated"]["f1"] >= RECOVERY_FRACTION*LABEL_CHOSEN_BEST):
        reading = "calibration-recovers-the-operating-point"
        reading_text = ("A threshold chosen from each board's own logits, with no labels and no swept "
                        "F1, puts the shifted read's operating point back above control's standing "
                        "boundary by the attribution floor and recovers at least 85% of the gain the "
                        "label-chosen boundary found, so the operating point is a determinable "
                        "property of the read.")
    elif treated["calibrated"]["f1"] > CONTROL_F1_AT_ZERO:
        reading = "calibration-moves-without-recovering"
        reading_text = ("The label-free threshold lifts the shifted read's operating point above "
                        "control's standing boundary but does not recover most of the label-chosen "
                        "gain, so the boundary is partly determinable and partly still dependent on "
                        "labels.")
    else:
        reading = "calibration-fails"
        reading_text = ("A threshold chosen from each board's own logits does not put the shifted read "
                        "above control's standing boundary at all, so the offset's boundary drift is "
                        "not a property the read can supply and the offset stays a ranking fix that "
                        "cannot be made operational at a fixed threshold.")
    if reading not in READINGS:
        raise ValueError("an undeclared reading was produced")
    calibrated_delta = round(treated["calibrated"]["f1"]
                             - arms[CONTROL]["calibrated"]["f1"], 6)
    secondary = ("offset-advantage-survives-calibration" if calibrated_delta >= ATTRIBUTION_DELTA
                 else "offset-advantage-was-a-threshold-effect")
    verdict = lever_verdict(treated["calibrated"]["f1"], summaries[TREATED]["verdict"],
                            calibrated_delta)
    scope = {
        "status": "passed",
        "instrument": INSTRUMENT,
        "predeclaration": PREDECLARATION,
        "scope": ("Three re-captured 64-step arms - the published read-out, the four-cell registration "
                  "field and the declared -1 row offset - scored read-only at two operating rules: the "
                  "standing 0.0 boundary and a per-board threshold derived from the arm's own logits "
                  "with no labels. The calibration adds no parameter, touches no weight and cannot "
                  "change ranking, so every AUC printed here is the uncalibrated arm's. No threshold, "
                  "population or gate change, and no real-image, complete-state or quality claim."),
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "rule": {"name": "otsu-logits", "label_free": True, "applied_to": list(ARMS),
                 "candidates": "midpoints of consecutive distinct logits, ties to the lower candidate",
                 "constant_board": "chooses its own value, so nothing is decided",
                 "auc_effect": "none by construction - a scoring rule does not touch ranking"},
        "envelope": {"steps": STEPS, "ceiling_bytes": CEILING_BYTES,
                     "sweep_ceiling_bytes": CEILING_BYTES,
                     "work_items": {"sweep": FORWARDS, "calibration": len(ARMS), "curation": 9},
                     "attribution_floor": ATTRIBUTION_DELTA,
                     "recovery_fraction": RECOVERY_FRACTION},
        "reference": {"control_f1_at_zero": CONTROL_F1_AT_ZERO,
                      "label_chosen_best_f1": LABEL_CHOSEN_BEST,
                      "declared_before_receipts": True,
                      "source": str(PRIOR.relative_to(ROOT))},
        "arms": arms,
        "contrasts": {
            "down_calibrated_control_calibrated": {"f1_delta": calibrated_delta,
                                                   "attribution_met": calibrated_delta >= ATTRIBUTION_DELTA},
            "down_control": {"auc_delta": round(arms[TREATED]["auc"]-arms[CONTROL]["auc"], 6),
                             "f1_at_zero_delta": round(treated["f1_at_zero"]
                                                       - arms[CONTROL]["f1_at_zero"], 6),
                             "calibrated_f1_delta": round(treated["calibrated"]["f1"]
                                                          - arms[CONTROL]["calibrated"]["f1"], 6)},
            "down_wide": {"auc_delta": round(arms[TREATED]["auc"]-arms[WIDE]["auc"], 6),
                          "best_f1_delta": round(treated["best"]["f1"]-arms[WIDE]["best"]["f1"], 6),
                          "calibrated_f1_delta": round(treated["calibrated"]["f1"]
                                                       - arms[WIDE]["calibrated"]["f1"], 6)},
            "wide_control": {"auc_delta": round(arms[WIDE]["auc"]-arms[CONTROL]["auc"], 6)},
        },
        "reading": reading,
        "reading_text": reading_text,
        "secondary_reading": secondary,
        "secondary_text": ("At their own label-free thresholds the offset arm and the control arm are "
                           "compared directly, so a shifted-boundary effect that calibration removes "
                           "cannot be mistaken for a ranking advantage."),
        "verdict": verdict,
        "gates": {"per_cell_f1": CELL_F1_THRESHOLD, "ranking_auc": AUC_THRESHOLD,
                  "ranking_f1": F1_THRESHOLD, "attribution_floor": ATTRIBUTION_DELTA,
                  "grid_points": 81},
        "fit_gate": {"exact_nonempty_boards": arms[CONTROL]["exact_boards"],
                     "boards": BOARDS, "cleared": False},
        "reproduction": reproduction,
        "population": {"split": "training", "images": DECIDING_IMAGES, "boards": BOARDS,
                       "occupied_cells": OCCUPIED_CELLS},
        "omitted": {"arm": "-2 row offset", "reason": "requires a fourth 64-step training round, which "
                                                     "this pass forbids; no receipt exists for an arm "
                                                     "that was not run"},
    }
    args.output.mkdir(parents=True)
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(scope, stream, indent=2, allow_nan=False)
    print(json.dumps({key: scope[key] for key in ("reading", "secondary_reading", "contrasts",
                                                  "verdict", "fit_gate")}, indent=2, default=str))


if __name__ == "__main__":
    main()
