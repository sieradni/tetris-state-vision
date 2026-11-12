"""Curate the two-sided supervised anchor against the published control arm.

`docs/M10_SUPERVISED_ANCHOR.md` fixes the margin, the weight and the reading rule before any receipt; this
recorder checks that both arms were captured in one catalogue under the declared envelope, that control reproduces
the recorded numbers exactly, that the arms differ in exactly the declared supervision, reports the anchor the
objective actually paid, and applies the declared primary and secondary readings to the swept numbers themselves.
The primary metric is the plain-binary F1 at the standing `0.0` boundary - no calibration, no sweep selection, no
post-processing - because `0.0` is the boundary the anchor claims to supervise.
"""
import argparse
import json
from pathlib import Path
import sys

from measure_row_error import boundaries
from occupancy_sweep import AUC_THRESHOLD, F1_THRESHOLD, sweep_verdict, threshold_grid
from percell_gate import CELL_F1_THRESHOLD
from probe_model import require_bounded_job
from record_demean_lever import arm_summary, components_of, lever_verdict
from record_representation_lever import read

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from tetris_vision.losses import ANCHOR_MARGIN, ANCHOR_WEIGHT  # noqa: E402

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
FAMILY = "m10-anchor"
INSTRUMENT = "m10-supervised-anchor-r1"
PREDECLARATION = "docs/M10_SUPERVISED_ANCHOR.md"
PRIOR = ROOT/"docs/evidence/m9-supervised-margin-r1/scope.json"
CONTROL = "control"
TREATED = "anchor"
# Declared run suffixes: the family is `m10-anchor-`, and the treated arm's run is `...-two-sided-r1`.
RUNS = {CONTROL: "control", TREATED: "two-sided"}
READINGS = ("anchor-clears-the-boundary", "anchor-moves-without-clearing", "anchor-fails")
SECONDARY = ("ranking-preserved", "ranking-degraded")


def require_sweep(name):
    """Both arms must be the declared measurement, of their own run, under the declared envelope."""
    study = f"{FAMILY}-{RUNS[name]}-r1"
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
    declared = 1 if name == TREATED else 0
    if (summary.get("arguments", {}).get("occupancy_anchor") != declared or
            summary.get("recovery_contract", {}).get("occupancy_anchor") != declared):
        raise ValueError(f"{name} run does not carry the declared anchor supervision")
    # The arms may differ in the supervised anchor and in nothing else that moves the read or zero.
    for key in ("occupancy_margin_hinge", "occupancy_offset_rows", "occupancy_normalise", "occupancy_deform",
                "occupancy_context", "occupancy_detail_stencil", "occupancy_resolution", "occupancy_column",
                "occupancy_lattice_free", "occupancy_placed"):
        if summary.get("config", {}).get(key, 0) or summary.get("arguments", {}).get(key, 0):
            raise ValueError(f"{name} carries an undeclared read lever: {key}")
    return study, summary, result


def training_anchor(study):
    """The anchor the arm actually paid, from the study's own step log (in-sample, not a measurement)."""
    log = ROOT/"runs"/(study+"-study.log")
    if not log.exists():
        raise ValueError(f"{study} step log is missing, so the paid anchor cannot be reported")
    values = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if '"terms":' not in line or '"anchor":' not in line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        terms = record.get("terms") or {}
        if "anchor" in terms and type(record.get("step")) is int:
            values.append((record["step"], float(terms["anchor"])))
    if len(values) != STEPS:
        raise ValueError(f"{study} does not log the anchor at every declared step")
    return {"steps": len(values), "first": round(values[0][1], 6), "last_step": values[-1][0],
            "last": round(values[-1][1], 6),
            "mean": round(sum(value for _, value in values)/len(values), 6),
            "fraction_satisfied": round(sum(1 for _, value in values if value == 0.0)/len(values), 6),
            "source": str(log.relative_to(ROOT)), "in_sample": True,
            "reading": "the anchor the objective paid on its own training batch; it constrains both the "
                       "separation and the location of the boundary at 0.0"}


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
    if prior.get("instrument") != "m9-supervised-margin-r1":
        raise ValueError("the reproduction reference is not the committed margin block")
    expected_auc, expected_f1 = prior["arms"][CONTROL]["auc"], prior["arms"][CONTROL]["f1_at_zero"]

    grid = tuple(threshold_grid())
    arms, reproduction = {}, {}
    for name in (CONTROL, TREATED):
        study, summary, result = require_sweep(name)
        scored = arm_summary(result["deciding"])
        counts = scored["operating_counts"]
        arms[name] = {
            "study": study, "sweep": f"{study}-sweep-r1",
            "occupancy_anchor": summary["arguments"]["occupancy_anchor"],
            "occupancy_detail": summary["config"].get("occupancy_detail", 0),
            "occupancy_offset_rows": summary["config"].get("occupancy_offset_rows", 0),
            "weights_sha256": result["weights_sha256"],
            "catalogue_sha256": summary["catalogue_sha256"],
            "study_elapsed_seconds": summary["elapsed_seconds"],
            "auc": result["deciding"]["auc"],
            "f1_at_zero": scored["per_cell_f1"],
            "operating_counts": counts,
            "precision": round(counts["true_positive"]
                               /max(1, counts["true_positive"]+counts["false_positive"]), 6),
            "recall": round(counts["true_positive"]
                            /max(1, counts["true_positive"]+counts["false_negative"]), 6),
            "best": {"threshold": result["deciding"]["best"]["threshold"],
                     "f1": result["deciding"]["best"]["f1"],
                     "precision": result["deciding"]["best"]["precision"]},
            "out_of_fold_f1": components_of([case for case in result["cases"]
                                             if case["split"] == "training"], grid)["out_of_fold_f1"],
            "exact_boards": result["deciding"]["best"]["exact_nonempty_boards"],
            "row_error_at_zero": row_error_at_zero(result),
            "training_anchor": training_anchor(study) if name == TREATED else None,
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

    hinge = prior["arms"]["hinge"]
    delta = round(treated["f1_at_zero"]-control["f1_at_zero"], 6)
    auc_delta = round(treated["auc"]-control["auc"], 6)
    if delta >= ATTRIBUTION_DELTA:
        reading = "anchor-clears-the-boundary"
        reading_text = ("Anchoring the declared margin to the boundary a consumer applies moves the "
                        "plain-binary operating point at `0.0` past control's by the attribution floor, so the "
                        "operating point is supervised rather than guessed and the boundary the project could "
                        "not find post-hoc or by re-parameterisation is placed by the objective.")
    elif delta > 0:
        reading = "anchor-moves-without-clearing"
        reading_text = ("The anchor moves the plain-binary boundary in the right direction but short of the "
                        "attribution floor: naming `0.0` in the objective shapes the boundary without placing "
                        "it far enough to matter.")
    else:
        reading = "anchor-fails"
        reading_text = ("Naming `0.0` in the objective on both sides of the margin does not put the plain-binary "
                        "boundary above control's, so a per-board margin cannot place the operating point at all "
                        "and the next term has to be a learned boundary parameter fitted by the loss.")
    if reading not in READINGS:
        raise ValueError("an undeclared reading was produced")
    secondary = "ranking-preserved" if auc_delta >= -AUC_TOLERANCE else "ranking-degraded"
    scope = {
        "status": "passed",
        "instrument": INSTRUMENT,
        "predeclaration": PREDECLARATION,
        "scope": ("Two 64-step arms - the published read-out and the same read-out with the declared two-sided "
                  "anchor added to the occupancy objective - captured in one catalogue and scored read-only at "
                  "the standing 0.0 boundary and over the standing 81-point grid. The anchor is supervision: it "
                  "names no threshold, counts no cells, reads no prevalence and adds no parameter, and 0 is "
                  "exactly the published objective. No threshold change, no population change and no "
                  "real-image, complete-state or quality claim."),
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "mechanism": {
            "name": "two-sided-supervised-anchor",
            "flag": "--occupancy-anchor 1",
            "definition": "max(0, margin - lowest_occupied_logit) + max(0, margin + highest_empty_logit), mean "
                          "over boards carrying both classes among their known visible cells",
            "margin": ANCHOR_MARGIN, "weight": ANCHOR_WEIGHT,
            "declared_before_receipts": True,
            "parameters_added": 0, "tensors_added": 0,
            "unknown_cells": "excluded from both sides, so an invisible cell is never read as empty",
            "identity_at_zero": "occupancy_anchor 0 is exactly the published objective",
            "supersedes": "the M9 one-sided hinge; the trainer refuses the two together",
        },
        "envelope": {"steps": STEPS, "ceiling_bytes": CEILING_BYTES,
                     "work_items": {"sweep": FORWARDS, "curation": 9},
                     "attribution_floor": ATTRIBUTION_DELTA, "auc_tolerance": AUC_TOLERANCE},
        "arms": arms,
        "contrasts": {
            "anchor_control": {"f1_at_zero_delta": delta, "auc_delta": auc_delta,
                               "best_f1_delta": round(treated["best"]["f1"]-control["best"]["f1"], 6),
                               "out_of_fold_delta": round(treated["out_of_fold_f1"]
                                                          - control["out_of_fold_f1"], 6),
                               "row_error_delta": round(treated["row_error_at_zero"]["mean"]
                                                        - control["row_error_at_zero"]["mean"], 6)},
        },
        "reference": {
            "hinge": {"auc": hinge["auc"], "f1_at_zero": hinge["f1_at_zero"],
                      "best_f1": hinge["best"]["f1"], "out_of_fold_f1": hinge["out_of_fold_f1"],
                      "gates_at_zero": hinge["operating_counts"],
                      "scope": str(PRIOR.relative_to(ROOT))},
            "role": "reference-only: the M9 one-sided hinge, measured in its own catalogue and not re-measured "
                    "or re-verdicted here, so a reader can see whether pinning the location cost the ranking "
                    "the hinge found or bought the operating point it lacked",
        },
        "reading": reading,
        "reading_text": reading_text,
        "secondary_reading": secondary,
        "secondary_text": ("Deciding AUC is reported beside the primary metric with M8/M9's declared tolerance "
                           "of -0.02 so the passes are comparable: an anchored margin cannot add information, so "
                           "a gain is a change in what was learned and a loss is a real cost."),
        "verdict": lever_verdict(treated["f1_at_zero"], treated["verdict"], delta),
        "gates": {"per_cell_f1": CELL_F1_THRESHOLD, "ranking_auc": AUC_THRESHOLD,
                  "ranking_f1": F1_THRESHOLD, "attribution_floor": ATTRIBUTION_DELTA,
                  "grid_points": 81},
        "fit_gate": {"exact_nonempty_boards": control["exact_boards"], "boards": BOARDS,
                     "cleared": control["exact_boards"] == BOARDS},
        "reproduction": reproduction,
        "population": {"split": "training", "images": DECIDING_IMAGES, "boards": BOARDS,
                       "occupied_cells": OCCUPIED_CELLS,
                       "prevalence": round(OCCUPIED_CELLS/(BOARDS*200), 6)},
    }
    args.output.mkdir(parents=True)
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(scope, stream, indent=2, allow_nan=False)
    print(json.dumps({key: scope[key] for key in ("reading", "secondary_reading", "contrasts",
                                                  "verdict", "fit_gate")}, indent=2, default=str))


if __name__ == "__main__":
    main()
