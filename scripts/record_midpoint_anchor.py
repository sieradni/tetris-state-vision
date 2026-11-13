"""Curate the midpoint anchor - supervision of where the boundary sits - against the published control arm.

`docs/M13_MIDPOINT_ANCHOR.md` fixes the target location, the weight, the class balance, the
expected reading and the AUC override rule before any receipt; this recorder checks that both arms were captured in one
catalogue under the declared envelope, that control reproduces the recorded numbers exactly, that the arms differ in
exactly the declared supervision (one location per board, no margin, equal class weight), reports the location charge the
objective actually paid, reports how much of the population each arm calls positive at `0.0`, and applies the declared
primary and secondary readings to the swept numbers themselves. The primary metric is the plain-binary F1 at the
standing `0.0` boundary - no calibration, no sweep selection, no post-processing - because `0.0` is the boundary the
term claims to supervise.
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
from tetris_vision.losses import (ANCHOR_WEIGHT, MIDPOINT_EMPTY_BALANCE, MIDPOINT_OCCUPIED_BALANCE,  # noqa: E402
                                  MIDPOINT_TARGET)

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 640*1024**2
CURATION_CEILING_BYTES = 1900*1024**2
STEPS = 64
FORWARDS = 54
BOARDS = 28
OCCUPIED_CELLS = 667
KNOWN_CELLS = BOARDS*200
DECIDING_IMAGES = 36
ATTRIBUTION_DELTA = .05
AUC_TOLERANCE = .02
FAMILY = "m13-midpoint"
INSTRUMENT = "m13-midpoint-r1"
PREDECLARATION = "docs/M13_MIDPOINT_ANCHOR.md"
PRIOR = ROOT/"docs/evidence/m10-supervised-anchor-r1/scope.json"
QUANTILE = ROOT/"docs/evidence/m11-quantile-anchor-r1/scope.json"
DENSE = ROOT/"docs/evidence/m12-dense-hinge-r1/scope.json"
HINGE = ROOT/"docs/evidence/m9-supervised-margin-r1/scope.json"
CONTROL = "control"
TREATED = "level"
# Declared run suffixes: the family is `m12-dense-hinge-`, and the treated arm's run is `...-level-r1`.
RUNS = {CONTROL: "control", TREATED: "level"}
READINGS = ("midpoint-clears-the-boundary", "midpoint-moves-without-clearing", "midpoint-fails")
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
    if (summary.get("arguments", {}).get("occupancy_midpoint") != declared or
            summary.get("recovery_contract", {}).get("occupancy_midpoint") != declared):
        raise ValueError(f"{name} run does not carry the declared location supervision")
    # The arms may differ in the midpoint anchor and in nothing else - least of all in the four existing terms,
    # which the declaration says do not stack with it.
    for key in ("occupancy_dense_hinge", "occupancy_quantile_anchor", "occupancy_anchor", "occupancy_margin_hinge",
                "occupancy_offset_rows", "occupancy_normalise",
                "occupancy_deform", "occupancy_context", "occupancy_detail_stencil", "occupancy_resolution",
                "occupancy_column", "occupancy_lattice_free", "occupancy_placed"):
        if summary.get("config", {}).get(key, 0) or summary.get("arguments", {}).get(key, 0):
            raise ValueError(f"{name} carries an undeclared read lever: {key}")
    return study, summary, result


def training_midpoint(study):
    """The location charge the arm actually paid, from the study's own step log (in-sample, not a measurement)."""
    log = ROOT/"runs"/(study+"-study.log")
    if not log.exists():
        raise ValueError(f"{study} step log is missing, so the paid term cannot be reported")
    values = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if '"terms":' not in line or '"midpoint":' not in line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        terms = record.get("terms") or {}
        if "midpoint" in terms and type(record.get("step")) is int:
            values.append((record["step"], float(terms["midpoint"])))
    if len(values) != STEPS:
        raise ValueError(f"{study} does not log the location charge at every declared step")
    return {"steps": len(values), "first": round(values[0][1], 6), "last_step": values[-1][0],
            "last": round(values[-1][1], 6),
            "mean": round(sum(value for _, value in values)/len(values), 6),
            "fraction_satisfied": round(sum(1 for _, value in values if value == 0.0)/len(values), 6),
            "source": str(log.relative_to(ROOT)), "in_sample": True,
            "reading": "the location charge the objective paid on its own training batch; it is zero only when a "
                       "board's two class means straddle the boundary symmetrically"}


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


def exact_boards_at_zero(result):
    """Boards the standing `0.0` decision reproduces exactly - the dense hinge's declared fit gate.

    The declaration's implication is that a satisfied band forces every known cell a full margin from `0.0` on its
    correct side, so satisfaction implies an exact `0.0` boundary on every board. Counting the boards `0.0` decides
    exactly therefore tests the arm against the implication rather than against a swept optimum.
    """
    config = result["config"]
    rows, columns = config["visible_rows"], config["columns"]
    exact = total = 0
    for case in result["cases"]:
        if case["split"] != "training" or not any(case["occupied"]):
            continue
        decided, occupied = boundaries(case["logits"], case["occupied"], rows, columns, 0.0)
        pairs = [(got, want) for got, want in zip(decided, occupied) if got is not None and want is not None]
        if not pairs:
            continue
        total += 1
        exact += all(got == want for got, want in pairs)
    if not total:
        raise ValueError("the fit gate carried no board")
    return {"boards": total, "exact": exact}


def reference(path, name):
    scope = read(path)
    arm = scope["arms"][name]
    return {"auc": arm["auc"], "f1_at_zero": arm["f1_at_zero"], "best_f1": arm["best"]["f1"],
            "out_of_fold_f1": arm["out_of_fold_f1"], "scope": str(path.relative_to(ROOT))}


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
    if prior.get("instrument") != "m10-supervised-anchor-r1":
        raise ValueError("the reproduction reference is not the committed extremal-anchor block")
    expected_auc, expected_f1 = prior["arms"][CONTROL]["auc"], prior["arms"][CONTROL]["f1_at_zero"]

    grid = tuple(threshold_grid())
    arms, reproduction = {}, {}
    for name in (CONTROL, TREATED):
        study, summary, result = require_sweep(name)
        scored = arm_summary(result["deciding"])
        counts = scored["operating_counts"]
        positives = counts["true_positive"]+counts["false_positive"]
        arms[name] = {
            "study": study, "sweep": f"{study}-sweep-r1",
            "occupancy_midpoint": summary["arguments"]["occupancy_midpoint"],
            "occupancy_dense_hinge": summary["arguments"].get("occupancy_dense_hinge", 0),
            "occupancy_quantile_anchor": summary["arguments"].get("occupancy_quantile_anchor", 0),
            "occupancy_anchor": summary["arguments"].get("occupancy_anchor", 0),
            "occupancy_margin_hinge": summary["arguments"].get("occupancy_margin_hinge", 0),
            "occupancy_detail": summary["config"].get("occupancy_detail", 0),
            "occupancy_offset_rows": summary["config"].get("occupancy_offset_rows", 0),
            "weights_sha256": result["weights_sha256"],
            "catalogue_sha256": summary["catalogue_sha256"],
            "study_elapsed_seconds": summary["elapsed_seconds"],
            "auc": result["deciding"]["auc"],
            "f1_at_zero": scored["per_cell_f1"],
            "operating_counts": counts,
            "positives_at_zero": positives,
            "positive_share_at_zero": round(positives/KNOWN_CELLS, 6),
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
            "exact_boards_at_zero": exact_boards_at_zero(result),
            "training_midpoint": training_midpoint(study) if name == TREATED else None,
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
        reading = "midpoint-clears-the-boundary"
        reading_text = ("Charging the midpoint of the two class means against the boundary itself moves the "
                        "plain-binary operating point at `0.0` past control's by the attribution floor. Location "
                        "was the obstacle: the shape was already there after M12, and the read simply sat about "
                        "three logits high.")
    elif delta > 0:
        reading = "midpoint-moves-without-clearing"
        reading_text = ("The midpoint anchor moves the plain-binary boundary in the right direction but short of the "
                        "attribution floor: pinning the location shifts the read without placing `0.0` far "
                        "enough to matter.")
    else:
        reading = "midpoint-fails"
        reading_text = ("Charging the midpoint of the two class means does not put the plain-binary boundary above "
                        "control's, so supervising the location of the published read-out cannot place `0.0` on "
                        "this data and the next arm has to remove the additive degree of freedom by "
                        "construction instead.")
    if reading not in READINGS:
        raise ValueError("an undeclared reading was produced")
    secondary = "ranking-preserved" if auc_delta >= -AUC_TOLERANCE else "ranking-degraded"
    override = None
    if secondary == "ranking-degraded":
        # Declared rule: a boundary placed by pulling the two class means together is not an operating point, so the
        # AUC tolerance overrules the primary and the record says so plainly.
        override = {"primary_reading": reading, "forced_reading": "midpoint-fails",
                    "rule": "the deciding AUC fell more than 0.02 below control's, which is the declared signature of "
                            "the two class means collapsing onto each other; the verdict is midpoint-fails whatever "
                            "the primary metric says"}
        reading = "midpoint-fails"
        reading_text = ("The midpoint anchor moved the boundary only by destroying separation - the deciding AUC "
                        "fell outside the declared tolerance, which is the collapse failure mode the declaration "
                        "named in advance - so the reading is `midpoint-fails` whatever the primary metric says.")
    scope = {
        "status": "passed",
        "instrument": INSTRUMENT,
        "predeclaration": PREDECLARATION,
        "scope": ("Two 64-step arms - the published read-out and the same read-out with the declared midpoint "
                  "anchor added to the occupancy objective - captured in one catalogue and scored read-only at the "
                  "standing 0.0 boundary and over the standing 81-point grid. The anchor is supervision of one "
                  "location per board: no margin, equal class weight (never a count), no threshold, no prevalence "
                  "and no parameter, and 0 is exactly the published objective. No threshold change, no population "
                  "change and no real-image, complete-state or quality claim."),
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "mechanism": {
            "name": "midpoint-anchored-boundary",
            "flag": "--occupancy-midpoint 1",
            "definition": "mean over qualifying boards of |0.5*mean_occupied + 0.5*mean_empty - MIDPOINT_TARGET| "
                          "over known visible occupancy logits",
            "weight": ANCHOR_WEIGHT, "target": MIDPOINT_TARGET, "margin": None,
            "occupied_balance": MIDPOINT_OCCUPIED_BALANCE, "empty_balance": MIDPOINT_EMPTY_BALANCE,
            "balance_is": "equal class weight, never a cell count or a population statistic",
            "declared_before_receipts": True,
            "parameters_added": 0, "tensors_added": 0,
            "unknown_cells": "excluded from both sides, so an invisible cell is never read as empty",
            "identity_at_zero": "occupancy_midpoint 0 is exactly the published objective",
            "supersedes": "the M12 dense hinge, the M11 quantile anchor, the M10 two-sided anchor and the M9 "
                          "one-sided hinge; the trainer refuses all five together",
            "expected_reading": "midpoint-clears-the-boundary",
            "expected_f1_at_zero_hint": .465582,
            "failure_mode": "satisfiable by pulling the occupied mean down to meet the empty mean; the declared "
                            "AUC tolerance overrides the primary if that happens",
        },
        "envelope": {"steps": STEPS, "ceiling_bytes": CEILING_BYTES,
                     "work_items": {"sweep": FORWARDS, "curation": 9},
                     "attribution_floor": ATTRIBUTION_DELTA, "auc_tolerance": AUC_TOLERANCE},
        "arms": arms,
        "contrasts": {
            "midpoint_control": {"f1_at_zero_delta": delta, "auc_delta": auc_delta,
                             "best_f1_delta": round(treated["best"]["f1"]-control["best"]["f1"], 6),
                             "out_of_fold_delta": round(treated["out_of_fold_f1"]
                                                        - control["out_of_fold_f1"], 6),
                             "positive_share_delta": round(treated["positive_share_at_zero"]
                                                           - control["positive_share_at_zero"], 6),
                             "row_error_delta": round(treated["row_error_at_zero"]["mean"]
                                                      - control["row_error_at_zero"]["mean"], 6)},
        },
        "reference": {
            "dense_hinge": reference(DENSE, "banded"),
            "extremal_anchor": reference(PRIOR, "anchor"),
            "quantile_anchor": reference(QUANTILE, "bulk"),
            "hinge": reference(HINGE, "hinge"),
            "role": "reference-only: the M12 dense class-balanced hinge, the M10 two-sided extremal anchor, the M11 "
                    "class-quantile anchor and the M9 one-sided hinge, each measured in its own catalogue and none "
                    "re-measured or re-verdicted here, so a reader can see the whole progression from two cells, to "
                    "a tenth of each class, to every cell, to one location",
        },
        "reading": reading,
        "reading_text": reading_text,
        "secondary_reading": secondary,
        "secondary_text": ("Deciding AUC is reported beside the primary metric with the standing -0.02 tolerance, "
                           "and it is an override rather than a footnote: this term carries no separation pressure "
                           "of its own and can be satisfied by collapsing the two class means together, so a "
                           "ranking loss outside the tolerance forces the reading to midpoint-fails."),
        "reading_override": override,
        "verdict": lever_verdict(treated["f1_at_zero"], treated["verdict"], delta),
        "gates": {"per_cell_f1": CELL_F1_THRESHOLD, "ranking_auc": AUC_THRESHOLD,
                  "ranking_f1": F1_THRESHOLD, "attribution_floor": ATTRIBUTION_DELTA,
                  "grid_points": 81},
        "fit_gate": {"exact_nonempty_boards": treated["exact_boards_at_zero"]["exact"],
                     "boards": treated["exact_boards_at_zero"]["boards"],
                     "cleared": (treated["exact_boards_at_zero"]["exact"]
                                 == treated["exact_boards_at_zero"]["boards"] == BOARDS),
                     "boundary": 0.0, "measured_on": TREATED,
                     "expectation": "placement was the only remaining obstacle after M12 reached 1 of 28, so this "
                                     "is the count the location term is expected to move"},
        "reproduction": reproduction,
        "population": {"split": "training", "images": DECIDING_IMAGES, "boards": BOARDS,
                       "occupied_cells": OCCUPIED_CELLS, "known_cells": KNOWN_CELLS,
                       "prevalence": round(OCCUPIED_CELLS/KNOWN_CELLS, 6)},
    }
    args.output.mkdir(parents=True)
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(scope, stream, indent=2, allow_nan=False)
    print(json.dumps({key: scope[key] for key in ("reading", "secondary_reading", "contrasts",
                                                  "verdict", "fit_gate")}, indent=2, default=str))


if __name__ == "__main__":
    main()
