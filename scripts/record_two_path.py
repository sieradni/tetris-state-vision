"""Curate the two-path read against the reproduced published read, with each channel scored alone on the same fold.

`docs/M20_TWO_PATH_READ.md` declares the arm, the numeric bar for `passes` and the failure vocabulary before any receipt.
This recorder asserts the two declared fields and nothing else, that each sweep carries its own study's checkpoint while
the neighbour, span-only and span-under-level blocks are excluded by name, that the control reproduces the published
numbers exactly, and it scores three readings on the same fold - the sum, the lattice-sampled channel alone under the
arm's own transforms, and the free channel alone - so the record can say whether the sum beat its own parts.
"""
import argparse
import json
from pathlib import Path

import torch
from tetris_vision.model import centre_board, otsu_board, stack_contiguous

from measure_row_error import boundaries
from occupancy_sweep import AUC_THRESHOLD, F1_THRESHOLD, evaluate, sweep_verdict, threshold_grid
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
KNOWN_CELLS = BOARDS*200
DECIDING_IMAGES = 36
ATTRIBUTION_DELTA = .05
AUC_TOLERANCE = .02
CHANNEL_GAIN = .005
CHANNEL_GAP = .01
FAMILY = "m20-two-path"
INSTRUMENT = "m20-two-path-read-r1"
PREDECLARATION = "docs/M20_TWO_PATH_READ.md"
PREVIOUS = ROOT/"docs/evidence/m15-stack-combined-r1/scope.json"
LEVEL = ROOT/"docs/evidence/m19-span-level-r1/scope.json"
SPAN = ROOT/"docs/evidence/m18-tap-span-r1/scope.json"
SKYLINE = ROOT/"docs/evidence/m17-skyline-recovery-r1/scope.json"
RULED = ROOT/"docs/evidence/m16-crossing-threshold-r1/scope.json"
ENVELOPE = ROOT/"docs/evidence/m15-stack-contiguity-r1/scope.json"
CENTRING = ROOT/"docs/evidence/m14-read-centring-r1/scope.json"
MIDPOINT = ROOT/"docs/evidence/m13-midpoint-r1/scope.json"
DENSE = ROOT/"docs/evidence/m12-dense-hinge-r1/scope.json"
QUANTILE = ROOT/"docs/evidence/m11-quantile-anchor-r1/scope.json"
PRIOR = ROOT/"docs/evidence/m10-supervised-anchor-r1/scope.json"
HINGE = ROOT/"docs/evidence/m9-supervised-margin-r1/scope.json"
CONTROL = "control"
TREATED = "two-path"
RUNS = {CONTROL: "control", TREATED: "two-path"}
READINGS = ("two-path-passes", "two-path-moves-without-clearing", "two-path-fails")
CHANNEL_VERDICTS = ("complementary", "one-channel-noise-dominates", "no-complementary-signal")
SECONDARY = ("ranking-preserved", "ranking-degraded")
SUPERVISION = ("occupancy_midpoint", "occupancy_dense_hinge", "occupancy_quantile_anchor", "occupancy_anchor",
               "occupancy_margin_hinge")
READ_LEVERS = ("occupancy_normalise", "occupancy_offset_rows", "occupancy_deform", "occupancy_context",
               "occupancy_detail_stencil", "occupancy_resolution", "occupancy_column", "occupancy_placed",
               "occupancy_centre", "occupancy_crossing_threshold", "occupancy_skyline_recovery",
               "occupancy_tap_span")
DECLARED = ("occupancy_lattice_free", "occupancy_stack_monotone")
RECORDED_CONFIG = ("occupancy_lattice_free", "occupancy_stack_monotone", "occupancy_detail", "occupancy_normalise",
                   "occupancy_centre", "occupancy_crossing_threshold", "occupancy_skyline_recovery",
                   "occupancy_tap_span", "dimension", "component_capacity")


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
    declared = 2 if name == TREATED else 0
    if (summary.get("config", {}).get("occupancy_lattice_free") != declared or
            summary.get("arguments", {}).get("occupancy_lattice_free") != declared or
            summary.get("recovery_contract", {}).get("occupancy_lattice_free") != declared):
        raise ValueError(f"{name} run does not carry occupancy_lattice_free={declared}")
    envelope = 1 if name == TREATED else 0
    if (summary.get("config", {}).get("occupancy_stack_monotone") != envelope or
            summary.get("arguments", {}).get("occupancy_stack_monotone") != envelope or
            summary.get("recovery_contract", {}).get("occupancy_stack_monotone") != envelope):
        raise ValueError(f"{name} run does not carry occupancy_stack_monotone={envelope}")
    if summary.get("config", {}).get("occupancy_detail") != 1:
        raise ValueError(f"{name} run does not carry the published detail tap")
    for key in SUPERVISION+READ_LEVERS:
        if summary.get("config", {}).get(key, 0) or summary.get("arguments", {}).get(key, 0):
            raise ValueError(f"{name} carries an undeclared lever beside the two-path read: {key}")
    if name == TREATED:
        for case in result["cases"]:
            if "logits_parts" not in case or sorted(case["logits_parts"]) != ["free", "lattice"]:
                raise ValueError("the two-path sweep did not record both channel logits per case")
    return study, summary, result


def training_occupancy(study):
    """The occupancy loss the arm paid, since this structural read adds no term of its own."""
    log = ROOT/"runs"/(study+"-study.log")
    if not log.exists():
        raise ValueError(f"{study} step log is missing")
    values = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if '"terms":' not in line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        terms = record.get("terms") or {}
        if "occupancy" in terms and type(record.get("step")) is int:
            values.append((record["step"], float(terms["occupancy"])))
    if len(values) != STEPS:
        raise ValueError(f"{study} does not log the occupancy term at every declared step")
    return {"steps": len(values), "first": round(values[0][1], 6), "last_step": values[-1][0],
            "last": round(values[-1][1], 6), "mean": round(sum(v for _, v in values)/len(values), 6),
            "source": str(log.relative_to(ROOT)), "in_sample": True,
            "reading": "the occupancy loss the arm paid; the read adds no term of its own - one score is formed from "
                       "two channels of evidence"}


def row_error_at_zero(result):
    config = result["config"]
    rows, columns = config["visible_rows"], config["columns"]
    errors = []
    for case in result["cases"]:
        if case["split"] != "training" or not any(case["occupied"]):
            continue
        decided, occupied = boundaries(case["logits"], case["occupied"], rows, columns, 0.0)
        errors.extend(got-want for got, want in zip(decided, occupied) if got is not None and want is not None)
    if not errors:
        raise ValueError("row diagnostic carried no columns")
    return {"mean": round(sum(errors)/len(errors), 6), "columns": len(errors)}


def exact_boards_at_zero(result):
    """Strict (V-139): every visible cell right, and every column's top right with an empty column compared."""
    config = result["config"]
    rows, columns = config["visible_rows"], config["columns"]
    exact = boundary = total = 0
    for case in result["cases"]:
        if case["split"] != "training" or not any(case["occupied"]):
            continue
        decided, occupied = boundaries(case["logits"], case["occupied"], rows, columns, 0.0)
        total += 1
        cells = [(value > 0.0) == bool(label) for value, label in zip(case["logits"], case["occupied"])]
        exact += all(cells)
        boundary += all(got == want for got, want in zip(decided, occupied))
    if not total:
        raise ValueError("the fit gate carried no board")
    return {"boards": total, "exact": exact, "boundary_exact": boundary}


def channel_reading(cases, channel, config, grid):
    """Score one channel alone on the arm's own fold, under the arm's own declared transforms."""
    rows, columns = config["visible_rows"], config["columns"]
    pseudo = []
    for case in cases:
        if case["split"] != "training":
            continue
        values = torch.tensor(case["logits_parts"][channel], dtype=torch.float32).reshape(1, rows, columns)
        if channel == "lattice":
            # The lattice part still gets the arm's declared transforms: it is the envelope arm's read.
            values = stack_contiguous(values, rows)
            if config.get("occupancy_centre"):
                values = centre_board(values, rows)
            if config.get("occupancy_crossing_threshold"):
                values = otsu_board(values, rows)
        pseudo.append({"split": "training", "logits": [float(value) for value in values.flatten()],
                       "occupied": case["occupied"]})
    summary = evaluate(pseudo, grid)
    row = next(item for item in summary["rows"] if item["threshold"] == 0.0)
    counts = {"true_positive": row["true_positive"], "false_positive": row["false_positive"],
              "false_negative": row["false_negative"]}
    # Scored from the grid row at the declared boundary rather than through the arm summary, because a channel on a
    # different scale can legitimately decide nothing positive at 0.0 - which is itself diagnostic, not an error.
    return {"auc": summary["auc"], "f1_at_zero": row["f1"], "best_f1": summary["best"]["f1"],
            "operating_counts": counts, "precision": row["precision"], "recall": row["recall"],
            "positives_at_zero": row["true_positive"]+row["false_positive"]}


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
    previous_scope = read(PREVIOUS)
    span_scope = read(SPAN)
    level_scope = read(LEVEL)
    if prior.get("instrument") != "m10-supervised-anchor-r1":
        raise ValueError("the reproduction reference is not the committed extremal-anchor block")
    if span_scope.get("instrument") != "m18-tap-span-r1" or level_scope.get("instrument") != "m19-span-level-r1":
        raise ValueError("the excluded blocks are not the committed span records")
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
            "occupancy_lattice_free": summary["config"]["occupancy_lattice_free"],
            "occupancy_stack_monotone": summary["config"]["occupancy_stack_monotone"],
            "weights_sha256": result["weights_sha256"],
            "config": {key: summary["config"][key] for key in RECORDED_CONFIG},
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
            "exact_boards_at_zero": exact_boards_at_zero(result),
            "row_error_at_zero": row_error_at_zero(result),
            "training_occupancy": training_occupancy(study),
            "verdict": result["verdict"],
        }
        reproduction[name] = {"auc": result["deciding"]["auc"], "f1_at_zero": scored["per_cell_f1"],
                              "prior_scope": str(PRIOR.relative_to(ROOT))}
    if len({arms[name]["catalogue_sha256"] for name in arms}) != 1:
        raise ValueError("the arms were captured under different catalogues")
    control, treated = arms[CONTROL], arms[TREATED]
    if control["auc"] != expected_auc or control["f1_at_zero"] != expected_f1:
        raise ValueError(f"control does not reproduce the recorded {expected_auc!r} / {expected_f1!r}")

    differing = sorted(key for key in control["config"] if control["config"][key] != treated["config"][key])
    if differing != sorted(DECLARED):
        raise ValueError(f"the arms differ in more than the declared fields: {differing}")
    if treated["weights_sha256"] == control["weights_sha256"]:
        raise ValueError("the two arms scored identical weights, so one of them did not measure its own run")
    left = next(name for name in previous_scope["arms"] if name != "control")
    for label, scope in (("neighbour block", previous_scope), ("span-only block", span_scope),
                         ("span-under-level block", level_scope)):
        for arm_name, arm in scope["arms"].items():
            if treated["weights_sha256"] == arm["weights_sha256"]:
                raise ValueError(f"the two-path arm scored the {label}'s checkpoint ({arm_name})")

    # The channel split: the sum against its own parts, on the arm's own fold and weights.
    sweep = read(ROOT/"runs"/(f"{FAMILY}-{RUNS[TREATED]}-r1-sweep-r1")/"result.json")
    cases, split_config = sweep["cases"], sweep["config"]
    summed = channel_reading([{**case, "logits_parts": {"free": case["logits"], "lattice": case["logits"]}}
                              for case in cases], "free", split_config, grid)
    lattice_only = channel_reading(cases, "lattice", split_config, grid)
    free_only = channel_reading(cases, "free", split_config, grid)
    if round(summed["auc"], 6) != round(treated["auc"], 6):
        raise ValueError("the channel split did not re-derive the summed read's AUC")
    if lattice_only["auc"]-free_only["auc"] >= CHANNEL_GAP and summed["auc"]-lattice_only["auc"] < CHANNEL_GAIN:
        channel_verdict = "one-channel-noise-dominates"
        channel_text = ("The free channel is the weaker read on its own and adding it did not lift the sum above the "
                        "lattice-sampled read: the weaker channel's noise is diluting the sum. The next lever is to "
                        "condition that channel - a registration-aware gate - rather than to add more evidence to it.")
    elif summed["auc"]-lattice_only["auc"] >= CHANNEL_GAIN:
        channel_verdict = "complementary"
        channel_text = ("The sum ranks cells better than its own lattice-sampled part on the same fold: the two "
                        "channels carry information on different axes, exactly as declared.")
    else:
        channel_verdict = "no-complementary-signal"
        channel_text = ("Both channels are individually informative and summing them does not rank better than the "
                        "lattice-sampled part alone: the two reads are redundant at the cell level, so the next lever "
                        "is not the read but the image - resolution and capture - i.e. the population.")
    if channel_verdict not in CHANNEL_VERDICTS:
        raise ValueError("an undeclared channel verdict was produced")

    delta = round(treated["f1_at_zero"]-control["f1_at_zero"], 6)
    auc_delta = round(treated["auc"]-control["auc"], 6)
    gate_moved = treated["exact_boards_at_zero"]["exact"] > 0
    ceiling = control["f1_at_zero"]
    if delta >= ATTRIBUTION_DELTA and gate_moved:
        reading = "two-path-passes"
        reading_text = ("The two-path read clears the published read's primary by the standing attribution floor and "
                        "puts a genuinely exact board on the gate: summing evidence from two disjoint error axes was "
                        "worth more than either channel alone could deliver.")
    elif delta > 0:
        reading = "two-path-moves-without-clearing"
        reading_text = ("The two-path read beats the published read at the declared boundary but does not clear the "
                        "attribution floor together with the gate: part of the two channels' evidence survives the "
                        "sum, not enough to count as a win.")
    else:
        reading = "two-path-fails"
        reading_text = ("The two-path read does not beat the published read at the declared boundary. The sum "
                        "hypothesis is falsified on this population, and the channel split says which part of it "
                        "failed; no gate weight, envelope variant or summing coefficient is offered in its place.")
    if reading not in READINGS:
        raise ValueError("an undeclared reading was produced")
    secondary = "ranking-preserved" if auc_delta >= -AUC_TOLERANCE else "ranking-degraded"
    envelope_primary = read(ENVELOPE)["arms"]["contiguous"]["f1_at_zero"]
    scope = {
        "status": "passed",
        "instrument": INSTRUMENT,
        "predeclaration": PREDECLARATION,
        "scope": ("The published read's own 64-step control against a 64-step arm whose only declared difference is "
                  "the summed two-path read - the lattice-free channel added to the lattice-sampled read, under the "
                  "M15 monotone envelope - captured in one catalogue and scored read-only at the standing 0.0 "
                  "boundary and over the standing 81-point grid. Both parts are structural and label-free: no "
                  "threshold, no fitted parameter, no prevalence, no statistic of the population, and no swept "
                  "constant. With both fields at 0 the model is exactly the published function. No population change "
                  "and no real-image, complete-state or quality claim."),
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "mechanism": {
            "name": "two-path-read",
            "flag": "--occupancy-detail 1 --occupancy-stack-monotone 1 --occupancy-lattice-free 2",
            "definition": ("one per-cell score is the sum of the lattice-sampled read and the gated lattice-free "
                           "channel, then the M15 monotone envelope places the crossing"),
            "where": "model head, before the read-out's transforms and before every consumer",
            "parameters_added": 0, "tensors_added": 0, "loss_terms_added": 0,
            "weight": None, "margin": None,
            "statistic": "none: both channels read the image and the board's own fixed coordinates; nothing about "
                         "the population is read",
            "gate": "one learned scalar, tanh(gate), starting at exactly zero so step zero is the published function; "
                    "never swept and never set by hand",
            "error_axes": {"lattice_path": "per-cell sampling error inherited from a misplaced decoded lattice "
                                           "(placement error 1.189 rows even in its best arm)",
                           "free_channel": "global registration error and lost local detail, with no dependence on "
                                           "the predicted lattice - the only read that has produced a 2-wrong-cell "
                                           "board"},
            "declared_before_receipts": True,
            "identity_at_zero": "both fields 0 is exactly the published function; the gate at zero makes the sum "
                                "reduce exactly to the lattice-sampled envelope arm",
            "composition": "the two channels are declared together or not at all: lattice_free 2 requires the "
                           "envelope in config validation and in the parser, and the single-channel forms remain "
                           "their own arms",
            "non_degeneracy": ("tests assert each channel moves the summed read, that a zeroed gate reduces the sum "
                               "exactly to the lattice-sampled read, and that zeroing the read-out reduces it "
                               "exactly to the free channel - so the two channels are not mutually inert and the "
                               "arm's readings cannot be produced by the control's geometry"),
            "expected_reading": "two-path-moves-without-clearing",
            "label_free": True,
            "label_free_proof": "both channels are functions of the image, the board's fixed coordinates and the "
                                "gate scalar; tests permute the labels and require bit-identical output and scan the "
                                "executable body for any label, target, occupancy-mask, prevalence or threshold "
                                "reference",
            "falsification": ("the arm is falsified if the primary does not exceed the control's or no strict exact "
                              "board appears; the mechanism is falsified if the channel split is not "
                              "`complementary`"),
            "failure_mode": "recorded as a channel verdict, never as a number that merely went down",
        },
        "envelope": {"steps": STEPS, "ceiling_bytes": CEILING_BYTES,
                     "work_items": {"sweep": FORWARDS, "curation": 9},
                     "attribution_floor": ATTRIBUTION_DELTA, "auc_tolerance": AUC_TOLERANCE},
        "arms": arms,
        "channels": {"summed": summed, "lattice_only": lattice_only, "free_only": free_only,
                     "verdict": channel_verdict, "verdict_text": channel_text,
                     "gain_over_lattice_auc": round(summed["auc"]-lattice_only["auc"], 6),
                     "lattice_minus_free_auc": round(lattice_only["auc"]-free_only["auc"], 6),
                     "declared_bar": {"channel_gain": CHANNEL_GAIN, "channel_gap": CHANNEL_GAP},
                     "fold": "the arm's own training fold, its own weights, the standing 81-point grid"},
        "contrasts": {
            "two_path_control": {"f1_at_zero_delta": delta, "auc_delta": auc_delta,
                                 "precision_delta": round(treated["precision"]-control["precision"], 6),
                                 "best_f1_delta": round(treated["best"]["f1"]-control["best"]["f1"], 6),
                                 "best_threshold_delta": round(treated["best"]["threshold"]
                                                               - control["best"]["threshold"], 6),
                                 "out_of_fold_delta": round(treated["out_of_fold_f1"]
                                                            - control["out_of_fold_f1"], 6),
                                 "positive_share_delta": round(treated["positive_share_at_zero"]
                                                               - control["positive_share_at_zero"], 6),
                                 "row_error_delta": round(treated["row_error_at_zero"]["mean"]
                                                          - control["row_error_at_zero"]["mean"], 6),
                                 "occupancy_loss_delta": round(treated["training_occupancy"]["mean"]
                                                               - control["training_occupancy"]["mean"], 6)},
            "incremental_over_envelope": {"f1_at_zero_delta": round(treated["f1_at_zero"]-envelope_primary, 6),
                                          "envelope_primary": envelope_primary,
                                          "scope": str(ENVELOPE.relative_to(ROOT))},
        },
        "aliasing_guard": {
            "arms_differ_in": differing,
            "weights_differ": treated["weights_sha256"] != control["weights_sha256"],
            "not_the_previous_block": all(treated["weights_sha256"] != arm["weights_sha256"]
                                          for arm in previous_scope["arms"].values()),
            "not_the_span_only_arm": all(treated["weights_sha256"] != arm["weights_sha256"]
                                         for arm in span_scope["arms"].values()),
            "not_the_span_under_level_arm": all(treated["weights_sha256"] != arm["weights_sha256"]
                                                for arm in level_scope["arms"].values()),
            "rule": "the read is formed from two channels, so the block proves it measured its own checkpoint: each "
                    "sweep's weights equal its own study's checkpoint, the arms' weights differ, the neighbour, "
                    "span-only and span-under-level checkpoints are excluded by name, and the configs differ in "
                    "exactly the declared fields",
        },
        "reference": {
            "span_under_level": reference(LEVEL, "composed"),
            "span_only": reference(SPAN, "tap"),
            "skyline_recovery": reference(SKYLINE, "recovered"),
            "crossing_rule": reference(RULED, "ruled"),
            "combined_read": reference(PREVIOUS, left),
            "envelope": reference(ENVELOPE, "contiguous"),
            "centring": reference(CENTRING, "centred"),
            "midpoint_anchor": reference(MIDPOINT, "level"),
            "dense_hinge": reference(DENSE, "banded"),
            "quantile_anchor": reference(QUANTILE, "bulk"),
            "extremal_anchor": reference(PRIOR, "anchor"),
            "hinge": reference(HINGE, "hinge"),
            "role": "reference-only: the M19 span-under-level arm, the M18 span-only arm, the M17 skyline arm, the "
                    "M16 crossing rule, the M15 combined read, the M15 envelope arm, the M14 centred read, the M13 "
                    "midpoint anchor, the M12 dense hinge, the M11 quantile anchor, the M10 extremal anchor and the "
                    "M9 hinge, each measured in its own catalogue and none re-measured or re-verdicted here",
        },
        "reading": reading,
        "reading_text": reading_text,
        "secondary_reading": secondary,
        "secondary_text": ("Deciding AUC with the standing -0.02 tolerance. A read formed from two channels can change "
                           "the ranking, so a ranking change here is the read's own doing and is reported as the "
                           "measured price or gain."),
        "fixed_threshold_ceiling": {"arm": CONTROL, "scope": "this block's reproduced published read",
                                    "prior_scope": str(PRIOR.relative_to(ROOT)),
                                    "f1_at_zero": ceiling, "two_path_primary": treated["f1_at_zero"]},
        "ceiling_broken": treated["f1_at_zero"] > ceiling,
        "gate_moved": gate_moved,
        "neighbour_excluded": {"arm": left, "scope": str(PREVIOUS.relative_to(ROOT))},
        "verdict": lever_verdict(treated["f1_at_zero"], treated["verdict"], delta),
        "gates": {"per_cell_f1": CELL_F1_THRESHOLD, "ranking_auc": AUC_THRESHOLD,
                  "ranking_f1": F1_THRESHOLD, "attribution_floor": ATTRIBUTION_DELTA, "grid_points": 81},
        "fit_gate": {"exact_nonempty_boards": treated["exact_boards_at_zero"]["exact"],
                     "boundary_exact_nonempty_boards": treated["exact_boards_at_zero"]["boundary_exact"],
                     "boards": treated["exact_boards_at_zero"]["boards"],
                     "cleared": (treated["exact_boards_at_zero"]["exact"] == BOARDS),
                     "boundary": 0.0, "measured_on": TREATED},
        "reproduction": reproduction,
        "population": {"split": "training", "images": DECIDING_IMAGES, "boards": BOARDS,
                       "occupied_cells": OCCUPIED_CELLS, "known_cells": KNOWN_CELLS,
                       "prevalence": round(OCCUPIED_CELLS/KNOWN_CELLS, 6)},
    }
    args.output.mkdir(parents=True)
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(scope, stream, indent=2, allow_nan=False)
    print(json.dumps({key: scope[key] for key in ("reading", "secondary_reading", "channels", "contrasts",
                                                  "aliasing_guard", "verdict", "fit_gate")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
