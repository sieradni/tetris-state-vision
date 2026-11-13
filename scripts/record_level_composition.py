"""Curate the level composition: the two-path read with and without the envelope, and with the crossing rule.

`docs/M21_LEVEL_COMPOSITION.md` declares four arms in one catalogue and the bar each is judged by before any receipt.
This recorder asserts each arm's declared fields and nothing else, that every sweep carries its own study's checkpoint,
that the control reproduces the published numbers exactly, scores the four arms at the standing boundary and over the
standing grid, rebuilds each arm's read stage by stage from its recorded parts (raw sum -> monotone envelope -> crossing
rule) to name the stage that moves the level, and scores the summed read, the lattice part and the free channel separately
for every arm that carries the free channel.
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
FAMILY = "m21-level"
INSTRUMENT = "m21-level-composition-r1"
PREDECLARATION = "docs/M21_LEVEL_COMPOSITION.md"
PRIOR = ROOT/"docs/evidence/m10-supervised-anchor-r1/scope.json"
TWO_PATH = ROOT/"docs/evidence/m20-two-path-r1/scope.json"
ENVELOPE = ROOT/"docs/evidence/m15-stack-contiguity-r1/scope.json"
RULED = ROOT/"docs/evidence/m16-crossing-threshold-r1/scope.json"
LEVEL = ROOT/"docs/evidence/m19-span-level-r1/scope.json"
COMBINED = ROOT/"docs/evidence/m15-stack-combined-r1/scope.json"
CENTRING = ROOT/"docs/evidence/m14-read-centring-r1/scope.json"
PRIOR_SCOPE = ROOT/"docs/evidence/m13-midpoint-r1/scope.json"
DECLARED = ("occupancy_lattice_free", "occupancy_stack_monotone", "occupancy_crossing_threshold")
ARMS = {
    "control": {"occupancy_lattice_free": 0, "occupancy_stack_monotone": 0,
                "occupancy_crossing_threshold": 0},
    "envelope": {"occupancy_lattice_free": 2, "occupancy_stack_monotone": 1,
                 "occupancy_crossing_threshold": 0},
    "crossing": {"occupancy_lattice_free": 2, "occupancy_stack_monotone": 1,
                 "occupancy_crossing_threshold": 1},
    "bare": {"occupancy_lattice_free": 2, "occupancy_stack_monotone": 0,
             "occupancy_crossing_threshold": 0},
}
CONTROL = "control"
REQUESTED = "crossing"
DECOUPLED = "bare"
COMPOSITIONS = ("control", "envelope", "crossing", "bare")
READINGS = ("level-passes", "level-without-ranking", "level-fails")
DECOUPLED_READINGS = ("decoupling-passes", "decoupling-with-ranking-price", "decoupling-fails")
CHANNEL_VERDICTS = ("complementary", "one-channel-noise-dominates", "no-complementary-signal")
SUPERVISION = ("occupancy_midpoint", "occupancy_dense_hinge", "occupancy_quantile_anchor", "occupancy_anchor",
               "occupancy_margin_hinge")
READ_LEVERS = ("occupancy_normalise", "occupancy_offset_rows", "occupancy_deform", "occupancy_context",
               "occupancy_detail_stencil", "occupancy_resolution", "occupancy_column", "occupancy_placed",
               "occupancy_centre", "occupancy_skyline_recovery", "occupancy_tap_span", "occupancy_lattice_board_refine",
               "occupancy_tap_span")
RECORDED_CONFIG = ("occupancy_lattice_free", "occupancy_stack_monotone", "occupancy_crossing_threshold",
                   "occupancy_detail", "occupancy_normalise", "occupancy_centre", "occupancy_skyline_recovery",
                   "occupancy_tap_span", "dimension", "component_capacity")


def study_of(name):
    return f"{FAMILY}-{name}-r1"


def require_sweep(name):
    """One arm's sweep is the declared measurement of its own run, under the declared budget and grid."""
    study = study_of(name)
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
    for key, declared in ARMS[name].items():
        for part in ("config", "arguments", "recovery_contract"):
            if summary.get(part, {}).get(key) != declared:
                raise ValueError(f"{name} run does not carry {key}={declared} in {part}")
    if summary.get("config", {}).get("occupancy_detail") != 1:
        raise ValueError(f"{name} run does not carry the published detail tap")
    for key in SUPERVISION+READ_LEVERS:
        if summary.get("config", {}).get(key, 0) or summary.get("arguments", {}).get(key, 0):
            raise ValueError(f"{name} carries an undeclared lever: {key}")
    if ARMS[name]["occupancy_lattice_free"] == 2:
        for case in result["cases"]:
            if "logits_parts" not in case or sorted(case["logits_parts"]) != ["free", "lattice"]:
                raise ValueError(f"{name} did not record both channel logits per case")
    return study, summary, result


def training_occupancy(study):
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
    return {"steps": len(values), "first": round(values[0][1], 6), "last": round(values[-1][1], 6),
            "mean": round(sum(v for _, v in values)/len(values), 6), "source": str(log.relative_to(ROOT)),
            "in_sample": True}


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


def chain(config):
    """The transforms the arm declares, in the order the model applies them after the sum."""
    steps = []
    if config.get("occupancy_stack_monotone"):
        steps.append("monotone")
    if config.get("occupancy_centre"):
        steps.append("centred")
    if config.get("occupancy_crossing_threshold"):
        steps.append("crossing")
    return steps


def apply_chain(values, config, rows):
    for step in chain(config):
        values = {"monotone": lambda v: stack_contiguous(v, rows),
                  "centred": lambda v: centre_board(v, rows),
                  "crossing": lambda v: otsu_board(v, rows)}[step](values)
    return values


def score(cases, grid, rows, columns, stages):
    """Score one read made of a list of per-case tensors; return the grid-row reading at the declared boundary."""
    pseudo = [{"split": "training", "logits": [float(value) for value in tensor.flatten()],
               "occupied": case["occupied"]} for case, tensor in zip(cases, stages)]
    summary = evaluate(pseudo, grid)
    row = next(item for item in summary["rows"] if item["threshold"] == 0.0)
    positives = row["true_positive"]+row["false_positive"]
    return {"auc": round(summary["auc"], 6), "f1_at_zero": round(row["f1"], 6),
            "positives_at_zero": positives,
            "share": round(positives/(len(cases)*rows*columns), 6),
            "precision": round(row["precision"], 6), "recall": round(row["recall"], 6),
            "best_f1": round(summary["best"]["f1"], 6),
            "best_threshold": summary["best"]["threshold"],
            "counts": {"true_positive": row["true_positive"], "false_positive": row["false_positive"],
                       "false_negative": row["false_negative"]}}


def base_of(case, rows, columns):
    """The sum of the recorded parts, or the published read for an arm that carries no free channel."""
    parts = case.get("logits_parts")
    if not parts:
        return torch.tensor(case["logits"], dtype=torch.float32).reshape(1, rows, columns)
    return (torch.tensor(parts["lattice"], dtype=torch.float32)
            + torch.tensor(parts["free"], dtype=torch.float32)).reshape(1, rows, columns)


def decomposition(result, grid):
    """Rebuild the read stage by stage and score each stage, so the level can be attributed to one transform."""
    config = result["config"]
    rows, columns = config["visible_rows"], config["columns"]
    cases = [case for case in result["cases"] if case["split"] == "training"]
    steps = chain(config)
    stages = [base_of(case, rows, columns) for case in cases]
    levels = []
    readings = [("raw_sum", score(cases, grid, rows, columns, stages))]
    for index, step in enumerate(steps, start=1):
        previous = stages
        stages = [{"monotone": lambda v: stack_contiguous(v, rows),
                   "centred": lambda v: centre_board(v, rows),
                   "crossing": lambda v: otsu_board(v, rows)}[step](tensor) for tensor in stages]
        if step == "crossing":
            levels = [float((before[0]-after[0]).mean()) for before, after in zip(previous, stages)]
        readings.append((step, score(cases, grid, rows, columns, stages)))
    chain_reading = [{"stage": name, **values} for name, values in readings]
    rises = [(later["share"]-earlier["share"], later["stage"]) for earlier, later
             in zip(chain_reading, chain_reading[1:])]
    worst = max(rises) if rises else (0.0, None)
    out = {"chain": [item for item in steps], "stages": chain_reading,
           "raising_stage": worst[1], "raising_share": round(worst[0], 6),
           "crossing_level": ({"boards": len(levels), "mean": round(sum(levels)/len(levels), 6),
                               "min": round(min(levels), 6), "max": round(max(levels), 6)}
                              if levels else None),
           "final_stage_matches_the_arm": (round(chain_reading[-1]["auc"], 6) == round(result["deciding"]["auc"], 6))}
    if not out["final_stage_matches_the_arm"]:
        raise ValueError("the rebuilt chain does not reproduce the arm's own read")
    return out


def channel_readings(result, grid):
    """The summed read against its own parts, each under that arm's own transforms."""
    config = result["config"]
    rows, columns = config["visible_rows"], config["columns"]
    cases = [case for case in result["cases"] if case["split"] == "training"]
    summed_base = [base_of(case, rows, columns) for case in cases]
    summed = score(cases, grid, rows, columns, [apply_chain(value, config, rows) for value in summed_base])
    lattice = score(cases, grid, rows, columns,
                    [apply_chain(torch.tensor(case["logits_parts"]["lattice"], dtype=torch.float32)
                                 .reshape(1, rows, columns), config, rows) for case in cases])
    free = score(cases, grid, rows, columns,
                 [torch.tensor(case["logits_parts"]["free"], dtype=torch.float32).reshape(1, rows, columns)
                  for case in cases])
    if summed["auc"] != round(result["deciding"]["auc"], 6):
        raise ValueError("the channel split did not re-derive the summed read's AUC")
    if lattice["auc"]-free["auc"] >= CHANNEL_GAP and summed["auc"]-lattice["auc"] < CHANNEL_GAIN:
        verdict = "one-channel-noise-dominates"
    elif summed["auc"]-lattice["auc"] >= CHANNEL_GAIN:
        verdict = "complementary"
    else:
        verdict = "no-complementary-signal"
    if verdict not in CHANNEL_VERDICTS:
        raise ValueError("an undeclared channel verdict was produced")
    return {"summed": summed, "lattice_only": lattice, "free_only": free, "verdict": verdict,
            "gain_over_lattice_auc": round(summed["auc"]-lattice["auc"], 6),
            "lattice_minus_free_auc": round(lattice["auc"]-free["auc"], 6),
            "declared_bar": {"channel_gain": CHANNEL_GAIN, "channel_gap": CHANNEL_GAP}}


def reference(path, name):
    arm = read(path)["arms"][name]
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
    arms, decompositions, channels = {}, {}, {}
    for name in COMPOSITIONS:
        study, summary, result = require_sweep(name)
        scored = arm_summary(result["deciding"])
        counts = scored["operating_counts"]
        positives = counts["true_positive"]+counts["false_positive"]
        arms[name] = {
            "study": study, "sweep": f"{study}-sweep-r1",
            "declared": ARMS[name],
            "weights_sha256": result["weights_sha256"],
            "config": {key: summary["config"][key] for key in RECORDED_CONFIG},
            "catalogue_sha256": summary["catalogue_sha256"],
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
                     "f1": result["deciding"]["best"]["f1"]},
            "out_of_fold_f1": components_of([case for case in result["cases"]
                                             if case["split"] == "training"], grid)["out_of_fold_f1"],
            "exact_boards_at_zero": exact_boards_at_zero(result),
            "row_error_at_zero": row_error_at_zero(result),
            "training_occupancy": training_occupancy(study),
            "sweep_verdict": result["verdict"],
        }
        decompositions[name] = decomposition(result, grid)
        if ARMS[name]["occupancy_lattice_free"] == 2:
            channels[name] = channel_readings(result, grid)
    if len({arms[name]["catalogue_sha256"] for name in arms}) != 1:
        raise ValueError("the arms were captured under different catalogues")
    control = arms[CONTROL]
    if control["auc"] != expected_auc or control["f1_at_zero"] != expected_f1:
        raise ValueError(f"control does not reproduce the recorded {expected_auc!r} / {expected_f1!r}")
    for name, arm in arms.items():
        differing = sorted(key for key in DECLARED if arm["config"][key] != control["config"][key])
        if differing != sorted(key for key in DECLARED if ARMS[name][key] != ARMS[CONTROL][key]):
            raise ValueError(f"{name} differs from the control in more than its declared fields")
        if name != CONTROL and arm["weights_sha256"] == control["weights_sha256"]:
            raise ValueError(f"{name} scored the control's weights")
    if len({arm["weights_sha256"] for arm in arms.values()}) != len(arms):
        raise ValueError("two arms scored the same checkpoint")

    def reading_of(name, readings):
        arm = arms[name]
        delta = round(arm["f1_at_zero"]-control["f1_at_zero"], 6)
        auc_delta = round(arm["auc"]-control["auc"], 6)
        if delta > 0 and auc_delta >= -AUC_TOLERANCE:
            return readings[0], delta, auc_delta
        if delta > 0:
            return readings[1], delta, auc_delta
        return readings[2], delta, auc_delta

    reading, requested_delta, requested_auc_delta = reading_of(REQUESTED, READINGS)
    decoupled_reading, decoupled_delta, decoupled_auc_delta = reading_of(DECOUPLED, DECOUPLED_READINGS)
    asked = decompositions[REQUESTED]
    bare = decompositions[DECOUPLED]
    envelope = arms["envelope"]

    diagnosis = {
        "requested_arm": REQUESTED,
        "raising_stage": asked["raising_stage"],
        "raising_share": asked["raising_share"],
        "crossing_stage": next((item for item in asked["stages"] if item["stage"] == "crossing"), None),
        "monotone_stage": next((item for item in asked["stages"] if item["stage"] == "monotone"), None),
        "bare_raising_stage": bare["raising_stage"],
        "envelope_raising_stage": decompositions["envelope"]["raising_stage"],
        "control_raising_stage": decompositions[CONTROL]["raising_stage"],
    }
    control_note = ("the published read carries no transform, so it has no stage to raise it"
                    if diagnosis["control_raising_stage"] is None else
                    f"the published read (`{diagnosis['control_raising_stage']}`)")
    text = (f"The stage that raises the positive share in the requested arm is `{asked['raising_stage']}` "
            f"(`{asked['raising_share']:+.6f}` share points); the crossing stage moves the share by "
            f"`{(diagnosis['crossing_stage'] or {}).get('share', 0)-(diagnosis['monotone_stage'] or {}).get('share', 0):+.6f}` "
            f"against the monotone stage and its F1 at `0.0` by "
            f"`{(diagnosis['crossing_stage'] or {}).get('f1_at_zero', 0)-(diagnosis['monotone_stage'] or {}).get('f1_at_zero', 0):+.6f}`. "
            f"The envelope arm reproduces the same raising stage (`{diagnosis['envelope_raising_stage']}`), and {control_note}; "
            f"the level movement therefore belongs to the transform rather than to the added channel.")

    contrasts = {}
    for name, arm in arms.items():
        if name == CONTROL:
            continue
        contrasts[f"{name}_vs_control"] = {
            "f1_at_zero_delta": round(arm["f1_at_zero"]-control["f1_at_zero"], 6),
            "auc_delta": round(arm["auc"]-control["auc"], 6),
            "best_f1_delta": round(arm["best"]["f1"]-control["best"]["f1"], 6),
            "best_threshold_delta": round(arm["best"]["threshold"]-control["best"]["threshold"], 6),
            "out_of_fold_delta": round(arm["out_of_fold_f1"]-control["out_of_fold_f1"], 6),
            "positive_share_delta": round(arm["positive_share_at_zero"]-control["positive_share_at_zero"], 6),
            "row_error_delta": round(arm["row_error_at_zero"]["mean"]-control["row_error_at_zero"]["mean"], 6),
            "occupancy_loss_delta": round(arm["training_occupancy"]["mean"]
                                          - control["training_occupancy"]["mean"], 6)}
    contrasts["bare_vs_envelope"] = {
        "f1_at_zero_delta": round(arms[DECOUPLED]["f1_at_zero"]-envelope["f1_at_zero"], 6),
        "auc_delta": round(arms[DECOUPLED]["auc"]-envelope["auc"], 6),
        "positive_share_delta": round(arms[DECOUPLED]["positive_share_at_zero"]
                                      - envelope["positive_share_at_zero"], 6),
        "row_error_delta": round(arms[DECOUPLED]["row_error_at_zero"]["mean"]
                                 - envelope["row_error_at_zero"]["mean"], 6),
        "out_of_fold_delta": round(arms[DECOUPLED]["out_of_fold_f1"]-envelope["out_of_fold_f1"], 6)}

    scope = {
        "status": "passed",
        "instrument": INSTRUMENT,
        "predeclaration": PREDECLARATION,
        "scope": ("Four 64-step arms in one catalogue: the published read, the M20 two-path read under the M15 monotone "
                  "envelope, the same read under the crossing rule, and the same sum without the envelope. Every arm is "
                  "scored read-only at the standing 0.0 boundary and over the standing 81-point grid, and every arm's read "
                  "is rebuilt stage by stage from its recorded parts. All mechanisms are structural and label-free."),
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "arms": arms,
        "decomposition": decompositions,
        "channels": channels,
        "contrasts": contrasts,
        "reading": reading,
        "requested_delta": requested_delta,
        "requested_auc_delta": requested_auc_delta,
        "decoupled_reading": decoupled_reading,
        "decoupled_delta": decoupled_delta,
        "decoupled_auc_delta": decoupled_auc_delta,
        "diagnosis": diagnosis,
        "diagnosis_text": text,
        "declaration_of_change": ("M20 refused the sum without the envelope in config validation and the parser; that "
                                  "coupling was declared before the envelope's effect on the level was measured. The "
                                  "coupling is relaxed to an option so the two mechanisms can be separated, and the M20 "
                                  "arm is re-measured unchanged as `envelope`."),
        "aliasing_guard": {
            "arms_differ_in": list(DECLARED),
            "weights_all_distinct": len({arm["weights_sha256"] for arm in arms.values()}) == len(arms),
            "reproduces_the_m20_checkpoint": sorted(
                name for name, arm in arms.items()
                if arm["weights_sha256"] == read(TWO_PATH)["arms"]["two-path"]["weights_sha256"]),
            "rule": "each sweep's weights equal its own study's checkpoint and the four arms' weights are all distinct. "
                    "The arm whose declared config equals M20's reproduces the M20 checkpoint bit-for-bit, which is this "
                    "block's determinism check on the same code path; every other arm differs from it and from the control "
                    "in exactly its declared fields",
        },
        "reference": {
            "two_path": reference(TWO_PATH, "two-path"),
            "envelope": reference(ENVELOPE, "contiguous"),
            "crossing_rule": reference(RULED, "ruled"),
            "span_under_level": reference(LEVEL, "composed"),
            "combined_read": reference(COMBINED, next(name for name in read(COMBINED)["arms"]
                                                      if name != "control")),
            "centring": reference(CENTRING, "centred"),
            "midpoint_anchor": reference(PRIOR_SCOPE, "level"),
            "role": "reference-only, each measured in its own catalogue and none re-measured or re-verdicted here",
        },
        "gates": {"per_cell_f1": CELL_F1_THRESHOLD, "ranking_auc": AUC_THRESHOLD,
                  "ranking_f1": F1_THRESHOLD, "attribution_floor": ATTRIBUTION_DELTA,
                  "auc_tolerance": AUC_TOLERANCE, "grid_points": 81},
        "verdict": {name: lever_verdict(arm["f1_at_zero"], arm["sweep_verdict"],
                                        round(arm["f1_at_zero"]-control["f1_at_zero"], 6))
                    for name, arm in arms.items()},
        "population": {"split": "training", "images": DECIDING_IMAGES, "boards": BOARDS,
                       "occupied_cells": OCCUPIED_CELLS, "known_cells": KNOWN_CELLS,
                       "prevalence": round(OCCUPIED_CELLS/KNOWN_CELLS, 6)},
    }
    args.output.mkdir(parents=True)
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(scope, stream, indent=2, allow_nan=False, default=str)
    print(json.dumps({key: scope[key] for key in ("reading", "requested_delta", "requested_auc_delta",
                                                  "decoupled_reading", "decoupled_delta", "decoupled_auc_delta",
                                                  "diagnosis_text")}, indent=2, default=str))


if __name__ == "__main__":
    main()
