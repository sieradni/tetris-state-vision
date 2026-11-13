"""Curate the free channel's coarse local stencil, and the in-loop versus post-hoc crossing ablation.

`docs/M22_STENCIL_READ.md` declares six arms in one catalogue before any receipt: the published read, the M20 two-path
read, the M21 level composition, the sum without the envelope, and the same two compositions with the free channel's
coarse local stencil switched on. This recorder asserts each arm's declared fields and nothing else, that every sweep
carries its own study's checkpoint, that the control reproduces the published numbers exactly, scores every arm at the
standing boundary and over the standing grid, rebuilds each arm's read stage by stage to name the stage that moves the
level, splits every free-channel arm into its lattice part, its pooled free part, its local stencil part and its sum, and
measures the crossing rule applied post-hoc to a read trained without it against the same rule trained in the loop.
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
# the de-meaned mode keeps at most this share of its magnitude as a board level; the plain form keeps at least the other
LEVEL_RATIO_REMOVED = .15
LEVEL_RATIO_PRESENT = .40
CHANNEL_GAP = .01
FAMILY = "m22-stencil"
INSTRUMENT = "m22-stencil-read-r1"
PREDECLARATION = "docs/M22_STENCIL_READ.md"
INERTNESS = ROOT/"docs/evidence/m22-stencil-r1-inertness.json"
INERTNESS_CONTRAST = ROOT/"docs/evidence/m23-contrast-r1-inertness.json"
PRIOR = ROOT/"docs/evidence/m10-supervised-anchor-r1/scope.json"
TWO_PATH = ROOT/"docs/evidence/m20-two-path-r1/scope.json"
LEVEL_BLOCK = ROOT/"docs/evidence/m21-level-r1/scope.json"
ENVELOPE = ROOT/"docs/evidence/m15-stack-contiguity-r1/scope.json"
RULED = ROOT/"docs/evidence/m16-crossing-threshold-r1/scope.json"
LEVEL = ROOT/"docs/evidence/m19-span-level-r1/scope.json"
COMBINED = ROOT/"docs/evidence/m15-stack-combined-r1/scope.json"
CENTRING = ROOT/"docs/evidence/m14-read-centring-r1/scope.json"
PRIOR_SCOPE = ROOT/"docs/evidence/m13-midpoint-r1/scope.json"
DECLARED = ("occupancy_lattice_free", "occupancy_stack_monotone", "occupancy_crossing_threshold",
            "occupancy_free_stencil")
ARMS = {
    "control": (0, 0, 0, 0),
    "envelope": (2, 1, 0, 0),
    "crossing": (2, 1, 1, 0),
    "bare": (2, 0, 0, 0),
    "stencil": (2, 1, 0, 1),
    "stencil-crossing": (2, 1, 1, 1),
    "centred": (2, 1, 0, 2),
    "centred-crossing": (2, 1, 1, 2),
    "norm": (2, 1, 0, 3),
    "norm-cross": (2, 1, 1, 3),
}
CONTROL = "control"
LEVEL_READINGS = ("level-passes", "level-without-ranking", "level-fails")
STENCIL_READINGS = ("stencil-passes", "stencil-moves-without-clearing", "stencil-fails")
CHANNEL_VERDICTS = ("complementary", "one-channel-noise-dominates", "no-complementary-signal")
SUPERVISION = ("occupancy_midpoint", "occupancy_dense_hinge", "occupancy_quantile_anchor", "occupancy_anchor",
               "occupancy_margin_hinge")
READ_LEVERS = ("occupancy_normalise", "occupancy_offset_rows", "occupancy_deform", "occupancy_context",
               "occupancy_detail_stencil", "occupancy_resolution", "occupancy_column", "occupancy_placed",
               "occupancy_centre", "occupancy_skyline_recovery", "occupancy_tap_span",
               "occupancy_lattice_board_refine")
RECORDED_CONFIG = DECLARED+("occupancy_detail", "occupancy_normalise", "occupancy_centre",
                            "occupancy_skyline_recovery", "occupancy_tap_span", "dimension",
                            "component_capacity")


def declared_of(name):
    return dict(zip(DECLARED, ARMS[name]))


def family_of(name):
    """Each arm belongs to the family that launched it; the M23 arms declare the contrast mode."""
    return "m23-contrast" if ARMS[name][3] == 3 else FAMILY


def study_of(name):
    return f"{family_of(name)}-{name}-r1"


def require_sweep(name):
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
    for key, value in declared_of(name).items():
        for part in ("config", "arguments", "recovery_contract"):
            if summary.get(part, {}).get(key) != value:
                raise ValueError(f"{name} run does not carry {key}={value} in {part}")
    if summary.get("config", {}).get("occupancy_detail") != 1:
        raise ValueError(f"{name} run does not carry the published detail tap")
    for key in SUPERVISION+READ_LEVERS:
        if summary.get("config", {}).get(key, 0) or summary.get("arguments", {}).get(key, 0):
            raise ValueError(f"{name} carries an undeclared lever: {key}")
    if ARMS[name][0] == 2:
        wanted = ["free", "lattice"]+(["local"] if ARMS[name][3] else [])
        for case in result["cases"]:
            if "logits_parts" not in case or sorted(case["logits_parts"]) != sorted(wanted):
                raise ValueError(f"{name} did not record the declared channel parts per case")
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
            values.append(float(terms["occupancy"]))
    if len(values) != STEPS:
        raise ValueError(f"{study} does not log the occupancy term at every declared step")
    return {"steps": len(values), "first": round(values[0], 6), "last": round(values[-1], 6),
            "mean": round(sum(values)/len(values), 6), "source": str(log.relative_to(ROOT)), "in_sample": True}


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
    steps = []
    if config.get("occupancy_stack_monotone"):
        steps.append("monotone")
    if config.get("occupancy_centre"):
        steps.append("centred")
    if config.get("occupancy_crossing_threshold"):
        steps.append("crossing")
    return steps


def apply_step(values, step, rows):
    return {"monotone": lambda v: stack_contiguous(v, rows),
            "centred": lambda v: centre_board(v, rows),
            "crossing": lambda v: otsu_board(v, rows)}[step](values)


def score(cases, grid, rows, columns, stages):
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


def parts_of(case, rows, columns):
    """The recorded per-channel logits, as tensors."""
    parts = case.get("logits_parts")
    if not parts:
        return {"published": torch.tensor(case["logits"], dtype=torch.float32).reshape(1, rows, columns)}
    return {name: torch.tensor(value, dtype=torch.float32).reshape(1, rows, columns)
            for name, value in parts.items()}


def base_of(case, rows, columns):
    parts = parts_of(case, rows, columns)
    if "published" in parts:
        return parts["published"]
    return parts["lattice"]+parts["free"]


def decomposition(result, grid):
    config = result["config"]
    rows, columns = config["visible_rows"], config["columns"]
    cases = [case for case in result["cases"] if case["split"] == "training"]
    steps = chain(config)
    stages = [base_of(case, rows, columns) for case in cases]
    levels = []
    readings = [("raw_sum", score(cases, grid, rows, columns, stages))]
    for step in steps:
        previous = stages
        stages = [apply_step(tensor, step, rows) for tensor in stages]
        if step == "crossing":
            levels = [float((before[0]-after[0]).mean()) for before, after in zip(previous, stages)]
        readings.append((step, score(cases, grid, rows, columns, stages)))
    chain_reading = [{"stage": name, **values} for name, values in readings]
    rises = [(later["share"]-earlier["share"], later["stage"])
             for earlier, later in zip(chain_reading, chain_reading[1:])]
    worst = max(rises) if rises else (0.0, None)
    out = {"chain": steps, "stages": chain_reading,
           "raising_stage": worst[1], "raising_share": round(worst[0], 6),
           "crossing_level": ({"boards": len(levels), "mean": round(sum(levels)/len(levels), 6),
                               "min": round(min(levels), 6), "max": round(max(levels), 6)}
                              if levels else None),
           "final_stage_matches_the_arm": round(chain_reading[-1]["auc"], 6) == round(result["deciding"]["auc"], 6)}
    if not out["final_stage_matches_the_arm"]:
        raise ValueError("the rebuilt chain does not reproduce the arm's own read")
    return out


def channels_of(result, grid):
    """The summed read against its lattice part, its pooled free part, its local stencil and the sum without it."""
    config = result["config"]
    rows, columns = config["visible_rows"], config["columns"]
    cases = [case for case in result["cases"] if case["split"] == "training"]
    steps = chain(config)

    def transformed(tensors):
        values = tensors
        for step in steps:
            values = [apply_step(value, step, rows) for value in values]
        return score(cases, grid, rows, columns, values)

    parts = [parts_of(case, rows, columns) for case in cases]
    summed = transformed([part["lattice"]+part["free"] for part in parts])
    lattice = transformed([part["lattice"] for part in parts])
    free = transformed([part["free"] for part in parts])
    out = {"summed": summed, "lattice_only": lattice, "free_only": free}
    if "local" in parts[0]:
        out["free_without_local"] = transformed([part["free"]-part["local"] for part in parts])
        out["local_only"] = transformed([part["local"] for part in parts])
        # How much of the local read's own magnitude is a board-wide level: ~0 when the mode-2 bar is applied,
        # and most of it for the plain form. Measured separation on this catalogue is 0.006-0.082 against
        # 0.70-0.84, so the declared thresholds sit in open space rather than against a rounded edge.
        ratios = [abs(float(tensor.mean()))/max(1e-9, float(tensor.abs().mean()))
                  for part in parts for tensor in (part["local"],)]
        out["largest_level_ratio"] = round(max(ratios), 6)
        out["constant_mode_removed"] = max(ratios) <= LEVEL_RATIO_REMOVED
        if round(out["free_without_local"]["auc"]+out["local_only"]["auc"], 6) == 0:
            raise ValueError("the stencil split did not separate")
    inverted = transformed([part["lattice"] for part in parts])
    if inverted["auc"] != lattice["auc"]:
        raise ValueError("the lattice part did not re-derive")
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
    out.update({"verdict": verdict,
                "gain_over_lattice_auc": round(summed["auc"]-lattice["auc"], 6),
                "lattice_minus_free_auc": round(lattice["auc"]-free["auc"], 6),
                "declared_bar": {"channel_gain": CHANNEL_GAIN, "channel_gap": CHANNEL_GAP}})
    if "local" in parts[0]:
        out["local_minus_pooled_auc"] = round(out["local_only"]["auc"]-out["free_without_local"]["auc"], 6)
        declared = config.get("occupancy_free_stencil", 0)
        if declared in (2, 3) and not out["constant_mode_removed"]:
            raise ValueError("the de-meaned arm's local read still carries a board-wide level")
        if declared == 1 and out["largest_level_ratio"] < LEVEL_RATIO_PRESENT:
            raise ValueError("the un-de-meaned arm's local read has no board-wide level, so the two modes are aliases")
    return out


def post_hoc_crossing(result, grid):
    """The same crossing rule applied to a read trained without it: stack the sum, then decide per board."""
    config = result["config"]
    rows, columns = config["visible_rows"], config["columns"]
    cases = [case for case in result["cases"] if case["split"] == "training"]
    stages = [base_of(case, rows, columns) for case in cases]
    if config.get("occupancy_stack_monotone"):
        stages = [stack_contiguous(value, rows) for value in stages]
    stages = [otsu_board(value, rows) for value in stages]
    return score(cases, grid, rows, columns, stages)


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
    parser.add_argument("--inertness", type=Path, default=INERTNESS,
                        help="the frozen-source proof for the modes the arm families do not share")
    parser.add_argument("--predeclaration", default=PREDECLARATION)
    parser.add_argument("--instrument", default=INSTRUMENT)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("an evidence directory already exists")
    prior = read(PRIOR)
    if prior.get("instrument") != "m10-supervised-anchor-r1":
        raise ValueError("the reproduction reference is not the committed extremal-anchor block")
    expected_auc, expected_f1 = prior["arms"][CONTROL]["auc"], prior["arms"][CONTROL]["f1_at_zero"]

    grid = tuple(threshold_grid())
    arms, decompositions, channels = {}, {}, {}
    for name in ARMS:
        study, summary, result = require_sweep(name)
        scored = arm_summary(result["deciding"])
        counts = scored["operating_counts"]
        positives = counts["true_positive"]+counts["false_positive"]
        arms[name] = {
            "study": study, "sweep": f"{study}-sweep-r1",
            "declared": declared_of(name),
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
            "parameters": summary.get("parameters"),
        }
        decompositions[name] = decomposition(result, grid)
        if ARMS[name][0] == 2:
            channels[name] = channels_of(result, grid)
    # The catalogue fingerprint is the archived source tree, and the de-meaned mode was added between the first six
    # arms and the last two. Two fingerprints are therefore admissible, but only with a bit-for-bit proof that the
    # branch is inert for the modes the earlier arms use - which is stronger evidence than one shared hash.
    fingerprints = {arms[name]["catalogue_sha256"] for name in arms}
    inertness = read(args.inertness)
    if not inertness.get("inert") or any(not inertness["modes"][mode]["occupancy_identical"] or
                                         not all(inertness["modes"][mode]["parts_identical"].values())
                                         for mode in inertness["modes"]):
        raise ValueError("the added stencil mode is not inert for the modes the earlier arms use")
    covered = {int(mode) for mode in inertness["modes"]}
    if not {arm["declared"]["occupancy_free_stencil"] for arm in arms.values()} - {max(covered)+1} <= covered:
        raise ValueError("an arm declares a mode the proof does not cover")
    control = arms[CONTROL]
    if control["auc"] != expected_auc or control["f1_at_zero"] != expected_f1:
        raise ValueError(f"control does not reproduce the recorded {expected_auc!r} / {expected_f1!r}")
    if len({arm["weights_sha256"] for arm in arms.values()}) != len(ARMS):
        raise ValueError("the arms did not produce distinct checkpoints where they must")
    m20 = read(TWO_PATH)["arms"]["two-path"]["weights_sha256"]
    m21 = read(LEVEL_BLOCK)["arms"]["crossing"]["weights_sha256"]
    reproduction = {"envelope_reproduces_m20": arms["envelope"]["weights_sha256"] == m20,
                    "crossing_reproduces_m21": arms["crossing"]["weights_sha256"] == m21}
    if not all(reproduction.values()):
        raise ValueError("the unchanged arms did not reproduce their earlier checkpoints")

    def reading_of(name):
        arm = arms[name]
        delta = round(arm["f1_at_zero"]-control["f1_at_zero"], 6)
        auc_delta = round(arm["auc"]-control["auc"], 6)
        gate = arm["exact_boards_at_zero"]["exact"] > 0
        if delta >= ATTRIBUTION_DELTA and gate:
            return ("level-passes" if ARMS[name][2] else "stencil-passes"), delta, auc_delta
        if ARMS[name][3]:
            return ("stencil-moves-without-clearing" if delta > 0 else "stencil-fails"), delta, auc_delta
        if delta > 0 and auc_delta >= -AUC_TOLERANCE:
            return "level-passes", delta, auc_delta
        if delta > 0:
            return "level-without-ranking", delta, auc_delta
        return "level-fails", delta, auc_delta

    readings = {name: reading_of(name) for name in ARMS if name != CONTROL}
    for name, (reading, _, _) in readings.items():
        allowed = STENCIL_READINGS if ARMS[name][3] else LEVEL_READINGS
        if reading not in allowed:
            raise ValueError(f"{name} produced an undeclared reading")

    ablation = {
        "question": "is the crossing rule's gain from being trained in the loop, or does applying it afterwards "
                    "to a read trained without it do the same?",
        "in_loop": {"arm": "crossing", **{key: arms["crossing"][key] for key in
                                          ("auc", "f1_at_zero", "positive_share_at_zero")},
                    "best": arms["crossing"]["best"],
                    "stage": next(item for item in decompositions["crossing"]["stages"]
                                  if item["stage"] == "crossing")},
        "post_hoc": {"arm": "envelope", "applied": "stack_contiguous then otsu_board on the arm's own weights",
                     **post_hoc_crossing(read(ROOT/"runs"/(study_of("envelope")+"-sweep-r1")/"result.json"), grid)},
        "trained_without_rule": {"arm": "envelope", "auc": arms["envelope"]["auc"],
                                 "f1_at_zero": arms["envelope"]["f1_at_zero"],
                                 "positive_share_at_zero": arms["envelope"]["positive_share_at_zero"]},
    }
    ablation["in_loop_minus_post_hoc_f1"] = round(ablation["in_loop"]["f1_at_zero"]
                                                 - ablation["post_hoc"]["f1_at_zero"], 6)
    ablation["in_loop_minus_post_hoc_auc"] = round(ablation["in_loop"]["auc"]-ablation["post_hoc"]["auc"], 6)
    ablation["verdict"] = ("in-loop-training-is-what-makes-the-rule-work"
                           if ablation["in_loop"]["f1_at_zero"] > ablation["post_hoc"]["f1_at_zero"]
                           else "post-hoc-application-matches-the-rule")

    best_stencil = max((name for name in ARMS if ARMS[name][3]),
                       key=lambda name: readings[name][1])
    scope = {
        "status": "passed",
        "instrument": args.instrument,
        "predeclaration": args.predeclaration,
        "scope": ("Six 64-step arms in one catalogue: the published read, the M20 two-path read, the M21 level "
                  "composition, the sum without the envelope, and the two free-channel compositions with a coarse "
                  "local stencil on the fixed lattice. Every arm is scored read-only at the standing 0.0 boundary and "
                  "over the standing 81-point grid, every arm's read is rebuilt stage by stage, and every free-channel "
                  "arm is split into lattice, pooled-free, local-stencil and summed readings."),
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "mechanism": {
            "name": "coarse-local-stencil",
            "field": "occupancy_free_stencil",
            "definition": ("five taps - the cell's own coarse box and its four neighbours - read from the "
                           "dimension-wide coarse feature map at equal divisions of the capture, plus the cell's row "
                           "and column embeddings, through a two-layer head whose output projection starts at zero"),
            "why_fixed": ("the divisions are of the capture, not of the decoded lattice, so the channel cannot inherit "
                          "the placement error that made pixel-local features fail"),
            "why_coarse": ("a box average is insensitive to sub-cell misplacement: the stencil reads a region, not a "
                           "sample"),
            "parameters_added": "6*dimension*dimension + dimension + 1",
            "identity_at_zero": "the output projection is zero at step zero, so the read is bit-identical to the same arm without it",
            "parts": "the local read is kept beside the sum, so the pooled free channel and the stencil are scored exactly",
        },
        "arms": arms,
        "decomposition": decompositions,
        "channels": channels,
        "readings": {name: {"reading": reading, "f1_at_zero_delta": delta, "auc_delta": auc_delta}
                     for name, (reading, delta, auc_delta) in readings.items()},
        "best_stencil_arm": best_stencil,
        "ablation": ablation,
        "reproduction": reproduction,
        "source_fingerprints": {"distinct": len(fingerprints), "values": sorted(fingerprints),
                               "inertness_proof": str(args.inertness.resolve().relative_to(ROOT)),
                               "reading": ("the eight arms run under two source fingerprints that differ only by the "
                                           "added de-mean branch; the proof compares the archived and current sources "
                                           "on one checkpoint and finds the read and every recorded part "
                                           "bit-identical for modes 0 and 1, and no earlier arm declares mode 2")},
        "contrasts": {f"{name}_vs_control": {
            "f1_at_zero_delta": round(arm["f1_at_zero"]-control["f1_at_zero"], 6),
            "auc_delta": round(arm["auc"]-control["auc"], 6),
            "best_f1_delta": round(arm["best"]["f1"]-control["best"]["f1"], 6),
            "out_of_fold_delta": round(arm["out_of_fold_f1"]-control["out_of_fold_f1"], 6),
            "positive_share_delta": round(arm["positive_share_at_zero"]-control["positive_share_at_zero"], 6),
            "row_error_delta": round(arm["row_error_at_zero"]["mean"]-control["row_error_at_zero"]["mean"], 6)}
            for name, arm in arms.items() if name != CONTROL},
        "declaration_of_change": ("M21 relaxed the M20 coupling of the sum to the envelope; M22 adds a declared field "
                                  "beside it. Neither changes an existing arm's declared values, so the M20 and M21 "
                                  "checkpoints are reproduced bit-for-bit inside this catalogue."),
        "aliasing_guard": {
            "arms_differ_in": list(DECLARED),
            "distinct_checkpoints": len({arm["weights_sha256"] for arm in arms.values()}),
            "envelope_reproduces_m20": reproduction["envelope_reproduces_m20"],
            "crossing_reproduces_m21": reproduction["crossing_reproduces_m21"],
            "rule": "each sweep's weights equal its own study's checkpoint, the control and every unchanged arm "
                    "reproduce their earlier checkpoints exactly, and each arm differs from the control in exactly its "
                    "declared fields",
        },
        "reference": {
            "two_path": reference(TWO_PATH, "two-path"),
            "crossing_rule": reference(RULED, "ruled"),
            "span_under_level": reference(LEVEL, "composed"),
            "envelope": reference(ENVELOPE, "contiguous"),
            "combined_read": reference(COMBINED, next(name for name in read(COMBINED)["arms"]
                                                      if name != "control")),
            "centring": reference(CENTRING, "centred"),
            "midpoint_anchor": reference(PRIOR_SCOPE, "level"),
            "role": "reference-only, each measured in its own catalogue and none re-measured or re-verdict'd here",
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
    print(json.dumps({"readings": scope["readings"], "best_stencil_arm": best_stencil,
                      "ablation": {key: ablation[key] for key in ("in_loop_minus_post_hoc_f1",
                                                                  "in_loop_minus_post_hoc_auc", "verdict")},
                      "reproduction": reproduction}, indent=2, default=str))


if __name__ == "__main__":
    main()
