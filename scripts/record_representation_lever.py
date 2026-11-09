"""Curate the declared occupancy representation lever against its matched control.

`docs/M3_OCCUPANCY_REPRESENTATION.md` fixes the rule; this recorder applies it to the two admitted
continuation studies and their two widened sweeps. It re-derives every published quantity from the
receipts, refuses a mismatched pair, and reports the cluster-respecting intervals that no verdict
reads.
"""
import argparse
import hashlib
import json
from pathlib import Path

from cluster_interval import cluster_bootstrap
from gate_power import auc_interval, f1_from_counts, f1_gap, precision_recall_interval
from occupancy_sweep import AUC_THRESHOLD, F1_THRESHOLD, pooled_auc, sweep_verdict
from percell_gate import CELL_F1_THRESHOLD
from probe_model import require_bounded_job

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 1152*1024**2
SWEEP_CEILING_BYTES = 640*1024**2
STEPS = 64
DECIDING_IMAGES = 36
DEVELOPMENT_IMAGES = 18
FORWARDS = DECIDING_IMAGES+DEVELOPMENT_IMAGES
ATTRIBUTION_DELTA = 0.05
BOOTSTRAP_DRAWS = 400
BOOTSTRAP_SEED = 20261026
SOURCE_RUN = "m3-percell-balance-r1"
ARMS = {"lever": 1, "control": 0}


def read(path, maximum=16*1024**2):
    if path.stat().st_size > maximum:
        raise ValueError("representation-lever evidence JSON byte budget exceeded")
    return json.loads(path.read_text())


def digest(path):
    value, consumed = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            consumed += len(block)
            if consumed > 64*1024**2:
                raise ValueError("representation-lever evidence artifact byte budget exceeded")
            value.update(block)
    return value.hexdigest()


def counts_of(row):
    return {key: row[key] for key in ("true_positive", "false_positive", "false_negative")}


def zero_row(deciding):
    return deciding["rows"][deciding["thresholds"].index(0.0)]


def ancestry_of(sample_id):
    """Two seed tokens (`training-20261026`) are the twelve ancestry clusters."""
    parts = sample_id.split("-")
    return "-".join(parts[:2])


def cells_of(rows):
    return [(logit, label) for row in rows
            for logit, label in zip(row["logits"], row["occupied"])]


def f1_at_threshold(rows, threshold=0.0):
    true_positive = false_positive = false_negative = 0
    for logit, label in cells_of(rows):
        predicted = logit > threshold
        if label and predicted:
            true_positive += 1
        elif label:
            false_negative += 1
        elif predicted:
            false_positive += 1
    return f1_from_counts(true_positive, false_positive, false_negative)


def arm_summary(deciding):
    row = zero_row(deciding)
    counts = counts_of(row)
    return {"auc": deciding["auc"], "auc_interval": auc_interval(
                deciding["auc"], deciding["occupied_cells"], deciding["empty_cells_on_nonempty_boards"]),
            "per_cell_f1": round(f1_from_counts(counts["true_positive"], counts["false_positive"],
                                                counts["false_negative"]), 6),
            "per_cell_gap": f1_gap(f1_from_counts(counts["true_positive"], counts["false_positive"],
                                                  counts["false_negative"]), CELL_F1_THRESHOLD),
            "operating_counts": counts,
            "operating_intervals": precision_recall_interval(counts["true_positive"],
                                                            counts["false_positive"],
                                                            counts["false_negative"]),
            "best": {key: deciding["best"][key] for key in
                     ("threshold", "true_positive", "false_positive", "false_negative",
                      "precision", "recall", "f1", "exact_nonempty_boards")},
            "best_gap": f1_gap(deciding["best"]["f1"], F1_THRESHOLD),
            "exact_boards_at_best": deciding["exact_boards_at_best"],
            "verdict": sweep_verdict(deciding)}


def lever_verdict(per_cell_f1, ranking_verdict, attribution_delta, *,
                  cell_bar=CELL_F1_THRESHOLD, attribution=ATTRIBUTION_DELTA):
    """All three declared conditions; a tie at any boundary counts as met."""
    conditions = {"per_cell_f1": per_cell_f1 >= cell_bar,
                  "ranking": ranking_verdict == "informative",
                  "attribution": attribution_delta >= attribution}
    return {"verdict": "cleared" if all(conditions.values()) else "not-cleared",
            "conditions": conditions,
            "failing": [name for name, met in conditions.items() if not met],
            "cell_bar": cell_bar, "attribution_floor": attribution}


def paired_bootstrap(lever_cases, control_cases):
    """Whole-ancestry-cluster resampling of both arms on the same sampled clusters."""
    grouped = {}
    for name, rows in (("lever", lever_cases), ("control", control_cases)):
        for row in rows:
            grouped.setdefault(ancestry_of(row["sample_id"]), {}).setdefault(name, []).append(row)
    if len(grouped) < 2:
        raise ValueError("at least two ancestry clusters required")
    for cluster, arms in grouped.items():
        if set(arms) != set(ARMS):
            raise ValueError("both arms must cover every ancestry cluster")
    ordered = tuple((name, arms) for name, arms in sorted(grouped.items()))

    def statistic(sample):
        values = {}
        for name in ("lever", "control"):
            rows = [row for entry in sample for row in entry[name]]
            positive = [(logit, label) for logit, label in cells_of(rows) if label]
            negative = [(logit, label) for logit, label in cells_of(rows) if not label]
            if not positive or not negative:
                return None
            values["auc_"+name] = pooled_auc([value for value, _ in positive],
                                             [value for value, _ in negative])
            values["f1_"+name] = f1_at_threshold(rows)
        values["f1_delta"] = values["f1_lever"]-values["f1_control"]
        values["auc_delta"] = values["auc_lever"]-values["auc_control"]
        return values

    return cluster_bootstrap(statistic, ordered,
                             ("auc_lever", "f1_lever", "auc_control", "f1_control",
                              "f1_delta", "auc_delta"),
                             draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)


def require_arm(name):
    context = ARMS[name]
    study = f"m3-occctx-{name}-r1"
    summary = read(ROOT/"runs"/study/"summary.json")
    job = read(ROOT/"runs"/(study+"-job.json"))
    if job.get("status") != "passed" or job.get("exit_code") != 0:
        raise ValueError(f"{name} study did not pass inside its declared envelope")
    if (summary.get("status") != "complete" or summary.get("steps") != STEPS or
            summary.get("requested_steps") != STEPS or not summary.get("checkpoint_round_trip_exact") or
            job.get("work_items") != STEPS or job.get("memory_bytes") != CEILING_BYTES):
        raise ValueError(f"{name} study is not a complete 64-step run inside 1,152 MiB")
    arguments = summary["arguments"]
    if (arguments.get("occupancy_context") != context or arguments.get("steps") != STEPS or
            arguments.get("seed") != 20261026 or arguments.get("capture_max_side") != 512 or
            summary["recovery_contract"].get("occupancy_context") != context):
        raise ValueError(f"{name} study did not run the declared lever/budget")
    origin = summary.get("initial_weights_from") or {}
    if not str(origin.get("run", "")).endswith(SOURCE_RUN) or not origin.get("weights_sha256"):
        raise ValueError(f"{name} study was not initialized from the published checkpoint")
    return study, summary, job, origin


def require_sweep(name):
    study = f"m3-occctx-{name}-r1"
    sweep = f"{study}-sweep-r1"
    result = read(ROOT/"runs"/sweep/"result.json")
    job = read(ROOT/"runs"/(sweep+"-job.json"))
    summary = read(ROOT/"runs"/study/"summary.json")
    profile = read(ROOT/"runs"/(sweep+"-profile-r1")/"summary.json")
    profile_job = read(ROOT/"runs"/(sweep+"-profile-r1-job.json"))
    if (job.get("status") != "passed" or job.get("exit_code") != 0 or
            job.get("memory_bytes") != SWEEP_CEILING_BYTES or job.get("work_items") != result.get("forwards")):
        raise ValueError(f"{name} sweep did not pass under the declared 640 MiB")
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
    if (declared.get("evaluated_groups") != 12 or
            declared.get("evaluated_cases") != DECIDING_IMAGES):
        raise ValueError(f"{name} sweep did not score the widened deciding population")
    if (profile_job.get("status") != "passed" or profile.get("status") != "complete" or
            profile.get("weights_sha256") != result.get("weights_sha256")):
        raise ValueError(f"{name} sweep admission profile differs")
    return sweep, result, job


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
    if arms["lever"]["summary"]["initial_weights_sha256"] == arms["control"]["summary"]["initial_weights_sha256"]:
        raise ValueError("the lever and control must not share one saved initial tensor file")
    lever_cases = [row for row in arms["lever"]["result"]["cases"] if row["split"] == "training"]
    control_cases = [row for row in arms["control"]["result"]["cases"] if row["split"] == "training"]
    if len(lever_cases) != DECIDING_IMAGES or len(control_cases) != DECIDING_IMAGES:
        raise ValueError("the paired bootstrap requires exactly the 36 deciding training images")
    if ([row["sample_id"] for row in lever_cases] != [row["sample_id"] for row in control_cases] or
            [(row["logits"], row["occupied"]) for row in lever_cases] ==
            [(row["logits"], row["occupied"]) for row in control_cases]):
        raise ValueError("the two sweeps must share one population and differ in logits")

    measured = {name: arm_summary(arms[name]["result"]["deciding"]) for name in ARMS}
    bootstrap = paired_bootstrap(lever_cases, control_cases)
    if bootstrap["clusters"] != 12:
        raise ValueError("the paired bootstrap must resample the twelve ancestry clusters")
    delta_best = measured["lever"]["best"]["f1"]-measured["control"]["best"]["f1"]
    verdict = lever_verdict(measured["lever"]["per_cell_f1"], measured["lever"]["verdict"], delta_best)

    scope = {
        "status": "passed",
        "instrument": "m3-occupancy-representation-r1",
        "predeclaration": "docs/M3_OCCUPANCY_REPRESENTATION.md",
        "scope": ("Matched 64-step continuation of one published checkpoint, lever = declared "
                  "zero-initialized occupancy read-out neighborhood, control = the same budget "
                  "without it. Scored read-only on the widened deciding population with the "
                  "standing gates and grid; no threshold, population, ceiling or gate changes, and "
                  "no real-image, complete-state, calibrated or quality claim."),
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "envelope": {"steps": STEPS, "ceiling_bytes": CEILING_BYTES,
                     "sweep_ceiling_bytes": SWEEP_CEILING_BYTES,
                     "attribution_floor": ATTRIBUTION_DELTA},
        "source": {"run": SOURCE_RUN, "weights_sha256": source["checkpoint_sha256"],
                   "steps": source["steps"]},
        "matched": {"seed": 20261026, "learning_rate": .001, "capture_max_side": 512,
                    "curriculum": "stream-v1", "config_difference": ["occupancy_context"],
                    "initial_weights": arms["lever"]["summary"]["initial_weights_from"]},
        "arms": {name: {"study": arms[name]["study"], "sweep": arms[name]["sweep"],
                        "occupancy_context": ARMS[name],
                        "weights_sha256": arms[name]["result"]["weights_sha256"],
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
                        "cluster_bootstrap": bootstrap},
        "verdict": verdict,
        "population": {"split": "training", "images": DECIDING_IMAGES,
                       "nonempty_boards": sum(1 for row in lever_cases if any(row["occupied"])),
                       "occupied_cells": arms["lever"]["result"]["deciding"]["occupied_cells"],
                       "empty_cells": arms["lever"]["result"]["deciding"]["empty_cells_on_nonempty_boards"],
                       "contexts": len({ancestry_of(row["sample_id"]) for row in lever_cases})},
        "unchanged": {"thresholds": {"per_cell_f1": CELL_F1_THRESHOLD, "ranking_auc": AUC_THRESHOLD,
                                     "ranking_f1": F1_THRESHOLD},
                      "population": "widened deciding population, twelve groups",
                      "source_checkpoint": source["checkpoint_sha256"],
                      "prior_evidence": "not touched",
                      "frozen_manifest": "not touched"},
    }
    args.output.mkdir(parents=True)
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(scope, stream, indent=2, allow_nan=False)
    print(json.dumps({key: scope[key] for key in ("verdict", "attribution", "arms")}, indent=2))


if __name__ == "__main__":
    main()
