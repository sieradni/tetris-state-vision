"""Curate the widened deciding-population instrument change into frozen evidence.

`docs/M3_WIDENED_DECIDING_POPULATION.md` predeclares the change. Nothing here declares a
threshold: the rules come from `fit_gate`, `percell_gate`, `occupancy_sweep` and
`identity_gate`, and `gate_power` supplies intervals that no verdict reads.

Every published number is re-derived from the receipts this recorder copies. The recorded
nine-case subset of each widened replay must still reproduce the earlier evidence exactly:
that join is what makes the narrow-population numbers superseded rather than invalidated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from fit_gate import FIT_GATE_THRESHOLD, fit_gate_opens, occupancy_mechanism
from gate_power import auc_interval, f1_from_counts, f1_gap, precision_recall_interval
from identity_gate import IDENTITY_CLASS_THRESHOLD, identity_gate_opens, identity_verdict
from m2_error_analysis import summarize
from m3_error_analysis import summarize_components
from occupancy_sweep import AUC_THRESHOLD, CHANCE_AUC, F1_THRESHOLD, sweep_verdict
from percell_gate import CELL_F1_THRESHOLD, decision_mechanism, occupied_cell_f1, percell_gate_opens
from probe_model import require_bounded_job
from probe_population import CANONICAL_TRAINING_GROUPS, RECORDED_TRAINING_GROUPS

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 640*1024**2
POPULATION = {"training": CANONICAL_TRAINING_GROUPS*3, "development": 18}
OLD_RESULT = {"occupancy": "docs/evidence/m3-percell-balance-r1-errors-r1/result.json",
              "identity": "docs/evidence/m3-identity-factor-length-r1-errors-r1/result.json"}
OLD_SWEEP = "docs/evidence/m3-sweep-occupancy-r1-ranking-r1/scope.json"
# A bounded launch names its sidecars after the receipt; both shapes are declared here so a
# missing log is a named refusal rather than a silent gap (V-053).
SIDECAR_CANDIDATES = ("-job.json", ".log", "-job.log", ".stderr.log", "-job.stderr.log")


def read(path, maximum=4*1024**2):
    if path.stat().st_size > maximum:
        raise ValueError("widened-population evidence JSON byte budget exceeded")
    return json.loads(path.read_text())


def digest(path):
    value, consumed = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            consumed += len(block)
            if consumed > 64*1024**2:
                raise ValueError("widened-population evidence artifact byte budget exceeded")
            value.update(block)
    return value.hexdigest()


def split_rows(result, split):
    return [row for row in result["cases"] if row["split"] == split]


def require_replay(name, run, expected_weights):
    job = read(ROOT/"runs"/(name+"-job.json"))
    result = read(ROOT/"runs"/name/"result.json")
    if (job.get("status") != "passed" or job.get("exit_code") != 0 or
            job.get("memory_bytes") != CEILING_BYTES):
        raise ValueError("widened replay job must pass under the declared ceiling")
    if (result.get("status") != "passed" or not result.get("automatic_metrics_reproduced") or
            not result.get("weights_unchanged") or result.get("run") != run):
        raise ValueError("widened replay must pass with reproduced metrics and unchanged weights")
    if (job.get("work_items") != result.get("forwards") or
            result["forwards"] != len(result["cases"]) or
            result["forwards"] != POPULATION["training"]+POPULATION["development"]):
        raise ValueError("widened replay forward count differs")
    declared = result.get("training_population") or {}
    if (declared.get("recorded_groups") != RECORDED_TRAINING_GROUPS or
            declared.get("evaluated_groups") != CANONICAL_TRAINING_GROUPS or
            declared.get("evaluated_cases") != POPULATION["training"] or
            declared.get("extension_cases") != POPULATION["training"]-RECORDED_TRAINING_GROUPS*3):
        raise ValueError("widened replay training population differs")
    if (len(split_rows(result, "training")) != POPULATION["training"] or
            len(split_rows(result, "development")) != POPULATION["development"]):
        raise ValueError("widened replay split sizes differ")
    if result["weights_sha256"] != expected_weights:
        raise ValueError("widened replay weights differ from the published evidence")
    return job, result


def require_sweep(name, expected_weights):
    job = read(ROOT/"runs"/(name+"-job.json"))
    result = read(ROOT/"runs"/name/"result.json")
    profile_dir = ROOT/"runs"/result["admission"]["run"]
    profile = read(profile_dir/"summary.json")
    profile_job = read(ROOT/"runs"/(result["admission"]["run"]+"-job.json"))
    if (job.get("status") != "passed" or job.get("exit_code") != 0 or
            job.get("memory_bytes") != CEILING_BYTES or job.get("work_items") != result["forwards"]):
        raise ValueError("widened sweep job must pass under the declared ceiling")
    if result.get("status") != "passed" or result["forwards"] != len(result["cases"]):
        raise ValueError("widened sweep result must pass")
    if result["forwards"] != POPULATION["training"]+POPULATION["development"]:
        raise ValueError("widened sweep forward count differs")
    if result["weights_sha256"] != expected_weights:
        raise ValueError("widened sweep weights differ from the published evidence")
    if (profile_job.get("status") != "passed" or profile.get("status") != "complete" or
            profile.get("cases") != POPULATION or profile.get("weights_sha256") != expected_weights):
        raise ValueError("widened sweep admission profile differs")
    if sweep_verdict(result["deciding"]) != result["verdict"]:
        raise ValueError("widened sweep verdict does not re-derive from its own rows")
    return job, result, profile


def counts(labels, prefix):
    return {key: labels[prefix+key] for key in ("true_positive", "false_positive", "false_negative")}


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occupancy-replay", required=True)
    parser.add_argument("--identity-replay", required=True)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("evidence directory already exists")

    old_occupancy = read(ROOT/OLD_RESULT["occupancy"])
    old_identity = read(ROOT/OLD_RESULT["identity"])
    old_sweep = read(ROOT/OLD_SWEEP)

    occupancy_job, occupancy = require_replay(args.occupancy_replay, "m3-percell-balance-r1",
                                              old_occupancy["weights_sha256"])
    identity_job, identity = require_replay(args.identity_replay, "m3-identity-factor-length-r1",
                                            old_identity["weights_sha256"])
    sweep_job, sweep, profile = require_sweep(args.sweep, old_occupancy["weights_sha256"])

    occupancy_training, occupancy_development = split_rows(occupancy, "training"), split_rows(occupancy, "development")
    identity_training = split_rows(identity, "training")

    # Supersession join: the recorded nine-case subset still reproduces the published numbers.
    saved_occupancy = [row for row in occupancy_training if row.get("saved", True)]
    saved_totals = summarize(saved_occupancy, max_cases=45)["training"]
    if saved_totals["locked_populations"] != old_occupancy["totals"]["training"]["locked_populations"]:
        raise ValueError("recorded nine-case subset no longer reproduces the published occupancy counts")
    saved_identity = [row for row in identity_training if row.get("saved", True)]
    saved_components = summarize_components(saved_identity)["training"]
    old_components = old_identity["totals"]["training"]["components"]
    for field in ("queue", "hold"):
        if saved_components[field]["predicted_classes"] != old_components[field]["predicted_classes"]:
            raise ValueError("recorded nine-case subset no longer reproduces the published identity classes")

    widened_training = summarize(occupancy_training, max_cases=45)["training"]
    widened_development = summarize(occupancy_development, max_cases=45)["development"]
    identity_totals = summarize(identity_training, max_cases=45)["training"]
    identity_components = summarize_components(identity_training)["training"]

    locked = widened_training["locked_populations"]["nonempty"]
    wide = counts(locked, "")
    fit_opens = fit_gate_opens(widened_training["locked_populations"])
    cell_f1 = occupied_cell_f1(widened_training["locked_populations"])
    operating_intervals = precision_recall_interval(wide["true_positive"], wide["false_positive"],
                                                    wide["false_negative"])
    zero_index = sweep["deciding"]["thresholds"].index(0.0)
    zero_row = sweep["deciding"]["rows"][zero_index]
    if counts(zero_row, "") != wide:
        raise ValueError("widened sweep zero row does not join the widened replay")
    best = sweep["deciding"]["best"]
    deciding_auc = sweep["deciding"]["auc"]
    deciding_positive = sweep["deciding"]["occupied_cells"]
    deciding_negative = sweep["deciding"]["empty_cells_on_nonempty_boards"]
    development_locked = widened_development["locked_populations"]["nonempty"]
    queue, hold = identity_components["queue"], identity_components["hold"]

    scope = {
        "status": "passed",
        "instrument_change": "widened-deciding-population-r1",
        "predeclaration": "docs/M3_WIDENED_DECIDING_POPULATION.md",
        "scope": ("Instrument change only: the deciding training population is widened from the "
                  "recorded three groups to the canonical twelve and the already-published rules are "
                  "re-derived on it. No model, loss, seed, capture or data change, no new gate "
                  "threshold, no threshold re-sweep of the closed occupancy family, and no "
                  "real-image, complete-state, calibrated or quality claim."),
        "populations": {
            "deciding_split": "training",
            "before": {"groups": RECORDED_TRAINING_GROUPS, "images": RECORDED_TRAINING_GROUPS*3,
                       "nonempty_images": old_occupancy["totals"]["training"]["locked_populations"]["nonempty"]["images"],
                       "occupied_cells": old_occupancy["totals"]["training"]["locked_populations"]["nonempty"]["true_positive"]
                       + old_occupancy["totals"]["training"]["locked_populations"]["nonempty"]["false_negative"]},
            "after": {"groups": CANONICAL_TRAINING_GROUPS, "images": POPULATION["training"],
                      "nonempty_images": locked["images"],
                      "occupied_cells": wide["true_positive"]+wide["false_negative"],
                      "empty_cells_on_nonempty_boards": sweep["deciding"]["empty_cells_on_nonempty_boards"],
                      "empty_images": widened_training["locked_populations"]["empty"]["images"],
                      "ancestry_groups": widened_training["ancestry_groups"],
                      "base_seed": occupancy["training_population"]["base_seed"]},
            "development": {"groups": 6, "images": POPULATION["development"],
                            "nonempty_images": development_locked["images"],
                            "occupied_cells": development_locked["true_positive"]+development_locked["false_negative"]},
        },
        "unchanged_thresholds": {
            "fit_gate_exact_nonempty_boards": FIT_GATE_THRESHOLD,
            "per_cell_f1": CELL_F1_THRESHOLD,
            "ranking_auc": AUC_THRESHOLD,
            "ranking_chance_auc": CHANCE_AUC,
            "ranking_f1": F1_THRESHOLD,
            "sweep_grid_points": len(sweep["deciding"]["thresholds"]),
            "identity_distinct_classes": IDENTITY_CLASS_THRESHOLD,
        },
        "occupancy": {
            "replay_operating_point": {
                "threshold": 0.0,
                **wide,
                "f1": round(f1_from_counts(wide["true_positive"], wide["false_positive"],
                                           wide["false_negative"]), 6),
                "f1_gap_to_bar": f1_gap(f1_from_counts(wide["true_positive"], wide["false_positive"],
                                                       wide["false_negative"]), CELL_F1_THRESHOLD),
                "intervals": operating_intervals,
                "exact_nonempty_boards": locked["exact_boards"],
                "nonempty_images": locked["images"],
            },
            "fit_gate": {"quantity": "exact nonempty training boards", "threshold": FIT_GATE_THRESHOLD,
                         "opens": fit_opens, "exact": locked["exact_boards"], "images": locked["images"],
                         "mechanism": occupancy_mechanism(widened_training["locked_populations"]),
                         "interval": None},
            "per_cell_gate": {"quantity": "occupied-cell F1", "threshold": CELL_F1_THRESHOLD,
                              "opens": percell_gate_opens(widened_training["locked_populations"]),
                              "f1": round(cell_f1, 6), "mechanism": decision_mechanism(widened_training["locked_populations"])},
            "ranking": {
                "auc": deciding_auc, "bar": AUC_THRESHOLD,
                "auc_interval": auc_interval(deciding_auc, deciding_positive, deciding_negative),
                "positives": deciding_positive, "negatives": deciding_negative,
                "verdict": sweep["verdict"],
                "best": {key: best[key] for key in ("threshold", "true_positive", "false_positive",
                                                    "false_negative", "precision", "recall", "f1",
                                                    "exact_nonempty_boards")},
                "best_f1_gap_to_bar": f1_gap(best["f1"], F1_THRESHOLD),
                "best_intervals": precision_recall_interval(best["true_positive"], best["false_positive"],
                                                            best["false_negative"]),
                "exact_boards_at_best": sweep["deciding"]["exact_boards_at_best"],
                "empty_population": sweep["deciding"]["empty_population"],
            },
        },
        "identity": {
            "queue": {key: queue.get(key) for key in ("images", "distinct_predicted_classes",
                                                      "constant_predictor", "predicted_classes")},
            "hold": {key: hold.get(key) for key in ("images", "distinct_predicted_classes",
                                                    "constant_predictor", "predicted_classes")},
            "verdict": identity_verdict(queue, hold),
            "queue_exact_rate": queue["piece_tokens"],
            "queue_exact_interval": None,
            "position_spread": identity_totals.get("position_spread"),
        },
        "superseded_without_invalidation": {
            "occupancy_counts": counts(old_occupancy["totals"]["training"]["locked_populations"]["nonempty"], ""),
            "occupancy_exact_nonempty_boards": old_occupancy["totals"]["training"]["locked_populations"]["nonempty"]["exact_boards"],
            "sweep_auc": old_sweep["auc"]["training"],
            "sweep_best_f1": old_sweep["best"]["training"]["f1"],
            "sweep_verdict": old_sweep["verdict"],
            "identity_queue_classes": old_components["queue"]["predicted_classes"],
            "identity_hold_classes": old_components["hold"]["predicted_classes"],
            "join": "recorded nine-case subset reproduced exactly from the widened replay's own cases",
        },
        "receipts": {"occupancy": args.occupancy_replay, "identity": args.identity_replay,
                     "sweep": args.sweep, "admission": sweep["admission"]["run"]},
        "forwards": {"occupancy": occupancy["forwards"], "identity": identity["forwards"],
                     "sweep": sweep["forwards"]},
        "weights_sha256": {"occupancy": occupancy["weights_sha256"], "identity": identity["weights_sha256"]},
    }
    from gate_power import wilson_interval
    scope["identity"]["queue_exact_interval"] = wilson_interval(queue["piece_tokens"]["correct"],
                                                                queue["piece_tokens"]["known"])
    scope["identity"]["hold_exact_interval"] = wilson_interval(hold["correct_pieces"], hold["known_piece_cases"])

    args.output.mkdir(parents=True)
    artifacts = {}
    for prefix, receipt_name, directory in (("occupancy", args.occupancy_replay, args.occupancy_replay),
                                            ("identity", args.identity_replay, args.identity_replay),
                                            ("sweep", args.sweep, args.sweep)):
        sources = [(prefix+"-result.json", ROOT/"runs"/directory/"result.json")]
        for candidate in SIDECAR_CANDIDATES:
            path = ROOT/"runs"/(receipt_name+candidate)
            if path.is_file():
                sources.append((prefix+candidate, path))
        for target, source in sources:
            shutil.copyfile(source, args.output/target)
            artifacts[target] = digest(args.output/target)
    shutil.copyfile(ROOT/"runs"/sweep["admission"]["run"]/"summary.json", args.output/"admission-summary.json")
    artifacts["admission-summary.json"] = digest(args.output/"admission-summary.json")
    scope["artifact_sha256"] = dict(sorted(artifacts.items()))
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(scope, stream, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in scope.items() if key != "artifact_sha256"}, indent=2))


if __name__ == "__main__":
    main()
