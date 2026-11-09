"""Record the clustering addendum: measured design effect and corrected intervals.

`docs/M3_WIDENED_DECIDING_POPULATION.md` predeclares the widened population; this addendum
measures how correlated its per-cell errors actually are and reports an interval method that
respects the nesting, beside the cell-level one already published. It is a methodological
addendum, not a new sweep: no gate, threshold, point estimate, model or evidence byte changes.

Every number is derived from committed artifacts whose bytes are hashed into the record, and
every published point estimate is re-derived and asserted equal before the record is written.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from cluster_interval import (AUC_DRAWS, BOOTSTRAP_SEED, COUNT_DRAWS, cluster_bootstrap,
                              clusters_of, design_effect, effective_size,
                              intra_cluster_correlation, rate_design_effect,
                              wilson_interval_with_design_effect)
from gate_power import auc_interval, f1_from_counts, f1_gap, precision_recall_interval, wilson_interval
from occupancy_sweep import pooled_auc
from probe_model import require_bounded_job

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/"docs/evidence/m3-widened-deciding-population-r1"
NARROW = ROOT/"docs/evidence/m3-sweep-occupancy-r1-ranking-r1/result.json"
INPUTS = {"scope": EVIDENCE/"scope.json", "sweep": EVIDENCE/"sweep-result.json",
          "identity": EVIDENCE/"identity-result.json", "narrow_sweep": NARROW}
OPERATING_THRESHOLD = 0.0


def read(path, maximum=8*1024**2):
    if path.stat().st_size > maximum:
        raise ValueError("clustering addendum JSON byte budget exceeded")
    return json.loads(path.read_text())


def digest(path):
    value, consumed = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            consumed += len(block)
            if consumed > 64*1024**2:
                raise ValueError("clustering addendum artifact byte budget exceeded")
            value.update(block)
    return value.hexdigest()


def ancestry_of(sample_id):
    """The seed token of a sample id is its ancestry group; every group contributes three views."""
    token, _, view = sample_id.rpartition("-v")
    if not token or not view.isdigit():
        raise ValueError("sample id must carry a seed token and view index")
    return token


def boards_of(sweep, split="training"):
    """Nonempty boards only: the deciding population the published sweep scores."""
    boards = []
    for case in sweep["cases"]:
        if case["split"] != split or not any(case["occupied"]):
            continue
        boards.append({"board": case["sample_id"], "group": ancestry_of(case["sample_id"]),
                       "view": case["view"],
                       "cells": [(logit, label) for logit, label in zip(case["logits"], case["occupied"])]})
    if not boards:
        raise ValueError("nonempty deciding boards required")
    return boards


def cells_of(boards):
    return [{"board": board["board"], "group": board["group"], "logit": logit, "label": label}
            for board in boards for logit, label in board["cells"]]


def errors_at(cells, threshold):
    return [0 if ((cell["logit"] > threshold) == bool(cell["label"])) else 1 for cell in cells]


def icc_block(cells, threshold):
    errors = errors_at(cells, threshold)
    block = {"threshold": threshold, "cells": len(cells),
             "occupied_cells": sum(int(cell["label"]) for cell in cells),
             "empty_cells": sum(1-int(cell["label"]) for cell in cells)}
    for level in ("board", "group"):
        ids = [cell[level] for cell in cells]
        icc = intra_cluster_correlation(errors, ids)
        mean_size = len(cells)/len(set(ids))
        deff = design_effect(icc, mean_size)
        block[level] = {"clusters": len(set(ids)), "mean_cells_per_cluster": round(mean_size, 3),
                        "icc": round(icc, 6), "design_effect": round(deff, 6),
                        "effective_cells": round(effective_size(len(cells), deff), 2),
                        "effective_occupied": round(effective_size(block["occupied_cells"], deff), 2),
                        "effective_empty": round(effective_size(block["empty_cells"], deff), 2)}
    return block


def auc_statistic(sample):
    """Cluster payloads in, pooled AUC out; a draw without both classes is degenerate."""
    positive, negative = [], []
    for cluster in sample:
        for board in cluster:
            for logit, label in board["cells"]:
                (positive if label else negative).append(logit)
    if not positive or not negative:
        return None
    return {"auc": pooled_auc(positive, negative)}


def counts_statistic(threshold):
    def statistic(sample):
        true_positive = false_positive = false_negative = 0
        for cluster in sample:
            for board in cluster:
                for logit, label in board["cells"]:
                    predicted = logit > threshold
                    if label and predicted:
                        true_positive += 1
                    elif label:
                        false_negative += 1
                    elif predicted:
                        false_positive += 1
        if not true_positive or not true_positive+false_positive:
            return None
        precision = true_positive/(true_positive+false_positive)
        recall = true_positive/(true_positive+false_negative)
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}
    return statistic


def gate_block(boards, cells, threshold, published, intervals, label):
    """Cell-level interval beside the board- and ancestry-cluster intervals at one operating point."""
    counts = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
    for cell in cells:
        predicted = cell["logit"] > threshold
        if cell["label"] and predicted:
            counts["true_positive"] += 1
        elif cell["label"]:
            counts["false_negative"] += 1
        elif predicted:
            counts["false_positive"] += 1
    f1 = f1_from_counts(counts["true_positive"], counts["false_positive"], counts["false_negative"])
    if abs(f1-published["f1"]) > 1e-6:
        raise ValueError(f"{label} F1 does not reproduce the published point estimate")
    cell_intervals = precision_recall_interval(counts["true_positive"], counts["false_positive"],
                                               counts["false_negative"])
    if cell_intervals != intervals:
        raise ValueError(f"{label} cell-level intervals do not reproduce the published ones")
    block = {"label": label, "threshold": threshold, **counts,
             "f1": round(f1, 6), "f1_gap_to_bar": f1_gap(f1), "cell_level": cell_intervals}
    statistic = counts_statistic(threshold)
    for level in ("board", "group"):
        block[level+"_cluster"] = cluster_bootstrap(statistic, clusters_of(boards, level),
                                                    ("precision", "recall", "f1"))
    # Design effect per proportion: how much larger the resampled standard error is than the
    # binomial standard error the closed-form Wilson interval assumes.
    block["standard_error_ratio"] = {}
    for name in ("precision", "recall"):
        passes, trials = block["cell_level"][name]["successes"], block["cell_level"][name]["trials"]
        estimate = passes/trials
        binomial_se = math.sqrt(estimate*(1-estimate)/trials)
        block["standard_error_ratio"][name] = {
            level: {"binomial_standard_error": round(binomial_se, 6),
                    "design_effect": round((block[level+"_cluster"][name]["standard_error"]/binomial_se)**2, 6)}
            for level in ("board", "group")}
    return block


def identity_block(identity, scope):
    training = [case for case in identity["cases"] if case["split"] == "training"]
    if len(training) != scope["populations"]["after"]["images"]:
        raise ValueError("widened identity training population required")
    queue = [(case["diagnostics"]["components"]["queue"]["piece_tokens"]["known"],
              case["diagnostics"]["components"]["queue"]["piece_tokens"]["correct"]) for case in training]
    queue_successes, queue_trials = sum(correct for _, correct in queue), sum(known for known, _ in queue)
    hold_rows = [{"board": case["sample_id"], "group": ancestry_of(case["sample_id"]),
                  "exact": int(bool(case["diagnostics"]["components"]["hold"]["piece_correct"]))}
                 for case in training if case["diagnostics"]["components"]["hold_identity_known"]]
    hold_successes, hold_trials = sum(row["exact"] for row in hold_rows), len(hold_rows)
    published = scope["identity"]
    if (queue_successes, queue_trials) != (published["queue_exact_rate"]["correct"],
                                           published["queue_exact_rate"]["known"]):
        raise ValueError("queue exact rate does not reproduce the published point estimate")
    if (hold_successes, hold_trials) != (published["hold_exact_interval"]["successes"],
                                         published["hold_exact_interval"]["trials"]):
        raise ValueError("hold exact rate does not reproduce the published point estimate")

    by_group = {}
    for row in hold_rows:
        entry = by_group.setdefault(row["group"], [0, 0])
        entry[0] += 1
        entry[1] += row["exact"]
    queue_deff = rate_design_effect(queue)
    hold_deff = rate_design_effect([(trials, successes) for trials, successes in by_group.values()])
    queue_corrected = wilson_interval_with_design_effect(queue_successes, queue_trials, queue_deff)
    hold_corrected = wilson_interval_with_design_effect(hold_successes, hold_trials, hold_deff)
    if queue_deff != 1.0:
        raise ValueError("an all-failure census cannot carry a between-image design effect")
    for field in ("successes", "trials", "estimate", "low", "high"):
        if queue_corrected[field] != published["queue_exact_interval"][field]:
            raise ValueError("cell-level queue interval does not reproduce the published one")
    hold_cluster = cluster_bootstrap(
        lambda sample: (lambda successes, trials: {"exact": successes/trials} if trials else None)(
            sum(row["exact"] for cluster in sample for row in cluster),
            sum(1 for cluster in sample for _ in cluster)),
        clusters_of(hold_rows, "group"), ("exact",))
    return {
        "queue": {"successes": queue_successes, "trials": queue_trials,
                  "images": len(training), "cell_level": published["queue_exact_interval"],
                  "design_effect": round(queue_deff, 6), "corrected": queue_corrected,
                  "corrected_equals_cell_level": queue_deff == 1.0,
                  "note": ("every supervised position fails, so no between-image variance is "
                           "estimable; the design effect is 1 and the census carries the claim")},
        "hold": {"successes": hold_successes, "trials": hold_trials, "images": hold_trials,
                 "groups": len(by_group), "cell_level": published["hold_exact_interval"],
                 "group_cluster": hold_cluster["exact"], "design_effect_group": round(hold_deff, 6),
                 "corrected": hold_corrected,
                 "note": ("an image is the observation unit and ancestry groups nest them; the "
                          "corrected interval evaluates the Wilson interval at the reduced sample size")},
    }


def quantify(payload):
    """The quantifiers the record quotes, each derived from a stored interval or design effect.

    The over-confidence multiplier is the square root of the precision design effect (a standard
    error or an interval width scales with the root of the variance ratio); the width ratio is
    measured directly from the two published endpoints, so the two must agree up to bootstrap
    asymmetry. Nothing here is typed in by hand.
    """
    points = {"0.0": payload["occupancy"]["operating_point"],
              "-0.90": payload["occupancy"]["best_point"]}
    clustering = payload["clustering"]
    per_point, effective = {}, {}
    for label, block in points.items():
        for level in ("board", "group"):
            record = block["standard_error_ratio"]["precision"][level]
            clustered, cell = block[level+"_cluster"]["precision"], block["cell_level"]["precision"]
            per_point[f"{label}/{level}"] = {
                "precision_design_effect": record["design_effect"],
                "precision_standard_error_inflation": round(math.sqrt(record["design_effect"]), 6),
                "precision_width_ratio": round((clustered["high"]-clustered["low"]) /
                                                (cell["high"]-cell["low"]), 6)}
            effective.setdefault(label, {})[level] = clustering[
                "operating_point" if label == "0.0" else "best_point"][level]["effective_occupied"]
    values = list(per_point.values())
    cells = [clustering[key][level]["effective_cells"]
             for key in ("operating_point", "best_point") for level in ("board", "group")]
    ranking = payload["occupancy"]["ranking_auc"]
    return {
        "derivation": ("precision_design_effect = (resampled standard error / binomial standard "
                       "error)^2; over-confidence multiplier = sqrt(precision_design_effect); "
                       "precision_width_ratio = clustered interval width / cell interval width"),
        "per_point": per_point,
        "precision_design_effect_range": {"low": min(v["precision_design_effect"] for v in values),
                                          "high": max(v["precision_design_effect"] for v in values)},
        "precision_standard_error_inflation_range": {
            "low": min(v["precision_standard_error_inflation"] for v in values),
            "high": max(v["precision_standard_error_inflation"] for v in values)},
        "precision_width_ratio_range": {"low": min(v["precision_width_ratio"] for v in values),
                                        "high": max(v["precision_width_ratio"] for v in values)},
        "precision_width_ratio_at": {f"{label}/{level}": per_point[f"{label}/{level}"]["precision_width_ratio"]
                                     for label in ("0.0", "-0.90") for level in ("board", "group")},
        "effective_occupied": effective,
        "effective_cells_range": {"low": min(cells), "high": max(cells)},
        "auc_width_ratio": {"board": round((ranking["board_cluster"]["high"]-ranking["board_cluster"]["low"]) /
                                            (ranking["cell_level"]["high"]-ranking["cell_level"]["low"]), 6),
                            "group": round((ranking["group_cluster"]["high"]-ranking["group_cluster"]["low"]) /
                                           (ranking["cell_level"]["high"]-ranking["cell_level"]["low"]), 6)},
    }


def published_strings(payload):
    """Canonical renderings of the quoted quantifiers, so a doc cannot drift from the artifact."""
    q, pop, ranking = payload["quantifiers"], payload["population"], payload["occupancy"]["ranking_auc"]
    op, best = payload["occupancy"]["operating_point"], payload["occupancy"]["best_point"]
    clustering, hold = payload["clustering"], payload["identity"]["hold"]

    def span(block, level, name, digits):
        interval = block["cell_level"][name] if level == "cell" else block[level+"_cluster"][name]
        return f"{interval['low']:.{digits}f}-{interval['high']:.{digits}f}"

    strings = {
        "boards": f"{pop['boards']} nonempty boards",
        "groups": f"{pop['ancestry_groups']} ancestry groups",
        "known_cells": f"{pop['known_cells']:,} known cells",
        "occupied_cells": f"{pop['occupied_cells']} occupied",
        "empty_cells": f"{pop['empty_cells']:,} empty",
        "cells_per_board": f"{pop['cells_per_board']:.0f} cells",
        "precision_design_effect_range": (f"{q['precision_design_effect_range']['low']:.1f}x-"
                                          f"{q['precision_design_effect_range']['high']:.1f}x"),
        "precision_design_effect_range_2dp": (f"{q['precision_design_effect_range']['low']:.2f}x-"
                                              f"{q['precision_design_effect_range']['high']:.2f}x"),
        "standard_error_inflation_range": (f"{q['precision_standard_error_inflation_range']['low']:.2f}x-"
                                           f"{q['precision_standard_error_inflation_range']['high']:.2f}x"),
        "width_ratio_range": (f"{q['precision_width_ratio_range']['low']:.2f}x to "
                              f"{q['precision_width_ratio_range']['high']:.2f}x"),
        "width_ratio_operating": f"{q['precision_width_ratio_at']['0.0/group']:.2f}x",
        "width_ratio_best": f"{q['precision_width_ratio_at']['-0.90/group']:.2f}x",
        "auc_width_ratio": f"{q['auc_width_ratio']['group']:.2f}x",
        "effective_occupied_operating": (f"{q['effective_occupied']['0.0']['group']:.0f}-"
                                         f"{q['effective_occupied']['0.0']['board']:.0f}"),
        "effective_occupied_best": (f"{q['effective_occupied']['-0.90']['group']:.0f}-"
                                    f"{q['effective_occupied']['-0.90']['board']:.0f}"),
        "effective_cells_range": (f"{q['effective_cells_range']['low']:,.0f}-"
                                  f"{q['effective_cells_range']['high']:,.0f}"),
        "auc": f"{ranking['point']:.4f}",
        "auc_cell_4": f"{ranking['cell_level']['low']:.4f}-{ranking['cell_level']['high']:.4f}",
        "auc_group_4": f"{ranking['group_cluster']['low']:.4f}-{ranking['group_cluster']['high']:.4f}",
        "auc_cell_6": f"{ranking['cell_level']['low']:.6f}-{ranking['cell_level']['high']:.6f}",
        "auc_group_6": f"{ranking['group_cluster']['low']:.6f}-{ranking['group_cluster']['high']:.6f}",
        "op_precision_cell_4": span(op, "cell", "precision", 4),
        "op_precision_group_4": span(op, "group", "precision", 4),
        "op_precision_cell_6": span(op, "cell", "precision", 6),
        "op_precision_group_6": span(op, "group", "precision", 6),
        "op_recall_cell_4": span(op, "cell", "recall", 4),
        "op_recall_group_4": span(op, "group", "recall", 4),
        "op_recall_cell_6": span(op, "cell", "recall", 6),
        "op_recall_group_6": span(op, "group", "recall", 6),
        "best_precision_cell_4": span(best, "cell", "precision", 4),
        "best_precision_group_4": span(best, "group", "precision", 4),
        "best_precision_cell_6": span(best, "cell", "precision", 6),
        "best_precision_group_6": span(best, "group", "precision", 6),
        "best_recall_cell_4": span(best, "cell", "recall", 4),
        "best_recall_group_4": span(best, "group", "recall", 4),
        "best_recall_cell_6": span(best, "cell", "recall", 6),
        "best_recall_group_6": span(best, "group", "recall", 6),
        "op_f1": f"{op['f1']:.4f}", "op_gap": f"{op['f1_gap_to_bar']:.4f}",
        "best_f1": f"{best['f1']:.4f}", "best_gap": f"{best['f1_gap_to_bar']:.4f}",
        "hold_deff": f"{hold['design_effect_group']:.1f}",
        "hold_corrected_4": f"{hold['corrected']['low']:.4f}-{hold['corrected']['high']:.4f}",
        "hold_cell_4": f"{hold['cell_level']['low']:.4f}-{hold['cell_level']['high']:.4f}",
        "hold_bootstrap_4": f"{hold['group_cluster']['low']:.4f}-{hold['group_cluster']['high']:.4f}",
        "hold_bootstrap_6": f"{hold['group_cluster']['low']:.6f}-{hold['group_cluster']['high']:.6f}",
        "hold_corrected_6": f"{hold['corrected']['low']:.6f}-{hold['corrected']['high']:.6f}",
        "queue_exact": f"{payload['identity']['queue']['successes']}/{payload['identity']['queue']['trials']}",
        "hold_exact": f"{hold['successes']}/{hold['trials']}",
        "f1_bar": "0.70", "auc_bar": f"{ranking['bar']:.2f}",
    }
    for name in ("operating_point", "best_point"):
        for level in ("board", "group"):
            strings[f"icc_{name}_{level}"] = f"{clustering[name][level]['icc']:.6f}"
            strings[f"deff_{name}_{level}"] = f"{clustering[name][level]['design_effect']:.2f}"
    return strings


# Each document must still carry the qualifiers it quotes (V-056's standing instrument).
PUBLISHED = {
    "docs/M3_WIDENED_DECIDING_POPULATION.md": (
        "boards", "groups", "known_cells", "occupied_cells", "empty_cells", "cells_per_board",
        "icc_operating_point_board", "icc_operating_point_group", "icc_best_point_board",
        "icc_best_point_group", "deff_operating_point_board", "deff_operating_point_group",
        "deff_best_point_board", "deff_best_point_group", "precision_design_effect_range",
        "standard_error_inflation_range", "width_ratio_range", "width_ratio_operating",
        "width_ratio_best", "auc", "auc_cell_6", "auc_group_6", "auc_width_ratio",
        "op_precision_cell_6", "op_precision_group_6", "op_recall_cell_6", "op_recall_group_6",
        "best_precision_cell_6", "best_precision_group_6", "best_recall_cell_6",
        "best_recall_group_6", "op_f1", "op_gap", "best_f1", "best_gap",
        "effective_occupied_operating", "effective_occupied_best", "effective_cells_range",
        "queue_exact", "hold_exact", "hold_deff", "hold_corrected_6", "hold_bootstrap_6",
        "auc_bar", "f1_bar"),
    "docs/CONTINUATION_STATE.md": (
        "precision_design_effect_range", "standard_error_inflation_range", "width_ratio_operating",
        "width_ratio_best", "effective_occupied_operating",
        "icc_operating_point_board", "icc_operating_point_group", "icc_best_point_board",
        "icc_best_point_group", "auc_group_4", "op_precision_group_4", "best_precision_group_4",
        "hold_bootstrap_4", "hold_deff", "queue_exact", "hold_exact", "auc_bar", "f1_bar"),
    "docs/ROADMAP.md": (
        "precision_design_effect_range", "standard_error_inflation_range",
        "effective_occupied_operating", "auc_group_4", "op_precision_group_4",
        "op_recall_group_4", "auc_bar"),
    "docs/BUG_LOG.md": (
        "precision_design_effect_range_2dp", "standard_error_inflation_range", "width_ratio_operating",
        "width_ratio_best", "auc_width_ratio", "effective_occupied_operating",
        "icc_operating_point_board", "icc_operating_point_group", "icc_best_point_board",
        "icc_best_point_group", "deff_operating_point_board", "deff_operating_point_group",
        "deff_best_point_board", "deff_best_point_group"),
}


def check_published(payload, root=ROOT):
    """Fail loudly when a quoted qualifier no longer matches the committed measurement."""
    strings, failures = published_strings(payload), []
    unknown = sorted({key for keys in PUBLISHED.values() for key in keys} - set(strings))
    if unknown:
        raise ValueError("published table names qualifiers that are never computed: "+", ".join(unknown))
    for name, keys in PUBLISHED.items():
        text = (root/name).read_text(encoding="utf-8")
        for key in keys:
            if strings[key] not in text:
                failures.append(f"{name}: {key} -> {strings[key]!r} not published")
    if failures:
        raise ValueError("published quantifiers drifted from the measurement: "+"; ".join(failures))
    return {"documents": len(PUBLISHED), "strings": sum(len(keys) for keys in PUBLISHED.values())}


def main():
    declare = {}
    declare["memory_bytes"] = require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("evidence directory already exists")
    if not (ROOT/args.receipt).is_file() and Path(args.receipt).suffix != ".json":
        raise ValueError("declared receipt must be a job receipt path")

    inputs = {name: read(path) for name, path in INPUTS.items()}
    scope, sweep, identity, narrow = inputs["scope"], inputs["sweep"], inputs["identity"], inputs["narrow_sweep"]
    for name, expected in scope["artifact_sha256"].items():
        if digest(EVIDENCE/name) != expected:
            raise ValueError("widened-population evidence artifact changed")
    if (sweep["weights_sha256"] != scope["weights_sha256"]["occupancy"] or
            identity["weights_sha256"] != scope["weights_sha256"]["identity"]):
        raise ValueError("clustering addendum would measure different weights")

    boards = boards_of(sweep)
    cells = cells_of(boards)
    published = scope["occupancy"]
    expected_population = (scope["populations"]["after"]["nonempty_images"],
                           published["ranking"]["positives"]+published["ranking"]["negatives"])
    if (len(boards), len(cells)) != expected_population:
        raise ValueError("clustering addendum population differs from the published one")
    groups = {board["group"] for board in boards}
    if len(groups) != scope["populations"]["after"]["ancestry_groups"]:
        raise ValueError("ancestry clusters differ from the published group count")

    point_auc = pooled_auc([cell["logit"] for cell in cells if cell["label"]],
                           [cell["logit"] for cell in cells if not cell["label"]])
    if abs(point_auc-published["ranking"]["auc"]) > 1e-12:
        raise ValueError("AUC does not reproduce the published point estimate")
    cell_auc = auc_interval(point_auc, published["ranking"]["positives"], published["ranking"]["negatives"])
    if cell_auc != published["ranking"]["auc_interval"]:
        raise ValueError("cell-level AUC interval does not reproduce the published one")

    ranking = {"point": point_auc, "cell_level": cell_auc, "bar": published["ranking"]["bar"],
               "verdict": published["ranking"]["verdict"],
               "board_cluster": cluster_bootstrap(auc_statistic, clusters_of(boards, "board"),
                                                  ("auc",), draws=AUC_DRAWS)["auc"],
               "group_cluster": cluster_bootstrap(auc_statistic, clusters_of(boards, "group"),
                                                  ("auc",), draws=AUC_DRAWS)["auc"]}
    ranking["design_effect"] = {
        level: round((ranking[level+"_cluster"]["standard_error"]/cell_auc["standard_error"])**2, 6)
        for level in ("board", "group")}

    best_threshold = published["ranking"]["best"]["threshold"]
    occupancy = {
        "ranking_auc": ranking,
        "operating_point": gate_block(boards, cells, OPERATING_THRESHOLD,
                                      published["replay_operating_point"],
                                      published["replay_operating_point"]["intervals"],
                                      "replay operating point"),
        "best_point": gate_block(boards, cells, best_threshold, published["ranking"]["best"],
                                 published["ranking"]["best_intervals"], "swept best-F1 point")}
    for key, source in (("operating_point", published["replay_operating_point"]),
                        ("best_point", published["ranking"]["best"])):
        for field in ("true_positive", "false_positive", "false_negative"):
            if occupancy[key][field] != source[field]:
                raise ValueError(f"{key} counts do not reproduce the published ones")

    narrow_boards = boards_of(narrow)
    narrow_cells = cells_of(narrow_boards)
    narrow_auc = pooled_auc([cell["logit"] for cell in narrow_cells if cell["label"]],
                            [cell["logit"] for cell in narrow_cells if not cell["label"]])
    if abs(narrow_auc-scope["superseded_without_invalidation"]["sweep_auc"]) > 1e-12:
        raise ValueError("narrow AUC does not reproduce the superseded point estimate")
    narrow_block = {
        "boards": len(narrow_boards), "groups": len({board["group"] for board in narrow_boards}),
        "point": narrow_auc,
        "cell_level": auc_interval(narrow_auc, sum(int(cell["label"]) for cell in narrow_cells),
                                   sum(1-int(cell["label"]) for cell in narrow_cells)),
        "board_cluster": cluster_bootstrap(auc_statistic, clusters_of(narrow_boards, "board"),
                                           ("auc",), draws=AUC_DRAWS)["auc"],
        "note": "seven boards over three ancestry groups; group resampling is uninformative"}

    clustering = {"operating_point": icc_block(cells, OPERATING_THRESHOLD),
                  "best_point": icc_block(cells, best_threshold),
                  "note": ("ICC(1) is the one-way variance-components estimate of the per-cell "
                           "error indicator; the design effect is 1+(mean cluster size-1)*ICC")}
    addendum = {
        "status": "passed",
        "addendum": "widened-deciding-population-clustering-r1",
        "kind": ("methodological addendum to the widened-population predeclaration; not a sweep, "
                 "lever, threshold or data change"),
        "predeclaration": "docs/M3_WIDENED_DECIDING_POPULATION.md",
        "scope": ("Measures the within-board and within-ancestry-group clustering of the widened "
                  "deciding population's per-cell errors and reports a correlation-respecting 95% "
                  "interval beside every published cell-level interval. No gate, threshold, point "
                  "estimate, model, loss, seed or prior evidence byte changes; no real-image, "
                  "complete-state or quality claim."),
        "inputs": {name: path.relative_to(ROOT).as_posix() for name, path in INPUTS.items()},
        "input_sha256": {name: digest(path) for name, path in INPUTS.items()},
        "declared_receipt": args.receipt,
        "declared_memory_bytes": declare["memory_bytes"],
        "bootstrap": {"method": "whole-cluster percentile resampling over sorted cluster ids",
                      "seed": BOOTSTRAP_SEED, "auc_draws": AUC_DRAWS, "count_draws": COUNT_DRAWS,
                      "note": "degenerate draws are skipped; each draw resamples as many clusters as exist"},
        "population": {"boards": len(boards), "ancestry_groups": len(groups), "known_cells": len(cells),
                       "occupied_cells": sum(int(cell["label"]) for cell in cells),
                       "empty_cells": sum(1-int(cell["label"]) for cell in cells),
                       "cells_per_board": round(len(cells)/len(boards), 3)},
        "clustering": clustering,
        "occupancy": occupancy,
        "identity": identity_block(identity, scope),
        "narrow_like_for_like": narrow_block,
        "unchanged": {"point_estimates": True, "thresholds": scope["unchanged_thresholds"],
                      "frozen_manifest": "not touched", "prior_evidence": "byte-identical"},
    }
    addendum["quantifiers"] = quantify(addendum)
    addendum["published_check"] = check_published(addendum)
    args.output.mkdir(parents=True)
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(addendum, stream, indent=2, allow_nan=False)
    print(json.dumps(addendum, indent=2))


if __name__ == "__main__":
    main()
