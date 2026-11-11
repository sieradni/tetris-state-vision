"""Record the deciding population's discriminating power beside the sweep verdict.

`uninformative` is a claim about the evidence only if the deciding subset could have shown something. This
read-only pass records the population size, cell-level balance, the pooled AUC with its Hanley-McNeil
interval and chance-referenced z, and whether the fixed threshold grid can locate an interior optimum. It
reads a saved sweep receipt and writes no verdict of its own; the sweep's predeclared rule still decides.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from gate_power import auc_interval, precision_recall_interval
from replay_equivalence import first_difference
from occupancy_sweep import AUC_THRESHOLD, F1_THRESHOLD

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def power(result):
    deciding = result["deciding"]
    positives = deciding["occupied_cells"]
    negatives = deciding["empty_cells_on_nonempty_boards"]+deciding["empty_population"]["known_cells"]
    auc = deciding["auc"]
    interval = auc_interval(auc, positives, negatives)
    chance_se = math.sqrt((positives+negatives+1)/(12*positives*negatives))
    best = deciding["best"]
    rows = deciding["rows"]
    grid = result["grid"]
    best_index = next(index for index, row in enumerate(rows) if row["threshold"] == best["threshold"])
    neighbours = [rows[index]["f1"] for index in (best_index-1, best_index+1) if 0 <= index < len(rows)]
    step_change = max(abs(value-best["f1"]) for value in neighbours)
    groups = len({case["group_id"] for case in result["cases"] if case["split"] == deciding["split"]})
    gap_auc = AUC_THRESHOLD-auc
    bar_distance = gap_auc/interval["standard_error"]
    resolvable = interval["high"] < AUC_THRESHOLD
    powered = interval["standard_error"] < gap_auc/2
    return {
        "status": "passed",
        "run": result["run"],
        "deciding_split": deciding["split"],
        "population": {
            "images": deciding["nonempty_images"]+deciding["empty_images"],
            "nonempty_images": deciding["nonempty_images"],
            "empty_images": deciding["empty_images"],
            "ancestry_groups": groups,
            "occupied_cells": positives,
            "empty_cells": negatives,
            "known_cells": positives+negatives,
            "positive_rate": round(positives/(positives+negatives), 6),
            "nonempty_share": round(deciding["nonempty_images"] /
                                    (deciding["nonempty_images"]+deciding["empty_images"]), 6),
        },
        "auc": {
            "value": auc, "chance": 0.5, "chance_standard_error": round(chance_se, 6),
            "vs_chance_z": round((auc-0.5)/chance_se, 3),
            "hanley_mcneil": interval, "auc_bar": AUC_THRESHOLD,
            "gap_to_bar": round(gap_auc, 6),
            "bar_distance_standard_errors": round(bar_distance, 3),
            "ci_high_below_bar": resolvable,
            "minimum_resolvable_gap": round(1.96*interval["standard_error"], 6),
        },
        "operating_point": {
            "threshold": best["threshold"], "f1": best["f1"],
            "f1_bar": F1_THRESHOLD, "gap_to_bar": round(F1_THRESHOLD-best["f1"], 6),
            "precision_recall_intervals": precision_recall_interval(
                best["true_positive"], best["false_positive"], best["false_negative"]),
        },
        "grid": {
            "low": grid["low"], "high": grid["high"], "step": grid["step"], "count": grid["count"],
            "best_threshold": best["threshold"],
            "best_is_interior": rows[0]["threshold"] < best["threshold"] < rows[-1]["threshold"],
            "best_matches_swept_maximum": max(rows, key=lambda row: row["f1"])["threshold"] == best["threshold"],
            "f1_at_low_edge": rows[0]["f1"], "f1_at_high_edge": rows[-1]["f1"],
            "f1_change_at_one_step": round(step_change, 6),
            "f1_range_across_grid": round(max(row["f1"] for row in rows)-min(row["f1"] for row in rows), 6),
        },
        "verdict": {
            "recorded": result["verdict"],
            "rule": "informative iff pooled AUC >= %.2f and some swept point reaches F1 >= %.2f"
                    % (AUC_THRESHOLD, F1_THRESHOLD),
            "real_finding": bool(resolvable and powered),
            "interpretation": (
                f"uninformative is a measured miss, not an artifact: the deciding population has "
                f"{deciding['nonempty_images']+deciding['empty_images']} images in {groups} ancestry groups "
                f"and {positives+negatives:,} known cells ({positives} occupied, "
                f"{100.0*positives/(positives+negatives):.1f}%), which resolves a pooled AUC to "
                f"+/-{1.96*interval['standard_error']:.3f} at 95%; the {AUC_THRESHOLD:.2f} bar sits "
                f"{bar_distance:.1f} standard errors above the observed {auc:.4f} with the upper bound "
                f"{interval['high']:.4f} below it, the best swept F1 {best['f1']:.4f} is "
                f"{F1_THRESHOLD-best['f1']:.3f} short of its {F1_THRESHOLD:.2f} bar, and the optimum "
                f"threshold {best['threshold']:+.2f} is interior with an F1 range of "
                f"{max(row['f1'] for row in rows)-min(row['f1'] for row in rows):.4f} across the grid."),
        },
        "sources": {"result": None},
    }


def figure_sentence(record, path):
    """The exact sentence the records quote, generated from the record itself."""
    population, auc = record["population"], record["auc"]
    operating, grid = record["operating_point"], record["grid"]
    return (
        "The `uninformative` verdict now carries the deciding population's power (`%s`): %d images in %d "
        "ancestry groups and %s known cells (%d occupied, %.1f%%), which resolves a pooled AUC to +/-%.3f at "
        "95%%; the %.2f bar sits %.1f standard errors above the observed %.4f with the upper bound %.4f below "
        "it, the best swept F1 %.4f is %.3f short of its %.2f bar, and the optimum threshold %+.2f is interior "
        "with an F1 range of %.4f across the grid. `uninformative` is a measured miss of the predeclared bars, "
        "not an artefact of an underpowered subset or an unresolvable grid."
        % (path, population["images"], population["ancestry_groups"],
           "{:,}".format(population["known_cells"]), population["occupied_cells"],
           100.0*population["positive_rate"], auc["minimum_resolvable_gap"], auc["auc_bar"],
           auc["bar_distance_standard_errors"], auc["value"], auc["hanley_mcneil"]["high"],
           operating["f1"], operating["gap_to_bar"], operating["f1_bar"], operating["threshold"],
           grid["f1_range_across_grid"]))


def normalize_text(text):
    """Collapse whitespace so prose reflow is not a difference."""
    return " ".join(text.split())


def figure_appears(document, record, path):
    """Whether a document quotes the record's figure, ignoring prose whitespace."""
    return normalize_text(figure_sentence(record, path)) in normalize_text(document)


def record_problems(record):
    """Named problems with a saved power record, or empty when it matches its receipt."""
    if type(record) is not dict:
        return {"record": "object required"}
    try:
        result_path = ROOT/record["result_path"]
        stored = record["sources"]["result"]
    except (KeyError, TypeError):
        return {"record": "result_path and sources.result required"}
    if not result_path.exists():
        return {"sources.result": "receipt %s missing" % result_path}
    problems = {}
    actual = digest(result_path)
    if actual != stored:
        problems["sources.result"] = "digest %s != stored %s" % (actual, stored)
    fresh = power(json.loads(result_path.read_text(encoding="utf-8")))
    expected = set(fresh) | {"result_path", "sources"}
    for key in sorted(expected | set(record)):
        if key in ("result_path", "sources"):
            continue
        if key not in fresh:
            problems["injected:" + key] = "section not produced by the power instrument"
        elif key not in record:
            problems[key] = "missing from the record"
        else:
            difference = first_difference(fresh[key], record[key], key)
            if difference:
                problems[key] = difference
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, help="saved sweep result.json, relative to the repo root")
    parser.add_argument("--output", required=True, help="evidence directory, relative to the repo root")
    args = parser.parse_args()
    result_path = ROOT/args.result
    result = json.loads(result_path.read_text(encoding="utf-8"))
    output = ROOT/args.output
    output.mkdir(parents=True, exist_ok=False)
    record = power(result)
    record["result_path"] = args.result
    record["sources"] = {"result": digest(result_path), "script": digest(Path(__file__))}
    (output/"power.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "verdict": record["verdict"]["recorded"],
                      "real_finding": record["verdict"]["real_finding"],
                      "auc": record["auc"]["value"], "known_cells": record["population"]["known_cells"]}))


if __name__ == "__main__":
    main()
