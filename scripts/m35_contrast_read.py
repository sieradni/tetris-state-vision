"""Score the M35 contrast read and diagnose closure if it falls short (read-only).

`docs/M35_CONTRAST_READ.md` fixes the bar before scoring: this arm is the operating point only if it
matches or beats the winner's re-scored F1 at `0.0` and best swept F1. If it does not, this read also
records where the per-cell decision fails - the error mix at the fixed boundary, the margins each error
class holds, and the row/column distribution of the errors - so the next structural step attacks a
measured cause rather than a guess.

No model, no forward pass, no training, no weight, threshold, gate or label change.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from oracle_pairing_read import MATCHED_CELLS, pairs, require_close, score  # noqa: E402
from pixel_query_deciding_read import population, rows_for  # noqa: E402
from probe_model import require_bounded_job  # noqa: E402

REFERENCE_OWN = 0.543974
REFERENCE_PIXEL = 0.987179
CONTROL_BEST = 0.219626
CELL_GATE = 0.70
AUC_GATE = 0.90
ATTRIBUTION_FLOOR = 0.05
CEILING_BYTES = 128*1024**2
BANDS = ((0, 4), (5, 9), (10, 14), (15, 19), (20, 25))


def read(path):
    return json.loads(Path(path).read_text())


def arm(name, registration):
    probe = read(ROOT/"runs"/(name+registration)/"result.json")
    job = read(ROOT/"runs"/(name+"-job.json"))
    if probe.get("status") != "passed" or job.get("status") != "passed":
        raise ValueError(f"{name}: a committed receipt this read joins did not pass")
    if probe.get("forwards") != 36:
        raise ValueError(f"{name}: the probe is not the 36-board population")
    return probe, job


def gates_for(read_, control_best):
    return {"per_cell_f1": read_["f1_at_zero"] >= CELL_GATE,
            "ranking": read_["auc"] >= AUC_GATE and read_["best"]["f1"] >= CELL_GATE,
            "attribution": (read_["best"]["f1"]-control_best) >= ATTRIBUTION_FLOOR}


def margins(probe, fixed):
    specific = {case["sample_id"]: case["pixel_logits"] for case in probe["cases"]}
    rows = []
    for sample, row, column, _, _ in fixed:
        rows.append((row, column, float(specific[sample][row][column])))
    return rows


def diagnose(probe, fixed):
    cells = margins(probe, fixed)
    truth = {(row, column): bool(identity) for _, row, column, identity, _ in fixed}
    errors = {"true_positive": [], "false_positive": [], "false_negative": [], "true_negative": []}
    for row, column, margin in cells:
        positive, occupied = margin > 0, truth[(row, column)]
        key = ("true_" if positive == occupied else "false_")+("positive" if positive else "negative")
        errors[key].append((row, column, margin))

    def summary(points):
        if not points:
            return {"count": 0}
        values = sorted(value for _, _, value in points)
        pick = lambda fraction: round(values[min(len(values)-1, int(fraction*len(values)))], 6)
        return {"count": len(points), "margin_low": pick(0.0), "margin_median": pick(0.5),
                "margin_high": pick(0.999)}

    bands = {}
    for low, high in BANDS:
        inside = [(row, column) for row, column, _ in errors["false_positive"] if low <= row <= high]
        truth_row = [(row, column) for _, row, column, identity, _ in fixed if identity and low <= row <= high]
        bands[f"rows_{low}_{high}"] = {
            "false_positives": len(inside),
            "occupied_cells": len(truth_row),
            "false_positive_rate": round(len(inside)/max(len(truth_row), 1), 6)}
    columns = {str(column): sum(1 for row, col, _ in errors["false_positive"] if col == column)
               for column in range(10)}
    f1_at_zero = score([(margin, truth[(row, column)]) for row, column, margin in cells])["f1_at_zero"]
    return {
        "errors_at_zero": {key: summary(points) for key, points in errors.items()},
        "false_positive_bands": bands,
        "false_positives_by_column": columns,
        "precision_at_zero": round(sum(1 for r, c, m in cells if m > 0 and truth[(r, c)]) /
                                   max(sum(1 for _, _, m in cells if m > 0), 1), 6),
        "recall_at_zero": round(sum(1 for r, c, m in cells if m > 0 and truth[(r, c)]) /
                                max(sum(1 for r, c, _ in cells if truth[(r, c)]), 1), 6),
        "f1_at_zero_recomputed": round(f1_at_zero, 6),
        "worst_false_positive": sorted(errors["false_positive"], key=lambda item: -item[2])[:5],
        "closest_false_negative": sorted(errors["false_negative"], key=lambda item: -item[2])[:5],
        "note": ("Margins are the read's own logits at the scored cell, so they show whether the error is a "
                 "boundary placement (large but wrong-signed margins) or a ranking overlap (errors packed "
                 "near zero). Truth here is the fixed identity vector used for the per-cell condition."),
    }


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--anchor", default="m30-level-calibrate-r5")
    parser.add_argument("--anchor-registration", default="-registration")
    parser.add_argument("--control", default="m28-pixelread-base-r1")
    parser.add_argument("--reference", default="m3-occscale-registration-r2")
    parser.add_argument("--pixel-ceiling", default="m3-occpix-ceiling-r1-r2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = read(ROOT/"runs"/args.reference/"result.json")
    pixel = read(ROOT/"runs"/args.pixel_ceiling/"result.json")
    fixed = population(reference)
    if len(fixed) != MATCHED_CELLS:
        raise ValueError("the fixed population is not the declared 2,986 cells")
    if (sum(1 for _, _, _, identity, _ in fixed if identity),
            sum(1 for _, _, _, _, anchored in fixed if anchored)) != (214, 309):
        raise ValueError("the label vectors are not the declared 214/309 occupied cells")
    pixel_read = score([(float(next(c for c in pixel["cases"] if c["sample_id"] == sample)
                               ["pixel_logits"][target]), anchored)
                        for sample, _, target, _, anchored in fixed])
    require_close("pixel read", pixel_read["best"]["f1"], REFERENCE_PIXEL)
    reference_own = score(rows_for(reference["cases"], fixed, "identity"))
    require_close("reference own", reference_own["best"]["f1"], REFERENCE_OWN)

    control, _ = arm(args.control, "-registration-r1")
    control_own = score(rows_for(control["cases"], fixed, "identity"))
    require_close("control own", control_own["best"]["f1"], CONTROL_BEST)

    anchor_probe, _ = arm(args.anchor, args.anchor_registration)
    anchor_own = score(rows_for(anchor_probe["cases"], fixed, "identity"))

    probe, job = arm(args.arm, args.registration)
    own = score(rows_for(probe["cases"], fixed, "identity"))
    anchored = score(rows_for(probe["cases"], fixed, "anchored"))
    span = REFERENCE_PIXEL-REFERENCE_OWN
    closed = own["best"]["f1"]-REFERENCE_OWN
    placement = anchored["best"]["f1"]-own["best"]["f1"]
    representation = REFERENCE_PIXEL-anchored["best"]["f1"]
    if abs((closed+placement+representation)-span) > 1e-9:
        raise ValueError("the point arithmetic does not reproduce the span")
    if abs((closed/span)+(placement/span)+(representation/span)-1.0) > 1e-9:
        raise ValueError("the closed span and its split do not sum to one span")
    gates, anchored_gates = gates_for(own, control_own["best"]["f1"]), gates_for(
        anchored, control_own["best"]["f1"])
    clears_bar = (own["f1_at_zero"] >= anchor_own["f1_at_zero"] and
                  own["best"]["f1"] >= anchor_own["best"]["f1"])
    diagnosis = diagnose(probe, fixed)
    require_close("diagnosis recomputes the per-cell read", diagnosis["f1_at_zero_recomputed"],
                  own["f1_at_zero"], tolerance=1e-6)
    result = {
        "status": "passed", "instrument": "m35-contrast-lattice-r1",
        "predeclaration": "docs/M35_CONTRAST_READ.md",
        "population": {"fixed_cells": len(fixed), "occupied_identity": 214,
                       "occupied_anchored": 309, "boards": 28},
        "arm": {"run": args.arm, "registration": args.registration,
                "weights_sha256": probe["weights_sha256"], "job_name": job.get("job_name"),
                "inputs": 60, "features": 30},
        "anchors": {"anchor_run": args.anchor,
                    "anchor_f1_at_zero": round(anchor_own["f1_at_zero"], 6),
                    "anchor_auc": round(anchor_own["auc"], 6),
                    "anchor_best": round(anchor_own["best"]["f1"], 6),
                    "control_best": round(control_own["best"]["f1"], 6),
                    "reference_own_best": REFERENCE_OWN, "pixel_best": REFERENCE_PIXEL,
                    "span": round(span, 6)},
        "own": own, "anchored": anchored, "full": pairs(probe["cases"], pairing="as_is"),
        "own_best": round(own["best"]["f1"], 6),
        "anchored_best": round(anchored["best"]["f1"], 6),
        "gain_over_anchor_f1": round(own["f1_at_zero"]-anchor_own["f1_at_zero"], 6),
        "gain_over_anchor_best": round(own["best"]["f1"]-anchor_own["best"]["f1"], 6),
        "auc_delta_vs_anchor": round(own["auc"]-anchor_own["auc"], 6),
        "gain_over_control": round(own["best"]["f1"]-control_own["best"]["f1"], 6),
        "span_points": round(closed, 6), "placement_points": round(placement, 6),
        "representation_points": round(representation, 6),
        "span_closed_fraction": round(closed/span, 6),
        "placement_fraction": round(placement/span, 6),
        "representation_fraction": round(representation/span, 6),
        "fractions_sum_to_one": True,
        "gates": gates, "gates_anchored": anchored_gates,
        "cleared": all(gates.values()) and all(anchored_gates.values()),
        "bar": {"f1_at_zero": round(anchor_own["f1_at_zero"], 6),
                "best": round(anchor_own["best"]["f1"], 6), "met": bool(clears_bar)},
        "diagnosis": diagnosis,
        "short_stage": min({"closure": closed, "placement": placement,
                            "representation": representation}.items(),
                           key=lambda item: item[1])[0],
        "peak_bytes": job["peak_job_commit_bytes"],
        "scope": ("Read-only scoring of one committed study on the fixed population under the unchanged "
                  "global 0.0 boundary, with no oracle and no fitted threshold; the diagnosis reads the "
                  "arm's own logits only."),
    }
    receipt = json.loads((ROOT/"runs"/(args.output.name+"-job.json")).read_text())
    if receipt.get("memory_bytes") != CEILING_BYTES:
        raise ValueError("the reader ceiling differs from the declared 128 MiB")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"scope.json").write_text(json.dumps({k: v for k, v in result.items()
                                                      if k != "diagnosis"},
                                                     indent=2, allow_nan=False)+"\n",
                                          encoding="utf-8")
    (args.output/"diagnosis.json").write_text(json.dumps(diagnosis, indent=2, allow_nan=False)+"\n",
                                              encoding="utf-8")
    print(json.dumps({"gates": gates, "gates_anchored": anchored_gates,
                      "f1_at_zero": round(own["f1_at_zero"], 6), "auc": round(own["auc"], 6),
                      "best": result["own_best"], "anchor_f1": result["anchors"]["anchor_f1_at_zero"],
                      "anchor_best": result["anchors"]["anchor_best"],
                      "bar_met": result["bar"]["met"], "short_stage": result["short_stage"],
                      "split": [result["span_points"], result["placement_points"],
                                result["representation_points"]],
                      "diagnosis": {"errors": diagnosis["errors_at_zero"],
                                    "precision": diagnosis["precision_at_zero"],
                                    "recall": diagnosis["recall_at_zero"],
                                    "bands": diagnosis["false_positive_bands"]}}, indent=2))


if __name__ == "__main__":
    main()
