"""Score the M28 pixel-query arms on the M27 matched population (read-only).

The M28 verdict was computed on each arm's own 3,316-cell co-scoreable set, while every
baseline it is compared with (0.3791 / 0.5440 / 0.7626 / 0.9872) was measured on the M27
matched population. This reader rebuilds that population from the committed record and
scores both trained arms on it, so the comparison needs no caveat.

The population is one fixed set of 2,986 (board, model cell) rows, taken from the M27
arm's own registration receipt: every cell whose committed mapping lands on a known
target cell. Two label vectors are then fixed for all arms because they are read from
that same receipt:

- ``identity`` - the truth at the row's own target index, which is what M27 scored as the
  model at its own lattice (0.439024 / 0.543974);
- ``anchored`` - the truth at the target cell the committed registration pairs with the
  row, which is what M27 scored as registration granted (0.574181 / 0.762590).

Only the logits move between arms. The arms' own mappings are deliberately not used:
they cover 1,861 and 1,583 of the reference targets respectively, so a target-space
re-score would silently become a per-arm population - the thing this pass exists to
remove. The pixel read-out is a function of the raster and the labels, so it is the same
on all arms (0.974763 / 0.987179) and anchors the far end of the span.

No model, no forward pass, no training, no weight, threshold, gate or label change.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from oracle_pairing_read import MATCHED_CELLS, nonempty, require_close, score  # noqa: E402
from probe_model import require_bounded_job  # noqa: E402

REFERENCE_OWN = 0.543974
REFERENCE_PIXEL = 0.987179
CELL_GATE = 0.70
AUC_GATE = 0.90
ATTRIBUTION_FLOOR = 0.05
PIXEL_FLOOR = 0.90
PLACEMENT_FLOOR = 0.10
CEILING_BYTES = 128*1024**2
SPAN = REFERENCE_PIXEL-REFERENCE_OWN


def read(path):
    return json.loads(Path(path).read_text())


def population(registration):
    """Fixed rows: (board, model cell, identity label, anchored label)."""
    rows = []
    for case in nonempty(registration["cases"]):
        for index in range(200):
            target = case["mapping"][index]
            if target is None or not case["known"][target]:
                continue
            rows.append((case["sample_id"], index, target, bool(case["occupied"][index]),
                         bool(case["occupied"][target])))
    return rows


def rows_for(cases, fixed, pairing):
    """The arm's logits on the fixed rows, with the fixed label vector."""
    by_board = {case["sample_id"]: case for case in cases}
    rows = []
    for sample_id, index, _, identity, anchored in fixed:
        logits = by_board[sample_id]["logits"]
        rows.append((float(logits[index]), identity if pairing == "identity" else anchored))
    return rows


def pixel_rows(pixel_cases, fixed):
    """The pixel read-out on the fixed rows, labelled by the anchored truth."""
    by_board = {case["sample_id"]: case for case in pixel_cases}
    rows = []
    for sample_id, _, target, _, anchored in fixed:
        rows.append((float(by_board[sample_id]["pixel_logits"][target]), anchored))
    return rows


def coverage(arm, reference):
    """Fixed rows whose reference target cell appears in the arm's own mapping."""
    mapping = {case["sample_id"]: set(value for value in case["mapping"] if value is not None)
               for case in arm["cases"]}
    covered = 0
    for case in nonempty(reference["cases"]):
        for index in range(200):
            target = case["mapping"][index]
            if target is None or not case["known"][target]:
                continue
            covered += target in mapping.get(case["sample_id"], set())
    return covered


def receipt_arm(name, suffix):
    data = read(ROOT/"runs"/(name+suffix)/"result.json")
    job = read(ROOT/"runs"/(name+"-job.json"))
    if data.get("status") != "passed" or job.get("status") != "passed":
        raise ValueError(f"{name}: a committed receipt this read joins did not pass")
    if data.get("forwards") != 36:
        raise ValueError(f"{name}: the probe is not the 36-board population")
    if len(data["cases"]) != 36:
        raise ValueError(f"{name}: not the declared deciding population")
    return data, job


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--pixel-ceiling", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--lever", required=True)
    parser.add_argument("--registration-suffix", default="-registration-r1")
    parser.add_argument("--previous-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = read(ROOT/"runs"/args.reference/"result.json")
    pixel = read(ROOT/"runs"/args.pixel_ceiling/"result.json")
    if reference.get("status") != "passed" or pixel.get("status") != "passed":
        raise ValueError("the committed reference receipts did not pass")
    fixed = population(reference)
    if len(fixed) != MATCHED_CELLS:
        raise ValueError(f"the fixed population is {len(fixed)} cells, not {MATCHED_CELLS}")
    if len({(sample, index) for sample, index, _, _, _ in fixed}) != MATCHED_CELLS:
        raise ValueError("the fixed population has duplicate rows")
    anchors = {sample for sample, _, _, _, _ in fixed}
    if len(anchors) != 28:
        raise ValueError("the fixed population is not the 28 nonempty boards")
    if sum(anchored for _, _, _, _, anchored in fixed) == 0:
        raise ValueError("the anchored label vector has no occupied cell")
    pixel_read = score(pixel_rows(pixel["cases"], fixed))
    reference_own = score(rows_for(reference["cases"], fixed, "identity"))
    reference_anchored = score(rows_for(reference["cases"], fixed, "anchored"))
    published = reference["summary"]
    require_close("reference own F1", reference_own["f1_at_zero"], published["as_is"]["f1"])
    require_close("reference own AUC", reference_own["auc"], published["as_is"]["auc"])
    require_close("reference anchored F1", reference_anchored["f1_at_zero"],
                  published["oracle"]["f1"])
    require_close("reference anchored AUC", reference_anchored["auc"], published["oracle"]["auc"])
    if abs(pixel_read["best"]["f1"]-REFERENCE_PIXEL) > 1e-6:
        raise ValueError("the pixel read does not reproduce the committed 0.987179")
    if abs(reference_own["best"]["f1"]-REFERENCE_OWN) > 1e-6:
        raise ValueError("the reference own read does not reproduce the committed 0.543974")

    previous = read(ROOT/args.previous_evidence/"scope.json")
    arms = {}
    for key in ("base", "lever"):
        data, job = receipt_arm(getattr(args, key), args.registration_suffix)
        arms[key] = {"registration": data, "job": job,
                     "own": score(rows_for(data["cases"], fixed, "identity")),
                     "anchored": score(rows_for(data["cases"], fixed, "anchored"))}
    if (arms["base"]["registration"]["weights_sha256"] ==
            arms["lever"]["registration"]["weights_sha256"]):
        raise ValueError("the two arms carry identical weights")
    if arms["base"]["job"]["peak_job_commit_bytes"] > 1152*1024**2:
        raise ValueError("an arm exceeded the declared study ceiling")

    arithmetic = {}
    for key, payload in arms.items():
        own = payload["own"]["best"]["f1"]
        anchored = payload["anchored"]["best"]["f1"]
        closed = own-REFERENCE_OWN
        placement = anchored-own
        representation = REFERENCE_PIXEL-anchored
        closed_fraction = closed/SPAN
        placement_fraction = placement/SPAN
        representation_fraction = representation/SPAN
        if abs(closed_fraction+placement_fraction+representation_fraction-1.0) > 1e-9:
            raise ValueError(f"{key}: the closed span and its split do not add up to one span")
        if abs((own-REFERENCE_OWN)+(anchored-own)+(REFERENCE_PIXEL-anchored)-SPAN) > 1e-9:
            raise ValueError(f"{key}: the point arithmetic does not reproduce the span")
        arithmetic[key] = {
            "own_best": round(own, 6), "anchored_best": round(anchored, 6),
            "pixel_best": round(pixel_read["best"]["f1"], 6),
            "span_points": round(closed, 6), "placement_points": round(placement, 6),
            "representation_points": round(representation, 6),
            "remaining_points": round(REFERENCE_PIXEL-own, 6),
            "span_closed_fraction": round(closed_fraction, 6),
            "placement_fraction": round(placement_fraction, 6),
            "representation_fraction": round(representation_fraction, 6),
            "remaining_fraction": round(1.0-closed_fraction, 6),
            "fractions_sum_to_one": True,
        }
        arithmetic[key]["split_sums_to_span"] = True
        arithmetic[key]["residual"] = ("representation" if representation > placement
                                       else "placement")
    base, lever = arithmetic["base"], arithmetic["lever"]
    attribution = lever["own_best"]-base["own_best"]
    gates = {"per_cell_f1": arms["lever"]["own"]["f1_at_zero"] >= CELL_GATE,
             "ranking": (arms["lever"]["own"]["auc"] >= AUC_GATE and
                         arms["lever"]["own"]["best"]["f1"] >= CELL_GATE),
             "attribution": attribution >= ATTRIBUTION_FLOOR}
    cleared = all(gates.values())
    anchored_delta = lever["anchored_best"]-base["anchored_best"]
    auc_delta = arms["lever"]["own"]["auc"]-arms["base"]["own"]["auc"]
    f1_delta = arms["lever"]["own"]["f1_at_zero"]-arms["base"]["own"]["f1_at_zero"]
    # The cause is read from the within-arm placement term, never from a cross-arm anchored
    # delta: anchoring can lower one arm and raise another for reasons that are not placement
    # (it also swaps the label vector to the 309 target-anchored cells), so a cross-arm
    # comparison of anchored reads cannot attribute anything.
    if cleared:
        cause = None
    elif lever["placement_fraction"] >= PLACEMENT_FLOOR:
        cause = "placement-limited"
    elif attribution >= ATTRIBUTION_FLOOR:
        cause = "representation-limited-partial-gain"
    elif pixel_read["best"]["f1"] >= PIXEL_FLOOR:
        cause = "features-unavailable-at-own-lattice"
    else:
        cause = "head-cannot-exploit"

    result = {
        "status": "passed", "instrument": "m28-pixel-query-deciding-r1",
        "predeclaration": "docs/M28_PIXEL_QUERY.md",
        "reference_receipt": args.reference, "pixel_ceiling_receipt": args.pixel_ceiling,
        "base_run": args.base, "lever_run": args.lever,
        "population": {
            "fixed_cells": len(fixed), "boards": len(anchors),
            "occupied_identity": sum(1 for _, _, _, identity, _ in fixed if identity),
            "occupied_anchored": sum(1 for _, _, _, _, anchored in fixed if anchored),
            "definition": ("the M27 arm's own registration receipt, cells whose committed mapping "
                           "lands on a known target cell; index-fixed, so every arm is scored on "
                           "identical rows and identical label vectors"),
            "reference_targets_covered_by_arm": {
                key: coverage(arms[key]["registration"], reference) for key in arms},
            "coverage_note": ("how many of the fixed rows' reference target cells appear anywhere in "
                              "an arm's own mapping; below the fixed cell count, which is why the "
                              "rows are index-fixed and the arms' own mappings are not used"),
        },
        "reads": {
            "pixel": pixel_read,
            "reference_own": reference_own, "reference_anchored": reference_anchored,
            "base_own": arms["base"]["own"], "base_anchored": arms["base"]["anchored"],
            "lever_own": arms["lever"]["own"], "lever_anchored": arms["lever"]["anchored"],
        },
        "references": {"own_best": REFERENCE_OWN, "pixel_best": REFERENCE_PIXEL,
                       "own_f1_at_zero": round(reference_own["f1_at_zero"], 6),
                       "own_auc": round(reference_own["auc"], 6),
                       "anchored_best": round(reference_anchored["best"]["f1"], 6),
                       "pixel_f1_at_zero": round(pixel_read["f1_at_zero"], 6),
                       "pixel_auc": round(pixel_read["auc"], 6)},
        "arithmetic": {"span": round(SPAN, 6), "arms": arithmetic,
                       "attribution_best_delta": round(attribution, 6),
                       "attribution_auc_delta": round(auc_delta, 6),
                       "attribution_f1_at_zero_delta": round(f1_delta, 6),
                       "anchored_best_delta": round(anchored_delta, 6),
                       "control_relative_span_closed": round(attribution/SPAN, 6),
                       "placement_floor": PLACEMENT_FLOOR,
                       "cause_source": ("within-arm placement fraction, not the cross-arm anchored "
                                        "delta: anchoring also swaps the label vector")},
        "gates": {"conditions": gates, "cell_bar": CELL_GATE, "auc_bar": AUC_GATE,
                  "attribution_floor": ATTRIBUTION_FLOOR,
                  "verdict": "cleared" if cleared else "not-cleared"},
        "verdict": "cleared" if cleared else "not-cleared", "cause": cause,
        "secondary_per_arm_population": previous["reads"],
        "secondary_arithmetic": previous["attribution"],
        "secondary_population_cells": previous["population"]["matched_cells"],
        "secondary_note": ("the 3,316-cell per-arm read retained from m28-pixel-query-r2; it is "
                           "per-arm, so it is not cross-arm comparable and is not used for any "
                           "verdict in this receipt"),
        "weights_sha256": {"base": arms["base"]["registration"]["weights_sha256"],
                           "lever": arms["lever"]["registration"]["weights_sha256"]},
        "study_peaks": {key: arms[key]["job"]["peak_job_commit_bytes"] for key in arms},
        "scope": ("Read-only re-score of two already-trained arms on one fixed population anchored "
                  "to the committed matched cells; no forward pass, no training, no weight, "
                  "threshold, gate or label change, and no accuracy, real-image or complete-state "
                  "claim."),
    }
    job = read(ROOT/"runs"/(args.output.name+"-job.json"))
    if job.get("memory_bytes") != CEILING_BYTES:
        raise ValueError("the reader ceiling differs from the declared 128 MiB")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"scope.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                                          encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"], "cause": cause,
        "population": {"cells": result["population"]["fixed_cells"],
                       "occupied_identity": result["population"]["occupied_identity"],
                       "occupied_anchored": result["population"]["occupied_anchored"]},
        "reads": {name: {"auc": round(value["auc"], 6), "f1_at_zero": round(value["f1_at_zero"], 6),
                         "best": round(value["best"]["f1"], 6),
                         "threshold": value["best"]["threshold"]}
                  for name, value in result["reads"].items()},
        "arithmetic": arithmetic, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
