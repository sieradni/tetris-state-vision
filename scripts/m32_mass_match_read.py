"""Score the M32 mass-shaping study on the fixed population (read-only).

`docs/M32_MASS_MATCH.md` declares the term and the unchanged gate. This joins the study's probe to
the campaign's fixed 2,986 cells and both label vectors, scores the **unchanged** global `0.0`
boundary, reports the gate, and re-derives the closure/placement/representation split with its
arithmetic asserted before the receipt is written. No oracle, no per-board boundary, no fitted
threshold anywhere.

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

    control, control_job = arm(args.control, "-registration-r1")
    control_own = score(rows_for(control["cases"], fixed, "identity"))
    require_close("control own", control_own["best"]["f1"], CONTROL_BEST)

    anchor_probe, anchor_job = arm(args.anchor, args.anchor_registration)
    anchor_own = score(rows_for(anchor_probe["cases"], fixed, "identity"))

    probe, job = arm(args.arm, args.registration)
    own = score(rows_for(probe["cases"], fixed, "identity"))
    anchored = score(rows_for(probe["cases"], fixed, "anchored"))
    full = score(pairs(probe["cases"], pairing="as_is"))
    span = REFERENCE_PIXEL-REFERENCE_OWN
    closed = own["best"]["f1"]-REFERENCE_OWN
    placement = anchored["best"]["f1"]-own["best"]["f1"]
    representation = REFERENCE_PIXEL-anchored["best"]["f1"]
    if abs((closed+placement+representation)-span) > 1e-9:
        raise ValueError("the point arithmetic does not reproduce the span")
    if abs((closed/span)+(placement/span)+(representation/span)-1.0) > 1e-9:
        raise ValueError("the closed span and its split do not sum to one span")
    gates = {"per_cell_f1": own["f1_at_zero"] >= CELL_GATE,
             "ranking": own["auc"] >= AUC_GATE and own["best"]["f1"] >= CELL_GATE,
             "attribution": (own["best"]["f1"]-control_own["best"]["f1"]) >= ATTRIBUTION_FLOOR}
    short = min({"closure": closed, "placement": placement, "representation": representation}.items(),
                key=lambda item: item[1])[0]
    if job["peak_job_commit_bytes"] > 1152*1024**2:
        raise ValueError("the study exceeded the declared ceiling")
    result = {
        "status": "passed", "instrument": "m32-mass-match-r1",
        "predeclaration": "docs/M32_MASS_MATCH.md",
        "population": {"fixed_cells": len(fixed), "occupied_identity": 214,
                       "occupied_anchored": 309, "boards": 28},
        "arm": {"run": args.arm, "registration": args.registration,
                "weights_sha256": probe["weights_sha256"], "job_name": job.get("job_name")},
        "anchors": {"anchor_run": args.anchor,
                    "anchor_f1_at_zero": round(anchor_own["f1_at_zero"], 6),
                    "anchor_auc": round(anchor_own["auc"], 6),
                    "anchor_best": round(anchor_own["best"]["f1"], 6),
                    "control_best": round(control_own["best"]["f1"], 6),
                    "reference_own_best": REFERENCE_OWN, "pixel_best": REFERENCE_PIXEL,
                    "span": round(span, 6)},
        "own": own, "anchored": anchored, "full": full,
        "own_best": round(own["best"]["f1"], 6),
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
        "gates": gates, "cleared": all(gates.values()), "short_stage": short,
        "peak_bytes": job["peak_job_commit_bytes"],
        "scope": ("Read-only scoring of one committed study on the fixed population under the "
                  "unchanged global 0.0 boundary, with no oracle and no fitted threshold; no "
                  "real-image or complete-state claim."),
    }
    receipt = json.loads((ROOT/"runs"/(args.output.name+"-job.json")).read_text())
    if receipt.get("memory_bytes") != CEILING_BYTES:
        raise ValueError("the reader ceiling differs from the declared 128 MiB")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"scope.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                                          encoding="utf-8")
    print(json.dumps({"gates": gates, "cleared": result["cleared"],
                      "f1_at_zero": round(own["f1_at_zero"], 6), "auc": round(own["auc"], 6),
                      "best": result["own_best"], "short_stage": short,
                      "gain_over_anchor_f1": result["gain_over_anchor_f1"],
                      "gain_over_anchor_best": result["gain_over_anchor_best"]}, indent=2))


if __name__ == "__main__":
    main()
