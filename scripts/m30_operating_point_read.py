"""Score the declared M30 level mechanisms on the fixed population (read-only).

`docs/M30_OPERATING_POINT.md` declares this instrument. Every variant is scored on the identical
2,986-cell population with both fixed label vectors, against the M29 winner re-scored here from its
own committed probe, so no anchor number is taken on trust. The reader then applies the declared
confirmation floor and, for the best confirmed mechanism, the full campaign gate, with the span
arithmetic asserted before the receipt is written.

No model, no forward pass, no training, no weight, threshold, gate or label change.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from oracle_pairing_read import (MATCHED_CELLS, nonempty, pairs, require_close,  # noqa: E402
                                 score)
from pixel_query_deciding_read import population, rows_for  # noqa: E402
from probe_model import require_bounded_job  # noqa: E402

REFERENCE_OWN = 0.543974
REFERENCE_PIXEL = 0.987179
CONTROL_BEST = 0.219626
GAIN_FLOOR = 0.05
CELL_GATE = 0.70
AUC_GATE = 0.90
ATTRIBUTION_FLOOR = 0.05
CEILING_BYTES = 128*1024**2
GOALS = ROOT/"docs/evidence/m29-pixel-form-r2/scope.json"


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
    parser.add_argument("--forms", nargs="+", required=True,
                        help="name:run:registration triples")
    parser.add_argument("--anchor", default="m29-pixelform-hidden-r2")
    parser.add_argument("--anchor-registration", default="-registration-r2")
    parser.add_argument("--control", default="m28-pixelread-base-r1")
    parser.add_argument("--reference", default="m3-occscale-registration-r2")
    parser.add_argument("--pixel-ceiling", default="m3-occpix-ceiling-r1-r2")
    parser.add_argument("--m29", type=Path, default=GOALS)
    parser.add_argument("--tensors", type=Path,
                        default=ROOT/"docs/evidence/m30-level-tensors-r1/scope.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = read(ROOT/"runs"/args.reference/"result.json")
    pixel = read(ROOT/"runs"/args.pixel_ceiling/"result.json")
    m29 = read(args.m29)
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
    committed = m29["variants"]["hidden-r2"]["own"]
    for key, value in (("f1_at_zero", anchor_own["f1_at_zero"]),
                       ("auc", anchor_own["auc"]), ("best", anchor_own["best"]["f1"])):
        stored = committed["best"]["f1"] if key == "best" else committed[key]
        require_close(f"anchor {key}", value, stored)

    variants = {}
    for spec in args.forms:
        parts = spec.split(":")
        name, run = parts[0], parts[1]
        registration = parts[2] if len(parts) > 2 else "-registration-r1"
        probe, job = arm(run, registration)
        own = score(rows_for(probe["cases"], fixed, "identity"))
        anchored = score(rows_for(probe["cases"], fixed, "anchored"))
        full = score(pairs(probe["cases"], pairing="as_is"))
        span = REFERENCE_PIXEL-REFERENCE_OWN
        closed = own["best"]["f1"]-REFERENCE_OWN
        placement = anchored["best"]["f1"]-own["best"]["f1"]
        representation = REFERENCE_PIXEL-anchored["best"]["f1"]
        if abs((closed+placement+representation)-span) > 1e-9:
            raise ValueError(f"{name}: the point arithmetic does not reproduce the span")
        if abs((closed/span)+(placement/span)+(representation/span)-1.0) > 1e-9:
            raise ValueError(f"{name}: the closed span and its split do not sum to one span")
        variants[name] = {
            "run": run, "weights_sha256": probe["weights_sha256"],
            "own": own, "anchored": anchored, "full": full,
            "own_best": round(own["best"]["f1"], 6),
            "gain_over_anchor": round(own["f1_at_zero"]-anchor_own["f1_at_zero"], 6),
            "gain_over_anchor_best": round(own["best"]["f1"]-anchor_own["best"]["f1"], 6),
            "auc_delta_vs_anchor": round(own["auc"]-anchor_own["auc"], 6),
            "gain_over_control": round(own["best"]["f1"]-control_own["best"]["f1"], 6),
            "span_points": round(closed, 6), "placement_points": round(placement, 6),
            "representation_points": round(representation, 6),
            "span_closed_fraction": round(closed/span, 6),
            "placement_fraction": round(placement/span, 6),
            "representation_fraction": round(representation/span, 6),
            "fractions_sum_to_one": True,
            "peak_bytes": job["peak_job_commit_bytes"], "job_name": job.get("job_name"),
        }
        if job["peak_job_commit_bytes"] > 1152*1024**2:
            raise ValueError(f"{name}: the study exceeded the declared ceiling")

    tensors = read(args.tensors)
    if tensors.get("status") != "passed":
        raise ValueError("the declared level-parameter read did not pass")
    for name in variants:
        reading = tensors["forms"].get(name)
        if reading is None:
            raise ValueError(f"{name}: no level-parameter reading")
        variants[name]["level_parameters"] = reading
        if reading["ordering_preserved"] is False:
            raise ValueError(f"{name}: the level mechanism reordered cells")

    confirmed = {name: value for name, value in variants.items()
                 if value["gain_over_anchor"] >= GAIN_FLOOR}
    winner = (max(confirmed.values(), key=lambda value: value["own_best"])
              if confirmed else None)
    gates = None
    if winner is not None:
        gates = {"per_cell_f1": winner["own"]["f1_at_zero"] >= CELL_GATE,
                 "ranking": (winner["own"]["auc"] >= AUC_GATE and
                             winner["own"]["best"]["f1"] >= CELL_GATE),
                 "attribution": winner["gain_over_control"] >= ATTRIBUTION_FLOOR}
    if winner is None:
        cause = "level:no-mechanism"
    elif gates["per_cell_f1"] and gates["ranking"] and gates["attribution"]:
        cause = "level:"+",".join(sorted(name for name, value in confirmed.items()
                                        if value["own_best"] == winner["own_best"]))
    else:
        cause = "level-inconclusive:"+",".join(sorted(
            name for name, value in confirmed.items() if value["own_best"] == winner["own_best"]))
    result = {
        "status": "passed", "instrument": "m30-operating-point-r1",
        "predeclaration": "docs/M30_OPERATING_POINT.md",
        "population": {"fixed_cells": len(fixed), "occupied_identity": 214,
                       "occupied_anchored": 309, "boards": 28},
        "anchors": {"reference_own_best": REFERENCE_OWN, "pixel_best": REFERENCE_PIXEL,
                    "control_best": round(control_own["best"]["f1"], 6),
                    "anchor_run": args.anchor,
                    "anchor_f1_at_zero": round(anchor_own["f1_at_zero"], 6),
                    "anchor_auc": round(anchor_own["auc"], 6),
                    "anchor_best": round(anchor_own["best"]["f1"], 6),
                    "span": round(REFERENCE_PIXEL-REFERENCE_OWN, 6)},
        "variants": variants,
        "anchor": {"run": args.anchor, "own": anchor_own, "job_name": anchor_job.get("job_name")},
        "control": {"run": args.control, "own": control_own,
                    "job_name": control_job.get("job_name")},
        "reference": {"run": args.reference, "own": reference_own},
        "pixel": pixel_read,
        "decision": {"cause": cause, "gain_floor": GAIN_FLOOR,
                     "confirmed": sorted(confirmed),
                     "excluded": sorted(name for name in variants if name not in confirmed),
                     "winner": None if winner is None else winner["run"],
                     "gates": gates, "cleared": None if gates is None else all(gates.values()),
                     "reading": ("A monotone calibration that lifts F1 at 0.0 while leaving AUC "
                                 "unchanged is evidence about the gate's operating point, not a "
                                 "pass of the ranking claim.")},
        "scope": ("Read-only scoring of committed arms on one fixed population plus one "
                  "checkpoint-scalar read; no training, no weight, threshold, gate or label "
                  "change, and no accuracy, real-image or complete-state claim."),
    }
    job = read(ROOT/"runs"/(args.output.name+"-job.json"))
    if job.get("memory_bytes") != CEILING_BYTES:
        raise ValueError("the reader ceiling differs from the declared 128 MiB")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"scope.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                                          encoding="utf-8")
    print(json.dumps({"cause": cause, "gates": gates,
                      "variants": {name: {"f1_at_zero": round(value["own"]["f1_at_zero"], 6),
                                          "auc": round(value["own"]["auc"], 6),
                                          "best": value["own_best"],
                                          "gain": value["gain_over_anchor"]}
                                   for name, value in variants.items()}}, indent=2))


if __name__ == "__main__":
    main()
