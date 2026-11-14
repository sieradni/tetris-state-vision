"""Discriminate the causes of the pixel-query collapse (read-only, M29).

`docs/M29_PIXEL_QUERY_FORM.md` declares this instrument. It scores the committed reference arm,
the unchanged M28 control and one trained variant per candidate mechanism on the **identical**
fixed 2,986-cell population and the same two label vectors the deciding receipt uses, so the
comparison needs no population caveat. It also reads the committed form-1 checkpoint's tensors to
test the amplitude (gradient-starvation) explanation arithmetically, before any variant result is
interpreted.

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
FORM_ONE_BEST = 0.425703
CONTROL_BEST = 0.219626
SPAN = REFERENCE_PIXEL-REFERENCE_OWN
GAIN_FLOOR = 0.05
AMPLITUDE_FLOOR = 0.05
CELL_GATE = 0.70
AUC_GATE = 0.90
ATTRIBUTION_FLOOR = 0.05
CEILING_BYTES = 128*1024**2
FORM_ONE = "m28-pixelread-lever-r1"


def read(path):
    return json.loads(Path(path).read_text())


def arm(name, registration_suffix="-registration-r1"):
    probe = read(ROOT/"runs"/(name+registration_suffix)/"result.json")
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
                        help="name:form_label pairs, e.g. centred:2 hidden:3 neighbours:4")
    parser.add_argument("--control", default="m28-pixelread-base-r1")
    parser.add_argument("--reference", default="m3-occscale-registration-r2")
    parser.add_argument("--pixel-ceiling", default="m3-occpix-ceiling-r1-r2")
    parser.add_argument("--deciding", type=Path,
                        default=ROOT/"docs/evidence/m28-pixel-query-deciding-r3/scope.json")
    parser.add_argument("--amplitude", type=Path,
                        default=ROOT/"docs/evidence/m29-amplitude-r1",
                        help="the committed amplitude diagnostic this read joins")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = read(ROOT/"runs"/args.reference/"result.json")
    pixel = read(ROOT/"runs"/args.pixel_ceiling/"result.json")
    deciding = read(args.deciding)
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

    control, control_job = arm(args.control)
    control_own = score(rows_for(control["cases"], fixed, "identity"))
    require_close("control own", control_own["best"]["f1"], CONTROL_BEST)
    require_close("control own F1", control_own["f1_at_zero"],
                  deciding["reads"]["base_own"]["f1_at_zero"])

    variants = {}
    for spec in args.forms:
        parts = spec.split(":")
        name, form = parts[0], parts[1]
        suffix = ("-registration-"+parts[2]) if len(parts) > 2 else "-registration-r1"
        probe, job = arm(f"m29-pixelform-{name}", suffix)
        own = score(rows_for(probe["cases"], fixed, "identity"))
        anchored = score(rows_for(probe["cases"], fixed, "anchored"))
        full = score(pairs(probe["cases"], pairing="as_is"))
        closed = own["best"]["f1"]-REFERENCE_OWN
        placement = anchored["best"]["f1"]-own["best"]["f1"]
        representation = REFERENCE_PIXEL-anchored["best"]["f1"]
        closed_fraction = closed/SPAN
        placement_fraction = placement/SPAN
        representation_fraction = representation/SPAN
        if abs(closed_fraction+placement_fraction+representation_fraction-1.0) > 1e-9:
            raise ValueError(f"{name}: the closed span and its split do not sum to one span")
        if abs((own["best"]["f1"]-REFERENCE_OWN)+(anchored["best"]["f1"]-own["best"]["f1"])+
               (REFERENCE_PIXEL-anchored["best"]["f1"])-SPAN) > 1e-9:
            raise ValueError(f"{name}: the point arithmetic does not reproduce the span")
        variants[name] = {
            "form": int(form), "run": f"m29-pixelform-{name}",
            "weights_sha256": probe["weights_sha256"],
            "own": own, "anchored": anchored, "full": full,
            "own_best": round(own["best"]["f1"], 6),
            "gain_over_form_one": round(own["best"]["f1"]-FORM_ONE_BEST, 6),
            "gain_over_control": round(own["best"]["f1"]-control_own["best"]["f1"], 6),
            "auc_gain_over_control": round(own["auc"]-control_own["auc"], 6),
            "f1_gain_over_control": round(own["f1_at_zero"]-control_own["f1_at_zero"], 6),
            "span_points": round(closed, 6),
            "placement_points": round(placement, 6),
            "representation_points": round(representation, 6),
            "span_closed_fraction": round(closed_fraction, 6),
            "placement_fraction": round(placement_fraction, 6),
            "representation_fraction": round(representation_fraction, 6),
            "fractions_sum_to_one": True,
            "residual": "representation" if representation > placement else "placement",
            "peak_bytes": job["peak_job_commit_bytes"],
            "job_name": job.get("job_name"),
        }
        if job["peak_job_commit_bytes"] > 1152*1024**2:
            raise ValueError(f"{name}: the study exceeded the declared ceiling")

    amplitude_reading = read(ROOT/args.amplitude/"scope.json")
    if amplitude_reading.get("status") != "passed":
        raise ValueError("the amplitude diagnostic did not pass")
    if amplitude_reading.get("run") != FORM_ONE:
        raise ValueError("the amplitude diagnostic read a different checkpoint")
    winners = {name: value for name, value in variants.items()
               if value["gain_over_form_one"] >= GAIN_FLOOR}
    if amplitude_reading["starved"]:
        cause = "gradient-starvation"
    elif winners:
        best = max(winners.values(), key=lambda value: value["own_best"])
        cause = "form:"+",".join(sorted(name for name, value in winners.items()
                                        if value["own_best"] == best["own_best"]))
    else:
        cause = "additive-form-expressivity"
    winner = (None if not winners else
              max(winners.values(), key=lambda value: value["own_best"]))
    gates = None
    if winner is not None:
        gates = {"per_cell_f1": winner["own"]["f1_at_zero"] >= CELL_GATE,
                 "ranking": (winner["own"]["auc"] >= AUC_GATE and
                             winner["own"]["best"]["f1"] >= CELL_GATE),
                 "attribution": winner["gain_over_control"] >= ATTRIBUTION_FLOOR}
    result = {
        "status": "passed", "instrument": "m29-pixel-form-r1",
        "predeclaration": "docs/M29_PIXEL_QUERY_FORM.md",
        "population": {"fixed_cells": len(fixed), "occupied_identity": 214,
                       "occupied_anchored": 309, "boards": 28,
                       "anchoring_receipt": str(args.deciding)},
        "anchors": {"form_one_best": FORM_ONE_BEST, "control_best": CONTROL_BEST,
                    "reference_own_best": REFERENCE_OWN, "pixel_best": REFERENCE_PIXEL,
                    "span": round(SPAN, 6)},
        "amplitude": {key: amplitude_reading[key] for key in
                      ("relative_gain", "starved", "reading", "amplitude_floor",
                       "projection_weight_norm", "projection_bias_norm",
                       "head_weight_norm", "statistics_to_logit_gain_norm",
                       "run", "checkpoint", "tensors", "job_name")},
        "variants": variants,
        "control": {"run": args.control, "own": control_own,
                    "own_best": round(control_own["best"]["f1"], 6),
                    "job_name": control_job.get("job_name")},
        "reference": {"run": args.reference, "own": reference_own},
        "pixel": pixel_read,
        "decision": {"cause": cause, "gain_floor": GAIN_FLOOR,
                     "confirmed": sorted(winners), "excluded": sorted(
                         name for name in variants if name not in winners),
                     "winner": None if winner is None else winner["run"],
                     "gates": gates,
                     "cleared": None if gates is None else all(gates.values())},
        "scope": ("Read-only mechanistic read of committed arms on one fixed population plus one "
                  "checkpoint-tensor amplitude read; no forward pass, no training, no weight, "
                  "threshold, gate or label change, and no accuracy, real-image or complete-state "
                  "claim."),
    }
    job = read(ROOT/"runs"/(args.output.name+"-job.json"))
    if job.get("memory_bytes") != CEILING_BYTES:
        raise ValueError("the reader ceiling differs from the declared 128 MiB")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"scope.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                                          encoding="utf-8")
    print(json.dumps({"cause": cause, "amplitude": amplitude_reading,
                      "variants": {name: {"own_best": value["own_best"],
                                          "gain_over_form_one": value["gain_over_form_one"],
                                          "auc": round(value["own"]["auc"], 6),
                                          "f1_at_zero": round(value["own"]["f1_at_zero"], 6)}
                                   for name, value in variants.items()},
                      "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
