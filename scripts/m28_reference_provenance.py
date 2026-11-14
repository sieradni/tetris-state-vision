"""Pin the population and arm behind every baseline the M28 table cites (read-only).

`docs/M28_PIXEL_QUERY.md` compares the M28 arms against `0.3791` F1 at `0.0`, `0.5440` best
swept F1, `0.7626` registration-granted and `0.9872` pixel read. Those figures were quoted as
one row in prose, which reads as a single population. This reader establishes, from the
committed receipts and by recomputation rather than by copying their own summary, which
population and which arm produced each one - and whether the M28 control's own-lattice read
(`0.129870` on the fixed matched population) contradicts the baseline it is set against.

It replays nothing: the reference arm's registration receipt carries per-case logits, labels,
the known mask and the mapping, and the pixel-ceiling receipt carries per-case pixel logits on
the same boards. From those it recomputes

- the full deciding read (`5,600` known cells, `667` occupied): `0.3791` / `0.8006`;
- the matched identity read (`2,986` cells, `214` occupied): `0.439024` / `0.933138`;
- the matched registration-granted read (`2,986` cells, `309` occupied): `0.574181` / `0.970233`;
- the pixel read on the same cells (`0.974763` / `0.987179`) and on the full set (`0.978006`).

and the M28 control's own full-population read, which must reproduce its committed sweep before
this pass may claim its matched read is a population effect rather than a defect.

No model, no forward pass, no training, no weight, threshold, gate or label change.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from oracle_pairing_read import (MATCHED_CELLS, matched_reads, nonempty, pairs,  # noqa: E402
                                 require_close, score)
from pixel_query_deciding_read import population, rows_for  # noqa: E402
from probe_model import require_bounded_job  # noqa: E402

FULL_CELLS = 5600
FULL_OCCUPIED = 667
MATCHED_IDENTITY_OCCUPIED = 214
MATCHED_ANCHORED_OCCUPIED = 309
CEILING_BYTES = 128*1024**2


def read(path):
    return json.loads(Path(path).read_text())


def receipt(name):
    data = read(ROOT/"runs"/name/"result.json")
    job = read(ROOT/"runs"/(name+"-job.json"))
    if data.get("status") != "passed" or job.get("status") != "passed":
        raise ValueError(f"{name}: a committed receipt this read joins did not pass")
    return data, job


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="m3-occscale-registration-r2")
    parser.add_argument("--pixel-ceiling", default="m3-occpix-ceiling-r1-r2")
    parser.add_argument("--control", default="m28-pixelread-base-r1")
    parser.add_argument("--control-sweep", default="m28-pixelread-base-r1-sweep-r1")
    parser.add_argument("--deciding", required=True,
                        help="the deciding receipt this provenance repairs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference, reference_job = receipt(args.reference)
    pixel, pixel_job = receipt(args.pixel_ceiling)
    control, control_job = receipt(args.control+"-registration-r1")
    control_sweep = read(ROOT/"runs"/args.control_sweep/"result.json")
    deciding = read(ROOT/"docs/evidence"/args.deciding/"scope.json")
    if deciding.get("status") != "passed":
        raise ValueError("the deciding receipt being repaired did not pass")
    if len(pixel.get("cases", [])) != 28 or control.get("forwards") != 36:
        raise ValueError("a receipt is not the declared board population")
    cases, pixel_cases = reference["cases"], pixel["cases"]
    if {case["sample_id"] for case in pixel_cases} != {case["sample_id"] for case in nonempty(cases)}:
        raise ValueError("the pixel receipt is not on the reference boards")

    matched = matched_reads(cases, pixel_cases)
    for name, rows in matched.items():
        if len(rows) != MATCHED_CELLS:
            raise ValueError(f"the {name} read is not on the matched population")
    full = score(pairs(cases, pairing="as_is"))
    if full["cells"] != FULL_CELLS or full["occupied"] != FULL_OCCUPIED:
        raise ValueError("the full deciding read is not the 5,600-cell population")
    if len([1 for _, label in matched["as_is"] if label]) != MATCHED_IDENTITY_OCCUPIED:
        raise ValueError("the matched identity label vector is not the declared 214 cells")
    if len([1 for _, label in matched["oracle"] if label]) != MATCHED_ANCHORED_OCCUPIED:
        raise ValueError("the matched anchored label vector is not the declared 309 cells")

    reference_full = score(pairs(cases, pairing="as_is"))
    reference_matched = score(matched["as_is"])
    reference_anchored = score(matched["oracle"])
    pixel_matched = score(matched["pixel"])
    pixel_full = score([(float(case["pixel_logits"][index]), bool(reference_case["occupied"][index]))
                        for reference_case, case in
                        ((next(c for c in nonempty(cases) if c["sample_id"] == case["sample_id"]), case)
                         for case in pixel_cases)
                        for index in range(200) if reference_case["known"][index]])
    published = reference["summary"]
    require_close("reference full F1", reference_full["f1_at_zero"], published["as_is_full"]["f1"])
    require_close("reference full AUC", reference_full["auc"], published["as_is_full"]["auc"])
    require_close("reference matched F1", reference_matched["f1_at_zero"], published["as_is"]["f1"])
    require_close("reference matched AUC", reference_matched["auc"], published["as_is"]["auc"])
    require_close("reference anchored F1", reference_anchored["f1_at_zero"],
                  published["oracle"]["f1"])
    require_close("reference anchored AUC", reference_anchored["auc"], published["oracle"]["auc"])
    require_close("pixel full F1", pixel_full["f1_at_zero"], pixel["measured"]["operating"]["f1"])
    require_close("pixel full AUC", pixel_full["auc"], pixel["measured"]["pooled_auc"])

    control_full = score(pairs(control["cases"], pairing="as_is"))
    require_close("control full F1", control_full["f1_at_zero"],
                  control_sweep["deciding"]["rows"][
                      [row["threshold"] for row in control_sweep["deciding"]["rows"]].index(0.0)]["f1"])
    require_close("control full AUC", control_full["auc"], control_sweep["deciding"]["auc"])
    fixed = population(reference)
    if len(fixed) != MATCHED_CELLS:
        raise ValueError("the fixed population is not the declared 2,986 cells")
    control_matched = score(rows_for(control["cases"], fixed, "identity"))
    require_close("control matched F1", control_matched["f1_at_zero"],
                  deciding["reads"]["base_own"]["f1_at_zero"])
    require_close("control matched AUC", control_matched["auc"],
                  deciding["reads"]["base_own"]["auc"])

    references = {
        "0.3791": {"figure": round(reference_full["f1_at_zero"], 6),
                   "measured_as": "F1 at threshold 0.0, own lattice",
                   "population": f"full deciding set, {FULL_CELLS} known cells, {FULL_OCCUPIED} occupied",
                   "population_name": "full-5600", "arm": args.reference,
                   "arm_weights_sha256": reference["weights_sha256"],
                   "label_vector": "target cell at the cell's own index (as_is)",
                   "source_receipt": f"runs/{args.reference}/result.json",
                   "source_block": "summary.as_is_full",
                   "job_receipt": f"runs/{args.reference}-job.json",
                   "job_name": reference_job.get("job_name")},
        "0.439024": {"figure": round(reference_matched["f1_at_zero"], 6),
                     "measured_as": "F1 at threshold 0.0, own lattice",
                     "population": f"matched set, {MATCHED_CELLS} cells, {MATCHED_IDENTITY_OCCUPIED} occupied",
                     "population_name": "matched-2986", "arm": args.reference,
                     "arm_weights_sha256": reference["weights_sha256"],
                     "label_vector": "target cell at the cell's own index (as_is)",
                     "source_receipt": f"runs/{args.reference}/result.json",
                     "source_block": "summary.as_is",
                     "job_receipt": f"runs/{args.reference}-job.json",
                     "job_name": reference_job.get("job_name")},
        "0.5440": {"figure": round(reference_matched["best"]["f1"], 6),
                   "measured_as": "best swept F1 over the standing grid",
                   "population": f"matched set, {MATCHED_CELLS} cells, {MATCHED_IDENTITY_OCCUPIED} occupied",
                   "population_name": "matched-2986", "arm": args.reference,
                   "arm_weights_sha256": reference["weights_sha256"],
                   "label_vector": "target cell at the cell's own index (as_is)",
                   "source_receipt": "docs/evidence/m27-three-factor-r3/scope.json",
                   "source_block": "reads.model_matched_as_is.best.f1",
                   "recomputed_from": f"runs/{args.reference}/result.json",
                   "job_receipt": f"runs/{args.reference}-job.json",
                   "job_name": reference_job.get("job_name")},
        "0.7626": {"figure": round(reference_anchored["best"]["f1"], 6),
                   "measured_as": "best swept F1 with the committed registration granted",
                   "population": f"matched set, {MATCHED_CELLS} cells, {MATCHED_ANCHORED_OCCUPIED} occupied",
                   "population_name": "matched-2986", "arm": args.reference,
                   "arm_weights_sha256": reference["weights_sha256"],
                   "label_vector": "target cell the committed mapping points at (oracle)",
                   "source_receipt": "docs/evidence/m27-three-factor-r3/scope.json",
                   "source_block": "reads.model_oracle.best.f1",
                   "recomputed_from": f"runs/{args.reference}/result.json",
                   "job_receipt": f"runs/{args.reference}-job.json",
                   "job_name": reference_job.get("job_name")},
        "0.9872": {"figure": round(pixel_matched["best"]["f1"], 6),
                   "measured_as": "best swept F1 of the 30-feature pixel read-out",
                   "population": f"matched set, {MATCHED_CELLS} cells, {MATCHED_ANCHORED_OCCUPIED} occupied",
                   "population_name": "matched-2986", "arm": args.pixel_ceiling + " (arm-independent read)",
                   "arm_weights_sha256": pixel.get("weights_sha256"),
                   "label_vector": "target cell the committed mapping points at (oracle)",
                   "source_receipt": f"runs/{args.pixel_ceiling}/result.json",
                   "source_block": "cases[].pixel_logits with the reference labels",
                   "job_receipt": f"runs/{args.pixel_ceiling}-job.json",
                   "job_name": pixel_job.get("job_name")},
        "0.978006": {"figure": round(pixel_full["f1_at_zero"], 6),
                     "measured_as": "F1 at threshold 0.0 of the pixel read-out",
                     "population": f"full deciding set, {FULL_CELLS} known cells, {FULL_OCCUPIED} occupied",
                     "population_name": "full-5600", "arm": args.pixel_ceiling + " (arm-independent read)",
                     "arm_weights_sha256": pixel.get("weights_sha256"),
                     "label_vector": "target cell at the cell's own index (as_is)",
                     "source_receipt": f"runs/{args.pixel_ceiling}/result.json",
                     "source_block": "measured.operating",
                     "job_receipt": f"runs/{args.pixel_ceiling}-job.json",
                     "job_name": pixel_job.get("job_name")},
    }
    control_entry = {
        "arm": args.control, "arm_weights_sha256": control["weights_sha256"],
        "full_f1_at_zero": round(control_full["f1_at_zero"], 6),
        "full_auc": round(control_full["auc"], 6),
        "full_population": f"{FULL_CELLS} known cells, {FULL_OCCUPIED} occupied",
        "full_source": f"runs/{args.control_sweep}/result.json",
        "full_job_name": control_job.get("job_name"),
        "matched_f1_at_zero": round(control_matched["f1_at_zero"], 6),
        "matched_auc": round(control_matched["auc"], 6),
        "matched_population": f"{MATCHED_CELLS} cells, {MATCHED_IDENTITY_OCCUPIED} occupied",
        "reproduces_its_own_committed_full_read": True,
        "note": ("the control's own committed full-population F1 at 0.0 is its sweep's deciding value, "
                 "which this read recomputes exactly; its matched-population value is lower because "
                 "the matched cells are the reference arm's co-scoreable subset, not because the "
                 "read path differs"),
    }
    result = {
        "status": "passed", "instrument": "m28-reference-provenance-r1",
        "question": ("which population and arm produced the 0.3791 / 0.5440 / 0.7626 / 0.9872 "
                     "baselines the M28 table is scored against"),
        "answer": {
            "0.3791": "reference arm, full 5,600-cell deciding population - not the M28 control",
            "0.5440": "reference arm, matched 2,986-cell co-scoreable population, best swept",
            "0.7626": "reference arm, matched 2,986-cell population, registration granted",
            "0.9872": "pixel read-out, matched 2,986-cell population",
            "table_sound": True,
            "control_defect": False,
            "reason": ("0.3791 is the reference arm's own-lattice read on the full deciding set; the "
                       "control is a different checkpoint with different weights, and its own committed "
                       "full-population read is reproduced exactly here before its matched read is used"),
        },
        "references": references,
        "recomputed": {
            "reference_full": reference_full, "reference_matched": reference_matched,
            "reference_anchored": reference_anchored, "pixel_matched": pixel_matched,
            "pixel_full": pixel_full, "control_full": control_full,
        },
        "control": control_entry,
        "deciding_receipt": args.deciding,
        "weights_sha256": {"reference": reference["weights_sha256"], "pixel": pixel.get("weights_sha256"),
                           "control": control["weights_sha256"]},
        "jobs": {args.reference: reference_job.get("job_name"),
                 args.pixel_ceiling: pixel_job.get("job_name"),
                 args.control: control_job.get("job_name")},
        "scope": ("Read-only provenance read of committed receipts: no forward pass, no training, no "
                  "weight, threshold, gate or label change, and no accuracy, real-image or "
                  "complete-state claim."),
    }
    job = read(ROOT/"runs"/(args.output.name+"-job.json"))
    if job.get("memory_bytes") != CEILING_BYTES:
        raise ValueError("the reader ceiling differs from the declared 128 MiB")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"scope.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                                          encoding="utf-8")
    print(json.dumps({"answer": result["answer"],
                      "figures": {key: value["figure"] for key, value in references.items()},
                      "control": control_entry}, indent=2))


if __name__ == "__main__":
    main()
