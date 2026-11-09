"""Curate the declared registration attribution fallback.

`docs/M3_OCCUPANCY_SCALED.md` fixes the rule and its precondition: this instrument may only be read
when the scaled-budget verdict refused scaling. The curator re-derives both pairings and the cluster
interval from the probe's own stored cells, refuses a probe that is not the declared 36-forward job,
and writes the evidence directory only after the arithmetic agrees.
"""
import argparse
import json
from pathlib import Path

from probe_model import require_bounded_job
from registration_attribution import (COLUMNS, F1_FLOOR, ROWS, attribution_verdict, summarize)
from record_representation_lever import read

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 640*1024**2
WALL_SECONDS = 300
FORWARDS = 36
DECIDING_IMAGES = 36
SCALED = ROOT/"docs/evidence/m3-occupancy-scaled-r1/scope.json"
PROBE = "m3-occscale-registration-r2"
RUN = "m3-occscale-lever-r1"


def require_scaled_refusal():
    """The declared precondition: the fallback exists only for a refused scaling verdict."""
    record = read(SCALED)
    if record.get("status") != "passed" or record.get("instrument") != "m3-occupancy-scaled-r1":
        raise ValueError("the scaled-budget receipt is not the committed record")
    verdict = record.get("verdict") or {}
    if verdict.get("verdict") != "not-cleared":
        raise ValueError("registration may only run after scaling is refused")
    return {"instrument": record["instrument"], "predeclaration": record["predeclaration"],
            "scaling_verdict": verdict.get("verdict"), "failing": verdict.get("failing"),
            "as_is_per_cell_f1": record["arms"]["lever"]["per_cell_f1"],
            "as_is_operating_counts": record["arms"]["lever"]["operating_counts"],
            "attribution_delta_vs_matched_budget_control":
                record["scaling"]["attribution_delta_vs_matched_budget_control"],
            "reference_best_f1": record["reference"]["arm"]["best"]["f1"]}


def require_probe():
    result = read(ROOT/"runs"/PROBE/"result.json")
    job = read(ROOT/"runs"/(PROBE+"-job.json"))
    summary = read(ROOT/"runs"/RUN/"summary.json")
    if (job.get("status") != "passed" or job.get("exit_code") != 0 or
            job.get("work_items") != FORWARDS or job.get("memory_bytes") != CEILING_BYTES or
            job.get("wall_seconds") != float(WALL_SECONDS)):
        raise ValueError("the probe did not pass as the declared 36-forward 640 MiB / 300 s job")
    if (result.get("status") != "passed" or result.get("forwards") != FORWARDS or
            result.get("instrument") != "m3-occupancy-registration-r1" or
            result.get("run") != RUN or len(result.get("cases", [])) != FORWARDS):
        raise ValueError("the probe is not a complete 36-board registration attribution")
    if result.get("weights_sha256") != summary.get("checkpoint_sha256"):
        raise ValueError("the probe measured different weights from its study")
    cases = result["cases"]
    if [case["split"] for case in cases] != ["training"]*FORWARDS:
        raise ValueError("the probe must score exactly the deciding training population")
    for case in cases:
        if len(case["logits"]) != COLUMNS*ROWS or len(case["mapping"]) != COLUMNS*ROWS:
            raise ValueError("a probed board does not carry 200 cells")
    if any(len(case["occupied"]) != COLUMNS*ROWS or len(case["known"]) != COLUMNS*ROWS for case in cases):
        raise ValueError("a probed board does not carry 200 labels")
    if (sum(1 for case in cases if any(case["occupied"])),
            len({"-".join(case["sample_id"].split("-")[:2]) for case in cases})) != (28, 12):
        raise ValueError("deciding arithmetic differs from the predeclaration (28 boards, 12 clusters)")
    return result, job, summary


def main():
    bounded = require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("evidence directory already exists")

    scaled = require_scaled_refusal()
    result, job, summary = require_probe()
    measured = summarize(result["cases"])
    if measured != result["summary"]:
        raise ValueError("the probe's stored summary does not re-derive from its own cells")
    verdict = attribution_verdict(measured)
    if verdict != result["verdict"]:
        raise ValueError("the probe's stored verdict does not re-derive")
    if measured["matched_cells"] == 0:
        raise ValueError("no cell survives the registration pairing")
    # Cross-receipt join: the full as-is pairing on the 28 nonempty boards must reproduce the
    # scaled sweep's published per-cell reading, or one of the two instruments is mis-scoring.
    if abs(measured["as_is_full"]["f1"]-scaled["as_is_per_cell_f1"]) > 1e-6:
        raise ValueError("the as-is pairing does not reproduce the study's published per-cell F1")
    published = scaled["as_is_operating_counts"]
    derived = measured["as_is_full"]["counts"]
    if any(derived[key] != published[key] for key in
           ("true_positive", "false_positive", "false_negative")):
        raise ValueError("the as-is pairing does not reproduce the study's operating counts")
    if measured["attribution"]["bootstrap"]["clusters"] != 12:
        raise ValueError("the paired bootstrap must resample the twelve ancestry clusters")

    scope = {
        "status": "passed",
        "instrument": "m3-occupancy-registration-r1",
        "predeclaration": "docs/M3_OCCUPANCY_SCALED.md",
        "scope": ("Read-only attribution of the scaled stencil arm's occupancy error to cell "
                  "placement: each model cell's centre is located in target cell coordinates using "
                  "the target lattice, and its logit is scored against the label of the cell it "
                  "physically covers (oracle registration) as well as against its own index, on the "
                  "same cells. No training, no weight, threshold, gate, label or prior-evidence "
                  "change, and no real-image, complete-state or calibrated claim."),
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "envelope": {"ceiling_bytes": CEILING_BYTES, "wall_seconds": WALL_SECONDS,
                     "forwards": FORWARDS, "work_items": {"probe": FORWARDS, "curation": 4}},
        "precondition": scaled,
        "source": {"run": RUN, "weights_sha256": result["weights_sha256"],
                   "steps": summary["steps"], "catalogue_sha256": summary["catalogue_sha256"]},
        "probe": {"job_status": job["status"], "work_items": job["work_items"],
                  "peak_commit_bytes": job.get("peak_job_commit_bytes"),
                  "elapsed_seconds": job.get("elapsed_seconds"),
                  "forwards": result["forwards"], "threshold": result["threshold"]},
        "population": result["population"],
        "measured": measured,
        "verdict": verdict,
        "limits": ("An attribution diagnostic under oracle pairing, not a model, a correction, an "
                   "accuracy estimate or a new label policy. Cells whose mapped target cell lies "
                   "outside the board are excluded from both pairings, and the excluded share is "
                   "published as mapped_share."),
        "unchanged": {"thresholds": {"decision": 0.0}, "population": "widened deciding population",
                      "source_checkpoint": summary["checkpoint_sha256"],
                      "prior_evidence": "not touched", "frozen_manifest": "not touched"},
    }
    args.output.mkdir(parents=True)
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(scope, stream, indent=2, allow_nan=False)
    print(json.dumps({"verdict": verdict, "measured": {
        key: measured[key] for key in ("images", "matched_cells", "mapped_share", "shifted_share",
                                       "mean_abs_offset_cells", "f1_delta")}}, indent=2))


if __name__ == "__main__":
    main()
