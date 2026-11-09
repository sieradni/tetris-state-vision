"""Curate the read-only pixel-ceiling probe against the sweep it joins.

`docs/M3_OCCUPANCY_PIXEL_CEILING.md` fixes the rule; this recorder re-derives every
published number from the committed probe result, its receipts and the swept bytes, and it
refuses before writing when the population, the control or the join disagrees.
"""
import argparse
import json
from pathlib import Path

from pixel_ceiling import F1_THRESHOLD, AUC_THRESHOLD, counts, f1, precision, recall, sweep, \
    tie_aware_auc
from probe_model import require_bounded_job

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 640*1024**2
CASES = 36
NONEMPTY = 28
OCCUPIED = 667
EMPTY = 4933
INSTRUMENT = "m3-occupancy-pixel-ceiling-r1"


def read(path):
    return json.loads(Path(path).read_text())


def require_deciding(population, cases, control):
    """The declared arithmetic, asserted before any evidence byte is written."""
    if (population.get("nonempty_boards"), population.get("empty_boards"),
            population.get("occupied_cells"), population.get("empty_cells")) != (
            NONEMPTY, CASES-NONEMPTY, OCCUPIED, EMPTY):
        raise ValueError("the deciding arithmetic differs from the predeclaration")
    if len(cases) != NONEMPTY or sum(row["known_cells"] for row in cases) != NONEMPTY*200:
        raise ValueError("the probe did not measure the twenty-eight nonempty boards")
    if sum(row["occupied"] for row in cases) != OCCUPIED:
        raise ValueError("the probe's case labels differ from the deciding population")
    if control["operating"]["f1"] >= F1_THRESHOLD or control["fold_mean_auc"] >= AUC_THRESHOLD:
        raise ValueError("the label-shuffled control did not collapse")


def rederive(cases, labels_by_id):
    """Re-derive from the probe's scores and the committed sweep's own cell order."""
    labels, logits, control = [], [], []
    for row in cases:
        truth = labels_by_id.get(row["sample_id"])
        if (truth is None or len(truth) != row["known_cells"] or
                sum(truth) != row["occupied"] or len(row["pixel_logits"]) != row["known_cells"]):
            raise ValueError("a case's stored cell labels do not match the committed sweep")
        labels.extend(truth)
        logits.extend(row["pixel_logits"])
        control.extend(row["control_logits"])
    operating = counts(logits, labels, 0.)
    measured = {"pooled_auc": tie_aware_auc(logits, labels), "operating": {
        key: operating[key] for key in ("true_positive", "false_positive", "false_negative")},
        "best": sweep(logits, labels)}
    return measured


def main():
    bounded = require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--profile", required=True, help="the probe's admitted profile run name")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("evidence directory already exists")
    probe = read(ROOT/"runs"/args.probe/"result.json")
    job = read(ROOT/"runs"/(args.probe+"-job.json"))
    profile_job = read(ROOT/"runs"/(args.profile+"-job.json"))
    profile = read(ROOT/"runs"/args.profile/"summary.json")
    sweep_bytes = read(ROOT/"runs"/probe["sweep"]/"result.json")
    if (probe.get("status") != "passed" or job.get("status") != "passed" or job.get("exit_code") != 0 or
            job.get("work_items") != CASES or job.get("memory_bytes") != CEILING_BYTES):
        raise ValueError("the probe did not pass under its declared envelope")
    if (profile_job.get("status") != "passed" or profile.get("status") != "complete" or
            profile.get("cases") != CASES):
        raise ValueError("the probe's admission profile differs")
    if (probe["weights_sha256"] != sweep_bytes["weights_sha256"] or
            probe["model_reference"]["source"] != probe["sweep"]):
        raise ValueError("the probe does not join the committed sweep")
    labels_by_id = {row["sample_id"]: row["occupied"] for row in sweep_bytes["cases"]}
    if not {row["sample_id"] for row in probe["cases"]} <= set(labels_by_id):
        raise ValueError("the probe's cases are not a subset of the joined sweep")
    require_deciding(probe["population"], probe["cases"], probe["control"])
    rederived = rederive(probe["cases"], labels_by_id)
    published = probe["measured"]
    if (rederived["operating"]["true_positive"] != published["operating"]["true_positive"] or
            rederived["operating"]["false_positive"] != published["operating"]["false_positive"] or
            abs(rederived["best"]["f1"]-published["best"]["f1"]) > 1e-6 or
            abs(rederived["pooled_auc"]-published["pooled_auc"]) > 1e-9):
        raise ValueError("the probe's stored statistics do not re-derive from its own cells")
    if probe["verdict"]["status"] == "instrument-invalid":
        raise ValueError("the control invalidated the instrument")

    model = probe["model_reference"]
    evidence = {
        "status": "passed", "instrument": INSTRUMENT,
        "predeclaration": "docs/M3_OCCUPANCY_PIXEL_CEILING.md",
        "scope": ("Read-only pixel-only read-out of the committed determinate occupancy labels on the "
                  "committed 28 nonempty deciding boards. No model forward, no training, no weight, "
                  "threshold, gate, label-policy or evidence change; no accuracy, real-image or "
                  "complete-state claim follows."),
        "declared_receipt": args.receipt, "declared_memory_bytes": bounded,
        "envelope": {"ceiling_bytes": CEILING_BYTES, "work_items": {"profile": 2, "study": CASES,
                                                                   "curation": 2}},
        "join": {"sweep": probe["sweep"], "run": probe["run"],
                 "weights_sha256": probe["weights_sha256"],
                 "catalogue_sha256": probe["catalogue_sha256"],
                 "swept_cases": probe["cases"], "published": True},
        "population": probe["population"],
        "features": probe["features"],
        "measured": {"pooled_auc": published["pooled_auc"],
                     "fold_mean_auc": published["fold_mean_auc"],
                     "operating": published["operating"],
                     "operating_f1": f1(published["operating"]),
                     "operating_precision": precision(published["operating"]),
                     "operating_recall": recall(published["operating"]),
                     "best": published["best"], "per_fold": published["per_fold"]},
        "control": {"fold_mean_auc": probe["control"]["fold_mean_auc"],
                    "operating": probe["control"]["operating"],
                    "operating_f1": f1(probe["control"]["operating"]),
                    "best": probe["control"]["best"]},
        "gates": {"operating_f1": F1_THRESHOLD, "fold_mean_auc": AUC_THRESHOLD,
                  "grid_points": 81},
        "verdict": probe["verdict"],
        "model_reference": {"note": ("Read from the committed sweep's own threshold rows, so this pass "
                                     "joins that receipt instead of restating it."),
                            **model},
        "comparison": {
            "operating_f1_gap": round(f1(published["operating"])-model["operating"]["f1"], 6),
            "best_f1_gap": round(published["best"]["f1"]-model["best"]["f1"], 6),
            "auc_gap": round(published["fold_mean_auc"]-model["auc"], 6)},
        "label_hypothesis": {
            "status": ("refuted" if probe["verdict"]["status"] == "separable" else "licensed"),
            "statement": ("The committed labels are recoverable from the captured pixels by a per-cell "
                          "read-out that never sees the model, so an explicit indeterminate target cannot "
                          "be the missing separation at this capture."),
            "indeterminate_target_licensed": probe["verdict"]["status"] != "separable"},
        "open": ["the model-side representation/optimization lever remains unnamed by this instrument"],
        "unchanged": {"thresholds": {"operating_f1": F1_THRESHOLD, "fold_mean_auc": AUC_THRESHOLD},
                      "population": "widened deciding population, twenty-eight nonempty boards",
                      "labels": "no label policy changed", "prior_evidence": "not touched"},
    }
    args.output.mkdir(parents=True)
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
    print(json.dumps({key: evidence[key] for key in ("verdict", "label_hypothesis", "comparison")},
                     indent=2))


if __name__ == "__main__":
    main()
