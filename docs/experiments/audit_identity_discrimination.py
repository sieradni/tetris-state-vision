"""Measure whether the M3 identity heads discriminate classes on the frozen runs.

Model-free. Reads only published per-case diagnostics plus the frozen
identity-exposure audit, whose `published_result_sha256` binds the baseline it was
derived from. No inference, training, target, mask, order or loss change.

Two independent measurements, both masked so unknown stays unknown:

* hold piece: the published `predicted_piece` of every supervised case, tallied by
  class. One predicted class across cases is a constant predictor, so its exact
  count is that class's frequency in the targets.
* queue piece tokens: the published `error_indices` are exactly the supervised
  positions the head got wrong, so for each case the set of classes whose constant
  prediction reproduces that error set is computable from the audited labels. A
  class that explains every case in every split is a proven constant predictor;
  zero consistent classes at any case proves the head is not constant there.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_model import require_bounded_job  # noqa: E402

PIECES = tuple("IOTSZJL")
BASELINE = "m3-identity-factor-r1-errors-r3"
EXPOSURE = Path("runs/m3-identity-exposure-r4/result.json")
RUNS = ("m3-identity-factor-r1-errors-r3", "m3-identity-factor-representation-r1-errors-r1",
        "m3-identity-factor-budget-r1-errors-r1", "m3-stream-identity-shared-r1-errors-r1")


def read(path, limit=2 * 1024**2):
    if path.stat().st_size > limit:
        raise ValueError("audit JSON byte budget exceeded")
    return json.loads(path.read_text())


def evidence(name):
    return ROOT / "docs" / "evidence" / name / "result.json"


def constant_classes(errors, labels):
    """Classes whose constant prediction is exactly consistent with the error set."""
    errors = set(errors)
    return [index for index, name in enumerate(PIECES)
            if errors == {position for position, token in labels if PIECES[token] != name}]


def measure(run, published, labels, split):
    rows = [case for case in published["cases"] if case["split"] == split]
    if not rows:
        raise ValueError("split present without published cases")
    hold_classes, hold_correct, queue_correct = {}, 0, 0
    consistent, joint = {}, None
    cases = []
    for case in rows:
        components = case["diagnostics"]["components"]
        piece = components["queue"]["piece_tokens"]
        # Published tokens include at most one EOS label; audited labels are pieces only.
        if components["known_queue_tokens"] - len(labels[case["sample_id"]]) not in (0, 1):
            raise ValueError("audited labels differ from the published supervised population")
        queue_correct += piece["correct"]
        classes = constant_classes(piece["error_indices"], labels[case["sample_id"]])
        if not classes:
            consistent.setdefault("not_constant", []).append(case["sample_id"])
        for index in classes:
            consistent[PIECES[index]] = consistent.get(PIECES[index], []) + [case["sample_id"]]
        joint = set(classes) if joint is None else joint & set(classes)
        hold = components["hold"]
        if hold["target_piece"]:
            hold_classes[hold["predicted_piece"]] = hold_classes.get(hold["predicted_piece"], 0) + 1
            hold_correct += int(hold["piece_correct"] is True)
        cases.append({"sample_id": case["sample_id"], "group_id": case["group_id"], "split": split,
                      "supervised_tokens": len(labels[case["sample_id"]]),
                      "tokens_correct": piece["correct"],
                      "consistent_constant_classes": [PIECES[index] for index in classes],
                      "target_hold": hold["target_piece"], "predicted_hold": hold["predicted_piece"],
                      "hold_correct": hold["piece_correct"]})
    supervised = sum(len(labels[case["sample_id"]]) for case in rows)
    hold_supervised = sum(hold_classes.values())
    constant = next(iter(hold_classes)) if len(hold_classes) == 1 else None
    matching = sum(1 for case in cases if case["target_hold"] == constant)
    return {"images": len(rows), "ancestry_groups": len({case["group_id"] for case in rows}),
            "queue": {"supervised_piece_tokens": supervised, "correct_tokens": queue_correct,
                      "consistent_cases_by_class": {name: sorted(set(ids)) for name, ids in sorted(consistent.items())},
                      "cases_with_no_consistent_constant": sorted(consistent.get("not_constant", [])),
                      "joint_constant_classes": sorted(PIECES[index] for index in (joint or ())),
                      "constant_predictor_proven": bool(joint)},
            "hold": {"supervised_pieces": hold_supervised, "correct_pieces": hold_correct,
                     "predicted_classes": dict(sorted(hold_classes.items())),
                     "distinct_predicted_classes": len(hold_classes),
                     "constant_predictor": bool(hold_supervised > 1 and len(hold_classes) == 1),
                     "targets_matching_predicted_class": matching,
                     "exact_equals_predicted_class_frequency": bool(
                         constant is not None and hold_supervised > 1 and hold_correct == matching)},
            "cases": cases}


def main():
    ceiling = require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    exposure = read(ROOT / EXPOSURE)
    baseline = evidence(BASELINE)
    digest = hashlib.sha256(baseline.read_bytes()).hexdigest()
    if exposure["published_result_sha256"] != digest or exposure["run"] != "m3-identity-factor-r1":
        raise ValueError("exposure audit does not bind the published baseline diagnostics")
    labels = {case["sample_id"]: case["labels"] for case in exposure["cases"]}
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    runs, sources = {}, {}
    for name in RUNS:
        path = evidence(name)
        sources["docs/evidence/" + name + "/result.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        published = read(path)
        if published["status"] != "passed" or len(published["cases"]) != 27:
            raise ValueError("complete 27-case published diagnostics required")
        if any(case["sample_id"] not in labels for case in published["cases"]):
            raise ValueError("published case is missing from the frozen exposure audit")
        runs[name] = {"queue_constant_threshold": published.get("component_identity_factor"),
                      "weights_sha256": published["weights_sha256"],
                      "splits": {split: measure(name, published, labels, split)
                                 for split in ("training", "development")}}
        (args.output / "progress.json").write_text(json.dumps({"runs": sorted(runs)}) + "\n")
    result = {
        "status": "passed", "audit_format_version": 1, "baseline": BASELINE,
        "exposure_audit": EXPOSURE.as_posix(),
        "exposure_result_sha256": hashlib.sha256((ROOT / EXPOSURE).read_bytes()).hexdigest(),
        "baseline_result_sha256": digest, "runs": runs, "sources": sources,
        "job_ceiling_bytes": ceiling, "elapsed_seconds": time.monotonic() - started,
        "scope": ("Identity discrimination of the queue-token and hold-piece heads on the four published "
                  "frozen M3 error runs, read from saved per-case diagnostics and the frozen exposure audit's "
                  "labels only. Constant-predictor consistency is computed from published error indices, so it "
                  "reports what the saved bytes prove and no per-slot predicted token class. No inference, "
                  "training, mask, target, order or loss change, and no complete-state, real-image or release "
                  "claim."),
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with (args.output / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({name: {split: {"queue_correct": values["queue"]["correct_tokens"],
                                     "queue_supervised": values["queue"]["supervised_piece_tokens"],
                                     "queue_joint_constant": values["queue"]["joint_constant_classes"],
                                     "hold_predicted": values["hold"]["predicted_classes"],
                                     "hold_correct": values["hold"]["correct_pieces"],
                                     "hold_supervised": values["hold"]["supervised_pieces"]}
                              for split, values in run["splits"].items()}
                      for name, run in runs.items()}, indent=1), flush=True)


if __name__ == "__main__":
    main()
