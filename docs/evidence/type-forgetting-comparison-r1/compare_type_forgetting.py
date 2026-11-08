"""Compare source-bound checkpoint observations and class distributions."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_model import require_bounded_job
from type_forgetting_analysis import analyze
from probe_type_forgetting import sha


def read(path):
    if path.stat().st_size > 32*1024**2:
        raise ValueError("finite evidence required")
    return json.loads(path.read_text())


def main():
    ceiling = require_bounded_job()
    banks, scopes, metrics, identities = [], [], {}, {}
    for arm in ("control", "candidate"):
        folder = ROOT / "docs/evidence" / f"type-forgetting-{arm}-r1"
        scope, bank = read(folder / "scope.json"), read(folder / "result.json")
        curator = read(ROOT / "runs" / f"type-forgetting-{arm}-curation-r1-job.json")
        if (scope["status"] != "passed" or bank["arm"] != arm or bank["forwards"] != 544 or
                sha(folder / "result.json") != scope["artifact_sha256"]["result.json"] or
                curator["status"] != "passed" or curator["exit_code"] != 0 or
                curator["memory_bytes"] != 128*1024**2 or curator["work_items"] != 544 or
                scope["source_catalogue_sha256"] != bank["source_catalogue_sha256"] or
                scope["weights_sha256"] != bank["weights_sha256"] or
                analyze(bank["observations"]) != scope["analysis"]):
            raise ValueError("independently curated observer identity differs")
        stats = []
        for point in bank["observations"]:
            predicted = Counter()
            correct = Counter()
            for case in point["cases"]:
                for i, role in enumerate(("active", "ghost")):
                    target = case["roles"][role]["target_type"]
                    values = case["logits"][i]
                    guess = max(range(8), key=values.__getitem__)
                    predicted[str(guess)] += 1
                    correct[str(target)] += int(guess == target)
            stats.append({"step": point["step"], "predicted_types": {str(i): predicted[str(i)] for i in range(8)},
                          "correct_by_target": {str(i): correct[str(i)] for i in range(8)}})
        metrics[arm] = {"class_counts": stats, "analysis": scope["analysis"], "weights_sha256": bank["weights_sha256"]}
        identities[arm] = {"scope_sha256": sha(folder / "scope.json"), "result_sha256": sha(folder / "result.json"),
                           "curator_receipt_sha256": sha(ROOT / "runs" / f"type-forgetting-{arm}-curation-r1-job.json")}
        banks.append(bank)
        scopes.append(scope)
    if banks[0]["source_catalogue_sha256"] != banks[1]["source_catalogue_sha256"]:
        raise ValueError("model source differs")
    for a, b in zip(banks[0]["observations"], banks[1]["observations"]):
        if a["step"] != b["step"]:
            raise ValueError("checkpoint steps differ")
        for x, y in zip(a["cases"], b["cases"]):
            if (x["sample_id"] != y["sample_id"] or x["group_id"] != y["group_id"] or
                    any(x["roles"][role]["target_type"] != y["roles"][role]["target_type"] or
                        x["roles"][role]["target_known"] != y["roles"][role]["target_known"] or
                        x["roles"][role]["target_pose_known"] != y["roles"][role]["target_pose_known"]
                        for role in ("active", "ghost"))):
                raise ValueError("matched public target identities differ")
    if banks[0]["observations"][0] != banks[1]["observations"][0]:
        raise ValueError("actual initial logits/features/head differ")
    output = ROOT / "docs/evidence/type-forgetting-comparison-r1"
    output.mkdir(exist_ok=False)
    result = {"status": "passed", "job_ceiling_bytes": ceiling,
              "source_catalogue_sha256": banks[0]["source_catalogue_sha256"], "inputs": identities,
              "comparison_source_sha256": sha(Path(__file__)), "identical_initial_head_features_logits": True,
              "matched_targets": True, "arms": metrics,
              "scope": "One seed, dependent fixed synthetic training fixtures, 64-update observations; local final-head gradient geometry is not an Adam or full joint-gradient analysis."}
    (output / "scope.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": "passed", "final": {arm: metrics[arm]["class_counts"][-1] for arm in metrics}}))


if __name__ == "__main__":
    main()
