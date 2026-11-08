"""Curate the identity-discrimination audit: receipt, result and re-derived counts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_model import require_bounded_job  # noqa: E402

RUNS = ("m3-identity-factor-r1-errors-r3", "m3-identity-factor-representation-r1-errors-r1",
        "m3-identity-factor-budget-r1-errors-r1", "m3-stream-identity-shared-r1-errors-r1")


def read(path, limit=2 * 1024**2):
    if path.stat().st_size > limit:
        raise ValueError("curation JSON byte budget exceeded")
    return json.loads(path.read_text())


def rederive(result):
    """Recompute every published count from the audit's own per-case rows."""
    derived = {}
    for name in RUNS:
        run = result["runs"][name]
        splits = {}
        for split in ("training", "development"):
            rows = run["splits"][split]["cases"]
            hold, correct, supervised = {}, 0, 0
            for row in rows:
                correct += row["tokens_correct"]
                supervised += row["supervised_tokens"]
                if row["target_hold"] is not None:
                    hold[row["predicted_hold"]] = hold.get(row["predicted_hold"], 0) + 1
            values = run["splits"][split]
            derived[name] = derived.get(name, {})
            derived[name][split] = {
                "queue": {"supervised_piece_tokens": supervised, "correct_tokens": correct},
                "hold": {"supervised_pieces": sum(hold.values()),
                         "correct_pieces": sum(1 for row in rows if row["hold_correct"] is True),
                         "predicted_classes": dict(sorted(hold.items())),
                         "distinct_predicted_classes": len(hold)},
                "cases": len(rows),
            }
            for section in ("queue", "hold"):
                for key, value in derived[name][split][section].items():
                    if values[section][key] != value:
                        raise ValueError("audit summary does not follow from its per-case rows: %s/%s" %
                                         (name, key))
            if values["images"] != len(rows):
                raise ValueError("audit image count does not follow from its per-case rows")
        for split in ("training", "development"):
            if run["splits"][split].get("images") is None:
                raise ValueError("both splits must carry their own measurement")
    return derived


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--stem", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]{1,64}", args.name) or not re.fullmatch(r"[a-z0-9-]{1,64}", args.stem):
        raise ValueError("finite owned evidence name and receipt stem required")
    runner = hashlib.sha256((ROOT / "scripts/run_bounded.py").read_bytes()).hexdigest()
    receipt = read(ROOT / "runs" / (args.stem + ".json"))
    if receipt["status"] != "passed" or receipt["work_items"] != len(RUNS):
        raise ValueError("complete four-run audited receipt required")
    if receipt["supervisor_script_sha256"] != runner:
        raise ValueError("audit receipt was produced by another supervisor")
    output = ROOT / "runs" / args.stem
    result = read(output / "result.json")
    if result["status"] != "passed" or result["audit_format_version"] != 1:
        raise ValueError("complete audited result required")
    if sorted(result["runs"]) != sorted(RUNS):
        raise ValueError("audited run population differs")
    derived = rederive(result)
    evidence = ROOT / "docs" / "evidence" / args.name
    evidence.mkdir(exist_ok=False)
    for suffix in (".json", ".log", ".stderr.log"):
        shutil.copyfile(ROOT / "runs" / (args.stem + suffix), evidence / (args.stem + suffix))
    shutil.copyfile(output / "result.json", evidence / "result.json")
    if hashlib.sha256((output / "result.json").read_bytes()).hexdigest() != \
            hashlib.sha256((evidence / "result.json").read_bytes()).hexdigest():
        raise ValueError("curated result bytes differ from the audited output")
    sources = {}
    for relative in ("docs/experiments/audit_identity_discrimination.py",
                     "docs/experiments/record_identity_discrimination.py",
                     "scripts/run_bounded.py", "scripts/probe_model.py",
                     "scripts/m3_error_analysis.py"):
        sources[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    scope = {
        "status": "passed", "audit_format_version": result["audit_format_version"],
        "baseline": result["baseline"], "exposure_audit": result["exposure_audit"],
        "exposure_result_sha256": result["exposure_result_sha256"],
        "baseline_result_sha256": result["baseline_result_sha256"],
        "runs": {name: {"weights_sha256": result["runs"][name]["weights_sha256"],
                        "splits": derived[name],
                        "queue_joint_constant_classes": result["runs"][name]["splits"]["training"]["queue"]["joint_constant_classes"],
                        "hold_constant_predictor": result["runs"][name]["splits"]["training"]["hold"]["constant_predictor"],
                        "hold_exact_equals_predicted_class_frequency":
                            result["runs"][name]["splits"]["training"]["hold"]["exact_equals_predicted_class_frequency"]}
                 for name in RUNS},
        "result_sha256": hashlib.sha256((evidence / "result.json").read_bytes()).hexdigest(),
        "audit_receipt": {"stem": args.stem, "status": receipt["status"], "work_items": receipt["work_items"],
                          "peak_job_commit_bytes": receipt["peak_job_commit_bytes"],
                          "declared_wall_seconds": receipt["wall_seconds"],
                          "declared_memory_bytes": receipt["memory_bytes"],
                          "elapsed_seconds": receipt["elapsed_seconds"]},
        "sources": sources, "supervisor_script_sha256": runner,
        "scope": result["scope"],
        "curator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (evidence / "scope.json").write_text(json.dumps(scope, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "evidence": str(evidence),
                      "queue": {name: derived[name]["training"]["queue"] for name in RUNS},
                      "hold": {name: derived[name]["training"]["hold"]["predicted_classes"] for name in RUNS}}))


if __name__ == "__main__":
    main()
