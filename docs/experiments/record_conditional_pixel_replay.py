"""Independently audit packed pixel predictions and finite native receipts."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_model import require_bounded_job
from m2_error_analysis import analyze_case, fit_metrics_from_diagnostics, summarize
from type_head_fit_contract import digest, load_bank, metrics, read


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("ordered", "interleaved"), required=True)
    args = parser.parse_args()
    run = ROOT / "runs" / ("conditional-pixel-" + args.study + "-r1")
    proof = json.loads((run / "intervention.json").read_text())
    ordinary = read(run / "result.json")
    job = read(run.parent / (run.name + "-job.json"))
    bank, model, identities = load_bank(args.study)
    head = read(ROOT / "runs" / ("type-head-" + args.study + "-fit-r1") / "result.json")
    expected = ["--study", args.study, "--output"]
    command = job["command"]
    if (proof["status"] != "passed" or proof["study"] != args.study or proof["model_run"] != model.name or
            proof["forwards"] != 82 or proof["inputs_sha256"] != identities or
            proof["ordinary_report_sha256"] != digest(run / "result.json") or
            proof["head_payload_sha256"] != head["payload_sha256"] or
            job["status"] != "passed" or job["exit_code"] != 0 or job["work_items"] != 82 or
            job["wall_seconds"] != 120 or job["memory_bytes"] != 640*1024**2 or
            job["system_pressure_guard"] != "GetPerformanceInfo/0.2s" or
            len(command) != 6 or Path(command[1]).resolve() != ROOT / "docs/experiments/conditional_pixel_replay.py" or
            command[2:5] != expected or Path(command[5]).resolve() != run):
        raise ValueError("native identity or finite resource contract differs")
    for path, expected_hash in proof["tracked_sha256"].items():
        if digest(Path(path)) != expected_hash:
            raise ValueError("replay execution/input source differs")
    records = proof["records"]
    if len(records) != 41 or len(ordinary["cases"]) != 41 or len(proof["changed_logits"]) != 82:
        raise ValueError("finite paired population differs")
    recomputed = []
    for index, (case, raw) in enumerate(zip(ordinary["cases"], records)):
        original, changed, target = raw["original_prediction"], raw["changed_prediction"], raw["target"]
        if analyze_case(original, target) != case["diagnostics"]:
            raise ValueError("raw original prediction/target differs from ordinary audit")
        for key in original:
            if key != "layers" and original[key] != changed[key]:
                raise ValueError("non-type packed prediction changed")
        for role, (a, b) in enumerate(zip(original["layers"], changed["layers"])):
            if any(a[k] != b[k] for k in a if k != "type"):
                raise ValueError("non-type packed role output changed")
            logits = proof["changed_logits"][2*index+role]
            if b["type"] != max(range(8), key=logits.__getitem__):
                raise ValueError("pixel type/logit mismatch")
            row = bank["features"][2*index+role]
            if (row["sample_id"] != case["sample_id"] or row["split"] != case["split"] or
                    row["role"] != ("active", "ghost")[role]):
                raise ValueError("pixel/feature identity order differs")
        diagnostic = analyze_case(changed, target)
        recomputed.append({**{k: case[k] for k in ("split", "sample_id", "group_id")}, "diagnostics": diagnostic})
    type_metrics = metrics(bank["features"], proof["changed_logits"])
    if type_metrics != proof["type_metrics"]:
        raise ValueError("pixel type metric mismatch")
    vector = head["logits"]["raw32"]
    discrepancy = max(abs(a-b) for actual, saved in zip(proof["changed_logits"], vector)
                      for a, b in zip(actual, saved))
    if (discrepancy != proof["pixel_vector_max_abs_difference"] or
            any(type_metrics[split]["correct"] != head["metrics"]["raw32"][split]["correct"]
                for split in ("training", "development"))):
        raise ValueError("pixel/vector comparison differs")
    totals = summarize(recomputed, max_cases=68)
    train = [row for row in recomputed if row["split"] == "training"]
    fits = [fit_metrics_from_diagnostics(row["diagnostics"]) for row in train]
    roles = [role for fit in fits for role in fit["roles"].values() if role["known"]]
    joint = {"boards_exact": sum(row["diagnostics"]["locked"]["exact_board"] is True for row in train),
             "roles_known": len(roles), "roles_exact": sum(role["exact"] is True for role in roles),
             "visible_known": sum(role["observed"] for role in roles),
             "visible_exact": sum(role["observed"] and role["exact"] is True for role in roles),
             "absent_known": sum(role["class"] == "absent" for role in roles),
             "absent_exact": sum(role["class"] == "absent" and role["exact"] is True for role in roles),
             "semantic_known": sum(fit["semantic_known"] for fit in fits),
             "semantic_correct": sum(fit["semantic_correct"] for fit in fits),
             "geometry_mean_cells": totals["training"]["geometry"]["mean_cells"]}
    joint["pass"] = (joint["boards_exact"] == 32 and joint["roles_known"] == joint["roles_exact"] == 64 and
                     joint["semantic_known"] == joint["semantic_correct"] == 128 and joint["geometry_mean_cells"] < .25)
    result = {"status": "passed", "study": args.study, "inputs_sha256": identities,
              "intervention_sha256": digest(run / "intervention.json"), "native_receipt_sha256": digest(run.parent / (run.name + "-job.json")),
              "collector_sha256": digest(Path(__file__)), "joint_fit": joint, "totals": totals,
              "type_metrics": type_metrics, "pixel_vector_max_abs_difference": proof["pixel_vector_max_abs_difference"],
              "scope": "Conditional copied-head synthetic pixel replay; dependent views, no real-image or generalization claim."}
    out = ROOT / "docs/evidence" / run.name
    out.mkdir(exist_ok=False)
    for source in (run / "intervention.json", run / "result.json", run.parent / (run.name + "-job.json"),
                   run.parent / (run.name + ".log"), run.parent / (run.name + ".stderr.log"),
                   Path(__file__), ROOT / "docs/experiments/conditional_pixel_replay.py", ROOT / "scripts/pixel_replay_contract.py"):
        shutil.copyfile(source, out / source.name)
        if digest(source) != digest(out / source.name):
            raise ValueError("evidence copy differs")
    (out / "scope.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "study": args.study, "joint_fit": joint}))


if __name__ == "__main__":
    main()
