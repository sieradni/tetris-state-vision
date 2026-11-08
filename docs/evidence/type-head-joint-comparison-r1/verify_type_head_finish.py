"""Compare completed fixed type-head-rate studies after ordinary pixel audits."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from m2_fit_contract import summarize_fit
from probe_model import require_bounded_job
from record_stream_run import checkpoint_architecture
from source_snapshot import verify_source


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    if path.stat().st_size > 2 * 1024**2:
        raise ValueError("finite report required")
    return json.loads(path.read_text())


def main():
    ceiling = require_bounded_job()
    sources = [Path(__file__), ROOT / "scripts/m2_fit_contract.py", ROOT / "scripts/record_stream_run.py"]
    source_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    names = ("m2-stream-headrate-control-s3", "m2-stream-headrate-candidate-s3")
    reports = ("m2-stream-headrate-control-errors-r1", "m2-stream-headrate-candidate-errors-r1")
    summaries, ledgers, outcomes, inputs = [], [], {}, {}
    for name, report_name, factor in zip(names, reports, (1, 4)):
        run = ROOT / "runs" / name
        summary, job = read(run / "summary.json"), read(ROOT / "runs" / (name + "-job.json"))
        parent = read(ROOT / "runs" / ("type-head-" + ("control" if factor == 1 else "candidate") + "-s3-controller-job.json"))
        scope, manifest = read(ROOT / "docs/evidence" / name / "scope.json"), read(run / "checkpoint/manifest.json")
        if (summary["status"] != "complete" or summary["steps"] != 1024 or summary["requested_steps"] != 1024 or
                summary["updates_this_run"] != 448 or summary["resume"]["step"] != 576 or
                summary["curriculum"] != "all-class-fit-v1" or summary["model_kind"] != "pilot" or
                summary["recovery_contract"]["type_head_rate_factor"] != factor or
                job["status"] != "passed" or job["exit_code"] != 0 or job["work_items"] != 448 or
                job["memory_bytes"] != 640*1024**2 or parent["status"] != "passed" or
                parent["memory_bytes"] != 1280*1024**2 or parent["work_items"] != 1513 or
                scope["status"] != "passed" or scope["steps"] != 1024 or scope["verified_saved_probe_images"] != 41 or
                scope["source_catalogue_sha256"] != summary["catalogue_sha256"] or
                scope["weights_sha256"] != summary["checkpoint_sha256"] or
                checkpoint_architecture(summary, manifest) != "TetrisStateNet-M2" or
                sha(run / "checkpoint/weights.pt") != summary["checkpoint_sha256"] or
                sha(run / "initial-weights.pt") != summary["initial_weights_sha256"]):
            raise ValueError("completed curated study identity differs")
        verify_source(run / "code", summary["catalogue_sha256"])
        ledger = (run / "training.jsonl").read_bytes()
        rows = [json.loads(line) for line in ledger.splitlines()]
        if len(rows) != 1024 or [row["step"] for row in rows] != list(range(1, 1025)):
            raise ValueError("complete ordered training ledger required")
        evidence = ROOT / "docs/evidence" / report_name
        pixel_scope, report = read(evidence / "scope.json"), read(evidence / "result.json")
        if (pixel_scope["status"] != "passed" or pixel_scope["global_steps"] != 1024 or
                pixel_scope["target_steps"] != 1024 or pixel_scope["forwards"] != 41 or
                pixel_scope["weights_sha256"] != summary["checkpoint_sha256"] or
                pixel_scope["source_catalogue_sha256"] != summary["catalogue_sha256"] or
                sha(evidence / "result.json") != pixel_scope["artifact_sha256"]["result.json"] or
                read(evidence / "input-summary.json") != summary or report["run"] != name or
                report["weights_unchanged"] is not True or report["automatic_metrics_reproduced"] is not True or
                report["label_inputs_to_model"] is not False):
            raise ValueError("independently curated ordinary pixel replay required")
        raw_fit = [{"sample_id": case["sample_id"], **case["ordinary_metrics"]} for case in report["cases"] if case["split"] == "training"]
        fit = summarize_fit(raw_fit)
        if fit != summary["final_training_probe"]["fit"]:
            raise ValueError("independently recomputed joint fit differs")
        outcomes[name] = {"training_fit": fit, "training_errors": report["totals"]["training"],
                          "development": report["totals"]["development"], "weights_sha256": summary["checkpoint_sha256"]}
        summaries.append(summary)
        ledgers.append(ledger)
        inputs[name] = {p: sha(run / p) for p in ("summary.json", "training.jsonl", "curve.jsonl", "initial-weights.pt",
                                                    "checkpoint/manifest.json", "checkpoint/weights.pt", "code/source.json")}
        inputs[name].update(pixel_scope_sha256=sha(evidence / "scope.json"), pixel_report_sha256=sha(evidence / "result.json"))
    control, candidate = summaries
    recipes = [{k: v for k, v in summary["recovery_contract"].items() if k != "type_head_rate_factor"} for summary in summaries]
    if (recipes[0] != recipes[1] or control["config"] != candidate["config"] or ledgers[0] != ledgers[1] or
            control["initial_weights_sha256"] != candidate["initial_weights_sha256"] or
            control["initial_training_probe"] != candidate["initial_training_probe"] or
            control["initial_development"] != candidate["initial_development"]):
        raise ValueError("matched actual initialization, recipes or complete data differ")
    start = read(ROOT / "docs/evidence/type-head-initial-admission-r2/type-head-matched-admission-r2.json")
    if (start["status"] != "passed" or start["initial_tensor_bytes_identical"] is not True or
            start["source_catalogue_sha256"] != control["catalogue_sha256"] or
            control["initial_weights_sha256"] != start["sources"]["runs/m2-stream-recovery-gate-r10/continuous/initial-weights.pt"]):
        raise ValueError("study starts do not match actual admitted tensors")
    if source_hashes != {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}:
        raise ValueError("comparison source changed")
    output = ROOT / "docs/evidence/type-head-joint-comparison-r1"
    output.mkdir(exist_ok=False)
    proof = {"status": "passed", "runs": names, "updates_each": 1024, "job_ceiling_bytes": ceiling,
             "source_catalogue_sha256": control["catalogue_sha256"], "identical_training_ledger": True,
             "identical_initial_tensors": True, "rates": [1, 4], "outcomes": outcomes, "input_sha256": inputs,
             "comparison_sources": source_hashes,
             "scope": "One seed and 32 related synthetic drawings; not independent generalization or real-image quality."}
    (output / "scope.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "candidate_fit": outcomes[names[1]]["training_fit"]}))


if __name__ == "__main__":
    main()
