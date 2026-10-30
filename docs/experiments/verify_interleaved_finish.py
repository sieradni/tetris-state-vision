"""Compare the two completed fixed studies only after independent ordinary audits."""
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

from verify_interleaved_start import read, rows, sha

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from m2_fit_contract import summarize_fit
from probe_model import require_bounded_job
from record_stream_run import checkpoint_architecture
from source_snapshot import verify_source


def main():
    require_bounded_job()
    sources = (Path(__file__), Path(__file__).with_name("verify_interleaved_start.py"), ROOT / "scripts/m2_fit_contract.py")
    source_hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in sources}
    names = ("m2-stream-allclass1024-s3", "m2-stream-interleave1024-s3")
    reports = ("m2-stream-allclass1024-errors-r1", "m2-stream-interleave1024-errors-r1")
    policies = ("all-class-fit-v1", "all-class-interleaved-v1")
    summaries, archives, populations, outcomes, inputs = [], [], [], {}, {}
    for name, report_name, policy in zip(names, reports, policies):
        run = ROOT / "runs" / name
        summary = read(run / "summary.json")
        job = read(ROOT / "runs" / (name + "-job.json"))
        parent = read(ROOT / "runs" / (name + "-pipeline-job.json"))
        scope = read(ROOT / "docs/evidence" / name / "scope.json")
        manifest = read(run / "checkpoint/manifest.json")
        if (summary["status"] != "complete" or summary["steps"] != 1024 or summary["requested_steps"] != 1024 or
                summary["updates_this_run"] != 384 or summary["resume"]["step"] != 640 or summary["curriculum"] != policy or
                summary["model_kind"] != "pilot" or job["status"] != "passed" or job["exit_code"] != 0 or
                job["work_items"] != 384 or job["memory_bytes"] != 640 * 1024**2 or parent["status"] != "passed" or
                parent["work_items"] != 384 or parent["memory_bytes"] != 896 * 1024**2 or
                scope["status"] != "passed" or scope["steps"] != 1024 or scope["verified_saved_probe_images"] != 41 or
                scope["source_catalogue_sha256"] != summary["catalogue_sha256"] or
                scope["weights_sha256"] != summary["checkpoint_sha256"] or
                checkpoint_architecture(summary, manifest) != "TetrisStateNet-M2" or
                sha(run / "checkpoint/weights.pt") != summary["checkpoint_sha256"]):
            raise ValueError("actual completed curated1024-update M2 studies required")
        archive = verify_source(run / "code", summary["catalogue_sha256"])
        if archive["files"]["scripts/m2_fit_contract.py"] != source_hashes["scripts/m2_fit_contract.py"]:
            raise ValueError("fit criterion implementation changed")
        records, curve = rows(run / "training.jsonl", 1024), rows(run / "curve.jsonl", 1024)
        if ([row["step"] for row in records] != list(range(1, 1025)) or
                [row["sample_id"] for row in curve] != [row["sample_id"] for row in records] or
                any(row.get("curriculum") != policy for row in records)):
            raise ValueError("actual complete optimizer/data order differs")
        evidence = ROOT / "docs/evidence" / report_name
        report_scope, report = read(evidence / "scope.json"), read(evidence / "result.json")
        if (report_scope["status"] != "passed" or report_scope["global_steps"] != 1024 or report_scope["target_steps"] != 1024 or
                report_scope["run_status"] != "complete" or report_scope["forwards"] != 41 or
                report_scope["automatic_metrics_reproduced"] is not True or report_scope["weights_unchanged"] is not True or
                report_scope["weights_sha256"] != summary["checkpoint_sha256"] or
                report_scope["source_catalogue_sha256"] != summary["catalogue_sha256"] or
                sha(evidence / "result.json") != report_scope["artifact_sha256"]["result.json"] or
                read(evidence / "input-summary.json") != summary or
                report["run"] != name or report["curriculum"] != policy or report["run_status"] != "complete" or
                report["global_steps"] != 1024 or report["target_steps"] != 1024 or report["config"] != manifest["config"] or
                report["component_capacity"] != 0 or
                report["weights_sha256"] != summary["checkpoint_sha256"] or report["label_inputs_to_model"] is not False or
                report["automatic_metrics_reproduced"] is not True or report["weights_unchanged"] is not True):
            raise ValueError("independent ordinary final audit identity differs")
        raw_fit = []
        for case in report["cases"]:
            if case["split"] == "training":
                raw_fit.append({"sample_id": case["sample_id"], **case["ordinary_metrics"]})
        fit = summarize_fit(raw_fit)
        if fit != summarize_fit(summary["final_training_probe"]["cases"]) or fit != summary["final_training_probe"]["fit"]:
            raise ValueError("raw independent final fit criterion differs")
        outcomes[name] = {"training_fit": fit, "development": report["totals"]["development"],
                          "training_errors": report["totals"]["training"], "weights_sha256": summary["checkpoint_sha256"]}
        summaries.append(summary)
        archives.append(archive)
        populations.append(records)
        inputs[name] = {file: sha(run / file) for file in
                        ("summary.json", "training.jsonl", "curve.jsonl", "training-probe.jsonl", "development-probe.jsonl",
                         "checkpoint/manifest.json", "checkpoint/weights.pt", "code/source.json")}
        inputs[name].update(ordinary_report_sha256=sha(evidence / "result.json"), ordinary_scope_sha256=sha(evidence / "scope.json"))
    recipes = [{key: value for key, value in summary["recovery_contract"].items() if key not in ("catalogue_sha256", "curriculum")}
               for summary in summaries]
    if recipes[0] != recipes[1] or summaries[0]["config"] != summaries[1]["config"]:
        raise ValueError("model/optimizer/input/step recipe differs beyond policy/source")
    for start in range(0, 1024, 32):
        values = [Counter(json.dumps({key: value for key, value in row.items() if key not in ("step", "curriculum")}, sort_keys=True)
                          for row in records[start:start + 32]) for records in populations]
        if values[0] != values[1] or len(values[0]) != 32 or any(count != 1 for count in values[0].values()):
            raise ValueError("matched32-pair population changed within a completed cycle")
    if [row["sample_id"] for row in populations[0]] == [row["sample_id"] for row in populations[1]]:
        raise ValueError("actual training order did not change")
    packages = [{path: value for path, value in archive["files"].items() if path.startswith("tetris_vision/")} for archive in archives]
    if packages[0] != packages[1]:
        raise ValueError("neural/loss/target package source changed")
    start_path = ROOT / "docs/evidence/interleave128-matched-start-r1/scope.json"
    matched = read(start_path)
    if (matched["status"] != "passed" or matched["initial_prediction_loss_metrics_exact"] is not True or
            matched["same_input_warm_actual_tensors_exact"] is not True or
            [name.rsplit("-", 1)[0] + "-s1" for name in names] != matched["runs"]):
        raise ValueError("actual matched fresh start missing")
    warm_path = ROOT / "docs/evidence/interleave-profile-numerics-r1/scope.json"
    warm = read(warm_path)
    if (warm["status"] != "passed" or warm["identical_actual_weights"] is not True or
            warm["source_catalogues"] != [summary["catalogue_sha256"] for summary in summaries]):
        raise ValueError("final sources differ from matched same-input warm proof")
    if source_hashes != {path.relative_to(ROOT).as_posix(): sha(path) for path in sources}:
        raise ValueError("comparison implementation changed during execution")
    output = ROOT / "docs/evidence/interleave1024-comparison-r1"
    output.mkdir(exist_ok=False)
    for path in sources:
        shutil.copyfile(path, output / path.name)
    proof = {"status": "passed", "runs": names, "policies": policies, "updates_each": 1024,
             "all32_completed_cycles_have_identical_pixel_state_pairs": True, "actual_training_order_differs": True,
             "neural_package_sources_identical": True, "common_training_recipe": recipes[0],
             "matched_start_scope_sha256": sha(start_path), "warm_scope_sha256": sha(warm_path),
             "outcomes": outcomes, "input_sha256": inputs,
             "comparison_sources": source_hashes,
             "scope": "One matched seed and32 controlled drawings;actual fit verdict only. Views share ancestry. Not a real screenshot,independent-generalization or complete-state release result."}
    (output / "scope.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "updates_each": 1024,
                      "candidate_fit_passed": outcomes[names[1]]["training_fit"]["passed"]}))


if __name__ == "__main__":
    main()
