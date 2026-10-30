"""Curate fixed read-only feature studies after independent ordinary replay."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_model import require_bounded_job
from record_stream_errors import digest, read
from record_stream_run import checkpoint_architecture, validate_completion
from source_snapshot import verify_source
from layer_feature_analysis import analyze

STUDIES = {
    "m2-stream-accuracy4096-features-r1": ("m2-stream-accuracy4096-s10", "m2-stream-accuracy4096-errors-r1", 4096, 27, 1024),
    "m2-stream-allclass1024-features-r1": ("m2-stream-allclass1024-s3", "m2-stream-allclass1024-errors-r1", 1024, 41, 640),
    "m2-stream-interleave1024-features-r1": ("m2-stream-interleave1024-s3", "m2-stream-interleave1024-errors-r1", 1024, 41, 640),
}


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=tuple(STUDIES), required=True)
    args = parser.parse_args()
    model_name, ordinary_name, steps, count, memory = STUDIES[args.name]
    run = ROOT / "runs" / args.name
    model_run = ROOT / "runs" / model_name
    ordinary_evidence = ROOT / "docs/evidence" / ordinary_name
    feature = read(run / "features.json")
    observed = read(run / "result.json")
    independent = read(ordinary_evidence / "result.json")
    independent_scope = read(ordinary_evidence / "scope.json")
    summary = read(model_run / "summary.json")
    manifest = read(model_run / "checkpoint/manifest.json")
    job = read(ROOT / "runs" / (args.name + "-job.json"))
    validate_completion(summary, read(ROOT / "runs" / (model_name + "-job.json")))
    verify_source(model_run / "code", summary["catalogue_sha256"])
    command = job["command"]
    if (job["status"] != "passed" or job["exit_code"] != 0 or job["work_items"] != count or
            job["memory_bytes"] != memory * 1024**2 or job["wall_seconds"] != 120 or
            job["system_pressure_guard"] != "GetPerformanceInfo/0.2s" or
            job["supervisor_script_sha256"] != digest(ROOT / "scripts/run_bounded.py") or
            len(command) != 6 or Path(command[1]).resolve() != Path(__file__).with_name("probe_layer_features.py").resolve() or
            command[2:5] != ["--run", model_name, "--output"] or Path(command[5]).resolve() != run):
        raise ValueError("actual native feature command/budget differs")
    architecture = checkpoint_architecture(summary, manifest)
    for result in (feature, observed, independent, independent_scope):
        if (result["status"] != "passed" or result["global_steps"] != steps or result["target_steps"] != steps or
                result["forwards"] != count or result["weights_sha256"] != summary["checkpoint_sha256"] or
                result["source_catalogue_sha256"] != summary["catalogue_sha256"] or
                result["core_architecture"] != architecture or result["weights_unchanged"] is not True or
                result["automatic_metrics_reproduced"] is not True):
            raise ValueError("feature and independent ordinary artifact identities differ")
    if (summary["steps"] != steps or summary["requested_steps"] != steps or
            feature["run"] != model_name or observed["run"] != model_name or independent["run"] != model_name or
            feature["config"] != manifest["config"] or feature["config"] != observed["config"] or
            observed["cases"] != independent["cases"] or observed["totals"] != independent["totals"] or
            observed["label_inputs_to_model"] is not False or
            feature["exact_linear_reconstruction"] is not True or feature["observer_returns_none"] is not True or
            feature["ordinary_report_sha256"] != digest(run / "result.json") or
            independent_scope["artifact_sha256"]["result.json"] != digest(ordinary_evidence / "result.json") or
            digest(model_run / "checkpoint/weights.pt") != summary["checkpoint_sha256"] or
            feature["job_ceiling_bytes"] != job["memory_bytes"]):
        raise ValueError("actual feature/ordinary/pixel-only replay differs")
    rows = feature["features"]
    if len(rows) != 2 * count or analyze(rows) != feature["analysis"]:
        raise ValueError("feature population or recomputed grouped analysis differs")
    for index, case in enumerate(independent["cases"]):
        for offset, role in enumerate(("active", "ghost")):
            row = rows[2 * index + offset]
            layer = case["diagnostics"]["layers"][role]
            if (row["split"] != case["split"] or row["sample_id"] != case["sample_id"] or
                    row["group_id"] != case["group_id"] or row["role"] != role or
                    row["known"] != layer["target_known"] or row["target_type"] != layer["target_type"] or
                    max(range(8), key=row["logits"].__getitem__) != layer["predicted_type"]):
                raise ValueError("feature row order/support differs from independent ordinary report")
    expected_sources = ["docs/experiments/probe_layer_features.py", "docs/experiments/layer_feature_analysis.py",
                        "scripts/run_bounded.py", "scripts/probe_stream_errors.py", "scripts/m2_error_analysis.py",
                        "scripts/m3_error_analysis.py", "scripts/record_stream_run.py", "scripts/source_snapshot.py",
                        "scripts/probe_model.py"]
    if feature["sources"] != {name: digest(ROOT / name) for name in expected_sources}:
        raise ValueError("feature execution source changed")
    for name, expected in feature["sources"].items():
        if digest(run / "execution-source" / name) != expected:
            raise ValueError("copied feature execution source differs")
    collector_sources = [Path(__file__).relative_to(ROOT).as_posix(), "scripts/record_stream_errors.py"]
    source_hashes = {name: digest(ROOT / name) for name in expected_sources + collector_sources}
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *source_hashes], cwd=ROOT, check=True, capture_output=True)
    output = ROOT / "docs/evidence" / args.name
    output.mkdir(exist_ok=False)
    copies = [(run / "features.json", "features.json"), (run / "result.json", "observed-ordinary.json"),
              (model_run / "summary.json", "input-summary.json"),
              (model_run / "checkpoint/manifest.json", "checkpoint-manifest.json")]
    copies += [(ROOT / "runs" / (args.name + suffix), "native" + suffix)
               for suffix in ("-job.json", ".log", ".stderr.log")]
    copies += [(ROOT / name, "execution-source/" + name) for name in source_hashes]
    hashes = {}
    for source, destination in copies:
        target = output / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[destination] = digest(source)
        if digest(target) != hashes[destination]:
            raise ValueError("raw feature evidence bytes differ")
    if source_hashes != {name: digest(ROOT / name) for name in source_hashes}:
        raise ValueError("feature collector source changed")
    scope = {"status": "passed", "run": model_name, "global_steps": steps, "target_steps": steps,
             "core_architecture": architecture, "config": manifest["config"], "forwards": count,
             "weights_sha256": summary["checkpoint_sha256"], "source_catalogue_sha256": summary["catalogue_sha256"],
             "independent_ordinary_scope_sha256": digest(ordinary_evidence / "scope.json"),
             "independent_ordinary_report_sha256": digest(ordinary_evidence / "result.json"),
             "observer_ordinary_cases_and_totals_identical": True, "grouped_analysis_recomputed": True,
             "native_elapsed_seconds": job["elapsed_seconds"], "native_peak_job_commit_bytes": job["peak_job_commit_bytes"],
             "artifact_sha256": hashes, "execution_sources": source_hashes, "analysis": feature["analysis"],
             "scope": feature["analysis"]["scope"]}
    (output / "scope.json").write_text(json.dumps(scope, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "forwards": count, "evidence": str(output)}))


if __name__ == "__main__":
    main()
