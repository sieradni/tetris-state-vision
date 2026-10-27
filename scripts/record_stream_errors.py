"""Curate bounded ordinary M2/M3 saved-probe diagnostics; no quality claim."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

from m2_error_analysis import summarize
from m3_error_analysis import summarize_components
from probe_model import require_bounded_job
from record_stream_run import checkpoint_architecture, validate_completion
from source_snapshot import verify_source

ROOT = Path(__file__).resolve().parents[1]
REPORT_SOURCES = tuple("scripts/"+name for name in
    ("probe_stream_errors.py", "m2_error_analysis.py", "m3_error_analysis.py", "record_stream_run.py", "source_snapshot.py", "probe_model.py"))


def read(path, maximum=2*1024**2):
    if path.stat().st_size > maximum:
        raise ValueError("error evidence JSON byte budget exceeded")
    return json.loads(path.read_text())


def digest(path):
    value, consumed = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            consumed += len(block)
            if consumed > 64*1024**2:
                raise ValueError("error evidence artifact byte budget exceeded")
            value.update(block)
    return value.hexdigest()


def validate_report(result, job, summary, manifest, recipes, *, name):
    """Pure metadata gate; neural/pixel replay belongs to the recorded worker."""
    if (result["status"] != "passed" or result.get("diagnostic_format_version") != 2 or
            job["status"] != "passed" or job["exit_code"] != 0 or
            type(result["forwards"]) is not int or not 1 <= result["forwards"] <= 45 or
            job["work_items"] != result["forwards"] or result["forwards"] != len(result["cases"]) or
            result["weights_unchanged"] is not True or result["automatic_metrics_reproduced"] is not True or
            result["label_inputs_to_model"] is not False):
        raise ValueError("complete finite ordinary diagnostic gate required")
    command = job["command"]
    if (len(command) != 6 or Path(command[1]).resolve() != ROOT/"scripts/probe_stream_errors.py" or
            command[2] != "--run" or command[3] != result["run"] or command[4] != "--output" or
            Path(command[5]).resolve() != ROOT/"runs"/name):
        raise ValueError("native report command/output identity differs")
    architecture = checkpoint_architecture(summary, manifest)
    if (result["run_status"] != summary["status"] or result["global_steps"] != summary["steps"] or
            result["target_steps"] != summary["requested_steps"] or
            result["weights_sha256"] != summary["checkpoint_sha256"] or manifest["weights_sha256"] != result["weights_sha256"] or
            result["source_catalogue_sha256"] != summary["catalogue_sha256"] or
            result["core_architecture"] != architecture or result["config"] != manifest["config"] or
            result["component_capacity"] != summary["config"].get("component_capacity", 0) or
            result["job_ceiling_bytes"] != job["memory_bytes"]):
        raise ValueError("report checkpoint/partial-status identity differs")
    capacity = result["component_capacity"]
    totals = summarize(result["cases"])
    for split, field in (("training", "final_training_probe"), ("development", "development")):
        expected = {row["sample_id"]: row for row in summary[field]["cases"]}
        labels = {row["sample_id"]: row for row in recipes[split]}
        cases = [row for row in result["cases"] if row["split"] == split]
        if (not 1 <= len(cases) <= (9 if split == "training" else 36) or len(expected) != len(summary[field]["cases"]) or
                len(labels) != len(recipes[split]) or len(cases) != len(expected) or
                {row["sample_id"] for row in cases} != set(expected) or set(labels) != set(expected)):
            raise ValueError("saved diagnostic population differs")
        for row in cases:
            recipe = labels[row["sample_id"]]
            if any(row[key] != recipe[source] for key, source in
                   (("group_id", "group_id"), ("view", "view"), ("skin", "skin"), ("density_recipe", "density"))):
                raise ValueError("saved sample ancestry/recipe differs")
            ordinary = row["ordinary_metrics"]
            if ordinary != {key: value for key, value in expected[row["sample_id"]].items() if key != "sample_id"}:
                raise ValueError("saved ordinary metrics differ")
            diagnostic = row["diagnostics"]
            layers = [value for value in diagnostic["layers"].values() if value["target_known"]]
            if (ordinary["known_cells"] != diagnostic["locked"]["known_cells"] or
                    ordinary["exact_locked_board"] != diagnostic["locked"]["exact_board"] or
                    not math.isclose(ordinary["cell_accuracy"], diagnostic["locked"]["cell_accuracy"], abs_tol=1e-6) or
                    not math.isclose(ordinary["mean_geometry_error_cells"], diagnostic["geometry"]["mean_cells"], rel_tol=1e-5, abs_tol=1e-5) or
                    ordinary["known_layers"] != len(layers) or ordinary["exact_known_layers"] != sum(layer["exact"] for layer in layers)):
                raise ValueError("decomposition disagrees with ordinary metrics")
            if bool(capacity) != ("components" in diagnostic):
                raise ValueError("component diagnostics differ from checkpoint capacity")
            if capacity and any(diagnostic["components"][key] != value for key, value in ordinary["components"].items()):
                raise ValueError("component decomposition disagrees with ordinary metrics")
    if capacity:
        for split, values in summarize_components(result["cases"]).items():
            totals[split]["components"] = values
    if result["totals"] != totals:
        raise ValueError("diagnostic aggregate differs")
    return architecture


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"m[23]-stream-[a-z0-9-]{1,40}", args.name):
        raise ValueError("owned error-report name required")
    run = ROOT/"runs"/args.name
    result = read(run/"result.json")
    if not re.fullmatch(r"m[23]-stream-[a-z0-9-]{1,40}", result["run"]):
        raise ValueError("owned input artifact required")
    source = ROOT/"runs"/result["run"]
    summary, manifest = read(source/"summary.json"), read(source/"checkpoint/manifest.json")
    source_job = read(ROOT/"runs"/(result["run"]+"-job.json"))
    validate_completion(summary, source_job, allow_partial=True)
    verify_source(source/"code", result["source_catalogue_sha256"])
    if digest(source/"checkpoint/weights.pt") != result["weights_sha256"]:
        raise ValueError("input weight bytes differ")
    job = read(ROOT/"runs"/(args.name+"-job.json"))
    if (job["supervisor_script_sha256"] != digest(ROOT/"scripts/run_bounded.py") or
            job["system_pressure_guard"] != "GetPerformanceInfo/0.2s"):
        raise ValueError("native supervisor source/guard differs")
    recipes = {}
    for split in ("training", "development"):
        path = source/(split+"-probe.jsonl")
        if path.stat().st_size > 1024**2:
            raise ValueError("probe ledger byte budget exceeded")
        recipes[split] = [json.loads(line) for line in path.read_text().splitlines()]
    architecture = validate_report(result, job, summary, manifest, recipes, name=args.name)
    if result["sources"] != {name: digest(ROOT/name) for name in REPORT_SOURCES}:
        raise ValueError("diagnostic execution source changed")
    relative = list(REPORT_SOURCES)+[Path(__file__).relative_to(ROOT).as_posix()]
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *relative], cwd=ROOT, check=True, capture_output=True)
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    evidence = ROOT/"docs/evidence"/args.name
    evidence.mkdir(exist_ok=False)
    selected = [(run/"result.json", "result.json"), (run/"progress.json", "progress.json"),
                (source/"summary.json", "input-summary.json"), (source/"code/source.json", "source-manifest.json"),
                (source/"checkpoint/manifest.json", "checkpoint-manifest.json"),
                (ROOT/"runs"/(result["run"]+"-job.json"), "input-run-job.json")]
    selected += [(source/(split+"-probe.jsonl"), split+"-probe.jsonl") for split in ("training", "development")]
    selected += [(ROOT/"runs"/(args.name+suffix), "native"+suffix) for suffix in ("-job.json", ".log", ".stderr.log")]
    hashes = {}
    for path, name in selected:
        hashes[name] = digest(path)
        shutil.copyfile(path, evidence/name)
        if digest(evidence/name) != hashes[name]:
            raise ValueError("error evidence copy differs")
    scope = {"status": "passed", "source_base_commit": base, "core_architecture": architecture,
             **{key: result[key] for key in ("source_catalogue_sha256", "weights_sha256", "global_steps", "target_steps",
                 "run_status", "forwards", "component_capacity", "automatic_metrics_reproduced", "weights_unchanged")},
             "native_peak_job_commit_bytes": job["peak_job_commit_bytes"], "native_elapsed_seconds": job["elapsed_seconds"],
             "diagnostic_sources": result["sources"], "artifact_sha256": hashes,
             "curator_sha256": digest(Path(__file__)), "scope": result["scope"]}
    (evidence/"scope.json").write_text(json.dumps(scope, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "forwards": result["forwards"], "scope": str(evidence/"scope.json")}))


if __name__ == "__main__":
    main()
