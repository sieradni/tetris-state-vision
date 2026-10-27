"""Preserve the oracle-geometry diagnostic without calling it parser accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from probe_model import require_bounded_job
from source_snapshot import verify_source
from geometry_contract import REPORT_SOURCES, input_contract, summarize

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    if path.stat().st_size > 64*1024**2:
        raise ValueError("geometry evidence byte budget exceeded")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    if path.stat().st_size > 1024**2:
        raise ValueError("geometry JSON byte budget exceeded")
    return json.loads(path.read_text())


def validate_report(result, job, summary, source_job, manifest, recipes, *, name):
    if (result["status"] != "passed" or result.get("diagnostic_format_version") != 2 or
            job["status"] != "passed" or job["exit_code"] != 0 or job["work_items"] != 54 or
            job["memory_bytes"] not in (640*1024**2, 768*1024**2, 1024**3) or result["forwards"] != 54 or
            result["weights_unchanged"] is not True or result["automatic_metrics_reproduced"] is not True or
            result["oracle_geometry_provided"] is not True or result["automatic_accuracy_claim_allowed"] is not False):
        raise ValueError("native exact-source conditional diagnostic-only gate required")
    command = job["command"]
    if (len(command) not in (6, 7) or Path(command[1]).resolve() != ROOT/"scripts/probe_stream_geometry.py" or
            command[2:5] != ["--run", result["run"], "--output"] or
            Path(command[5]).resolve() != ROOT/"runs"/name or len(command) == 7 and command[6] != "--allow-partial"):
        raise ValueError("native geometry command/output identity differs")
    architecture = input_contract(summary, source_job, manifest, allow_partial=len(command) == 7)
    if (result["weights_sha256"] != summary["checkpoint_sha256"] or
            result["run_status"] != summary["status"] or result["global_steps"] != summary["steps"] or
            result["target_steps"] != summary["requested_steps"] or result["core_architecture"] != architecture or
            result["config"] != manifest["config"] or result["component_capacity"] != 0 or
            result["model_kind"] != summary["model_kind"] or result["job_ceiling_bytes"] != job["memory_bytes"] or
            result["source_catalogue_sha256"] != summary["catalogue_sha256"]):
        raise ValueError("geometry checkpoint/partial-status identity differs")
    totals = summarize(result["cases"])
    for split, field in (("training", "final_training_probe"), ("development", "development")):
        expected = {row["sample_id"]: row for row in summary[field]["cases"]}
        labels = {row["sample_id"]: row for row in recipes[split]}
        cases = [row for row in result["cases"] if row["split"] == split]
        if (len(expected) != len(summary[field]["cases"]) or len(labels) != len(recipes[split]) or
                {row["sample_id"] for row in cases} != set(expected) or set(labels) != set(expected)):
            raise ValueError("geometry case population differs")
        for row in cases:
            recipe = labels[row["sample_id"]]
            if any(row[key] != recipe[key] for key in ("group_id", "view", "skin", "density")):
                raise ValueError("geometry ancestry/recipe differs")
            if any(value != expected[row["sample_id"]][key] for key, value in row["automatic"].items()):
                raise ValueError("ordinary artifact metrics differ")
            if row["automatic"]["known_layers"] != row["oracle_geometry"]["known_layers"]:
                raise ValueError("oracle target population changed")
    if result["totals"] != totals:
        raise ValueError("geometry aggregate differs")
    return architecture


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"m2-stream-[a-z0-9-]{1,40}", args.name):
        raise ValueError("owned geometry-probe name required")
    run = ROOT/"runs"/args.name
    result = read(run/"result.json")
    job = read(ROOT/"runs"/(args.name+"-job.json"))
    if (result["script_sha256"] != digest(ROOT/"scripts/probe_stream_geometry.py") or
            result["sources"] != {name: digest(ROOT/name) for name in REPORT_SOURCES} or
            job["supervisor_script_sha256"] != digest(ROOT/"scripts/run_bounded.py") or
            job["system_pressure_guard"] != "GetPerformanceInfo/0.2s"):
        raise ValueError("native exact-source diagnostic-only gate required")
    source_name = result["run"]
    if not re.fullmatch(r"m2-stream-[a-z0-9-]{1,40}", source_name):
        raise ValueError("owned source artifact required")
    source = ROOT/"runs"/source_name
    verify_source(source/"code", result["source_catalogue_sha256"])
    summary = read(source/"summary.json")
    source_job = read(ROOT/"runs"/(source_name+"-job.json"))
    manifest = read(source/"checkpoint/manifest.json")
    if result["weights_sha256"] != summary["checkpoint_sha256"] or digest(source/"checkpoint/weights.pt") != result["weights_sha256"]:
        raise ValueError("geometry input artifact identity differs")
    recipes = {}
    for split in ("training", "development"):
        path = source/(split+"-probe.jsonl")
        if path.stat().st_size > 1024**2:
            raise ValueError("finite geometry probe ledger required")
        recipes[split] = [json.loads(line) for line in path.read_text().splitlines()]
    architecture = validate_report(result, job, summary, source_job, manifest, recipes, name=args.name)
    evidence = ROOT/"docs/evidence"/args.name
    evidence.mkdir(exist_ok=False)
    selected = [(run/"result.json", "result.json"), (run/"progress.json", "progress.json"),
                (source/"code/source.json", "source-manifest.json"), (source/"checkpoint/manifest.json", "input-checkpoint-manifest.json"),
                (source/"summary.json", "input-summary.json"),
                (ROOT/"runs"/(source_name+"-job.json"), "input-job.json")]
    selected += [(source/(split+"-probe.jsonl"), split+"-probe.jsonl") for split in ("training", "development")]
    selected += [(ROOT/"runs"/(args.name+suffix), args.name+suffix) for suffix in ("-job.json", ".log", ".stderr.log")]
    hashes = {}
    for file, name in selected:
        hashes[name] = digest(file)
        shutil.copyfile(file, evidence/name)
        if digest(evidence/name) != hashes[name]:
            raise ValueError("geometry evidence copy differs")
    scope = {"status": "passed", "input_source_archive": str(source/"code"),
             "run_status": summary["status"], "global_steps": summary["steps"], "target_steps": summary["requested_steps"],
             "core_architecture": architecture, "component_capacity": 0, "diagnostic_sources": result["sources"],
             "source_catalogue_sha256": result["source_catalogue_sha256"], "weights_sha256": result["weights_sha256"],
             "artifact_sha256": hashes, "curator_sha256": digest(Path(__file__)),
             "scope": "54-forward conditional oracle-geometry diagnostic; ordinary metrics reproduced and weights unchanged. Labels enter oracle forward; not automatic accuracy, an upper bound or deployed correction."}
    (evidence/"scope.json").write_text(json.dumps(scope, indent=2)+"\n")
    print(json.dumps({"status": "passed", "diagnostic_only": True, "scope": str(evidence/"scope.json")}))


if __name__ == "__main__":
    main()
