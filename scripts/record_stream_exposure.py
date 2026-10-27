"""Curate native frozen target exposure with independently recomputed counts."""
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from probe_stream_exposure import ROOT, REPORT_SOURCES, LEDGERS, digest, input_contract, load_recipes, read, recipe_digest
from probe_model import require_bounded_job
from source_snapshot import verify_source
from target_exposure import require, summarize


def validate_report(result, job, summary, source_job, manifest, records, recipes, *, name):
    architecture = input_contract(summary, source_job, manifest)
    expected_count = sum(map(len, recipes.values()))
    require(result["status"] == "passed" and result["format"] == "masked-target-exposure-v1" and
            job["status"] == "passed" and job["exit_code"] == 0 and job["memory_bytes"] == 640*1024**2 and
            job["work_items"] == expected_count == result["target_records"] == len(records) and
            type(result["model_forwards"]) is int and result["model_forwards"] == 0 and
            type(result["new_updates"]) is int and result["new_updates"] == 0 and result["weights_unchanged"] is True,
            "complete finite no-forward target replay required")
    command = job["command"]
    require(len(command) == 6 and Path(command[1]).resolve() == ROOT/"scripts/probe_stream_exposure.py" and
            command[2:5] == ["--run", result["run"], "--output"] and Path(command[5]).resolve() == ROOT/"runs"/name,
            "target replay command/output differs")
    require(result["run_status"] == summary["status"] and result["global_steps"] == summary["steps"] and
            result["target_steps"] == summary["requested_steps"] and result["config"] == manifest["config"] and
            result["core_architecture"] == architecture and result["torch_version"] == manifest["torch_version"] and
            result["weights_sha256"] == summary["checkpoint_sha256"] and result["job_ceiling_bytes"] == job["memory_bytes"] and
            result["source_catalogue_sha256"] == summary["catalogue_sha256"] and
            result["type_labels"] == ["absent", *"IOTSZJL"], "target exposure artifact/dialect/partial identity differs")
    index = 0
    for population in LEDGERS:
        for recipe in recipes[population]:
            row = records[index]
            require(row["population"] == population and row["step"] == recipe.get("step") and
                    row["recipe_sha256"] == recipe_digest(recipe) and
                    all(row[key] == recipe[key] for key in ("sample_id", "group_id", "view", "density", "skin")),
                    "target replay recipe order/identity differs")
            index += 1
    require(result["counts"] == summarize(records), "target exposure aggregate differs")
    return architecture


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    require(re.fullmatch(r"m2-stream-[a-z0-9-]{1,40}", args.name), "owned exposure output required")
    output = ROOT/"runs"/args.name
    result, job = read(output/"result.json"), read(ROOT/"runs"/(args.name+"-job.json"))
    require(re.fullmatch(r"m2-stream-[a-z0-9-]{1,40}", result["run"]), "owned exposure source required")
    source = ROOT/"runs"/result["run"]
    summary, manifest = read(source/"summary.json"), read(source/"checkpoint/manifest.json")
    source_job = read(ROOT/"runs"/(result["run"]+"-job.json"))
    verify_source(source/"code", result["source_catalogue_sha256"])
    require(digest(source/"checkpoint/weights.pt") == result["weights_sha256"] and
            result["sources"] == {name: digest(ROOT/name) for name in REPORT_SOURCES} and
            job["supervisor_script_sha256"] == digest(ROOT/"scripts/run_bounded.py") and
            job["system_pressure_guard"] == "GetPerformanceInfo/0.2s", "native exposure source/weight/guard differs")
    recipes = load_recipes(source, summary)
    require(result["input_ledger_sha256"] == {key: digest(source/filename) for key, filename in LEDGERS.items()},
            "input exposure ledger bytes differ")
    ledger = output/"targets.jsonl"
    require(ledger.stat().st_size <= 16*1024**2 and digest(ledger) == result["targets_sha256"], "target ledger bytes differ")
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    architecture = validate_report(result, job, summary, source_job, manifest, records, recipes, name=args.name)
    relative = list(REPORT_SOURCES)+[Path(__file__).relative_to(ROOT).as_posix()]
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *relative], cwd=ROOT, check=True, capture_output=True)
    evidence = ROOT/"docs/evidence"/args.name
    evidence.mkdir(exist_ok=False)
    selected = [(output/name, name) for name in ("result.json", "progress.json", "targets.jsonl")]
    selected += [(source/"summary.json", "input-summary.json"), (source/"checkpoint/manifest.json", "checkpoint-manifest.json"),
                 (source/"code/source.json", "source-manifest.json"),
                 (ROOT/"runs"/(result["run"]+"-job.json"), "input-job.json")]
    selected += [(source/filename, "input-"+filename) for filename in LEDGERS.values()]
    selected += [(ROOT/"runs"/(args.name+suffix), "native"+suffix) for suffix in ("-job.json", ".log", ".stderr.log")]
    hashes = {}
    for original, filename in selected:
        shutil.copyfile(original, evidence/filename)
        hashes[filename] = digest(original)
        require(digest(evidence/filename) == hashes[filename], "exposure evidence copy differs")
    scope = {"status": "passed", "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "core_architecture": architecture, **{key: result[key] for key in
                ("source_catalogue_sha256", "weights_sha256", "global_steps", "target_steps", "run_status", "target_records",
                 "model_forwards", "new_updates", "weights_unchanged", "type_labels")},
             "sources": result["sources"], "curator_sha256": digest(Path(__file__)), "artifact_sha256": hashes,
             "native_seconds": job["elapsed_seconds"], "native_peak_bytes": job["peak_job_commit_bytes"], "scope": result["scope"]}
    (evidence/"scope.json").write_text(json.dumps(scope, indent=2)+"\n")
    print(json.dumps(scope), flush=True)


if __name__ == "__main__":
    main()
