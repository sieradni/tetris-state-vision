"""Curate a verified controlled pilot analysis and its exact source scope."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from probe_model import require_bounded_job

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from tetris_vision.dataset import source_digest


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--analysis", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"m2-learn[0-9]{1,3}", args.name) or not re.fullmatch(r"m2-analysis[0-9]{1,3}", args.analysis):
        raise ValueError("owned finite M2 names required")
    import hashlib
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    def read(path):
        return json.loads(path.read_text())
    directory = ROOT/"docs/evidence"/args.name
    analysis_root = ROOT/"runs"/args.analysis
    analysis = read(analysis_root/"analysis.json")
    summary = read(ROOT/"runs"/args.name/"summary.json")
    analysis_job = read(ROOT/"runs"/(args.analysis+"-job.json"))
    admission = read(ROOT/"runs"/(args.name+"-admission.json"))
    if analysis["status"] != "passed" or analysis_job["status"] != "passed" or not analysis["controlled_repeat"]["prefix_exact"]:
        raise ValueError("controlled analysis gate missing")
    if analysis["checkpoint_sha256"] != summary["checkpoint_sha256"] or summary["runtime_source_sha256"] != source_digest():
        raise ValueError("artifact/runtime source identity mismatch")
    if analysis["analysis_script_sha256"] != sha(ROOT/"scripts/analyze_m2_pilot.py"):
        raise ValueError("analysis source changed")
    for path, expected in admission["sources"].items():
        if sha(ROOT/path) != expected:
            raise ValueError("admitted computation source changed")
    changed = subprocess.check_output(["git", "diff", "--name-only", "e6adce8", "HEAD", "--", "src/"], cwd=ROOT, text=True).splitlines()
    if changed != ["src/tetris_vision/runtime.py"]:
        raise ValueError("unexpected source delta from initial optimizer pilot")
    destination = directory/"analysis.json"
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copyfile(analysis_root/"analysis.json", destination)
    for filename in (args.analysis+"-job.json", args.analysis+".log", args.analysis+".stderr.log", args.name+"-admission.json"):
        shutil.copyfile(ROOT/"runs"/filename, directory/filename)
    shutil.copyfile(analysis_root/"locked-errors.png", ROOT/"examples"/(args.name+"-locked-errors.png"))
    scope = {"status": "passed", "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "weight_sha256": analysis["checkpoint_sha256"], "analysis_sha256": sha(destination),
             "analysis_script_sha256": analysis["analysis_script_sha256"], "admitted_source_hashes": admission["sources"],
             "source_delta_since_initial_pilot": changed, "delta_effect": "Unused runtime caller only; optimizer/model/loss/preprocess/checkpoint identical.",
             "prefix_steps_exact": analysis["controlled_repeat"]["prefix_steps"], "manifest_identical": analysis["controlled_repeat"]["identical_manifest"],
             "analysis_peak_job_commit_bytes": analysis_job["peak_job_commit_bytes"],
             "scope": "Same data, initialization and optimizer prefix; independent checkpoint confusion/layer counts agree with run summary. Not real or complete-state accuracy.",
             "curator_sha256": sha(Path(__file__))}
    (directory/"comparison.json").write_text(json.dumps(scope, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "exact_prefix_steps": scope["prefix_steps_exact"], "comparison": str(directory/"comparison.json")}))


if __name__ == "__main__":
    main()
