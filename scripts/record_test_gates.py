"""Curate finite named green test gates with committed-source identities."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from probe_model import require_bounded_job

ROOT = Path(__file__).resolve().parents[1]


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--gate", action="append", required=True, help="receipt-stem:case-count")
    parser.add_argument("--source", action="append", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]{1,64}", args.name) or not 1 <= len(args.gate) <= 8 or not 1 <= len(args.source) <= 32:
        raise ValueError("finite owned evidence name/gate/source selection required")
    base = subprocess.check_output(["git", "rev-parse", "--verify", args.base+"^{commit}"], cwd=ROOT, text=True).strip()
    sources = {}
    for relative in args.source:
        path = (ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or path.stat().st_size > 1024**2:
            raise ValueError("source path/byte budget invalid")
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    subprocess.run(["git", "diff", "--exit-code", base, "--", *sources], cwd=ROOT, check=True, capture_output=True)
    runner_sha = hashlib.sha256((ROOT/"scripts/run_bounded.py").read_bytes()).hexdigest()
    gates = []
    for declaration in args.gate:
        stem, separator, count = declaration.partition(":")
        if not separator or not re.fullmatch(r"[a-z0-9-]{1,64}", stem) or not count.isdigit() or not 1 <= int(count) <= 1000:
            raise ValueError("bounded gate stem:count required")
        receipt = json.loads((ROOT/"runs"/(stem+".json")).read_text())
        stderr = (ROOT/"runs"/(stem+".stderr.log")).read_text()
        if (receipt["status"] != "passed" or receipt["work_items"] != int(count) or
                receipt["supervisor_script_sha256"] != runner_sha or
                not re.search(rf"Ran {int(count)} tests? in ", stderr) or not stderr.rstrip().endswith("OK")):
            raise ValueError("complete exact-source green gate missing")
        gates.append({"stem": stem, "tests": int(count), "raw_peak_job_commit_bytes": receipt["peak_job_commit_bytes"]})
    evidence = ROOT/"docs/evidence"/args.name
    evidence.mkdir(exist_ok=False)
    for gate in gates:
        for suffix in (".json", ".log", ".stderr.log"):
            source = ROOT/"runs"/(gate["stem"]+suffix)
            shutil.copyfile(source, evidence/source.name)
    scope = {"status": "passed", "source_base_commit": base, "sources": sources, "gates": gates,
             "tests_across_scoped_runs": sum(gate["tests"] for gate in gates), "supervisor_script_sha256": runner_sha,
             "scope": "Only the named selected test cases on the hashed sources; not a whole-suite or model-accuracy claim.",
             "curator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (evidence/"scope.json").write_text(json.dumps(scope, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "tests": scope["tests_across_scoped_runs"], "scope": str(evidence/"scope.json")}))


if __name__ == "__main__":
    main()
