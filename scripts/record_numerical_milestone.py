"""Curate the M2 numerical gate on exact runtime sources; no quality claim."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from tetris_vision.dataset import source_digest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    probe = json.loads((ROOT/"runs/m2-probe-r2-result.json").read_text())
    tests = json.loads((ROOT/"runs/m2-fixtures-r1.json").read_text())
    log = (ROOT/"runs/m2-fixtures-r1.stderr.log").read_text()
    if probe["status"] != "passed" or not probe["cpu_only"] or probe["runtime_source_sha256"] != source_digest():
        raise ValueError("probe did not pass on the exact current runtime source")
    if tests["status"] != "passed" or not re.search(r"Ran 85 tests in [\d.]+s\s+OK\s*$", log):
        raise ValueError("expected 85 green tests without skips")
    output = ROOT/"docs/evidence/m2-numerics"
    output.mkdir(exist_ok=False)
    for name in ("m2-setup", "m2-lock", "m2-probe-job", "m2-probe", "m2-job-diagnostic",
                 "m2-probe-r2-job", "m2-probe-r2", "m2-fixtures-r1"):
        for suffix in (".json", ".log", ".stderr.log"):
            source = ROOT/f"runs/{name}{suffix}"
            if source.exists():
                shutil.copyfile(source, output/source.name)
    shutil.copyfile(ROOT/"runs/m2-probe-r2-result.json", output/"m2-probe-r2-result.json")
    paths = list((ROOT/"src").rglob("*.py"))+list((ROOT/"tests").glob("*.py"))+[
        ROOT/"scripts/probe_model.py", ROOT/"scripts/run_bounded.py", Path(__file__)]
    result = {"milestone": "M2 source and numerical characterization", "tests": 85,
              "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "runtime_source_sha256": source_digest(), "uv_lock_sha256": sha(ROOT/"uv.lock"),
              "parameters_pilot": probe["parameters"], "cpu_only": probe["cpu_only"],
              "one_step_probe": "m2-probe-r2-result.json", "full_suite": "m2-fixtures-r1.json",
              "verified_sources": {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in sorted(paths)},
              "accuracy_config_status": "Implemented, not trained or resource-profiled by this small-model gate",
              "not_achieved": ["clean/no-grid learning gate", "queue/hold joint model", "real-image quality",
                               "calibration", "export", "released parser"],
              "claim": "Finite forward/backward and numerical contracts, not trained-model accuracy."}
    with (output/"numerical.json").open("x", encoding="utf-8") as receipt:
        json.dump(result, receipt, indent=2)
        receipt.write("\n")
    print(json.dumps({"status": "passed", "tests": 85, "source_sha256": source_digest()}))


if __name__ == "__main__":
    main()
