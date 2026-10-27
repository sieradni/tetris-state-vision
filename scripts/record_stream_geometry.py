"""Preserve the oracle-geometry diagnostic without calling it parser accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from probe_model import require_bounded_job
from source_snapshot import verify_source

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    if path.stat().st_size > 64*1024**2:
        raise ValueError("geometry evidence byte budget exceeded")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"m2-stream-[a-z0-9-]{1,40}", args.name):
        raise ValueError("owned geometry-probe name required")
    run = ROOT/"runs"/args.name
    result = json.loads((run/"result.json").read_text())
    job = json.loads((ROOT/"runs"/(args.name+"-job.json")).read_text())
    if (job["status"] != "passed" or job["work_items"] != 54 or result["status"] != "passed" or
            result["forwards"] != 54 or not result["weights_unchanged"] or not result["automatic_metrics_reproduced"] or
            result["oracle_geometry_provided"] is not True or result["automatic_accuracy_claim_allowed"] is not False or
            result["script_sha256"] != digest(ROOT/"scripts/probe_stream_geometry.py")):
        raise ValueError("native exact-source diagnostic-only gate required")
    command = job["command"]
    source_name = command[command.index("--run")+1]
    if not re.fullmatch(r"m2-stream-[a-z0-9-]{1,40}", source_name):
        raise ValueError("owned source artifact required")
    source = ROOT/"runs"/source_name
    verify_source(source/"code", result["source_catalogue_sha256"])
    summary = json.loads((source/"summary.json").read_text())
    if result["weights_sha256"] != summary["checkpoint_sha256"] or digest(source/"checkpoint/weights.pt") != result["weights_sha256"]:
        raise ValueError("geometry input artifact identity differs")
    for split, field in (("training", "final_training_probe"), ("development", "development")):
        expected = {row["sample_id"]: row for row in summary[field]["cases"]}
        cases = [row for row in result["cases"] if row["split"] == split]
        if len(cases) != len(expected) or {row["sample_id"] for row in cases} != set(expected):
            raise ValueError("geometry diagnostic case population differs")
        for row in cases:
            if any(value != expected[row["sample_id"]][key] for key, value in row["automatic"].items()):
                raise ValueError("ordinary artifact metrics differ")
        if len({row["group_id"] for row in cases}) != result["totals"][split]["ancestry_groups"]:
            raise ValueError("geometry ancestry count differs")
    evidence = ROOT/"docs/evidence"/args.name
    evidence.mkdir(exist_ok=False)
    selected = [(run/"result.json", "result.json"), (run/"progress.json", "progress.json"),
                (source/"code/source.json", "source-manifest.json"), (source/"checkpoint/manifest.json", "input-checkpoint-manifest.json")]
    selected += [(ROOT/"runs"/(args.name+suffix), args.name+suffix) for suffix in ("-job.json", ".log", ".stderr.log")]
    hashes = {}
    for file, name in selected:
        hashes[name] = digest(file)
        shutil.copyfile(file, evidence/name)
        if digest(evidence/name) != hashes[name]:
            raise ValueError("geometry evidence copy differs")
    scope = {"status": "passed", "input_source_archive": str(source/"code"),
             "source_catalogue_sha256": result["source_catalogue_sha256"], "weights_sha256": result["weights_sha256"],
             "artifact_sha256": hashes, "curator_sha256": digest(Path(__file__)),
             "scope": "54-forward conditional oracle-geometry diagnostic; ordinary metrics reproduced and weights unchanged. Labels enter oracle forward; not automatic accuracy, an upper bound or deployed correction."}
    (evidence/"scope.json").write_text(json.dumps(scope, indent=2)+"\n")
    print(json.dumps({"status": "passed", "diagnostic_only": True, "scope": str(evidence/"scope.json")}))


if __name__ == "__main__":
    main()
