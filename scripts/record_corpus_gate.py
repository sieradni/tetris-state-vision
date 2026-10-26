"""Curate finite corpus integrity evidence, without copying any user assets."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from probe_model import require_bounded_job
    require_bounded_job()
    name = "corpus-fixtures-r3"
    receipt = json.loads((ROOT/f"runs/{name}.json").read_text())
    log = ROOT/f"runs/{name}.stderr.log"
    if receipt["status"] != "passed" or receipt["work_items"] != 26 or "Ran 26 tests" not in log.read_text() or not log.read_text().rstrip().endswith("OK"):
        raise ValueError("complete named 26-case corpus gate required")
    tested = ("scripts/corpus_tools.py", "tests/test_corpus_tools.py")
    hashes = {path: sha(ROOT/path) for path in tested}
    # The launch checkpoint pins these hashes before this curator is invoked.
    launch = json.loads((ROOT/"runs/corpus-fixtures-r3-source.json").read_text())
    if launch["sources"] != hashes:
        raise ValueError("source changed after the named gate launched")
    evidence = ROOT/"docs/evidence/corpus-integrity-final"
    evidence.mkdir(exist_ok=False)
    for suffix in (".json", ".log", ".stderr.log", "-source.json"):
        shutil.copyfile(ROOT/f"runs/{name}{suffix}", evidence/f"{name}{suffix}")
    scope = {"status": "passed", "tests": 26, "base_commit": launch["base_commit"],
             "sources": hashes, "guard": receipt["guard"], "peak_job_commit_bytes": receipt["peak_job_commit_bytes"],
             "scope": "Structural admission, grouping, pixel identity, review/permission declarations, held-out identity and actual guarded CLI.",
             "not_claimed": "Real-image labeling, human review authenticity, perceptual near-duplicate discovery, dataset permission or model accuracy.",
             "raw_receipts": [f"{name}.json", f"{name}.log", f"{name}.stderr.log"],
             "curator_sha256": sha(Path(__file__)),
             "current_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    with (evidence/"scope.json").open("x", encoding="utf-8") as output:
        json.dump(scope, output, indent=2)
        output.write("\n")
    print(json.dumps({"status": "passed", "tests": 26, "scope": str(evidence/"scope.json")}))


if __name__ == "__main__":
    main()
