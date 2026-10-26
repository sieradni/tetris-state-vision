"""Preserve the named565-step failure and finite native-cache mechanism probes."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from probe_model import require_bounded_job
from source_snapshot import verify_source

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    if path.stat().st_size > 16*1024**2:
        raise ValueError("failure evidence file budget exceeded")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    require_bounded_job()
    name = "m2-stream-learn768-r2"
    run = ROOT/"runs"/name
    job = json.loads((ROOT/"runs"/(name+"-job.json")).read_text())
    archive = verify_source(run/"code")
    curves, ledger = ([json.loads(line) for line in (run/filename).read_text().splitlines()]
                      for filename in ("curve.jsonl", "training.jsonl"))
    if (job["status"] != "failed" or job["exit_code"] != 1 or job["work_items"] != 768 or
            job["memory_bytes"] != 640*1024**2 or
            [row["step"] for row in curves] != list(range(1, 566)) or
            [row["step"] for row in ledger] != list(range(1, 566)) or
            [row["sample_id"] for row in curves] != [row["sample_id"] for row in ledger] or
            (run/"summary.json").exists() or (run/"checkpoint/manifest.json").exists()):
        raise ValueError("named failure/no-final-weights contract differs")
    cache = {}
    selected = [(run/filename, "failed-study/"+filename) for filename in
                ("curve.jsonl", "training.jsonl", "evaluations.jsonl", "progress.json", "training-probe.jsonl", "development-probe.jsonl")]
    selected += [(run/"code/source.json", "failed-study/source-manifest.json")]
    selected += [(ROOT/"runs"/(name+suffix), "failed-study/"+name+suffix) for suffix in ("-job.json", ".log", ".stderr.log")]
    for variant, expected in (("default", "cache_hit"), ("bounded16", "cache_miss")):
        stem = "native-cache-"+variant+"-exercise"
        record = json.loads((ROOT/"runs"/(stem+".json")).read_text())
        receipt = json.loads((ROOT/"runs"/(stem+"-job.json")).read_text())
        lines = (ROOT/"runs"/(stem+".log")).read_text().splitlines()
        operations = [line for line in lines if ",primitive,create:cache_" in line and ",cpu,convolution," in line and "ih80oh80" in line]
        if (receipt["status"] != "passed" or receipt["memory_bytes"] != 512*1024**2 or receipt["work_items"] != 19 or
                record["finite_convolutions"] != 19 or len(operations) != 3 or expected not in operations[-1]):
            raise ValueError("controlled native-cache exercise differs")
        cache[variant] = {"final_repeated_shape": expected, "last_native_creation": operations[-1],
                          "requested_environment": record["environment"], "native_capacity_api_available": record["native_query_available"],
                          "raw_peak_job_commit_bytes": receipt["peak_job_commit_bytes"]}
        selected += [(ROOT/"runs"/(stem+suffix), "cache-probes/"+stem+suffix) for suffix in
                     (".json", "-job.json", ".log", ".stderr.log")]
    # Bind the exercise implementation to its committed version, not a guess.
    subprocess.run(["git", "diff", "--exit-code", "4ab205c", "--", "scripts/probe_native_cache.py"], cwd=ROOT, check=True, capture_output=True)
    evidence = ROOT/"docs/evidence/v013-failure"
    evidence.mkdir(exist_ok=False)
    hashes = {}
    for source, relative in selected:
        hashes[relative] = digest(source)
        target = evidence/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest(target) != hashes[relative]:
            raise ValueError("failure evidence copy changed bytes")
    scope = {"curation_status": "verified_copy", "run_status": "failed_no_final_weights", "completed_updates": 565,
             "requested_updates": 768, "weights_available": False, "source_archive": str(run/"code"),
             "source_catalogue_sha256": archive["catalogue_sha256"], "artifact_sha256": hashes,
             "cache_exercises": cache, "cache_probe_source_commit": "4ab205c",
             "cache_probe_sha256": digest(ROOT/"scripts/probe_native_cache.py"),
             "curator_sha256": digest(Path(__file__)),
             "scope": "Raw failed640MiB study and native cache eviction evidence. Exact cache capacity API unavailable; no attribution of every allocation, completed model, or accuracy claim."}
    (evidence/"scope.json").write_text(json.dumps(scope, indent=2)+"\n")
    print(json.dumps({"curation_status": "verified_copy", "completed_updates": 565, "final_weights": False, "scope": str(evidence/"scope.json")}))


if __name__ == "__main__":
    main()
