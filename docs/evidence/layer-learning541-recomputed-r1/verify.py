"""Read-only comparison of the541 logged versus512 banked pressure boundary."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
from probe_model import require_bounded_job
from source_snapshot import verify_source


def read(path):
    if path.stat().st_size > 1024**2:
        raise ValueError("JSON budget")
    return json.loads(path.read_text())


def prefix(path):
    with path.open("rb") as stream:
        lines = [stream.readline(65536) for _ in range(541)]
    if not all(line.endswith(b"\n") for line in lines):
        raise ValueError("incomplete541-row prefix")
    rows = [json.loads(line) for line in lines]
    if [row["step"] for row in rows] != list(range(1, 542)):
        raise ValueError("noncontiguous prefix")
    return b"".join(lines), rows


def main():
    require_bounded_job()
    old, new = (ROOT / "runs" / ("m2-stream-layer-evidence768-" + suffix) for suffix in ("s2", "s3"))
    catalogue = "f095d3526162e5bbee45e0800c017296d05d45ec78cd76f4583642c2c5b7ce47"
    for run in (old, new):
        verify_source(run / "code", catalogue)
    job = read(ROOT / "runs/m2-stream-layer-evidence768-s2-pipeline-job.json")
    if job["status"] != "memory_pressure" or job["exit_code"] != 125:
        raise ValueError("native stop identity")
    pointer = read(old / "latest-training.json")
    manifest_path = old / "training-snapshots" / pointer["generation"] / "manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != pointer["manifest_sha256"]:
        raise ValueError("parent manifest changed")
    manifest = read(manifest_path)
    summary = read(new / "summary.json")
    if (manifest["step"] != 512 or manifest["contract"]["steps"] != 768 or
            summary["steps"] != 768 or summary["status"] != "complete" or summary["resume"]["step"] != 512 or
            summary["resume"]["pointer"] != pointer):
        raise ValueError("banked/resumed cursor identity")
    if sum(1 for _ in (old / "curve.jsonl").open("rb")) != 541:
        raise ValueError("original final logged count")
    selected = {}
    for filename in ("training.jsonl", "curve.jsonl"):
        prior_bytes, prior = prefix(old / filename)
        new_bytes, current = prefix(new / filename)
        if filename == "training.jsonl":
            if prior_bytes != new_bytes:
                raise ValueError("data recipe bytes differ")
        else:
            strip = lambda rows: [{key: value for key, value in row.items() if key != "elapsed_seconds"} for row in rows]
            if strip(prior) != strip(current):
                raise ValueError("loss/terms/gradient/sample prefix differs")
        selected["old-" + filename], selected["new-" + filename] = prior_bytes, new_bytes
    output = ROOT / "docs/evidence/layer-learning541-recomputed-r1"
    output.mkdir(exist_ok=False)
    for name, data in selected.items():
        (output / name).write_bytes(data)
    shutil.copyfile(__file__, output / "verify.py")
    proof = {"status": "passed", "banked_step": 512, "logged_steps": 541, "recomputed_updates": 29,
             "all_prefix_rows_compared": 541, "source_catalogue_sha256": catalogue, "parent_pointer": pointer,
             "exact_recipe_bytes": True, "exact_loss_terms_gradients_samples_excluding_elapsed": True,
             "artifact_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in selected.items()},
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "scope": "Exact recovery/recomputed trajectory only. No old541 tensor artifact exists for equality;541 was not a saved checkpoint."}
    (output / "scope.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({"status": "passed", "recomputed_updates": 29}), flush=True)


if __name__ == "__main__":
    main()
