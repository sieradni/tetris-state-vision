"""Replay and preserve the complete32-image positive-control fixture set."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
from probe_model import require_bounded_job


def read(path):
    if path.stat().st_size > 1024**2:
        raise ValueError("metadata budget")
    return json.loads(path.read_text())


def sha(path):
    if path.stat().st_size > 1024**2:
        raise ValueError("fixture file budget")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    require_bounded_job()
    from PIL import Image
    import m2_positive_control_data as data
    from tetris_vision.dataset import source_digest
    run = ROOT / "runs/m2-positive-control-fixtures-r1"
    summary = read(run / "complete.json")
    job = read(ROOT / "runs/positive-control-generation-r1-job.json")
    expected_command = [".venv/Scripts/python.exe", "scripts/m2_positive_control_data.py", "--output", "runs/m2-positive-control-fixtures-r1"]
    if (job["status"] != "passed" or job["exit_code"] != 0 or job["memory_bytes"] != 128 * 1024**2 or
            job["work_items"] != 32 or job["wall_seconds"] != 30 or job["command"] != expected_command or
            summary["images"] != 32 or summary["ancestry_groups"] != 16 or summary["split"] != "fit" or
            summary["generator_sha256"] != sha(Path(data.__file__)) or summary["package_sha256"] != source_digest() or
            summary["manifest_sha256"] != sha(run / "manifest.jsonl")):
        raise ValueError("native command/complete fixture/source identity differs")
    recipes = [json.loads(line) for line in (run / "manifest.jsonl").read_text().splitlines()]
    if [(row["index"], row["view"]) for row in recipes] != [(i, view) for i in range(16) for view in range(2)]:
        raise ValueError("complete finite fixture population required")
    counts = {role: Counter() for role in ("active", "ghost")}
    files = [(run / "manifest.jsonl", "manifest.jsonl"), (run / "complete.json", "complete.json")]
    for record in recipes:
        case = data.sample_control(record["index"], record["view"])
        stem = case.record["sample_id"]
        actual = {**case.record, "image": stem + ".png", "state": stem + ".json", "image_sha256": sha(run / (stem + ".png"))}
        if json.loads(json.dumps(actual)) != record or sha(run / record["state"]) != record["state_sha256"]:
            raise ValueError("raw image/state/recipe identity differs")
        with Image.open(run / record["image"]) as image:
            if image.mode != case.frame.image.mode or image.size != case.frame.image.size or image.tobytes() != case.frame.image.tobytes():
                raise ValueError("actual PNG raster differs from replay")
        for role in counts:
            public = getattr(case.frame.state, role)
            counts[role][public.piece if public.status == "observed" else public.status] += 1
        files += [(run / record[key], record[key]) for key in ("image", "state")]
    if counts != summary["role_class_counts"]:
        raise ValueError("aggregate class/status counts differ")
    sources = {path.relative_to(ROOT).as_posix(): sha(path) for path in (ROOT / "src/tetris_vision").glob("*.py")}
    sources["scripts/m2_positive_control_data.py"] = sha(Path(data.__file__))
    files += [(Path(data.__file__), "generator-source.py"), (Path(__file__), "curator-source.py")]
    files += [(ROOT / "runs" / ("positive-control-generation-r1" + suffix), "positive-control-generation-r1" + suffix)
              for suffix in ("-job.json", ".log", ".stderr.log")]
    out = ROOT / "docs/evidence/positive-control-fixtures-r1"
    out.mkdir(exist_ok=False)
    hashes = {}
    for source, name in files:
        target = out / name
        shutil.copyfile(source, target)
        hashes[name] = sha(source)
        if sha(target) != hashes[name]:
            raise ValueError("evidence copy differs")
    scope = {"status": "passed", "replayed_images": 32, "ancestry_groups": 16, "split": "fit",
             "role_class_counts": counts, "sources": sources, "artifact_sha256": hashes,
             "source_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "native_generation_peak_bytes": job["peak_job_commit_bytes"], "curator_sha256": sha(Path(__file__)),
             "scope": "Public-observer labels, complete class/absence coverage, exact PNG/state/manifest/source replay. No model training, admission or accuracy claim."}
    (out / "scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    print(json.dumps({"status": "passed", "replayed_images": 32}), flush=True)


if __name__ == "__main__":
    main()
