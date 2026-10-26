"""Verify and curate this milestone's evidence; never assert model accuracy."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from tetris_vision.dataset import source_digest
from tetris_vision.state import VisionState


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-brief", type=Path, required=True)
    parser.add_argument("--test-run", required=True)
    args = parser.parse_args()
    archive = ROOT/"docs/research/Tetris_Vision_Research.md"
    if args.source_brief.read_bytes() != archive.read_bytes():
        raise ValueError("archived brief differs from supplied source")
    if sha(archive).upper() not in (ROOT/"docs/CONTINUATION_STATE.md").read_text():
        raise ValueError("checkpoint source checksum mismatch")
    receipt = json.loads((ROOT/f"runs/{args.test_run}.json").read_text())
    log = (ROOT/f"runs/{args.test_run}.stderr.log").read_text()
    match = re.search(r"Ran (\d+) tests in ([\d.]+)s\s+OK\s*$", log)
    if receipt["status"] != "passed" or not match or int(match[1]) != receipt["work_items"]:
        raise ValueError("test receipt, count or log is not green")
    dataset = ROOT/"runs/demo-r1"
    demo = json.loads((ROOT/"runs/demo-r1.json").read_text())
    if demo["status"] != "passed":
        raise ValueError("demo process not green")
    completion = json.loads((dataset/"complete.json").read_text())
    if completion["generator_source_sha256"] != source_digest():
        raise ValueError("dataset was generated with a different runtime source")
    records = [json.loads(line) for line in (dataset/"manifest.jsonl").read_text().splitlines()]
    if len(records) != 15 or len({r["sample_id"] for r in records}) != 15:
        raise ValueError("expected fifteen unique demo images")
    groups = {}
    decisions = Counter()
    for record in records:
        if sha(dataset/record["image"]) != record["image_sha256"] or sha(dataset/record["state"]) != record["state_sha256"]:
            raise ValueError("demo image or state checksum mismatch")
        state = VisionState.from_json((dataset/record["state"]).read_text())
        decisions[state.decision] += 1
        groups.setdefault(record["group_id"], set()).add(record["split"])
    if len(groups) != 3 or any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("counterfactual split integrity failure")
    evidence = ROOT/"docs/evidence"
    for name in ("setup", "fixtures-r1", "fixtures-r2", "fixtures-r3", args.test_run, "demo-r1"):
        for suffix in (".json", ".log", ".stderr.log"):
            origin = ROOT/f"runs/{name}{suffix}"
            target = evidence/origin.name
            if target.exists():
                raise FileExistsError(f"refusing to replace evidence: {target}")
            shutil.copyfile(origin, target)
    shutil.copyfile(dataset/"montage.png", ROOT/"examples/montage.png")
    for family, name in (("clean", "state-complete.json"), ("cropped", "state-cropped.json")):
        record = next(r for r in records if r["family"] == family and r["density"] == "dense")
        shutil.copyfile(dataset/record["state"], ROOT/"examples"/name)
    shutil.copyfile(dataset/"manifest.jsonl", evidence/"demo-manifest.jsonl")
    source_files = list((ROOT/"src").rglob("*.py")) + list((ROOT/"tests").glob("*.py")) + list((ROOT/"scripts").glob("*.py"))
    result = {"milestone": "M0/M1 initial foundation (scoped pilot)",
              "recorded_utc": datetime.now(timezone.utc).isoformat(),
              "source_brief_sha256": sha(archive), "runtime_source_sha256": source_digest(),
              "uv_lock_sha256": sha(ROOT/"uv.lock"), "tests": int(match[1]),
              "test_seconds": float(match[2]), "test_receipt": f"{args.test_run}.json",
              "synthetic_demo": {"images": 15, "groups": 3, "decisions": dict(decisions),
                                  "montage_sha256": sha(ROOT/"examples/montage.png")},
              "verified_sources": {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
                                   for path in sorted(source_files)},
              "not_measured": ["trained model", "real-image accuracy", "legacy baseline",
                               "human-reviewed pilot", "training throughput", "export", "CPU inference latency"],
              "visual_review": "Montage inspected: bottom-first asymmetric stack, sparse/empty, "
                               "five appearance families; partial crops visibly require review.",
              "next": "Expand layout/queue/occlusion families and real annotation pilot; "
                      "then implement scratch lattice-query model with a resource-admitted learning pilot."}
    with (evidence/"foundation.json").open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps({"status": "passed", "tests": int(match[1]), "images_verified": len(records),
                      "brief_sha256": sha(archive), "evidence": str(evidence/"foundation.json")}))


if __name__ == "__main__":
    main()
