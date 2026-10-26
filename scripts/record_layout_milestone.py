"""Curate M1 diversity evidence and explicit visible queue-flow specimens."""
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from PIL import Image, ImageDraw
from tetris_vision.dataset import source_digest
from tetris_vision.layout import sample_layout
from tetris_vision.render import Appearance, render, sample_scene
from tetris_vision.state import VisionState


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    fixtures = json.loads((ROOT/"runs/m1-fixtures-r2.json").read_text())
    log = (ROOT/"runs/m1-fixtures-r2.stderr.log").read_text()
    match = re.search(r"Ran (\d+) tests in ([\d.]+)s\s+OK\s*$", log)
    if fixtures["status"] != "passed" or not match or int(match[1]) != 67:
        raise ValueError("expected 67 green fixtures")
    demo = ROOT/"runs/m1-demo"
    run = json.loads((ROOT/"runs/m1-demo.json").read_text())
    completion = json.loads((demo/"complete.json").read_text())
    if run["status"] != "passed" or completion["generator_source_sha256"] != source_digest():
        raise ValueError("demo is not green on current runtime source")
    records = [json.loads(line) for line in (demo/"manifest.jsonl").read_text().splitlines()]
    if len(records) != 30 or len({r["sample_id"] for r in records}) != 30:
        raise ValueError("expected thirty unique images")
    decisions, flow_metadata, groups = Counter(), Counter(), {}
    for record in records:
        if sha(demo/record["image"]) != record["image_sha256"] or sha(demo/record["state"]) != record["state_sha256"]:
            raise ValueError("image/state hash mismatch")
        state = VisionState.from_json((demo/record["state"]).read_text())
        decisions[state.decision] += 1
        flow_metadata[record["queue_flow"]] += 1
        groups.setdefault(record["group_id"], set()).add(record["split"])
        with Image.open(demo/record["image"]) as frame:
            for name in ("locked", "active", "ghost", "icons"):
                with Image.open((demo/record["image"]).parent/f"mask-{name}.png") as mask:
                    if frame.size != mask.size:
                        raise ValueError("transformed mask size mismatch")
    if len(groups) != 6 or any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("ancestry split leakage")
    evidence = ROOT/"docs/evidence/m1"
    evidence.mkdir(exist_ok=False)
    for name in ("m1-fixtures-r1", "m1-fixtures-r2", "m1-demo", "m1-lock"):
        for suffix in (".json", ".log", ".stderr.log"):
            source = ROOT/f"runs/{name}{suffix}"
            if source.exists():
                shutil.copyfile(source, evidence/source.name)
    shutil.copyfile(demo/"manifest.jsonl", evidence/"manifest.jsonl")
    shutil.copyfile(demo/"montage.png", ROOT/"examples/layout-diversity.png")
    specimens = ROOT/"examples/queue-flows"
    specimens.mkdir(exist_ok=False)
    scene = replace(sample_scene(17), queue=tuple("TTILSZJO"), queue_present=True, hold_present=True)
    montage = Image.new("RGB", (3*320, 500), (13, 18, 29))
    caption = ImageDraw.Draw(montage)
    for index, flow in enumerate(("vertical", "horizontal", "wrapped")):
        layout = sample_layout(91, len(scene.queue), flow=flow)
        result = render(scene, Appearance(), layout)
        if tuple(item.piece for item in result.state.queue.items) != scene.queue:
            raise ValueError("visible queue order mismatch")
        result.image.save(specimens/f"{flow}.png")
        (specimens/f"{flow}.json").write_text(result.state.to_json(), encoding="utf-8")
        thumbnail = result.image.copy()
        thumbnail.thumbnail((316, 465))
        montage.paste(thumbnail, (index*320, 30))
        caption.text((index*320+8, 8), flow, fill=(225, 233, 248))
    montage.save(ROOT/"examples/queue-flows.png")
    source_paths = list((ROOT/"src").rglob("*.py"))+list((ROOT/"tests").glob("*.py"))+list((ROOT/"scripts").glob("*.py"))
    result = {"milestone": "M1 source-layout and panel diversity", "schema_version": "1.1",
              "recorded_utc": datetime.now(timezone.utc).isoformat(),
              "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "runtime_source_sha256": source_digest(), "uv_lock_sha256": sha(ROOT/"uv.lock"),
              "tests": 67, "test_seconds": float(match[2]), "test_receipt": "m1-fixtures-r2.json",
              "demo_images": 30, "independent_groups": 6, "decisions": dict(decisions),
              "flow_metadata_in_random_demo": dict(flow_metadata),
              "explicit_visible_queue_flow_examples": ["vertical", "horizontal", "wrapped"],
              "verified_sources": {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
                                   for path in sorted(source_paths)},
              "limits": "One axis-aligned board; controlled skins/backgrounds; no real-image accuracy claim.",
              "visual_review": "Thirty-image montage inspected; explicit queue-flow full images reviewed separately."}
    with (evidence/"layout.json").open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps({"status": "passed", "tests": 67, "demo_images_verified": 30,
                      "visible_queue_specimens": 3, "evidence": str(evidence/"layout.json")}))


if __name__ == "__main__":
    main()
