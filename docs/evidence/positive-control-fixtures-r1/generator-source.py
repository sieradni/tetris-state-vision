"""Finite balanced all-class drawings for a future memorization diagnostic.

Not training admission, a physics simulator, or independent accuracy evidence.
"""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import struct

from tetris_vision.dataset import source_digest
from tetris_vision.layout import Layout
from tetris_vision.render import Appearance, Pose, Scene, render

PIECES = tuple("IOTSZJL")
LAYOUT = Layout(canvas=(512, 512), board_left=72, board_bottom=480, cell_size=16,
                queue_origin=(300, 40), hold_origin=(8, 40), icon_size=8)


@dataclass
class ControlCase:
    frame: object
    record: dict


def sample_control(index, view):
    if type(index) is not int or not 0 <= index < 16 or type(view) is not int or view not in (0, 1):
        raise ValueError("control index0..15 and view0/1 required")
    variant = index % 2
    piece = PIECES[index // 2] if index < 14 else None
    rows = [[None] * 10 for _ in range(20)]
    if variant == 0:
        rows[0][:8] = list("IOTSZJLG")
    active = Pose(piece, 1 if variant == 0 else 5, 16 if variant == 0 else 10, variant) if piece else None
    ghost = Pose(piece, 1 if variant == 0 else 5, 2 if variant == 0 else 6, variant) if piece else None
    scene = Scene(tuple(tuple(row) for row in rows), active, ghost, queue=("T", "O"), hold="J")
    style = Appearance(seed=20261027, grid=view == 0, border=view == 0)
    frame = render(scene, style, LAYOUT)
    if (frame.image.size != (512, 512) or not LAYOUT.contains(LAYOUT.spawn_box) or
            frame.state.board.lattice is None or
            sum(cell.occupied is not None for row in frame.state.board.rows for cell in row) != 200):
        raise ValueError("fixture lacks complete image-supported board/spawn domain")
    expected_status = "observed" if piece else "absent"
    for role, pose in (("active", active), ("ghost", ghost)):
        public = getattr(frame.state, role)
        if public.status != expected_status or piece and (public.piece != piece or set(public.cells) != set(pose.cells)):
            raise ValueError("public observer does not support the requested control role")
    pixel_hash = hashlib.sha256(struct.pack("<II", *frame.image.size) + frame.image.mode.encode() + frame.image.tobytes()).hexdigest()
    group = f"all-class-{index:02d}"
    record = {"sample_id": f"{group}-v{view}", "group_id": group, "split": "fit", "index": index, "view": view,
              "layout": asdict(LAYOUT), "appearance": asdict(style), "image_pixels_sha256": pixel_hash,
              "state_sha256": hashlib.sha256(frame.state.to_json().encode()).hexdigest(),
              "policy": "All-class fixed-geometry fit-only drawing; existing public observer; no latent supervision."}
    return ControlCase(frame, record)


def generate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    package_sha = source_digest()
    module_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    counts = {role: Counter() for role in ("active", "ghost")}
    records = []
    for index in range(16):
        for view in range(2):
            case = sample_control(index, view)
            stem = case.record["sample_id"]
            case.frame.image.save(output / (stem + ".png"))
            (output / (stem + ".json")).write_bytes(case.frame.state.to_json().encode("utf-8"))
            record = {**case.record, "image": stem + ".png", "state": stem + ".json",
                      "image_sha256": hashlib.sha256((output / (stem + ".png")).read_bytes()).hexdigest()}
            for role in counts:
                public = getattr(case.frame.state, role)
                counts[role][public.piece if public.status == "observed" else public.status] += 1
            records.append(record)
    expected = {**{piece: 4 for piece in PIECES}, "absent": 4}
    if any(dict(value) != expected for value in counts.values()) or source_digest() != package_sha or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != module_sha:
        raise ValueError("control coverage/source changed")
    manifest = "".join(json.dumps(record) + "\n" for record in records)
    (output / "manifest.jsonl").write_bytes(manifest.encode("utf-8"))
    summary = {"status": "complete", "images": 32, "ancestry_groups": 16, "split": "fit",
               "role_class_counts": counts, "generator_sha256": module_sha, "package_sha256": package_sha,
               "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
               "scope": "Image-supported synthetic fit fixtures only. No model forward, training admission or real accuracy."}
    (output / "complete.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    from probe_model import require_bounded_job
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.output)), flush=True)
