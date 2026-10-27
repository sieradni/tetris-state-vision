"""Deterministic fresh-state curriculum, with no Torch or real-image inputs."""
from dataclasses import asdict, dataclass, replace
import hashlib
import random
import struct

from tetris_vision.layout import Layout, sample_layout
from tetris_vision.model_capture import model_capture_style
from tetris_vision.render import Appearance, render, sample_scene


@dataclass
class StreamCase:
    frame: object
    record: dict


def sample_case(group_index, view, split, seed, *, profile_square=False, final_capture=False):
    if type(group_index) is not int or not 0 <= group_index < 4096 or type(view) is not int or view not in (0, 1, 2):
        raise ValueError("group 0..4095 and view 0/1/2 required")
    if split not in {"training", "development"} or type(seed) is not int or not 0 <= seed <= 2**31-1:
        raise ValueError("explicit curriculum split and finite seed required")
    if type(final_capture) is not bool:
        raise ValueError("explicit final-capture policy required")
    scene_seed = seed+group_index+(10_000_000 if split == "development" else 0)
    density = ("empty", "sparse", "dense")[group_index % 3]
    scene = sample_scene(scene_seed, density, varied=True)
    layout = sample_layout(scene_seed+0xB04D, len(scene.queue), flow="vertical")
    if profile_square:
        if (group_index, view, split) != (0, 0, "training"):
            raise ValueError("square resource profile is the first training sample only")
        layout = Layout(canvas=(512, 512), board_left=80, board_bottom=480, cell_size=16,
                        queue_origin=(290, 40), hold_origin=(8, 40), icon_size=8)
    grid = view == 0 or view == 2 and group_index % 2 == 0
    appearance = Appearance(seed=scene_seed, grid=grid, border=grid)
    intervention = None
    if view == 2:
        blocked = {cell for layer in (scene.active, scene.ghost) if layer for cell in layer.cells}
        desired_before = bool(group_index//3 % 2)
        choices = [(c, r) for r in range(20) for c in range(10) if (c, r) not in blocked and
                   (scene.locked[r][c] is not None) == desired_before]
        if not choices:
            choices = [(c, r) for r in range(20) for c in range(10) if (c, r) not in blocked]
        c, r = random.Random(scene_seed+0xCE11).choice(choices)
        locked = [list(row) for row in scene.locked]
        before = locked[r][c] is not None
        locked[r][c] = None if before else "IOTSZJLG"[(scene_seed+c+3*r) % 8]
        scene = replace(scene, locked=tuple(tuple(row) for row in locked))
        intervention = {"cell": [c, r], "before": before, "after": not before}
    if final_capture:
        appearance = model_capture_style(appearance, layout, max_side=512)
    frame = render(scene, appearance, layout)
    if frame.state.board is None or frame.state.board.lattice is None or any(cell.occupied is None for row in frame.state.board.rows for cell in row):
        raise ValueError("clean curriculum requires 200 supported source cells")
    pixel_hash = hashlib.sha256(struct.pack("<II", *frame.image.size)+frame.image.mode.encode()+frame.image.tobytes()).hexdigest()
    record = {"sample_id": f"{split}-{scene_seed}-v{view}", "group_id": f"ancestry-{split}-{scene_seed}",
              "split": split, "group_index": group_index, "view": view, "seed": scene_seed, "base_seed": seed,
              "density": density, "skin": "clean" if grid else "no_grid", "intervention": intervention,
              "profile_square": profile_square,
              "layout": asdict(layout), "appearance": asdict(appearance),
              "image_pixels_sha256": pixel_hash,
              "state_sha256": hashlib.sha256(frame.state.to_json().encode()).hexdigest(),
              "policy": "M2-stream-v1: full visible clean/no-grid/single-cell; existing observe.py; no latent target fields."}
    if final_capture:
        record.update(final_capture=True, capture_max_side=512,
                      policy="M3-stream-final-capture-v1: observe the actual bounded model frame; unknown components stay masked; no latent target fields.")
    return StreamCase(frame, record)
