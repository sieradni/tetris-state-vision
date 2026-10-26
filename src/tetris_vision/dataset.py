"""Finite, streaming generation with deterministic ancestry groups and provenance."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, __version__ as pillow_version

from . import __version__
from .render import families, render, sample_scene
from .state import check


def split_group(group_id: str, split_seed: int = 0):
    bucket = int.from_bytes(hashlib.sha256(f"{split_seed}:{group_id}".encode()).digest()[:4], "big") % 100
    return "train" if bucket < 80 else "development" if bucket < 90 else "test"


def source_digest():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def overlay(image, state):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    if state.board and state.board.lattice:
        for r, row in enumerate(state.board.rows):
            for c, cell in enumerate(row):
                x, y = state.board.lattice.center(c, r)
                fill = (255, 209, 91) if cell.occupied else (104, 213, 184) if cell.occupied is False else (255, 88, 119)
                draw.ellipse((x-1, y-1, x+1, y+1), fill=fill)
    return result


def generate(output: Path, *, count: int = 3, seed: int = 20261025):
    check(type(count) is int and 1 <= count <= 64, "group count must be 1..64 (five images each)")
    # No append/reuse: an interrupted dataset is evidence, not an overwrite target.
    output.mkdir(parents=True, exist_ok=False)
    records = []
    source_hash = source_digest()
    montage = Image.new("RGB", (5*240, count*294), (13, 18, 29))
    caption = ImageDraw.Draw(montage)
    with (output / "manifest.jsonl").open("x", encoding="utf-8") as manifest:
        for index in range(count):
            density = ("empty", "sparse", "dense")[index % 3]
            scene_seed = seed + index
            group_id = f"state-{scene_seed}-{density}"
            scene = sample_scene(scene_seed, density)
            for column, (family, style) in enumerate(families(scene_seed).items()):
                sample_id = f"{group_id}-{family}"
                item_dir = output / sample_id
                item_dir.mkdir()
                result = render(scene, style)
                result.image.save(item_dir / "image.png")
                (item_dir / "state.json").write_text(result.state.to_json(), encoding="utf-8")
                (item_dir / "latent.json").write_text(json.dumps({"scene": asdict(scene), "appearance": asdict(style)},
                                                               indent=2) + "\n", encoding="utf-8")
                for name, mask in result.layer_masks.items():
                    mask.save(item_dir / f"mask-{name}.png")
                record = {"sample_id": sample_id, "group_id": group_id, "family": family,
                          "split": split_group(group_id), "density": density, "seed": scene_seed,
                          "generator_version": __version__, "image": f"{sample_id}/image.png",
                          "generator_source_sha256": source_hash, "pillow_version": pillow_version,
                          "state": f"{sample_id}/state.json", "latent": f"{sample_id}/latent.json",
                          "image_sha256": hashlib.sha256((item_dir / "image.png").read_bytes()).hexdigest(),
                          "state_sha256": hashlib.sha256((item_dir / "state.json").read_bytes()).hexdigest()}
                manifest.write(json.dumps(record) + "\n")
                manifest.flush()
                records.append(record)
                thumbnail = overlay(result.image, result.state)
                thumbnail.thumbnail((236, 264))
                montage.paste(thumbnail, (column*240 + (240-thumbnail.width)//2, index*294 + 26))
                caption.text((column*240+8, index*294+6), f"{density} / {family}", fill=(225, 233, 248))
                caption.text((column*240+8, index*294+276), result.state.decision, fill=(172, 188, 211))
    montage.save(output / "montage.png")
    receipt = {"status": "complete", "groups": count, "images": len(records), "seed": seed,
               "generator_version": __version__, "max_groups": 64, "max_image_pixels": 1536**2,
               "generator_source_sha256": source_hash, "pillow_version": pillow_version,
               "note": "Synthetic annotation/label-integrity pilot; no trained model or real-image score."}
    (output / "complete.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt
