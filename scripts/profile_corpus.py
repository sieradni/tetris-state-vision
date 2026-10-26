"""Private draft-corpus contact sheets and raster metadata, never labels."""
from collections import Counter
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from corpus_tools import Limits, asset, audit, canonical_digest, digest_file
from probe_model import require_bounded_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=200)
    args = parser.parse_args()
    require_bounded_job()
    limits = Limits(records=args.max_records, asset_bytes=128*1024**2)
    manifest_sha256 = digest_file(args.manifest, 16*1024**2)
    result = audit(args.manifest, limits)
    if result.errors:
        raise ValueError("corpus integrity audit failed")
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(result.header["image_root"]).resolve()
    dimensions, frames, pages = Counter(), [], []
    per_page, columns, tile_width, tile_height, header_height = 24, 4, 320, 240, 50
    for start in range(0, len(result.entries), per_page):
        rows = result.entries[start:start+per_page]
        sheet = Image.new("RGB", (columns*tile_width, ((len(rows)+columns-1)//columns)*tile_height+header_height), (24, 27, 32))
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 10), "UNLABELED SCREENSHOT STRESS SET - no board answers or accuracy claim", fill="white")
        draw.text((12, 27), f"Images {start+1}-{start+len(rows)} of {len(result.entries)}; complete frames, aspect preserved", fill=(170, 185, 200))
        for index, record in enumerate(rows):
            path = asset(source, record["image"])
            with Image.open(path) as image:
                width, height = image.size
                # audit() verified these bytes; the second read has its own
                # pixel admission in case the source changes between reads.
                if not 0 < width*height <= limits.image_pixels:
                    raise ValueError("source changed past pixel budget")
                dimensions[(width, height)] += 1
                alpha = image.convert("RGBA")
                background = Image.new("RGBA", alpha.size, (0, 0, 0, 255))
                background.alpha_composite(alpha)
                thumbnail = ImageOps.contain(background.convert("RGB"), (tile_width-12, tile_height-38))
            if digest_file(path) != record["image_sha256"]:
                raise ValueError("image changed while profiling")
            left, top = (index % columns)*tile_width, header_height+(index//columns)*tile_height
            sheet.paste(thumbnail, (left+(tile_width-thumbnail.width)//2, top+2))
            draw.text((left+8, top+tile_height-31), record["sample_id"], fill="white")
            draw.text((left+8, top+tile_height-17), f"{width} x {height} | {record['review']['status']} | {record['population']}", fill=(170, 185, 200))
            frames.append({"sample_id": record["sample_id"], "image": record["image"], "width": width, "height": height,
                           "image_sha256": record["image_sha256"], "pixel_sha256": record["pixel_sha256"]})
        page = args.output/f"overview-{len(pages)+1:02d}.png"
        sheet.save(page)
        pages.append({"path": page.name, "sha256": digest_file(page)})
    if digest_file(args.manifest, 16*1024**2) != manifest_sha256:
        raise ValueError("manifest changed while profiling")
    report = {"status": "complete", "images": len(frames), "contact_pages": pages,
              "dimensions": [{"width": width, "height": height, "images": count} for (width, height), count in sorted(dimensions.items())],
              "manifest_sha256": manifest_sha256, "record_identity_sha256": canonical_digest(result.entries),
              "script_sha256": digest_file(Path(__file__)), "frames": frames,
              "scope": "Raster dimensions and unlabeled visual coverage only. No annotations, accuracy metric or training admission."}
    (args.output/"profile.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "images", "contact_pages", "dimensions", "scope")}, indent=2))


if __name__ == "__main__":
    main()
