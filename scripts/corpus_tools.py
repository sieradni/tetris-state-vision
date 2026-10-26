"""Real-image corpus integrity. Review declarations never prove model accuracy.

This module changes no model/training code. CLI workloads require the existing
named Windows Job. Draft/model proposals are not eligible training labels.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from PIL import Image
from tetris_vision.state import VisionState

GROUPS = ("session", "ancestry", "skin_family", "renderer_family")
SPLITS = frozenset(("train", "development", "calibration", "test"))
POPULATIONS = {"readable": "accepted", "ambiguous": "needs_review", "no_board": "no_board", "unsupported": "unsupported"}
HEX = re.compile(r"[0-9a-f]{64}")
RECORD_KEYS = {"sample_id", "image", "image_sha256", "pixel_sha256", "annotation", "annotation_sha256",
               "split", "population", "source", "groups", "permission", "review"}


class CorpusError(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise CorpusError(reason)


@dataclass(frozen=True)
class Limits:
    records: int = 500
    asset_bytes: int = 1024**3
    image_pixels: int = 8_000_000

    def __post_init__(self):
        require(type(self.records) is int and 1 <= self.records <= 10000, "record budget must be 1..10000")
        require(type(self.asset_bytes) is int and 1 <= self.asset_bytes <= 8*1024**3, "asset-byte budget must be 1..8 GiB")
        require(type(self.image_pixels) is int and 1 <= self.image_pixels <= 8_000_000, "image pixel budget must be <=8 million")


def digest_file(path, maximum=64*1024**2):
    require(path.stat().st_size <= maximum, f"file byte budget exceeded: {path.name}")
    digest = hashlib.sha256()
    consumed = 0
    with path.open("rb") as data:
        for chunk in iter(lambda: data.read(1024**2), b""):
            consumed += len(chunk)
            require(consumed <= maximum, f"file grew past byte budget: {path.name}")
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def parse_json(value):
    def fields(pairs):
        result = {}
        for key, item in pairs:
            require(key not in result, f"duplicate JSON field: {key}")
            result[key] = item
        return result

    def invalid_constant(value):
        raise CorpusError(f"nonfinite JSON value: {value}")

    return json.loads(value, object_pairs_hook=fields, parse_constant=invalid_constant)


def asset(root, relative):
    require(isinstance(relative, str) and 0 < len(relative) <= 2048 and "\x00" not in relative, "invalid asset path")
    require(not Path(relative).is_absolute(), "asset paths must be relative to the declared root")
    candidate = (root/relative).resolve()
    require(candidate.is_relative_to(root), "asset path escapes declared root")
    require(candidate.is_file(), f"missing asset: {relative}")
    return candidate


def image_identity(path, limits):
    encoded = digest_file(path)
    with Image.open(path) as image:
        width, height = image.size
        require(0 < width*height <= limits.image_pixels, "source image pixel budget exceeded")
        require(getattr(image, "n_frames", 1) == 1, "extract and group animation/video frames explicitly")
        rgba = image.convert("RGBA")
        black = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        black.alpha_composite(rgba)
        digest = hashlib.sha256(struct.pack("<II", width, height))
        digest.update(black.convert("RGB").tobytes())
        digest.update(rgba.getchannel("A").tobytes())
    return encoded, digest.hexdigest(), (width, height)


def text(value, reason, maximum=256):
    require(isinstance(value, str) and 0 < len(value.strip()) <= maximum, reason)
    return value.strip()


def load_manifest(path, limits):
    require(path.stat().st_size <= 16*1024**2, "manifest byte budget exceeded")
    entries = []
    with path.open(encoding="utf-8") as data:
        header_line = data.readline(65537)
        require(len(header_line.encode()) <= 65536, "header line budget exceeded")
        header = parse_json(header_line)
        require(isinstance(header, dict), "corpus header must be an object")
        require(set(header) == {"format", "version", "image_root", "annotation_root", "grouping", "client_holdouts"}, "invalid corpus header fields")
        require(header["format"] == "tetris-vision-corpus" and header["version"] == 1, "unsupported corpus format")
        require(header["grouping"] == list(GROUPS), "session/ancestry/skin/renderer grouping cannot be relaxed")
        require(isinstance(header["client_holdouts"], dict), "invalid client holdouts")
        require(all(isinstance(k, str) and v in SPLITS for k, v in header["client_holdouts"].items()), "invalid holdout split")
        for key in ("image_root", "annotation_root"):
            text(header[key], "invalid root", 2048)
            require(Path(header[key]).is_absolute() and Path(header[key]).is_dir(), "declared roots must be existing absolute directories")
        while True:
            line = data.readline(65537)
            if not line:
                break
            require(len(line.encode()) <= 65536, "record line budget exceeded")
            require(line.strip(), "blank record line")
            require(len(entries) < limits.records, "record budget exceeded")
            entries.append(parse_json(line))
    require(entries, "corpus has no records")
    return header, entries


def validate_annotation(path, expected_hash, population, size):
    require(isinstance(expected_hash, str) and HEX.fullmatch(expected_hash), "invalid annotation SHA256")
    require(digest_file(path, 1024**2) == expected_hash, "annotation SHA256 mismatch")
    document = parse_json(path.read_text(encoding="utf-8"))
    require(isinstance(document, dict), "annotation must be an object")
    require(document.get("schema_version") == "1.1", "review current schema explicitly; no silent legacy migration")
    state = VisionState.from_dict(document)
    require(population in POPULATIONS and state.decision == POPULATIONS[population], "population/annotation decision mismatch")
    require(state.model_id is None and state.producer in {"human-annotation", "human-reviewed", "human-corrected"},
            "reviewed ground truth must be separate from model/synthetic proposals")
    if state.board:
        lattice = state.board.lattice
        require(lattice is not None or all(cell.occupied is None for row in state.board.rows for cell in row),
                "known canonical cells require supported geometry")
        if lattice is not None:
            def inside(c, r):
                x = lattice.origin[0]+c*lattice.u[0]+r*lattice.v[0]
                y = lattice.origin[1]+c*lattice.u[1]+r*lattice.v[1]
                return -1e-6 <= x <= size[0]+1e-6 and -1e-6 <= y <= size[1]+1e-6
            if population == "readable":
                require(all(inside(c, r) for c, r in ((0, 0), (state.board.width, 0),
                        (0, state.board.height), (state.board.width, state.board.height))), "readable board geometry outside image")
            for r, row in enumerate(state.board.rows):
                for c, cell in enumerate(row):
                    if cell.occupied is not None:
                        require(all(inside(c+dx, r+dy) for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1))),
                                "known canonical cell lies outside source image")
            for layer in (state.active, state.ghost):
                if layer.status == "observed":
                    require(all(inside(c+dx, r+dy) for c, r in layer.cells for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1))),
                            "observed canonical piece lies outside source image")
    return state


@dataclass
class Audit:
    header: dict
    entries: list
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    eligible: dict = field(default_factory=dict)
    bytes_admitted: int = 0

    def report(self):
        return {"status": "passed" if not self.errors else "failed", "records": len(self.entries),
                "drafts": sum(isinstance(row, dict) and isinstance(row.get("review"), dict) and
                              row["review"].get("status") == "draft" for row in self.entries),
                "eligible_by_split": {split: sum(value == split for value in self.eligible.values()) for split in sorted(SPLITS)},
                "asset_bytes_admitted": self.bytes_admitted, "errors": self.errors, "warnings": self.warnings,
                "scope": "Structural integrity and review/permission declarations, not proof that human labels are correct."}


def audit(manifest, limits=Limits()):
    header, entries = load_manifest(Path(manifest), limits)
    result = Audit(header, entries)
    image_root, annotation_root = (Path(header[key]).resolve() for key in ("image_root", "annotation_root"))
    ids, all_pixels, eligible_pixels, groups = set(), {}, {}, {}
    for index, row in enumerate(entries):
        sample_id = row.get("sample_id", f"record-{index}") if isinstance(row, dict) else f"record-{index}"
        try:
            require(isinstance(row, dict) and set(row) == RECORD_KEYS, "invalid record fields")
            sample_id = text(sample_id, "invalid sample id", 128)
            require(sample_id not in ids, "duplicate sample id")
            ids.add(sample_id)
            require(row["split"] in SPLITS, "invalid split")
            require(all(isinstance(row[key], dict) for key in ("source", "groups", "permission", "review")), "record declarations must be objects")
            require(set(row["source"]) == {"kind", "reference"} and row["source"]["kind"] == "real", "real-image corpus requires declared real provenance")
            text(row["source"]["reference"], "source reference required", 2048)
            require(set(row["groups"]) == {*GROUPS, "client_family"}, "invalid grouping fields")
            for value in row["groups"].values():
                text(value, "nonempty group id required", 256)
            permission, review = row["permission"], row["review"]
            require(set(permission) == {"training", "evaluation", "evidence"} and
                    type(permission["training"]) is bool and type(permission["evaluation"]) is bool, "invalid permission declaration")
            if permission["training"] or permission["evaluation"]:
                text(permission["evidence"], "permission evidence required", 2048)
            require(set(review) == {"status", "annotator", "annotator_kind", "reviewer", "reviewer_kind", "reviewed_at", "evidence"}, "invalid review fields")
            require(review["status"] in {"draft", "reviewed"}, "invalid review status")
            image = asset(image_root, row["image"])
            result.bytes_admitted += image.stat().st_size
            require(result.bytes_admitted <= limits.asset_bytes, "aggregate asset-byte budget exceeded")
            encoded, pixels, size = image_identity(image, limits)
            require(encoded == row["image_sha256"] and pixels == row["pixel_sha256"], "image byte/pixel SHA256 mismatch")
            if pixels in all_pixels:
                result.warnings.append({"sample_id": sample_id, "reason": "duplicate pixel candidate", "other": all_pixels[pixels]})
            else:
                all_pixels[pixels] = sample_id
            require((row["annotation"] is None) == (row["annotation_sha256"] is None), "annotation path/hash must appear together")
            if review["status"] == "draft":
                require(row["population"] in {*POPULATIONS, "unclassified"}, "invalid draft population")
                continue  # A valid-looking proposal never promotes itself.
            require(row["annotation"] is not None, "reviewed annotation missing")
            require(review["annotator_kind"] in {"human", "agent", "model"} and review["reviewer_kind"] == "human", "independent human review required")
            annotator = text(review["annotator"], "annotator required", 128)
            reviewer = text(review["reviewer"], "reviewer required", 128)
            require(annotator != reviewer, "independent reviewer must differ from annotator")
            text(review["evidence"], "review evidence required", 2048)
            reviewed_at = datetime.fromisoformat(text(review["reviewed_at"], "review timestamp required"))
            require(reviewed_at.tzinfo is not None, "review timestamp needs timezone")
            require(all(value.strip().casefold() not in {"unresolved", "unknown"} for value in row["groups"].values()), "resolve grouping before review admission")
            annotation = asset(annotation_root, row["annotation"])
            result.bytes_admitted += annotation.stat().st_size
            require(result.bytes_admitted <= limits.asset_bytes, "aggregate asset-byte budget exceeded")
            validate_annotation(annotation, row["annotation_sha256"], row["population"], size)
            use = "training" if row["split"] == "train" else "evaluation"
            if not permission[use]:
                result.warnings.append({"sample_id": sample_id, "reason": f"not eligible: {use} permission absent"})
                continue
            holdout = header["client_holdouts"].get(row["groups"]["client_family"])
            require(holdout is None or row["split"] == holdout, "client holdout leaked to another split")
            require(pixels not in eligible_pixels, "duplicate eligible pixels inflate independent image count")
            eligible_pixels[pixels] = sample_id
            for key in GROUPS:
                group = (key, row["groups"][key])
                require(group not in groups or groups[group] == row["split"], f"{key} leakage across splits")
                groups[group] = row["split"]
            result.eligible[sample_id] = row["split"]
        except (ValueError, TypeError, KeyError, OSError) as error:
            result.errors.append({"sample_id": sample_id, "reason": str(error)})
    return result


def init_corpus(image_root, output, limits=Limits()):
    image_root, output = Path(image_root).resolve(), Path(output).resolve()
    require(image_root.is_dir(), "image root is not a directory")
    paths = []
    for scanned, path in enumerate(image_root.iterdir(), 1):
        require(scanned <= 10000, "directory scan budget exceeded")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            require(len(paths) < limits.records, "record budget exceeded; no silent subset selection")
            paths.append(path)
    require(paths, "no supported screenshots in the specified directory")
    entries, byte_count = [], 0
    for index, path in enumerate(sorted(paths, key=lambda p: p.name.casefold())):
        path = asset(image_root, str(path.relative_to(image_root)))
        byte_count += path.stat().st_size
        require(byte_count <= limits.asset_bytes, "aggregate asset-byte budget exceeded")
        encoded, pixels, _ = image_identity(path, limits)
        entries.append({"sample_id": f"frame-{index:05d}-{encoded[:12]}", "image": str(path.relative_to(image_root)),
                        "image_sha256": encoded, "pixel_sha256": pixels, "annotation": None, "annotation_sha256": None,
                        "split": "development", "population": "unclassified", "source": {"kind": "real", "reference": "local screenshot; provenance pending review"},
                        "groups": {key: "unresolved" for key in (*GROUPS, "client_family")},
                        "permission": {"training": False, "evaluation": False, "evidence": None},
                        "review": {"status": "draft", "annotator": None, "annotator_kind": None,
                                   "reviewer": None, "reviewer_kind": None, "reviewed_at": None, "evidence": None}})
    output.mkdir(parents=True, exist_ok=False)
    (output/"annotations").mkdir()
    header = {"format": "tetris-vision-corpus", "version": 1, "image_root": str(image_root),
              "annotation_root": str(output/"annotations"), "grouping": list(GROUPS), "client_holdouts": {}}
    with (output/"manifest.jsonl").open("x", encoding="utf-8") as data:
        for value in (header, *entries):
            data.write(json.dumps(value, allow_nan=False)+"\n")
    return {"status": "created", "drafts": len(entries), "eligible": 0, "manifest": str(output/"manifest.jsonl")}


def freeze(manifest, snapshot, splits=("test",), limits=Limits()):
    result = audit(manifest, limits)
    require(not result.errors, "corpus audit failed; cannot seal")
    require(splits and set(splits) <= SPLITS, "invalid frozen split selection")
    selected = sorted((row for row in result.entries if row["split"] in splits), key=lambda row: row["sample_id"])
    require(selected and all(any(row["split"] == split for row in selected) for split in splits), "every selected split needs records")
    require(all(row["sample_id"] in result.eligible for row in selected), "every held-out record must be reviewed and permitted")
    payload = {"format": "tetris-vision-corpus-seal", "version": 1, "splits": sorted(set(splits)),
               "policy": {key: result.header[key] for key in ("grouping", "client_holdouts")},
               "records_sha256": canonical_digest(selected), "records": selected}
    with Path(snapshot).open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.write("\n")
    return {"status": "sealed", "records": len(selected), "snapshot_sha256": digest_file(Path(snapshot), 16*1024**2),
            "instruction": "Pin this snapshot SHA256 in the experiment receipt; verification requires that external digest."}


def verify_snapshot(manifest, snapshot, expected_sha, limits=Limits()):
    require(isinstance(expected_sha, str) and HEX.fullmatch(expected_sha), "external expected snapshot SHA256 required")
    require(digest_file(Path(snapshot), 16*1024**2) == expected_sha, "snapshot byte identity changed")
    frozen = parse_json(Path(snapshot).read_text(encoding="utf-8"))
    require(isinstance(frozen, dict), "snapshot must be an object")
    require(frozen.get("format") == "tetris-vision-corpus-seal" and frozen.get("version") == 1, "unsupported snapshot format")
    result = audit(manifest, limits)
    require(not result.errors, "current corpus audit failed")
    policy = {key: result.header[key] for key in ("grouping", "client_holdouts")}
    require(policy == frozen["policy"], "frozen split policy changed")
    selected = sorted((row for row in result.entries if row["split"] in frozen["splits"]), key=lambda row: row["sample_id"])
    require(canonical_digest(selected) == frozen["records_sha256"], "frozen records changed")
    require(all(row["sample_id"] in result.eligible for row in selected), "frozen record lost review/permission eligibility")
    return {"status": "verified", "records": len(selected), "snapshot_sha256": expected_sha}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-records", type=int, default=500)
    parser.add_argument("--max-asset-mib", type=int, default=1024)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--image-root", type=Path, required=True)
    init.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("audit")
    check.add_argument("manifest", type=Path)
    seal = commands.add_parser("freeze")
    seal.add_argument("manifest", type=Path)
    seal.add_argument("--snapshot", type=Path, required=True)
    seal.add_argument("--splits", nargs="+", choices=sorted(SPLITS), default=["test"])
    verify = commands.add_parser("verify")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--snapshot", type=Path, required=True)
    verify.add_argument("--expected-sha256", required=True)
    args = parser.parse_args(argv)
    limits = Limits(args.max_records, args.max_asset_mib*1024**2)
    from probe_model import require_bounded_job
    require_bounded_job()
    try:
        if args.command == "init":
            report = init_corpus(args.image_root, args.output, limits)
        elif args.command == "audit":
            report = audit(args.manifest, limits).report()
        elif args.command == "freeze":
            report = freeze(args.manifest, args.snapshot, args.splits, limits)
        else:
            report = verify_snapshot(args.manifest, args.snapshot, args.expected_sha256, limits)
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(json.dumps({"status": "failed", "reason": str(error)}))
        return 1
    print(json.dumps(report, indent=2))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
