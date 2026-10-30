"""Bounded source + current-artifact handoff ZIP; safe artifact-only restore."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import time
import zipfile

from probe_model import require_bounded_job

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "https://github.com/sieradni/tetris-state-vision"
MAX_FILES = 12000
MAX_BYTES = 2*1024**3
MAX_FILE_BYTES = 256*1024**2
MANIFEST = ".handoff/manifest.json"
MODELS = ("m2-stream-allclass1024-s3", "m2-stream-interleave1024-s3", "m2-stream-accuracy4096-s10")
RUNS = ["runs/"+name for name in MODELS]
RUNS += ["runs/"+name+"-pipeline" for name in MODELS]
RUNS += ["runs/m2-stream-accuracy-probe", "runs/m2-stream-recovery-gate-r5/profile", "runs/m2-stream-recovery-gate-r6/profile"]
RUNS += ["runs/type-head-"+study+"-"+mode+"-r1" for study in ("ordered", "interleaved") for mode in ("profile", "fit")]
RUNS += ["runs/m2-stream-"+model+"-"+kind+"-r1" for model in ("allclass1024", "interleave1024", "accuracy4096")
         for kind in ("features", "errors")]
SIDECARS = ("-job.json", ".log", ".stderr.log", "-startup.json", "-observed-startup.json",
            "-curation-job.json", "-curation.log", "-curation.stderr.log", "-curation-startup.json")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8").strip()


def member_path(name):
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or any(part in ("..", ".git", ".venv", "data") for part in path.parts):
        raise ValueError("unsafe or excluded archive path")
    if ":" in name or name.startswith("/"):
        raise ValueError("absolute archive path")
    return path


def hash_stream(stream):
    value, size = hashlib.sha256(), 0
    for block in iter(lambda: stream.read(1024**2), b""):
        size += len(block)
        if size > MAX_FILE_BYTES:
            raise ValueError("per-file byte budget exceeded")
        value.update(block)
    return value.hexdigest(), size


def hash_file(path):
    with path.open("rb") as stream:
        return hash_stream(stream)[0]


def artifacts():
    found, latest = {}, {}
    for relative in RUNS:
        folder = ROOT / relative
        if not folder.is_dir() or not folder.resolve().is_relative_to(ROOT / "runs"):
            raise ValueError("required continuation artifact root missing: "+relative)
        pointer = folder / "latest-training.json"
        generation = None
        if pointer.exists():
            state = json.loads(pointer.read_text())
            generation = state["generation"]
            if PurePosixPath(generation).name != generation or not generation.startswith("step-"):
                raise ValueError("invalid latest snapshot generation")
            manifest = folder / "training-snapshots" / generation / "manifest.json"
            if hash_file(manifest) != state["manifest_sha256"]:
                raise ValueError("actual latest optimizer manifest differs")
            latest[relative] = generation
        for current, directories, files in os.walk(folder, topdown=True, followlinks=False):
            current = Path(current)
            directories[:] = [name for name in directories if name not in ("__pycache__", ".git", ".venv")]
            if current.name == "training-snapshots":
                if generation is None or generation not in directories:
                    raise ValueError("latest snapshot root missing")
                directories[:] = [generation]
            for name in directories + files:
                candidate = current / name
                if candidate.is_symlink() or getattr(candidate, "is_junction", lambda: False)():
                    raise ValueError("linked artifact path refused")
            for name in files:
                if name.endswith((".pyc", ".pyo", ".tmp")):
                    continue
                path = current / name
                if not path.resolve().is_relative_to(ROOT / "runs"):
                    raise ValueError("artifact escaped owned runs")
                found[path.relative_to(ROOT).as_posix()] = path
        for suffix in SIDECARS:
            path = folder.parent / (folder.name+suffix)
            if path.is_file():
                if path.is_symlink():
                    raise ValueError("linked sidecar refused")
                found[path.relative_to(ROOT).as_posix()] = path
    return found, latest


def plan():
    extra, latest = artifacts()
    tracked = git("ls-tree", "-r", "--name-only", "-z", "HEAD").split("\0")
    tracked = [name for name in tracked if name]
    for name in tracked + list(extra):
        member_path(name)
    if set(tracked) & set(extra) or len(tracked)+len(extra) > MAX_FILES:
        raise ValueError("duplicate member or file budget exceeded")
    total = sum(path.stat().st_size for path in extra.values())
    if total > MAX_BYTES or any(path.stat().st_size > MAX_FILE_BYTES for path in extra.values()):
        raise ValueError("artifact byte budget exceeded")
    return {"commit": git("rev-parse", "HEAD"), "repository": REMOTE, "source_files": len(tracked),
            "artifact_files": len(extra), "artifact_bytes": total, "artifact_roots": RUNS,
            "latest_snapshots": latest, "work_limit_files": MAX_FILES, "byte_limit": MAX_BYTES}, extra


def build(destination):
    if git("status", "--porcelain"):
        raise ValueError("commit/review the clean source milestone before packaging")
    info, extra = plan()
    destination = destination.resolve()
    if (destination.is_relative_to(ROOT) or destination.exists() or
            destination.with_suffix(destination.suffix+".sha256").exists() or not destination.parent.is_dir()):
        raise ValueError("new archive outside repository with existing parent required")
    partial = destination.with_suffix(destination.suffix+".partial")
    with partial.open("xb"):
        pass
    subprocess.run(["git", "archive", "--format=zip", "--output", str(partial), info["commit"]], cwd=ROOT, check=True)
    files = []
    with zipfile.ZipFile(partial, "a", compression=zipfile.ZIP_DEFLATED, compresslevel=6, strict_timestamps=False) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            member_path(item.filename)
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("source symlink refused")
            with archive.open(item) as stream:
                digest, size = hash_stream(stream)
            files.append({"path": item.filename, "sha256": digest, "bytes": size, "kind": "source"})
        for name, path in sorted(extra.items()):
            archive.write(path, name)
            with path.open("rb") as stream:
                digest, size = hash_stream(stream)
            files.append({"path": name, "sha256": digest, "bytes": size, "kind": "artifact"})
        if len(files) > MAX_FILES or sum(item["bytes"] for item in files) > MAX_BYTES:
            raise ValueError("full handoff budget exceeded")
        info.update(format="tetris-vision-contributor-handoff-v1", files=files,
                    exclusions=[".git", ".venv", "data/private screenshots", "neighbor tetrguide",
                                "unselected historical runs", "older optimizer snapshot generations"])
        archive.writestr(MANIFEST, json.dumps(info, indent=2, allow_nan=False).encode("utf-8"))
    if git("rev-parse", "HEAD") != info["commit"] or git("status", "--porcelain"):
        raise ValueError("source checkout changed during packaging")
    partial.rename(destination)
    digest = hashlib.sha256()
    with destination.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    with destination.with_suffix(destination.suffix+".sha256").open("x", encoding="ascii") as stream:
        stream.write(digest.hexdigest()+"  "+destination.name+"\n")
    return {"status": "created", "zip": str(destination), "sha256": digest.hexdigest(),
            "commit": info["commit"], "files": len(files), "bytes": destination.stat().st_size}


def verify(path):
    with zipfile.ZipFile(path) as archive:
        manifest_info = archive.getinfo(MANIFEST)
        if manifest_info.file_size > 4*1024**2:
            raise ValueError("manifest byte budget exceeded")
        info = json.loads(archive.read(MANIFEST))
        files = info["files"]
        if (info["format"] != "tetris-vision-contributor-handoff-v1" or info["repository"] != REMOTE or
                len(files) > MAX_FILES or sum(item["bytes"] for item in files) > MAX_BYTES):
            raise ValueError("unsupported or oversized handoff")
        expected = {item["path"] for item in files} | {MANIFEST}
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != len(expected) or {item.filename for item in members} != expected:
            raise ValueError("duplicate,missing or undeclared archive members")
        for item in files:
            member_path(item["path"])
            if item["kind"] not in ("source", "artifact") or item["kind"] == "artifact" and not item["path"].startswith("runs/"):
                raise ValueError("artifact namespace differs")
            if archive.getinfo(item["path"]).file_size != item["bytes"]:
                raise ValueError("member byte count differs")
            with archive.open(item["path"]) as stream:
                digest, size = hash_stream(stream)
            if digest != item["sha256"] or size != item["bytes"]:
                raise ValueError("member integrity differs: "+item["path"])
    return info


def restore(path):
    info = verify(path)
    if git("rev-parse", "HEAD") != info["commit"] or git("status", "--porcelain"):
        raise ValueError("clean clone at the archive commit required;tracked source is never overwritten")
    files = [item for item in info["files"] if item["kind"] == "artifact"]
    # Preflight every collision before writing anything; existing run data wins.
    for item in files:
        target = ROOT / item["path"]
        if not target.resolve().is_relative_to(ROOT / "runs") or target.is_symlink():
            raise ValueError("restore target escaped runs")
        if target.exists() and (not target.is_file() or hash_file(target) != item["sha256"]):
            raise ValueError("different existing artifact;no files restored: "+item["path"])
    restored = 0
    with zipfile.ZipFile(path) as archive:
        for item in files:
            target = ROOT / item["path"]
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item["path"]) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=1024**2)
            if hash_file(target) != item["sha256"]:
                raise ValueError("restored bytes differ")
            restored += 1
    if git("status", "--porcelain"):
        raise ValueError("restore unexpectedly dirtied tracked source")
    return {"status": "restored", "commit": info["commit"], "new_artifact_files": restored,
            "tracked_files_written": 0}


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "build", "verify", "restore"))
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    if args.mode == "plan":
        result, _ = plan()
    else:
        if args.zip is None:
            parser.error("--zip required")
        if args.mode == "build":
            result = build(args.zip)
        elif args.mode == "restore":
            result = restore(args.zip)
        else:
            info = verify(args.zip)
            result = {"status": "verified", "commit": info["commit"], "files": len(info["files"])}
    result["elapsed_seconds"] = time.monotonic()-started
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
