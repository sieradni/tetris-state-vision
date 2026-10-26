"""Bounded byte-verified source archives for isolated training imports."""
import hashlib
import json
from pathlib import Path
import shutil

SCRIPTS = ("train_m2_stream.py", "m2_stream_data.py", "source_snapshot.py", "probe_model.py", "run_bounded.py",
           "training_snapshot.py", "atomic_write.py")


def sha(path):
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as data:
        for chunk in iter(lambda: data.read(65536), b""):
            count += len(chunk)
            if count > 1024**2:
                raise ValueError("source file grew past byte budget")
            digest.update(chunk)
    return digest.hexdigest()


def catalogue_digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def capture_source(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    source = root/"src/tetris_vision" if (root/"src/tetris_vision").is_dir() else root/"tetris_vision"
    paths = {}
    # The repository package is finite and task-owned; no runs, data, native
    # binaries or external packages are copied. Recursive modules are covered.
    for path in source.rglob("*.py"):
        if "__pycache__" not in path.parts:
            relative = "tetris_vision/"+path.relative_to(source).as_posix()
            paths[relative] = path
            if len(paths) > 128:
                raise ValueError("source module budget exceeded")
    for name in SCRIPTS:
        paths["scripts/"+name] = root/"scripts"/name
    for name in ("pyproject.toml", "uv.lock"):
        paths[name] = root/name
    if "tetris_vision/__init__.py" not in paths:
        raise ValueError("package source missing")
    total, files = 0, {}
    for relative, path in paths.items():
        if not path.resolve().is_relative_to(root) or not path.is_file() or path.stat().st_size > 1024**2:
            raise ValueError("source path or per-file byte budget invalid")
        total += path.stat().st_size
        if total > 16*1024**2:
            raise ValueError("aggregate source byte budget exceeded")
        files[relative] = sha(path)
    output.mkdir(parents=True, exist_ok=False)
    for relative, path in paths.items():
        target = output/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha(target) != files[relative]:
            raise ValueError("source changed while copying")
    record = {"format": "tetris-vision-training-source", "version": 1, "files": files,
              "catalogue_sha256": catalogue_digest(files), "bytes": total}
    (output/"source.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    verify_source(output)
    return record


def verify_source(root, expected_catalogue=None):
    root = Path(root).resolve()
    path = root/"source.json"
    if path.stat().st_size > 128*1024:
        raise ValueError("source manifest byte budget exceeded")
    record = json.loads(path.read_text())
    if record["format"] != "tetris-vision-training-source" or record["version"] != 1 or not 1 <= len(record["files"]) <= 128+len(SCRIPTS)+2:
        raise ValueError("unsupported source manifest")
    digest = catalogue_digest(record["files"])
    if digest != record["catalogue_sha256"] or expected_catalogue is not None and digest != expected_catalogue:
        raise ValueError("source catalogue identity changed")
    for relative, expected in record["files"].items():
        candidate = (root/relative).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file() or candidate.stat().st_size > 1024**2 or sha(candidate) != expected:
            raise ValueError("archived source identity changed")
    # Extra importable modules must not enter the archive unnoticed.
    declared = {key for key in record["files"] if key.endswith(".py")}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*.py") if "__pycache__" not in path.parts}
    if actual != declared:
        raise ValueError("unlisted source module present")
    return record
