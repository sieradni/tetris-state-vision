"""Versioned, hash-checked, tensor-only M2/M3 checkpoint artifacts."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from .model import ModelConfig, TetrisStateNet


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_checkpoint(directory, model, metadata):
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    temporary = directory/"weights.tmp"
    weights = directory/"weights.pt"
    torch.save(model.state_dict(), temporary)
    temporary.replace(weights)
    record = {"format_version": 1, "architecture": "TetrisStateNet-M3" if model.config.component_capacity else "TetrisStateNet-M2",
              "kind": "uncalibrated-synthetic-prototype", "config": asdict(model.config),
              "weights_sha256": file_digest(weights), "torch_version": str(torch.__version__),
              "parameters": sum(parameter.numel() for parameter in model.parameters()),
              "metadata": metadata}
    (directory/"manifest.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return record


def load_checkpoint(directory):
    directory = Path(directory)
    record = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if record.get("format_version") != 1 or record.get("architecture") not in {"TetrisStateNet-M2", "TetrisStateNet-M3"}:
        raise ValueError("unsupported checkpoint format/architecture")
    expected = "TetrisStateNet-M3" if record["config"].get("component_capacity", 0) else "TetrisStateNet-M2"
    if record["architecture"] != expected:
        raise ValueError("component capacity/architecture identity differs")
    if file_digest(directory/"weights.pt") != record["weights_sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    config = dict(record["config"])
    config["widths"] = tuple(config["widths"])
    model = TetrisStateNet(ModelConfig(**config))
    model.load_state_dict(torch.load(directory/"weights.pt", map_location="cpu", weights_only=True), strict=True)
    return model, record
