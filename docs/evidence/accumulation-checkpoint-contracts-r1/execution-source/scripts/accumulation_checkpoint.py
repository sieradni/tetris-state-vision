"""Durable group-boundary journal/snapshot bridge for a future trainer.

No training admission, source capture, scheduler or process launcher is
provided here. A caller must run inside the existing bounded Job and abort on
any append/snapshot/optimizer failure rather than publish an uncertain state.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re

from accumulation_contract import IMAGE_BUDGET, POLICIES, update_count
from accumulation_journal import require_optimizer_ownership, validate_ledger, verify_resume_cursor
from training_snapshot import MAX_LEDGER, load_training_snapshot, save_training_snapshot

LEDGER = "groups.jsonl"


def make_contract(policy, *, source_catalogue_sha256, config):
    """Bind the fixed proposal's training recipe to its frozen actual source."""
    target_updates = update_count(policy)
    if type(source_catalogue_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", source_catalogue_sha256):
        raise ValueError("actual source catalogue digest required")
    if type(config) is not dict or not config or config.get("component_capacity", 0) != 0 or config.get("layer_evidence", False):
        raise ValueError("plain M2 configuration required")
    config = json.loads(json.dumps(config, allow_nan=False))
    return {"format": "fixed-fit-accumulation-v1", "policy": policy,
            "source_catalogue_sha256": source_catalogue_sha256, "config": config,
            "target_images": IMAGE_BUDGET, "target_updates": target_updates,
            "images_per_update": POLICIES[policy][0], "seed": 20261026,
            "optimizer": "AdamW", "learning_rate": .001, "weight_decay": .01,
            "betas": [.9, .999], "epsilon": 1e-8,
            "clip_norm": 1., "checkpoint_every_images": 64, "evaluate_every_images": 128,
            "input_policy": "M2-prepared-resize-v1", "rng": "torch_cpu_only", "threads": 1}


def _verify_optimizer(optimizer, contract, *, expected_updates=None):
    """Refuse a different AdamW recipe or incorrectly advanced optimizer."""
    groups = optimizer.param_groups
    if len(groups) != 1:
        raise ValueError("one shared AdamW optimizer group required")
    group = groups[0]
    if (group.get("lr") != contract["learning_rate"] or group.get("weight_decay") != contract["weight_decay"] or
            list(group.get("betas", ())) != contract["betas"] or group.get("eps") != contract["epsilon"] or
            group.get("amsgrad", False) is not False or group.get("maximize", False) is not False):
        raise ValueError("actual optimizer hyperparameters differ from the contract")
    if expected_updates is None:
        return
    for parameter in group["params"]:
        state = optimizer.state.get(parameter, {})
        step = state.get("step")
        if hasattr(step, "item"):
            step = step.item()
        if type(step) not in (int, float) or not math.isfinite(step) or step != expected_updates:
            raise ValueError("actual AdamW step differs from committed update cursor")


def _ledger_path(run):
    run = Path(run).resolve()
    path = run / LEDGER
    if not run.is_dir() or not path.resolve().is_relative_to(run):
        raise ValueError("owned run directory and contained journal required")
    return path


def create_journal(run):
    """Create only in a fresh owned directory; never replace a prior journal."""
    with _ledger_path(run).open("xb") as stream:
        stream.flush()
        os.fsync(stream.fileno())


def _parse_prefix(data, policy):
    if len(data) > MAX_LEDGER or data and not data.endswith(b"\n"):
        raise ValueError("bounded complete-line journal prefix required")
    try:
        entries = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    except (UnicodeError, ValueError) as error:
        raise ValueError("invalid UTF-8 JSON journal") from error
    return entries, validate_ledger(policy, entries)


def append_group(run, policy, entry):
    """Append only a validated whole group after its successful optimizer step."""
    path = _ledger_path(run)
    if path.stat().st_size > MAX_LEDGER:
        raise ValueError("journal byte budget exceeded")
    entries, _ = _parse_prefix(path.read_bytes(), policy)
    state = validate_ledger(policy, entries + [entry])
    wire = (json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode("utf-8")
    if path.stat().st_size + len(wire) > MAX_LEDGER:
        raise ValueError("journal byte budget exceeded")
    with path.open("ab") as stream:
        stream.write(wire)
        stream.flush()
        os.fsync(stream.fileno())
    return state


def save_boundary(run, model, optimizer, contract):
    """Publish model, Adam, RNG and the validated committed-group prefix."""
    policy = contract["policy"]
    if contract != make_contract(policy, source_catalogue_sha256=contract["source_catalogue_sha256"], config=contract["config"]):
        raise ValueError("fixed recovery contract differs")
    require_optimizer_ownership(model, optimizer)
    if any(getattr(parameter, "grad", None) is not None for parameter in model.parameters()):
        raise ValueError("clear gradients at a complete group boundary before snapshot")
    entries, state = _parse_prefix(_ledger_path(run).read_bytes(), policy)
    if not entries:
        raise ValueError("at least one complete update is required")
    verify_resume_cursor(policy, entries, snapshot_images=state["images_seen"], snapshot_updates=state["optimizer_updates"])
    _verify_optimizer(optimizer, contract, expected_updates=state["optimizer_updates"])
    return save_training_snapshot(run, model, optimizer, step=state["optimizer_updates"], contract=contract, ledgers=(LEDGER,))


def _preflight_source(source, contract):
    """Verify pointer, exact saved journal prefix and cursor before model mutation."""
    source = Path(source).resolve()
    pointer_path = source / "latest-training.json"
    if pointer_path.stat().st_size > 64*1024:
        raise ValueError("bounded pointer required")
    pointer = json.loads(pointer_path.read_text())
    generation = pointer.get("generation", "")
    if (pointer.get("format") != "tetris-vision-training-snapshot" or pointer.get("version") != 1 or
            not isinstance(generation, str) or not re.fullmatch(r"step-[0-9]{6}-[0-9a-f]{32}", generation)):
        raise ValueError("invalid snapshot pointer")
    manifest = (source / "training-snapshots" / generation / "manifest.json").resolve()
    if not manifest.is_relative_to(source) or manifest.stat().st_size > 64*1024:
        raise ValueError("invalid snapshot manifest path")
    data = manifest.read_bytes()
    if hashlib.sha256(data).hexdigest() != pointer.get("manifest_sha256"):
        raise ValueError("snapshot manifest identity differs")
    record = json.loads(data)
    if record.get("contract") != contract or set(record.get("ledgers", {})) != {LEDGER}:
        raise ValueError("snapshot contract or journal set differs")
    cursor = record["ledgers"][LEDGER]
    size = cursor.get("bytes")
    if type(size) is not int or not 0 <= size <= MAX_LEDGER:
        raise ValueError("invalid journal byte cursor")
    with _ledger_path(source).open("rb") as stream:
        prefix = stream.read(size)
    if len(prefix) != size or hashlib.sha256(prefix).hexdigest() != cursor.get("sha256"):
        raise ValueError("committed journal prefix differs")
    entries, state = _parse_prefix(prefix, contract["policy"])
    verify_resume_cursor(contract["policy"], entries, snapshot_images=state["images_seen"], snapshot_updates=record.get("step"))
    return prefix, state


def restore_boundary(source, destination, model, optimizer, contract):
    """Restore into a new directory, excluding all uncommitted suffix data."""
    policy = contract["policy"]
    if contract != make_contract(policy, source_catalogue_sha256=contract["source_catalogue_sha256"], config=contract["config"]):
        raise ValueError("fixed recovery contract differs")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or _ledger_path(destination).exists():
        raise ValueError("fresh separate destination journal required")
    require_optimizer_ownership(model, optimizer)
    _verify_optimizer(optimizer, contract)
    prefix, state = _preflight_source(source, contract)
    loaded = load_training_snapshot(source, model, optimizer, contract=contract)
    if loaded["step"] != state["optimizer_updates"] or loaded["ledger_prefixes"] != {LEDGER: prefix}:
        raise ValueError("loaded snapshot differs from preflight")
    _verify_optimizer(optimizer, contract, expected_updates=state["optimizer_updates"])
    with _ledger_path(destination).open("xb") as stream:
        stream.write(prefix)
        stream.flush()
        os.fsync(stream.fileno())
    return {**state, "source_pointer": loaded["pointer"]}
