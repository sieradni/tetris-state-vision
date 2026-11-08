"""Exact CPU AdamW recovery primitive; caller runs inside the native bounded Job.

Only Torch CPU randomness is captured. Data generation must be a deterministic
function of the recorded global step and recipe. No inference format changes.
Uncommitted generations are retained; latest.json is the final commit point.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
from atomic_write import replace_retry

MAX_TENSORS = 256*1024**2
MAX_LEDGER = 16*1024**2
MAX_JSON = 64*1024


def _json_bytes(value):
    data = (json.dumps(value, sort_keys=True, allow_nan=False)+"\n").encode()
    if len(data) > MAX_JSON:
        raise ValueError("snapshot JSON byte budget exceeded")
    return data


def _read_json(path):
    if path.stat().st_size > MAX_JSON:
        raise ValueError("snapshot JSON byte budget exceeded")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, value):
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with temporary.open("xb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    replace_retry(temporary, path)


def _digest(path, limit):
    result, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            size += len(block)
            if size > limit:
                raise ValueError("snapshot artifact byte budget exceeded")
            result.update(block)
    return result.hexdigest()


def _signature(model, optimizer):
    import torch
    if type(optimizer) is not torch.optim.AdamW:
        raise ValueError("snapshot protocol supports CPU AdamW only")
    parameters = dict(model.named_parameters())
    if any(value.device.type != "cpu" for value in model.state_dict().values()):
        raise ValueError("CPU snapshot required")
    names = {id(parameter): name for name, parameter in parameters.items()}
    groups = [[names.get(id(parameter)) for parameter in group["params"]] for group in optimizer.param_groups]
    flattened = [name for group in groups for name in group]
    if None in flattened or len(flattened) != len(set(flattened)) or set(flattened) != set(parameters):
        raise ValueError("optimizer must own each named model parameter exactly once")
    return {"tensors": {name: [list(value.shape), str(value.dtype)] for name, value in model.state_dict().items()},
            "parameter_groups": groups}


def _ledger_path(run, name):
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,39}\.jsonl", name):
        raise ValueError("bounded local ledger basename required")
    path = run/name
    if not path.resolve().is_relative_to(run):
        raise ValueError("ledger escaped run directory")
    return path


def save_training_snapshot(run, model, optimizer, *, step, contract, ledgers=()):
    import torch
    if type(step) is not int or not 1 <= step <= 4096 or type(contract) is not dict:
        raise ValueError("finite global step and explicit recipe contract required")
    contract = json.loads(_json_bytes(contract))
    if len(ledgers) > 3 or len(set(ledgers)) != len(ledgers):
        raise ValueError("at most three distinct ledgers required")
    run = Path(run).resolve()
    cursors = {}
    for name in ledgers:
        path = _ledger_path(run, name)
        # Caller must flush its writers before this commit operation.
        with path.open("r+b") as stream:
            os.fsync(stream.fileno())
        cursors[name] = {"bytes": path.stat().st_size, "sha256": _digest(path, MAX_LEDGER)}
    signature = _signature(model, optimizer)
    tensor_bytes = sum(value.numel()*value.element_size() for value in model.state_dict().values())
    tensor_bytes += sum(value.numel()*value.element_size() for state in optimizer.state.values()
                        for value in state.values() if torch.is_tensor(value))
    if tensor_bytes+1024**2 > MAX_TENSORS:
        raise ValueError("snapshot tensor byte budget exceeded before serialization")
    generation = f"step-{step:06d}-"+uuid.uuid4().hex
    folder = run/"training-snapshots"/generation
    if not folder.resolve().is_relative_to(run):
        raise ValueError("snapshot directory escaped run")
    folder.mkdir(parents=True, exist_ok=False)
    artifact = folder/"tensors.pt"
    with artifact.open("xb") as stream:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "cpu_rng": torch.get_rng_state()}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    record = {"format": "tetris-vision-training-snapshot", "version": 1, "step": step,
              "torch_version": str(torch.__version__), "contract": contract, "signature": signature,
              "module_training": {name: module.training for name, module in model.named_modules()},
              "tensors_sha256": _digest(artifact, MAX_TENSORS), "ledgers": cursors}
    _write_json(folder/"manifest.json", record)
    pointer = {"format": record["format"], "version": 1, "generation": generation,
               "manifest_sha256": _digest(folder/"manifest.json", MAX_JSON)}
    _write_json(run/"latest-training.json", pointer)
    return pointer


def load_training_snapshot(run, model, optimizer, *, contract):
    import torch
    run = Path(run).resolve()
    pointer = _read_json(run/"latest-training.json")
    if (pointer.get("format") != "tetris-vision-training-snapshot" or pointer.get("version") != 1 or
            not re.fullmatch(r"step-[0-9]{6}-[0-9a-f]{32}", pointer.get("generation", ""))):
        raise ValueError("unsupported snapshot pointer/generation")
    folder = run/"training-snapshots"/pointer["generation"]
    if (not folder.resolve().is_relative_to(run) or not (folder/"manifest.json").resolve().is_relative_to(run) or
            _digest(folder/"manifest.json", MAX_JSON) != pointer["manifest_sha256"]):
        raise ValueError("snapshot manifest identity changed")
    record = _read_json(folder/"manifest.json")
    if (record.get("format") != pointer["format"] or record.get("version") != 1 or
            type(record.get("step")) is not int or not 1 <= record["step"] <= 4096 or
            record.get("contract") != json.loads(_json_bytes(contract)) or
            record.get("torch_version") != str(torch.__version__) or record.get("signature") != _signature(model, optimizer)):
        raise ValueError("snapshot source/recipe/runtime/model identity differs")
    if not isinstance(record.get("ledgers"), dict) or len(record["ledgers"]) > 3:
        raise ValueError("invalid snapshot ledger catalogue")
    prefixes = {}
    for name, cursor in record["ledgers"].items():
        size = cursor["bytes"]
        if type(size) is not int or not 0 <= size <= MAX_LEDGER:
            raise ValueError("invalid ledger byte cursor")
        with _ledger_path(run, name).open("rb") as stream:
            data = stream.read(size)
        if len(data) != size or hashlib.sha256(data).hexdigest() != cursor["sha256"]:
            raise ValueError("snapshot ledger prefix changed/truncated")
        prefixes[name] = data
    artifact = folder/"tensors.pt"
    if not artifact.resolve().is_relative_to(run) or _digest(artifact, MAX_TENSORS) != record["tensors_sha256"]:
        raise ValueError("snapshot tensor identity changed")
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    if (payload.keys() != {"model", "optimizer", "cpu_rng"} or
            {name: [list(value.shape), str(value.dtype)] for name, value in payload["model"].items()} != record["signature"]["tensors"] or
            record["module_training"].keys() != dict(model.named_modules()).keys()):
        raise ValueError("snapshot tensor/module contract differs")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    for name, module in model.named_modules():
        module.training = record["module_training"][name]
    # Last: model construction/loading must not shift the next stochastic draw.
    torch.set_rng_state(payload["cpu_rng"])
    return {"step": record["step"], "ledger_prefixes": prefixes, "pointer": pointer}
