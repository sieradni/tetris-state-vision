"""Bounded analysis-only fit of a copied linear head on curated frozen features."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time

from type_head_fit_contract import ROOT, RECIPE, STUDIES, digest, load_bank, metrics, read, training_indices
from probe_model import require_bounded_job

SOURCE_NAMES = ["docs/experiments/type_head_fit.py", "docs/experiments/type_head_fit_contract.py"] + [
    "scripts/"+name for name in ("probe_model.py", "run_bounded.py", "record_stream_errors.py",
                                 "record_stream_run.py", "source_snapshot.py", "m2_error_analysis.py", "m3_error_analysis.py")]


def source_hashes():
    return {name: digest(ROOT / name) for name in SOURCE_NAMES}


def equal_tree(left, right, torch):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_tree(left[key], right[key], torch) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal_tree(a, b, torch) for a, b in zip(left, right))
    return left == right


def finite_tree(value, torch):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item, torch) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(item, torch) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def validate_admission(path, study, identities, sources):
    profile = read(path / "result.json")
    scope = read(ROOT / "docs/evidence" / path.name / "scope.json")
    job = read(path.parent / (path.name+"-job.json"))
    expected = ["--study", study, "--mode", "profile", "--output", str(path)]
    command = job["command"]
    if (profile["status"] != "passed" or profile["mode"] != "profile" or profile["study"] != study or
            profile["recipe"] != RECIPE or profile["steps"] != 2 or profile["optimizer_updates_executed"] != 4 or
            profile["inputs_sha256"] != identities or profile["sources"] != sources or
            profile["warm_reload_next_update_exact"] is not True or profile["tensor_optimizer_reload_exact"] is not True or
            scope["status"] != "passed" or scope["result_sha256"] != digest(path / "result.json") or
            scope["collector_sha256"] != digest(Path(__file__).with_name("record_type_head_fit.py")) or
            job["status"] != "passed" or job["exit_code"] != 0 or job["work_items"] != 4 or
            job["memory_bytes"] != 640*1024**2 or job["wall_seconds"] != 60 or
            job["system_pressure_guard"] != "GetPerformanceInfo/0.2s" or
            job["supervisor_script_sha256"] != sources["scripts/run_bounded.py"] or len(command) != 8 or
            Path(command[1]).resolve() != Path(__file__).resolve() or command[2:7] != expected[:5] or
            Path(command[7]).resolve() != path):
        raise ValueError("source/input/recipe-bound native warm admission required")
    if digest(path / "head-state.pt") != profile["payload_sha256"]:
        raise ValueError("warm tensor payload changed")
    return profile


def update(torch, weight, bias, optimizer, features, targets, step):
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(torch.nn.functional.linear(features, weight, bias), targets)
    loss.backward()
    norm = torch.sqrt(sum(value.grad.square().sum() for value in (weight, bias)))
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)):
        raise ValueError("nonfinite diagnostic loss/gradient")
    optimizer.step()
    if not all(bool(torch.isfinite(value).all()) for value in (weight, bias)):
        raise ValueError("nonfinite diagnostic head")
    with torch.no_grad():
        correct = int((torch.nn.functional.linear(features, weight, bias).argmax(-1) == targets).sum())
    return {"step": step, "cross_entropy_before_update": float(loss.detach()),
            "gradient_l2": float(norm), "training_correct_after_update": correct}


def main():
    started = time.monotonic()
    ceiling = require_bounded_job()
    if ceiling != 640*1024**2:
        raise ValueError("exact640MiB diagnostic Job required")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=tuple(STUDIES), required=True)
    parser.add_argument("--mode", choices=("profile", "fit"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admission", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if (output.parent != ROOT / "runs" or output.exists() or
            not re.fullmatch("type-head-"+args.study+"-"+args.mode+r"-r[1-9]", output.name) or
            (args.mode == "fit") != (args.admission is not None)):
        raise ValueError("new owned output and explicit fit admission required")
    sources = source_hashes()
    if (os.environ.get("TETRIS_VISION_SUPERVISOR_SHA256") != sources["scripts/run_bounded.py"] or
            os.environ.get("TETRIS_VISION_SYSTEM_GUARD") != "GetPerformanceInfo/0.2s" or
            any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                       "OPENBLAS_DEFAULT_NUM_THREADS", "GOTO_NUM_THREADS"))):
        raise ValueError("current supervisor/pressure/pre-import CPU policy required")
    bank, model, identities = load_bank(args.study)
    admission = None
    if args.admission is not None:
        admission_path = args.admission.resolve()
        if admission_path.parent != ROOT / "runs":
            raise ValueError("owned warm profile required")
        admission = validate_admission(admission_path, args.study, identities, sources)
    output.mkdir()
    (output / "progress.json").write_text(json.dumps({"pid": os.getpid(), "phase": "initial_replay", "step": 0})+"\n")
    os.environ.update({"ONEDNN_PRIMITIVE_CACHE_CAPACITY": "16", "DNNL_PRIMITIVE_CACHE_CAPACITY": "16"})
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if str(torch.__version__) != read(model / "checkpoint/manifest.json")["torch_version"]:
        raise ValueError("actual image-artifact Torch runtime differs")
    rows = bank["features"]
    x32 = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
    recorded = torch.tensor([row["logits"] for row in rows], dtype=torch.float32)
    weights = torch.load(model / "checkpoint/weights.pt", map_location="cpu", weights_only=True)
    original_w, original_b = weights["layer_type.weight"], weights["layer_type.bias"]
    if (not torch.equal(original_w, torch.tensor(bank["type_head"]["weight"], dtype=torch.float32)) or
            not torch.equal(original_b, torch.tensor(bank["type_head"]["bias"], dtype=torch.float32)) or
            original_w.shape != (8, 64) or original_b.shape != (8,)):
        raise ValueError("copied head differs from actual image checkpoint")
    for index in range(0, 82, 2):
        logits = torch.nn.functional.linear(x32[index:index+2].reshape(1, 2, 64), original_w, original_b)
        if not torch.equal(logits[0], recorded[index:index+2]):
            raise ValueError("original per-image float32 logits fail exact replay")
    del weights
    training_ids = training_indices(rows)
    x = x32.double()
    train_x = x[training_ids]
    mean = train_x.mean(0)
    scale = train_x.std(0, correction=0).clamp_min(RECIPE["scale_floor"])
    normalized = (x-mean)/scale
    if not all(bool(torch.isfinite(value).all()) for value in (mean, scale, normalized)) or not bool((scale > 0).all()):
        raise ValueError("finite positive training-only conditioning required")
    training = normalized[training_ids]
    target = torch.tensor([rows[i]["target_type"] for i in training_ids], dtype=torch.long)
    weight = torch.nn.Parameter(original_w.double()*scale)
    bias = torch.nn.Parameter(original_b.double()+original_w.double()@mean)
    with torch.no_grad():
        initial64 = torch.nn.functional.linear(normalized, weight, bias)
        raw_initial64 = torch.nn.functional.linear(x, original_w.double(), original_b.double())
        initial_error = float((initial64-raw_initial64).abs().max())
        if initial_error > 1e-10 or not torch.equal(initial64.argmax(-1), recorded.argmax(-1)):
            raise ValueError("conditioning changed initial head function/classes")
    def optimizer_for(w, b):
        return torch.optim.AdamW([w, b], lr=RECIPE["learning_rate"], weight_decay=RECIPE["weight_decay"],
                                betas=tuple(RECIPE["betas"]), eps=RECIPE["eps"], foreach=False)
    optimizer = optimizer_for(weight, bias)
    steps = 2 if args.mode == "profile" else RECIPE["steps"]
    curve = []
    with (output / "curve.jsonl").open("x", encoding="utf-8", newline="\n") as log:
        for step in range(1, steps+1):
            if time.monotonic()-started > (45 if args.mode == "profile" else 100):
                raise RuntimeError("inner diagnostic wall cap")
            record = update(torch, weight, bias, optimizer, training, target, step)
            curve.append(record)
            log.write(json.dumps(record, allow_nan=False)+"\n")
            if admission is not None and step <= 2 and record != admission["curve"][step-1]:
                raise ValueError("fit fails exact admitted first-two update replay")
            if step % 128 == 0 or step == steps:
                (output / "progress.json").write_text(json.dumps({"pid": os.getpid(), "phase": "head_fit", "step": step})+"\n")
    payload = {"format": RECIPE["format"], "step": steps, "weight": weight.detach().clone(),
               "bias": bias.detach().clone(), "mean": mean, "scale": scale, "optimizer": optimizer.state_dict()}
    if not finite_tree(payload, torch):
        raise ValueError("nonfinite optimizer/tensor payload")
    torch.save(payload, output / "head-state.pt")
    reloaded = torch.load(output / "head-state.pt", map_location="cpu", weights_only=True)
    if not equal_tree(payload, reloaded, torch):
        raise ValueError("head/tensor/optimizer reload differs")
    # Bank is the saved step2/1024, independent of the extra profile parity updates.
    bank_w, bank_b = reloaded["weight"].clone(), reloaded["bias"].clone()
    parity = None
    executed = steps
    if args.mode == "profile":
        reload_w, reload_b = torch.nn.Parameter(bank_w.clone()), torch.nn.Parameter(bank_b.clone())
        reload_optimizer = optimizer_for(reload_w, reload_b)
        reload_optimizer.load_state_dict(copy.deepcopy(reloaded["optimizer"]))
        a = update(torch, weight, bias, optimizer, training, target, 3)
        b = update(torch, reload_w, reload_b, reload_optimizer, training, target, 3)
        if (a != b or not torch.equal(weight, reload_w) or not torch.equal(bias, reload_b) or
                not equal_tree(optimizer.state_dict(), reload_optimizer.state_dict(), torch) or
                not torch.equal(torch.nn.functional.linear(normalized, weight, bias),
                                torch.nn.functional.linear(normalized, reload_w, reload_b))):
            raise ValueError("warm continuous/reloaded next update differs")
        parity, executed = a, 4
    with torch.no_grad():
        normalized_logits = torch.nn.functional.linear(normalized, bank_w, bank_b)
        raw_w = bank_w/scale
        raw_b = bank_b-raw_w@mean
        raw64 = torch.nn.functional.linear(x, raw_w, raw_b)
        raw32 = torch.nn.functional.linear(x32, raw_w.float(), raw_b.float())
        if not all(bool(torch.isfinite(tensor).all()) for tensor in (normalized_logits, raw_w, raw_b, raw64, raw32)):
            raise ValueError("nonfinite raw-head conversion")
        conversion_error = float((normalized_logits-raw64).abs().max())
        converted_classes_match = torch.equal(normalized_logits.argmax(-1), raw64.argmax(-1))
    if sources != source_hashes() or any(digest(ROOT / path) != value for path, value in identities.items()):
        raise ValueError("diagnostic input/source changed")
    for name in sources:
        destination = output / "execution-source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        if digest(destination) != sources[name]:
            raise ValueError("diagnostic source byte copy differs")
    result = {"status": "passed", "study": args.study, "mode": args.mode, "recipe": RECIPE,
              "steps": steps, "optimizer_updates_executed": executed, "curve": curve,
              "job_ceiling_bytes": ceiling, "pid": os.getpid(), "torch_version": str(torch.__version__),
              "model_run": model.name, "core_architecture": bank["core_architecture"], "model_config": bank["config"],
              "weights_sha256": bank["weights_sha256"], "source_catalogue_sha256": bank["source_catalogue_sha256"],
              "inputs_sha256": identities, "sources": sources, "original_per_image_logits_exact": True,
              "initial_conditioning_max_abs_error": initial_error,
              "tensor_optimizer_reload_exact": True, "warm_reload_next_update_exact": parity is not None,
              "warm_next_record": parity, "payload_sha256": digest(output / "head-state.pt"),
              "head": {"weight_normalized": bank_w.tolist(), "bias_normalized": bank_b.tolist(),
                       "mean": mean.tolist(), "scale": scale.tolist(), "weight_raw64": raw_w.tolist(),
                       "bias_raw64": raw_b.tolist(), "weight_raw32": raw_w.float().tolist(), "bias_raw32": raw_b.float().tolist()},
              "logits": {"initial32": recorded.tolist(), "normalized64": normalized_logits.tolist(),
                         "raw64": raw64.tolist(), "raw32": raw32.tolist()},
              "metrics": {"initial32": metrics(rows, recorded.tolist()), "normalized64": metrics(rows, normalized_logits.tolist()),
                          "raw64": metrics(rows, raw64.tolist()), "raw32": metrics(rows, raw32.tolist())},
              "raw_conversion_max_abs_error": conversion_error, "raw64_classes_match": converted_classes_match,
              "raw32_training_fit": metrics(rows, raw32.tolist())["training"]["correct"] == 64,
              "elapsed_seconds": time.monotonic()-started,
              "scope": "Fixed-budget copied linear-head fit on frozen synthetic feature vectors. Separate fresh optimizer;original image model/source/weights unchanged. Raw-float32 training64/64 is a type-only representation diagnostic,not whole-state,image generalization,convergence or real accuracy. Failure is not proof of inseparability."}
    with (output / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    (output / "progress.json").write_text(json.dumps({"pid": os.getpid(), "phase": "complete", "step": steps})+"\n")
    print(json.dumps({"status": "passed", "mode": args.mode, "steps": steps,
                      "raw32_training_fit": result["raw32_training_fit"]}), flush=True)


if __name__ == "__main__":
    main()
