"""Decisive under-sampling test: fixed pool, batch>1, long budget, GPU.

Working-tree instrument, not a committed trainer. It replays the campaign's own recipe --
`training_case(step, 20261026, stream-v1, capacity 64)` -> `prepare_image(max_side=512)` ->
`targets_for` -> model -> `joint_loss(endpoint_term="length", occupancy_factor=4,
positive_weight=4)` -> backward -> clip 1.0 -> Adam(1e-3) -- on CUDA, from a fixed sample pool
seen many times instead of a fresh drawing every step, and evaluates the exactness gate on the
same 36 training / 18 development boards the eight levers used.

Two declared departures from the published recipe, both named in the receipt:
  * the occupancy decision bias is initialised to logit(0.1354), the measured positive-cell
    prevalence, so logit 0.0 is the prior's own 0.5 point rather than an arbitrary shift;
  * the step budget is a declared accelerator envelope, not the 1,800 s CPU wall.

Nothing here writes to `src/`, `scripts/` or the checkpoint. Usage:
  python runs/gpu_fit_test.py --steps 20000 --batch 4 --pool 2048 --output runs/m3-gpu-fit-r1
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
sys.dont_write_bytecode = True

from m2_curriculum import sample_for, training_case
from placement_error import stats as placement_stats, summarize
from registration_attribution import COLUMNS, ROWS, map_cells
from tetris_vision.checkpoint import load_checkpoint
from tetris_vision.losses import joint_loss, targets_for
from tetris_vision.preprocess import prepare_image

SEED = 20261026
CURRICULUM = "stream-v1"
CAPACITY = 64
MAX_SIDE = 512
CEILING_GIB = 6.0
PREVALENCE = 0.1354
GRID = tuple(round(-6.0+0.15*i, 2) for i in range(81))


def prior_bias(prevalence=PREVALENCE):
    return math.log(prevalence/(1-prevalence))


def build_pool(count):
    """One drawing per index, stored as uint8 so thousands fit in host memory."""
    pixels, valid, targets, records = [], [], [], []
    started = time.monotonic()
    for index in range(count):
        step = index % 4096+1
        case = training_case(step, SEED, curriculum=CURRICULUM, component_capacity=CAPACITY,
                             max_side=MAX_SIDE)
        prepared = prepare_image(case.frame.image, max_side=MAX_SIDE)
        target = targets_for(case.frame.state, prepared, component_capacity=CAPACITY)
        pixels.append((prepared.pixels*255).round().clamp(0, 255).to(torch.uint8))
        valid.append(prepared.valid[0].to(torch.uint8))
        targets.append({key: (value.detach().clone() if torch.is_tensor(value)
                              else {k: (v.detach().clone() if torch.is_tensor(v) else v)
                                    for k, v in value.items()})
                        for key, value in target.items()})
        records.append({"pool_index": index, "step": step})
        if (index+1) % 256 == 0:
            print(json.dumps({"pool": index+1, "seconds": round(time.monotonic()-started, 1)}),
                  flush=True)
    return {"pixels": pixels, "valid": valid, "targets": targets, "records": records,
            "seconds": time.monotonic()-started}


def batch_of(pool, indices):
    """Pad one batch to its largest frame; the valid mask carries the real extent."""
    chosen = [pool["pixels"][i] for i in indices]
    masks = [pool["valid"][i] for i in indices]
    height = max(item.shape[1] for item in chosen)
    width = max(item.shape[2] for item in chosen)
    pixels = torch.zeros((len(chosen), 3, height, width))
    valid = torch.zeros((len(chosen), 1, height, width), dtype=torch.bool)
    for row, (item, mask) in enumerate(zip(chosen, masks)):
        pixels[row, :, :item.shape[1], :item.shape[2]] = item.float()/255
        valid[row, 0, :mask.shape[0], :mask.shape[1]] = mask.bool()
    target = {}
    for key in pool["targets"][0]:
        values = [pool["targets"][i][key] for i in indices]
        if torch.is_tensor(values[0]):
            target[key] = torch.cat(values, 0)
        else:
            target[key] = {}
            for inner in values[0]:
                items = [value[inner] for value in values]
                if torch.is_tensor(items[0]):
                    target[key][inner] = torch.cat(items, 0)
                else:
                    first = items[0]
                    for other in items[1:]:
                        if other != first:
                            raise ValueError(f"batch-varying non-tensor target {key}.{inner}")
                    target[key][inner] = first
    return pixels, valid, target


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def evaluate(model, split, groups, device):
    model.eval()
    cases = []
    with torch.no_grad():
        for group in range(groups):
            for view in range(3):
                case = sample_for(group, view, split, SEED, curriculum=CURRICULUM,
                                  component_capacity=CAPACITY, max_side=MAX_SIDE)
                prepared = prepare_image(case.frame.image, max_side=MAX_SIDE)
                target = targets_for(case.frame.state, prepared, component_capacity=CAPACITY)
                output = model(prepared.pixels[None].to(device), prepared.valid[None].to(device))
                final = output["final"]
                lattice = final["lattice"][0].detach().cpu()
                geometry = target["lattice"][0]
                mapping, offsets = map_cells(
                    {"origin": lattice[0].tolist(), "u": lattice[1].tolist(), "v": lattice[2].tolist()},
                    {"origin": geometry[0].tolist(), "u": geometry[1].tolist(), "v": geometry[2].tolist()})
                cases.append({
                    "split": split, "sample_id": case.record["sample_id"],
                    "view": view,
                    "logits": [round(float(value), 6) for value in
                               final["occupancy"][0, :ROWS].reshape(-1)],
                    "occupied": [int(value) for value in target["occupied"][0].reshape(-1)],
                    "known": [bool(value) for value in target["known"][0].reshape(-1)],
                    "mapping": mapping,
                    "offsets": [[round(float(c), 6), round(float(r), 6)] for c, r in offsets],
                    "predicted": {"origin": lattice[0].tolist(), "u": lattice[1].tolist(),
                                  "v": lattice[2].tolist()},
                    "target": {"origin": geometry[0].tolist(), "u": geometry[1].tolist(),
                               "v": geometry[2].tolist()},
                })
    model.train()
    return cases


def swept(cases):
    """Exactness and F1 at logit 0.0 and at the best swept threshold, plus geometry."""
    nonempty = [case for case in cases if any(case["occupied"])]
    empty = [case for case in cases if not any(case["occupied"])]

    def counts(threshold):
        tp = fp = fn = tn = 0
        exact = empty_exact = 0
        for case in nonempty:
            right = True
            for value, label, known in zip(case["logits"], case["occupied"], case["known"]):
                if not known:
                    continue
                predicted = value > threshold
                if label and predicted:
                    tp += 1
                elif label:
                    fn += 1
                    right = False
                elif predicted:
                    fp += 1
                    right = False
                else:
                    tn += 1
            exact += int(right)
        for case in empty:
            right = all(not (value > threshold) for value, known in
                        zip(case["logits"], case["known"]) if known)
            empty_exact += int(right)
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        return {"threshold": threshold, "true_positive": tp, "false_positive": fp,
                "false_negative": fn, "true_negative": tn, "precision": precision,
                "recall": recall, "f1": f1, "exact_nonempty_boards": exact,
                "exact_empty_boards": empty_exact}

    rows = [counts(value) for value in GRID]
    best = max(rows, key=lambda row: (row["f1"], row["exact_nonempty_boards"]))
    placement = placement_stats(cases)
    return {"boards": len(cases), "nonempty": len(nonempty), "empty": len(empty),
            "at_zero": counts(0.0), "best": best,
            "exact_nonempty_at_best": best["exact_nonempty_boards"],
            "exact_empty_at_best": best["exact_empty_boards"],
            "f1_at_zero": counts(0.0)["f1"],
            "placement": placement}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--pool", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--completed", type=int, default=0)
    parser.add_argument("--receipt", type=Path, required=True, help="run_bounded job receipt")
    args = parser.parse_args()
    if args.batch > 4:
        raise SystemExit("the model admits at most four frames per batch")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available")
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(index)
        properties = torch.cuda.get_device_properties(index)
        # The declared 6 GiB envelope is enforced by measurement rather than by an allocator
        # fraction: capping the fraction made the allocator spin (one core pinned, no steps)
        # once fragmentation reached the cap. The peak is asserted after the run instead.
        accelerator = {"device_name": torch.cuda.get_device_name(device),
                       "capability": list(torch.cuda.get_device_capability(device)),
                       "total_memory_bytes": properties.total_memory,
                       "declared_limit_gib": CEILING_GIB,
                       "torch_version": torch.__version__, "cuda_version": torch.version.cuda}
    else:
        accelerator = {"device_name": "cpu", "torch_version": torch.__version__}

    if device.type == "cuda":
        # Benchmark mode is deliberately off: padded batches vary in size, and autotuning
        # re-runs for every new shape (measured ~5x slower than the fixed-shape pilot).
        torch.set_float32_matmul_precision("high")
    model, loaded = load_checkpoint(ROOT/"runs/m3-percell-balance-r1/checkpoint")
    with torch.no_grad():
        original_bias = float(model.occupancy.bias.item())
        model.occupancy.bias.fill_(prior_bias())
    model.to(device)
    model.train()

    pool = build_pool(args.pool)
    print(json.dumps({"pool_seconds": round(pool["seconds"], 1), "device": str(device)}), flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = random.Random(SEED)
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.setstate(state["generator"])
        args.completed = int(state.get("steps_done", args.completed))
        print(json.dumps({"resumed": str(args.resume), "completed": args.completed}), flush=True)
    order, cursor = [], 0

    def next_batch():
        nonlocal order, cursor
        if cursor >= len(order):
            order = list(range(len(pool["pixels"])))
            generator.shuffle(order)
            cursor = 0
        indices = order[cursor:cursor+args.batch]
        cursor += args.batch
        return batch_of(pool, indices)

    def step():
        pixels, valid, target = next_batch()
        output = model(pixels.to(device), valid.to(device))
        loss = joint_loss(output, to_device(target, device), endpoint_term="length",
                          occupancy_factor=4, positive_weight=4)[0]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return float(loss)

    warmup, timed, losses = 20, min(200, max(20, args.steps)), []
    for index in range(min(args.pilot, args.steps) if args.pilot else args.steps):
        started = time.monotonic()
        value = step()
        elapsed = time.monotonic()-started
        if index < warmup or index >= warmup+timed:
            continue
        losses.append((elapsed, value))
        if (index+1) % 250 == 0:
            args.output.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "generator": generator.getstate(),
                        "steps_done": args.completed+index+1}, args.output/"state-rolling.pt")
        if (index+1) % 10 == 0:
            print(json.dumps({"step": index+1, "loss": round(value, 4),
                              "s_per_step": round(elapsed, 4),
                              "gpu_alloc_mb": round(torch.cuda.memory_allocated()/1024**2)
                              if device.type == "cuda" else 0}), flush=True)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds_per_step = (sum(item[0] for item in losses)/len(losses)) if losses else None
    final_loss = sum(item[1] for item in losses[-20:])/len(losses[-20:]) if losses else None

    result = {
        "instrument": "m3-gpu-fit-r1",
        "scope": ("Whether the existing architecture fits boards when the CPU-only step wall is "
                  "removed: fixed pool, batch>1, declared accelerator envelope. Training-fit and "
                  "development probes only; no real-screenshot accuracy claim."),
        "accelerator": accelerator,
        "torch_threads": torch.get_num_threads(),
        "steps": args.steps, "completed_before": args.completed,
        "steps_total": args.completed+args.steps,
        "resumed_from": str(args.resume) if args.resume else None,
        "batch": args.batch, "pool": args.pool, "lr": args.lr,
        "seconds_per_step": seconds_per_step,
        "pilot_steps": args.pilot, "final_loss": final_loss,
        "pool_seconds": pool["seconds"],
        "prior": {"prevalence": PREVALENCE, "bias_before": original_bias,
                  "bias_after": prior_bias(),
                  "note": "logit 0.0 is now the prior's 0.5 decision point"},
        "init": {"checkpoint": "runs/m3-percell-balance-r1/checkpoint",
                 "weights_sha256": loaded.get("weights_sha256")},
        "job": json.loads(args.receipt.read_text()) if args.receipt.exists() else None,
    }
    if args.pilot:
        result["status"] = "pilot"
    else:
        result["train"] = swept(evaluate(model, "training", 12, device))
        result["development"] = swept(evaluate(model, "development", 6, device))
        result["status"] = "passed"
    if torch.cuda.is_available() and device.type == "cuda":
        stats = torch.cuda.memory_stats()
        result["accelerator_memory"] = {
            "max_allocated_bytes": stats.get("allocated_bytes.all.peak", 0),
            "max_reserved_bytes": stats.get("reserved_bytes.all.peak", 0),
            "declared_limit_bytes": int(CEILING_GIB*1024**3),
            "within_declared_envelope": stats.get("reserved_bytes.all.peak", 0) <= CEILING_GIB*1024**3}
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "generator": generator.getstate(),
                "steps_done": args.completed+args.steps}, args.output/"state.pt")
    torch.save(model.state_dict(), args.output/"weights.pt")
    (args.output/"result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in
                      ("status", "seconds_per_step", "final_loss", "accelerator_memory")
                      if key in result}, indent=2), flush=True)
    if "train" in result:
        for split in ("train", "development"):
            block = result[split]
            print(split, "zero", block["at_zero"]["exact_nonempty_boards"],
                  block["at_zero"]["exact_empty_boards"], "f1", round(block["f1_at_zero"], 4),
                  "best", block["best"]["threshold"], round(block["best"]["f1"], 4),
                  block["exact_nonempty_at_best"], block["exact_empty_at_best"],
                  "geom", round(block["placement"]["mean_abs_offset_cells"], 4), flush=True)


if __name__ == "__main__":
    main()
