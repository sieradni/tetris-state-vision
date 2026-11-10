"""Price a wider trunk on this machine. PRICING ONLY - no training, no checkpoint, no identity change.

Rebuilds the streaming trainer's exact step (streaming curriculum case -> prepare_image at 512 -> forward ->
joint_loss -> backward -> clip -> Adam step) at the published widths and at two wider declared widths, and
records parameter counts, tensor counts and measured per-step wall time. Nothing here is training evidence: the
models are scratch constructions, are never saved, and the published config defaults are untouched.
"""
import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

def repository_root():
    """Find the repository by its scripts directory, so this file runs from docs/evidence too."""
    for parent in Path(__file__).resolve().parents:
        if (parent/"scripts"/"m2_curriculum.py").exists():
            return parent
    raise SystemExit("cannot locate the repository root")


ROOT = repository_root()
sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(ROOT/"src"))

import torch  # noqa: E402
from m2_curriculum import training_case  # noqa: E402
from tetris_vision.losses import joint_loss, targets_for  # noqa: E402
from tetris_vision.model import ModelConfig, TetrisStateNet  # noqa: E402
from tetris_vision.preprocess import prepare_image  # noqa: E402

SEED = 20261026
CAPTURE = 512
CAPACITY = 64
TIMED_STEPS = 3
WARMUP_STEPS = 1
# Published identity, then two wider trunks inside ModelConfig's own bounds (widths <= 256, dimension %6 == 0).
TRUNKS = (("published", (48, 96, 192, 256), 192),
          ("wide_a", (64, 128, 224, 256), 192),
          ("wide_b", (96, 192, 256, 256), 240))
PUBLISHED_PARAMETERS = 8018505


def config(widths, dimension):
    return ModelConfig(widths=widths, dimension=dimension, heads=6, blocks=6, refinements=2,
                       component_capacity=CAPACITY, component_identity_readout="positional",
                       component_endpoint_term="length", occupancy_detail=1)


def case_step(model, optimizer, step):
    """One trainer-shaped step: fresh streaming case, real 512 input, joint loss, backward, Adam."""
    case = training_case(step, SEED, curriculum="stream-v1", component_capacity=CAPACITY)
    prepared = prepare_image(case.frame.image, max_side=CAPTURE)
    target = targets_for(case.frame.state, prepared, component_capacity=CAPACITY)
    if int(target["known"].sum()) != 200:
        raise ValueError("curriculum admission requires all 200 visible locked cells")
    model.train()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    output = model(prepared.pixels[None], prepared.valid[None])
    loss, _ = joint_loss(output, target, endpoint_term="length", occupancy_factor=4, positive_weight=4)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    elapsed = time.perf_counter() - started
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("nonfinite pricing loss")
    return elapsed, list(prepared.pixels.shape), float(loss.detach())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--memory-mib", type=int, default=1152)
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    records = []
    for name, widths, dimension in TRUNKS:
        if args.only and name != args.only:
            continue
        print(json.dumps({"trunk_start": name, "widths": list(widths), "dimension": dimension}), flush=True)
        model = TetrisStateNet(config(widths, dimension))
        census = {"parameters": sum(p.numel() for p in model.parameters()),
                  "tensors": len(model.state_dict())}
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        timings, shapes, losses = [], None, []
        for step in range(1, WARMUP_STEPS+TIMED_STEPS+1):
            elapsed, shapes, loss = case_step(model, optimizer, step)
            losses.append(loss)
            print(json.dumps({"trunk": name, "step": step, "seconds": round(elapsed, 4)}), flush=True)
            if step > WARMUP_STEPS:
                timings.append(elapsed)
        records.append({"trunk": name, "widths": list(widths), "dimension": dimension,
                        **census, "input_shape": shapes,
                        "timed_steps": len(timings), "per_step_seconds": timings,
                        "median_step_seconds": statistics.median(timings),
                        "losses": losses})
        del model, optimizer
        gc.collect()

    measured_names = {record["trunk"] for record in records}
    published = next((record for record in records if record["trunk"] == "published"), None)
    if published is None:
        published = records[0]
    elif published["parameters"] != PUBLISHED_PARAMETERS:
        raise ValueError(f"the reference census is not the published one: {published['parameters']}")
    for record in records:
        record["parameter_ratio"] = record["parameters"]/published["parameters"]
        record["step_ratio"] = record["median_step_seconds"]/published["median_step_seconds"]

    # Whole-step figures the trainer actually spent at capacity 64 (fit-budget pass receipts).
    measured = {"baseline_256": {"steps": 256, "elapsed_seconds": 1046.2},
                "lever_384": {"steps": 384, "elapsed_seconds": 1522.4}}
    whole_step = {name: value["elapsed_seconds"]/value["steps"] for name, value in measured.items()}
    for record in records:
        record["whole_step_seconds_derived"] = whole_step["baseline_256"]*record["step_ratio"]
        record["steps_in_1800s_derived"] = int(1800//record["whole_step_seconds_derived"])

    result = {"status": "passed", "instrument": "m3-trunk-width-pricing-r1", "purpose": "pricing only",
              "published_identity_untouched": True, "capacity": CAPACITY, "capture_max_side": CAPTURE,
              "seed": SEED, "declared_ceiling_mib": args.memory_mib, "warmup_steps": WARMUP_STEPS, "timed_steps": TIMED_STEPS,
              "trainer_wall_seconds": 1800, "measured_trainer": measured,
              "measured_whole_step_seconds": whole_step, "trunks": records}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"pricing.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    receipt = Path(args.receipt)
    if receipt.exists():
        (args.output/"job.json").write_bytes(receipt.read_bytes())
    print(json.dumps({"trunks": [{key: record[key] for key in
                                  ("trunk", "parameters", "tensors", "median_step_seconds", "step_ratio",
                                   "whole_step_seconds_derived", "steps_in_1800s_derived")}
                                 for record in records]}, indent=2))


if __name__ == "__main__":
    main()
