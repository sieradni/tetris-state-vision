"""Finite CPU learning pilot. Always invoke through scripts/run_bounded.py.

This overfit/development diagnostic cannot establish complete-state or real-image
accuracy. It uses no external weights, detector, oracle crop or solver at inference.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from probe_model import require_bounded_job


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--max-side", type=int, default=384)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--wall-seconds", type=float, default=90)
    parser.add_argument("--evaluate-every", type=int, default=16)
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 512 or not 1 <= args.states <= 8:
        parser.error("optimizer steps must be 1..512 and states 1..8")
    if not 128 <= args.max_side <= 640 or args.threads not in (1, 2):
        parser.error("input max side 128..640 and threads 1..2 required")
    if not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= .01:
        parser.error("learning rate must be finite and in (0, .01]")
    if not math.isfinite(args.wall_seconds) or not 0 < args.wall_seconds <= 1800:
        parser.error("inner wall budget must be finite and in (0, 1800]")
    if not 1 <= args.evaluate_every <= 128:
        parser.error("evaluation interval must be 1..128")
    return args


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    args = arguments(argv)
    ceiling = require_bounded_job()  # No heavy import before kernel admission.
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    progress = {"status": "preparing", "step": 0, "requested_steps": args.steps,
                "job_ceiling_bytes": ceiling, "pid": os.getpid(), "threads": args.threads}
    atomic_json(args.output/"progress.json", progress)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import torch
    from PIL import Image, ImageDraw
    from tetris_vision.checkpoint import file_digest, load_checkpoint, save_checkpoint
    from tetris_vision.dataset import source_digest
    from tetris_vision.layout import Layout
    from tetris_vision.losses import diagnostic_metrics, joint_loss, targets_for
    from tetris_vision.model import ModelConfig, TetrisStateNet
    from tetris_vision.preprocess import prepare_image
    from tetris_vision.render import Appearance, render, sample_scene

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    source = source_digest()
    manifest, tensor_bytes = [], 0

    def cases(seed, states, split):
        nonlocal tensor_bytes
        result = []
        directory = args.output/split
        directory.mkdir()
        for index in range(states):
            sample_seed = seed+index
            rng = random.Random(sample_seed)
            density = ("dense", "sparse", "empty")[index % 3]
            scene = sample_scene(sample_seed, density, varied=True)
            k, left, bottom = rng.randint(20, 22), rng.randint(60, 120), rng.randint(580, 615)
            layout = Layout(canvas=(480, 640), board_left=left, board_bottom=bottom,
                            cell_size=k, queue_origin=(left+10*k+20, 40), hold_origin=(8, 40))
            for skin in ("clean", "no_grid"):
                style = Appearance(seed=sample_seed, grid=skin == "clean", border=skin == "clean")
                frame = render(scene, style, layout)
                prepared = prepare_image(frame.image, max_side=args.max_side)
                target = targets_for(frame.state, prepared)
                if int(target["known"].sum()) != 200:
                    raise ValueError("pilot admission requires all 200 canonical cells readable; raise source detail")
                tensor_bytes += prepared.pixels.numel()*prepared.pixels.element_size()+prepared.valid.numel()
                if tensor_bytes > 64*1024**2:
                    raise ValueError("64 MiB tensor-cache budget exceeded")
                sample_id = f"{sample_seed}-{density}-{skin}"
                image_path, state_path = directory/f"{sample_id}.png", directory/f"{sample_id}.json"
                frame.image.save(image_path)
                state_path.write_text(frame.state.to_json(), encoding="utf-8")
                manifest.append({"sample_id": sample_id, "group_id": f"{split}-{sample_seed}", "split": split,
                                 "density": density, "skin": skin, "seed": sample_seed,
                                 "layout": asdict(layout), "appearance": asdict(style),
                                 "image": str(image_path.relative_to(args.output)), "image_sha256": file_digest(image_path),
                                 "state": str(state_path.relative_to(args.output)), "state_sha256": file_digest(state_path)})
                result.append({"id": sample_id, "image": frame.image, "state": frame.state,
                               "prepared": prepared, "target": target, "skin": skin})
        return result

    train = cases(args.seed, args.states, "training")
    development = cases(args.seed+100000, 3, "development")
    (args.output/"manifest.jsonl").write_text("".join(json.dumps(item)+"\n" for item in manifest), encoding="utf-8")
    model = TetrisStateNet(ModelConfig.pilot())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)

    @torch.no_grad()
    def evaluate(samples):
        model.eval()
        loss_sum = geometry_sum = cell_sum = exact = layer_right = layer_total = 0
        terms_sum = {}
        for case in samples:
            prepared, target = case["prepared"], case["target"]
            output = model(prepared.pixels[None], prepared.valid[None])
            loss, terms = joint_loss(output, target)
            loss_sum += float(loss)
            for key, value in terms.items():
                terms_sum[key] = terms_sum.get(key, 0)+float(value)
            metrics = diagnostic_metrics(output, target)
            geometry_sum += metrics["mean_geometry_error_cells"]
            cell_sum += metrics["cell_accuracy"]
            exact += int(metrics["exact_locked_board"])
            final = output["final"]
            for index in range(2):
                if not bool(target["layer_known"][0, index]):
                    continue
                layer_total += 1
                right = (int(final["layer_type"][0, index].argmax()) == int(target["layer_type"][0, index])
                         and bool(final["layer_known"][0, index] > 0))
                if bool(target["pose_known"][0, index]):
                    right = (right and int(final["pose_column"][0, index].argmax()) == int(target["column"][0, index])
                             and int(final["pose_row"][0, index].argmax()) == int(target["row"][0, index])
                             and bool(target["rotations"][0, index, final["pose_rotation"][0, index].argmax()]))
                layer_right += int(right)
        return {"images": len(samples), "loss": loss_sum/len(samples),
                "terms": {key: value/len(samples) for key, value in terms_sum.items()},
                "cell_accuracy": cell_sum/len(samples), "exact_locked_boards": exact,
                "mean_geometry_error_cells": geometry_sum/len(samples),
                "exact_known_layers": layer_right, "known_layers": layer_total,
                "note": "Uncalibrated raw-head synthetic diagnostics; not automatic full-state accuracy."}

    initial = evaluate(train)
    evaluations = [{"step": 0, "training": initial}]
    completed = 0
    with (args.output/"curve.jsonl").open("x", encoding="utf-8") as curve:
        for step in range(1, args.steps+1):
            if time.monotonic()-started >= args.wall_seconds:
                break
            model.train()
            case = train[(step-1) % len(train)]
            prepared = case["prepared"]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(prepared.pixels[None], prepared.valid[None])
            loss, terms = joint_loss(outputs, case["target"])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            completed = step
            item = {"step": step, "sample_id": case["id"], "loss": float(loss.detach()),
                    "terms": {key: float(value.detach()) for key, value in terms.items()},
                    "unclipped_gradient_norm": float(norm), "elapsed_seconds": time.monotonic()-started}
            curve.write(json.dumps(item, allow_nan=False)+"\n")
            curve.flush()
            progress.update(status="training", step=step, elapsed_seconds=item["elapsed_seconds"], loss=item["loss"])
            atomic_json(args.output/"progress.json", progress)
            print(json.dumps(item, allow_nan=False), flush=True)
            if step % args.evaluate_every == 0:
                evaluations.append({"step": step, "training": evaluate(train)})
    final, held_out = evaluate(train), evaluate(development)
    if source_digest() != source:
        raise RuntimeError("runtime source changed during training; refusing unscoped checkpoint")
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata = {"trained_steps": completed, "runtime_source_sha256": source,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "arguments": configuration,
                "manifest_sha256": file_digest(args.output/"manifest.jsonl"), "cpu_only": torch.version.cuda is None}
    checkpoint = save_checkpoint(args.output/"checkpoint", model.eval(), metadata)
    loaded, record = load_checkpoint(args.output/"checkpoint")
    example = train[0]["prepared"]
    with torch.no_grad():
        expected = model(example.pixels[None], example.valid[None])["final"]["occupancy"]
        actual = loaded.eval()(example.pixels[None], example.valid[None])["final"]["occupancy"]
    if not torch.equal(expected, actual):
        raise RuntimeError("saved artifact changes predictions")
    overlays = []
    for case in (train[0], development[0]):
        prepared = case["prepared"]
        with torch.no_grad():
            output = model(prepared.pixels[None], prepared.valid[None])["final"]
        overlay = case["image"].copy()
        draw = ImageDraw.Draw(overlay)
        vectors = output["lattice"][0].detach().tolist()
        factors = (prepared.padded_size[0]/prepared.scale_x, prepared.padded_size[1]/prepared.scale_y)
        origin, u, v = [[point[i]*factors[i] for i in range(2)] for point in vectors]
        def point(c, r):
            return (origin[0]+c*u[0]+r*v[0], origin[1]+c*u[1]+r*v[1])
        for c in range(11):
            draw.line((*point(c, 0), *point(c, 20)), fill=(245, 118, 110), width=1)
        for r in range(21):
            draw.line((*point(0, r), *point(10, r)), fill=(245, 118, 110), width=1)
        draw.text((8, 620), "M2 uncalibrated predicted lattice / "+case["id"], fill=(240, 241, 245))
        overlays.append(overlay)
    preview = Image.new("RGB", (960, 640))
    for index, overlay in enumerate(overlays):
        preview.paste(overlay, (480*index, 0))
    preview.save(args.output/"lattice-preview.png")
    summary = {"status": "complete" if completed == args.steps else "wall_partial", "steps": completed,
               "requested_steps": args.steps, "initial_training": initial, "final_training": final,
               "development": held_out, "evaluations": evaluations, "tensor_cache_bytes": tensor_bytes,
               "max_tensor_cache_bytes": 64*1024**2, "threads": args.threads,
               "max_worker_processes": 0, "max_prefetch": 0, "cpu_only": torch.version.cuda is None,
               "elapsed_seconds": time.monotonic()-started, "checkpoint_sha256": checkpoint["weights_sha256"],
               "checkpoint_round_trip_exact": True, **metadata,
               "claim": "Bounded scratch-learning diagnostic; no complete-state or real-image acceptance."}
    atomic_json(args.output/"summary.json", summary)
    progress.update(status=summary["status"], step=completed, summary="summary.json", checkpoint="checkpoint/manifest.json")
    atomic_json(args.output/"progress.json", progress)
    print(json.dumps(summary, allow_nan=False), flush=True)
    return 0 if completed == args.steps else 2


if __name__ == "__main__":
    raise SystemExit(main())
