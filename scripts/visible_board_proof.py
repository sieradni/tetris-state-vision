"""Recompute a recorded evaluation's loss and metrics under one source tree (V-089).

The development population of an existing arm is replayed through the trainer's own evaluation recipe -
each saved probe case re-rendered, the checkpoint forwarded, `joint_loss` and `diagnostic_metrics` read -
and the aggregate is compared, to the digit, with the numbers the arm published in
`summary["development"]`. `--source current` reads the working tree, `--source archive` reads the arm's
frozen pre-change source, so running both proves the derived board window changed no reported number.

Read-only: no training, no weight change, no new parameter.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from occupancy_sweep import replay_keywords
from probe_model import require_bounded_job
from record_stream_run import require_owned_run, validate_completion
from source_snapshot import verify_source

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 1024*1024**2


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--source", choices=("current", "archive"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_owned_run(args.run)
    ceiling = json.loads((ROOT/"runs"/(args.output.name+"-job.json")).read_text())["memory_bytes"]
    if ceiling != CEILING_BYTES:
        raise ValueError("proof ceiling differs from the declared 1024 MiB")
    args.output.mkdir(parents=True, exist_ok=False)
    run = ROOT/"runs"/args.run
    summary = json.loads((run/"summary.json").read_text())
    manifest = json.loads((run/"checkpoint/manifest.json").read_text())
    validate_completion(summary, json.loads((ROOT/"runs"/(args.run+"-job.json")).read_text()))
    if manifest["weights_sha256"] != summary["checkpoint_sha256"] or manifest["config"] != summary["config"]:
        raise ValueError("checkpoint identity differs")
    started = time.monotonic()
    if args.source == "archive":
        code = run/"code"
        verify_source(code, summary["catalogue_sha256"])
        for stale in [name for name in sys.modules
                      if name.split(".")[0] in ("tetris_vision", "m2_curriculum", "m2_stream_data",
                                                "m2_fit_contract", "m2_interleaved_order")]:
            del sys.modules[stale]
        sys.path[:0] = [str(code/"scripts"), str(code)]
    sys.dont_write_bytecode = True
    import torch
    from PIL import Image
    from m2_curriculum import replay_case
    from tetris_vision.checkpoint import file_digest, load_checkpoint
    from tetris_vision.losses import diagnostic_metrics, joint_loss, targets_for
    from tetris_vision.preprocess import prepare_image

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    config = summary["config"]
    capacity = config.get("component_capacity", 0)
    curriculum = summary.get("curriculum", "stream-v1")
    max_side = summary.get("max_side", 512)
    arguments = summary.get("arguments", {})
    loss_kwargs = {"identity_factor": arguments.get("component_identity_factor", 1),
                   "endpoint_term": arguments.get("component_endpoint_term", "joint"),
                   "occupancy_factor": arguments.get("occupancy_factor", 1),
                   "positive_weight": arguments.get("occupancy_positive_weight", 1),
                   "geometry_final_only": arguments.get("geometry_final_only", 0),
                   "box_extent_weight": arguments.get("box_extent_weight", 0.0),
                   "support_anchor_weight": arguments.get("support_anchor_weight", 0.0),
                   "occupancy_placed": arguments.get("occupancy_placed", 0)}
    recipes = [json.loads(line) for line in (run/"development-probe.jsonl").read_text().splitlines()]
    model, loaded = load_checkpoint(run/"checkpoint")
    if loaded["weights_sha256"] != summary["checkpoint_sha256"]:
        raise ValueError("proof weights differ from the recorded run")
    model.eval()
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    per_case, forwards = [], 0
    for recipe in recipes:
        sample_kwargs, target_kwargs = replay_keywords(recipe, capacity)
        case = replay_case(recipe, split="development", curriculum=curriculum,
                           component_capacity=capacity, max_side=max_side)
        if json.loads(json.dumps(case.record)) != recipe:
            raise ValueError("proof recipe/pixel/label identity differs")
        stem = run/"development-probe"/recipe["sample_id"]
        with Image.open(stem.with_suffix(".png")) as image:
            if (image.mode != case.frame.image.mode or image.size != case.frame.image.size or
                    image.tobytes() != case.frame.image.tobytes()):
                raise ValueError("saved probe raster differs")
        if file_digest(stem.with_suffix(".json")) != recipe["state_sha256"]:
            raise ValueError("saved probe annotation differs")
        prepared = prepare_image(case.frame.image, max_side=max_side)
        target = targets_for(case.frame.state, prepared, **target_kwargs)
        with torch.no_grad():
            output = model(prepared.pixels[None], prepared.valid[None])
        loss, _ = joint_loss(output, target, **loss_kwargs)
        forwards += 1
        criteria = diagnostic_metrics(output, target)
        per_case.append({"sample_id": recipe["sample_id"], "loss": float(loss),
                         "cell_accuracy": criteria["cell_accuracy"],
                         "exact_locked_board": criteria["exact_locked_board"],
                         "mean_geometry_error_cells": criteria["mean_geometry_error_cells"]})
    for key, value in weights.items():
        if not torch.equal(value, model.state_dict()[key]):
            raise ValueError("read-only proof changed model tensors")
    aggregate = {"images": len(per_case),
                 "ancestry_groups": len({row["group_id"] for row in recipes}),
                 "loss": sum(row["loss"] for row in per_case)/len(per_case),
                 "exact_locked_boards": sum(row["exact_locked_board"] for row in per_case),
                 "cell_accuracy": sum(row["cell_accuracy"] for row in per_case)/len(per_case),
                 "mean_geometry_error_cells": sum(row["mean_geometry_error_cells"] for row in per_case)/len(per_case)}
    published = summary["development"]
    differences = {}
    for key in ("images", "ancestry_groups", "loss", "exact_locked_boards", "cell_accuracy",
                "mean_geometry_error_cells"):
        differences[key] = aggregate[key]-published[key] if isinstance(aggregate[key], float) else (
            0 if aggregate[key] == published[key] else None)
    published_cases = {row["sample_id"]: row for row in published["cases"]}
    for row in per_case:
        reference = published_cases[row["sample_id"]]
        for key in ("loss", "cell_accuracy", "mean_geometry_error_cells"):
            if row[key] != reference[key]:
                differences["%s:%s" % (row["sample_id"], key)] = row[key]-reference[key]
        if row["exact_locked_board"] != reference["exact_locked_board"]:
            differences["%s:exact_locked_board" % row["sample_id"]] = row["exact_locked_board"]
    if any(value != 0 for value in differences.values()):
        raise ValueError("proof does not reproduce the published evaluation: %s" % differences)
    result = {"status": "passed", "source": args.source, "run": args.run,
              "weights_sha256": summary["checkpoint_sha256"], "config": config,
              "forwards": forwards, "job_ceiling_bytes": ceiling,
              "aggregate": aggregate, "published": {key: published[key] for key in aggregate},
              "differences": differences, "cases": per_case,
              "elapsed_seconds": time.monotonic()-started,
              "scope": ("Reproduces the arm's published development loss and metrics from its saved raster "
                        "population, to the digit, under one source tree. No training, no weight change, no "
                        "complete-state or real-image claim.")}
    with (args.output/"result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": "passed", "source": args.source, "aggregate": aggregate,
                      "elapsed_seconds": round(result["elapsed_seconds"], 1)}, default=str), flush=True)


if __name__ == "__main__":
    main()
