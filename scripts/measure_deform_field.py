"""Measure the applied registration offset field of a committed run, in cells.

Read-only: the committed checkpoint is reloaded from its own frozen archive, one forward per saved
probe board recomputes the offsets the detail tap actually applied, and the field is summarised in
cell units so it can be set beside the placement residual the campaign measured. No training, no
weight change, no new claim, no decoded state.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from probe_model import require_bounded_job
from probe_population import (RECORDED_TRAINING_GROUPS, require_training_population,
                              training_extension_specs)
from record_stream_run import checkpoint_architecture, require_owned_run, validate_completion
from source_snapshot import verify_source

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 640*1024**2
DECIDING_IMAGES = 36
DECIDING_GROUPS = 12
OUTSIDE_CELL = .5
AT_BOUND = .95
REFERENCE_MEAN_CELLS = 2.57
REFERENCE_WRONG_CELL_FRACTION = .483


def digest(path, maximum=64*1024**2):
    value, consumed = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            consumed += len(block)
            if consumed > maximum:
                raise ValueError("field-measurement source byte budget exceeded")
            value.update(block)
    return value.hexdigest()


def percentile(values, fraction):
    if not values or not 0 <= fraction <= 1:
        raise ValueError("percentile of a nonempty list and a fraction in [0, 1] required")
    ordered = sorted(values)
    return ordered[min(len(ordered)-1, max(0, int(round(fraction*(len(ordered)-1)))))]


def summary_of(values):
    if not values:
        raise ValueError("summary of an empty field")
    return {"mean": round(sum(values)/len(values), 6), "p50": round(percentile(values, .5), 6),
            "p90": round(percentile(values, .9), 6), "p99": round(percentile(values, .99), 6),
            "max": round(max(values), 6), "cells": len(values)}


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_owned_run(args.run)
    require_owned_run(args.output.name)
    run = ROOT/"runs"/args.run
    summary = json.loads((run/"summary.json").read_text())
    manifest = json.loads((run/"checkpoint/manifest.json").read_text())
    validate_completion(summary, json.loads((ROOT/"runs"/(args.run+"-job.json")).read_text()),
                        allow_partial=True)
    architecture = checkpoint_architecture(summary, manifest)
    if (manifest["weights_sha256"] != summary["checkpoint_sha256"] or
            manifest["config"] != summary["config"]):
        raise ValueError("measured checkpoint identity differs")
    config = manifest["config"]
    if config.get("occupancy_detail") != 1 or config.get("occupancy_deform") not in (1, 2):
        raise ValueError("this instrument measures a declared deformation arm with the detail tap")
    job = json.loads((ROOT/"runs"/(args.output.name+"-job.json")).read_text())
    if job["memory_bytes"] != CEILING_BYTES:
        raise ValueError("field-measurement ceiling differs from the predeclared 640 MiB")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    code = run/"code"
    verify_source(code, summary["catalogue_sha256"])
    # Modules imported above came from the working tree, not the run's archive; drop them so what
    # follows reads the recorded implementation, exactly as the sweep does.
    for stale in [name for name in sys.modules
                  if name.split(".")[0] in ("m2_curriculum", "m2_stream_data", "m2_fit_contract",
                                            "m2_interleaved_order", "tetris_vision")]:
        del sys.modules[stale]
    sys.path[:0] = [str(code/"scripts"), str(code)]
    sys.dont_write_bytecode = True
    os.environ.update({"ONEDNN_PRIMITIVE_CACHE_CAPACITY": "16", "DNNL_PRIMITIVE_CACHE_CAPACITY": "16"})
    import torch
    from board_window import visible_board
    from m2_curriculum import replay_case, sample_for
    from m2_stream_data import sample_case
    from occupancy_sweep import replay_keywords
    from tetris_vision import model as model_module
    from tetris_vision.checkpoint import load_checkpoint
    from tetris_vision.losses import targets_for
    from tetris_vision.preprocess import prepare_image
    if not Path(model_module.__file__).resolve().is_relative_to(code):
        raise ValueError("field-measurement model import is not frozen")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model, loaded = load_checkpoint(run/"checkpoint")
    if loaded["weights_sha256"] != summary["checkpoint_sha256"]:
        raise ValueError("measured weights differ from the recorded run")
    model.eval()
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    capacity = config.get("component_capacity", 0)
    curriculum = summary.get("curriculum", "stream-v1")
    max_side = summary.get("max_side", 512)
    recipes = [json.loads(line) for line in (run/"training-probe.jsonl").read_text().splitlines()]
    evaluated = require_training_population(RECORDED_TRAINING_GROUPS, DECIDING_GROUPS)
    for spec in training_extension_specs(recipes[0]["base_seed"], RECORDED_TRAINING_GROUPS,
                                         evaluated):
        case = sample_for(spec["group_index"], spec["view"], "training", spec["base_seed"],
                          curriculum=curriculum, component_capacity=capacity, max_side=max_side)
        recipe = json.loads(json.dumps(case.record))
        if (recipe.get("split") != "training" or
                (recipe["group_index"], recipe["view"], recipe["base_seed"]) !=
                (spec["group_index"], spec["view"], spec["base_seed"])):
            raise ValueError("field-measurement population identity differs")
        recipes.append(recipe)
    if len(recipes) != DECIDING_IMAGES:
        raise ValueError("field measurement requires exactly the 36 deciding images")
    rows = config["visible_rows"]+config["spawn_rows"]
    columns = config["columns"]
    bound = float(model_module.DEFORM_REACH if config["occupancy_deform"] == 2
                  else model_module.DEFORM_LIMIT)
    magnitudes, norms, axes = [], [], ([], [])
    signed = ([], [])
    occupied_magnitudes, occupied_norms = [], []
    cases, known_total, occupied_total = [], 0, 0
    for recipe in recipes:
        sample_kwargs, target_kwargs = replay_keywords(recipe, capacity)
        if config.get("layer_evidence", False):
            target_kwargs["layer_evidence"] = True
        case = (replay_case(recipe, split="training", curriculum=curriculum,
                            component_capacity=capacity, max_side=max_side)
                if "curriculum" in summary else
                sample_case(recipe["group_index"], recipe["view"], "training", recipe["base_seed"],
                            max_side=max_side, **sample_kwargs))
        prepared = prepare_image(case.frame.image, max_side=max_side)
        target = targets_for(case.frame.state, prepared, **target_kwargs)
        with torch.no_grad():
            output = model(prepared.pixels[None], prepared.valid[None])
        field = output["final"]["occupancy_deform_offsets"].reshape(1, rows, columns, 2)
        offsets = visible_board(field, config, "occupancy deform offsets")[0].reshape(-1, 2)
        known = target["known"][0].flatten().bool()
        occupied = target["occupied"][0].flatten().bool()
        if int(known.sum()) != 200 or offsets.shape[0] != known.numel():
            raise ValueError("field measurement cannot cover the labelled visible board")
        magnitudes_case = offsets.abs().amax(1)
        norms_case = offsets.norm(dim=1)
        for index in range(len(magnitudes_case)):
            if not bool(known[index]):
                continue
            magnitudes.append(float(magnitudes_case[index]))
            norms.append(float(norms_case[index]))
            axes[0].append(abs(float(offsets[index, 0])))
            axes[1].append(abs(float(offsets[index, 1])))
            signed[0].append(float(offsets[index, 0]))
            signed[1].append(float(offsets[index, 1]))
            if bool(occupied[index]):
                occupied_magnitudes.append(float(magnitudes_case[index]))
                occupied_norms.append(float(norms_case[index]))
        known_total += int(known.sum())
        occupied_total += int(occupied.sum())
        cases.append({"sample_id": recipe["sample_id"], "group_id": recipe["group_id"],
                      "known_cells": int(known.sum()), "occupied_cells": int(occupied.sum()),
                      "mean_magnitude": round(float(magnitudes_case[known].mean()), 6),
                      "max_magnitude": round(float(magnitudes_case.max()), 6),
                      "outside_own_cell": round(float((magnitudes_case[known] > OUTSIDE_CELL).float().mean()), 6)})
    if not all(torch.equal(value, model.state_dict()[name]) for name, value in weights.items()):
        raise ValueError("read-only field measurement changed model tensors")
    if (known_total, occupied_total) != (DECIDING_IMAGES*200, 667):
        raise ValueError("field measurement decided a different population from the sweep")
    at_bound = sum(1 for value in magnitudes if value >= AT_BOUND*bound)/len(magnitudes)
    mean_dx, mean_dy = sum(signed[0])/len(signed[0]), sum(signed[1])/len(signed[1])

    def spread(values, mean):
        return (sum((value-mean)**2 for value in values)/len(values))**.5

    # Share of the displaced energy that a single constant vector already accounts for: the squared
    # length of the mean offset over the mean squared length of the offsets. A field that is really a
    # lattice translation scores near one however large its per-cell variation is, and a genuinely
    # per-cell field scores near zero because its mean vector is short.
    mean_square = sum(value*value for value in signed[0])/len(signed[0]) + \
        sum(value*value for value in signed[1])/len(signed[1])
    constant_share = ((mean_dx**2+mean_dy**2)/mean_square) if mean_square else 1.0
    mean_magnitude = sum(magnitudes)/len(magnitudes)
    result = {
        "status": "passed", "run": args.run, "weights_sha256": summary["checkpoint_sha256"],
        "config": config, "core_architecture": architecture, "forwards": len(cases),
        "job_ceiling_bytes": job["memory_bytes"],
        "arm": {"occupancy_deform": config["occupancy_deform"], "bound_cells": bound,
                "fine_limit_cells": float(model_module.DEFORM_LIMIT),
                "coarse_limit_cells": (float(model_module.DEFORM_WIDE_LIMIT)
                                       if config["occupancy_deform"] == 2 else 0.0),
                "deform_reach_cells": float(model_module.DEFORM_REACH)},
        "field": {
            "note": ("Offsets are the ones the tap applied, in cell units, L-infinity per cell. The "
                     "fraction outside its own cell is the sampling-side analogue of the oracle's "
                     "wrong-target-cell coverage; a field working correctly moves samples off their "
                     "nominal cell when the lattice is misplaced."),
            "magnitude_cells": summary_of(magnitudes),
            "norm_cells": summary_of(norms),
            "axis_mean_abs_cells": [round(sum(axes[0])/len(axes[0]), 6),
                                    round(sum(axes[1])/len(axes[1]), 6)],
            "mean_vector_cells": [round(mean_dx, 6), round(mean_dy, 6)],
            "spread_cells": [round(spread(signed[0], mean_dx), 6),
                             round(spread(signed[1], mean_dy), 6)],
            "constant_share": round(constant_share, 6),
            "coherence": round(((mean_dx**2+mean_dy**2)**.5)/mean_magnitude, 6),
            "at_bound_fraction": round(at_bound, 6),
            "outside_own_cell_fraction": round(
                sum(1 for value in magnitudes if value > OUTSIDE_CELL)/len(magnitudes), 6),
            "occupied_magnitude_cells": summary_of(occupied_magnitudes),
            "occupied_norm_cells": summary_of(occupied_norms),
            "occupied_outside_own_cell_fraction": round(
                sum(1 for value in occupied_magnitudes if value > OUTSIDE_CELL)
                /len(occupied_magnitudes), 6)},
        "reference": {"mean_placement_offset_cells": REFERENCE_MEAN_CELLS,
                      "wrong_target_cell_fraction": REFERENCE_WRONG_CELL_FRACTION,
                      "source": "M3_OCCUPANCY_SCALED.md, oracle registration at 4x budget"},
        "population": {"split": "training", "images": len(cases), "known_cells": known_total,
                       "occupied_cells": occupied_total, "groups": DECIDING_GROUPS},
        "cases": cases,
        "sources": {name: digest(ROOT/name) for name in
                    ("scripts/measure_deform_field.py", "scripts/board_window.py")},
        "elapsed_seconds": time.monotonic()-started,
        "scope": ("Read-only measurement of the registration offsets one committed checkpoint applied; "
                  "no training, no weight change, no decoded state, no accuracy claim."),
    }
    with (args.output/"result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": "passed", "field": result["field"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
