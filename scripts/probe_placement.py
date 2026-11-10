"""Read-only placement probe for one committed checkpoint.

`docs/M3_OCCUPANCY_GEOMETRY.md` declares this job: 36 forwards, one at a time, 640 MiB, 300 s. It
replays the widened deciding population exactly as the occupancy sweep does, stores each board's
logits, labels and the model-cell -> target-cell offsets, and scores the **full** deciding
population - never the oracle pairing's co-scoreable subset.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

from probe_model import require_bounded_job
from probe_population import (RECORDED_TRAINING_GROUPS, require_training_population,
                              training_extension_specs)
from placement_error import summarize
from record_stream_run import checkpoint_architecture, require_owned_run, validate_completion
from registration_attribution import COLUMNS, ROWS, map_cells
from source_snapshot import verify_source

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 640*1024**2
FORWARDS = 36
SOURCES = tuple("scripts/"+name for name in
                ("probe_placement.py", "placement_error.py", "registration_attribution.py",
                 "occupancy_sweep.py", "gate_power.py", "percell_gate.py",
                 "record_stream_run.py", "probe_population.py", "source_snapshot.py",
                 "probe_model.py"))


def digest(path, maximum=64*1024**2):
    value, consumed = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            consumed += len(block)
            if consumed > maximum:
                raise ValueError("placement source byte budget exceeded")
            value.update(block)
    return value.hexdigest()


def replay_keywords(recipe, capacity):
    from occupancy_sweep import replay_keywords as declared
    return declared(recipe, capacity)


def saved_recipe(recipe, run):
    return (run/"training-probe"/recipe["sample_id"]).with_suffix(".png").exists()


def activation_summary(vectors, scale=.25):
    """Declared activation diagnostic: the engaged magnitude of the per-board refinement.

    Each vector is one board's raw four-vector; the applied correction is the model's own
    `scale*tanh` map, so these are cell offsets and relative axis scale changes, not weights.
    """
    if not vectors:
        return {"available": False}
    import math
    cells = [abs(scale*math.tanh(value)) for row in vectors for value in row[:2]]
    logs = [abs(math.exp(scale*math.tanh(value))-1) for row in vectors for value in row[2:]]
    return {"available": True, "boards": len(vectors),
            "mean_abs_origin_cells": round(sum(cells)/len(cells), 6),
            "mean_abs_log_scale": round(sum(logs)/len(logs), 6),
            "max_abs_origin_cells": round(max(cells), 6),
            "max_abs_log_scale": round(max(logs), 6)}


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
    job = json.loads((ROOT/"runs"/(args.run+"-job.json")).read_text())
    validate_completion(summary, job, allow_partial=True)
    architecture = checkpoint_architecture(summary, manifest)
    if manifest["weights_sha256"] != summary["checkpoint_sha256"] or manifest["config"] != summary["config"]:
        raise ValueError("probed checkpoint identity differs")
    ceiling = json.loads((ROOT/"runs"/(args.output.name+"-job.json")).read_text())["memory_bytes"]
    if ceiling != CEILING_BYTES:
        raise ValueError("probe ceiling differs from the declared 640 MiB")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    code = run/"code"
    verify_source(code, summary["catalogue_sha256"])
    sys.path[:0] = [str(code/"scripts"), str(code)]
    sys.dont_write_bytecode = True
    os.environ.update({"ONEDNN_PRIMITIVE_CACHE_CAPACITY": "16", "DNNL_PRIMITIVE_CACHE_CAPACITY": "16"})
    import torch
    from m2_curriculum import replay_case, sample_for
    from tetris_vision.checkpoint import file_digest, load_checkpoint
    from tetris_vision.losses import targets_for
    from tetris_vision.preprocess import prepare_image
    import tetris_vision.model as model_module
    for function in (targets_for, replay_case, sample_for):
        if not Path(inspect.unwrap(function).__code__.co_filename).resolve().is_relative_to(code):
            raise ValueError("placement probe model/label implementation is not frozen")
    if not Path(model_module.__file__).resolve().is_relative_to(code):
        raise ValueError("placement probe model import is not frozen")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model, loaded = load_checkpoint(run/"checkpoint")
    if loaded["weights_sha256"] != summary["checkpoint_sha256"]:
        raise ValueError("probed weights differ from the recorded run")
    model.eval()
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    capacity = summary["config"].get("component_capacity", 0)
    curriculum = summary.get("curriculum", "stream-v1")
    max_side = summary.get("max_side", 512)
    ledger = run/"training-probe.jsonl"
    if ledger.stat().st_size > 1024**2:
        raise ValueError("probe ledger byte budget exceeded")
    recipes = [json.loads(line) for line in ledger.read_text().splitlines()]
    evaluated = require_training_population(RECORDED_TRAINING_GROUPS, 12)
    if (len(recipes) != RECORDED_TRAINING_GROUPS*3 or
            {(row["group_index"], row["view"]) for row in recipes} !=
            {(group, view) for group in range(RECORDED_TRAINING_GROUPS) for view in range(3)}):
        raise ValueError("recorded training probe population is three complete groups")
    for spec in training_extension_specs(recipes[0]["base_seed"], RECORDED_TRAINING_GROUPS, evaluated):
        case = sample_for(spec["group_index"], spec["view"], "training", spec["base_seed"],
                          curriculum=curriculum, component_capacity=capacity, max_side=max_side)
        recipe = json.loads(json.dumps(case.record))
        if (recipe.get("split") != "training" or
                (recipe["group_index"], recipe["view"], recipe["base_seed"]) !=
                (spec["group_index"], spec["view"], spec["base_seed"])):
            raise ValueError("probe training extension identity differs")
        recipes.append(recipe)
    if len(recipes) != FORWARDS:
        raise ValueError("placement probe requires exactly 36 deciding boards")

    cases, activations = [], []
    for recipe in recipes:
        sample_kwargs, target_kwargs = replay_keywords(recipe, capacity)
        if manifest["config"].get("layer_evidence", False):
            target_kwargs["layer_evidence"] = True
        case = replay_case(recipe, split="training", curriculum=curriculum,
                           component_capacity=capacity, max_side=max_side)
        if json.loads(json.dumps(case.record)) != recipe:
            raise ValueError("probe recipe/pixel/label identity differs")
        if saved_recipe(recipe, run):
            stem = run/"training-probe"/recipe["sample_id"]
            if file_digest(stem.with_suffix(".json")) != recipe["state_sha256"]:
                raise ValueError("saved probe annotation differs")
        prepared = prepare_image(case.frame.image, max_side=max_side)
        target = targets_for(case.frame.state, prepared, **target_kwargs)
        if int(target["known"].sum()) != COLUMNS*ROWS or not bool(target["registered"]):
            raise ValueError("placement probe requires a fully known registered board")
        with torch.no_grad():
            output = model(prepared.pixels[None], prepared.valid[None])
        refinement = output["final"].get("board_refinement")
        if refinement is not None:
            activations.append([float(value) for value in refinement[0]])
        predicted = output["final"]["lattice"][0].detach().cpu()
        geometry = target["lattice"][0]
        predicted = {"origin": predicted[0].tolist(), "u": predicted[1].tolist(),
                     "v": predicted[2].tolist()}
        geometry = {"origin": geometry[0].tolist(), "u": geometry[1].tolist(),
                    "v": geometry[2].tolist()}
        mapping, offsets = map_cells(predicted, geometry)
        cases.append({
            "split": "training", "sample_id": recipe["sample_id"], "group_id": recipe["group_id"],
            "view": recipe["view"],
            "logits": [round(float(value), 6) for value in output["final"]["occupancy"][0, :ROWS].reshape(-1)],
            "occupied": [int(value) for value in target["occupied"][0].reshape(-1)],
            "known": [bool(value) for value in target["known"][0].reshape(-1)],
            "mapping": mapping,
            "offsets": [[round(float(column), 6), round(float(row), 6)] for column, row in offsets],
            "predicted": predicted, "target": geometry,
        })
        (args.output/"progress.json").write_text(json.dumps({"cases": len(cases)})+"\n")
    if not all(torch.equal(value, model.state_dict()[name]) for name, value in weights.items()):
        raise ValueError("read-only placement probe changed model tensors")
    measured = summarize(cases)
    result = {"status": "passed", "instrument": "m3-occupancy-geometry-probe-r1", "run": args.run,
              "weights_sha256": summary["checkpoint_sha256"], "config": manifest["config"],
              "core_architecture": architecture, "forwards": len(cases),
              "job_ceiling_bytes": ceiling, "component_capacity": capacity, "curriculum": curriculum,
              "geometry_final_only": summary.get("arguments", {}).get("geometry_final_only", 0),
              "population": {"split": "training", "deciding_images": len(cases),
                             "nonempty_boards": sum(1 for case in cases if any(case["occupied"])),
                             "occupied_cells": sum(sum(case["occupied"]) for case in cases),
                             "clusters": len({"-".join(case["sample_id"].split("-")[:2])
                                              for case in cases})},
              "threshold": 0.0, "measured": measured, "cases": cases,
              "activation": activation_summary(activations),
              "sources": {name: digest(ROOT/name) for name in SOURCES},
              "elapsed_seconds": time.monotonic()-started,
              "scope": ("Read-only placement and occupancy probe on the widened deciding population: "
                        "one forward per saved board, no training, no weight change, no new label, and "
                        "no real-image or calibrated accuracy claim. The rule was predeclared in "
                        "docs/M3_OCCUPANCY_GEOMETRY.md before this receipt existed.")}
    with (args.output/"result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in ("status", "forwards", "population", "measured")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
