"""Read-only threshold sweep of a committed run's locked occupancy logits.

No training, no weight changes, no new model: the committed checkpoint is reloaded, one
forward recomputes the occupancy logits for the same saved probe boards, and the predeclared
fixed grid (`scripts/occupancy_sweep.py`, `docs/M3_OCCUPANCY_RANKING.md`) is swept over the
binary decision. The rule that reads this receipt was committed before it existed.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

from occupancy_sweep import evaluate, replay_keywords, sweep_verdict, threshold_grid
from probe_model import require_bounded_job
from probe_population import (RECORDED_TRAINING_GROUPS, require_training_population,
                              training_extension_specs)
from record_stream_run import checkpoint_architecture, require_owned_run, validate_completion
from source_snapshot import verify_source

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 640*1024**2
SWEEP_SOURCES = tuple("scripts/"+name for name in
    ("sweep_stream_occupancy.py", "occupancy_sweep.py", "record_stream_run.py",
     "source_snapshot.py", "probe_model.py"))


def digest(path, maximum=64*1024**2):
    value, consumed = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            consumed += len(block)
            if consumed > maximum:
                raise ValueError("sweep source byte budget exceeded")
            value.update(block)
    return value.hexdigest()


def sweep_logits(occupancy, config):
    """The swept logits of one case: the visible board of the map, sized by the declared config."""
    from board_window import visible_board
    return visible_board(occupancy, config, "occupancy")[0].flatten()


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-groups", type=int, default=None,
                        help="widen the deciding training population to the canonical group set")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--profile", action="store_true")
    mode.add_argument("--admission", type=Path)
    args = parser.parse_args()
    require_owned_run(args.run)
    require_owned_run(args.output.name)
    run = ROOT/"runs"/args.run
    summary = json.loads((run/"summary.json").read_text())
    manifest = json.loads((run/"checkpoint/manifest.json").read_text())
    validate_completion(summary, json.loads((ROOT/"runs"/(args.run+"-job.json")).read_text()),
                        allow_partial=True)
    architecture = checkpoint_architecture(summary, manifest)
    if (manifest["weights_sha256"] != summary["checkpoint_sha256"] or manifest["config"] != summary["config"]):
        raise ValueError("swept checkpoint identity differs")
    ceiling = json.loads((ROOT/"runs"/(args.output.name+"-job.json")).read_text())["memory_bytes"]
    if ceiling != CEILING_BYTES:
        raise ValueError("sweep ceiling differs from the predeclared 640 MiB")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    code = run/"code"
    verify_source(code, summary["catalogue_sha256"])
    # Modules imported above came from the working tree, not from the run's frozen archive; drop
    # them so the frozen-source check below reads the recorded implementation.
    for stale in [name for name in sys.modules
                  if name.split(".")[0] in ("m2_curriculum", "m2_stream_data", "m2_fit_contract",
                                            "m2_interleaved_order", "tetris_vision")]:
        del sys.modules[stale]
    sys.path[:0] = [str(code/"scripts"), str(code)]
    sys.dont_write_bytecode = True
    os.environ.update({"ONEDNN_PRIMITIVE_CACHE_CAPACITY": "16", "DNNL_PRIMITIVE_CACHE_CAPACITY": "16"})
    import torch
    from PIL import Image
    from m2_curriculum import replay_case, sample_for
    from m2_stream_data import sample_case
    from tetris_vision.checkpoint import file_digest, load_checkpoint
    from tetris_vision.losses import targets_for
    from tetris_vision.preprocess import prepare_image
    import tetris_vision.model as model_module
    for function in (targets_for, sample_case, replay_case, sample_for):
        if not Path(inspect.unwrap(function).__code__.co_filename).resolve().is_relative_to(code):
            raise ValueError("sweep model/label implementation is not frozen")
    if not Path(model_module.__file__).resolve().is_relative_to(code):
        raise ValueError("sweep model import is not frozen")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model, loaded = load_checkpoint(run/"checkpoint")
    if loaded["weights_sha256"] != summary["checkpoint_sha256"]:
        raise ValueError("swept weights differ from the recorded run")
    model.eval()
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    capacity = summary["config"].get("component_capacity", 0)
    curriculum = summary.get("curriculum", "stream-v1")
    max_side = summary.get("max_side", 512)
    thresholds = list(threshold_grid())
    recipes = {}
    for split in ("training", "development"):
        ledger = run/(split+"-probe.jsonl")
        if ledger.stat().st_size > 1024**2:
            raise ValueError("probe ledger byte budget exceeded")
        recipes[split] = [json.loads(line) for line in ledger.read_text().splitlines()]
    evaluated_training = (RECORDED_TRAINING_GROUPS if args.training_groups is None
                          else require_training_population(RECORDED_TRAINING_GROUPS, args.training_groups))
    saved_ids = {split: {row["sample_id"] for row in rows} for split, rows in recipes.items()}
    saved_training_cases = len(recipes["training"])
    if evaluated_training != RECORDED_TRAINING_GROUPS:
        if (saved_training_cases != RECORDED_TRAINING_GROUPS*3 or
                {(row["group_index"], row["view"]) for row in recipes["training"]} !=
                {(group, view) for group in range(RECORDED_TRAINING_GROUPS) for view in range(3)}):
            raise ValueError("recorded training probe population is three complete groups")
        # The same widening the error replay applies, from the same declaration, so the two
        # independent executions sweep and replay an identical deciding population.
        for spec in training_extension_specs(recipes["training"][0]["base_seed"],
                                             RECORDED_TRAINING_GROUPS, evaluated_training):
            case = sample_for(spec["group_index"], spec["view"], "training", spec["base_seed"],
                              curriculum=curriculum, component_capacity=capacity, max_side=max_side)
            recipe = json.loads(json.dumps(case.record))
            if (recipe.get("split") != "training" or
                    (recipe["group_index"], recipe["view"], recipe["base_seed"]) !=
                    (spec["group_index"], spec["view"], spec["base_seed"])):
                raise ValueError("sweep training extension identity differs")
            recipes["training"].append(recipe)
    population = {split: len(rows) for split, rows in recipes.items()}

    def sweep_case(recipe, split, *, saved=True):
        sample_kwargs, target_kwargs = replay_keywords(recipe, capacity)
        if manifest["config"].get("layer_evidence", False):
            target_kwargs["layer_evidence"] = True
        case = (replay_case(recipe, split=split, curriculum=curriculum, component_capacity=capacity,
                            max_side=max_side) if "curriculum" in summary else
                sample_case(recipe["group_index"], recipe["view"], split, recipe["base_seed"],
                            max_side=max_side, **sample_kwargs))
        if json.loads(json.dumps(case.record)) != recipe:
            raise ValueError("sweep recipe/pixel/label identity differs")
        if saved:
            stem = run/(split+"-probe")/recipe["sample_id"]
            with Image.open(stem.with_suffix(".png")) as image:
                if (image.mode != case.frame.image.mode or image.size != case.frame.image.size or
                        image.tobytes() != case.frame.image.tobytes()):
                    raise ValueError("saved probe raster differs")
            if file_digest(stem.with_suffix(".json")) != recipe["state_sha256"]:
                raise ValueError("saved probe annotation differs")
        prepared = prepare_image(case.frame.image, max_side=max_side)
        target = targets_for(case.frame.state, prepared, **target_kwargs)
        if int(target["known"].sum()) != 200:
            raise ValueError("saved streaming sweep requires200 supported known cells")
        with torch.no_grad():
            output = model(prepared.pixels[None], prepared.valid[None])
        logits = sweep_logits(output["final"]["occupancy"], manifest["config"])
        known = target["known"][0].flatten().bool()
        occupied = target["occupied"][0].flatten().bool()
        row = {"split": split, "sample_id": recipe["sample_id"], "group_id": recipe["group_id"],
               "view": recipe["view"], "known_cells": int(known.sum()),
               "logits": [round(float(value), 6) for value in logits[known]],
               "occupied": [int(value) for value in occupied[known]]}
        # Declared two-path read (M20): the sum keeps its two parts beside it so a measurement can score each
        # channel alone on the same fold and say whether the sum beats its own parts. Recorded raw (only the
        # visible-board window is applied), because each channel's own transforms are the curator's to apply.
        parts = output["final"].get("occupancy_parts")
        if parts is not None:
            row["logits_parts"] = {name: [round(float(value), 6)
                                          for value in sweep_logits(part, manifest["config"])[known]]
                                   for name, part in sorted(parts.items())}
        return row

    if args.profile:
        profile_cases = [sweep_case(recipes["training"][0], "training"),
                         sweep_case(recipes["development"][0], "development")]
        (args.output/"summary.json").write_text(json.dumps(
            {"status": "complete", "run": args.run, "weights_sha256": summary["checkpoint_sha256"],
             "config": summary["config"], "core_architecture": architecture, "thresholds": thresholds,
             "admission_ceiling_bytes": ceiling, "cases": population,
             "profile_cases": [{"split": row["split"], "sample_id": row["sample_id"],
                                "known_cells": row["known_cells"],
                                "occupied_cells": sum(row["occupied"])} for row in profile_cases],
             "scope": "Resource profile only: two forwards, no sweep receipt, no claim."},
            indent=2)+"\n", encoding="utf-8")
        print(json.dumps({"status": "complete", "profile": True, "cases": population}), flush=True)
        return
    profile = args.admission
    profile_summary = json.loads((profile/"summary.json").read_text())
    profile_job = json.loads((ROOT/"runs"/(profile.name+"-job.json")).read_text())
    if (profile_summary.get("status") != "complete" or profile_job.get("status") != "passed" or
            profile_job.get("exit_code") != 0 or
            profile_summary.get("weights_sha256") != summary["checkpoint_sha256"] or
            profile_summary.get("config") != summary["config"] or
            profile_summary.get("thresholds") != thresholds or
            profile_summary.get("admission_ceiling_bytes") != ceiling or
            profile_summary.get("cases") != population):
        raise ValueError("sweep admission profile differs")
    cases = []
    for split in ("training", "development"):
        for recipe in recipes[split]:
            cases.append(sweep_case(recipe, split, saved=recipe["sample_id"] in saved_ids[split]))
            (args.output/"progress.json").write_text(json.dumps({"cases": len(cases)})+"\n")
    if not all(torch.equal(value, model.state_dict()[name]) for name, value in weights.items()):
        raise ValueError("read-only sweep changed model tensors")
    sources = {name: digest(ROOT/name) for name in SWEEP_SOURCES}
    deciding = evaluate([row for row in cases if row["split"] == "training"], thresholds)
    development = evaluate([row for row in cases if row["split"] == "development"], thresholds)
    result = {"status": "passed", "run": args.run, "run_status": summary["status"],
              "weights_sha256": summary["checkpoint_sha256"], "config": manifest["config"],
              "core_architecture": architecture, "forwards": len(cases),
              "job_ceiling_bytes": ceiling, "component_capacity": capacity, "curriculum": curriculum,
              "training_population": {"recorded_groups": RECORDED_TRAINING_GROUPS,
                                      "evaluated_groups": evaluated_training,
                                      "recorded_cases": saved_training_cases,
                                      "evaluated_cases": len(recipes["training"]),
                                      "extension_cases": len(recipes["training"])-saved_training_cases},
              "admission": {"run": args.admission.name, "weights_sha256": profile_summary["weights_sha256"],
                            "ceiling_bytes": ceiling, "cases": population},
              "grid": {"low": thresholds[0], "high": thresholds[-1],
                       "step": round(thresholds[1]-thresholds[0], 2), "count": len(thresholds)},
              "deciding": deciding, "development": development,
              "verdict": sweep_verdict(deciding), "cases": cases, "sources": sources,
              "elapsed_seconds": time.monotonic()-started,
              "scope": ("Read-only threshold sweep of saved synthetic probe boards from one committed "
                        "checkpoint; no training, no weight change, no decoded complete-state, no "
                        "real-image or calibrated accuracy claim. Unknown cells are excluded; the "
                        "grid and rule were predeclared before this receipt existed.")}
    with (args.output/"result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}), flush=True)


if __name__ == "__main__":
    main()
