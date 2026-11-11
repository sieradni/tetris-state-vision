"""Read the declared trunk-width lever: one knob (decision-path dimension) at a matched fresh start.

The two committed arms are the fresh-baseline pair predeclared in docs/M3_TRUNK_WIDTH_LEVER.md:
identical seed, catalogue, curriculum, budget and every other argument, differing only in
`dimension`. This pass reloads both checkpoints, replays the saved deciding population, and
answers the declared verdict rule. Read-only: no training, no weight change, no new parameter.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

from occupancy_sweep import evaluate, replay_keywords, threshold_grid
from probe_model import require_bounded_job
from probe_population import (RECORDED_TRAINING_GROUPS, require_training_population,
                              training_extension_specs)
from record_stream_run import checkpoint_architecture, require_owned_run, validate_completion
from source_snapshot import verify_source

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 1024*1024**2
F1_BAR, AUC_BAR, ATTRIBUTION_FLOOR = 0.70, 0.90, 0.05
ACTIVATION_FLOOR, CONTROL_FLOOR, GATE_BOARDS = 0.05, 0.70, 3
READ_SOURCES = tuple("scripts/"+name for name in
    ("width_lever_read.py", "occupancy_sweep.py", "record_stream_run.py", "source_snapshot.py",
     "probe_model.py", "probe_population.py"))


def digest(path, maximum=64*1024**2):
    value, consumed = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            consumed += len(block)
            if consumed > maximum:
                raise ValueError("lever source byte budget exceeded")
            value.update(block)
    return value.hexdigest()


def f1_at(summary, threshold=0.0):
    for row in summary["rows"]:
        if row["threshold"] == threshold:
            return row["f1"]
    raise ValueError("declared threshold missing from the swept grid")


def visible_board_logits(occupancy, config):
    """The visible-board logits, sliced by the declared board rather than a literal.

    `final["occupancy"]` is `(batch, visible_rows+spawn_rows, columns)`: one channel of logits per
    board cell plus the spawn margin. The spawn rows carry no target, so the visible read is the first
    `visible_rows` rows, and sizing that from the verified config is what stops a structural change
    from silently truncating the read.
    """
    rows = int(config["visible_rows"])+int(config["spawn_rows"])
    columns = int(config["columns"])
    if tuple(occupancy.shape) != (1, rows, columns):
        raise ValueError("occupancy logits must be (1, %d, %d), got %s" % (rows, columns, tuple(occupancy.shape)))
    return occupancy[0, :int(config["visible_rows"])].flatten()


def readout_statistics(occupancy, config):
    """Shape and per-block logit statistics of the map the trunk produces, for the record."""
    visible = int(config["visible_rows"])
    rows = occupancy.detach()[0]
    cells, spawn = rows[:visible], rows[visible:]
    return {"shape": list(occupancy.shape), "visible_rows": visible,
            "spawn_rows": int(config["spawn_rows"]), "columns": int(config["columns"]),
            "visible_mean_abs": round(float(cells.abs().mean()), 6),
            "visible_std": round(float(cells.std(unbiased=False)), 6),
            "spawn_mean_abs": round(float(spawn.abs().mean()), 6) if spawn.numel() else 0.0}


def lever_verdict(control, lever, *, flips, known_cells, auc_bar=AUC_BAR, f1_bar=F1_BAR,
                  floor=ATTRIBUTION_FLOOR, activation=ACTIVATION_FLOOR, control_floor=CONTROL_FLOOR,
                  gate_boards=GATE_BOARDS):
    """The predeclared rule of docs/M3_TRUNK_WIDTH_LEVER.md; no unstated branch."""
    for summary in (control, lever):
        if (type(summary) is not dict or not isinstance(summary.get("auc"), (int, float)) or
                type(summary.get("best")) is not dict or
                not isinstance(summary["best"].get("f1"), (int, float)) or
                type(summary.get("rows")) is not list):
            raise ValueError("swept arm summary required")
    if type(flips) is not int or type(known_cells) is not int or known_cells <= 0 or not 0 <= flips <= known_cells:
        raise ValueError("decision-flip count required over a positive known population")
    attribution = lever["best"]["f1"]-control["best"]["f1"]
    fraction = flips/known_cells
    opened = lever["exact_boards_at_best"]["nonempty"] >= gate_boards
    if fraction < activation:
        label = "inactive"
    elif control["auc"] < control_floor:
        label = "floor_limited"
    elif attribution >= floor and (opened or lever["auc"] >= auc_bar):
        label = "capacity_helps"
    elif attribution >= floor:
        label = "capacity_fits_not_decides"
    else:
        label = "capacity_null"
    return {"label": label, "attribution": attribution,
            "control_best_f1": control["best"]["f1"], "lever_best_f1": lever["best"]["f1"],
            "control_auc": control["auc"], "lever_auc": lever["auc"],
            "f1_at_zero_delta": f1_at(lever)-f1_at(control),
            "flip_fraction": fraction, "activation_floor": activation,
            "control_floor": control_floor, "fit_gate_boards": gate_boards,
            "fit_gate_opened": opened,
            "control_exact_nonempty": control["exact_boards_at_best"]["nonempty"],
            "lever_exact_nonempty": lever["exact_boards_at_best"]["nonempty"],
            "auc_bar": auc_bar, "f1_bar": f1_bar, "attribution_floor": floor}


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs=2, required=True, metavar=("CONTROL", "LEVER"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-groups", type=int, required=True,
                        help="the declared deciding population size; never defaulted (V-087)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--profile", action="store_true")
    mode.add_argument("--admission", type=Path)
    args = parser.parse_args()
    for name in args.arms:
        require_owned_run(name)
    ceiling = json.loads((ROOT/"runs"/(args.output.name+"-job.json")).read_text())["memory_bytes"]
    if ceiling != CEILING_BYTES:
        raise ValueError("lever read ceiling differs from the predeclared 1024 MiB")
    args.output.mkdir(parents=True, exist_ok=False)
    runs, summaries, manifests = {}, {}, {}
    for name in args.arms:
        run = ROOT/"runs"/name
        summary = json.loads((run/"summary.json").read_text())
        manifest = json.loads((run/"checkpoint/manifest.json").read_text())
        validate_completion(summary, json.loads((ROOT/"runs"/(name+"-job.json")).read_text()))
        if (manifest["weights_sha256"] != summary["checkpoint_sha256"] or
                manifest["config"] != summary["config"]):
            raise ValueError("lever checkpoint identity differs")
        runs[name], summaries[name], manifests[name] = run, summary, manifest
    control, lever = args.arms
    if summaries[control]["catalogue_sha256"] != summaries[lever]["catalogue_sha256"]:
        raise ValueError("lever arms were not trained from one frozen catalogue")
    settings = dict(summaries[control]["config"])
    widened = dict(summaries[lever]["config"])
    dimensions = {settings.pop("dimension", None), widened.pop("dimension", None)}
    if settings != widened or len(dimensions) != 2 or None in dimensions:
        raise ValueError("the arms differ in more than the declared decision-path dimension")
    if not dimensions == {summaries[control]["config"]["dimension"],
                          summaries[lever]["config"]["dimension"]}:
        raise ValueError("declared dimension difference missing")
    if (summaries[control]["arguments"]["seed"] != summaries[lever]["arguments"]["seed"] or
            summaries[control]["arguments"].get("trunk_dimension", 192) !=
            summaries[control]["config"]["dimension"] or
            summaries[lever]["arguments"].get("trunk_dimension", 192) !=
            summaries[lever]["config"]["dimension"]):
        raise ValueError("lever arms differ in seed or in the recorded knob")
    for name in args.arms:
        if summaries[name].get("initial_weights_from") is not None:
            raise ValueError("the declared lever is a fresh-baseline pair, not a continued run")
        if summaries[name]["arguments"].get("lr_final") is not None:
            raise ValueError("the declared lever holds the constant published rate")
    architectures = {name: checkpoint_architecture(summaries[name], manifests[name]) for name in args.arms}
    if len(set(architectures.values())) != 2:
        raise ValueError("the arms are not two declared identities")
    started = time.monotonic()
    code = runs[control]/"code"
    verify_source(code, summaries[control]["catalogue_sha256"])
    # Anything imported above came from the working tree, not from the arm's frozen archive; drop
    # those modules so the frozen-source check below reads the arm's own recorded implementation.
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
            raise ValueError("lever model/label implementation is not frozen")
    if not Path(model_module.__file__).resolve().is_relative_to(code):
        raise ValueError("lever model import is not frozen")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    models, weights = {}, {}
    for name in args.arms:
        model, loaded = load_checkpoint(runs[name]/"checkpoint")
        if loaded["weights_sha256"] != summaries[name]["checkpoint_sha256"]:
            raise ValueError("lever weights differ from the recorded run")
        model.eval()
        models[name] = model
        weights[name] = {key: value.clone() for key, value in model.state_dict().items()}
    config = summaries[control]["config"]
    board_cells = int(config["visible_rows"])*int(config["columns"])
    capacity = summaries[control]["config"].get("component_capacity", 0)
    curriculum = summaries[control].get("curriculum", "stream-v1")
    max_side = summaries[control].get("max_side", 512)
    thresholds = list(threshold_grid())
    recipes = {}
    for split in ("training", "development"):
        ledger = runs[control]/(split+"-probe.jsonl")
        if ledger.stat().st_size > 1024**2:
            raise ValueError("probe ledger byte budget exceeded")
        recipes[split] = [json.loads(line) for line in ledger.read_text().splitlines()]
        other = runs[lever]/(split+"-probe.jsonl")
        if other.read_bytes() != ledger.read_bytes():
            raise ValueError("the arms hold different saved probe populations")
    saved_ids = {split: {row["sample_id"] for row in rows} for split, rows in recipes.items()}
    evaluated_training = require_training_population(RECORDED_TRAINING_GROUPS, args.training_groups)
    if evaluated_training != RECORDED_TRAINING_GROUPS:
        if (len(recipes["training"]) != RECORDED_TRAINING_GROUPS*3 or
                {(row["group_index"], row["view"]) for row in recipes["training"]} !=
                {(group, view) for group in range(RECORDED_TRAINING_GROUPS) for view in range(3)}):
            raise ValueError("recorded training probe population is three complete groups")
        for spec in training_extension_specs(recipes["training"][0]["base_seed"],
                                             RECORDED_TRAINING_GROUPS, evaluated_training):
            case = sample_for(spec["group_index"], spec["view"], "training", spec["base_seed"],
                              curriculum=curriculum, component_capacity=capacity, max_side=max_side)
            recipe = json.loads(json.dumps(case.record))
            if (recipe.get("split") != "training" or
                    (recipe["group_index"], recipe["view"], recipe["base_seed"]) !=
                    (spec["group_index"], spec["view"], spec["base_seed"])):
                raise ValueError("lever training extension identity differs")
            recipes["training"].append(recipe)
    population = {split: len(rows) for split, rows in recipes.items()}

    def forward_case(recipe, split, *, saved=True):
        sample_kwargs, target_kwargs = replay_keywords(recipe, capacity)
        case = (replay_case(recipe, split=split, curriculum=curriculum, component_capacity=capacity,
                            max_side=max_side) if "curriculum" in summaries[control] else
                sample_case(recipe["group_index"], recipe["view"], split, recipe["base_seed"],
                            max_side=max_side, **sample_kwargs))
        if json.loads(json.dumps(case.record)) != recipe:
            raise ValueError("lever recipe/pixel/label identity differs")
        if saved:
            stem = runs[control]/(split+"-probe")/recipe["sample_id"]
            with Image.open(stem.with_suffix(".png")) as image:
                if (image.mode != case.frame.image.mode or image.size != case.frame.image.size or
                        image.tobytes() != case.frame.image.tobytes()):
                    raise ValueError("saved probe raster differs")
            if file_digest(stem.with_suffix(".json")) != recipe["state_sha256"]:
                raise ValueError("saved probe annotation differs")
        prepared = prepare_image(case.frame.image, max_side=max_side)
        target = targets_for(case.frame.state, prepared, **target_kwargs)
        known = target["known"][0].flatten().bool()
        occupied = target["occupied"][0].flatten().bool()
        if known.numel() != board_cells or int(known.sum()) != board_cells:
            raise ValueError("lever read requires the fully supported visible board")
        vectors, readouts = [], {}
        for name in args.arms:
            with torch.no_grad():
                output = models[name](prepared.pixels[None], prepared.valid[None])
            vectors.append([round(float(value), 6)
                            for value in visible_board_logits(output["final"]["occupancy"], config)[known]])
            readouts[name] = readout_statistics(output["final"]["occupancy"], config)
        return {"split": split, "sample_id": recipe["sample_id"], "group_id": recipe["group_id"],
                "view": recipe["view"], "known_cells": int(known.sum()),
                "logits": vectors, "occupied": [int(value) for value in occupied[known]],
                "readouts": readouts}

    if args.profile:
        profile_cases = [forward_case(recipes["training"][0], "training"),
                         forward_case(recipes["development"][0], "development")]
        (args.output/"summary.json").write_text(json.dumps(
            {"status": "complete", "arms": list(args.arms),
             "weights_sha256": {name: summaries[name]["checkpoint_sha256"] for name in args.arms},
             "architectures": architectures, "dimensions": sorted(dimensions),
             "thresholds": thresholds, "admission_ceiling_bytes": ceiling, "cases": population,
             "training_groups": evaluated_training, "profile_forwards": 2*len(profile_cases),
             "scope": "Resource profile only: two arms resident, no sweep receipt, no claim."},
            indent=2)+"\n", encoding="utf-8")
        print(json.dumps({"status": "complete", "profile": True, "forwards": 4}), flush=True)
        return
    profile = args.admission
    profile_summary = json.loads((profile/"summary.json").read_text())
    profile_job = json.loads((ROOT/"runs"/(profile.name+"-job.json")).read_text())
    if (profile_summary.get("status") != "complete" or profile_job.get("status") != "passed" or
            profile_job.get("exit_code") != 0 or
            profile_summary.get("weights_sha256") !=
            {name: summaries[name]["checkpoint_sha256"] for name in args.arms} or
            profile_summary.get("architectures") != architectures or
            profile_summary.get("thresholds") != thresholds or
            profile_summary.get("admission_ceiling_bytes") != ceiling or
            profile_summary.get("cases") != population or
            profile_summary.get("training_groups") != evaluated_training):
        raise ValueError("lever admission profile differs")
    cases = []
    for split in ("training", "development"):
        for recipe in recipes[split]:
            cases.append(forward_case(recipe, split, saved=recipe["sample_id"] in saved_ids[split]))
            (args.output/"progress.json").write_text(json.dumps({"cases": len(cases)})+"\n")
    for name in args.arms:
        if not all(torch.equal(value, models[name].state_dict()[key])
                   for key, value in weights[name].items()):
            raise ValueError("read-only lever pass changed model tensors")
    readout = {name: {"readout_shape": cases[0]["readouts"][name]["shape"],
                      "visible_rows": cases[0]["readouts"][name]["visible_rows"],
                      "spawn_rows": cases[0]["readouts"][name]["spawn_rows"],
                      "columns": cases[0]["readouts"][name]["columns"],
                      "visible_cells": board_cells,
                      "derived_from": "config.visible_rows x config.columns",
                      **{key: round(sum(case["readouts"][name][key] for case in cases)/len(cases), 6)
                         for key in ("visible_mean_abs", "visible_std", "spawn_mean_abs")}}
               for name in args.arms}
    if len({tuple(value["readout_shape"]) for value in readout.values()}) != 1:
        raise ValueError("lever arms produce different read-out maps")
    split_cases = {split: [case for case in cases if case["split"] == split]
                   for split in ("training", "development")}
    arm_cases = {name: {split: [{"split": split, "sample_id": case["sample_id"],
                                 "logits": case["logits"][index], "occupied": case["occupied"]}
                                for case in split_cases[split]]
                        for split in split_cases}
                 for index, name in enumerate(args.arms)}
    deciding = {name: evaluate(arm_cases[name]["training"], thresholds) for name in args.arms}
    development = {name: evaluate(arm_cases[name]["development"], thresholds) for name in args.arms}
    flips = known_cells = 0
    for case in split_cases["training"]:
        for first, second in zip(case["logits"][0], case["logits"][1]):
            known_cells += 1
            flips += int((first > 0.0) != (second > 0.0))
    verdict = lever_verdict(deciding[control], deciding[lever], flips=flips, known_cells=known_cells)
    placement = {name: summaries[name]["development"]["mean_geometry_error_cells"] for name in args.arms}
    result = {"status": "passed", "arms": list(args.arms),
              "weights_sha256": {name: summaries[name]["checkpoint_sha256"] for name in args.arms},
              "architectures": architectures,
              "config": {"control": summaries[control]["config"], "lever": summaries[lever]["config"]},
              "parameters": {name: manifests[name]["parameters"] for name in args.arms},
              "forwards": len(cases)*2, "job_ceiling_bytes": ceiling,
              "population": {"training_groups": evaluated_training,
                             "training_cases": population["training"],
                             "development_cases": population["development"],
                             "nonempty_deciding_boards": deciding[control]["nonempty_images"],
                             "known_cells": known_cells},
              "admission": {"profile": profile.name, "forwards": profile_summary["profile_forwards"],
                            "ceiling_bytes": ceiling, "cases": population},
              "grid": {"low": thresholds[0], "high": thresholds[-1],
                       "step": round(thresholds[1]-thresholds[0], 2), "count": len(thresholds)},
              "arms_read": {name: {"auc": deciding[name]["auc"],
                                   "f1_at_zero": f1_at(deciding[name]),
                                   "best": deciding[name]["best"],
                                   "exact_boards_at_best": deciding[name]["exact_boards_at_best"],
                                   "development": {"auc": development[name]["auc"],
                                                   "best": development[name]["best"],
                                                   "mean_geometry_error_cells": placement[name]}}
                            for name in args.arms},
              "decisions": {"flips": flips, "known_cells": known_cells},
              "readout": readout,
              "verdict": verdict, "sources": {name: digest(ROOT/name) for name in READ_SOURCES},
              "elapsed_seconds": time.monotonic()-started,
              "scope": ("Read-only paired read of the declared trunk-width lever: two fresh-baseline arms "
                        "differing only in dimension. No training, no weight change, no gate or "
                        "complete-state claim; the verdict rule was predeclared in "
                        "docs/M3_TRUNK_WIDTH_LEVER.md before this receipt existed.")}
    with (args.output/"result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    (args.output/"scope.json").write_text(json.dumps(
        {key: result[key] for key in ("status", "arms", "weights_sha256", "architectures", "parameters",
                                      "population", "readout", "arms_read", "decisions", "verdict", "grid",
                                      "forwards", "job_ceiling_bytes", "sources", "scope")},
        indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("verdict", "readout", "arms_read", "decisions")
                      if key in result}, default=str), flush=True)


if __name__ == "__main__":
    main()
