"""Read-only bias-versus-variance headroom read on the deciding occupancy population.

Two committed arms of the identical published architecture - same initial tensor file, seed
and data order, differing only in schedule - are reloaded and swept in one execution. The
predeclared ensemble is their unweighted per-cell mean occupancy logit, so the pass measures
how much of the residual deciding error is trajectory variance rather than representation.
No training, no weight change, no new parameter, no architecture change. The rule that reads
this receipt is docs/M3_DECIDING_HEADROOM.md, written before the receipt existed.
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
CEILING_BYTES = 640*1024**2
F1_BAR, AUC_BAR, ATTRIBUTION_FLOOR = 0.70, 0.90, 0.05
HEADROOM_SOURCES = tuple("scripts/"+name for name in
    ("ensemble_headroom.py", "occupancy_sweep.py", "record_stream_run.py", "source_snapshot.py",
     "probe_model.py", "probe_population.py"))


def digest(path, maximum=64*1024**2):
    value, consumed = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            consumed += len(block)
            if consumed > maximum:
                raise ValueError("headroom source byte budget exceeded")
            value.update(block)
    return value.hexdigest()


def mean_logits(vectors):
    """Unweighted per-cell mean of the arms' raw occupancy logits (the declared ensemble)."""
    if type(vectors) is not list or len(vectors) < 2:
        raise ValueError("two arm logit vectors required")
    width = None
    for vector in vectors:
        if type(vector) is not list or not vector:
            raise ValueError("non-empty logit vector per arm required")
        if width is None:
            width = len(vector)
        elif len(vector) != width:
            raise ValueError("arm logit widths differ")
        if any(type(value) not in (int, float) for value in vector):
            raise ValueError("numeric logits required")
    return [round(sum(values)/len(vectors), 6) for values in zip(*vectors)]


def f1_at(summary, threshold=0.0):
    """The swept F1 at one declared grid point (the standing decision boundary)."""
    for row in summary["rows"]:
        if row["threshold"] == threshold:
            return row["f1"]
    raise ValueError("declared threshold missing from the swept grid")


def headroom_verdict(ensemble, members, *, auc_bar=AUC_BAR, f1_bar=F1_BAR,
                     floor=ATTRIBUTION_FLOOR):
    """The predeclared three-branch rule; there is no fourth branch."""
    if (type(ensemble) is not dict or not isinstance(ensemble.get("auc"), (int, float)) or
            type(ensemble.get("best")) is not dict or not isinstance(ensemble["best"].get("f1"),
                                                                    (int, float)) or
            not isinstance(ensemble.get("rows"), list)):
        raise ValueError("ensemble sweep summary required")
    if type(members) is not list or not members:
        raise ValueError("member sweep summaries required")
    for member in members:
        if (type(member) is not dict or type(member.get("best")) is not dict or
                not isinstance(member["best"].get("f1"), (int, float)) or
                not isinstance(member.get("rows"), list)):
            raise ValueError("member sweep summaries required")
    member_best = max(member["best"]["f1"] for member in members)
    best = ensemble["best"]["f1"]
    if ensemble["auc"] >= auc_bar and best >= f1_bar:
        label = "ensemble_clears"
    elif ensemble["auc"] >= auc_bar and best-member_best >= floor:
        label = "variance_limited"
    else:
        label = "bias_limited"
    return {"label": label, "ensemble_best_f1": best, "ensemble_auc": ensemble["auc"],
            "member_best_f1": member_best, "best_delta": best-member_best,
            "f1_at_zero_delta": f1_at(ensemble)-max(f1_at(member) for member in members),
            "auc_bar": auc_bar, "f1_bar": f1_bar, "attribution_floor": floor}


def visible_units(occupancy, config):
    """Deciding logits of one case: the visible board of the map, sized by the declared config."""
    from board_window import visible_board
    return visible_board(occupancy, config, "occupancy")[0].flatten()


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
        raise ValueError("headroom ceiling differs from the predeclared 640 MiB")
    args.output.mkdir(parents=True, exist_ok=False)
    runs, summaries, manifests = {}, {}, {}
    for name in args.arms:
        run = ROOT/"runs"/name
        summary = json.loads((run/"summary.json").read_text())
        manifest = json.loads((run/"checkpoint/manifest.json").read_text())
        validate_completion(summary, json.loads((ROOT/"runs"/(name+"-job.json")).read_text()))
        if (manifest["weights_sha256"] != summary["checkpoint_sha256"] or
                manifest["config"] != summary["config"]):
            raise ValueError("headroom checkpoint identity differs")
        runs[name], summaries[name], manifests[name] = run, summary, manifest
    control, lever = args.arms
    if summaries[control]["catalogue_sha256"] != summaries[lever]["catalogue_sha256"]:
        raise ValueError("headroom arms were not trained from one frozen catalogue")
    if summaries[control]["config"] != summaries[lever]["config"]:
        raise ValueError("headroom arms differ in architecture or configuration")
    if summaries[control]["arguments"]["seed"] != summaries[lever]["arguments"]["seed"]:
        raise ValueError("headroom arms differ in seed")
    if summaries[control]["initial_weights_sha256"] != summaries[lever]["initial_weights_sha256"]:
        raise ValueError("headroom arms were not started from one initial tensor file")
    if summaries[control]["checkpoint_sha256"] == summaries[lever]["checkpoint_sha256"]:
        raise ValueError("headroom arms are one checkpoint")
    architecture = checkpoint_architecture(summaries[control], manifests[control])
    if architecture != checkpoint_architecture(summaries[lever], manifests[lever]):
        raise ValueError("headroom arm architectures differ")
    started = time.monotonic()
    code = runs[control]/"code"
    verify_source(code, summaries[control]["catalogue_sha256"])
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
            raise ValueError("headroom model/label implementation is not frozen")
    if not Path(model_module.__file__).resolve().is_relative_to(code):
        raise ValueError("headroom model import is not frozen")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    models, weights = {}, {}
    for name in args.arms:
        model, loaded = load_checkpoint(runs[name]/"checkpoint")
        if loaded["weights_sha256"] != summaries[name]["checkpoint_sha256"]:
            raise ValueError("headroom weights differ from the recorded run")
        model.eval()
        models[name] = model
        weights[name] = {key: value.clone() for key, value in model.state_dict().items()}
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
            raise ValueError("headroom arms hold different saved probe populations")
    saved_ids = {split: {row["sample_id"] for row in rows} for split, rows in recipes.items()}
    evaluated_training = require_training_population(RECORDED_TRAINING_GROUPS, args.training_groups)
    saved_training_cases = len(recipes["training"])
    if evaluated_training != RECORDED_TRAINING_GROUPS:
        if (saved_training_cases != RECORDED_TRAINING_GROUPS*3 or
                {(row["group_index"], row["view"]) for row in recipes["training"]} !=
                {(group, view) for group in range(RECORDED_TRAINING_GROUPS) for view in range(3)}):
            raise ValueError("recorded training probe population is three complete groups")
        # The same widening the error replay and the sweeps apply, from the same declaration.
        for spec in training_extension_specs(recipes["training"][0]["base_seed"],
                                             RECORDED_TRAINING_GROUPS, evaluated_training):
            case = sample_for(spec["group_index"], spec["view"], "training", spec["base_seed"],
                              curriculum=curriculum, component_capacity=capacity, max_side=max_side)
            recipe = json.loads(json.dumps(case.record))
            if (recipe.get("split") != "training" or
                    (recipe["group_index"], recipe["view"], recipe["base_seed"]) !=
                    (spec["group_index"], spec["view"], spec["base_seed"])):
                raise ValueError("headroom training extension identity differs")
            recipes["training"].append(recipe)
    population = {split: len(rows) for split, rows in recipes.items()}

    def forward_case(recipe, split, *, saved=True):
        sample_kwargs, target_kwargs = replay_keywords(recipe, capacity)
        if manifests[control]["config"].get("layer_evidence", False):
            target_kwargs["layer_evidence"] = True
        case = (replay_case(recipe, split=split, curriculum=curriculum, component_capacity=capacity,
                            max_side=max_side) if "curriculum" in summaries[control] else
                sample_case(recipe["group_index"], recipe["view"], split, recipe["base_seed"],
                            max_side=max_side, **sample_kwargs))
        if json.loads(json.dumps(case.record)) != recipe:
            raise ValueError("headroom recipe/pixel/label identity differs")
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
        if int(target["known"].sum()) != 200:
            raise ValueError("headroom read requires 200 supported known cells")
        known = target["known"][0].flatten().bool()
        occupied = target["occupied"][0].flatten().bool()
        vectors = []
        for name in args.arms:
            with torch.no_grad():
                output = models[name](prepared.pixels[None], prepared.valid[None])
            vectors.append([round(float(value), 6)
                            for value in visible_units(output["final"]["occupancy"],
                                                       manifests[control]["config"])[known]])
        return {"split": split, "sample_id": recipe["sample_id"], "group_id": recipe["group_id"],
                "view": recipe["view"], "known_cells": int(known.sum()),
                "logits": vectors, "occupied": [int(value) for value in occupied[known]]}

    if args.profile:
        profile_cases = [forward_case(recipes["training"][0], "training"),
                         forward_case(recipes["development"][0], "development")]
        (args.output/"summary.json").write_text(json.dumps(
            {"status": "complete", "arms": list(args.arms),
             "weights_sha256": {name: summaries[name]["checkpoint_sha256"] for name in args.arms},
             "config": summaries[control]["config"], "core_architecture": architecture,
             "thresholds": thresholds, "admission_ceiling_bytes": ceiling, "cases": population,
             "training_groups": evaluated_training,
             "profile_forwards": 2*len(profile_cases),
             "profile_cases": [{"sample_id": row["sample_id"], "known_cells": row["known_cells"],
                                "occupied_cells": sum(row["occupied"])} for row in profile_cases],
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
            profile_summary.get("config") != summaries[control]["config"] or
            profile_summary.get("thresholds") != thresholds or
            profile_summary.get("admission_ceiling_bytes") != ceiling or
            profile_summary.get("cases") != population or
            profile_summary.get("training_groups") != evaluated_training):
        raise ValueError("headroom admission profile differs")
    cases = []
    for split in ("training", "development"):
        for recipe in recipes[split]:
            cases.append(forward_case(recipe, split,
                                      saved=recipe["sample_id"] in saved_ids[split]))
            (args.output/"progress.json").write_text(json.dumps({"cases": len(cases)})+"\n")
    for name in args.arms:
        if not all(torch.equal(value, models[name].state_dict()[key])
                   for key, value in weights[name].items()):
            raise ValueError("read-only headroom pass changed model tensors")
    sources = {name: digest(ROOT/name) for name in HEADROOM_SOURCES}
    split_cases = {split: [case for case in cases if case["split"] == split]
                   for split in ("training", "development")}
    arm_cases = {name: {split: [{"split": split, "sample_id": case["sample_id"],
                                 "logits": case["logits"][index], "occupied": case["occupied"]}
                                for case in split_cases[split]]
                        for split in split_cases}
                 for index, name in enumerate(args.arms)}
    ensemble_cases = {split: [{"split": split, "sample_id": case["sample_id"],
                              "logits": mean_logits(case["logits"]), "occupied": case["occupied"]}
                             for case in split_cases[split]]
                      for split in split_cases}
    deciding = {name: evaluate(arm_cases[name]["training"], thresholds) for name in args.arms}
    ensemble_deciding = evaluate(ensemble_cases["training"], thresholds)
    ensemble_development = evaluate(ensemble_cases["development"], thresholds)
    verdict = headroom_verdict(ensemble_deciding, list(deciding.values()))
    agreement = {"both_right": 0, "control_only": 0, "lever_only": 0, "both_wrong": 0,
                 "empty_both_right": 0}
    for case in split_cases["training"]:
        for first, second, label in zip(case["logits"][0], case["logits"][1], case["occupied"]):
            right_first, right_second = (first > 0.0) == bool(label), (second > 0.0) == bool(label)
            if not label:
                agreement["empty_both_right"] += int(right_first and right_second)
            elif right_first and right_second:
                agreement["both_right"] += 1
            elif right_first:
                agreement["control_only"] += 1
            elif right_second:
                agreement["lever_only"] += 1
            else:
                agreement["both_wrong"] += 1
    result = {"status": "passed", "arms": list(args.arms),
              "weights_sha256": {name: summaries[name]["checkpoint_sha256"] for name in args.arms},
              "config": manifests[control]["config"], "core_architecture": architecture,
              "forwards": len(cases)*2, "job_ceiling_bytes": ceiling,
              "population": {"training_groups": evaluated_training,
                             "training_cases": population["training"],
                             "development_cases": population["development"],
                             "nonempty_deciding_boards": ensemble_deciding["nonempty_images"]},
              "admission": {"profile": profile.name, "forwards": profile_summary["profile_forwards"],
                            "ceiling_bytes": ceiling, "cases": population},
              "grid": {"low": thresholds[0], "high": thresholds[-1],
                       "step": round(thresholds[1]-thresholds[0], 2), "count": len(thresholds)},
              "members": {name: {"auc": deciding[name]["auc"],
                                 "f1_at_zero": f1_at(deciding[name]),
                                 "best": deciding[name]["best"],
                                 "exact_boards_at_best": deciding[name]["exact_boards_at_best"]}
                          for name in args.arms},
              "ensemble": {"auc": ensemble_deciding["auc"],
                           "f1_at_zero": f1_at(ensemble_deciding),
                           "best": ensemble_deciding["best"],
                           "exact_boards_at_best": ensemble_deciding["exact_boards_at_best"],
                           "development": {"auc": ensemble_development["auc"],
                                           "best": ensemble_development["best"]}},
              "deciding": ensemble_deciding,
              "agreement_at_zero": agreement, "verdict": verdict, "cases": cases,
              "sources": sources, "elapsed_seconds": time.monotonic()-started,
              "scope": ("Read-only bias-versus-variance read of two committed checkpoints "
                        "differing only in schedule; unweighted mean logit; no training, no weight "
                        "change, no new parameter, no gate or complete-state claim, and the rule "
                        "was predeclared before this receipt existed.")}
    with (args.output/"result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    (args.output/"scope.json").write_text(json.dumps(
        {key: result[key] for key in ("status", "arms", "weights_sha256", "population", "members",
                                      "ensemble", "agreement_at_zero", "verdict", "grid", "forwards",
                                      "job_ceiling_bytes", "sources", "scope")}, indent=2)+"\n",
        encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("cases", "deciding", "f1_at_zero")}), flush=True)


if __name__ == "__main__":
    main()
