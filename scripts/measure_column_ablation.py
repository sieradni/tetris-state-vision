"""Measure what the declared column read-out contributes, by zeroing it on the frozen checkpoint.

Read-only: the committed checkpoint is reloaded from its own frozen archive, the deciding population is
run twice - as trained, then with the column projection's weight and bias zeroed - and the two sides are
scored on the same cells. The state dict is compared before and after and the run is refused if any value
moved. No training, no weight change, no decoded state, no accuracy claim.
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
KNOWN_CELLS = DECIDING_IMAGES*200
OCCUPIED_CELLS = 667


def digest(path, maximum=64*1024**2):
    value, consumed = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            consumed += len(block)
            if consumed > maximum:
                raise ValueError("ablation source byte budget exceeded")
            value.update(block)
    return value.hexdigest()


def pooled_auc(positive, negative):
    if not positive or not negative:
        raise ValueError("AUC needs both classes")
    ordered = sorted([(value, 1) for value in negative]+[(value, 0) for value in positive])
    rank, positive_rank, index = 0, 0.0, 0
    while index < len(ordered):
        stop = index
        while stop+1 < len(ordered) and ordered[stop+1][0] == ordered[index][0]:
            stop += 1
        average = (index+stop)/2+1
        positive_rank += average*sum(1 for entry in ordered[index:stop+1] if entry[1] == 0)
        rank = stop+1
        index = stop+1
    count = len(positive)*len(negative)
    return positive_rank/count


def at_zero(logits, labels):
    true_positive = sum(1 for logit, label in zip(logits, labels) if label and logit > 0.0)
    false_positive = sum(1 for logit, label in zip(logits, labels) if not label and logit > 0.0)
    false_negative = sum(1 for logit, label in zip(logits, labels) if label and logit <= 0.0)
    denominator = 2*true_positive+false_positive+false_negative
    return {"f1": (2*true_positive/denominator if denominator else 0.0),
            "precision": (true_positive/max(1, true_positive+false_positive)),
            "true_positive": true_positive, "false_positive": false_positive,
            "false_negative": false_negative}


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
    if config.get("occupancy_column") != 1:
        raise ValueError("this instrument measures a declared column read-out arm")
    job = json.loads((ROOT/"runs"/(args.output.name+"-job.json")).read_text())
    if job["memory_bytes"] != CEILING_BYTES:
        raise ValueError("ablation ceiling differs from the predeclared 640 MiB")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    code = run/"code"
    verify_source(code, summary["catalogue_sha256"])
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
        raise ValueError("ablation model import is not frozen")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model, loaded = load_checkpoint(run/"checkpoint")
    if loaded["weights_sha256"] != summary["checkpoint_sha256"]:
        raise ValueError("measured weights differ from the recorded run")
    model.eval()
    if model.occupancy_column is None:
        raise ValueError("the declared arm has no column projection to ablate")
    before = {name: value.clone() for name, value in model.state_dict().items()}
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
            raise ValueError("ablation population identity differs")
        recipes.append(recipe)
    if len(recipes) != DECIDING_IMAGES:
        raise ValueError("the ablation requires exactly the 36 deciding images")

    def measure():
        logits, labels, contributions, cases = [], [], [], []
        for recipe in recipes:
            sample_kwargs, target_kwargs = replay_keywords(recipe, capacity)
            if config.get("layer_evidence", False):
                target_kwargs["layer_evidence"] = True
            case = (replay_case(recipe, split="training", curriculum=curriculum,
                                component_capacity=capacity, max_side=max_side)
                    if "curriculum" in summary else
                    sample_case(recipe["group_index"], recipe["view"], "training",
                                recipe["base_seed"], max_side=max_side, **sample_kwargs))
            prepared = prepare_image(case.frame.image, max_side=max_side)
            target = targets_for(case.frame.state, prepared, **target_kwargs)
            with torch.no_grad():
                full = model(prepared.pixels[None], prepared.valid[None])["final"]["occupancy"]
                full = visible_board(full, config, "occupancy")[0].flatten()
                weight, bias = model.occupancy_column.weight, model.occupancy_column.bias
                with torch.no_grad():
                    weight.zero_()
                    bias.zero_()
                plain = model(prepared.pixels[None], prepared.valid[None])["final"]["occupancy"]
                plain = visible_board(plain, config, "occupancy")[0].flatten()
                with torch.no_grad():
                    weight.copy_(before["occupancy_column.weight"])
                    bias.copy_(before["occupancy_column.bias"])
            known = target["known"][0].flatten().bool()
            occupied = target["occupied"][0].flatten().bool()
            if int(known.sum()) != 200 or full.shape[0] != known.numel():
                raise ValueError("the ablation cannot cover the labelled visible board")
            replacement = plain.clone()
            replacement[~known] = 0.0
            chosen = full[known]
            discarded = replacement[known]
            labels.extend(bool(value) for value in occupied[known])
            logits.append((chosen, discarded))
            difference = (chosen-discarded).abs()
            contributions.extend(float(value) for value in difference)
            cases.append({"sample_id": recipe["sample_id"], "group_id": recipe["group_id"],
                          "known_cells": int(known.sum()),
                          "mean_contribution": round(float(difference.mean()), 6),
                          "mean_abs_logit": round(float(chosen.abs().mean()), 6)})
        return logits, labels, contributions, cases

    trained, labels, contributions, cases = measure()
    trained_logits = [float(value) for values, _ in trained for value in values]
    ablated_logits = [float(value) for _, values in trained for value in values]
    if not all(torch.equal(value, model.state_dict()[name]) for name, value in before.items()):
        raise ValueError("read-only ablation changed model tensors")
    positive = [value for value, label in zip(trained_logits, labels) if label]
    negative = [value for value, label in zip(trained_logits, labels) if not label]
    ablated_positive = [value for value, label in zip(ablated_logits, labels) if label]
    ablated_negative = [value for value, label in zip(ablated_logits, labels) if not label]
    if (len(trained_logits), len(positive)) != (KNOWN_CELLS, OCCUPIED_CELLS):
        raise ValueError("ablation decided a different population from the sweep")
    with_read = {"auc": pooled_auc(positive, negative), **at_zero(trained_logits, labels)}
    without = {"auc": pooled_auc(ablated_positive, ablated_negative),
               **at_zero(ablated_logits, labels)}
    # Coherence guard: the read-out's own contribution cannot exceed the logits it is part of by more
    # than the logit magnitude allows; a contribution larger than every logit means the ablation
    # measured a different pass, not a read-out (V-106 class).
    mean_abs_logit = sum(abs(value) for value in trained_logits)/len(trained_logits)
    mean_contribution = sum(contributions)/len(contributions)
    if mean_contribution > 10*max(mean_abs_logit, 1e-6):
        raise ValueError("the ablation contribution is implausible against its own logits")
    result = {
        "status": "passed", "run": args.run, "weights_sha256": summary["checkpoint_sha256"],
        "config": config, "core_architecture": architecture, "forwards": 2*len(cases),
        "job_ceiling_bytes": job["memory_bytes"],
        "arm": {"occupancy_column": config["occupancy_column"],
                "occupancy_deform": config.get("occupancy_deform", 0),
                "projection_in_features": int(model.occupancy_column.in_features),
                "projection_out_features": int(model.occupancy_column.out_features)},
        "with": {"auc": round(with_read["auc"], 6), "f1": round(with_read["f1"], 6),
                 "precision": round(with_read["precision"], 6),
                 "true_positive": with_read["true_positive"],
                 "false_positive": with_read["false_positive"],
                 "false_negative": with_read["false_negative"]},
        "without": {"auc": round(without["auc"], 6), "f1": round(without["f1"], 6),
                    "precision": round(without["precision"], 6),
                    "true_positive": without["true_positive"],
                    "false_positive": without["false_positive"],
                    "false_negative": without["false_negative"]},
        "delta": {"auc": round(with_read["auc"]-without["auc"], 6),
                  "f1": round(with_read["f1"]-without["f1"], 6),
                  "precision": round(with_read["precision"]-without["precision"], 6)},
        "contribution": {"mean_logit": round(mean_contribution, 6),
                         "mean_abs_logit": round(mean_abs_logit, 6),
                         "share": round(mean_contribution/max(mean_abs_logit, 1e-9), 6),
                         "max_logit": round(max(contributions), 6)},
        "population": {"split": "training", "images": len(cases), "known_cells": KNOWN_CELLS,
                       "occupied_cells": OCCUPIED_CELLS, "groups": DECIDING_GROUPS},
        "cases": cases,
        "sources": {name: digest(ROOT/name) for name in
                    ("scripts/measure_column_ablation.py", "scripts/board_window.py")},
        "elapsed_seconds": time.monotonic()-started,
        "scope": ("Read-only ablation of one committed read-out: the same weights scored with and "
                  "without the column projection. No training, no weight change, no decoded state, "
                  "no accuracy claim."),
    }
    with (args.output/"result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": "passed", "with": result["with"], "without": result["without"],
                      "delta": result["delta"], "contribution": result["contribution"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
