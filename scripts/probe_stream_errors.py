"""Decompose saved synthetic M2/M3 errors; one ordinary forward per verified probe.

Read-only analysis, not training, oracle geometry, calibrated parsing or real
accuracy. Run after the training Job terminates, under a separate native cap.
"""
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import time

from m2_error_analysis import analyze_case, fit_metrics_from_diagnostics, recipe_strata, summarize
from m3_error_analysis import analyze_components, pack_components, replay_keywords, summarize_components
from probe_model import require_bounded_job
from record_stream_run import checkpoint_architecture, validate_completion
from source_snapshot import verify_source

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    if path.stat().st_size > 1024**2:
        raise ValueError("error probe JSON byte budget exceeded")
    return json.loads(path.read_text())


def require_frozen_callables(code, *functions):
    # Torch's no_grad wrapper lives in Torch, while __wrapped__ keeps the actual
    # verified metric implementation. Validate that implementation's source.
    if any(not Path(inspect.unwrap(function).__code__.co_filename).resolve().is_relative_to(code) for function in functions):
        raise ValueError("error probe implementation is not frozen")


def pack(final, target):
    prediction = {"occupancy_logits": final["occupancy"][0, :20].tolist(),
                  "observability_logits": final["observability"][0, :20].tolist(),
                  "semantic": final["semantic"][0, :20].argmax(-1).tolist(),
                  "lattice": final["lattice"][0].tolist(), "registration_logit": float(final["registration"][0]),
                  "layers": [{"known_logit": float(final["layer_known"][0, i]),
                              "type": int(final["layer_type"][0, i].argmax()),
                              "rotation": int(final["pose_rotation"][0, i].argmax()),
                              "column": int(final["pose_column"][0, i].argmax()),
                              "row": int(final["pose_row"][0, i].argmax())} for i in range(2)]}
    truth = {key: target[key][0].tolist() for key in ("known", "typed", "semantic", "lattice", "registered")}
    truth["occupied"] = target["occupied"][0].bool().tolist()
    truth["layers"] = [{"known": bool(target["layer_known"][0, i]), "pose_known": bool(target["pose_known"][0, i]),
                        "type": int(target["layer_type"][0, i]), "rotation": 0,
                        "column": int(target["column"][0, i]), "row": int(target["row"][0, i]),
                        "rotations": target["rotations"][0, i].tolist()} for i in range(2)]
    return prediction, truth


def main():
    ceiling = require_bounded_job()
    source_paths = [ROOT/"scripts"/name for name in
                    ("probe_stream_errors.py", "m2_error_analysis.py", "m3_error_analysis.py", "record_stream_run.py", "source_snapshot.py", "probe_model.py")]
    sources = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"m[23]-stream-[a-z0-9-]{1,40}", args.run):
        raise ValueError("owned source-frozen stream run required")
    run = ROOT/"runs"/args.run
    summary = read(run/"summary.json")
    job = read(ROOT/"runs"/(args.run+"-job.json"))
    validate_completion(summary, job, allow_partial=True)
    code = run/"code"
    verify_source(code, summary["catalogue_sha256"])
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    sys.path[:0] = [str(code/"scripts"), str(code)]
    sys.dont_write_bytecode = True
    os.environ.update({"ONEDNN_PRIMITIVE_CACHE_CAPACITY": "16", "DNNL_PRIMITIVE_CACHE_CAPACITY": "16"})
    import torch
    from PIL import Image
    from m2_stream_data import sample_case
    from tetris_vision.checkpoint import file_digest, load_checkpoint
    from tetris_vision.losses import diagnostic_metrics, joint_loss, targets_for
    from tetris_vision.preprocess import prepare_image
    import tetris_vision.model as model_module
    require_frozen_callables(code, sample_case, targets_for)
    curriculum = summary.get("curriculum", "stream-v1")
    if "curriculum" in summary:
        from m2_curriculum import replay_case
        require_frozen_callables(code, replay_case)
    if not Path(model_module.__file__).resolve().is_relative_to(code):
        raise ValueError("error probe model/sampler/labels import is not frozen")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model, manifest = load_checkpoint(run/"checkpoint")
    architecture = checkpoint_architecture(summary, manifest)
    capacity = summary["config"].get("component_capacity", 0)
    if capacity:
        from tetris_vision.components import component_metrics
        require_frozen_callables(code, component_metrics)
    model.eval()
    if (manifest["weights_sha256"] != summary["checkpoint_sha256"] or manifest["config"] != summary["config"] or
            manifest["torch_version"] != str(torch.__version__)):
        raise ValueError("error probe weight/config/runtime identity differs")
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    records = []
    with torch.no_grad():
        for split, field in (("training", "final_training_probe"), ("development", "development")):
            ledger = run/(split+"-probe.jsonl")
            if ledger.stat().st_size > 1024**2:
                raise ValueError("probe ledger byte budget exceeded")
            recipes = [json.loads(line) for line in ledger.read_text().splitlines()]
            expected = {row["sample_id"]: row for row in summary[field]["cases"]}
            training_limit = 32 if curriculum == "all-class-fit-v1" else 9
            if not 1 <= len(recipes) <= (training_limit if split == "training" else 36) or len(recipes) != len(expected):
                raise ValueError("finite declared probe budget or saved population differs")
            for recipe in recipes:
                sample_kwargs, target_kwargs = replay_keywords(recipe, capacity)
                if manifest["config"].get("layer_evidence", False):
                    target_kwargs["layer_evidence"] = True
                case = (replay_case(recipe, split=split, curriculum=curriculum, component_capacity=capacity) if "curriculum" in summary else
                        sample_case(recipe["group_index"], recipe["view"], split, recipe["base_seed"], **sample_kwargs))
                if json.loads(json.dumps(case.record)) != recipe:
                    raise ValueError("error probe recipe/pixel/label identity differs")
                stem = run/(split+"-probe")/recipe["sample_id"]
                with Image.open(stem.with_suffix(".png")) as image:
                    if (image.mode != case.frame.image.mode or image.size != case.frame.image.size or
                            image.tobytes() != case.frame.image.tobytes()):
                        raise ValueError("saved probe raster differs")
                    prepared = prepare_image(image, max_side=512)
                if file_digest(stem.with_suffix(".json")) != recipe["state_sha256"]:
                    raise ValueError("saved probe annotation differs")
                target = targets_for(case.frame.state, prepared, **target_kwargs)
                if int(target["known"].sum()) != 200 or not bool(target["registered"].all()):
                    raise ValueError("saved streaming probe requires200 supported cells and registered grid")
                output = model(prepared.pixels[None], prepared.valid[None])
                loss, _ = joint_loss(output, target)
                baseline = diagnostic_metrics(output, target)
                diagnostics = analyze_case(*pack(output["final"], target))
                known_layers = [layer for layer in diagnostics["layers"].values() if layer["target_known"]]
                baseline.update(loss=float(loss), known_layers=len(known_layers),
                                exact_known_layers=sum(layer["exact"] for layer in known_layers))
                if curriculum == "all-class-fit-v1" and split == "training":
                    baseline.update(group_id=recipe["group_id"], fit_case=fit_metrics_from_diagnostics(diagnostics))
                if capacity:
                    baseline["components"] = component_metrics(output["final"]["components"], target["components"])[0]
                    components = analyze_components(*pack_components(output["final"]["components"], target["components"]))
                    if any(components[key] != value for key, value in baseline["components"].items()):
                        raise ValueError("component decomposition disagrees with frozen metrics")
                    diagnostics["components"] = components
                saved = expected[recipe["sample_id"]]
                if any(value != saved[key] for key, value in baseline.items()):
                    raise ValueError("ordinary predictions fail to reproduce saved metrics")
                if (baseline["exact_locked_board"] != diagnostics["locked"]["exact_board"] or
                        not math.isclose(baseline["cell_accuracy"], diagnostics["locked"]["cell_accuracy"], abs_tol=1e-6) or
                        not math.isclose(baseline["mean_geometry_error_cells"], diagnostics["geometry"]["mean_cells"], rel_tol=1e-5, abs_tol=1e-5)):
                    raise ValueError("independent decomposition disagrees with frozen metrics")
                density, skin = recipe_strata(recipe)
                records.append({"split": split, "sample_id": recipe["sample_id"], "group_id": recipe["group_id"],
                                "density_recipe": density, "skin": skin, "view": recipe["view"],
                                "ordinary_metrics": baseline, "diagnostics": diagnostics})
                (args.output/"progress.json").write_text(json.dumps({"forwards": len(records), "sample_id": recipe["sample_id"]})+"\n")
    if not all(torch.equal(value, model.state_dict()[name]) for name, value in weights.items()):
        raise ValueError("read-only error probe changed model tensors")
    if sources != {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}:
        raise ValueError("diagnostic source changed during execution")
    totals = summarize(records, max_cases=68 if curriculum == "all-class-fit-v1" else 45)
    if capacity:
        for split, values in summarize_components(records).items():
            totals[split]["components"] = values
    result = {"status": "passed", "diagnostic_format_version": 2,
              "run": args.run, "run_status": summary["status"], "global_steps": summary["steps"],
              "target_steps": summary["requested_steps"], "forwards": len(records), "job_ceiling_bytes": ceiling,
              "source_catalogue_sha256": summary["catalogue_sha256"], "weights_sha256": summary["checkpoint_sha256"],
              "automatic_metrics_reproduced": True, "weights_unchanged": True, "label_inputs_to_model": False,
              "core_architecture": architecture, "config": manifest["config"], "component_capacity": capacity,
              "curriculum": curriculum,
              "totals": totals, "cases": records, "sources": sources,
              "elapsed_seconds": time.monotonic()-started,
              "scope": "Saved synthetic raw-head error decomposition only. Matched views share ancestry; no independent-sample interval, calibrated confidence, decoded complete-state or real accuracy claim. Unknown component targets are excluded. No training/label/production-output changes."}
    with (args.output/"result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}), flush=True)


if __name__ == "__main__":
    main()
