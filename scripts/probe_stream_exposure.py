"""Replay frozen masked training/probe targets without a model or optimizer.

Run serially in a640MiB native Job,wall120s,work=updates+saved probes (<=4141).
Counts describe actual target exposure,not successful learning or accuracy.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys

from atomic_write import replace_retry
from probe_model import require_bounded_job
from record_stream_run import checkpoint_architecture, validate_completion
from source_snapshot import verify_source
from target_exposure import pack_layers, require, summarize

ROOT = Path(__file__).resolve().parents[1]
REPORT_SOURCES = tuple("scripts/"+name for name in
    ("probe_stream_exposure.py", "target_exposure.py", "record_stream_run.py", "source_snapshot.py",
     "probe_model.py", "run_bounded.py", "atomic_write.py"))
LEDGERS = {"training_updates": "training.jsonl", "training_probes": "training-probe.jsonl",
           "development_probes": "development-probe.jsonl"}


def read(path):
    require(path.stat().st_size <= 1024**2, "exposure JSON budget exceeded")
    return json.loads(path.read_text())


def digest(path):
    require(path.stat().st_size <= 64*1024**2, "exposure artifact budget exceeded")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def recipe_digest(recipe):
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    replace_retry(temporary, path)


def input_contract(summary, job, manifest):
    validate_completion(summary, job, allow_partial=True)
    architecture = checkpoint_architecture(summary, manifest)
    require(architecture == "TetrisStateNet-M2" and summary["profile_square"] is False and
            manifest["weights_sha256"] == summary["checkpoint_sha256"], "saved ordinary M2 curriculum artifact required")
    return architecture


def load_recipes(run, summary):
    result = {}
    for population, filename in LEDGERS.items():
        path = run/filename
        require(path.stat().st_size <= 16*1024**2, "finite exposure recipe ledger required")
        result[population] = [json.loads(line) for line in path.read_text().splitlines()]
    require([r["step"] for r in result["training_updates"]] == list(range(1, summary["steps"]+1)), "training cursor differs")
    require(len(result["training_probes"]) == 9 and len(result["development_probes"]) == 3*summary["development_groups"] and
            9 <= len(result["development_probes"]) <= 36, "saved probe population differs")
    return result


def main():
    ceiling = require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(re.fullmatch(r"m2-stream-[a-z0-9-]{1,40}", args.run), "owned original stream source required")
    run = ROOT/"runs"/args.run
    summary, job, manifest = read(run/"summary.json"), read(ROOT/"runs"/(args.run+"-job.json")), read(run/"checkpoint/manifest.json")
    architecture = input_contract(summary, job, manifest)
    code = run/"code"
    verify_source(code, summary["catalogue_sha256"])
    recipes = load_recipes(run, summary)
    sources = {name: digest(ROOT/name) for name in REPORT_SOURCES}
    inputs = {key: digest(run/filename) for key, filename in LEDGERS.items()}
    weights = digest(run/"checkpoint/weights.pt")
    require(weights == summary["checkpoint_sha256"], "input weight bytes differ")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(code/"scripts"), str(code)]
    sys.dont_write_bytecode = True
    os.environ.update({"ONEDNN_PRIMITIVE_CACHE_CAPACITY": "16", "DNNL_PRIMITIVE_CACHE_CAPACITY": "16"})
    import torch
    from m2_stream_data import sample_case
    from tetris_vision.losses import PIECE_NAMES, targets_for
    from tetris_vision.preprocess import prepare_image
    require(all(Path(inspect.unwrap(f).__code__.co_filename).resolve().is_relative_to(code)
                for f in (sample_case, targets_for, prepare_image)), "target replay imports are not frozen")
    require(str(torch.__version__) == manifest["torch_version"], "target replay Torch runtime differs")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    records = []
    with torch.no_grad(), (args.output/"targets.jsonl").open("x") as ledger:
        for population, source_rows in recipes.items():
            for recipe in source_rows:
                case = sample_case(recipe["group_index"], recipe["view"], recipe["split"], recipe["base_seed"],
                                   profile_square=recipe["profile_square"])
                require(json.loads(json.dumps(case.record)) == {k: v for k, v in recipe.items() if k != "step"},
                        "source pixels/labels/recipe differ")
                prepared = prepare_image(case.frame.image, max_side=512)
                target = targets_for(case.frame.state, prepared)
                require(prepared.pixels.shape[-2]*prepared.pixels.shape[-1] <= 512**2 and int(target["known"].sum()) == 200,
                        "actual trainer input/target admission differs")
                raw = {key: target[key][0].tolist() for key in ("layer_known", "pose_known", "layer_type", "column", "row", "rotations")}
                row = {"population": population, "step": recipe.get("step"),
                       **{key: recipe[key] for key in ("sample_id", "group_id", "view", "density", "skin")},
                       "recipe_sha256": recipe_digest(recipe), "layers": pack_layers(raw)}
                ledger.write(json.dumps(row)+"\n")
                records.append(row)
                if len(records) % 16 == 0:
                    ledger.flush()
                    write(args.output/"progress.json", {"phase": "target_replay", "records": len(records)})
    require(sources == {name: digest(ROOT/name) for name in REPORT_SOURCES} and
            inputs == {key: digest(run/filename) for key, filename in LEDGERS.items()} and
            digest(run/"checkpoint/weights.pt") == weights, "source/input/weights changed during target replay")
    result = {"status": "passed", "format": "masked-target-exposure-v1", "run": args.run,
              "run_status": summary["status"], "global_steps": summary["steps"], "target_steps": summary["requested_steps"],
              "core_architecture": architecture, "config": manifest["config"], "weights_sha256": weights,
              "source_catalogue_sha256": summary["catalogue_sha256"], "torch_version": str(torch.__version__),
              "type_labels": ["absent", *PIECE_NAMES], "target_records": len(records), "model_forwards": 0, "new_updates": 0,
              "weights_unchanged": True, "job_ceiling_bytes": ceiling, "counts": summarize(records),
              "targets_sha256": digest(args.output/"targets.jsonl"), "input_ledger_sha256": inputs, "sources": sources,
              "scope": "Exact frozen image/resize/masked-target exposure only. Repeated views are dependent; group type exposures may overlap. Allowed rotation counts are multi-label. No model forward, optimizer, learning/accuracy or hidden-state supervision claim."}
    write(args.output/"result.json", result)
    write(args.output/"progress.json", {"status": "passed", "records": len(records)})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
