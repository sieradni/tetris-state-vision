"""Oracle-lattice intervention on saved synthetic probes, not parser accuracy.

Labels enter only an explicitly separate diagnostic forward. Production model,
weights and automatic parser are unchanged. Oracle effects are conditional on
this trained model; they are not a capacity ceiling or real-image benchmark.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import time

from probe_model import require_bounded_job
from source_snapshot import verify_source
from geometry_contract import REPORT_SOURCES, input_contract, summarize

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    if path.stat().st_size > 1024**2:
        raise ValueError("geometry JSON byte budget exceeded")
    return json.loads(path.read_text())


def main():
    ceiling = require_bounded_job()
    sources = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in REPORT_SOURCES}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true", help="explicit saved controlled partial; preserve its original status")
    args = parser.parse_args()
    if not re.fullmatch(r"m2-stream-[a-z0-9-]{1,40}", args.run):
        raise ValueError("owned source-frozen stream run required")
    run = ROOT/"runs"/args.run
    summary = read(run/"summary.json")
    source_job = read(ROOT/"runs"/(args.run+"-job.json"))
    checkpoint = read(run/"checkpoint/manifest.json")
    architecture = input_contract(summary, source_job, checkpoint, allow_partial=args.allow_partial)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    code = run/"code"
    verify_source(code, summary["catalogue_sha256"])
    sys.path[:0] = [str(code/"scripts"), str(code)]
    sys.dont_write_bytecode = True
    os.environ.update({"ONEDNN_PRIMITIVE_CACHE_CAPACITY": "16", "DNNL_PRIMITIVE_CACHE_CAPACITY": "16"})
    import torch
    from PIL import Image
    from m2_stream_data import sample_case
    from tetris_vision.checkpoint import file_digest, load_checkpoint
    from tetris_vision.losses import diagnostic_metrics, targets_for
    from tetris_vision.preprocess import prepare_image
    import tetris_vision.model as model_module
    if not Path(model_module.__file__).resolve().is_relative_to(code):
        raise ValueError("geometry probe model import is not frozen")
    if any(not Path(inspect.unwrap(function).__code__.co_filename).resolve().is_relative_to(code)
           for function in (sample_case, load_checkpoint, diagnostic_metrics, targets_for, prepare_image)):
        raise ValueError("geometry probe callable implementation is not frozen")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model, manifest = load_checkpoint(run/"checkpoint")
    model.eval()
    if manifest != checkpoint or manifest["weights_sha256"] != summary["checkpoint_sha256"]:
        raise ValueError("geometry probe weight identity differs")
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    original_decode = model_module.decode_lattice
    records, forwards = [], 0

    def score(output, target):
        result = diagnostic_metrics(output, target)
        result.pop("mean_geometry_error_cells")  # Oracle zero would be tautological.
        right, known = 0, 0
        final = output["final"]
        for layer in range(2):
            if not bool(target["layer_known"][0, layer]):
                continue
            known += 1
            correct = (bool(final["layer_known"][0, layer] > 0) and
                       int(final["layer_type"][0, layer].argmax()) == int(target["layer_type"][0, layer]))
            if bool(target["pose_known"][0, layer]):
                correct = (correct and int(final["pose_column"][0, layer].argmax()) == int(target["column"][0, layer]) and
                           int(final["pose_row"][0, layer].argmax()) == int(target["row"][0, layer]) and
                           bool(target["rotations"][0, layer, final["pose_rotation"][0, layer].argmax()]))
            right += int(correct)
        return {**result, "exact_known_layers": right, "known_layers": known}

    with torch.no_grad():
        for split, field in (("training", "final_training_probe"), ("development", "development")):
            recipes = [json.loads(line) for line in (run/(split+"-probe.jsonl")).read_text().splitlines()]
            expected = {row["sample_id"]: row for row in summary[field]["cases"]}
            if not 1 <= len(recipes) <= (9 if split == "training" else 18):
                raise ValueError("finite27-probe/54-forward budget exceeded")
            for recipe in recipes:
                case = sample_case(recipe["group_index"], recipe["view"], split, recipe["base_seed"])
                if json.loads(json.dumps(case.record)) != recipe:
                    raise ValueError("geometry probe recipe/pixel/label identity differs")
                stem = run/(split+"-probe")/recipe["sample_id"]
                with Image.open(stem.with_suffix(".png")) as image:
                    if image.mode != case.frame.image.mode or image.size != case.frame.image.size or image.tobytes() != case.frame.image.tobytes():
                        raise ValueError("saved geometry probe image differs")
                    prepared = prepare_image(image, max_side=512)
                if file_digest(stem.with_suffix(".json")) != recipe["state_sha256"]:
                    raise ValueError("saved geometry probe annotation differs")
                target_kwargs = {"layer_evidence": True} if manifest["config"].get("layer_evidence", False) else {}
                target = targets_for(case.frame.state, prepared, **target_kwargs)
                if int(target["known"].sum()) != 200 or not bool(target["registered"].all()):
                    raise ValueError("oracle geometry requires fully supported synthetic board")
                automatic = model(prepared.pixels[None], prepared.valid[None])
                baseline = score(automatic, target)
                forwards += 1
                saved = expected[recipe["sample_id"]]
                if any(baseline[key] != saved[key] for key in baseline):
                    raise ValueError("automatic predictions do not reproduce stored artifact metrics")
                try:
                    model_module.decode_lattice = lambda raw, columns=10, rows=20: target["lattice"]
                    oracle = score(model(prepared.pixels[None], prepared.valid[None]), target)
                    forwards += 1
                finally:
                    model_module.decode_lattice = original_decode
                records.append({"split": split, "sample_id": recipe["sample_id"], "group_id": recipe["group_id"],
                                "density": recipe["density"], "skin": recipe["skin"], "view": recipe["view"],
                                "automatic": baseline, "oracle_geometry": oracle})
                (args.output/"progress.json").write_text(json.dumps({"forwards": forwards, "sample_id": recipe["sample_id"]})+"\n")
    if forwards > 54 or not all(torch.equal(value, model.state_dict()[name]) for name, value in weights.items()) or model_module.decode_lattice is not original_decode:
        raise ValueError("geometry intervention mutated weights/decoder or exceeded budget")
    totals = summarize(records, layer_evidence=manifest["config"].get("layer_evidence", False))
    if sources != {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in REPORT_SOURCES}:
        raise ValueError("geometry diagnostic sources changed during execution")
    result = {"status": "passed", "forwards": forwards, "source_catalogue_sha256": summary["catalogue_sha256"],
              "diagnostic_format_version": 2, "run": args.run, "run_status": summary["status"],
              "global_steps": summary["steps"], "target_steps": summary["requested_steps"],
              "core_architecture": architecture, "config": manifest["config"], "component_capacity": 0,
              "weights_sha256": summary["checkpoint_sha256"], "model_kind": summary["model_kind"],
              "job_ceiling_bytes": ceiling, "automatic_metrics_reproduced": True, "weights_unchanged": True,
              "oracle_geometry_provided": True, "automatic_accuracy_claim_allowed": False,
              "totals": totals, "cases": records, "elapsed_seconds": time.monotonic()-started,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "sources": sources,
              "scope": "Conditional oracle geometry intervention on saved synthetic probes. Labels deliberately enter the diagnostic forward; not deployment accuracy, a capacity ceiling, training, or a production parser correction."}
    (args.output/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}), flush=True)


if __name__ == "__main__":
    main()
