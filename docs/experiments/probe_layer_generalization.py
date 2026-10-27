"""Predeclared groups100..163, three views; source-frozen read-only pilot probe."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
from probe_model import require_bounded_job
from probe_stream_errors import pack, require_frozen_callables
from m2_error_analysis import analyze_case
from record_stream_run import checkpoint_architecture, validate_completion
from source_snapshot import verify_source


def read(path):
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("JSON budget")
    return json.loads(path.read_text())


def sha(path):
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError("hash budget")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ceiling = require_bounded_job()
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {path: sha(ROOT / path) for path in ("scripts/probe_stream_errors.py",
               "scripts/m2_error_analysis.py", "scripts/m3_error_analysis.py", "scripts/probe_model.py",
               "scripts/record_stream_run.py", "scripts/source_snapshot.py", "scripts/run_bounded.py")}
    summaries, runs = {}, {}
    for label, name, enabled in (("control", args.control, False), ("candidate", args.candidate, True)):
        run = (ROOT / "runs" / name).resolve()
        if run.parent != ROOT / "runs":
            raise ValueError("owned run")
        summary = read(run / "summary.json")
        validate_completion(summary, read(ROOT / "runs" / (name + "-job.json")))
        if (summary["steps"] != 768 or summary["config"]["layer_evidence"] is not enabled or
                summary["config"]["component_capacity"] != 0):
            raise ValueError("complete768 pilot with actual option required")
        verify_source(run / "code", summary["catalogue_sha256"])
        summaries[label], runs[label] = summary, run
    configs = [dict(summaries[label]["config"]) for label in ("control", "candidate")]
    for config in configs:
        config.pop("layer_evidence")
    if (configs[0] != configs[1] or summaries["control"]["catalogue_sha256"] != summaries["candidate"]["catalogue_sha256"] or
            (runs["control"] / "training.jsonl").read_bytes() != (runs["candidate"] / "training.jsonl").read_bytes()):
        raise ValueError("matched frozen models/data required")
    code = runs["control"] / "code"
    sys.path[:0] = [str(code / "scripts"), str(code)]
    sys.dont_write_bytecode = True
    os.environ.update({"ONEDNN_PRIMITIVE_CACHE_CAPACITY": "16", "DNNL_PRIMITIVE_CACHE_CAPACITY": "16"})
    import torch
    from m2_stream_data import sample_case
    from tetris_vision.checkpoint import load_checkpoint
    from tetris_vision.losses import targets_for, diagnostic_metrics
    from tetris_vision.preprocess import prepare_image
    require_frozen_callables(code, sample_case, targets_for, diagnostic_metrics, prepare_image, load_checkpoint)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    models, weights, identities = {}, {}, {}
    for label, run in runs.items():
        model, manifest = load_checkpoint(run / "checkpoint")
        checkpoint_architecture(summaries[label], manifest)
        if sha(run / "checkpoint/weights.pt") != summaries[label]["checkpoint_sha256"]:
            raise ValueError("actual weights differ")
        models[label] = model.eval()
        weights[label] = {key: tensor.clone() for key, tensor in model.state_dict().items()}
        identities[label] = {"run": run.name, "weights_sha256": manifest["weights_sha256"], "config": manifest["config"]}
    args.output.mkdir(parents=True, exist_ok=False)
    records = {label: [] for label in models}
    recipes = []
    with torch.no_grad():
        for group in range(100, 164):
            for view in range(3):
                case = sample_case(group, view, "development", 20261026)
                recipes.append(case.record)
                prepared = prepare_image(case.frame.image, max_side=512)
                for label, model in models.items():
                    target = targets_for(case.frame.state, prepared, layer_evidence=label == "candidate")
                    if int(target["known"].sum()) != 200 or not bool(target["registered"].all()):
                        raise ValueError("fully supported visible target required")
                    output = model(prepared.pixels[None], prepared.valid[None])
                    prediction, truth = pack(output["final"], target)
                    diagnostic = analyze_case(prediction, truth)
                    metrics = diagnostic_metrics(output, target)
                    if metrics["exact_locked_board"] != diagnostic["locked"]["exact_board"]:
                        raise ValueError("independent board metric disagreement")
                    row = {"sample_id": case.record["sample_id"], "group_id": case.record["group_id"],
                           "prediction": prediction, "target": truth, "metrics": metrics, "diagnostics": diagnostic}
                    if label == "candidate":
                        row["dense_prediction"] = output["final"]["layer_cells"][0].tolist()
                        row["dense_target"] = {key: value[0].tolist() for key, value in target["layer_cells"].items()}
                    records[label].append(row)
            if (group - 99) % 16 == 0:
                progress = {"groups": group - 99, "images_per_model": len(recipes), "forwards": 2 * len(recipes)}
                (args.output / "progress.json").write_text(json.dumps(progress) + "\n")
                print(json.dumps(progress), flush=True)
    results = {}
    for label, rows in records.items():
        if not all(torch.equal(value, models[label].state_dict()[key]) for key, value in weights[label].items()):
            raise ValueError("read-only inference changed tensors")
        if sha(runs[label] / "checkpoint/weights.pt") != identities[label]["weights_sha256"]:
            raise ValueError("read-only inference changed artifact")
        totals = {"images": len(rows), "ancestry_groups": len({row["group_id"] for row in rows}),
                  "populations": {}, "roles_by_target_class": {}, "pose_rows_by_role": {},
                  "mean_geometry_error_cells": sum(row["metrics"]["mean_geometry_error_cells"] for row in rows) / len(rows)}
        for population in ("empty", "nonempty"):
            cells = [row["diagnostics"]["locked"] for row in rows if row["diagnostics"]["locked"]["population"] == population]
            totals["populations"][population] = {"images": len(cells), "exact_boards": sum(item["exact_board"] for item in cells),
                **{key: sum(item[key] for item in cells) for key in ("known_cells", "true_positive", "true_negative", "false_positive", "false_negative")}}
        for index, role in enumerate(("active", "ghost")):
            classes, pose_rows = {}, Counter()
            for row in rows:
                item = row["diagnostics"]["layers"][role]
                key = str(item["target_type"]) if item["target_known"] else "unknown"
                counts = classes.setdefault(key, Counter())
                counts["images"] += 1
                counts["exact"] += int(item["exact"] is True)
                counts["type_errors"] += int(item["type_correct"] is False)
                counts["pose_known"] += int(item["target_pose_known"])
                for dimension in ("column", "row", "rotation"):
                    counts[dimension + "_errors"] += int(item[dimension + "_correct"] is False)
                if item["target_pose_known"]:
                    pose_rows[str(row["target"]["layers"][index]["row"])] += 1
            totals["roles_by_target_class"][role] = classes
            totals["pose_rows_by_role"][role] = pose_rows
        for key in rows[0]["metrics"]:
            if key.startswith("dense_"):
                totals[key] = sum(row["metrics"][key] for row in rows)
        results[label] = totals
        data = "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
        if len(data.encode()) > 32 * 1024**2:
            raise ValueError("raw prediction budget")
        (args.output / (label + ".jsonl")).write_text(data)
    if sources != {path: sha(ROOT / path) for path in sources}:
        raise ValueError("diagnostic source changed")
    (args.output / "recipes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in recipes))
    shutil.copyfile(__file__, args.output / "analysis-source.py")
    files = ("control.jsonl", "candidate.jsonl", "recipes.jsonl", "analysis-source.py")
    result = {"status": "passed", "population": "development groups100..163, views0..2, seed20261026",
              "source_catalogue_sha256": summaries["control"]["catalogue_sha256"], "models": identities,
              "images_per_model": 192, "forwards": 384, "job_ceiling_bytes": ceiling,
              "type_index": ["absent", "I", "O", "T", "S", "Z", "J", "L"], "results": results,
              "sources": sources, "artifact_sha256": {name: sha(args.output / name) for name in files},
              "all_model_tensors_unchanged": True,
              "scope": "Disjoint-ancestry synthetic diagnostic within the same generator. Masks govern support; queue/hold and real accuracy untested. Related views are not independent trials. No model-selection or whole-state release claim."}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "passed", "forwards": 384}), flush=True)


if __name__ == "__main__":
    main()
