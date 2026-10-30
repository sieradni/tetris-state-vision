"""Observe existing type-head inputs during unchanged ordinary saved-probe replay.

Use only after the fixed larger baseline completes. No training or model edits.
The existing ordinary worker replays pixels/labels, frozen metrics and weights;
a global observer returns None and records the actual layer_type Linear input.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_model import require_bounded_job
import probe_stream_errors as ordinary
from layer_feature_analysis import analyze


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ceiling = require_bounded_job()
    if (Path(inspect.getfile(analyze)).resolve() != Path(__file__).with_name("layer_feature_analysis.py").resolve() or
            Path(ordinary.__file__).resolve() != ROOT / "scripts/probe_stream_errors.py"):
        raise ValueError("feature observer/analyzer execution origin differs")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=("m2-stream-accuracy4096-s10", "m2-stream-allclass1024-s3",
                                          "m2-stream-interleave1024-s3"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Enforce the declared study order, independent of which artifact is inspected.
    final = ordinary.read(ROOT / "runs/m2-stream-accuracy4096-s10/summary.json")
    ordinary.validate_completion(final, ordinary.read(ROOT / "runs/m2-stream-accuracy4096-s10-job.json"))
    final_scope = ordinary.read(ROOT / "docs/evidence/m2-stream-accuracy4096-s10/scope.json")
    if final["status"] != "complete" or final["steps"] != 4096 or final["requested_steps"] != 4096:
        raise ValueError("finish original fixed4096 study before feature diagnostics")
    if (final_scope["status"] != "passed" or final_scope["steps"] != 4096 or
            final_scope["weights_sha256"] != final["checkpoint_sha256"]):
        raise ValueError("independent complete4096 replay required")
    run = ROOT / "runs" / args.run
    summary = ordinary.read(run / "summary.json")
    larger = args.run == "m2-stream-accuracy4096-s10"
    if (summary["status"] != "complete" or summary["config"].get("component_capacity", 0) or
            summary["config"].get("layer_evidence", False) or
            summary["steps"] != (4096 if larger else 1024) or summary["requested_steps"] != summary["steps"] or
            len(summary["final_training_probe"]["cases"]) != (9 if larger else 32) or
            len(summary["development"]["cases"]) != (18 if larger else 9)):
        raise ValueError("completed original M2 baseline without optional heads required")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "runs") or output.exists():
        raise ValueError("new owned output required")
    paths = [Path(__file__).resolve(), Path(__file__).with_name("layer_feature_analysis.py"),
             ROOT / "scripts/run_bounded.py"] + [ROOT / "scripts" / name for name in (
                 "probe_stream_errors.py", "m2_error_analysis.py", "m3_error_analysis.py",
                 "record_stream_run.py", "source_snapshot.py", "probe_model.py")]
    sources = {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}
    os.environ.update({"ONEDNN_PRIMITIVE_CACHE_CAPACITY": "16", "DNNL_PRIMITIVE_CACHE_CAPACITY": "16"})
    import torch
    pending, observations = [], []
    head = {}

    def observe(module, inputs, result):
        if isinstance(module, torch.nn.Linear) and tuple(result.shape) == (1, 2, 8):
            if len(pending) >= 6 or inputs[0].shape[-1] != summary["config"]["dimension"]:
                raise ValueError("unexpected type-head observation shape/count")
            pending.append((module, inputs[0].detach().clone(), result.detach().clone()))
        if type(module).__name__ == "TetrisStateNet":
            code = run / "code"
            if (not Path(inspect.getfile(type(module))).resolve().is_relative_to(code) or
                    len(pending) != summary["config"]["blocks"] or
                    len(pending) != len(result["stages"]) or len(observations) >= 68):
                raise ValueError("frozen model/stage/forward contract differs")
            for event, stage in zip(pending, result["stages"]):
                reader, features, logits = event
                if reader is not module.layer_type or not torch.equal(logits, stage["layer_type"]):
                    raise ValueError("observed module is not actual stage type head")
                if not torch.equal(torch.nn.functional.linear(features, reader.weight, reader.bias), logits):
                    raise ValueError("captured input fails exact linear-logit reconstruction")
            if not head:
                head.update(weight=module.layer_type.weight.detach().tolist(), bias=module.layer_type.bias.detach().tolist())
            _, features, logits = pending[-1]
            observations.append({"features": features[0].tolist(), "logits": logits[0].tolist()})
            pending.clear()
        return None  # Torch hook cannot replace any module output.

    handle = torch.nn.modules.module.register_module_forward_hook(observe)
    previous = sys.argv
    try:
        sys.argv = [str(ROOT / "scripts/probe_stream_errors.py"), "--run", args.run, "--output", str(output)]
        ordinary.main()
    finally:
        sys.argv = previous
        handle.remove()
    report = json.loads((output / "result.json").read_text())
    if pending or len(observations) != report["forwards"] or report["forwards"] != (27 if larger else 41) or not head:
        raise ValueError("feature/ordinary forward populations differ")
    rows = []
    for case, observation in zip(report["cases"], observations):
        for index, role in enumerate(("active", "ghost")):
            diagnostic = case["diagnostics"]["layers"][role]
            logits = observation["logits"][index]
            if max(range(8), key=logits.__getitem__) != diagnostic["predicted_type"]:
                raise ValueError("actual feature order differs from ordinary predictions")
            rows.append({"split": case["split"], "sample_id": case["sample_id"], "group_id": case["group_id"],
                         "role": role, "known": diagnostic["target_known"], "target_type": diagnostic["target_type"],
                         "features": observation["features"][index], "logits": logits})
    analysis = analyze(rows)
    if sources != {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}:
        raise ValueError("feature diagnostic sources changed")
    for path in paths:
        destination = output / "execution-source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if sha(destination) != sources[path.relative_to(ROOT).as_posix()]:
            raise ValueError("execution source byte copy differs")
    proof = {"status": "passed", "run": args.run, "job_ceiling_bytes": ceiling,
             "global_steps": report["global_steps"], "target_steps": report["target_steps"],
             "weights_sha256": report["weights_sha256"], "source_catalogue_sha256": report["source_catalogue_sha256"],
             "config": report["config"], "core_architecture": report["core_architecture"],
             "forwards": report["forwards"], "ordinary_report_sha256": sha(output / "result.json"),
             "automatic_metrics_reproduced": report["automatic_metrics_reproduced"],
             "weights_unchanged": report["weights_unchanged"], "exact_linear_reconstruction": True,
             "observer_returns_none": True, "sources": sources, "type_head": head,
             "analysis": analysis, "features": rows}
    with (output / "features.json").open("x", encoding="utf-8") as stream:
        json.dump(proof, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": "passed", "forwards": report["forwards"], "feature_rows": len(rows)}))


if __name__ == "__main__":
    main()
