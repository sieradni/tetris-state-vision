"""Replay all supported fit images at each retained 64-step checkpoint.

Pixels/validity are the only model inputs. Saved features/logits support a
separate analytic final-head CE-gradient diagnostic, not an optimizer replay.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_model import require_bounded_job
from source_snapshot import verify_source
from m2_error_analysis import analyze_case
from probe_stream_errors import pack


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    if path.stat().st_size > 2*1024**2:
        raise ValueError("bounded JSON required")
    return json.loads(path.read_text())


def checkpoints(arm):
    result = [(0, ROOT / "runs" / f"m2-stream-headrate-{arm}-s1/initial-weights.pt", None)]
    for segment, low, high in ((1, 64, 128), (2, 192, 576), (3, 640, 1024)):
        run = ROOT / "runs" / f"m2-stream-headrate-{arm}-s{segment}"
        summary = read(run / "summary.json")
        verify_source(run / "code", summary["catalogue_sha256"])
        for step in range(low, high+1, 64):
            manifests = list((run / "training-snapshots").glob(f"step-{step:06d}-*/manifest.json"))
            if len(manifests) != 1:
                raise ValueError("unique retained checkpoint required")
            manifest = read(manifests[0])
            tensor = manifests[0].parent / "tensors.pt"
            if (manifest["step"] != step or manifest["contract"] != summary["recovery_contract"] or
                    sha(tensor) != manifest["tensors_sha256"]):
                raise ValueError("snapshot identity differs")
            for ledger, cursor in manifest["ledgers"].items():
                data = (run / ledger).read_bytes()
                if len(data) < cursor["bytes"] or hashlib.sha256(data[:cursor["bytes"]]).hexdigest() != cursor["sha256"]:
                    raise ValueError("snapshot ledger prefix differs")
            result.append((step, tensor, manifests[0]))
    if [step for step, _, _ in result] != list(range(0, 1025, 64)):
        raise ValueError("complete fixed checkpoint timeline required")
    return result


def main():
    ceiling = require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("control", "candidate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != ROOT / "runs" or output.exists():
        raise ValueError("new owned output required")
    run = ROOT / "runs" / f"m2-stream-headrate-{args.arm}-s3"
    summary = read(run / "summary.json")
    scope = read(ROOT / "docs/evidence" / run.name / "scope.json")
    report_path = ROOT / "docs/evidence" / f"m2-stream-headrate-{args.arm}-errors-r1/result.json"
    report_scope = read(report_path.parent / "scope.json")
    if (summary["status"] != "complete" or summary["steps"] != 1024 or scope["status"] != "passed" or
            scope["weights_sha256"] != summary["checkpoint_sha256"] or report_scope["status"] != "passed" or
            report_scope["weights_sha256"] != summary["checkpoint_sha256"] or
            sha(report_path) != report_scope["artifact_sha256"]["result.json"]):
        raise ValueError("complete independently audited model required")
    code = run / "code"
    verify_source(code, summary["catalogue_sha256"])
    sys.path[:0] = [str(code / "scripts"), str(code)]
    sys.dont_write_bytecode = True
    os.environ.update({"ONEDNN_PRIMITIVE_CACHE_CAPACITY": "16", "DNNL_PRIMITIVE_CACHE_CAPACITY": "16"})
    import torch
    from PIL import Image
    from m2_curriculum import replay_case
    from tetris_vision.checkpoint import load_checkpoint
    from tetris_vision.losses import targets_for
    from tetris_vision.preprocess import prepare_image
    if not Path(replay_case.__code__.co_filename).resolve().is_relative_to(code):
        raise ValueError("fixture implementation is not frozen")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model, manifest = load_checkpoint(run / "checkpoint")
    if manifest["weights_sha256"] != summary["checkpoint_sha256"]:
        raise ValueError("checkpoint identity differs")
    model.eval()
    recipes = [json.loads(line) for line in (run / "training-probe.jsonl").read_text().splitlines()]
    if len(recipes) != 32 or len({r["sample_id"] for r in recipes}) != 32:
        raise ValueError("fixed32 public images required")
    original_report = {r["sample_id"]: r for r in read(report_path)["cases"] if r["split"] == "training"}
    points = checkpoints(args.arm)
    inputs = {str(path.relative_to(ROOT)): sha(path) for _, tensor, meta in points for path in (tensor, meta) if path is not None}
    inputs.update({str(path.relative_to(ROOT)): sha(path) for path in (run / "summary.json", run / "training-probe.jsonl", report_path)})
    output.mkdir()
    observations, features = [], []

    def capture(module, inputs_, result):
        features.append((inputs_[0].detach().clone(), result.detach().clone()))

    handle = model.layer_type.register_forward_hook(capture)
    try:
        with torch.no_grad():
            for step, path, meta in points:
                state = torch.load(path, map_location="cpu", weights_only=True)
                model.load_state_dict(state if meta is None else state["model"], strict=True)
                model.eval()
                head = {"weight": model.layer_type.weight.tolist(), "bias": model.layer_type.bias.tolist()}
                cases = []
                for recipe in recipes:
                    case = replay_case(recipe, split="training", curriculum="all-class-fit-v1", component_capacity=0)
                    stem = run / "training-probe" / recipe["sample_id"]
                    with Image.open(stem.with_suffix(".png")) as image:
                        if image.mode != case.frame.image.mode or image.size != case.frame.image.size or image.tobytes() != case.frame.image.tobytes():
                            raise ValueError("saved fixture pixels differ")
                        prepared = prepare_image(image, max_side=512)
                    if sha(stem.with_suffix(".json")) != recipe["state_sha256"]:
                        raise ValueError("public target file differs")
                    target = targets_for(case.frame.state, prepared)
                    output_ = model(prepared.pixels[None], prepared.valid[None])
                    if len(features) != len(output_["stages"]) or not torch.equal(features[-1][1], output_["final"]["layer_type"]):
                        raise ValueError("actual final type-head observation differs")
                    diagnostic = analyze_case(*pack(output_["final"], target))
                    if step == 1024 and diagnostic != original_report[recipe["sample_id"]]["diagnostics"]:
                        raise ValueError("final checkpoint differs from independent pixel replay")
                    layers = diagnostic["layers"]
                    cases.append({"sample_id": recipe["sample_id"], "group_id": recipe["group_id"],
                                  "features": features[-1][0][0].tolist(), "logits": features[-1][1][0].tolist(),
                                  "roles": {role: {key: layers[role][key] for key in
                                            ("target_known", "target_type", "target_pose_known", "type_correct", "exact")}
                                            for role in ("active", "ghost")}})
                    features.clear()
                observations.append({"step": step, "head": head, "cases": cases})
                (output / "progress.json").write_text(json.dumps({"step": step, "forwards": len(observations)*32})+"\n")
    finally:
        handle.remove()
    if len(observations) != 17 or any(sha(ROOT / path) != value for path, value in inputs.items()):
        raise ValueError("timeline incomplete or inputs changed")
    proof = {"status": "passed", "arm": args.arm, "forwards": 544, "job_ceiling_bytes": ceiling,
             "source_catalogue_sha256": summary["catalogue_sha256"], "weights_sha256": summary["checkpoint_sha256"],
             "inputs_sha256": inputs, "execution_sha256": sha(Path(__file__)), "observations": observations,
             "scope": "Supported fixed synthetic training images; captured final type-head features/logits. No training, optimizer-step, independent generalization or real-image claim."}
    (output / "result.json").write_text(json.dumps(proof, separators=(",", ":"), allow_nan=False)+"\n")
    print(json.dumps({"status": "passed", "arm": args.arm, "checkpoints": 17, "forwards": 544}))


if __name__ == "__main__":
    main()
