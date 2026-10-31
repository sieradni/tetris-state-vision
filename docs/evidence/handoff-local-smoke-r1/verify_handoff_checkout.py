"""Restore a handoff into a fresh GitHub clone and check relocated current assets.

Uses an existing pinned runtime; does not install dependencies, train, or push.
All children inherit the caller's enforced Job and pre-import CPU policy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_model import require_bounded_job
from handoff_bundle import REMOTE, verify

MODELS = ("m2-stream-allclass1024-s3", "m2-stream-interleave1024-s3", "m2-stream-accuracy4096-s10")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def run(command, cwd):
    environment = dict(os.environ, GH_HOST="github.com", GH_PROMPT_DISABLED="1", GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never")
    return subprocess.check_output(command, cwd=cwd, env=environment, text=True).strip()


def smoke(name):
    # Import metadata helpers from this relocated checkout before any frozen package.
    sys.path.insert(0, str(ROOT / "docs/experiments"))
    from type_head_fit_contract import load_bank
    from source_snapshot import verify_source
    study = "ordered" if name == MODELS[0] else "interleaved" if name == MODELS[1] else None
    model_run = ROOT / "runs" / name
    if study:
        bank, model_run, _ = load_bank(study)
    else:
        bank = json.loads((ROOT / "docs/evidence/m2-stream-accuracy4096-features-r1/features.json").read_text())
    code = model_run / "code"
    verify_source(code, bank["source_catalogue_sha256"])
    if digest(model_run / "checkpoint/weights.pt") != bank["weights_sha256"]:
        raise ValueError("relocated image weights differ")
    sys.path[:0] = [str(code / "scripts"), str(code)]
    os.environ.update(ONEDNN_PRIMITIVE_CACHE_CAPACITY="16", DNNL_PRIMITIVE_CACHE_CAPACITY="16")
    import torch
    from PIL import Image
    import tetris_vision.checkpoint as checkpoint
    import tetris_vision.preprocess as preprocess
    if any(not Path(module.__file__).resolve().is_relative_to(code) for module in (checkpoint, preprocess)):
        raise ValueError("smoke test imported an original-machine package instead of relocated archive")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model, manifest = checkpoint.load_checkpoint(model_run / "checkpoint")
    if manifest["torch_version"] != str(torch.__version__):
        raise ValueError("pinned runtime differs")
    model.eval()
    recipe = json.loads((model_run / "training-probe.jsonl").read_text().splitlines()[0])
    expected = [row for row in bank["features"] if row["split"] == "training" and row["sample_id"] == recipe["sample_id"]]
    if [row["role"] for row in expected] != ["active", "ghost"]:
        raise ValueError("paired saved pixel evidence missing")
    with Image.open(model_run / "training-probe" / (recipe["sample_id"]+".png")) as image:
        prepared = preprocess.prepare_image(image, max_side=512)
    with torch.no_grad():
        output = model(prepared.pixels[None], prepared.valid[None])
    if not torch.equal(output["final"]["layer_type"][0], torch.tensor([row["logits"] for row in expected], dtype=torch.float32)):
        raise ValueError("relocated ordinary pixel prediction differs from saved evidence")
    result = {"model": name, "frozen_package_origin_verified": True,
              "ordinary_pixel_forwards": 1, "original_type_logits_exact": True,
              "weights_sha256": manifest["weights_sha256"]}
    if study:
        fitted = ROOT / "runs" / ("type-head-"+study+"-fit-r1")
        fit = json.loads((fitted / "result.json").read_text())
        if digest(fitted / "head-state.pt") != fit["payload_sha256"]:
            raise ValueError("relocated head-fit payload differs")
        state = torch.load(fitted / "head-state.pt", map_location="cpu", weights_only=True)
        raw_weight = (state["weight"]/state["scale"]).float()
        raw_bias = (state["bias"]-(state["weight"]/state["scale"])@state["mean"]).float()
        if not torch.equal(raw_weight, torch.tensor(fit["head"]["weight_raw32"], dtype=torch.float32)) or not torch.equal(
                raw_bias, torch.tensor(fit["head"]["bias_raw32"], dtype=torch.float32)):
            raise ValueError("relocated raw head conversion differs")
        with torch.no_grad():
            logits = torch.nn.functional.linear(torch.tensor([row["features"] for row in bank["features"]], dtype=torch.float32),
                                                raw_weight, raw_bias)
        if not torch.equal(logits, torch.tensor(fit["logits"]["raw32"], dtype=torch.float32)):
            raise ValueError("relocated copied-head vector prediction differs")
        result.update(head_feature_rows=82, copied_head_logits_exact=True,
                      raw32_training_correct=fit["metrics"]["raw32"]["training"]["correct"])
    return result


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--smoke-model", choices=MODELS)
    args = parser.parse_args()
    if args.smoke_model:
        print(json.dumps(smoke(args.smoke_model)), flush=True)
        return
    if args.zip is None or args.checkout is None:
        parser.error("--zip and --checkout required")
    archive = args.zip.resolve()
    info = verify(archive)
    target = args.checkout.resolve()
    if target.exists() or not target.is_relative_to(ROOT / "runs"):
        raise ValueError("fresh ignored verification checkout under runs/ required")
    target.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "-c", "pack.threads=1", "-c", "pack.windowMemory=64m", "-c", "credential.helper=",
         "-c", "credential.helper=!gh auth git-credential", "clone", REMOTE+".git", str(target)], ROOT)
    run(["git", "checkout", "--detach", info["commit"]], target)
    if run(["git", "rev-parse", "HEAD"], target) != info["commit"]:
        raise ValueError("cloned source commit differs")
    restore = json.loads(run([sys.executable, str(target / "scripts/handoff_bundle.py"), "restore", "--zip", str(archive)], target))
    snapshots = []
    for relative, generation in info["latest_snapshots"].items():
        if not relative.startswith("runs/") or ".." in Path(relative).parts or Path(generation).name != generation:
            raise ValueError("snapshot namespace differs")
        folder = target / relative
        pointer = json.loads((folder / "latest-training.json").read_text())
        snapshot = folder / "training-snapshots" / generation
        state = json.loads((snapshot / "manifest.json").read_text())
        if (pointer["generation"] != generation or digest(snapshot / "manifest.json") != pointer["manifest_sha256"] or
                digest(snapshot / "tensors.pt") != state["tensors_sha256"]):
            raise ValueError("relocated latest optimizer snapshot differs")
        snapshots.append({"run": relative, "generation": generation, "payload_sha256": state["tensors_sha256"]})
    results = []
    for model in MODELS:
        results.append(json.loads(run([sys.executable, str(target / "scripts/verify_handoff_checkout.py"), "--smoke-model", model], target)))
    if run(["git", "status", "--porcelain"], target):
        raise ValueError("verification changed cloned source")
    result = {"status": "passed", "repository": REMOTE, "commit": info["commit"], "checkout": str(target),
              "archive_sha256": digest(archive), "archive_members_verified": len(info["files"]),
              "restore": restore, "models": results, "latest_optimizer_snapshots": snapshots, "source_clean": True,
              "runtime_scope": "Existing pinned Windows CPU runtime reused;fresh dependency installation not performed.",
              "research_scope": "Transport/relocated-original-model smoke and saved-head vector replay only;not the planned82-forward conditional pixel intervention or a model-quality gate."}
    receipt = archive.with_suffix(".verification.json")
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
