"""Independent finite metadata/head-metric collector; no Torch imports."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess

from type_head_fit_contract import ROOT, RECIPE, digest, load_bank, metrics, read
from type_head_fit import source_hashes, validate_admission
from probe_model import require_bounded_job


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    run = (ROOT / "runs" / args.name).resolve()
    if run.parent != ROOT / "runs":
        raise ValueError("owned result required")
    result, job = read(run / "result.json"), read(run.parent / (run.name+"-job.json"))
    study, mode = result["study"], result["mode"]
    if mode not in ("profile", "fit") or not run.name.startswith("type-head-"+study+"-"+mode+"-r"):
        raise ValueError("owned fixed diagnostic mode/name required")
    bank, model, identities = load_bank(study)
    sources = source_hashes()
    own_sha = digest(Path(__file__))
    steps, work, wall = (2, 4, 60) if mode == "profile" else (1024, 1024, 120)
    command = job["command"]
    expected = ["--study", study, "--mode", mode, "--output"]
    if (result["status"] != "passed" or result["recipe"] != RECIPE or result["steps"] != steps or
            result["optimizer_updates_executed"] != work or result["sources"] != sources or
            result["inputs_sha256"] != identities or result["model_run"] != model.name or
            result["weights_sha256"] != bank["weights_sha256"] or
            result["source_catalogue_sha256"] != bank["source_catalogue_sha256"] or
            result["core_architecture"] != bank["core_architecture"] or result["model_config"] != bank["config"] or
            result["torch_version"] != read(model / "checkpoint/manifest.json")["torch_version"] or
            result["original_per_image_logits_exact"] is not True or
            not 0 <= result["initial_conditioning_max_abs_error"] <= 1e-10 or
            result["tensor_optimizer_reload_exact"] is not True or
            result["warm_reload_next_update_exact"] != (mode == "profile") or
            digest(run / "head-state.pt") != result["payload_sha256"] or
            result["job_ceiling_bytes"] != 640*1024**2 or
            job["status"] != "passed" or job["exit_code"] != 0 or job["memory_bytes"] != 640*1024**2 or
            job["wall_seconds"] != wall or job["work_items"] != work or
            job["supervisor_script_sha256"] != sources["scripts/run_bounded.py"] or
            job["system_pressure_guard"] != "GetPerformanceInfo/0.2s" or
            len(command) != (8 if mode == "profile" else 10) or
            Path(command[1]).resolve() != Path(__file__).with_name("type_head_fit.py").resolve() or
            command[2:7] != expected or Path(command[7]).resolve() != run):
        raise ValueError("actual input/source/recipe/head/native identity differs")
    if mode == "profile":
        if result["warm_next_record"]["step"] != 3:
            raise ValueError("warm parity update missing")
    else:
        if command[8] != "--admission":
            raise ValueError("explicit warm admission missing")
        admission = validate_admission(Path(command[9]).resolve(), study, identities, sources)
        if result["curve"][:2] != admission["curve"]:
            raise ValueError("native admitted first-two update prefix differs")
    curve_path = run / "curve.jsonl"
    if curve_path.stat().st_size > 1024**2:
        raise ValueError("finite curve byte budget exceeded")
    curve = [json.loads(line) for line in curve_path.read_text().splitlines()]
    if (curve != result["curve"] or len(curve) != steps or [row["step"] for row in curve] != list(range(1, steps+1)) or
            any(not 0 <= row["training_correct_after_update"] <= 64 or
                not math.isfinite(row["gradient_l2"]) or row["gradient_l2"] < 0 or
                not math.isfinite(row["cross_entropy_before_update"]) for row in curve)):
        raise ValueError("actual finite optimizer curve differs")
    rows = bank["features"]
    for name, logits in result["logits"].items():
        if metrics(rows, logits) != result["metrics"][name]:
            raise ValueError("recomputed role/class/loss metrics differ")
    if result["logits"]["initial32"] != [row["logits"] for row in rows]:
        raise ValueError("initial predictions differ from actual feature report")
    head = result["head"]
    for name in ("normalized64", "raw64", "raw32"):
        weight = head["weight_normalized" if name == "normalized64" else "weight_"+name]
        bias = head["bias_normalized" if name == "normalized64" else "bias_"+name]
        if len(weight) != 8 or any(len(vector) != 64 for vector in weight) or len(bias) != 8:
            raise ValueError("actual shared8x64 affine head required")
        for row, logits in zip(rows, result["logits"][name]):
            values = row["features"]
            if name == "normalized64":
                values = [(value-mean)/scale for value, mean, scale in zip(values, head["mean"], head["scale"])]
            for index in range(8):
                products = [a*b for a, b in zip(values, weight[index])]
                reconstructed = math.fsum(products)+bias[index]
                # Float32 native GEMM may fuse/reorder reductions. Bound its roundoff
                # from actual product magnitudes instead of claiming exact double replay.
                tolerance = (65*2**-24/(1-65*2**-24)*
                             (math.fsum(abs(x) for x in products)+abs(bias[index]))+1e-7 if name == "raw32" else
                             1e-9*(1+math.fsum(abs(x) for x in products)+abs(bias[index])))
                if not math.isfinite(reconstructed) or abs(reconstructed-logits[index]) > tolerance:
                    raise ValueError("saved head/logit numerical reconstruction differs")
    if result["raw32_training_fit"] != (result["metrics"]["raw32"]["training"]["correct"] == 64):
        raise ValueError("actual raw float32 training-fit verdict differs")
    if own_sha != digest(Path(__file__)) or sources != source_hashes():
        raise ValueError("collector/execution source changed")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *sources, str(Path(__file__).relative_to(ROOT))],
                   cwd=ROOT, check=True, capture_output=True)
    output = ROOT / "docs/evidence" / run.name
    output.mkdir(exist_ok=False)
    copies = [(run / file, file) for file in ("result.json", "curve.jsonl", "progress.json", "head-state.pt")]
    copies += [(ROOT / "runs" / (run.name+suffix), "native"+suffix) for suffix in ("-job.json", ".log", ".stderr.log")]
    copies += [(ROOT / source, "execution-source/"+source) for source in sources]
    copies += [(Path(__file__), "collector.py")]
    artifact_hashes = {}
    for path, name in copies:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        artifact_hashes[name] = digest(path)
        if digest(destination) != artifact_hashes[name]:
            raise ValueError("raw diagnostic evidence copy differs")
    scope = {"status": "passed", "study": study, "mode": mode, "steps": steps,
             "result_sha256": digest(run / "result.json"), "collector_sha256": own_sha,
             "recipe": RECIPE, "inputs_sha256": identities, "execution_sources": sources,
             "raw32_training_fit": result["raw32_training_fit"], "metrics_recomputed": True,
             "native_seconds": job["elapsed_seconds"], "native_peak_bytes": job["peak_job_commit_bytes"],
             "artifact_sha256": artifact_hashes, "scope": result["scope"]}
    (output / "scope.json").write_text(json.dumps(scope, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "mode": mode, "raw32_training_fit": result["raw32_training_fit"]}))


if __name__ == "__main__":
    main()
