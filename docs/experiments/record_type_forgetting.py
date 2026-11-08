"""Independently derive checkpoint transitions and local head-gradient geometry."""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_model import require_bounded_job
from type_forgetting_analysis import analyze
from probe_type_forgetting import sha


def read(path, maximum=32*1024**2):
    if path.stat().st_size > maximum:
        raise ValueError("bounded artifact required")
    return json.loads(path.read_text())


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("control", "candidate"), required=True)
    args = parser.parse_args()
    name = f"type-forgetting-{args.arm}-r1"
    run = ROOT / "runs" / name
    result, job = read(run / "result.json"), read(ROOT / "runs" / (name+"-job.json"))
    command = job["command"]
    if (result["status"] != "passed" or result["arm"] != args.arm or result["forwards"] != 544 or
            result["execution_sha256"] != sha(Path(__file__).with_name("probe_type_forgetting.py")) or
            job["status"] != "passed" or job["exit_code"] != 0 or job["work_items"] != 544 or
            job["memory_bytes"] != 640*1024**2 or job["wall_seconds"] != 600 or
            job["system_pressure_guard"] != "GetPerformanceInfo/0.2s" or
            len(command) != 6 or Path(command[1]).resolve() != Path(__file__).with_name("probe_type_forgetting.py").resolve() or
            command[2:5] != ["--arm", args.arm, "--output"] or Path(command[5]).resolve() != run):
        raise ValueError("actual observer/source/native identity differs")
    for relative, value in result["inputs_sha256"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT / "runs") and not path.is_relative_to(ROOT / "docs/evidence"):
            raise ValueError("diagnostic input path outside owned roots")
        if sha(path) != value:
            raise ValueError("diagnostic input changed")
        if path.name == "manifest.json" and "training-snapshots" in path.parts:
            snapshot = read(path)
            expected_factor = 1 if args.arm == "control" else 4
            if (snapshot["contract"]["catalogue_sha256"] != result["source_catalogue_sha256"] or
                    snapshot["contract"].get("type_head_rate_factor") != expected_factor or
                    snapshot["contract"].get("curriculum") != "all-class-fit-v1"):
                raise ValueError("checkpoint source/optimizer contract differs")
    model_run = ROOT / "runs" / f"m2-stream-headrate-{args.arm}-s3"
    model_summary = read(model_run / "summary.json")
    model_scope = read(ROOT / "docs/evidence" / model_run.name / "scope.json")
    pixel_scope = read(ROOT / "docs/evidence" / f"m2-stream-headrate-{args.arm}-errors-r1/scope.json")
    if (model_scope["status"] != "passed" or pixel_scope["status"] != "passed" or
            model_summary["catalogue_sha256"] != result["source_catalogue_sha256"] or
            model_summary["checkpoint_sha256"] != result["weights_sha256"] or
            model_scope["weights_sha256"] != result["weights_sha256"] or
            pixel_scope["weights_sha256"] != result["weights_sha256"] or
            sha(model_run / "checkpoint/weights.pt") != result["weights_sha256"]):
        raise ValueError("curated model identity differs")
    # Independently reconstruct each actual final head logit with a float32
    # dot-product roundoff bound; do not demand exact float64 arithmetic.
    worst = 0.0
    for point in result["observations"]:
        head = point["head"]
        for case in point["cases"]:
            for features, logits in zip(case["features"], case["logits"]):
                for row, bias, actual in zip(head["weight"], head["bias"], logits):
                    products = [a*b for a, b in zip(row, features)]
                    value = math.fsum(products)+bias
                    error = abs(value-actual)
                    tolerance = 65*2**-24/(1-65*2**-24)*(math.fsum(abs(v) for v in products)+abs(bias))+1e-7
                    if not math.isfinite(actual) or error > tolerance:
                        raise ValueError("captured head/logit reconstruction differs")
                    worst = max(worst, error)
    analysis = analyze(result["observations"])
    output = ROOT / "docs/evidence" / name
    output.mkdir(exist_ok=False)
    sources = (Path(__file__), Path(__file__).with_name("type_forgetting_analysis.py"),
               Path(__file__).with_name("probe_type_forgetting.py"), Path(__file__).with_name("test_type_forgetting_analysis.py"))
    copies = [(run / "result.json", "result.json"), (run / "progress.json", "progress.json")]
    copies += [(ROOT / "runs" / (name+suffix), "native"+suffix) for suffix in ("-job.json", ".log", ".stderr.log")]
    copies += [(path, path.name) for path in sources]
    hashes = {}
    for source, name_ in copies:
        target = output / name_
        shutil.copyfile(source, target)
        hashes[name_] = sha(source)
        if sha(target) != hashes[name_]:
            raise ValueError("curated source bytes differ")
    scope = {"status": "passed", "arm": args.arm, "source_catalogue_sha256": result["source_catalogue_sha256"],
             "weights_sha256": result["weights_sha256"], "forwards": result["forwards"],
             "native_peak_bytes": job["peak_job_commit_bytes"], "native_seconds": job["elapsed_seconds"],
             "max_float64_reconstruction_error": worst, "inputs_sha256": result["inputs_sha256"],
             "artifact_sha256": hashes, "analysis": analysis}
    (output / "scope.json").write_text(json.dumps(scope, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": "passed", "arm": args.arm, "initial_correct": analysis["timeline"][0]["correct_types"],
                      "final_correct": analysis["timeline"][-1]["correct_types"]}))


if __name__ == "__main__":
    main()
