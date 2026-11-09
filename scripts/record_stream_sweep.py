"""Curate a bounded read-only occupancy sweep into frozen evidence; no quality claim."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from occupancy_sweep import evaluate, sweep_verdict, threshold_grid
from probe_model import require_bounded_job
from record_stream_run import checkpoint_architecture, require_owned_run
from source_snapshot import verify_source
from sweep_stream_occupancy import CEILING_BYTES, SWEEP_SOURCES

ROOT = Path(__file__).resolve().parents[1]
NATIVE_SIDECARS = ("-job.json", ".log", ".stderr.log")
REFERENCE = ROOT/"docs/evidence/m3-percell-balance-r1-errors-r1/result.json"


def read(path, maximum=2*1024**2):
    if path.stat().st_size > maximum:
        raise ValueError("sweep evidence JSON byte budget exceeded")
    return json.loads(path.read_text())


def digest(path):
    value, consumed = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            consumed += len(block)
            if consumed > 64*1024**2:
                raise ValueError("sweep evidence artifact byte budget exceeded")
            value.update(block)
    return value.hexdigest()


def validate(result, job, summary, manifest, *, name):
    """Recompute the whole receipt from its own per-cell logits; no saved number is trusted."""
    if (result["status"] != "passed" or job["status"] != "passed" or job["exit_code"] != 0 or
            job["work_items"] != result["forwards"] or result["forwards"] != len(result["cases"]) or
            job["memory_bytes"] != result["job_ceiling_bytes"] or result["job_ceiling_bytes"] != CEILING_BYTES):
        raise ValueError("complete finite sweep gate required")
    command = job["command"]
    profile_dir = Path(command[7]) if len(command) == 8 else Path(".")
    if (len(command) != 8 or Path(command[1]).resolve() != ROOT/"scripts/sweep_stream_occupancy.py" or
            command[2] != "--run" or command[3] != result["run"] or command[4] != "--output" or
            Path(command[5]).resolve() != ROOT/"runs"/name or command[6] != "--admission" or
            profile_dir.is_absolute() or profile_dir.parts[:1] != ("runs",) or
            not (ROOT/profile_dir).is_dir() or profile_dir.name != result["admission"]["run"]):
        raise ValueError("native sweep command/output identity differs")
    profile = read(ROOT/profile_dir/"summary.json")
    if (profile.get("status") != "complete" or profile.get("weights_sha256") != result["weights_sha256"] or
            profile.get("config") != result["config"] or
            profile.get("admission_ceiling_bytes") != CEILING_BYTES or
            profile.get("cases") != result["admission"]["cases"] or
            profile.get("thresholds") != [row["threshold"] for row in result["deciding"]["rows"]]):
        raise ValueError("sweep admission summary differs")
    architecture = checkpoint_architecture(summary, manifest)
    if (result["run_status"] != summary["status"] or
            result["weights_sha256"] != summary["checkpoint_sha256"] or
            manifest["weights_sha256"] != result["weights_sha256"] or
            result["config"] != manifest["config"] or result["core_architecture"] != architecture or
            result["component_capacity"] != summary["config"].get("component_capacity", 0) or
            result.get("curriculum", "stream-v1") != summary.get("curriculum", "stream-v1") or
            result["admission"]["ceiling_bytes"] != CEILING_BYTES or
            result["admission"]["weights_sha256"] != result["weights_sha256"]):
        raise ValueError("sweep checkpoint/partial-status identity differs")
    thresholds = list(threshold_grid())
    if (result["grid"] != {"low": thresholds[0], "high": thresholds[-1],
                           "step": round(thresholds[1]-thresholds[0], 2), "count": len(thresholds)} or
            any(len(result[key]["thresholds"]) != len(thresholds) for key in ("deciding", "development"))):
        raise ValueError("sweep grid differs from the predeclaration")
    splits = {}
    for case in result["cases"]:
        if (type(case["known_cells"]) is not int or case["known_cells"] != len(case["logits"]) or
                len(case["logits"]) != len(case["occupied"]) or
                any(type(value) is not int or value not in (0, 1) for value in case["occupied"])):
            raise ValueError("sweep per-cell logits/labels differ")
        splits.setdefault(case["split"], []).append(case)
    if set(splits) != {"training", "development"}:
        raise ValueError("sweep must cover both splits")
    if len(splits["training"]) != 9 or len(splits["development"]) != 18:
        raise ValueError("sweep population differs from the canonical probe set")
    for key, split in (("deciding", "training"), ("development", "development")):
        if evaluate(splits[split], thresholds) != result[key]:
            raise ValueError("sweep aggregate does not re-derive from its own cells")
    if result["verdict"] != sweep_verdict(result["deciding"]):
        raise ValueError("sweep verdict does not re-derive from its own counts")
    if result["sources"] != {name: digest(ROOT/name) for name in SWEEP_SOURCES}:
        raise ValueError("sweep execution source changed")
    return architecture


def reference_join(result):
    """The grid's zero row must reproduce the replay's ordinary locked counts."""
    if not REFERENCE.exists():
        return None
    reference = read(REFERENCE)["totals"]
    zero = next(row for row in result["deciding"]["rows"] if row["threshold"] == 0.0)
    expected = reference["training"]["locked_populations"]["nonempty"]
    if (zero["true_positive"], zero["false_positive"], zero["false_negative"],
            zero["true_negative"]) != (expected["true_positive"], expected["false_positive"],
                                       expected["false_negative"], expected["true_negative"]):
        raise ValueError("sweep zero row does not join the replay's locked counts")
    return {"reference": "docs/evidence/m3-percell-balance-r1-errors-r1/result.json",
            "threshold": 0.0, "training_nonempty": {"true_positive": zero["true_positive"],
                                                    "false_positive": zero["false_positive"],
                                                    "false_negative": zero["false_negative"]}}


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    require_owned_run(args.name)
    run = ROOT/"runs"/args.name
    result = read(run/"result.json")
    require_owned_run(result["run"])
    source = ROOT/"runs"/result["run"]
    summary, manifest = read(source/"summary.json"), read(source/"checkpoint/manifest.json")
    source_job = read(ROOT/"runs"/(result["run"]+"-job.json"))
    job = read(ROOT/"runs"/(args.name+"-job.json"))
    if (job["supervisor_script_sha256"] != digest(ROOT/"scripts/run_bounded.py") or
            job["system_pressure_guard"] != "GetPerformanceInfo/0.2s"):
        raise ValueError("native supervisor source/guard differs")
    architecture = validate(result, job, summary, manifest, name=args.name)
    join = reference_join(result)
    verify_source(source/"code", summary["catalogue_sha256"])
    if digest(source/"checkpoint/weights.pt") != result["weights_sha256"]:
        raise ValueError("swept weight bytes differ")
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *SWEEP_SOURCES, Path(__file__).relative_to(ROOT).as_posix()],
                   cwd=ROOT, check=True, capture_output=True)
    evidence = ROOT/"docs/evidence"/args.name
    evidence.mkdir(exist_ok=False)
    profile = ROOT/"runs"/result["admission"]["run"]
    selected = [(run/"result.json", "result.json"), (run/"progress.json", "progress.json"),
                (source/"summary.json", "input-summary.json"),
                (source/"code/source.json", "source-manifest.json"),
                (source/"checkpoint/manifest.json", "checkpoint-manifest.json"),
                (ROOT/"runs"/(result["run"]+"-job.json"), "input-run-job.json"),
                (profile/"summary.json", "admission-summary.json")]
    selected += [(ROOT/"runs"/(args.name+suffix), "native"+suffix) for suffix in NATIVE_SIDECARS]
    hashes = {}
    for path, name in selected:
        hashes[name] = digest(path)
        shutil.copyfile(path, evidence/name)
        if digest(evidence/name) != hashes[name]:
            raise ValueError("sweep evidence copy differs")
    scope = {"status": "passed", "source_base_commit": base, "core_architecture": architecture,
             "run": result["run"], "weights_sha256": result["weights_sha256"],
             "grid": result["grid"], "verdict": result["verdict"],
             "reference_join": join,
             "auc": {"training": result["deciding"]["auc"], "development": result["development"]["auc"],
                     "chance": result["deciding"]["chance_auc"]},
             "best": {"training": result["deciding"]["best"], "development": result["development"]["best"]},
             "native_peak_job_commit_bytes": job["peak_job_commit_bytes"],
             "native_elapsed_seconds": job["elapsed_seconds"],
             "diagnostic_sources": result["sources"], "artifact_sha256": hashes,
             "curator_sha256": digest(Path(__file__)), "scope": result["scope"]}
    (evidence/"scope.json").write_text(json.dumps(scope, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "verdict": result["verdict"], "forwards": result["forwards"],
                      "scope": str(evidence/"scope.json")}))


if __name__ == "__main__":
    main()
