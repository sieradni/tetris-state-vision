"""Drive one seed's three-arm replicate block, each child under the bounded runner.

`docs/M3_COMPUTE_REPLICATION.md` fixes the arms, seeds, envelope and rule. This driver launches only
what that document declares, one heavy child at a time, and refuses to reuse any output or receipt:
profile -> arm -> sweep profile -> sweep for each of `base128`, `budget192`, `cosine128`. It writes a
per-seed manifest binding every job receipt to the arm summary it produced, and stops on the first
child that is not `passed` so a partial block is never reported as a finished one.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs"
PYTHON = str(Path(sys.executable).resolve())

SEEDS = (20261101, 20261102, 20261103)
ARMS = {"base128": {"steps": 128, "schedule": []},
        "budget192": {"steps": 192, "schedule": []},
        "cosine128": {"steps": 128, "schedule": ["--lr-final", "0.0001"]}}
COMMON = ("--model", "accuracy", "--component-capacity", "64", "--component-identity-factor", "4",
          "--component-identity-readout", "positional", "--component-endpoint-term", "length",
          "--occupancy-factor", "4", "--occupancy-positive-weight", "4", "--curriculum", "stream-v1",
          "--capture-max-side", "512", "--learning-rate", "0.001", "--evaluate-every", "64",
          "--checkpoint-every", "32", "--development-groups", "6", "--wall-seconds", "1800")
ARM_CEILING_MIB, SWEEP_CEILING_MIB, PARENT_CEILING_MIB = 2048, 640, 256
SWEEP_CASES = 54
DRIVER_SOURCES = tuple("scripts/"+name for name in
                       ("run_compute_replication.py", "run_bounded.py", "train_m2_stream.py",
                        "sweep_stream_occupancy.py"))


def digest(path, maximum=64*1024**2):
    value, consumed = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            consumed += len(block)
            if consumed > maximum:
                raise ValueError("driver source byte budget exceeded")
            value.update(block)
    return value.hexdigest()


def run_name(arm, seed):
    return f"m3-stream-repl-{arm}-s{seed}"


def launch(label, memory_mib, wall_seconds, work_items, worker):
    """One bounded child, with its receipt path claimed before launch and verified after."""
    receipt = RUNS/(label+"-job.json")
    log_path = RUNS/(label+".log")
    for path in (receipt, log_path):
        if path.exists():
            raise SystemExit(f"declared output already exists; refusing to reuse: {path}")
    command = [PYTHON, str(ROOT/"scripts/run_bounded.py"), "--memory-mib", str(memory_mib),
               "--wall-seconds", str(wall_seconds), "--work-items", str(work_items),
               "--receipt", str(receipt), "--", PYTHON, *worker]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with log_path.open("xb") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log, creationflags=flags)
        code = process.wait()
    data = json.loads(receipt.read_text())
    if code != 0 or data.get("status") != "passed":
        raise SystemExit(f"{label} did not pass: exit {code}, status {data.get('status')}")
    print(json.dumps({"job": label, "status": data["status"],
                      "peak": data.get("peak_job_commit_bytes"),
                      "elapsed": round(data.get("elapsed_seconds", 0.), 1)}), flush=True)
    return {"receipt": str(receipt.relative_to(ROOT)), "status": data["status"],
            "peak_job_commit_bytes": data.get("peak_job_commit_bytes"),
            "memory_bytes": data.get("memory_bytes"),
            "elapsed_seconds": round(data.get("elapsed_seconds", 0.), 1)}


def plan(seed):
    if seed not in SEEDS:
        raise SystemExit("seed outside the declared block")
    steps = [{"label": f"{run_name(arm, seed)}-profile", "kind": "profile", "arm": arm}
             for arm in ARMS]
    steps += [{"label": run_name(arm, seed), "kind": "arm", "arm": arm} for arm in ARMS]
    steps += [{"label": f"{run_name(arm, seed)}-sweep-profile", "kind": "sweep-profile", "arm": arm}
              for arm in ARMS]
    steps += [{"label": f"{run_name(arm, seed)}-sweep-r1", "kind": "sweep", "arm": arm}
              for arm in ARMS]
    return steps


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    steps = plan(args.seed)
    manifest_path = RUNS/f"compute-replication-s{args.seed}.json"
    if manifest_path.exists():
        raise SystemExit("this seed's manifest already exists; refusing to reuse")
    if args.plan_only:
        print(json.dumps({"seed": args.seed, "steps": [step["label"] for step in steps],
                          "work_items": [ARMS[step["arm"]]["steps"] for step in steps]}))
        return 0
    records, started = [], {}
    for step in steps:
        arm, name = step["arm"], run_name(step["arm"], args.seed)
        if step["kind"] == "profile":
            worker = [str(ROOT/"scripts/train_m2_stream.py"), "--output", str(RUNS/step["label"]),
                      "--profile-square", "--steps", "2", "--seed", str(args.seed), *COMMON]
            record = launch(step["label"], ARM_CEILING_MIB, 300, 2, worker)
            admission = RUNS/step["label"]
        elif step["kind"] == "arm":
            worker = [str(ROOT/"scripts/train_m2_stream.py"), "--output", str(RUNS/name),
                      "--steps", str(ARMS[arm]["steps"]),
                      "--max-new-steps", str(ARMS[arm]["steps"]), "--seed", str(args.seed),
                      "--admission", str(RUNS/f"{name}-profile"), *COMMON, *ARMS[arm]["schedule"]]
            record = launch(name, ARM_CEILING_MIB, 2400, ARMS[arm]["steps"], worker)
            summary = json.loads((RUNS/name/"summary.json").read_text())
            if summary.get("status") != "complete" or summary.get("steps") != ARMS[arm]["steps"]:
                raise SystemExit(f"{name} did not complete its declared steps")
            record.update(weights_sha256=summary["checkpoint_sha256"],
                          initial_weights_from=summary.get("initial_weights_from"),
                          steps=summary["steps"], seed=summary["arguments"]["seed"])
        elif step["kind"] == "sweep-profile":
            record = launch(step["label"], SWEEP_CEILING_MIB, 300, 4,
                            [str(ROOT/"scripts/sweep_stream_occupancy.py"), "--run", name,
                             "--output", str(RUNS/step["label"]), "--profile",
                             "--training-groups", "12"])
        else:
            worker = [str(ROOT/"scripts/sweep_stream_occupancy.py"), "--run", name,
                      "--output", str(RUNS/step["label"]), "--training-groups", "12",
                      "--admission", str(RUNS/f"{name}-sweep-profile")]
            record = launch(step["label"], SWEEP_CEILING_MIB, 600, SWEEP_CASES*2+4, worker)
            result = json.loads((RUNS/step["label"]/"result.json").read_text())
            record.update(sweep_status=result["status"], deciding_cases=len(result["cases"]))
        records.append({**step, **record})
    result = {"status": "passed", "seed": args.seed, "arms": sorted(ARMS),
              "runs": {arm: run_name(arm, args.seed) for arm in ARMS},
              "sources": {name: digest(ROOT/name) for name in DRIVER_SOURCES},
              "jobs": records,
              "scope": ("One seed's three declared replicate arms and their sweeps, each under the "
                        "bounded runner. No claim: the study's read is compute_replication.py.")}
    manifest_path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "manifest": str(manifest_path.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
