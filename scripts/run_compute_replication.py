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


def _job_passed(label):
    path = RUNS/(label+"-job.json")
    if not path.is_file():
        return False
    data = json.loads(path.read_text())
    return data.get("status") == "passed" and data.get("exit_code") == 0


def _summary_status(path, steps=None):
    if not path.is_file():
        return False
    data = json.loads(path.read_text())
    return data.get("status") == "complete" and (steps is None or data.get("steps") == steps)


def step_state(step, name):
    """`verified` only when the output and its own job receipt both say the step finished."""
    label = step["label"]
    if step["kind"] == "arm":
        done = (_summary_status(RUNS/name/"summary.json", ARMS[step["arm"]]["steps"])
                and _job_passed(name))
        present = (RUNS/name).exists() or (RUNS/(name+"-job.json")).exists()
    elif step["kind"] == "sweep":
        done = (RUNS/label/"result.json").is_file() and _job_passed(label)
        present = (RUNS/label).exists() or (RUNS/(label+"-job.json")).exists()
    elif step["kind"] == "profile":
        done = _summary_status(RUNS/label/"summary.json", 2) and _job_passed(label)
        present = (RUNS/label).exists() or (RUNS/(label+"-job.json")).exists()
    else:
        done = _summary_status(RUNS/label/"summary.json") and _job_passed(label)
        present = (RUNS/label).exists() or (RUNS/(label+"-job.json")).exists()
    return "verified" if done else ("partial" if present else "absent")


def preserve(label, note):
    """Move a partial step's output and receipts aside rather than deleting them (V-050)."""
    destination = RUNS/"attempts"/("m3-repl-"+label)
    destination.mkdir(parents=True, exist_ok=False)
    moved = []
    for path in (RUNS/label, RUNS/(label+"-job.json"), RUNS/(label+".log")):
        if path.exists():
            path.rename(destination/path.name)
            moved.append(path.name)
    (destination/"note.txt").write_text(note+"\n", encoding="utf-8")
    print(json.dumps({"preserved": label, "moved": moved}), flush=True)


def lock_path(seed):
    return RUNS/f".repl-block-s{seed}.lock"


def _pid_alive(pid):
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True
    import ctypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
    if handle:
        kernel32.CloseHandle(handle)
        return True
    return ctypes.get_last_error() == 5  # access denied still means the process exists


def acquire_lock(path, pid):
    """One block per seed. A holder that is no longer alive is replaced, never trusted."""
    if path.exists():
        try:
            holder = int(path.read_text().strip() or 0)
        except ValueError:
            holder = 0
        if _pid_alive(holder):
            raise SystemExit(f"another block for this seed is live (pid {holder}); refusing to run a "
                             f"second one against the same outputs")
        path.write_text(str(pid)+"\n", encoding="utf-8")
        return "replaced_stale"
    path.write_text(str(pid)+"\n", encoding="utf-8")
    return "acquired"


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
    parser.add_argument("--resume", action="store_true",
                        help="reuse only steps whose summary and job receipt both verify; preserve "
                             "any partial output before re-running it")
    args = parser.parse_args(argv)
    steps = plan(args.seed)
    acquire_lock(lock_path(args.seed), os.getpid())
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
        if args.resume:
            state = step_state(step, name)
            if state == "verified":
                records.append({**step, "reused": True})
                print(json.dumps({"job": step["label"], "status": "reused"}), flush=True)
                continue
            if state == "partial":
                preserve(step["label"], "Interrupted by a host runner restart; preserved before "
                                        "re-running. Partial output is not used as evidence.")
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
