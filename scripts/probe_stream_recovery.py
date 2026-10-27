"""Finite native integration gate: square profile2, full6, partial3, resume3.

Run under a 1 GiB parent Job. Each serialized child has its own 640 MiB Job,
90-second wall limit and exact update cap. No real images or unbounded loops.
"""
import argparse
import copy
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from probe_model import require_bounded_job
from source_snapshot import verify_source

ROOT = Path(__file__).resolve().parents[1]


def json_read(path):
    if path.stat().st_size > 1024**2:
        raise ValueError("gate JSON byte budget exceeded")
    return json.loads(path.read_text())


def verify_dead(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if handle:
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259:
                raise RuntimeError(f"owned gate process still live: {pid}")
        finally:
            kernel.CloseHandle(handle)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--component-capacity", type=int, default=0)
    parser.add_argument("--layer-evidence", action="store_true")
    parser.add_argument("--curriculum", choices=("stream-v1", "all-class-fit-v1"), default="stream-v1")
    args = parser.parse_args(argv)
    if args.component_capacity != 0 and not 10 <= args.component_capacity <= 64:
        parser.error("joint gate capacity must cover renderer10 slots and stay<=64;0 keeps M2")
    if args.curriculum == "all-class-fit-v1" and args.component_capacity:
        parser.error("all-class fit is capacity0 only")
    return args


def main():
    require_bounded_job()
    args = arguments()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runs, receipts = {}, {}

    def execute(name, updates, *, profile=False, resume=None, expected=0):
        run = output/name
        receipt = output/(name+"-job.json")
        worker = [sys.executable, str(ROOT/"scripts/train_m2_stream.py"), "--output", str(run),
                  "--model", "pilot", "--steps", "2" if profile else "6", "--wall-seconds", "60",
                  "--component-capacity", str(args.component_capacity),
                  "--development-groups", "3", "--evaluate-every", "3", "--checkpoint-every", "3"]
        if args.layer_evidence:
            worker += ["--layer-evidence"]
        if args.curriculum != "stream-v1":
            worker += ["--curriculum", args.curriculum]
        if profile:
            worker += ["--profile-square"]
        else:
            worker += ["--admission", str(output/"profile"), "--max-new-steps", str(updates)]
        if resume is not None:
            worker += ["--resume-from", str(resume)]
        command = [sys.executable, str(ROOT/"scripts/run_bounded.py"), "--memory-mib", "640",
                   "--wall-seconds", "90", "--work-items", str(updates), "--receipt", str(receipt), "--", *worker]
        print(json.dumps({"case": name, "updates_cap": updates, "command": command}), flush=True)
        with (output/(name+".log")).open("xb") as log, (output/(name+".stderr.log")).open("xb") as stderr:
            process = subprocess.Popen(command, stdout=log, stderr=stderr, cwd=ROOT)
            (output/"progress.json").write_text(json.dumps({"case": name, "supervisor_pid": process.pid})+"\n")
            actual = process.wait(timeout=105)
        job = json_read(receipt)
        if actual != expected or job["exit_code"] != expected or job["status"] != ("passed" if expected == 0 else "failed"):
            raise RuntimeError(f"gate case failed: {name}: {job}")
        summary, progress = json_read(run/"summary.json"), json_read(run/"progress.json")
        for pid in (process.pid, job["supervisor_pid"], job["child_pid"], progress["pid"]):
            verify_dead(pid)
        if job["memory_bytes"] != 640*1024**2 or job["work_items"] != updates or summary["updates_this_run"] != updates:
            raise ValueError("gate resource/work envelope differs")
        if (summary["status"] != ("complete" if expected == 0 else "step_partial") or
                not summary["checkpoint_round_trip_exact"] or summary["component_capacity"] != args.component_capacity or
                summary["config"]["component_capacity"] != args.component_capacity or
                summary["config"].get("layer_evidence", False) is not args.layer_evidence or
                summary.get("curriculum", "stream-v1") != args.curriculum):
            raise ValueError("gate completion/partial/parity contract differs")
        if not Path(summary["driver_source_path"]).resolve().is_relative_to(run/"code"):
            raise ValueError("trainer main source path is not archived")
        runs[name], receipts[name] = summary, job
        print(json.dumps({"case": name, "status": summary["status"], "steps": summary["steps"],
                          "verified_dead_worker": progress["pid"]}), flush=True)
        return run

    execute("profile", 2, profile=True)
    full = execute("continuous", 6)
    partial = execute("partial", 3, expected=2)
    resumed = execute("resumed", 3, resume=partial)
    catalogue = runs["continuous"]["catalogue_sha256"]
    for name, summary in runs.items():
        if summary["catalogue_sha256"] != catalogue:
            raise ValueError("gate cases did not use identical frozen sources")
        verify_source(output/name/"code", catalogue)
    if runs["partial"]["steps"] != 3 or runs["resumed"]["resume"]["step"] != 3 or runs["resumed"]["steps"] != 6:
        raise ValueError("resume cursor differs from predeclared3+3 boundary")

    def rows(path):
        if path.stat().st_size > 16*1024**2:
            raise ValueError("gate ledger byte budget exceeded")
        return [json.loads(line) for line in path.read_text().splitlines()]

    if (full/"training.jsonl").read_bytes() != (resumed/"training.jsonl").read_bytes():
        raise ValueError("resumed global data stream differs")
    if rows(partial/"training.jsonl") != rows(full/"training.jsonl")[:3]:
        raise ValueError("partial data prefix differs")
    def curve(path):
        return [{key: value for key, value in row.items() if key != "elapsed_seconds"} for row in rows(path)]
    if curve(full/"curve.jsonl") != curve(resumed/"curve.jsonl") or curve(partial/"curve.jsonl") != curve(full/"curve.jsonl")[:3]:
        raise ValueError("resumed loss/terms/gradient/data sequence differs")
    if rows(full/"evaluations.jsonl") != rows(resumed/"evaluations.jsonl"):
        raise ValueError("resumed evaluation sequence differs")
    for field in ("initial_training_probe", "initial_development", "final_training_probe", "development"):
        if runs["continuous"][field] != runs["resumed"][field]:
            raise ValueError("resumed fixed-probe outputs differ")

    code = full/"code"
    sys.path[:0] = [str(code/"scripts"), str(code)]
    sys.dont_write_bytecode = True
    import torch
    torch.set_num_threads(1)
    from tetris_vision.checkpoint import load_checkpoint
    from tetris_vision.model import ModelConfig, TetrisStateNet
    from training_snapshot import load_training_snapshot
    from train_m2_stream import equal_tree
    if (not Path(load_training_snapshot.__code__.co_filename).resolve().is_relative_to(code) or
            not Path(equal_tree.__code__.co_filename).resolve().is_relative_to(code)):
        raise ValueError("gate recovery import is not frozen")
    expected_model, inference_record = load_checkpoint(full/"checkpoint")
    resumed_model, _ = load_checkpoint(resumed/"checkpoint")
    if not equal_tree(expected_model.state_dict(), resumed_model.state_dict()):
        raise ValueError("resumed inference model tensors differ")
    saved = []
    config = dict(runs["continuous"]["config"])
    config["widths"] = tuple(config["widths"])
    for run in (full, resumed):
        model = TetrisStateNet(ModelConfig(**config))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        recovery = load_training_snapshot(run, model, optimizer, contract=runs["continuous"]["recovery_contract"])
        if recovery["step"] != 6 or not equal_tree(expected_model.state_dict(), model.state_dict()):
            raise ValueError("recovery artifact does not match final inference model")
        saved.append(copy.deepcopy((model.state_dict(), optimizer.state_dict(), torch.get_rng_state(),
                                    {name: module.training for name, module in model.named_modules()})))
    if not equal_tree(*saved):
        raise ValueError("resumed snapshot model/optimizer/RNG differs")
    proof = {"status": "passed", "source_catalogue_sha256": catalogue,
             "architecture": inference_record["architecture"], "component_capacity": args.component_capacity,
             "config": runs["continuous"]["config"],
             "curriculum": args.curriculum,
             "updates": {name: summary["updates_this_run"] for name, summary in runs.items()},
             "partial_status": runs["partial"]["status"], "global_steps": 6,
             "exact_data_ledger": True, "exact_loss_terms_gradient_curve": True,
             "exact_evaluations": True, "exact_model_optimizer_rng": True,
             "all_final_tensors_reload_exact": True,
             "weights_sha256": {name: summary["checkpoint_sha256"] for name, summary in runs.items()},
             "child_peak_job_commit_bytes": {name: job["peak_job_commit_bytes"] for name, job in receipts.items()},
             "orchestrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "claim": "Finite source-frozen stream recovery equivalence only; no accuracy or release acceptance."}
    (output/"proof.json").write_text(json.dumps(proof, indent=2)+"\n")
    (output/"progress.json").write_text(json.dumps({"status": "passed", "proof": "proof.json"})+"\n")
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
