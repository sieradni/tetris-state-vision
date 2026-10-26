"""One CPU forward/backward characterization step, only inside a bounded Job."""
import argparse
import ctypes
from ctypes import wintypes
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from run_bounded import Extended


def bounded_job_snapshot():
    if os.name != "nt":
        raise RuntimeError("Windows Job Object enforcement required")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.OpenJobObjectW.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                                wintypes.DWORD, ctypes.c_void_p]
    name = os.environ.get("TETRIS_VISION_JOB_NAME")
    if not name:
        raise RuntimeError("use run_bounded.py; named supervisor Job identity is required")
    job = kernel.OpenJobObjectW(0x0004, False, name)  # Query-only handle.
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        member = wintypes.BOOL()
        if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), job, ctypes.byref(member)) or not member.value:
            raise RuntimeError("worker is not a member of the supervisor's named Job")
        snapshot = Extended()
        if not kernel.QueryInformationJobObject(job, 9, ctypes.byref(snapshot), ctypes.sizeof(snapshot), None):
            raise ctypes.WinError(ctypes.get_last_error())
        if snapshot.basic.flags & 0x2200 != 0x2200 or not 0 < snapshot.job_memory <= 2*1024**3:
            raise RuntimeError("use run_bounded.py with <=2 GiB aggregate commit and a wall cap")
        return {"memory_bytes": snapshot.job_memory, "peak_job_commit_bytes": snapshot.peak_job}
    finally:
        kernel.CloseHandle(job)


def require_bounded_job():
    return bounded_job_snapshot()["memory_bytes"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ceiling = require_bounded_job()  # Before importing the optional heavy runtime.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    started = time.monotonic()
    import torch
    from tetris_vision.dataset import source_digest
    from tetris_vision.losses import diagnostic_metrics, joint_loss, targets_for
    from tetris_vision.model import ModelConfig, TetrisStateNet
    from tetris_vision.preprocess import prepare_image
    from tetris_vision.render import render, sample_scene

    torch.set_num_threads(1)
    torch.manual_seed(44)
    source = source_digest()
    result = render(sample_scene(44))
    prepared = prepare_image(result.image, max_side=384)
    target = targets_for(result.state, prepared)
    model = TetrisStateNet(ModelConfig.pilot())
    outputs = model(prepared.pixels[None], prepared.valid[None])
    loss, terms = joint_loss(outputs, target)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("nonfinite forward loss")
    loss.backward()
    if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
        raise RuntimeError("nonfinite gradient")
    if source_digest() != source:
        raise RuntimeError("runtime source changed during probe")
    record = {"status": "passed", "cpu_only": torch.version.cuda is None,
              "torch_version": str(torch.__version__), "threads": torch.get_num_threads(),
              "config": asdict(model.config), "parameters": sum(p.numel() for p in model.parameters()),
              "input_shape": list(prepared.pixels.shape), "loss": float(loss.detach()),
              "terms": {key: float(value.detach()) for key, value in terms.items()},
              "diagnostics": diagnostic_metrics(outputs, target), "steps": 1,
              "job_ceiling_bytes": ceiling, "runtime_source_sha256": source,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "state_sha256": hashlib.sha256(result.state.to_json().encode()).hexdigest(),
              "image_pixels_sha256": hashlib.sha256(result.image.tobytes()).hexdigest(),
              "elapsed_seconds": time.monotonic()-started,
              "claim": "Numerical/resource characterization only; no trained-model accuracy."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(record, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
