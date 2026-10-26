"""Windows workload supervisor: aggregate commit cap, wall cap, durable receipt.

The finite work count is declared here and must also be enforced by the child
interface (generation --count, training --max-steps, finite test selection).
This does not make an arbitrary unbounded child into a step-bounded workload.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


class Basic(ctypes.Structure):
    _fields_ = [("user", ctypes.c_int64), ("job_user", ctypes.c_int64),
                ("flags", wintypes.DWORD), ("min_ws", ctypes.c_size_t),
                ("max_ws", ctypes.c_size_t), ("active", wintypes.DWORD),
                ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                ("scheduling", wintypes.DWORD)]


class IO(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ("read_ops", "write_ops", "other_ops", "read", "write", "other")]


class Extended(ctypes.Structure):
    _fields_ = [("basic", Basic), ("io", IO),
                ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
        (name, ctypes.c_uint64) for name in
        ("physical", "available", "page", "available_page", "virtual",
         "available_virtual", "extended")]


class PerformanceInfo(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in
        ("commit", "commit_limit", "commit_peak", "physical", "available_physical",
         "system_cache", "kernel_total", "kernel_paged", "kernel_nonpaged", "page_size")] + [
        (name, wintypes.DWORD) for name in ("handles", "processes", "threads")]


@lru_cache(maxsize=1)
def _performance_reader():
    if os.name != "nt":
        raise RuntimeError("Windows OS commit counters required")
    library = ctypes.WinDLL("kernel32", use_last_error=True)
    reader = library.K32GetPerformanceInfo
    reader.argtypes = [ctypes.POINTER(PerformanceInfo), wintypes.DWORD]
    reader.restype = wintypes.BOOL
    return library, reader


def system_memory_snapshot():
    # PSAPI reports system pages, independently of the current/nested Job.
    info = PerformanceInfo()
    info.cb = ctypes.sizeof(info)
    if not _performance_reader()[1](ctypes.byref(info), info.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    if not info.page_size or not info.physical or not info.commit_limit:
        raise RuntimeError("invalid OS performance counters")
    return {"commit_bytes": info.commit*info.page_size,
            "commit_limit_bytes": info.commit_limit*info.page_size,
            "physical_bytes": info.physical*info.page_size,
            "available_physical_bytes": info.available_physical*info.page_size,
            "counter_source": "GetPerformanceInfo"}


def pressure_status(snapshot, reserved_bytes=0):
    fields = ("commit_bytes", "commit_limit_bytes", "physical_bytes", "available_physical_bytes")
    if any(type(snapshot.get(key)) is not int or snapshot[key] < 0 for key in fields):
        raise ValueError("invalid OS memory counter types/ranges")
    physical = snapshot["physical_bytes"]
    if not physical or not snapshot["commit_limit_bytes"] or snapshot["available_physical_bytes"] > physical or type(reserved_bytes) is not int or reserved_bytes < 0:
        raise ValueError("invalid OS memory counter bounds/reservation")
    reserve = min(2*1024**3, physical//8)
    stop_commit = min(physical-reserve, snapshot["commit_limit_bytes"]*9//10)
    reason = None
    if snapshot["available_physical_bytes"] < reserve:
        reason = "physical_pressure"
    elif snapshot["commit_bytes"] >= stop_commit:
        reason = "os_commit_pressure"
    elif snapshot["commit_bytes"]+reserved_bytes >= stop_commit:
        reason = "os_commit_reservation"
    return {"reason": reason, "stop_commit_bytes": stop_commit,
            "minimum_available_physical_bytes": reserve, "reserved_job_bytes": reserved_bytes}


def arm_job(memory_bytes: int, name: str | None = None):
    if os.name != "nt":
        raise RuntimeError("Windows Job Object enforcement required; refusing launch")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise ctypes.WinError(ctypes.get_last_error())
    clamp = min(memory.physical // 3, 8 * 1024**3)
    if not 0 < memory_bytes <= clamp:
        raise ValueError(f"memory ceiling must be positive and <= {clamp} bytes")
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                               ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
        ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
    kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    job = kernel.CreateJobObjectW(None, name)
    info = Extended()
    info.basic.flags = 0x200 | 0x2000  # JOB_MEMORY | KILL_ON_JOB_CLOSE
    info.job_memory = memory_bytes
    if not job or not kernel.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    # Assign the supervisor before spawning: every descendant inherits the Job.
    if not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())

    def peak():
        snapshot = Extended()
        if not kernel.QueryInformationJobObject(job, 9, ctypes.byref(snapshot), ctypes.sizeof(snapshot), None):
            raise ctypes.WinError(ctypes.get_last_error())
        if snapshot.job_memory != memory_bytes or snapshot.basic.flags & 0x2200 != 0x2200:
            raise RuntimeError("kernel memory/kill-on-close readback failed")
        return snapshot.peak_job

    peak()
    return kernel, job, peak


def main() -> int:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("child command must follow --")
    boundary = argv.index("--")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-mib", type=int, required=True)
    parser.add_argument("--wall-seconds", type=float, required=True)
    parser.add_argument("--work-items", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv[:boundary])
    command = argv[boundary + 1:]
    if not command or not math.isfinite(args.wall_seconds) or not 0 < args.wall_seconds <= 3600:
        parser.error("command and wall cap in (0, 3600] required")
    if args.work_items <= 0:
        parser.error("positive child-enforced work budget required")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental reuse of an earlier receipt.
    with args.receipt.open("x", encoding="utf-8") as output:
        json.dump({"status": "admitting", "command": command}, output)
    started = time.monotonic()
    state = {"supervisor_pid": os.getpid(), "command": command,
             "memory_bytes": args.memory_mib * 1024**2,
             "wall_seconds": args.wall_seconds, "work_items": args.work_items}

    def receipt(status, code=None):
        state.update(status=status, exit_code=code,
                     elapsed_seconds=round(time.monotonic() - started, 3))
        temporary = args.receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.receipt)

    try:
        job_name = "Local\\TetrisStateVision-"+uuid.uuid4().hex
        kernel, job, peak = arm_job(state["memory_bytes"], job_name)
        state["job_name"] = job_name
        state["guard"] = "readback-verified Windows aggregate Job commit + kill-on-close"
        state["supervisor_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        state["system_pressure_guard"] = "GetPerformanceInfo/0.2s"
        state["system_at_admission"] = system_memory_snapshot()
        state["system_pressure"] = pressure_status(state["system_at_admission"], state["memory_bytes"])

        def park(status, code):
            state["peak_job_commit_bytes"] = peak()
            receipt(status, code)
            print(json.dumps(state), flush=True)
            # The supervisor is a member too: persist BEFORE native termination.
            if not kernel.TerminateJobObject(job, code):
                raise ctypes.WinError(ctypes.get_last_error())
            return code

        if state["system_pressure"]["reason"] is not None:
            return park("memory_pressure", 125)
        receipt("admitted")
        print(json.dumps(state), flush=True)
        child_environment = dict(os.environ)
        child_environment["TETRIS_VISION_JOB_NAME"] = job_name
        child_environment["TETRIS_VISION_SUPERVISOR_SHA256"] = state["supervisor_script_sha256"]
        child_environment["TETRIS_VISION_SYSTEM_GUARD"] = state["system_pressure_guard"]
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, env=child_environment)
        state["child_pid"] = process.pid
        receipt("running")
        while process.poll() is None:
            remaining = args.wall_seconds-(time.monotonic()-started)
            if remaining <= 0:
                return park("wall_timeout", 124)
            state["last_system_memory"] = system_memory_snapshot()
            state["system_pressure"] = pressure_status(state["last_system_memory"])
            if state["system_pressure"]["reason"] is not None:
                return park("memory_pressure", 125)
            try:
                process.wait(timeout=min(.2, remaining))
            except subprocess.TimeoutExpired:
                pass
        code = process.returncode
        state["peak_job_commit_bytes"] = peak()
        receipt("passed" if code == 0 else "failed", code)
        print(json.dumps(state), flush=True)
        return code
    except Exception as error:
        state["error"] = repr(error)
        receipt("failed", 1)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
