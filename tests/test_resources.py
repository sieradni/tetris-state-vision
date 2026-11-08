"""Live enforcement probes: tiny limits and timeouts, no large allocations."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


GIB = 1024**3


def memory_snapshot(**overrides):
    snapshot = {"commit_bytes": 30*GIB, "commit_limit_bytes": 48*GIB,
                "physical_bytes": 32*GIB, "available_physical_bytes": 8*GIB}
    snapshot.update(overrides)
    return snapshot


class PressureDecisionTests(unittest.TestCase):
    """Standing instrument for the resource-profile detection class.

    V-013, V-039 and V-040 all landed in that class, so the admission decision is
    pinned here as a matrix over OS counters rather than by whatever the host
    happens to be doing. The suite job of V-040 peaked at 343 MiB; nothing in that
    evidence is about the size of the job, only about the line it was judged against.
    """

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_bounded
        cls.pressure_status = staticmethod(run_bounded.pressure_status)

    def test_stop_line_tracks_the_commit_limit_not_physical_ram(self):
        # V-040: min(physical-reserve, limit*9//10) resolved to physical-reserve,
        # which a page-file machine exceeds by its normal steady state.
        status = self.pressure_status(memory_snapshot())
        self.assertEqual(status["stop_commit_bytes"], 48*GIB*9//10)

    def test_commit_above_physical_ram_but_below_the_limit_admits_a_small_job(self):
        # The exact host state that refused five suite runs: commit well above
        # RAM-minus-headroom, gigabytes of physical memory still free.
        status = self.pressure_status(memory_snapshot(commit_bytes=38*GIB), 640*1024**2)
        self.assertIsNone(status["reason"])
        self.assertEqual(status["reserved_job_bytes"], 640*1024**2)

    def test_exhausted_commit_budget_still_refuses(self):
        status = self.pressure_status(memory_snapshot(commit_bytes=48*GIB*9//10))
        self.assertEqual(status["reason"], "os_commit_pressure")

    def test_job_reservation_crossing_the_line_still_refuses(self):
        status = self.pressure_status(memory_snapshot(commit_bytes=48*GIB*9//10-256*1024**2),
                                      640*1024**2)
        self.assertEqual(status["reason"], "os_commit_reservation")

    def test_physical_residency_floor_still_refuses(self):
        # Protective intent kept: low free physical memory refuses even when the
        # commit budget looks comfortable.
        status = self.pressure_status(memory_snapshot(commit_bytes=8*GIB,
                                                      available_physical_bytes=GIB))
        self.assertEqual(status["reason"], "physical_pressure")
        self.assertEqual(status["minimum_available_physical_bytes"], min(2*GIB, 32*GIB//8))


@unittest.skipUnless(os.name == "nt", "Windows Job Object probes")
class ResourceGuardTests(unittest.TestCase):
    def run_child(self, root, source, *, memory=64, wall=5, environment=None):
        receipt = root / "receipt.json"
        command = [sys.executable, "scripts/run_bounded.py", "--memory-mib", str(memory),
                   "--wall-seconds", str(wall), "--work-items", "1", "--receipt", str(receipt),
                   "--", sys.executable, "-c", source]
        result = subprocess.run(command, capture_output=True, text=True, timeout=12, env=environment)
        return result, json.loads(receipt.read_text())

    def run_command(self, root, executable, source):
        receipt = root / "receipt.json"
        command = [sys.executable, "scripts/run_bounded.py", "--memory-mib", "64",
                   "--wall-seconds", "5", "--work-items", "1", "--receipt", str(receipt),
                   "--", executable, "-c", source]
        result = subprocess.run(command, capture_output=True, text=True, timeout=12, cwd=ROOT)
        return result, json.loads(receipt.read_text())

    def test_relative_forward_slash_child_executable_is_refused_before_launch(self):
        # V-030: this spelling was executed by the shell for the supervisor but
        # not resolved by CreateProcess for the child. It must refuse first.
        with tempfile.TemporaryDirectory() as scratch:
            result, receipt = self.run_command(Path(scratch), ".venv/Scripts/python.exe", "print('SHOULD_NOT_RUN')")
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(receipt["status"], "failed")
            self.assertTrue(receipt["launch_refused"])
            self.assertIn("V-030", receipt["error"])
            self.assertNotIn("child_pid", receipt)
            self.assertNotIn("SHOULD_NOT_RUN", result.stdout)

    def test_backslash_relative_and_absolute_child_executables_launch(self):
        for executable in (os.path.join(".venv", "Scripts", "python.exe"), sys.executable):
            with tempfile.TemporaryDirectory() as scratch:
                result, receipt = self.run_command(Path(scratch), executable, "print('CHILD_RAN')")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("CHILD_RAN", result.stdout)
                self.assertEqual(receipt["status"], "passed")
                self.assertEqual(receipt["child_executable"], str((ROOT/executable).resolve()))

    def test_cpu_library_limits_override_host_before_child_imports(self):
        variables = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "OPENBLAS_DEFAULT_NUM_THREADS", "GOTO_NUM_THREADS")
        source = ("import os; expected="+repr(dict.fromkeys(variables, "1"))+"; "
                  "assert {name:os.environ.get(name) for name in expected}==expected; "
                  "print('PREIMPORT_CPU_LIMITS_VERIFIED')")
        with tempfile.TemporaryDirectory() as scratch:
            environment = {**os.environ, **dict.fromkeys(variables, "8")}
            result, receipt = self.run_child(Path(scratch), source, environment=environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("PREIMPORT_CPU_LIMITS_VERIFIED", result.stdout)
            self.assertEqual(receipt["cpu_library_threads"], dict.fromkeys(variables, "1"))

    def test_kernel_denies_allocation_over_small_job_ceiling(self):
        with tempfile.TemporaryDirectory() as scratch:
            # Test the native commit boundary directly, avoiding allocator retry
            # behavior and an assumption about Windows' historical peak counter.
            source = ("import ctypes; from ctypes import wintypes; "
                      "k=ctypes.WinDLL('kernel32', use_last_error=True); "
                      "k.VirtualAlloc.argtypes=[ctypes.c_void_p,ctypes.c_size_t,wintypes.DWORD,wintypes.DWORD]; "
                      "k.VirtualAlloc.restype=ctypes.c_void_p; "
                      "small=k.VirtualAlloc(None,4*1024*1024,0x3000,4); assert small; "
                      "large=k.VirtualAlloc(None,80*1024*1024,0x3000,4); "
                      "error=ctypes.get_last_error(); assert not large, 'limit did not deny allocation'; "
                      "assert error in (8,1455,1816), error; print('ALLOCATION_DENIED',error)")
            result, receipt = self.run_child(Path(scratch), source)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ALLOCATION_DENIED", result.stdout)
            self.assertEqual(receipt["status"], "passed")
            self.assertEqual(receipt["memory_bytes"], 64*1024**2)
            self.assertGreater(receipt["peak_job_commit_bytes"], 0)

    def test_ceiling_above_machine_clamp_is_refused_before_spawn(self):
        with tempfile.TemporaryDirectory() as scratch:
            result, receipt = self.run_child(Path(scratch), "raise SystemExit(9)", memory=8193)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(receipt["status"], "failed")
            self.assertNotIn("child_pid", receipt)

    def test_worker_queries_named_ceiling_beneath_another_nested_job(self):
        with tempfile.TemporaryDirectory() as scratch:
            source = ("import sys, ctypes; from ctypes import wintypes; sys.path.insert(0,'scripts'); "
                      "from probe_model import require_bounded_job; from run_bounded import Extended; "
                      "k=ctypes.WinDLL('kernel32',use_last_error=True); "
                      "k.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR]; k.CreateJobObjectW.restype=wintypes.HANDLE; "
                      "k.GetCurrentProcess.restype=wintypes.HANDLE; "
                      "k.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]; "
                      "k.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]; "
                      "nested=k.CreateJobObjectW(None,None); info=Extended(); info.basic.flags=0x3000; "
                      "assert k.SetInformationJobObject(nested,9,ctypes.byref(info),ctypes.sizeof(info)); "
                      "assert k.AssignProcessToJobObject(nested,k.GetCurrentProcess()); "
                      "assert require_bounded_job()==64*1024**2; print('EXACT_JOB_VERIFIED')")
            result, receipt = self.run_child(Path(scratch), source)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("EXACT_JOB_VERIFIED", result.stdout)
            self.assertIn("TetrisStateVision-", receipt["job_name"])

    def test_worker_refuses_a_forged_job_identity(self):
        with tempfile.TemporaryDirectory() as scratch:
            source = ("import sys,os; sys.path.insert(0,'scripts'); from probe_model import require_bounded_job; "
                      "os.environ['TETRIS_VISION_JOB_NAME']='Local\\\\nonexistent-tetris-vision-job'; "
                      "require_bounded_job()")
            result, receipt = self.run_child(Path(scratch), source)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(receipt["status"], "failed")

    def test_wall_timeout_kills_child_and_grandchild_with_durable_receipt(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            pid_file = root / "grandchild.pid"
            source = ("import pathlib, subprocess, sys, time; "
                      "p=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                      f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(60)")
            result, receipt = self.run_child(root, source, wall=.8)
            self.assertEqual(result.returncode, 124, result.stderr)
            self.assertEqual(receipt["status"], "wall_timeout")
            self.assertTrue(pid_file.exists())
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            for pid in (receipt["child_pid"], int(pid_file.read_text())):
                handle = kernel.OpenProcess(0x1000, False, pid)
                if handle:
                    code = wintypes.DWORD()
                    self.assertTrue(kernel.GetExitCodeProcess(handle, ctypes.byref(code)))
                    kernel.CloseHandle(handle)
                    self.assertNotEqual(code.value, 259, f"orphan process {pid}")
