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


@unittest.skipUnless(os.name == "nt", "Windows Job Object probes")
class ResourceGuardTests(unittest.TestCase):
    def run_child(self, root, source, *, memory=64, wall=5):
        receipt = root / "receipt.json"
        command = [sys.executable, "scripts/run_bounded.py", "--memory-mib", str(memory),
                   "--wall-seconds", str(wall), "--work-items", "1", "--receipt", str(receipt),
                   "--", sys.executable, "-c", source]
        result = subprocess.run(command, capture_output=True, text=True, timeout=12)
        return result, json.loads(receipt.read_text())

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
