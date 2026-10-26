"""Synthetic pressure with tiny native Jobs; never allocate host-scale RAM."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import run_bounded as runner

GiB = 1024**3


def healthy():
    return {"commit_bytes": 20*GiB, "commit_limit_bytes": 48*GiB,
            "physical_bytes": 32*GiB, "available_physical_bytes": 8*GiB,
            "counter_source": "synthetic-test"}


@unittest.skipUnless(os.name == "nt", "Windows pressure and Job contracts")
class PressureGuardTests(unittest.TestCase):
    def wrapper(self, root, child_source, counter_source):
        script = root/"forced_counter.py"
        script.write_text("import sys\nsys.path.insert(0,'scripts')\nimport run_bounded as runner\n"+
                          counter_source+"\nrunner.system_memory_snapshot=snapshot\nraise SystemExit(runner.main())\n", encoding="utf-8")
        receipt = root/"receipt.json"
        child = subprocess.run([sys.executable, str(script), "--memory-mib", "96", "--wall-seconds", "5",
                                "--work-items", "1", "--receipt", str(receipt), "--", sys.executable, "-c", child_source],
                               capture_output=True, text=True, timeout=12)
        return child, json.loads(receipt.read_text())

    def assert_dead(self, pids):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        for pid in pids:
            handle = kernel.OpenProcess(0x1000, False, pid)
            if handle:
                code = wintypes.DWORD()
                self.assertTrue(kernel.GetExitCodeProcess(handle, ctypes.byref(code)))
                kernel.CloseHandle(handle)
                self.assertNotEqual(code.value, 259, f"owned process survived guard: {pid}")

    def test_commit_admission_refuses_before_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root/"must-not-exist"
            high = {**healthy(), "commit_bytes": 31*GiB}
            child, receipt = self.wrapper(root, f"from pathlib import Path; Path({str(marker)!r}).write_text('spawned')",
                                          f"def snapshot():\n    return {high!r}\n")
            self.assertEqual(child.returncode, 125, child.stderr)
            self.assertEqual(receipt["status"], "memory_pressure")
            self.assertNotIn("child_pid", receipt)
            self.assertFalse(marker.exists())
            print(json.dumps({"probe": "pressure_admission", "receipt": receipt}))

    def tree_case(self, failure=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree, grand_file = root/"tree.json", root/"grand.pid"
            grand_source = f"import os,time;from pathlib import Path;Path({str(grand_file)!r}).write_text(str(os.getpid()));time.sleep(30)"
            source = ("import json,os,subprocess,sys,time\nfrom pathlib import Path\n"+
                      f"p=subprocess.Popen([sys.executable,'-c',{grand_source!r}])\n"+
                      f"grand_file=Path({str(grand_file)!r})\n"+
                      "for _ in range(100):\n    if grand_file.exists(): break\n    time.sleep(.01)\n"+
                      f"Path({str(tree)!r}).write_text(json.dumps([os.getpid(),p.pid,int(grand_file.read_text())]))\ntime.sleep(30)\n")
            counter = ("from pathlib import Path\ndef snapshot():\n"+f"    if Path({str(tree)!r}).exists():\n"+
                       ("        raise OSError('synthetic counter failure')\n" if failure else
                        f"        return {{**{healthy()!r}, 'commit_bytes': {31*GiB}}}\n")+
                       f"    return {healthy()!r}\n")
            child, receipt = self.wrapper(root, source, counter)
            self.assertEqual(child.returncode, 1 if failure else 125, child.stderr)
            self.assertEqual(receipt["status"], "failed" if failure else "memory_pressure")
            self.assertTrue(tree.exists())
            pids = [receipt["child_pid"], *json.loads(tree.read_text())]
            self.assert_dead(pids)
            print(json.dumps({"probe": "counter_failure" if failure else "runtime_pressure", "receipt": receipt, "verified_dead": pids}))

    def test_pressure_stops_child_and_grandchild(self):
        self.tree_case()

    def test_counter_query_failure_kills_owned_tree(self):
        self.tree_case(failure=True)

    def test_os_snapshot_reports_machine_not_small_nested_job(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory)/"receipt.json"
            source = ("import json,sys;sys.path.insert(0,'scripts');from run_bounded import system_memory_snapshot;"
                      "s=system_memory_snapshot();assert s['physical_bytes']>4*1024**3;"
                      "assert s['commit_bytes']>96*1024**2;print(json.dumps(s))")
            child = subprocess.run([sys.executable, "scripts/run_bounded.py", "--memory-mib", "96", "--wall-seconds", "5",
                                    "--work-items", "1", "--receipt", str(receipt), "--", sys.executable, "-c", source],
                                   capture_output=True, text=True, timeout=12)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertIn("GetPerformanceInfo", child.stdout)
            print(child.stdout.strip())

    def test_pressure_policy_boundaries_and_invalid_counters(self):
        self.assertIsNone(runner.pressure_status(healthy(), 2*GiB)["reason"])
        self.assertEqual(runner.pressure_status({**healthy(), "commit_bytes": 30*GiB})["reason"], "os_commit_pressure")
        self.assertEqual(runner.pressure_status({**healthy(), "commit_bytes": 29*GiB}, 2*GiB)["reason"], "os_commit_reservation")
        self.assertEqual(runner.pressure_status({**healthy(), "available_physical_bytes": GiB})["reason"], "physical_pressure")
        for field, value in (("physical_bytes", 0), ("commit_limit_bytes", 0), ("commit_bytes", -1),
                             ("available_physical_bytes", 33*GiB), ("physical_bytes", True)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.pressure_status({**healthy(), field: value})


if __name__ == "__main__":
    unittest.main()
