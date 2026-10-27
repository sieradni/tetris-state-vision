import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0, str(SCRIPTS))
from probe_stream_recovery import arguments


class NativeStreamGateTests(unittest.TestCase):
    def test_m2_default_and_joint_capacity_are_explicit(self):
        self.assertEqual(arguments(["--output", "unused"]).component_capacity, 0)
        self.assertEqual(arguments(["--output", "unused", "--component-capacity", "64"]).component_capacity, 64)
        for capacity in ("-1", "1", "9", "65"):
            with self.subTest(capacity=capacity), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(["--output", "unused", "--component-capacity", capacity])

    @unittest.skipUnless(os.name == "nt", "native Windows admission")
    def test_joint_gate_without_named_job_never_creates_or_launches_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"never-created"
            environment = dict(os.environ)
            environment.pop("TETRIS_VISION_JOB_NAME", None)
            result = subprocess.run([sys.executable, str(SCRIPTS/"probe_stream_recovery.py"), "--output", str(output),
                                     "--component-capacity", "64"], capture_output=True, text=True, env=environment, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("named supervisor Job identity", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
