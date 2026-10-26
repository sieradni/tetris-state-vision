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
from train_m2_pilot import arguments, main


class TrainingAdmissionTests(unittest.TestCase):
    def test_invalid_work_or_numeric_budgets_fail_before_creating_output(self):
        invalid = (("--steps", "0"), ("--steps", "513"), ("--states", "9"),
                   ("--threads", "3"), ("--max-side", "641"), ("--wall-seconds", "nan"),
                   ("--learning-rate", "inf"), ("--evaluate-every", "0"))
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch)/"never-created"
            for flag, value in invalid:
                argv = ["--output", str(output), "--steps", "16", flag, value]
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(argv)
                self.assertFalse(output.exists())
            self.assertEqual(arguments(["--output", str(output), "--steps", "16"]).steps, 16)

    @unittest.skipUnless(os.name == "nt", "Windows worker admission")
    def test_worker_without_supervisor_identity_refuses_before_output_or_torch(self):
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch)/"never-created"
            environment = dict(os.environ)
            environment.pop("TETRIS_VISION_JOB_NAME", None)
            result = subprocess.run([sys.executable, str(SCRIPTS/"train_m2_pilot.py"),
                                     "--output", str(output), "--steps", "1"], env=environment,
                                    capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("named supervisor Job identity is required", result.stderr)
            self.assertFalse(output.exists())
