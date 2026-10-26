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
from train_m2_stream import arguments, validate_admission


class StreamAdmissionTests(unittest.TestCase):
    def test_bounded_arguments_and_required_admission(self):
        base = ["--output", "unused", "--model", "pilot", "--steps", "2", "--profile-square"]
        for flag, value in (("--steps", "0"), ("--steps", "4097"), ("--seed", "-1"),
                            ("--wall-seconds", "nan"), ("--learning-rate", "inf"),
                            ("--evaluate-every", "0"), ("--development-groups", "13")):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments([*base, flag, value])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arguments(base[:-1])
        self.assertTrue(arguments(base).profile_square)

    def admission(self):
        config = {"dimension": 64}
        summary = {"status": "complete", "steps": 2, "profile_square": True, "checkpoint_round_trip_exact": True,
                   "resource_profile_version": 2, "warm_optimizer_backward": {"step": 2, "input_pixels": 512**2,
                   "all_parameters_initialized": True, "state_bytes": 100, "backward_completed": True},
                   "max_input_pixels_seen": 512**2, "max_side": 512, "threads": 1, "workers": 0, "prefetch": 0,
                   "tensor_cache_bytes": 0, "catalogue_sha256": "fixture", "model_kind": "pilot", "config": config,
                   "job_ceiling_bytes": 1024**3, "development_groups": 6, "cpu_only": True,
                   "supervisor_script_sha256": "supervisor-fixture", "system_pressure_guard": "GetPerformanceInfo/0.2s"}
        job = {"status": "passed", "exit_code": 0, "work_items": 2, "memory_bytes": 1024**3,
               "supervisor_script_sha256": "supervisor-fixture", "system_pressure_guard": "GetPerformanceInfo/0.2s"}
        expected = {"catalogue": "fixture", "model": "pilot", "config": config, "ceiling": 1024**3, "development_groups": 6, "supervisor_sha": "supervisor-fixture"}
        return summary, job, expected

    def test_incomplete_or_different_envelopes_never_admit_long_training(self):
        summary, job, expected = self.admission()
        validate_admission(summary, job, **expected)
        for field, value in (("status", "wall_partial"), ("max_input_pixels_seen", 384**2),
                             ("catalogue_sha256", "changed"), ("model_kind", "accuracy"),
                             ("config", {"dimension": 192}), ("job_ceiling_bytes", 2*1024**3),
                             ("development_groups", 3), ("cpu_only", False), ("workers", 2),
                             ("tensor_cache_bytes", 100), ("checkpoint_round_trip_exact", False)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_admission({**summary, field: value}, job, **expected)
        for field, value in (("supervisor_script_sha256", "old-supervisor"), ("system_pressure_guard", None)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_admission(summary, {**job, field: value}, **expected)
        with self.assertRaises(ValueError):
            validate_admission(summary, {**job, "status": "wall_timeout", "exit_code": 124}, **expected)

    def test_cold_or_incomplete_optimizer_profiles_cannot_admit_a_study(self):
        summary, job, expected = self.admission()
        for field, value in (("steps", 1), ("resource_profile_version", 1), ("warm_optimizer_backward", {})):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_admission({**summary, field: value}, job, **expected)
        for field, value in (("step", 1), ("input_pixels", 384**2), ("all_parameters_initialized", False),
                             ("backward_completed", False), ("state_bytes", 0)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_admission({**summary, "warm_optimizer_backward": {**summary["warm_optimizer_backward"], field: value}}, job, **expected)
        with self.assertRaises(ValueError):
            validate_admission(summary, {**job, "work_items": 1}, **expected)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arguments(["--output", "unused", "--model", "pilot", "--steps", "1", "--profile-square"])

    @unittest.skipUnless(os.name == "nt", "native worker admission")
    def test_worker_refuses_without_job_identity_before_source_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"never-created"
            environment = dict(os.environ)
            environment.pop("TETRIS_VISION_JOB_NAME", None)
            child = subprocess.run([sys.executable, str(SCRIPTS/"train_m2_stream.py"), "--output", str(output),
                                    "--steps", "2", "--model", "pilot", "--profile-square"],
                                   capture_output=True, text=True, timeout=15, env=environment)
            self.assertNotEqual(child.returncode, 0)
            self.assertIn("named supervisor Job identity", child.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
