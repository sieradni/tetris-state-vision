"""A safely saved time-limited checkpoint must never become a completed run."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from record_learning_pilot import validate_completion


class PilotEvidenceTests(unittest.TestCase):
    def test_partial_is_refused_by_default_and_is_explicit_when_admitted(self):
        summary = {"status": "wall_partial", "steps": 511, "requested_steps": 512}
        job = {"status": "failed", "exit_code": 2}
        with self.assertRaises(ValueError):
            validate_completion(summary, job, 511)
        self.assertEqual(validate_completion(summary, job, 511, True), "wall_partial")
        with self.assertRaises(ValueError):
            validate_completion({**summary, "status": "complete"}, {"status": "passed", "exit_code": 0}, 511, True)

    def test_outer_kill_and_other_failures_cannot_admit_saved_partial(self):
        summary = {"status": "wall_partial", "steps": 511, "requested_steps": 512}
        for job in ({"status": "wall_timeout", "exit_code": 124},
                    {"status": "failed", "exit_code": 1}, {"status": "passed", "exit_code": 0}):
            with self.subTest(job=job), self.assertRaises(ValueError):
                validate_completion(summary, job, 511, True)
        for steps in (0, 510, 512, 513):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                validate_completion(summary, {"status": "failed", "exit_code": 2}, steps, True)

    def test_completed_budget_requires_matching_terminal_success(self):
        summary = {"status": "complete", "steps": 128, "requested_steps": 128}
        self.assertEqual(validate_completion(summary, {"status": "passed", "exit_code": 0}, 128), "complete")
        with self.assertRaises(ValueError):
            validate_completion(summary, {"status": "failed", "exit_code": 2}, 128, True)


if __name__ == "__main__":
    unittest.main()
