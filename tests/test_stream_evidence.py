from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from record_stream_run import checkpoint_architecture, validate_completion


class StreamEvidenceTests(unittest.TestCase):
    def test_fit_policy_is_derived_from_matching_checkpoint_recovery_metadata(self):
        summary = {"config": {"component_capacity": 0}, "component_capacity": 0,
                   "curriculum": "all-class-fit-v1", "recovery_contract": {"curriculum": "all-class-fit-v1"}}
        manifest = {"config": summary["config"], "architecture": "TetrisStateNet-M2",
                    "metadata": {"recovery_contract": {"curriculum": "all-class-fit-v1"}}}
        self.assertEqual(checkpoint_architecture(summary, manifest), "TetrisStateNet-M2")
        manifest["metadata"]["recovery_contract"]["curriculum"] = "stream-v1"
        with self.assertRaises(ValueError):
            checkpoint_architecture(summary, manifest)
    def test_architecture_identity_comes_from_matching_checkpoint_config(self):
        old = {"config": {"dimension": 64}}
        manifest = {"architecture": "TetrisStateNet-M2", "config": old["config"]}
        self.assertEqual(checkpoint_architecture(old, manifest), "TetrisStateNet-M2")
        joint = {"config": {"dimension": 64, "component_capacity": 64}, "component_capacity": 64}
        manifest = {"architecture": "TetrisStateNet-M3", "config": joint["config"]}
        self.assertEqual(checkpoint_architecture(joint, manifest), "TetrisStateNet-M3")
        for changed in ({**manifest, "architecture": "TetrisStateNet-M2"},
                        {**manifest, "config": {"dimension": 192, "component_capacity": 64}}):
            with self.assertRaises(ValueError):
                checkpoint_architecture(joint, changed)
        with self.assertRaises(ValueError):
            checkpoint_architecture({**joint, "component_capacity": 0}, manifest)
        with self.assertRaises(ValueError):
            checkpoint_architecture({"config": {"component_capacity": True}, "component_capacity": True}, manifest)

    def test_completed_resume_counts_new_updates_separately_from_global_target(self):
        summary = {"status": "complete", "steps": 128, "requested_steps": 128,
                   "updates_this_run": 35, "checkpoint_round_trip_exact": True}
        job = {"status": "passed", "exit_code": 0, "work_items": 35}
        self.assertFalse(validate_completion(summary, job))
        with self.assertRaises(ValueError):
            validate_completion(summary, {**job, "work_items": 34})

    def test_wall_and_step_partials_need_explicit_admission_and_keep_partial_tag(self):
        for status, steps, target, cap in (("wall_partial", 93, 128, 128), ("step_partial", 512, 4096, 512)):
            with self.subTest(status=status):
                summary = {"status": status, "steps": steps, "requested_steps": target,
                           "updates_this_run": steps, "checkpoint_round_trip_exact": True}
                job = {"status": "failed", "exit_code": 2, "work_items": cap}
                with self.assertRaises(ValueError):
                    validate_completion(summary, job)
                self.assertTrue(validate_completion(summary, job, allow_partial=True))
                self.assertEqual(summary["status"], status)

    def test_outer_timeout_pressure_crash_or_missing_reload_cannot_be_curated_as_partial(self):
        summary = {"status": "wall_partial", "steps": 93, "requested_steps": 128,
                   "updates_this_run": 93, "checkpoint_round_trip_exact": True}
        for status, code in (("wall_timeout", 124), ("memory_pressure", 125), ("failed", 1), ("passed", 0)):
            with self.subTest(status=status), self.assertRaises(ValueError):
                validate_completion(summary, {"status": status, "exit_code": code, "work_items": 128}, allow_partial=True)
        with self.assertRaises(ValueError):
            validate_completion({**summary, "checkpoint_round_trip_exact": False}, {"status": "failed", "exit_code": 2, "work_items": 128}, allow_partial=True)


if __name__ == "__main__":
    unittest.main()
