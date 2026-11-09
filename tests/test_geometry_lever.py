"""The declared geometry-stage lever: admission join, ownership and the flag's objective rule."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from record_geometry_lever import STEPS, arm_matches
from record_stream_run import OWNED_RUN_PATTERN
from tetris_vision.losses import GEOMETRY_STAGE_POLICIES, require_geometry_final_only


def summary(flag=1, *, steps=STEPS, detail=1, stencil=0, context=0, factor=4, weight=4):
    return {"status": "complete", "steps": steps, "requested_steps": steps,
            "checkpoint_round_trip_exact": True,
            "arguments": {"geometry_final_only": flag, "occupancy_detail": detail,
                          "occupancy_detail_stencil": stencil, "occupancy_context": context,
                          "steps": steps, "seed": 20261026, "capture_max_side": 512,
                          "occupancy_factor": factor, "occupancy_positive_weight": weight},
            "recovery_contract": {"geometry_final_only": flag, "steps": steps}}


def job(steps=STEPS, ceiling=1152*1024**2, status="passed"):
    return {"status": status, "exit_code": 0 if status == "passed" else 1,
            "work_items": steps, "memory_bytes": ceiling}


class AdmissionTests(unittest.TestCase):
    def test_both_policies_are_admitted_only_for_their_own_flag(self):
        self.assertTrue(arm_matches(summary(1), job(), 1))
        self.assertTrue(arm_matches(summary(0), job(), 0))
        with self.assertRaises(ValueError):
            arm_matches(summary(1), job(), 0)

    def test_a_wrong_budget_ceiling_or_receipt_is_refused(self):
        with self.assertRaises(ValueError):
            arm_matches(summary(1, steps=64), job(steps=64), 1)
        with self.assertRaises(ValueError):
            arm_matches(summary(1), job(ceiling=640*1024**2), 1)
        with self.assertRaises(ValueError):
            arm_matches(summary(1), job(status="failed"), 1)

    def test_another_read_out_in_the_arm_is_refused(self):
        for override in ({"detail": 0}, {"stencil": 1}, {"context": 1}, {"factor": 1}, {"weight": 1}):
            with self.assertRaises(ValueError):
                arm_matches(summary(1, **override), job(), 1)

    def test_the_objective_flag_is_a_two_value_declaration(self):
        self.assertEqual(GEOMETRY_STAGE_POLICIES, (0, 1))
        self.assertIsNone(require_geometry_final_only(1, summary(1)))
        self.assertIsNone(require_geometry_final_only(0, summary(0)))
        with self.assertRaises(ValueError):
            require_geometry_final_only(1, summary(0))
        with self.assertRaises(ValueError):
            require_geometry_final_only(2, summary(1))
        with self.assertRaises(ValueError):
            require_geometry_final_only(True, summary(1))

    def test_the_owned_family_is_declared(self):
        for name in ("m3-occgeom-lever-r1", "m3-occgeom-baseline-r1",
                     "m3-occgeom-lever-r1-sweep-r1", "m3-occgeom-baseline-r1-probe-r1",
                     "m3-occgeom-lever-r1-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)


if __name__ == "__main__":
    unittest.main()
