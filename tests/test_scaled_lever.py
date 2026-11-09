"""The declared 4x scaling rule: matched-budget attribution, and no reference in the verdict."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from record_scaled_lever import (ATTRIBUTION_DELTA, BASELINE_STEPS, CEILING_BYTES, STEPS,
                                 require_scaled_arm, scaling_statement)
from record_stream_run import OWNED_RUN_PATTERN
from record_representation_lever import lever_verdict


def summary(stencil=1, *, steps=STEPS, detail=1, context=0, factor=4, weight=4, seed=20261026):
    return {"status": "complete", "steps": steps, "requested_steps": steps,
            "checkpoint_round_trip_exact": True,
            "arguments": {"occupancy_detail": detail, "occupancy_detail_stencil": stencil,
                          "occupancy_context": context, "steps": steps, "seed": seed,
                          "capture_max_side": 512, "occupancy_factor": factor,
                          "occupancy_positive_weight": weight},
            "recovery_contract": {"occupancy_detail": detail, "occupancy_detail_stencil": stencil,
                                  "steps": steps}}


def job(steps=STEPS, ceiling=CEILING_BYTES, status="passed"):
    return {"status": status, "exit_code": 0 if status == "passed" else 1,
            "work_items": steps, "memory_bytes": ceiling}


class AdmissionTests(unittest.TestCase):
    def test_the_declared_budget_is_four_times_the_baseline(self):
        self.assertEqual((STEPS, BASELINE_STEPS), (256, 64))

    def test_a_matching_arm_is_admitted(self):
        self.assertTrue(require_scaled_arm(summary(1), job(), 1))
        self.assertTrue(require_scaled_arm(summary(0), job(), 0))

    def test_a_sixty_four_step_summary_is_refused_as_a_scaled_arm(self):
        with self.assertRaises(ValueError):
            require_scaled_arm(summary(1, steps=64), job(steps=64), 1)

    def test_the_wrong_stencil_form_is_refused(self):
        with self.assertRaises(ValueError):
            require_scaled_arm(summary(0), job(), 1)

    def test_a_smaller_ceiling_or_failed_receipt_is_refused(self):
        with self.assertRaises(ValueError):
            require_scaled_arm(summary(1), job(ceiling=640*1024**2), 1)
        with self.assertRaises(ValueError):
            require_scaled_arm(summary(1), job(status="failed"), 1)

    def test_another_lever_in_the_arm_is_refused(self):
        for override in ({"context": 1}, {"factor": 1}, {"weight": 1}, {"detail": 0}, {"seed": 7}):
            with self.assertRaises(ValueError):
                require_scaled_arm(summary(1, **override), job(), 1)


class ScalingRuleTests(unittest.TestCase):
    def verdict(self, per_cell, ranking, delta):
        return lever_verdict(per_cell, ranking, delta)

    def test_scaling_is_licensed_only_by_the_standing_conditions(self):
        cleared = self.verdict(.75, "informative", .10)
        self.assertEqual(cleared["verdict"], "cleared")
        statement = scaling_statement(cleared, .10, .4913)
        self.assertTrue(statement["licensed"])
        self.assertEqual(statement["attribution_delta_vs_matched_budget_control"], .10)

    def test_a_lever_short_of_any_condition_refuses_scaling(self):
        for verdict in (self.verdict(.69, "informative", .10),
                        self.verdict(.75, "uninformative", .10),
                        self.verdict(.75, "informative", .049)):
            statement = scaling_statement(verdict, .049, .4913)
            self.assertFalse(statement["licensed"])
            self.assertIn("refused", statement["statement"])

    def test_the_attribution_floor_is_the_standing_five_points(self):
        self.assertEqual(ATTRIBUTION_DELTA, .05)

    def test_the_unmatched_reference_cannot_change_the_verdict(self):
        short = self.verdict(.60, "uninformative", .01)
        long = self.verdict(.60, "uninformative", .01)
        self.assertEqual(scaling_statement(short, .01, .0)["licensed"],
                         scaling_statement(long, .01, 9.9)["licensed"])

    def test_the_owned_family_is_declared(self):
        for name in ("m3-occscale-lever-r1", "m3-occscale-control-r1",
                     "m3-occscale-lever-r1-sweep-r1", "m3-occscale-control-r1-profile-r1",
                     "m3-occscale-registration-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)


if __name__ == "__main__":
    unittest.main()
