"""Integrity controls for the M32 mass-shaping receipt.

`docs/M32_MASS_MATCH.md` declares one supervision-side term on the read's mass, the unchanged gate
and both branch interpretations. These tests bind the receipt: it joins the M30/M31 winner on the
same cells, the gate block is the standing one, the measured result is the refusal the docs quote,
the split sums to the span, and no oracle or fitted threshold enters the verdict.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPE = json.loads((ROOT/"docs/evidence/m32-mass-match-r1/scope.json").read_text())
DECISION = json.loads((ROOT/"docs/evidence/m31-decision-rule-r2/scope.json").read_text())


class JoinTest(unittest.TestCase):
    def test_the_anchor_is_the_m31_arm(self):
        anchors = SCOPE["anchors"]
        identity = DECISION["reads"]["identity"]
        self.assertEqual(anchors["anchor_f1_at_zero"], identity["global_f1_at_zero"])
        self.assertEqual(anchors["anchor_best"], identity["global_best"])
        self.assertEqual(anchors["control_best"], 0.219626)
        self.assertEqual(anchors["reference_own_best"], 0.543974)
        self.assertEqual(SCOPE["population"]["fixed_cells"], 2986)

    def test_the_arm_is_the_declared_study(self):
        self.assertEqual(SCOPE["arm"]["run"], "m32-mass-v4")
        self.assertEqual(SCOPE["predeclaration"], "docs/M32_MASS_MATCH.md")


class GateTest(unittest.TestCase):
    def test_the_gate_block_is_the_standing_one(self):
        self.assertEqual(SCOPE["gates"], {"per_cell_f1": False, "ranking": False,
                                          "attribution": True})
        self.assertFalse(SCOPE["cleared"])
        self.assertLess(SCOPE["own"]["f1_at_zero"], 0.70)
        self.assertGreaterEqual(SCOPE["own"]["auc"], 0.90)
        self.assertLess(SCOPE["own_best"], 0.70)

    def test_no_oracle_or_fitted_threshold_enters_the_verdict(self):
        fields = {key: value for key, value in SCOPE.items() if key != "scope"}
        body = json.dumps(fields).lower()
        self.assertNotIn("oracle", body)
        self.assertNotIn("otsu", body)
        self.assertNotIn("label_free", body)

    def test_the_mass_term_moved_the_read_the_wrong_way(self):
        self.assertAlmostEqual(SCOPE["gain_over_anchor_f1"], -0.095104, places=6)
        self.assertAlmostEqual(SCOPE["gain_over_anchor_best"], -0.087577, places=6)
        self.assertAlmostEqual(SCOPE["auc_delta_vs_anchor"], -0.012205, places=6)
        self.assertAlmostEqual(SCOPE["gain_over_control"], 0.382572, places=6)


class SplitTest(unittest.TestCase):
    def test_the_split_sums_and_names_placement(self):
        span = SCOPE["anchors"]["span"]
        self.assertTrue(SCOPE["fractions_sum_to_one"])
        self.assertAlmostEqual(SCOPE["span_closed_fraction"]+SCOPE["placement_fraction"]+
                               SCOPE["representation_fraction"], 1.0, delta=2e-6)
        self.assertAlmostEqual(SCOPE["span_points"]+SCOPE["placement_points"]+
                               SCOPE["representation_points"], span, places=5)
        self.assertLess(SCOPE["placement_points"], 0.0)
        self.assertLess(SCOPE["placement_points"], SCOPE["span_points"])
        self.assertLess(SCOPE["placement_points"], SCOPE["representation_points"])
        self.assertEqual(SCOPE["short_stage"], "placement")

    def test_the_study_stayed_under_the_ceiling(self):
        self.assertLessEqual(SCOPE["peak_bytes"], 1152*1024**2)


if __name__ == "__main__":
    unittest.main()
