"""Integrity controls for the M33 fixed-lattice receipt.

`docs/M33_FIXED_LATTICE.md` declares one spatial change, the unchanged gate and both branch
interpretations. These tests bind the receipt: it joins the winner on the same cells, both fixed label
vectors are reported, the gate block is the standing one on both, the measured result is the refusal
the docs quote, and the split sums to the span.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPE = json.loads((ROOT/"docs/evidence/m33-fixed-lattice-r1/scope.json").read_text())
M32 = json.loads((ROOT/"docs/evidence/m32-mass-match-r1/scope.json").read_text())


class JoinTest(unittest.TestCase):
    def test_the_anchor_is_the_same_winner_as_the_previous_pass(self):
        self.assertEqual(SCOPE["anchors"]["anchor_run"], "m30-level-calibrate-r5")
        self.assertEqual(SCOPE["anchors"]["anchor_best"], M32["anchors"]["anchor_best"])
        self.assertEqual(SCOPE["anchors"]["anchor_f1_at_zero"], M32["anchors"]["anchor_f1_at_zero"])
        self.assertEqual(SCOPE["population"]["fixed_cells"], 2986)

    def test_both_fixed_label_vectors_are_reported(self):
        self.assertEqual(SCOPE["population"]["occupied_identity"], 214)
        self.assertEqual(SCOPE["population"]["occupied_anchored"], 309)
        for key in ("own", "anchored"):
            self.assertEqual(SCOPE[key]["cells"], 2986, key)

    def test_the_arm_is_the_declared_study(self):
        self.assertEqual(SCOPE["arm"]["run"], "m33-fixed")
        self.assertEqual(SCOPE["predeclaration"], "docs/M33_FIXED_LATTICE.md")


class GateTest(unittest.TestCase):
    def test_the_standing_gate_is_reported_on_both_vectors(self):
        for key in ("gates", "gates_anchored"):
            self.assertEqual(SCOPE[key], {"per_cell_f1": False, "ranking": False,
                                          "attribution": True}, key)
        self.assertFalse(SCOPE["cleared"])
        self.assertFalse(SCOPE["cleared_identity"])
        self.assertFalse(SCOPE["cleared_anchored"])

    def test_no_oracle_or_fitted_threshold_enters_the_verdict(self):
        fields = {key: value for key, value in SCOPE.items() if key != "scope"}
        body = json.dumps(fields).lower()
        for forbidden in ("oracle", "otsu", "label_free"):
            self.assertNotIn(forbidden, body)

    def test_the_measured_result_is_the_refusal_the_docs_quote(self):
        self.assertAlmostEqual(SCOPE["own"]["f1_at_zero"], 0.255630, places=6)
        self.assertAlmostEqual(SCOPE["own_best"], 0.632327, places=6)
        self.assertAlmostEqual(SCOPE["anchored_best"], 0.513678, places=6)
        self.assertAlmostEqual(SCOPE["gain_over_anchor_f1"], -0.117251, places=6)
        self.assertAlmostEqual(SCOPE["gain_over_anchor_best"], -0.057448, places=6)
        self.assertAlmostEqual(SCOPE["auc_delta_vs_anchor"], -0.009444, places=6)


class SplitTest(unittest.TestCase):
    def test_the_split_sums_and_placement_is_still_short(self):
        span = SCOPE["anchors"]["span"]
        self.assertTrue(SCOPE["fractions_sum_to_one"])
        self.assertAlmostEqual(SCOPE["span_closed_fraction"]+SCOPE["placement_fraction"]+
                               SCOPE["representation_fraction"], 1.0, delta=2e-6)
        self.assertAlmostEqual(SCOPE["span_points"]+SCOPE["placement_points"]+
                               SCOPE["representation_points"], span, places=5)
        self.assertLess(SCOPE["placement_points"], 0.0)
        self.assertLess(SCOPE["placement_points"], M32["placement_points"])
        self.assertEqual(SCOPE["short_stage"], "placement")
        self.assertLessEqual(SCOPE["peak_bytes"], 1152*1024**2)


if __name__ == "__main__":
    unittest.main()
