"""Integrity controls for the M34 support-lattice receipt.

`docs/M34_SUPPORT_LATTICE.md` declares the placement, the unchanged gate, and the bar that decides
placement reducibility before any run. These tests bind the receipt: it joins the same winner on the same
cells, the gate block is the standing one, the measured refusal and the reducibility verdict are the ones
the docs quote, and the split sums to the span and to one.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPE = json.loads((ROOT/"docs/evidence/m34-support-lattice-r1/scope.json").read_text())
M30 = json.loads((ROOT/"docs/evidence/m30-operating-point-r1/scope.json").read_text())
M32 = json.loads((ROOT/"docs/evidence/m32-mass-match-r1/scope.json").read_text())
M33 = json.loads((ROOT/"docs/evidence/m33-fixed-lattice-r1/scope.json").read_text())


class JoinTest(unittest.TestCase):
    def test_the_anchor_is_the_campaign_winner(self):
        self.assertEqual(SCOPE["anchors"]["anchor_run"], "m30-level-calibrate-r5")
        self.assertEqual(SCOPE["anchors"]["anchor_best"], M33["anchors"]["anchor_best"])
        self.assertEqual(SCOPE["population"]["fixed_cells"], 2986)

    def test_both_fixed_label_vectors_are_reported(self):
        self.assertEqual(SCOPE["population"]["occupied_identity"], 214)
        self.assertEqual(SCOPE["population"]["occupied_anchored"], 309)
        for key in ("own", "anchored"):
            self.assertEqual(SCOPE[key]["cells"], 2986, key)


class GateTest(unittest.TestCase):
    def test_the_standing_gate_is_reported_on_both_vectors(self):
        for key in ("gates", "gates_anchored"):
            self.assertEqual(SCOPE[key], {"per_cell_f1": False, "ranking": False,
                                          "attribution": True}, key)
        self.assertFalse(SCOPE["cleared"])
        self.assertGreaterEqual(SCOPE["gain_over_control"], 0.05)

    def test_the_measured_result_is_the_refusal_the_docs_quote(self):
        self.assertAlmostEqual(SCOPE["own"]["f1_at_zero"], 0.274869, places=6)
        self.assertAlmostEqual(SCOPE["own_best"], 0.385925, places=6)
        self.assertAlmostEqual(SCOPE["anchored_best"], 0.381148, places=6)
        self.assertAlmostEqual(SCOPE["gain_over_anchor_f1"], -0.098012, places=6)
        self.assertAlmostEqual(SCOPE["gain_over_anchor_best"], -0.303850, places=6)
        self.assertAlmostEqual(SCOPE["auc_delta_vs_anchor"], -0.071268, places=6)


class PlacementRuleTest(unittest.TestCase):
    def test_the_reducibility_verdict_follows_the_predeclared_bar(self):
        floor = SCOPE["placement_reducible_floor"]
        self.assertEqual(floor["floor"], 0.02)
        self.assertAlmostEqual(floor["best_previous"], -0.103647, places=6)
        self.assertAlmostEqual(floor["bar"], -0.083647, places=6)
        self.assertAlmostEqual(SCOPE["placement_points"], -0.004778, places=6)
        self.assertTrue(SCOPE["placement_reducible"])
        self.assertGreaterEqual(SCOPE["placement_points"], floor["bar"]+0.078)

    def test_every_recipe_in_the_comparison_is_a_recorded_number(self):
        recipes = SCOPE["placement_recipes"]
        self.assertAlmostEqual(recipes["sampled-at-predicted (m30 winner)"],
                               M30["variants"]["calibrate"]["placement_points"], places=6)
        self.assertAlmostEqual(recipes["sampled-at-predicted (m32 mass arm)"],
                               M32["placement_points"], places=6)
        self.assertAlmostEqual(recipes["constant-over-capture (m33 fixed)"],
                               M33["placement_points"], places=6)
        self.assertAlmostEqual(recipes["support-derived (m34 this arm)"],
                               SCOPE["placement_points"], places=6)
        self.assertAlmostEqual(SCOPE["placement_band"], 0.113871, places=6)


class SplitTest(unittest.TestCase):
    def test_the_split_sums_and_closure_is_the_short_stage(self):
        span = SCOPE["anchors"]["span"]
        self.assertTrue(SCOPE["fractions_sum_to_one"])
        self.assertAlmostEqual(SCOPE["span_closed_fraction"]+SCOPE["placement_fraction"]+
                               SCOPE["representation_fraction"], 1.0, delta=2e-6)
        self.assertAlmostEqual(SCOPE["span_points"]+SCOPE["placement_points"]+
                               SCOPE["representation_points"], span, places=5)
        self.assertEqual(SCOPE["short_stage"], "closure")
        self.assertGreater(SCOPE["placement_points"], M33["placement_points"])
        self.assertLess(abs(SCOPE["placement_fraction"]), 0.02)
        self.assertLessEqual(SCOPE["peak_bytes"], 1152*1024**2)


if __name__ == "__main__":
    unittest.main()
