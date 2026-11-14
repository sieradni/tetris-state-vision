"""Integrity controls for the M30 operating-point receipt.

`docs/M30_OPERATING_POINT.md` predeclares three level mechanisms, a 0.05 confirmation floor and the
campaign gate. These tests bind the receipt's claims: the anchor is the M29 winner's own re-scored
read, only one mechanism clears the floor, the split sums to the span, the gate verdict follows the
declared rule, and the level parameters are reported for every form.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPE = json.loads((ROOT/"docs/evidence/m30-operating-point-r1/scope.json").read_text())
TENSORS = json.loads((ROOT/"docs/evidence/m30-level-tensors-r1/scope.json").read_text())
M29 = json.loads((ROOT/"docs/evidence/m29-pixel-form-r2/scope.json").read_text())


class AnchorTest(unittest.TestCase):
    def test_the_anchor_is_the_m29_winner(self):
        anchors = SCOPE["anchors"]
        committed = M29["variants"]["hidden-r2"]["own"]
        self.assertEqual(anchors["anchor_run"], "m29-pixelform-hidden-r2")
        self.assertEqual(anchors["anchor_f1_at_zero"], round(committed["f1_at_zero"], 6))
        self.assertEqual(anchors["anchor_auc"], round(committed["auc"], 6))
        self.assertEqual(anchors["anchor_best"], round(committed["best"]["f1"], 6))
        self.assertEqual(anchors["control_best"], 0.219626)
        self.assertEqual(anchors["reference_own_best"], 0.543974)
        self.assertEqual(anchors["span"], 0.443205)

    def test_every_form_is_scored_on_the_fixed_population(self):
        self.assertEqual(SCOPE["population"]["fixed_cells"], 2986)
        self.assertEqual((SCOPE["population"]["occupied_identity"],
                          SCOPE["population"]["occupied_anchored"]), (214, 309))
        for name, item in SCOPE["variants"].items():
            self.assertEqual(item["own"]["cells"], 2986, name)
            self.assertEqual(item["anchored"]["cells"], 2986, name)


class MechanismTest(unittest.TestCase):
    def test_only_the_calibration_clears_the_floor(self):
        decision = SCOPE["decision"]
        self.assertEqual(decision["confirmed"], ["calibrate"])
        self.assertEqual(sorted(decision["excluded"]), ["balance", "bias"])
        self.assertGreaterEqual(SCOPE["variants"]["calibrate"]["gain_over_anchor"],
                                decision["gain_floor"])
        for name in ("bias", "balance"):
            self.assertLess(SCOPE["variants"][name]["gain_over_anchor"], decision["gain_floor"])

    def test_the_learned_scale_stayed_positive_and_the_gate_is_reported(self):
        self.assertGreater(TENSORS["forms"]["bias"]["scale"], 0.0)
        self.assertTrue(TENSORS["forms"]["bias"]["ordering_preserved"])
        self.assertGreaterEqual(TENSORS["forms"]["calibrate"]["gate"], 0.0)
        for name in ("bias", "balance", "calibrate"):
            self.assertIn(name, SCOPE["variants"])
            self.assertTrue(SCOPE["variants"][name]["level_parameters"])


class SplitTest(unittest.TestCase):
    def test_every_split_reproduces_the_span(self):
        span = SCOPE["anchors"]["span"]
        for name, item in SCOPE["variants"].items():
            self.assertTrue(item["fractions_sum_to_one"], name)
            self.assertAlmostEqual(item["span_closed_fraction"]+item["placement_fraction"]+
                                   item["representation_fraction"], 1.0, delta=2e-6)
            self.assertAlmostEqual(item["span_points"]+item["placement_points"]+
                                   item["representation_points"], span, places=5)
            self.assertAlmostEqual(item["span_points"], item["own_best"]-0.543974, places=5)


class GateTest(unittest.TestCase):
    def test_the_gate_verdict_follows_the_declared_rule(self):
        decision, winner = SCOPE["decision"], SCOPE["variants"]["calibrate"]
        self.assertEqual(decision["gates"], {"per_cell_f1": False, "ranking": False,
                                             "attribution": True})
        self.assertFalse(decision["cleared"])
        self.assertEqual(decision["cause"], "level-inconclusive:calibrate")
        self.assertLess(winner["own"]["f1_at_zero"], 0.70)
        self.assertGreaterEqual(winner["own"]["auc"], 0.90)
        self.assertLess(winner["own_best"], 0.70)
        self.assertGreaterEqual(winner["gain_over_control"], 0.05)

    def test_the_calibration_gain_is_recorded_as_operating_point_evidence(self):
        winner = SCOPE["variants"]["calibrate"]
        self.assertAlmostEqual(winner["gain_over_anchor"], 0.081417, places=6)
        self.assertLess(abs(winner["auc_delta_vs_anchor"]), 0.01)
        self.assertIn("operating point", SCOPE["decision"]["reading"])

    def test_the_studies_stay_under_the_ceiling(self):
        for name, item in SCOPE["variants"].items():
            self.assertLessEqual(item["peak_bytes"], 1152*1024**2, name)


if __name__ == "__main__":
    unittest.main()
