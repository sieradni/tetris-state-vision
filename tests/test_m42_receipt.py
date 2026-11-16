"""Integrity controls for the M42 decision-side receipt.

`docs/M42_SUPPORT_CROSSING.md` fixes the bar (`0.372881` / `0.689775`) and five branches before the run, and
declares the arm as the winner's configuration with `occupancy_support_crossing` as the only varied recorded
fact. These tests bind the receipt to that: the isolation is asserted against the matched control the pass
trained, the verdict is the predeclared harmful branch rather than a post-hoc reading, the split sums, and the
diagnosis is the one the document quotes.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPE = json.loads((ROOT/"docs/evidence/m42-support-crossing-r1/scope.json").read_text())
DIAGNOSIS = json.loads((ROOT/"docs/evidence/m42-support-crossing-r1/diagnosis.json").read_text())


def manifest(run):
    return json.loads((ROOT/"runs"/run/"checkpoint/manifest.json").read_text())


def effective(config):
    """An older manifest omits a field that later code records as its declared default."""
    return {key: (0 if value is None else value) for key, value in config.items()}


class IsolationTest(unittest.TestCase):
    def test_the_arm_is_the_winners_configuration_with_one_varied_fact(self):
        arm, control, winner = (manifest("m42-support-crossing"),
                                manifest("m42-control-winner-config"),
                                manifest("m30-level-calibrate-r5"))
        self.assertEqual(SCOPE["arm"]["run"], "m42-support-crossing")
        self.assertEqual(SCOPE["arm"]["inputs"], 30)
        self.assertEqual(arm["parameters"], control["parameters"])
        self.assertEqual(arm["parameters"], winner["parameters"])
        self.assertEqual(arm["parameters"], 8023562)
        differing = sorted(key for key in set(arm["config"]) | set(control["config"])
                           if effective(arm["config"]).get(key) != effective(control["config"]).get(key))
        self.assertEqual(differing, ["occupancy_support_crossing"])
        self.assertEqual((arm["config"]["occupancy_support_crossing"],
                          control["config"]["occupancy_support_crossing"]), (1, 0))

    def test_the_control_is_the_winners_configuration_recorded_under_current_code(self):
        control, winner = manifest("m42-control-winner-config"), manifest("m30-level-calibrate-r5")
        self.assertEqual(control["weights_sha256"], winner["weights_sha256"])
        self.assertEqual(control["parameters"], winner["parameters"])

    def test_the_arm_is_a_distinct_checkpoint(self):
        arm = manifest("m42-support-crossing")
        others = {run: manifest(run)["weights_sha256"] for run in
                  ("m30-level-calibrate-r5", "m42-control-winner-config", "m41-support-contrast",
                   "m40-composed-r2", "m34-support", "m36-monotone", "m38-capacity-r2")}
        self.assertNotIn(arm["weights_sha256"], set(others.values()))


class DeclaredTest(unittest.TestCase):
    def test_the_bar_is_the_winners_rescored_numbers(self):
        self.assertEqual(SCOPE["bar"]["f1_at_zero"], 0.372881)
        self.assertEqual(SCOPE["bar"]["best"], 0.689775)
        self.assertFalse(SCOPE["bar"]["met"])
        self.assertFalse(SCOPE["cleared"])

    def test_the_standing_gate_is_reported(self):
        self.assertEqual(SCOPE["gates"], {"per_cell_f1": False, "ranking": False, "attribution": True})
        self.assertEqual(SCOPE["gates_anchored"], SCOPE["gates"])

    def test_the_verdict_is_the_predeclared_harmful_branch(self):
        # harmful required dAUC < -0.005 or dF1@0 < -0.010; both held
        self.assertLess(SCOPE["auc_delta_vs_anchor"], -0.005)
        self.assertLess(SCOPE["gain_over_anchor_f1"], -0.010)
        # and it is not the inert branch: the read really changed
        self.assertNotEqual(SCOPE["own"]["auc"], SCOPE["anchors"]["anchor_auc"])


class MeasuredTest(unittest.TestCase):
    def test_the_receipt_is_the_one_the_document_quotes(self):
        self.assertAlmostEqual(SCOPE["own"]["f1_at_zero"], 0.189455, places=6)
        self.assertAlmostEqual(SCOPE["own"]["auc"], 0.838212, places=6)
        self.assertAlmostEqual(SCOPE["own_best"], 0.374034, places=6)
        self.assertAlmostEqual(SCOPE["own"]["best"]["threshold"], 1.65, places=6)

    def test_the_deltas_are_the_winners_own_numbers(self):
        self.assertAlmostEqual(SCOPE["gain_over_anchor_f1"], -0.183426, places=6)
        self.assertAlmostEqual(SCOPE["auc_delta_vs_anchor"], -0.116317, places=6)
        self.assertAlmostEqual(SCOPE["gain_over_anchor_best"], -0.315741, places=6)

    def test_the_split_sums_to_the_measured_span(self):
        self.assertTrue(SCOPE["fractions_sum_to_one"])
        self.assertAlmostEqual(SCOPE["span_closed_fraction"]+SCOPE["placement_fraction"]+
                               SCOPE["representation_fraction"], 1.0, places=6)
        self.assertAlmostEqual(SCOPE["span_points"]+SCOPE["placement_points"]+
                               SCOPE["representation_points"], SCOPE["anchors"]["span"], places=5)
        self.assertEqual(SCOPE["short_stage"], "closure")

    def test_the_harmful_signature_is_the_one_the_document_records(self):
        errors = DIAGNOSIS["errors_at_zero"]
        self.assertEqual(errors["false_positive"]["count"], 1812)
        self.assertAlmostEqual(errors["false_positive"]["margin_median"], 0.835996, places=6)
        self.assertAlmostEqual(errors["true_positive"]["margin_low"], 0.048825, places=6)
        bands = DIAGNOSIS["false_positive_bands"]
        self.assertEqual(bands["rows_10_14"]["false_positives"], 473)
        self.assertEqual(bands["rows_15_19"]["false_positives"], 475)
        self.assertFalse(SCOPE["comparison_to_m35"]["textured_empty_separated"])


if __name__ == "__main__":
    unittest.main()
