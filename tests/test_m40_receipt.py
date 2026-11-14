"""Integrity controls for the M40 composed-alignment receipt.

`docs/M40_COMPOSED_ALIGNMENT.md` fixes the bar (`0.372881` / `0.689775`) before the run, declares the arm as
a half-strength blend that adds no parameter, and pre-commits the reading of a refusal. These tests bind the
receipt to those commitments: the width is `30`, the standing gate block is reported, the deltas are the
winner's re-scored numbers, the split sums, and the diagnosis is the swamped reading the document records.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPE = json.loads((ROOT/"docs/evidence/m40-composed-alignment-r6/scope.json").read_text())
DIAGNOSIS = json.loads((ROOT/"docs/evidence/m40-composed-alignment-r6/diagnosis.json").read_text())
M34 = json.loads((ROOT/"docs/evidence/m34-support-lattice-r1/scope.json").read_text())
M35 = json.loads((ROOT/"docs/evidence/m35-contrast-lattice-r1/scope.json").read_text())


class DeclaredTest(unittest.TestCase):
    def test_the_arm_is_the_declared_thirty_input_form(self):
        self.assertEqual(SCOPE["arm"]["inputs"], 30)
        self.assertEqual(SCOPE["arm"]["features"], 30)
        self.assertEqual(SCOPE["predeclaration"], "docs/M40_COMPOSED_ALIGNMENT.md")

    def test_the_bar_is_the_winners_rescored_numbers(self):
        self.assertEqual(SCOPE["bar"]["f1_at_zero"], 0.372881)
        self.assertEqual(SCOPE["bar"]["best"], 0.689775)
        self.assertFalse(SCOPE["bar"]["met"])
        self.assertAlmostEqual(SCOPE["anchors"]["anchor_auc"], 0.954529, places=6)


class MeasuredTest(unittest.TestCase):
    def test_the_receipt_is_the_one_the_document_quotes(self):
        self.assertAlmostEqual(SCOPE["own"]["f1_at_zero"], 0.260599, places=6)
        self.assertAlmostEqual(SCOPE["own"]["auc"], 0.897918, places=6)
        self.assertAlmostEqual(SCOPE["own_best"], 0.439535, places=6)
        self.assertAlmostEqual(SCOPE["own"]["best"]["threshold"], 1.65, places=6)

    def test_the_deltas_are_the_winners_own_numbers(self):
        self.assertAlmostEqual(SCOPE["gain_over_anchor_f1"], -0.112283, places=6)
        self.assertAlmostEqual(SCOPE["auc_delta_vs_anchor"], -0.056611, places=6)
        self.assertAlmostEqual(SCOPE["gain_over_anchor_best"], -0.250240, places=6)

    def test_the_standing_gate_is_reported(self):
        self.assertEqual(SCOPE["gates"], {"per_cell_f1": False, "ranking": False, "attribution": True})
        self.assertEqual(SCOPE["gates_anchored"], SCOPE["gates"])
        self.assertFalse(SCOPE["cleared"])

    def test_the_split_sums_to_the_measured_span(self):
        self.assertTrue(SCOPE["fractions_sum_to_one"])
        self.assertAlmostEqual(SCOPE["span_closed_fraction"]+SCOPE["placement_fraction"]+
                               SCOPE["representation_fraction"], 1.0, places=6)
        # the components are rounded to six places in the receipt, so their printed sum can differ from the
        # printed span by one unit in the sixth place; the decomposition is asserted, not re-derived
        self.assertAlmostEqual(SCOPE["span_points"]+SCOPE["placement_points"]+
                               SCOPE["representation_points"], SCOPE["anchors"]["span"], places=5)
        self.assertEqual(SCOPE["short_stage"], "closure")


class BetweenTheEndpointsTest(unittest.TestCase):
    """The dose-response the document reads: ranking falls with the share of replaced geometry."""

    def test_the_arm_sits_between_the_two_measured_endpoints(self):
        winner = SCOPE["anchors"]["anchor_auc"]
        support = M34["own"]["auc"]
        self.assertGreater(SCOPE["own"]["auc"], support)
        self.assertLess(SCOPE["own"]["auc"], winner)
        self.assertGreater(SCOPE["own_best"], M34["own_best"])
        self.assertLess(SCOPE["own_best"], SCOPE["anchors"]["anchor_best"])

    def test_the_losing_family_is_the_one_that_moves_the_sampled_geometry(self):
        # arms that keep the predicted lattice sampled land above every arm that moves off it
        keeping = [M35["own"]["auc"]]
        moving = [M34["own"]["auc"], SCOPE["own"]["auc"]]
        self.assertGreater(min(keeping), max(moving))
        self.assertGreater(min([M35["own_best"]]), max([M34["own_best"], SCOPE["own_best"]]))

    def test_the_swamped_signature_is_the_one_the_document_records(self):
        errors = DIAGNOSIS["errors_at_zero"]
        self.assertEqual(errors["false_positive"]["count"], 1181)
        self.assertAlmostEqual(errors["false_positive"]["margin_median"], 1.233860, places=6)
        self.assertLess(errors["false_positive"]["margin_median"],
                        errors["true_positive"]["margin_median"])
        bands = DIAGNOSIS["false_positive_bands"]
        self.assertEqual(bands["rows_0_4"]["false_positives"]+bands["rows_5_9"]["false_positives"], 910)
        self.assertEqual(bands["rows_15_19"]["false_positives"], 136)
        self.assertAlmostEqual(DIAGNOSIS["errors_at_zero"]["true_positive"]["margin_low"], 0.168971,
                               places=6)

    def test_the_textured_empty_reading_matches_the_document(self):
        comparison = SCOPE["comparison_to_m35"]
        self.assertFalse(comparison["textured_empty_separated"])
        self.assertEqual(comparison["false_positives_before"], 1088)
        self.assertEqual(comparison["false_positives_now"], 1181)


if __name__ == "__main__":
    unittest.main()
