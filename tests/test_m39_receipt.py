"""Integrity controls for the M39 dual-placement receipt.

`docs/M39_DUAL_PLACEMENT.md` fixes the bar (`0.372881` / `0.689775`) before the run and pre-commits the three
possible readings of a refusal. These tests bind the receipt to those commitments: the width is `60`, the
gate block is the standing one, the deltas are the winner's re-scored numbers, the split sums, and the
diagnosis distinguishes swamped from absent using the numbers the document quotes.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPE = json.loads((ROOT/"docs/evidence/m39-dual-placement-r1/scope.json").read_text())
DIAGNOSIS = json.loads((ROOT/"docs/evidence/m39-dual-placement-r1/diagnosis.json").read_text())
M35 = json.loads((ROOT/"docs/evidence/m35-contrast-lattice-r1/scope.json").read_text())
M37 = json.loads((ROOT/"docs/evidence/m37-stencil-fusion-r2/scope.json").read_text())


class DeclaredTest(unittest.TestCase):
    def test_the_arm_is_the_declared_sixty_input_form(self):
        self.assertEqual(SCOPE["arm"]["inputs"], 60)
        self.assertEqual(SCOPE["arm"]["features"], 30)
        self.assertEqual(SCOPE["predeclaration"], "docs/M39_DUAL_PLACEMENT.md")

    def test_the_bar_is_the_winners_rescored_numbers(self):
        self.assertEqual(SCOPE["bar"]["f1_at_zero"], 0.372881)
        self.assertEqual(SCOPE["bar"]["best"], 0.689775)
        self.assertFalse(SCOPE["bar"]["met"])
        self.assertAlmostEqual(SCOPE["anchors"]["anchor_auc"], 0.954529, places=6)


class MeasuredTest(unittest.TestCase):
    def test_the_receipt_is_the_one_the_document_quotes(self):
        self.assertAlmostEqual(SCOPE["own"]["f1_at_zero"], 0.246228, places=6)
        self.assertAlmostEqual(SCOPE["own"]["auc"], 0.836907, places=6)
        self.assertAlmostEqual(SCOPE["own_best"], 0.351779, places=6)
        self.assertAlmostEqual(SCOPE["gain_over_anchor_best"], -0.337996, places=6)
        self.assertAlmostEqual(SCOPE["auc_delta_vs_anchor"], -0.117621, places=6)

    def test_the_standing_gate_is_reported(self):
        self.assertEqual(SCOPE["gates"], {"per_cell_f1": False, "ranking": False, "attribution": True})
        self.assertGreaterEqual(SCOPE["gain_over_control"], 0.05)


class SwampedTest(unittest.TestCase):
    def test_the_signal_is_neither_absent_nor_merely_insufficient(self):
        comparison = SCOPE["comparison_to_m35"]
        self.assertFalse(comparison["textured_empty_separated"])
        self.assertGreater(comparison["false_positives_now"], comparison["false_positives_before"])
        self.assertGreater(comparison["false_positive_median_now"],
                           comparison["false_positive_median_before"])
        self.assertGreater(comparison["false_positive_median_now"],
                           comparison["weakest_true_positive_margin_now"])
        self.assertGreater(comparison["per_band"]["rows_15_19"]["now"],
                           comparison["per_band"]["rows_15_19"]["before"])

    def test_it_is_the_weakest_arm_on_both_reading_measures(self):
        self.assertLess(SCOPE["own"]["auc"], M35["own"]["auc"])
        self.assertLess(SCOPE["own"]["auc"], M37["own"]["auc"])
        self.assertLess(SCOPE["own_best"], M35["own_best"])
        self.assertLess(SCOPE["own_best"], M37["own_best"])


class DiagnosisTest(unittest.TestCase):
    def test_the_diagnosis_accounts_for_every_fixed_cell(self):
        self.assertEqual(DIAGNOSIS["cells"], 2986)
        self.assertEqual(sum(entry["count"] for entry in DIAGNOSIS["errors_at_zero"].values()), 2986)
        self.assertEqual(DIAGNOSIS["occupied_cells"], 214)


class SplitTest(unittest.TestCase):
    def test_the_split_sums_to_the_span_and_to_one(self):
        span = SCOPE["anchors"]["span"]
        self.assertTrue(SCOPE["fractions_sum_to_one"])
        self.assertAlmostEqual(SCOPE["span_closed_fraction"]+SCOPE["placement_fraction"]+
                               SCOPE["representation_fraction"], 1.0, delta=2e-6)
        self.assertAlmostEqual(SCOPE["span_points"]+SCOPE["placement_points"]+
                               SCOPE["representation_points"], span, places=5)
        self.assertLessEqual(SCOPE["peak_bytes"], 1152*1024**2)


if __name__ == "__main__":
    unittest.main()
