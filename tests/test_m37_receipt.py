"""Integrity controls for the M37 fusion and M38 capacity receipts.

`docs/M37_STENCIL_FUSION.md` and `docs/M38_CAPACITY.md` fix the same promotion bar before their runs
(`0.372881` / `0.689775`, the winner's re-scored numbers). These tests bind both receipts: the widths are
stated by the arm and not inherited, the ladder numbers are the ones the documents quote, the gate block is
the standing one, the diagnoses account for every cell, and the splits sum to the span and to one.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M37 = json.loads((ROOT/"docs/evidence/m37-stencil-fusion-r2/scope.json").read_text())
M37_DIAGNOSIS = json.loads((ROOT/"docs/evidence/m37-stencil-fusion-r2/diagnosis.json").read_text())
M38 = json.loads((ROOT/"docs/evidence/m38-capacity-r1/scope.json").read_text())
M38_DIAGNOSIS = json.loads((ROOT/"docs/evidence/m38-capacity-r1/diagnosis.json").read_text())
M35 = json.loads((ROOT/"docs/evidence/m35-contrast-lattice-r1/scope.json").read_text())


class WidthTest(unittest.TestCase):
    def test_each_receipt_states_its_own_declared_width(self):
        self.assertEqual(M37["arm"]["inputs"], 40)
        self.assertEqual(M37["arm"]["features"], 30)
        self.assertEqual(M38["arm"]["inputs"], 40)
        self.assertNotEqual(M37["arm"]["inputs"], M35["arm"]["inputs"])

    def test_the_bar_is_the_winners_rescored_numbers(self):
        for receipt in (M37, M38):
            self.assertEqual(receipt["bar"]["f1_at_zero"], 0.372881)
            self.assertEqual(receipt["bar"]["best"], 0.689775)
            self.assertFalse(receipt["bar"]["met"])


class LadderTest(unittest.TestCase):
    def test_the_measured_rows_are_the_ones_the_documents_quote(self):
        self.assertAlmostEqual(M37["own"]["f1_at_zero"], 0.270096, places=6)
        self.assertAlmostEqual(M37["own"]["auc"], 0.919512, places=6)
        self.assertAlmostEqual(M37["own_best"], 0.589286, places=6)
        self.assertAlmostEqual(M38["own"]["f1_at_zero"], 0.273082, places=6)
        self.assertAlmostEqual(M38["own"]["auc"], 0.895041, places=6)
        self.assertAlmostEqual(M38["own_best"], 0.463551, places=6)

    def test_capacity_was_worse_at_the_same_inputs(self):
        self.assertLess(M38["own_best"], M37["own_best"])
        self.assertLess(M38["own"]["auc"], M37["own"]["auc"])
        self.assertLess(M38["gain_over_anchor_best"], M37["gain_over_anchor_best"])

    def test_every_support_lattice_arm_ranks_below_the_winner(self):
        for receipt in (M37, M38):
            self.assertLess(receipt["own"]["auc"], receipt["anchors"]["anchor_auc"])
            self.assertLess(receipt["own_best"], receipt["anchors"]["anchor_best"])


class DiagnosisTest(unittest.TestCase):
    def test_the_coarse_channel_did_not_separate_the_textured_empty_cells(self):
        self.assertFalse(M37["comparison_to_m35"]["textured_empty_separated"])
        self.assertGreaterEqual(M37["comparison_to_m35"]["false_positives_now"],
                                M37["comparison_to_m35"]["false_positives_before"])
        self.assertGreater(M37["comparison_to_m35"]["false_positive_median_now"],
                           M37["comparison_to_m35"]["false_positive_median_before"])

    def test_each_diagnosis_accounts_for_every_fixed_cell(self):
        for diagnosis in (M37_DIAGNOSIS, M38_DIAGNOSIS):
            self.assertEqual(diagnosis["cells"], 2986)
            self.assertEqual(sum(entry["count"] for entry in diagnosis["errors_at_zero"].values()), 2986)
            self.assertEqual(diagnosis["occupied_cells"], 214)


class SplitTest(unittest.TestCase):
    def test_both_splits_sum_to_the_span_and_to_one(self):
        for receipt in (M37, M38):
            span = receipt["anchors"]["span"]
            self.assertTrue(receipt["fractions_sum_to_one"])
            self.assertAlmostEqual(receipt["span_closed_fraction"]+receipt["placement_fraction"]+
                                   receipt["representation_fraction"], 1.0, delta=2e-6)
            self.assertAlmostEqual(receipt["span_points"]+receipt["placement_points"]+
                                   receipt["representation_points"], span, places=5)
            self.assertLessEqual(receipt["peak_bytes"], 1152*1024**2)


if __name__ == "__main__":
    unittest.main()
