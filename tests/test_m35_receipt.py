"""Integrity controls for the M35 contrast read and the M36 monotone contrast receipts.

`docs/M35_CONTRAST_READ.md` fixes the bar before scoring and `docs/M36_MONOTONE_CONTRAST.md` fixes the
same bar for the structural arm. These tests bind both receipts: same winner, same population, the
standing gate block, the measured numbers the documents quote, the diagnosis accounting for every cell,
and splits that sum to the span and to one.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M35 = json.loads((ROOT/"docs/evidence/m35-contrast-lattice-r1/scope.json").read_text())
M35_DIAGNOSIS = json.loads((ROOT/"docs/evidence/m35-contrast-lattice-r1/diagnosis.json").read_text())
M36 = json.loads((ROOT/"docs/evidence/m36-monotone-contrast-r1/scope.json").read_text())
M36_DIAGNOSIS = json.loads((ROOT/"docs/evidence/m36-monotone-contrast-r1/diagnosis.json").read_text())
M34 = json.loads((ROOT/"docs/evidence/m34-support-lattice-r1/scope.json").read_text())


class BarTest(unittest.TestCase):
    def test_the_bar_is_the_winners_rescored_numbers(self):
        for receipt in (M35, M36):
            self.assertEqual(receipt["bar"]["f1_at_zero"], receipt["anchors"]["anchor_f1_at_zero"])
            self.assertEqual(receipt["bar"]["best"], receipt["anchors"]["anchor_best"])
            self.assertFalse(receipt["bar"]["met"])

    def test_the_declared_input_width_is_thirty_features_twice(self):
        self.assertEqual(M35["arm"]["features"], 30)
        self.assertEqual(M35["arm"]["inputs"], 60)
        self.assertAlmostEqual(M35["own_best"], 0.570048, places=6)
        self.assertAlmostEqual(M36["own_best"], 0.420935, places=6)


class JoinTest(unittest.TestCase):
    def test_both_arms_join_the_same_winner_and_population(self):
        for receipt in (M35, M36):
            self.assertEqual(receipt["anchors"]["anchor_run"], "m30-level-calibrate-r5")
            self.assertEqual(receipt["population"]["fixed_cells"], 2986)
            self.assertEqual(receipt["population"]["occupied_identity"], 214)
            self.assertEqual(receipt["population"]["occupied_anchored"], 309)
            self.assertEqual(receipt["anchors"]["anchor_best"], M34["anchors"]["anchor_best"])

    def test_the_standing_gate_is_reported_on_both_vectors(self):
        for receipt in (M35, M36):
            for key in ("gates", "gates_anchored"):
                self.assertEqual(receipt[key], {"per_cell_f1": False, "ranking": False,
                                                "attribution": True}, key)
            self.assertGreaterEqual(receipt["gain_over_control"], 0.05)


class MeasuredTest(unittest.TestCase):
    def test_the_contrast_receipt_is_the_one_the_document_quotes(self):
        self.assertAlmostEqual(M35["own"]["f1_at_zero"], 0.277778, places=6)
        self.assertAlmostEqual(M35["own"]["auc"], 0.945596, places=6)
        self.assertAlmostEqual(M35["gain_over_anchor_best"], -0.119726, places=6)
        self.assertAlmostEqual(M35["span_points"], 0.026074, places=6)

    def test_the_monotone_receipt_is_the_one_the_document_quotes(self):
        self.assertAlmostEqual(M36["own"]["f1_at_zero"], 0.313667, places=6)
        self.assertAlmostEqual(M36["own"]["auc"], 0.886350, places=6)
        self.assertAlmostEqual(M36["gain_over_anchor_best"], -0.268839, places=6)
        self.assertAlmostEqual(M36["span_points"], -0.123039, places=6)
        self.assertLess(M36["own_best"], M35["own_best"])


class DiagnosisTest(unittest.TestCase):
    def test_each_diagnosis_accounts_for_every_fixed_cell(self):
        for diagnosis in (M35_DIAGNOSIS, M36_DIAGNOSIS):
            self.assertEqual(diagnosis["cells"], 2986)
            self.assertEqual(diagnosis["boards"], 28)
            self.assertEqual(sum(entry["count"] for entry in diagnosis["errors_at_zero"].values()), 2986)
            self.assertEqual(diagnosis["occupied_cells"], 214)

    def test_the_monotone_envelope_clears_the_lower_bands(self):
        bands_35 = M35_DIAGNOSIS["false_positive_bands"]
        bands_36 = M36_DIAGNOSIS["false_positive_bands"]
        self.assertEqual(bands_36["rows_10_14"]["false_positives"], 0)
        self.assertEqual(bands_36["rows_15_19"]["false_positives"], 0)
        self.assertGreater(bands_35["rows_15_19"]["false_positives"], 0)
        self.assertGreater(M36_DIAGNOSIS["precision_at_zero"], M35_DIAGNOSIS["precision_at_zero"])


class SplitTest(unittest.TestCase):
    def test_both_splits_sum_to_the_span_and_to_one(self):
        for receipt in (M35, M36):
            span = receipt["anchors"]["span"]
            self.assertTrue(receipt["fractions_sum_to_one"])
            self.assertAlmostEqual(receipt["span_closed_fraction"]+receipt["placement_fraction"]+
                                   receipt["representation_fraction"], 1.0, delta=2e-6)
            self.assertAlmostEqual(receipt["span_points"]+receipt["placement_points"]+
                                   receipt["representation_points"], span, places=5)
            self.assertLessEqual(receipt["peak_bytes"], 1152*1024**2)


if __name__ == "__main__":
    unittest.main()
