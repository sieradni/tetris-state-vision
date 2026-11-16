"""Integrity controls for the M50 board-box read receipt.

`docs/M50_BOARD_BOX.md` fixed the detector, its constants, the instrument check, the falsifier and the branches before
the run. These tests bind the receipt to that: the target baseline still reproduces the committed artifact, the
detected box is the declared label-free sampling of the same statistics, the branch is the predeclared arithmetic on
its own numbers (including the displacement gate it misses), the late profile is a like-for-like reading, and the
campaign's claim - that a label-free lattice buys the ceiling back - is rebuilt from the receipts.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from m50_board_box_read import branch_for  # noqa: E402

SCOPE = json.loads((ROOT/"docs/evidence/m50-board-box-read/scope.json").read_text())
DETECT = json.loads((ROOT/"runs/m50-board-box-detect-r9/detection.json").read_text())
M49 = json.loads((ROOT/"docs/evidence/m49-registration-read/scope.json").read_text())


class DeclarationOrderTest(unittest.TestCase):
    def test_the_declaration_precedes_the_receipt(self):
        self.assertEqual(SCOPE["predeclaration"], "docs/M50_BOARD_BOX.md")
        self.assertIn("predeclared before any scoring run", (ROOT/"docs/M50_BOARD_BOX.md").read_text())


class ProvenanceTest(unittest.TestCase):
    def test_the_readout_is_the_committed_one_and_reproduces_it(self):
        readout = SCOPE["readout"]
        self.assertEqual(readout["provenance"], "m3-occpix-ceiling-r1-r2")
        self.assertTrue(readout["parameters_imported_unchanged"])
        self.assertTrue(readout["reconstruction"]["committed_logits_reproduced"])
        self.assertLess(readout["reconstruction"]["max_abs_logit_difference"], 1e-6)

    def test_the_baselines_still_reproduce_the_earlier_receipts(self):
        target = SCOPE["measured"]["target_lattice"]
        predicted = SCOPE["measured"]["predicted_lattice"]
        self.assertAlmostEqual(target["f1_at_zero"], 0.974763, places=6)
        self.assertAlmostEqual(target["auc"], 0.999195, places=6)
        self.assertAlmostEqual(predicted["f1_at_zero"], 0.037152, places=6)
        self.assertAlmostEqual(predicted["auc"], 0.442341, places=6)

    def test_the_detection_is_label_free_and_declared(self):
        detection = SCOPE["geometry"]["detection"]
        self.assertTrue(detection["label_free"])
        # V-187: the frozen receipt carries the first draft's label; the executing code is pinned by digest
        self.assertEqual(detection["recipe"], "bright-frame interior of the captured well")
        self.assertEqual(detection["constants"]["edge"], 12.0)
        self.assertEqual(detection["constants"]["wall_share"], 0.5)
        self.assertEqual(len(detection["boards"]), 28)


class MeasuredTest(unittest.TestCase):
    def test_the_detected_box_readings_are_the_documented_ones(self):
        detected = SCOPE["measured"]["detected_box"]
        self.assertAlmostEqual(detected["f1_at_zero"], 0.9792, places=6)
        self.assertAlmostEqual(detected["auc"], 0.99915, places=6)
        self.assertEqual(detected["counts_at_zero"], {"true_positive": 306, "false_positive": 10,
                                                      "false_negative": 3})
        self.assertAlmostEqual(detected["best"]["f1"], 0.985507, places=6)

    def test_the_late_profile_is_like_for_like(self):
        bands = SCOPE["diagnosis"]["detected_box"]["bands"]
        rows = SCOPE["diagnosis"]["detected_box"]["false_positives_by_row"]
        self.assertEqual(sum(int(value) for value in rows.values()), 10)
        self.assertEqual(sum(int(rows[str(row)]) for row in range(10, 20)), 10)
        self.assertEqual(bands["rows_0_4"]+bands["rows_5_9"], 0)
        # unlike M49 the arm fires on nearly every occupied cell, so the small late count is real precision
        self.assertGreater(SCOPE["measured"]["detected_box"]["recall_at_zero"], 0.99)

    def test_the_margins_separate_the_classes(self):
        errors = SCOPE["diagnosis"]["detected_box"]["errors_at_zero"]
        self.assertGreater(errors["true_positive"]["margin_median"], 0.0)
        self.assertLess(errors["true_negative"]["margin_median"], 0.0)
        self.assertLess(errors["false_negative"]["margin_median"], -5.0)


class DetectorTest(unittest.TestCase):
    def test_the_receipt_and_the_instrument_check_agree(self):
        self.assertAlmostEqual(SCOPE["detector"]["mean_magnitude"], 0.802227, places=6)
        self.assertAlmostEqual(DETECT["summary"]["mean_magnitude"], 0.802227, places=6)
        self.assertAlmostEqual(DETECT["summary"]["median_of_medians"], 0.061328, places=6)
        self.assertAlmostEqual(DETECT["summary"]["worst_board_mean"], 13.509043, places=6)

    def test_the_two_named_captures_are_the_only_outliers(self):
        rows = sorted(DETECT["boards"], key=lambda row: -row["displacement"]["mean_magnitude"])
        self.assertEqual([row["sample_id"] for row in rows[:2]],
                         ["training-20261035-v2", "training-20261030-v1"])
        self.assertAlmostEqual(rows[2]["displacement"]["mean_magnitude"], 0.098, places=3)

    def test_the_scale_error_beat_the_recipe_it_replaced(self):
        self.assertLess(SCOPE["detector"]["mean_width_error_cells"], 0.1)
        self.assertLess(SCOPE["detector"]["mean_height_error_cells"], 0.1)
        self.assertGreater(M49["geometry_shift"]["registered_vs_target"]["mean_magnitude"], 8.0)
        self.assertLess(SCOPE["geometry_shift"]["detected_vs_target"]["mean_magnitude"], 1.0)


class BranchTest(unittest.TestCase):
    def test_the_receipt_lands_in_the_predeclared_branch_for_its_own_numbers(self):
        detected = SCOPE["measured"]["detected_box"]
        self.assertEqual(SCOPE["branch"]["verdict"],
                         branch_for(SCOPE["detector"]["mean_magnitude"], detected["f1_at_zero"], detected["auc"]))
        self.assertEqual(SCOPE["branch"]["verdict"], "box-partial")

    def test_only_the_displacement_gate_misses(self):
        detected = SCOPE["measured"]["detected_box"]
        branch = SCOPE["branch"]
        self.assertGreater(SCOPE["detector"]["mean_magnitude"], branch["detection_gate_cells"])
        self.assertGreaterEqual(detected["auc"], branch["auc_gate"])
        self.assertGreaterEqual(detected["f1_at_zero"], branch["f1_gate_at_zero"])


class CampaignClaimTest(unittest.TestCase):
    def test_a_label_free_lattice_buys_the_ceiling_back(self):
        target = SCOPE["measured"]["target_lattice"]
        detected = SCOPE["measured"]["detected_box"]
        predicted = SCOPE["measured"]["predicted_lattice"]
        control = SCOPE["anchors"]
        self.assertGreater(detected["f1_at_zero"], target["f1_at_zero"])
        self.assertLess(abs(target["auc"]-detected["auc"]), 0.001)
        self.assertGreater(detected["auc"]-predicted["auc"], 0.5)
        self.assertGreater(detected["auc"], control["control_auc"])
        self.assertGreater(detected["f1_at_zero"], control["control_f1_at_zero"])


if __name__ == "__main__":
    unittest.main()
