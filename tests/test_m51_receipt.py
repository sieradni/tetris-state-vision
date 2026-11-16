"""Integrity controls for the M51 detector-repair receipt.

`docs/M51_DETECTOR_REPAIR.md` fixed the candidate search, its constants, the instrument check, the falsifier, the
no-regression rule and the branches before the run. These tests bind the receipt to that: the target baseline still
reproduces the committed artifact, the detected box is the declared label-free sampling of the same statistics, the
branch is the predeclared arithmetic on its own numbers (the displacement gate it misses and the AUC gate it clears),
the per-board table covers all 28 boards including both former outliers, the no-regression rule that fails is reported
as failed, and the declaration carries the leakage disclosure and the prediction it judged.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from box_search import CONSTANTS  # noqa: E402
from m51_board_box_read import AUC_GATE, DETECTION_GATE_CELLS, branch_for  # noqa: E402

SCOPE = json.loads((ROOT/"docs/evidence/m51-board-box-read/scope.json").read_text())
DETECT = json.loads((ROOT/"runs/m51-detector-check-r10/detection.json").read_text())
M50 = json.loads((ROOT/"docs/evidence/m50-board-box-read/scope.json").read_text())
M49 = json.loads((ROOT/"docs/evidence/m49-registration-read/scope.json").read_text())


class DeclarationOrderTest(unittest.TestCase):
    def test_the_declaration_precedes_the_receipt(self):
        self.assertEqual(SCOPE["predeclaration"], "docs/M51_DETECTOR_REPAIR.md")
        self.assertEqual(SCOPE["status"], "passed")
        declaration = (ROOT/"docs/M51_DETECTOR_REPAIR.md").read_text()
        self.assertIn("predeclared before any scoring run", declaration)

    def test_the_declaration_discloses_the_leakage_and_keeps_the_prediction(self):
        declaration = (ROOT/"docs/M51_DETECTOR_REPAIR.md").read_text()
        self.assertIn("Leakage disclosure", declaration)
        self.assertIn("motivated by the two named outlier captures", declaration)
        self.assertIn("full 28-board population", declaration)
        self.assertIn("Prediction, standing as written whatever the outcome", declaration)
        # the prediction the run was judged against is still in the text, unedited
        self.assertIn("mean displacement in `[0.30, 0.50]` cells", declaration)
        self.assertIn("`repair-under-specified`**, with `repair-general` requiring", declaration)


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

    def test_the_population_is_the_same_population(self):
        self.assertEqual(SCOPE["population"]["cells"], 2986)
        self.assertEqual(SCOPE["population"]["boards"], 28)
        self.assertTrue(SCOPE["population"]["target_features_match"])
        self.assertEqual(SCOPE["measured"]["detected_box"]["cells"], 2986)

    def test_the_detection_is_label_free_and_declared(self):
        detection = SCOPE["geometry"]["detection"]
        self.assertTrue(detection["label_free"])
        self.assertEqual(detection["recipe"], "single-objective search over wall and grid candidates")
        # the receipt records the detector module's own mapping, so a mutation of either side breaks this (V-188)
        self.assertEqual(detection["constants"], CONSTANTS)
        self.assertEqual(detection["constants"]["edge"], 12.0)
        self.assertEqual(detection["constants"]["board_span"], 0.3)
        self.assertEqual(detection["constants"]["stack_weight"], 0.5)
        self.assertEqual(len(detection["boards"]), 28)


class MeasuredTest(unittest.TestCase):
    def test_the_detected_box_readings_are_the_documented_ones(self):
        detected = SCOPE["measured"]["detected_box"]
        self.assertAlmostEqual(detected["f1_at_zero"], 0.904376, places=6)
        self.assertAlmostEqual(detected["auc"], 0.933275, places=6)
        self.assertEqual(detected["counts_at_zero"], {"true_positive": 279, "false_positive": 29,
                                                      "false_negative": 30})
        self.assertAlmostEqual(detected["best"]["f1"], 0.908497, places=6)

    def test_the_read_did_not_beat_the_winner_head_on_auc(self):
        detected = SCOPE["measured"]["detected_box"]
        self.assertGreater(detected["f1_at_zero"], SCOPE["anchors"]["control_f1_at_zero"])
        self.assertLess(detected["auc"], SCOPE["anchors"]["control_auc"])

    def test_the_early_row_profile_is_the_pitch_signature(self):
        rows = SCOPE["diagnosis"]["detected_box"]["false_positives_by_row"]
        bands = SCOPE["diagnosis"]["detected_box"]["bands"]
        self.assertEqual(sum(int(value) for value in rows.values()), 29)
        self.assertEqual(bands["rows_0_4"], 12)
        self.assertEqual(bands["rows_5_9"], 6)
        self.assertEqual(bands["rows_10_14"], 0)
        self.assertEqual(bands["rows_15_19"], 11)

    def test_the_margins_separate_the_classes(self):
        errors = SCOPE["diagnosis"]["detected_box"]["errors_at_zero"]
        self.assertGreater(errors["true_positive"]["margin_median"], 0.0)
        self.assertLess(errors["true_negative"]["margin_median"], 0.0)
        self.assertLess(errors["false_negative"]["margin_median"], -5.0)


class DetectorTest(unittest.TestCase):
    def test_the_receipt_and_the_instrument_check_agree(self):
        self.assertAlmostEqual(SCOPE["detector"]["mean_magnitude"], 0.383358, places=6)
        self.assertAlmostEqual(DETECT["summary"]["mean_magnitude"], 0.383358, places=6)
        self.assertAlmostEqual(DETECT["summary"]["median_of_medians"], 0.066438, places=6)
        self.assertAlmostEqual(DETECT["summary"]["worst_board_mean"], 5.209784, places=6)
        self.assertEqual(DETECT["summary"]["boards_over_gate"], 4)

    def test_both_former_outliers_are_measured_on_all_28_boards(self):
        boards = {row["sample_id"]: row["displacement"]["mean_magnitude"]
                  for row in DETECT["boards"]}
        self.assertEqual(len(boards), 28)
        self.assertAlmostEqual(boards["training-20261035-v2"], 1.9285, places=3)
        self.assertAlmostEqual(boards["training-20261030-v1"], 5.2098, places=3)
        table = {row["sample_id"]: row for row in SCOPE["no_regression"]["boards"]}
        self.assertEqual(len(table), 28)
        self.assertAlmostEqual(table["training-20261030-v1"]["m51_mean_cells"], 5.209784, places=6)
        self.assertAlmostEqual(table["training-20261030-v1"]["m50_mean_cells"], 7.389254, places=6)

    def test_the_pitch_is_tighter_and_the_tail_is_shorter_than_m50(self):
        self.assertLess(SCOPE["detector"]["mean_width_error_cells"], M50["detector"]["mean_width_error_cells"])
        self.assertLess(SCOPE["detector"]["mean_height_error_cells"], M50["detector"]["mean_height_error_cells"])
        self.assertLess(SCOPE["detector"]["mean_magnitude"], M50["detector"]["mean_magnitude"])
        self.assertLess(SCOPE["detector"]["worst_board_mean"], M50["detector"]["worst_board_mean"])
        self.assertLess(M49["geometry_shift"]["registered_vs_target"]["mean_magnitude"], 8.26)

    def test_the_no_regression_rule_fails_and_says_so(self):
        no_regression = SCOPE["no_regression"]
        self.assertEqual(no_regression["previously_good_boards"], 26)
        self.assertFalse(no_regression["cleared"])
        self.assertEqual(no_regression["breaches"], ["training-20261031-v1", "training-20261031-v2"])


class BranchTest(unittest.TestCase):
    def test_the_receipt_lands_in_the_predeclared_branch_for_its_own_numbers(self):
        detected = SCOPE["measured"]["detected_box"]
        self.assertEqual(SCOPE["branch"]["verdict"],
                         branch_for(SCOPE["detector"]["mean_magnitude"], SCOPE["detector"]["median_of_medians"],
                                    detected["f1_at_zero"], detected["auc"]))
        self.assertEqual(SCOPE["branch"]["verdict"], "repair-under-specified")

    def test_the_gates_in_the_receipt_are_the_executing_constants(self):
        branch = SCOPE["branch"]
        self.assertEqual(branch["detection_gate_cells"], DETECTION_GATE_CELLS)
        self.assertEqual(branch["auc_gate"], AUC_GATE)
        self.assertEqual(branch["detection_gate_cells"], 0.25)
        self.assertEqual(branch["auc_gate"], 0.90)

    def test_the_falsifier_fires_on_the_displacement_leg_only(self):
        detected = SCOPE["measured"]["detected_box"]
        branch = SCOPE["branch"]
        self.assertGreater(SCOPE["detector"]["mean_magnitude"], branch["detection_gate_cells"])
        self.assertGreaterEqual(detected["auc"], branch["auc_gate"])
        self.assertLess(SCOPE["detector"]["median_of_medians"], branch["detection_gate_cells"])

    def test_the_displacement_is_the_declared_one_over_the_declared_population(self):
        shift = SCOPE["geometry_shift"]
        self.assertAlmostEqual(shift["detected_vs_target"]["mean_magnitude"], 0.383358, places=6)
        self.assertAlmostEqual(shift["predicted_vs_target"]["mean_magnitude"], 2.479843, places=6)
        self.assertEqual(shift["detected_vs_target"]["mean_magnitude"],
                         SCOPE["detector"]["mean_magnitude"])


if __name__ == "__main__":
    unittest.main()
