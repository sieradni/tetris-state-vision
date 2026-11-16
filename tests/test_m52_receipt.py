"""Integrity controls for the M52 lattice-refinement receipt.

`docs/M52_LATTICE_FIT.md` fixed the refinement, its constants, the instrument check, the falsifier, the no-regression
rule and the branches before the run. These tests bind the receipt to that: the target baseline still reproduces the
committed artifact, M51's box is re-measured and reproduces M51's own receipt, the refined read is the declared
label-free sampling, the branch is the predeclared arithmetic on its own numbers (the displacement gate it misses and
the AUC gate it clears), the no-regression pair is reported against both baselines, and the prediction the run judged
is still in the text.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from lattice_fit import CONSTANTS  # noqa: E402
from m52_lattice_fit_read import AUC_GATE, DETECTION_GATE_CELLS, branch_for  # noqa: E402

SCOPE = json.loads((ROOT/"docs/evidence/m52-lattice-fit-read/scope.json").read_text())
DETECT = json.loads((ROOT/"runs/m52-lattice-fit-detect-r3/detection.json").read_text())
M51 = json.loads((ROOT/"docs/evidence/m51-board-box-read/scope.json").read_text())
M50 = json.loads((ROOT/"docs/evidence/m50-board-box-read/scope.json").read_text())


class DeclarationOrderTest(unittest.TestCase):
    def test_the_declaration_precedes_the_receipt(self):
        self.assertEqual(SCOPE["predeclaration"], "docs/M52_LATTICE_FIT.md")
        self.assertEqual(SCOPE["status"], "passed")
        self.assertIn("predeclared before any scoring run", (ROOT/"docs/M52_LATTICE_FIT.md").read_text())

    def test_the_prediction_the_run_was_judged_against_is_still_in_the_text(self):
        declaration = (ROOT/"docs/M52_LATTICE_FIT.md").read_text()
        self.assertIn("Prediction, standing as written whatever the outcome", declaration)
        self.assertIn("mean displacement in `[0.30, 0.45]` cells", declaration)
        self.assertIn("F1 at `0.0` in `[0.93, 0.99]`", declaration)
        # the measured section must judge that prediction, not restate it
        self.assertIn("| F1 at `0.0` | `[0.93, 0.99]` | `0.893268` | **missed low** |", declaration)


class ProvenanceTest(unittest.TestCase):
    def test_the_readout_is_the_committed_one_and_reproduces_it(self):
        readout = SCOPE["readout"]
        self.assertEqual(readout["provenance"], "m3-occpix-ceiling-r1-r2")
        self.assertTrue(readout["parameters_imported_unchanged"])
        self.assertTrue(readout["reconstruction"]["committed_logits_reproduced"])
        self.assertLess(readout["reconstruction"]["max_abs_logit_difference"], 1e-6)

    def test_the_population_and_the_baselines_are_the_committed_ones(self):
        self.assertEqual(SCOPE["population"], {"cells": 2986, "boards": 28, "target_features_match": True})
        target = SCOPE["measured"]["target_lattice"]
        predicted = SCOPE["measured"]["predicted_lattice"]
        self.assertAlmostEqual(target["f1_at_zero"], 0.974763, places=6)
        self.assertAlmostEqual(target["auc"], 0.999195, places=6)
        self.assertAlmostEqual(predicted["f1_at_zero"], 0.037152, places=6)
        self.assertAlmostEqual(predicted["auc"], 0.442341, places=6)

    def test_m51s_box_is_re_measured_and_reproduces_m51s_receipt(self):
        box = SCOPE["measured"]["detected_box"]
        recorded = M51["measured"]["detected_box"]
        self.assertAlmostEqual(box["f1_at_zero"], recorded["f1_at_zero"], places=6)
        self.assertAlmostEqual(box["auc"], recorded["auc"], places=6)
        self.assertEqual(box["counts_at_zero"], recorded["counts_at_zero"])

    def test_the_refinement_is_declared_label_free_with_its_constants(self):
        refinement = SCOPE["geometry"]["refinement"]
        self.assertTrue(refinement["label_free"])
        self.assertEqual(refinement["constants"], CONSTANTS)
        self.assertEqual(refinement["constants"]["search_share"], 0.35)
        self.assertEqual(refinement["constants"]["missing_share"], 0.35)
        self.assertEqual(refinement["constants"]["outlier_cells"], 0.25)
        self.assertEqual(len(SCOPE["geometry"]["detection"]["boards"]), 28)


class MeasuredTest(unittest.TestCase):
    def test_the_refined_readings_are_the_documented_ones(self):
        refined = SCOPE["measured"]["refined_lattice"]
        self.assertAlmostEqual(refined["f1_at_zero"], 0.893268, places=6)
        self.assertAlmostEqual(refined["auc"], 0.908824, places=6)
        self.assertEqual(refined["counts_at_zero"], {"true_positive": 272, "false_positive": 28,
                                                     "false_negative": 37})
        self.assertAlmostEqual(refined["best"]["f1"], 0.896211, places=6)

    def test_the_refinement_costs_the_read_it_was_meant_to_buy(self):
        refined = SCOPE["measured"]["refined_lattice"]
        box = SCOPE["measured"]["detected_box"]
        self.assertLess(refined["f1_at_zero"], box["f1_at_zero"])
        self.assertLess(refined["auc"], box["auc"])
        self.assertLess(refined["auc"], SCOPE["anchors"]["control_auc"])
        self.assertGreater(refined["f1_at_zero"], SCOPE["anchors"]["control_f1_at_zero"])

    def test_the_row_profile_and_margins_are_the_documented_ones(self):
        diagnosis = SCOPE["diagnosis"]["refined_lattice"]
        rows = diagnosis["false_positives_by_row"]
        self.assertEqual(sum(int(value) for value in rows.values()), 28)
        self.assertEqual(diagnosis["bands"], {"rows_0_4": 12, "rows_5_9": 6, "rows_10_14": 0, "rows_15_19": 10})
        errors = diagnosis["errors_at_zero"]
        self.assertEqual(errors["true_positive"]["count"], 272)
        self.assertEqual(errors["false_negative"]["count"], 37)
        self.assertGreater(errors["true_positive"]["margin_median"], 0.0)
        self.assertLess(errors["false_negative"]["margin_median"], -5.0)


class DisplacementTest(unittest.TestCase):
    def test_the_receipt_and_the_instrument_check_agree(self):
        self.assertAlmostEqual(SCOPE["detector"]["mean_magnitude"], 0.380890, places=6)
        self.assertAlmostEqual(DETECT["summary"]["mean_magnitude"], 0.380890, places=6)
        self.assertAlmostEqual(DETECT["summary"]["median_of_medians"], 0.074716, places=6)
        self.assertAlmostEqual(DETECT["summary"]["worst_board_mean"], 5.177659, places=6)
        self.assertEqual(DETECT["summary"]["boards_over_gate"], 4)

    def test_all_28_boards_are_reported_with_their_m51_values(self):
        table = {row["sample_id"]: row for row in SCOPE["no_regression"]["against_m51"]["boards"]}
        self.assertEqual(len(table), 28)
        self.assertEqual(len(SCOPE["no_regression"]["against_m50"]["boards"]), 28)
        self.assertAlmostEqual(table["training-20261030-v1"]["refined_mean_cells"], 5.177659, places=6)
        self.assertAlmostEqual(table["training-20261030-v1"]["box_mean_cells"], 5.209784, places=6)
        self.assertAlmostEqual(table["training-20261030-v1"]["baseline_mean_cells"], 5.209784, places=6)

    def test_the_scale_is_tighter_and_the_location_is_a_wash(self):
        detector = SCOPE["detector"]
        self.assertLess(detector["mean_width_error_cells"], M51["detector"]["mean_width_error_cells"])
        self.assertLess(detector["mean_height_error_cells"], M51["detector"]["mean_height_error_cells"])
        self.assertLess(detector["mean_magnitude"], M51["detector"]["mean_magnitude"])
        self.assertGreater(detector["median_of_medians"], M51["detector"]["median_of_medians"])
        self.assertEqual(detector["boards_kept_by_fit"], 22)

    def test_the_frame_extent_recipe_is_still_the_one_beaten(self):
        declaration = (ROOT/"docs/M52_LATTICE_FIT.md").read_text()
        self.assertIn("3.325530", declaration)
        self.assertIn("1.736611", declaration)
        self.assertLess(SCOPE["detector"]["mean_width_error_cells"], 0.01)


class NoRegressionTest(unittest.TestCase):
    def test_the_rule_fails_on_both_baselines_and_names_the_pair(self):
        no_regression = SCOPE["no_regression"]
        self.assertFalse(no_regression["cleared"])
        self.assertEqual(no_regression["against_m50"]["placed_boards"], 26)
        self.assertEqual(no_regression["against_m51"]["placed_boards"], 24)
        self.assertEqual(no_regression["breaches"], ["training-20261031-v1", "training-20261031-v2"])

    def test_the_named_pair_is_judged_against_m50_not_m51(self):
        named = SCOPE["no_regression"]["named"]
        for sample in ("training-20261031-v1", "training-20261031-v2"):
            self.assertAlmostEqual(named[sample]["m50_mean_cells"], 0.058827, places=6)
            self.assertAlmostEqual(named[sample]["m51_mean_cells"], 0.924144, places=6)
            self.assertGreater(named[sample]["m52_refined_mean_cells"], 0.924144)
            self.assertGreater(named[sample]["m52_refined_mean_cells"], named[sample]["m50_mean_cells"]+0.10)


class BranchTest(unittest.TestCase):
    def test_the_receipt_lands_in_the_predeclared_branch_for_its_own_numbers(self):
        refined = SCOPE["measured"]["refined_lattice"]
        self.assertEqual(SCOPE["branch"]["verdict"],
                         branch_for(SCOPE["detector"]["mean_magnitude"], SCOPE["detector"]["median_of_medians"],
                                    refined["f1_at_zero"], refined["auc"]))
        self.assertEqual(SCOPE["branch"]["verdict"], "fit-under-specified")

    def test_the_gates_in_the_receipt_are_the_executing_constants(self):
        branch = SCOPE["branch"]
        self.assertEqual(branch["detection_gate_cells"], DETECTION_GATE_CELLS)
        self.assertEqual(branch["auc_gate"], AUC_GATE)
        self.assertEqual(branch["detection_gate_cells"], 0.25)
        self.assertEqual(branch["auc_gate"], 0.90)

    def test_the_falsifier_fires_on_the_displacement_leg_only(self):
        scope = SCOPE
        self.assertGreater(scope["detector"]["mean_magnitude"], scope["branch"]["detection_gate_cells"])
        self.assertGreaterEqual(scope["measured"]["refined_lattice"]["auc"], scope["branch"]["auc_gate"])
        self.assertLess(scope["detector"]["median_of_medians"], scope["branch"]["detection_gate_cells"])


if __name__ == "__main__":
    unittest.main()
