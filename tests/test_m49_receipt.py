"""Integrity controls for the M49 registration read receipt.

`docs/M49_REGISTRATION_READ.md` fixed the read-out, the registered lattice, the three comparable reads and the branch
thresholds before the run. These tests bind the receipt to that: the target baseline still reproduces the committed
artifact, the registered read is the declared label-free sampling of the same statistics, the branch is the
predeclared one for its own measured numbers, the failure is a chance-level ordering rather than a misplaced
threshold, and the measured cause (the valid extent is larger than the board's own box) is the one the diagnosis
records.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from m49_registration_read import branch_for  # noqa: E402

SCOPE = json.loads((ROOT/"docs/evidence/m49-registration-read/scope.json").read_text())
EXTENT = json.loads((ROOT/"docs/evidence/m49-frame-extent-r2/scope.json").read_text())
M48 = json.loads((ROOT/"docs/evidence/m48-geometry-read-r2/scope.json").read_text())


class DeclarationOrderTest(unittest.TestCase):
    def test_the_declaration_precedes_the_receipt(self):
        self.assertEqual(SCOPE["predeclaration"], "docs/M49_REGISTRATION_READ.md")
        self.assertIn("predeclared before any scoring run", (ROOT/"docs/M49_REGISTRATION_READ.md").read_text())


class ProvenanceTest(unittest.TestCase):
    def test_the_readout_is_the_committed_one_and_reproduces_it(self):
        readout = SCOPE["readout"]
        self.assertEqual(readout["provenance"], "m3-occpix-ceiling-r1-r2")
        self.assertTrue(readout["parameters_imported_unchanged"])
        self.assertTrue(readout["reconstruction"]["committed_logits_reproduced"])
        self.assertLess(readout["reconstruction"]["max_abs_logit_difference"], 1e-6)

    def test_the_baseline_still_reproduces_the_ceiling_and_m48s_readings(self):
        target = SCOPE["measured"]["target_lattice"]
        predicted = SCOPE["measured"]["predicted_lattice"]
        self.assertAlmostEqual(target["f1_at_zero"], 0.974763, places=6)
        self.assertAlmostEqual(target["auc"], 0.999195, places=6)
        self.assertAlmostEqual(predicted["f1_at_zero"], M48["measured"]["predicted_lattice"]["f1_at_zero"], places=6)
        self.assertAlmostEqual(predicted["auc"], M48["measured"]["predicted_lattice"]["auc"], places=6)

    def test_the_registration_is_the_declared_label_free_recipe(self):
        registration = SCOPE["geometry"]["registration"]
        self.assertEqual(registration["recipe"], "pixel_support_lattice(valid, 20, 10)")
        self.assertTrue(registration["label_free"])
        self.assertIn("tetris_vision", registration["source_file"])


class MeasuredTest(unittest.TestCase):
    def test_the_registered_readings_are_the_documented_ones(self):
        registered = SCOPE["measured"]["registered_lattice"]
        self.assertEqual(registered["f1_at_zero"], 0.0)
        self.assertAlmostEqual(registered["auc"], 0.505163, places=6)
        self.assertEqual(registered["counts_at_zero"], {"true_positive": 0, "false_positive": 0,
                                                        "false_negative": 309})
        self.assertEqual(registered["best"]["f1"], 0.0)

    def test_the_failure_is_a_chance_ordering_not_a_misplaced_threshold(self):
        registered = SCOPE["measured"]["registered_lattice"]
        self.assertGreater(registered["auc"], 0.45)
        self.assertLess(registered["auc"], 0.55)
        errors = SCOPE["diagnosis"]["registered_lattice"]["errors_at_zero"]
        # every cell is called empty, so no threshold in the swept range can produce a positive
        self.assertEqual(errors["true_positive"]["count"], 0)
        self.assertLess(errors["false_negative"]["margin_median"], 0.0)
        self.assertEqual(sum(SCOPE["diagnosis"]["registered_lattice"]["false_positives_by_row"].values()), 0)

    def test_the_registration_is_measurably_further_out_than_the_predicted_lattice(self):
        shift = SCOPE["geometry_shift"]
        self.assertAlmostEqual(shift["registered_vs_target"]["mean_magnitude"], 8.256949, places=6)
        self.assertAlmostEqual(shift["predicted_vs_target"]["mean_magnitude"], 2.479843, places=6)
        self.assertGreater(shift["registered_vs_target"]["mean_magnitude"],
                           3*shift["predicted_vs_target"]["mean_magnitude"])


class CauseTest(unittest.TestCase):
    def test_the_valid_extent_is_larger_than_the_board_box(self):
        extents = EXTENT["extents"]
        self.assertAlmostEqual(extents["valid_to_board_width_ratio"]["mean"], 3.32553, places=6)
        self.assertAlmostEqual(extents["valid_to_board_height_ratio"]["mean"], 1.736611, places=6)
        self.assertLess(extents["board_area_share_of_valid"]["mean"], 0.25)
        self.assertEqual(extents["board_inside_valid_boards"], EXTENT["population"]["boards"])

    def test_the_registered_cell_is_wrongly_sized_by_the_same_ratio(self):
        extents = EXTENT["extents"]
        # the registered cell inherits the extent ratio exactly: this is a scale error, not a rounding one
        self.assertAlmostEqual(extents["registered_cell_width_ratio"]["mean"],
                               extents["valid_to_board_width_ratio"]["mean"], places=6)
        self.assertAlmostEqual(extents["registered_cell_height_ratio"]["mean"],
                               extents["valid_to_board_height_ratio"]["mean"], places=6)


class BranchTest(unittest.TestCase):
    def test_the_receipt_lands_in_the_predeclared_branch_for_its_own_number(self):
        registered = SCOPE["measured"]["registered_lattice"]
        self.assertEqual(SCOPE["branch"]["verdict"],
                         branch_for(registered["f1_at_zero"], registered["auc"]))
        self.assertEqual(SCOPE["branch"]["verdict"], "registration-fails")
        self.assertEqual(SCOPE["branch"]["f1_gate_at_zero"], 0.70)
        self.assertEqual(SCOPE["branch"]["auc_gate"], 0.90)


class CampaignClaimTest(unittest.TestCase):
    def test_the_ceiling_is_not_reachable_by_this_registration(self):
        registered = SCOPE["measured"]["registered_lattice"]
        target = SCOPE["measured"]["target_lattice"]
        control = SCOPE["anchors"]
        # the same estimator at the same cells: only the sampling differs, and this sampling destroys it
        self.assertGreater(target["auc"]-registered["auc"], 0.4)
        self.assertGreater(target["f1_at_zero"]-registered["f1_at_zero"], 0.9)
        # worse than the winner's own learned read, which is what "registration first" promised to beat
        self.assertLess(registered["auc"], control["control_auc"])
        self.assertLess(registered["f1_at_zero"], control["control_f1_at_zero"])


if __name__ == "__main__":
    unittest.main()
