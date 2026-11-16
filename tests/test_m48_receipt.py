"""Integrity controls for the M48 geometry read receipt.

`docs/M48_GEOMETRY_READ.md` fixed the read-out, the reconstruction it must verify, the two samplings and the branch
thresholds before the run. These tests bind the receipt to that: the reconstruction reproduces the committed
artifact, the branch is the predeclared one for the measured F1, the contingency table and the late-count profile are
the ones the document quotes (with the profile reconciled rather than read as an improvement), and the campaign's
statistics-versus-geometry claim is rebuilt from the receipts themselves.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from m48_geometry_read import branch_for  # noqa: E402

SCOPE = json.loads((ROOT/"docs/evidence/m48-geometry-read-r2/scope.json").read_text())
SHIFT = json.loads((ROOT/"docs/evidence/m48-lattice-displacement/scope.json").read_text())
V47 = json.loads((ROOT/"docs/evidence/m47-pixel-sole-r1/scope.json").read_text())


class DeclarationOrderTest(unittest.TestCase):
    def test_the_declaration_precedes_the_receipt(self):
        self.assertEqual(SCOPE["predeclaration"], "docs/M48_GEOMETRY_READ.md")
        declaration = (ROOT/"docs/M48_GEOMETRY_READ.md").read_text()
        self.assertIn("## Protocol, isolation and falsifier", declaration)
        self.assertIn("no parameter is fitted on this arm's inputs", declaration.lower())
        self.assertIn("Measured", declaration)


class ProvenanceTest(unittest.TestCase):
    def test_the_readout_is_the_committed_one(self):
        readout = SCOPE["readout"]
        self.assertEqual(readout["provenance"], "m3-occpix-ceiling-r1-r2")
        self.assertTrue(readout["parameters_imported_unchanged"])
        self.assertTrue(readout["reconstruction"]["committed_logits_reproduced"])
        self.assertLess(readout["reconstruction"]["max_abs_logit_difference"], 1e-6)
        self.assertIn("not coefficients", readout["refit_note"])

    def test_the_geometry_source_is_the_committed_winner(self):
        manifest = json.loads((ROOT/"runs/m47-control-winner-config-r6/checkpoint/manifest.json").read_text())
        self.assertEqual(SCOPE["geometry"]["weights_sha256"], manifest["weights_sha256"])
        self.assertTrue(SCOPE["geometry"]["weights_sha256"].startswith("eddec1fd4af2"))

    def test_the_target_lattice_reproduces_the_committed_readout_exactly(self):
        target = SCOPE["measured"]["target_lattice"]
        self.assertAlmostEqual(target["f1_at_zero"], 0.974763, places=6)
        self.assertAlmostEqual(target["auc"], 0.999195, places=6)
        self.assertAlmostEqual(target["best"]["f1"], 0.987179, places=6)


class MeasuredTest(unittest.TestCase):
    def test_the_predicted_lattice_readings_are_the_documented_ones(self):
        predicted = SCOPE["measured"]["predicted_lattice"]
        self.assertAlmostEqual(predicted["f1_at_zero"], 0.037152, places=6)
        self.assertAlmostEqual(predicted["auc"], 0.442341, places=6)
        self.assertAlmostEqual(predicted["best"]["f1"], 0.037152, places=6)

    def test_the_contingency_table_at_the_fixed_crossing(self):
        predicted = SCOPE["measured"]["predicted_lattice"]
        self.assertEqual(predicted["counts_at_zero"], {"true_positive": 6, "false_positive": 8,
                                                       "false_negative": 303})
        self.assertEqual(predicted["cells"], 2986)
        self.assertEqual(predicted["occupied"], 309)
        self.assertAlmostEqual(predicted["precision_at_zero"], 0.428571, places=6)
        self.assertAlmostEqual(predicted["recall_at_zero"], 0.019417, places=6)

    def test_the_late_profile_is_reconciled_and_not_an_improvement(self):
        diagnosis = SCOPE["diagnosis"]["predicted_lattice"]
        rows = diagnosis["false_positives_by_row"]
        self.assertEqual(sum(int(value) for value in rows.values()), 8)
        self.assertEqual(sum(int(rows[str(row)]) for row in range(10, 20)), 0)
        # the arm under-fires: nearly every occupied cell is called empty, so the small late count is not precision
        self.assertEqual(SCOPE["measured"]["predicted_lattice"]["counts_at_zero"]["false_negative"], 303)

    def test_the_degradation_is_ordering_not_threshold_placement(self):
        predicted = SCOPE["measured"]["predicted_lattice"]
        self.assertLess(predicted["auc"], 0.5)
        errors = SCOPE["diagnosis"]["predicted_lattice"]["errors_at_zero"]
        self.assertLess(errors["false_negative"]["margin_median"], -20.0)
        self.assertGreater(errors["true_positive"]["margin_median"], 0.0)


class GeometryTest(unittest.TestCase):
    def test_the_displacement_is_measured_and_large(self):
        displacement = SHIFT["displacement_cells"]
        self.assertAlmostEqual(displacement["mean_magnitude"], 2.479843, places=6)
        self.assertAlmostEqual(displacement["median_magnitude"], 2.048453, places=6)
        self.assertAlmostEqual(displacement["mean_rows"], -1.116965, places=6)
        self.assertEqual(SHIFT["population"]["cells"], 5600)

    def test_the_feature_distribution_drifts_off_the_fitted_scale(self):
        drift = SHIFT["feature_drift"]
        self.assertAlmostEqual(drift["mean_shift_in_target_std"], 0.764765, places=6)
        self.assertEqual(drift["features_over_three_std"], 3)


class BranchTest(unittest.TestCase):
    def test_the_receipt_lands_in_the_predeclared_branch_for_its_own_number(self):
        predicted = SCOPE["measured"]["predicted_lattice"]["f1_at_zero"]
        self.assertEqual(SCOPE["branch"]["verdict"], branch_for(predicted))
        self.assertEqual(SCOPE["branch"]["verdict"], "geometry-binds")
        self.assertEqual(SCOPE["branch"]["cell_gate_f1_at_zero"], 0.70)
        self.assertAlmostEqual(SCOPE["branch"]["control_floor_f1_at_zero"], 0.382881, places=6)

    def test_the_branch_boundaries_are_the_declared_arithmetic(self):
        self.assertEqual(branch_for(0.70), "readout-binds")
        self.assertEqual(branch_for(0.382881), "geometry-costs")
        self.assertEqual(branch_for(0.38288), "geometry-binds")


class CampaignClaimTest(unittest.TestCase):
    def test_geometry_not_the_statistics_explains_the_ceiling(self):
        target = SCOPE["measured"]["target_lattice"]
        predicted = SCOPE["measured"]["predicted_lattice"]
        control = SCOPE["anchors"]
        # same estimator, same statistics, same cells: only the sampling differs
        self.assertGreater(target["f1_at_zero"]-predicted["f1_at_zero"], 0.9)
        self.assertGreater(target["auc"]-predicted["auc"], 0.5)
        # and the model's own learned read sits between them, at the displaced geometry
        self.assertLess(control["control_auc"], target["auc"])
        self.assertGreater(control["control_auc"], predicted["auc"])
        self.assertGreater(V47["own"]["auc"], predicted["auc"])


if __name__ == "__main__":
    unittest.main()
