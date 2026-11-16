"""Integrity controls for the M45 interior-contrast receipt.

`docs/M45_INTERIOR_CONTRAST.md` fixed the ranked candidates, the bar, the reference and every branch before the run,
and declared the arm as the winner's configuration plus one zero-initialised projection of exactly `2,496`
parameters. These tests bind the receipt to that: the control reproduces the winner bit-exactly, the arm differs by
exactly the declared change, the verdict is the predeclared harmful branch, and the diagnosis is the
precision-fell-with-recall-held reading the document quotes, per row.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPE = json.loads((ROOT/"docs/evidence/m45-interior-contrast-r1/scope.json").read_text())
DIAGNOSIS = json.loads((ROOT/"docs/evidence/m45-interior-contrast-r1/diagnosis.json").read_text())
DECLARATION = (ROOT/"docs/M45_INTERIOR_CONTRAST.md").read_text()


def manifest(run):
    return json.loads((ROOT/"runs"/run/"checkpoint/manifest.json").read_text())


def effective(config):
    return {key: (0 if value is None else value) for key, value in config.items()}


class DeclarationTest(unittest.TestCase):
    def test_the_declaration_precedes_the_receipt(self):
        self.assertEqual(SCOPE["predeclaration"], "docs/M45_INTERIOR_CONTRAST.md")
        self.assertIn("predeclared before any scoring run", DECLARATION)
        for element in ("interior contrast", "contrast-*gated*", "residual", "margin-conditioned",
                        "retired, must not be re-run"):
            self.assertIn(element, DECLARATION, element)

    def test_the_retired_ring_variants_are_named_as_retired(self):
        self.assertIn("every offset/width/sample-count variant of the M44 ring", DECLARATION)

    def test_the_declaration_states_the_row_set_reconciliation(self):
        self.assertIn("M43's `173`", DECLARATION)
        self.assertIn("**the control's `140` false positives in rows 10-19", DECLARATION)


class IsolationTest(unittest.TestCase):
    def test_the_control_is_the_winner_bit_for_bit(self):
        control, winner = manifest("m45-control-winner-config"), manifest("m30-level-calibrate-r5")
        self.assertEqual(control["weights_sha256"], winner["weights_sha256"])
        self.assertEqual(control["weights_sha256"][:12], "eddec1fd4af2")
        self.assertEqual(control["parameters"], winner["parameters"])

    def test_the_arm_differs_by_exactly_its_declared_projection(self):
        arm, control = manifest("m45-interior-contrast"), manifest("m45-control-winner-config")
        self.assertEqual(arm["parameters"]-control["parameters"], 12*192+192)
        self.assertEqual(arm["parameters"]-control["parameters"], 2496)
        differing = sorted(key for key in set(arm["config"]) | set(control["config"])
                           if effective(arm["config"]).get(key) != effective(control["config"]).get(key))
        self.assertEqual(differing, ["occupancy_pixel_contrast"])
        self.assertEqual((arm["config"]["occupancy_pixel_contrast"],
                          effective(control["config"]).get("occupancy_pixel_contrast")), (1, 0))

    def test_the_arm_is_a_distinct_and_non_inert_checkpoint(self):
        arm = manifest("m45-interior-contrast")
        self.assertNotEqual(arm["weights_sha256"], manifest("m45-control-winner-config")["weights_sha256"])
        self.assertEqual(SCOPE["arm"]["weights_sha256"], arm["weights_sha256"])
        self.assertEqual(SCOPE["arm"]["weights_sha256"][:12], "05ef2e09eac5")


class BarTest(unittest.TestCase):
    def test_the_bar_and_the_reference_are_the_predeclared_ones(self):
        self.assertEqual(SCOPE["bar"], {"best": 0.689775, "f1_at_zero": 0.372881, "met": False})
        self.assertAlmostEqual(SCOPE["anchors"]["anchor_auc"], 0.954529, places=6)
        self.assertFalse(SCOPE["cleared"])
        self.assertEqual(SCOPE["gates"], {"attribution": True, "per_cell_f1": False, "ranking": False})

    def test_the_verdict_is_the_predeclared_harmful_branch(self):
        # harmful: dF1@0 <= -0.010, or AUC below 0.949529 - both held
        self.assertLessEqual(SCOPE["gain_over_anchor_f1"], -0.010)
        self.assertLess(SCOPE["own"]["auc"], 0.949529)
        self.assertLess(SCOPE["own"]["auc"], SCOPE["anchors"]["anchor_auc"])
        self.assertAlmostEqual(SCOPE["gain_over_anchor_f1"], -0.189332, places=6)
        self.assertAlmostEqual(SCOPE["auc_delta_vs_anchor"], -0.038217, places=6)
        self.assertAlmostEqual(SCOPE["gain_over_anchor_best"], -0.130104, places=6)

    def test_the_cleared_branch_would_have_needed_a_precision_gain(self):
        # the reading is not "cleared": per-cell precision at the fixed crossing fell
        errors = DIAGNOSIS["errors_at_zero"]
        self.assertLess(errors["true_positive"]["count"]/(errors["true_positive"]["count"] +
                                                          errors["false_positive"]["count"]), 0.230430)


class MeasuredTest(unittest.TestCase):
    def test_the_readings_are_the_ones_the_document_quotes(self):
        self.assertAlmostEqual(SCOPE["own"]["f1_at_zero"], 0.183550, places=6)
        self.assertAlmostEqual(SCOPE["own"]["auc"], 0.916312, places=6)
        self.assertAlmostEqual(SCOPE["own_best"], 0.559671, places=6)
        self.assertAlmostEqual(SCOPE["own"]["best"]["threshold"], 2.85, places=6)
        self.assertAlmostEqual(SCOPE["anchored"]["f1_at_zero"], 0.249480, places=6)
        self.assertAlmostEqual(SCOPE["anchored_best"], 0.488812, places=6)

    def test_the_split_sums_to_the_measured_span(self):
        self.assertTrue(SCOPE["fractions_sum_to_one"])
        self.assertAlmostEqual(SCOPE["span_closed_fraction"]+SCOPE["placement_fraction"]+
                               SCOPE["representation_fraction"], 1.0, places=5)
        self.assertAlmostEqual(SCOPE["span_points"]+SCOPE["placement_points"]+
                               SCOPE["representation_points"], SCOPE["anchors"]["span"], places=5)
        self.assertEqual(SCOPE["short_stage"], "placement")

    def test_per_cell_precision_fell_at_the_fixed_crossing(self):
        errors = DIAGNOSIS["errors_at_zero"]
        self.assertEqual(errors["true_positive"]["count"], 212)
        self.assertEqual(errors["false_negative"]["count"], 2)
        self.assertEqual(errors["false_positive"]["count"], 1884)
        self.assertAlmostEqual(DIAGNOSIS["precision_at_zero"], 0.101145, places=6)
        self.assertAlmostEqual(DIAGNOSIS["precision_at_zero"]-209/(209+698), -0.129285, places=6)
        self.assertAlmostEqual(DIAGNOSIS["recall_at_zero"], 0.990654, places=6)

    def test_the_tail_below_zero_is_the_lost_mass(self):
        # the arm's lift is not an offset: the empty median barely moved while 1,186 negatives crossed up
        self.assertEqual(DIAGNOSIS["cells"], 2986)
        self.assertEqual(DIAGNOSIS["occupied_cells"], 214)
        true_negatives_arm = DIAGNOSIS["cells"]-DIAGNOSIS["occupied_cells"]-1884
        true_negatives_control = 2986-214-698
        self.assertEqual(true_negatives_arm, 888)
        self.assertEqual(true_negatives_control-true_negatives_arm, 1186)

    def test_the_late_row_mass_and_its_per_row_counts(self):
        comparison = SCOPE["comparison_to_m35"]
        self.assertEqual(comparison["anchor_false_positives"], 698)
        self.assertEqual(comparison["late_rows_10_19_before"], 140)
        self.assertEqual(comparison["late_rows_10_19_now"], 883)
        rows = comparison["per_row"]
        self.assertEqual(rows["15"], {"before": 111, "now": 169})
        self.assertEqual(rows["3"], {"before": 107, "now": 109})
        for row in ("10", "11", "12", "13", "18", "19"):
            self.assertEqual(rows[row]["before"], 0, row)
            self.assertGreater(rows[row]["now"], 0, row)
        self.assertEqual(sum(row["now"] for row in rows.values()), 1884)
        self.assertEqual(sum(row["before"] for row in rows.values()), 698)


if __name__ == "__main__":
    unittest.main()
