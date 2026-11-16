"""Integrity controls for the M47 pixel-sole receipt.

`docs/M47_PIXEL_SOLE.md` fixed the routing restriction, the zero parameter delta, the protocol and the falsifier
before the run. These tests bind the receipt to that: the control reproduces the winner bit-exactly, the arm carries
exactly the same parameters, the verdict is the predeclared harmful branch, the complete contingency table and the
per-row late profile are the ones the document quotes, and the mechanism numbers are the measured ones.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPE = json.loads((ROOT/"docs/evidence/m47-pixel-sole-r1/scope.json").read_text())
DIAGNOSIS = json.loads((ROOT/"docs/evidence/m47-pixel-sole-r1/diagnosis.json").read_text())
DECLARATION = (ROOT/"docs/M47_PIXEL_SOLE.md").read_text()
V44 = json.loads((ROOT/"docs/evidence/m44-pixel-edge-r1/scope.json").read_text())
V45 = json.loads((ROOT/"docs/evidence/m45-interior-contrast-r1/scope.json").read_text())
V46 = json.loads((ROOT/"docs/evidence/m46-edge-gated-contrast-r1/scope.json").read_text())


def manifest(run):
    return json.loads((ROOT/"runs"/run/"checkpoint/manifest.json").read_text())


def effective(config):
    return {key: (0 if value is None else value) for key, value in config.items()}


class DeclarationTest(unittest.TestCase):
    def test_the_declaration_precedes_the_receipt(self):
        self.assertEqual(SCOPE["predeclaration"], "docs/M47_PIXEL_SOLE.md")
        self.assertIn("predeclared before any scoring run", DECLARATION)
        self.assertIn("No decoder-latent path reaches the decision", DECLARATION)

    def test_the_falsifier_is_stated_with_its_derivation(self):
        for element in ("HELPFUL", "NEUTRAL", "HARMFUL", "0.382881", "0.954529", "0.974763"):
            self.assertIn(element, DECLARATION, element)
        self.assertIn("derived from already-committed receipts", DECLARATION)

    def test_the_routing_matrix_is_stated_before_the_run(self):
        self.assertIn("the latent route was the problem", DECLARATION)
        self.assertIn("not the binding constraint", DECLARATION)

    def test_the_zero_parameter_delta_is_declared(self):
        self.assertIn("the expected parameter delta is exactly `0`", DECLARATION)

    def test_the_successors_are_ranked_with_falsifiers_and_the_cheapest_named(self):
        self.assertIn("committed read-out on the predicted-lattice statistics", DECLARATION)
        self.assertIn("frozen-trunk, head-only training", DECLARATION)
        self.assertIn("registration-first arm", DECLARATION)
        self.assertIn("seed-spread replicate", DECLARATION)
        self.assertIn("The cheapest decisive test is rank 1", DECLARATION)


class IsolationTest(unittest.TestCase):
    def test_the_control_is_the_winner_bit_for_bit(self):
        control, winner = manifest("m47-control-winner-config-r6"), manifest("m30-level-calibrate-r5")
        self.assertEqual(control["weights_sha256"], winner["weights_sha256"])
        self.assertEqual(control["weights_sha256"][:12], "eddec1fd4af2")
        self.assertEqual(control["parameters"], winner["parameters"])

    def test_the_arm_differs_by_exactly_the_declared_field_and_no_parameter(self):
        arm, control = manifest("m47-pixel-sole"), manifest("m47-control-winner-config-r6")
        self.assertEqual(arm["parameters"], control["parameters"])
        self.assertEqual(arm["parameters"]-control["parameters"], 0)
        differing = sorted(key for key in set(arm["config"]) | set(control["config"])
                           if effective(arm["config"]).get(key) != effective(control["config"]).get(key))
        self.assertEqual(differing, ["occupancy_pixel_sole"])
        self.assertEqual((arm["config"]["occupancy_pixel_sole"],
                          effective(control["config"]).get("occupancy_pixel_sole")), (1, 0))

    def test_the_arm_is_a_distinct_and_non_inert_checkpoint(self):
        arm = manifest("m47-pixel-sole")
        self.assertNotEqual(arm["weights_sha256"], manifest("m47-control-winner-config-r6")["weights_sha256"])
        self.assertEqual(SCOPE["arm"]["weights_sha256"], arm["weights_sha256"])
        self.assertEqual(SCOPE["arm"]["weights_sha256"][:12], "b6959ba0e7ae")


class BarTest(unittest.TestCase):
    def test_the_bar_and_the_gates_are_the_predeclared_ones(self):
        self.assertEqual(SCOPE["bar"], {"best": 0.689775, "f1_at_zero": 0.372881, "met": False})
        self.assertAlmostEqual(SCOPE["anchors"]["anchor_auc"], 0.954529, places=6)
        self.assertFalse(SCOPE["cleared"])
        self.assertEqual(SCOPE["gates"], {"attribution": True, "per_cell_f1": False, "ranking": False})

    def test_the_verdict_is_the_predeclared_harmful_branch(self):
        self.assertLessEqual(SCOPE["gain_over_anchor_f1"], -0.010)
        self.assertLess(SCOPE["own"]["auc"], 0.949529)
        self.assertAlmostEqual(SCOPE["gain_over_anchor_f1"], -0.239131, places=6)
        self.assertAlmostEqual(SCOPE["auc_delta_vs_anchor"], -0.094155, places=6)
        self.assertAlmostEqual(SCOPE["gain_over_anchor_best"], -0.202818, places=6)


class MeasuredTest(unittest.TestCase):
    def test_the_readings_are_the_ones_the_document_quotes(self):
        self.assertAlmostEqual(SCOPE["own"]["f1_at_zero"], 0.13375, places=6)
        self.assertAlmostEqual(SCOPE["own"]["auc"], 0.860374, places=6)
        self.assertAlmostEqual(SCOPE["own_best"], 0.486957, places=6)
        self.assertAlmostEqual(SCOPE["own"]["best"]["threshold"], 0.60, places=6)
        self.assertAlmostEqual(SCOPE["anchored"]["f1_at_zero"], 0.187557, places=6)
        self.assertAlmostEqual(SCOPE["anchored_best"], 0.485616, places=6)

    def test_the_complete_contingency_table_at_the_fixed_crossing(self):
        errors = DIAGNOSIS["errors_at_zero"]
        self.assertEqual(errors["true_positive"]["count"], 214)
        self.assertEqual(errors["false_negative"]["count"], 0)
        self.assertEqual(errors["false_positive"]["count"], 2772)
        self.assertEqual(errors["true_negative"]["count"], 0)
        self.assertEqual(DIAGNOSIS["cells"]-DIAGNOSIS["occupied_cells"], 2772)
        self.assertAlmostEqual(DIAGNOSIS["precision_at_zero"], 0.071668, places=6)
        self.assertAlmostEqual(DIAGNOSIS["recall_at_zero"], 1.0, places=6)

    def test_every_margin_is_positive_which_is_the_level_finding(self):
        errors = DIAGNOSIS["errors_at_zero"]
        self.assertGreater(errors["false_positive"]["margin_low"], 0.0)
        self.assertGreater(errors["true_positive"]["margin_low"], 0.0)
        self.assertAlmostEqual(errors["false_positive"]["margin_median"], 0.238912, places=5)
        self.assertAlmostEqual(errors["true_positive"]["margin_median"], 0.978908, places=5)

    def test_the_auc_loss_is_ordering_and_not_a_threshold_shift(self):
        # an offset cannot move AUC, so the -0.094155 is the read's own ordering
        self.assertLess(SCOPE["own"]["auc"], SCOPE["anchors"]["anchor_auc"]-0.05)
        self.assertLess(SCOPE["own_best"], SCOPE["anchors"]["anchor_best"]-0.10)
        self.assertEqual(SCOPE["short_stage"], "closure")

    def test_the_split_sums_to_the_measured_span(self):
        self.assertTrue(SCOPE["fractions_sum_to_one"])
        self.assertAlmostEqual(SCOPE["span_closed_fraction"]+SCOPE["placement_fraction"]+
                               SCOPE["representation_fraction"], 1.0, places=5)
        self.assertAlmostEqual(SCOPE["span_points"]+SCOPE["placement_points"]+
                               SCOPE["representation_points"], SCOPE["anchors"]["span"], places=5)

    def test_the_late_row_mass_and_its_per_row_counts(self):
        comparison = SCOPE["comparison_to_m35"]
        self.assertEqual(comparison["anchor_false_positives"], 698)
        self.assertEqual(comparison["late_rows_10_19_before"], 140)
        self.assertEqual(comparison["late_rows_10_19_now"], 1685)
        rows = comparison["per_row"]
        self.assertEqual(rows["15"], {"before": 111, "now": 173})
        self.assertEqual(rows["3"], {"before": 107, "now": 109})
        self.assertEqual(sum(row["now"] for row in rows.values()), 2772)
        self.assertEqual(sum(row["before"] for row in rows.values()), 698)
        self.assertEqual(DIAGNOSIS["false_positive_bands"]["rows_10_14"]["false_positives"], 863)


class FamilyTest(unittest.TestCase):
    def test_no_arm_of_this_record_raises_fixed_crossing_precision(self):
        control = SCOPE["anchors"]
        for member in (V44, V45, V46, SCOPE):
            self.assertLess(member["own"]["f1_at_zero"], control["anchor_f1_at_zero"])
            self.assertLess(member["own"]["auc"], control["anchor_auc"])

    def test_the_routing_restriction_is_worse_than_every_additive_member_on_f1(self):
        self.assertLess(SCOPE["own"]["f1_at_zero"], min(
            member["own"]["f1_at_zero"] for member in (V44, V45, V46)))

    def test_the_ceiling_reference_the_successors_are_measured_against(self):
        self.assertAlmostEqual(SCOPE["anchors"]["pixel_best"], 0.987179, places=6)
        self.assertGreater(SCOPE["anchors"]["pixel_best"], 0.9)


if __name__ == "__main__":
    unittest.main()
