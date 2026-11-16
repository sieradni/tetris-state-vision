"""Integrity controls for the M46 edge-gated contrast receipt.

`docs/M46_EDGE_GATE.md` fixed the gate rule and its threshold - a contrast passes only where it exceeds the cell's
own in-cell spread, `|k| > s + 1e-6` - the ranked candidates, the bar and every branch before the run, and declared
the arm as the winner's configuration plus one zero-initialised projection of exactly `3,072` parameters. These
tests bind the receipt to that: the control reproduces the winner bit-exactly, the arm differs by exactly the
declared features, the verdict is the predeclared harmful branch, and the diagnosis is the
precision-fell-with-recall-rose reading the document quotes, per row.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPE = json.loads((ROOT/"docs/evidence/m46-edge-gated-contrast-r1/scope.json").read_text())
DIAGNOSIS = json.loads((ROOT/"docs/evidence/m46-edge-gated-contrast-r1/diagnosis.json").read_text())
DECLARATION = (ROOT/"docs/M46_EDGE_GATE.md").read_text()
V45 = json.loads((ROOT/"docs/evidence/m45-interior-contrast-r1/scope.json").read_text())
V44 = json.loads((ROOT/"docs/evidence/m44-pixel-edge-r1/scope.json").read_text())


def manifest(run):
    return json.loads((ROOT/"runs"/run/"checkpoint/manifest.json").read_text())


def effective(config):
    return {key: (0 if value is None else value) for key, value in config.items()}


class DeclarationTest(unittest.TestCase):
    def test_the_gate_rule_and_threshold_are_written_before_the_receipt(self):
        self.assertEqual(SCOPE["predeclaration"], "docs/M46_EDGE_GATE.md")
        self.assertIn("predeclared before any scoring run", DECLARATION)
        self.assertIn("|k_c,p| > s_c + phi", DECLARATION)
        self.assertIn("phi = 1e-6", DECLARATION)
        self.assertIn("in-cell spread", DECLARATION)

    def test_the_threshold_is_derived_from_the_pipeline_not_from_rows(self):
        self.assertIn("the read's own division floor", DECLARATION)
        self.assertIn("occupancy_read_standardise", DECLARATION)
        self.assertIn("No constant is fitted", DECLARATION)

    def test_the_family_question_and_the_ranked_successors_are_stated(self):
        self.assertIn("No member of the family raises fixed-crossing precision", DECLARATION)
        for element in ("pixel-sole occupancy read", "frozen raster projection",
                        "the ceiling read as the decision", "seed-spread replicate"):
            self.assertIn(element, DECLARATION, element)
        self.assertIn("retired, must not be re-run", DECLARATION)


class IsolationTest(unittest.TestCase):
    def test_the_control_is_the_winner_bit_for_bit(self):
        control, winner = manifest("m46-control-winner-config"), manifest("m30-level-calibrate-r5")
        self.assertEqual(control["weights_sha256"], winner["weights_sha256"])
        self.assertEqual(control["weights_sha256"][:12], "eddec1fd4af2")
        self.assertEqual(control["parameters"], winner["parameters"])

    def test_the_arm_differs_by_exactly_its_declared_projection(self):
        arm, control = manifest("m46-edge-gated-contrast"), manifest("m46-control-winner-config")
        self.assertEqual(arm["parameters"]-control["parameters"], 15*192+192)
        self.assertEqual(arm["parameters"]-control["parameters"], 3072)
        differing = sorted(key for key in set(arm["config"]) | set(control["config"])
                           if effective(arm["config"]).get(key) != effective(control["config"]).get(key))
        self.assertEqual(differing, ["occupancy_pixel_edge_gate"])
        self.assertEqual((arm["config"]["occupancy_pixel_edge_gate"],
                          effective(control["config"]).get("occupancy_pixel_edge_gate")), (1, 0))

    def test_the_arm_is_a_distinct_and_non_inert_checkpoint(self):
        arm = manifest("m46-edge-gated-contrast")
        self.assertNotEqual(arm["weights_sha256"], manifest("m46-control-winner-config")["weights_sha256"])
        self.assertEqual(SCOPE["arm"]["weights_sha256"], arm["weights_sha256"])
        self.assertEqual(SCOPE["arm"]["weights_sha256"][:12], "186734bdaed7")


class BarTest(unittest.TestCase):
    def test_the_bar_is_the_predeclared_one_and_is_not_met(self):
        self.assertEqual(SCOPE["bar"], {"best": 0.689775, "f1_at_zero": 0.372881, "met": False})
        self.assertAlmostEqual(SCOPE["anchors"]["anchor_auc"], 0.954529, places=6)
        self.assertFalse(SCOPE["cleared"])
        self.assertEqual(SCOPE["gates"], {"attribution": False, "per_cell_f1": False, "ranking": False})

    def test_the_verdict_is_the_predeclared_harmful_branch(self):
        self.assertLessEqual(SCOPE["gain_over_anchor_f1"], -0.010)
        self.assertLess(SCOPE["own"]["auc"], 0.949529)
        self.assertAlmostEqual(SCOPE["gain_over_anchor_f1"], -0.232148, places=6)
        self.assertAlmostEqual(SCOPE["auc_delta_vs_anchor"], -0.197029, places=6)
        self.assertAlmostEqual(SCOPE["gain_over_anchor_best"], -0.437413, places=6)


class MeasuredTest(unittest.TestCase):
    def test_the_readings_are_the_ones_the_document_quotes(self):
        self.assertAlmostEqual(SCOPE["own"]["f1_at_zero"], 0.140733, places=6)
        self.assertAlmostEqual(SCOPE["own"]["auc"], 0.757500, places=6)
        self.assertAlmostEqual(SCOPE["own_best"], 0.252362, places=6)
        self.assertAlmostEqual(SCOPE["anchored"]["f1_at_zero"], 0.192825, places=6)
        self.assertAlmostEqual(SCOPE["anchored_best"], 0.256466, places=6)

    def test_the_split_sums_to_the_measured_span(self):
        self.assertTrue(SCOPE["fractions_sum_to_one"])
        self.assertAlmostEqual(SCOPE["span_closed_fraction"]+SCOPE["placement_fraction"]+
                               SCOPE["representation_fraction"], 1.0, places=5)
        self.assertAlmostEqual(SCOPE["span_points"]+SCOPE["placement_points"]+
                               SCOPE["representation_points"], SCOPE["anchors"]["span"], places=5)
        self.assertEqual(SCOPE["short_stage"], "closure")

    def test_per_cell_precision_fell_while_recall_rose(self):
        errors = DIAGNOSIS["errors_at_zero"]
        self.assertEqual(errors["true_positive"]["count"], 213)
        self.assertEqual(errors["false_negative"]["count"], 1)
        self.assertEqual(errors["false_positive"]["count"], 2600)
        self.assertEqual(errors["true_negative"]["count"], 172)
        self.assertAlmostEqual(DIAGNOSIS["precision_at_zero"], 0.075720, places=6)
        self.assertAlmostEqual(DIAGNOSIS["precision_at_zero"]-209/(209+698), -0.154710, places=6)
        self.assertAlmostEqual(DIAGNOSIS["recall_at_zero"], 0.995327, places=6)

    def test_the_empty_distribution_moved_onto_the_boundary_not_off_it(self):
        # the mechanism, measured: the empty margin median rises and nearly every empty cell crosses
        self.assertAlmostEqual(DIAGNOSIS["errors_at_zero"]["true_negative"]["margin_median"], -0.160302, places=5)
        self.assertAlmostEqual(DIAGNOSIS["errors_at_zero"]["false_positive"]["margin_median"], 2.421909, places=5)
        self.assertEqual(DIAGNOSIS["cells"]-DIAGNOSIS["occupied_cells"], 2772)

    def test_the_late_row_mass_and_its_per_row_counts(self):
        comparison = SCOPE["comparison_to_m35"]
        self.assertEqual(comparison["anchor_false_positives"], 698)
        self.assertEqual(comparison["late_rows_10_19_before"], 140)
        self.assertEqual(comparison["late_rows_10_19_now"], 1513)
        rows = comparison["per_row"]
        self.assertEqual(rows["15"], {"before": 111, "now": 173})
        self.assertEqual(rows["3"], {"before": 107, "now": 109})
        for row in ("10", "11", "12", "13", "16", "18", "19"):
            self.assertEqual(rows[row]["before"], 0 if row != "16" else 2, row)
            self.assertGreater(rows[row]["now"], 100, row)
        self.assertEqual(sum(row["now"] for row in rows.values()), 2600)
        self.assertEqual(sum(row["before"] for row in rows.values()), 698)


class FamilyTest(unittest.TestCase):
    def test_no_additive_raster_member_raises_fixed_crossing_precision(self):
        # the campaign-level claim the document makes, rebuilt from the three receipts
        control = SCOPE["anchors"]
        for member in (V44, V45, SCOPE):
            self.assertLess(member["own"]["f1_at_zero"], control["anchor_f1_at_zero"])
            self.assertLess(member["own"]["auc"], control["anchor_auc"])
        self.assertLess(V45["own"]["f1_at_zero"], V44["own"]["f1_at_zero"])
        self.assertLess(SCOPE["own"]["f1_at_zero"], V45["own"]["f1_at_zero"])

    def test_the_price_range_the_document_quotes(self):
        deltas = [member["auc_delta_vs_anchor"] for member in (V44, V45, SCOPE)]
        self.assertTrue(all(-0.20 < value < -0.02 for value in deltas))
        best = max(member["gain_over_anchor_f1"] for member in (V44, V45, SCOPE))
        self.assertAlmostEqual(best, -0.068217, places=6)


if __name__ == "__main__":
    unittest.main()
