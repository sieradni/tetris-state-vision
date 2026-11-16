"""Integrity controls for the M44 pixel edge ring receipt.

`docs/M44_PIXEL_EDGE.md` fixed the bar, the ranking floor and the branch readings before the run, and declared the
arm as the winner's configuration plus one zero-initialised projection of exactly 2,496 parameters. These tests
bind the receipt to that: the control reproduces the winner bit-exactly, the arm differs by exactly the declared
change, the verdict is the predeclared harmful branch, and the error mix is the one the document quotes - equal
true positives, more false positives, and the per-row late-row counts that rose.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPE = json.loads((ROOT/"docs/evidence/m44-pixel-edge-r1/scope.json").read_text())
DIAGNOSIS = json.loads((ROOT/"docs/evidence/m44-pixel-edge-r1/diagnosis.json").read_text())


def manifest(run):
    return json.loads((ROOT/"runs"/run/"checkpoint/manifest.json").read_text())


def effective(config):
    return {key: (0 if value is None else value) for key, value in config.items()}


class IsolationTest(unittest.TestCase):
    def test_the_control_is_the_winner_bit_for_bit(self):
        control, winner = manifest("m44-control-winner-config"), manifest("m30-level-calibrate-r5")
        self.assertEqual(control["weights_sha256"], winner["weights_sha256"])
        self.assertEqual(control["weights_sha256"][:12], "eddec1fd4af2")
        self.assertEqual(control["parameters"], winner["parameters"])

    def test_the_arm_differs_by_exactly_its_declared_projection(self):
        arm, control = manifest("m44-pixel-edge"), manifest("m44-control-winner-config")
        self.assertEqual(arm["parameters"]-control["parameters"], 12*192+192)
        self.assertEqual(arm["parameters"], 8026058)
        differing = sorted(key for key in set(arm["config"]) | set(control["config"])
                           if effective(arm["config"]).get(key) != effective(control["config"]).get(key))
        self.assertEqual(differing, ["occupancy_pixel_edge"])
        self.assertEqual((arm["config"]["occupancy_pixel_edge"],
                          control["config"]["occupancy_pixel_edge"]), (1, 0))

    def test_the_arm_is_a_distinct_checkpoint_and_not_inert(self):
        arm = manifest("m44-pixel-edge")
        self.assertNotEqual(arm["weights_sha256"], manifest("m44-control-winner-config")["weights_sha256"])
        self.assertEqual(SCOPE["arm"]["weights_sha256"], arm["weights_sha256"])


class DeclaredTest(unittest.TestCase):
    def test_the_bar_is_the_winners_rescored_readings(self):
        self.assertEqual(SCOPE["bar"], {"best": 0.689775, "f1_at_zero": 0.372881, "met": False})
        self.assertFalse(SCOPE["cleared"])
        self.assertEqual(SCOPE["gates"], {"attribution": True, "per_cell_f1": False, "ranking": False})

    def test_the_verdict_is_the_predeclared_harmful_branch(self):
        # harmful required dF1@0 <= -0.010 or AUC below the 0.949529 floor; both held
        self.assertLessEqual(SCOPE["gain_over_anchor_f1"], -0.010)
        self.assertLess(SCOPE["own"]["auc"], 0.949529)
        self.assertAlmostEqual(SCOPE["gain_over_anchor_f1"], -0.068217, places=6)
        self.assertAlmostEqual(SCOPE["auc_delta_vs_anchor"], -0.020521, places=6)
        self.assertAlmostEqual(SCOPE["gain_over_anchor_best"], -0.139153, places=6)


class MeasuredTest(unittest.TestCase):
    def test_the_readings_are_the_ones_the_document_quotes(self):
        self.assertAlmostEqual(SCOPE["own"]["f1_at_zero"], 0.304665, places=6)
        self.assertAlmostEqual(SCOPE["own"]["auc"], 0.934008, places=6)
        self.assertAlmostEqual(SCOPE["own_best"], 0.550622, places=6)
        self.assertAlmostEqual(SCOPE["anchored_best"], 0.486322, places=6)

    def test_the_split_sums_to_the_measured_span(self):
        self.assertTrue(SCOPE["fractions_sum_to_one"])
        self.assertAlmostEqual(SCOPE["span_closed_fraction"]+SCOPE["placement_fraction"]+
                               SCOPE["representation_fraction"], 1.0, places=5)
        self.assertAlmostEqual(SCOPE["span_points"]+SCOPE["placement_points"]+
                               SCOPE["representation_points"], SCOPE["anchors"]["span"], places=5)
        self.assertEqual(SCOPE["short_stage"], "placement")

    def test_precision_fell_and_recall_was_untouched(self):
        errors = DIAGNOSIS["errors_at_zero"]
        self.assertEqual(errors["true_positive"]["count"], 209)
        self.assertEqual(errors["false_negative"]["count"], 5)
        self.assertEqual(errors["false_positive"]["count"], 949)
        anchor_precision = 209/(209+698)
        self.assertAlmostEqual(DIAGNOSIS["precision_at_zero"], 0.180484, places=6)
        self.assertAlmostEqual(DIAGNOSIS["precision_at_zero"]-anchor_precision, -0.049946, places=6)
        self.assertAlmostEqual(DIAGNOSIS["recall_at_zero"], 0.976636, places=6)

    def test_the_late_row_mass_rose_and_its_per_row_counts(self):
        comparison = SCOPE["comparison_to_m35"]
        self.assertEqual(comparison["anchor_false_positives"], 698)
        self.assertEqual(comparison["late_rows_10_19_before"], 140)
        self.assertEqual(comparison["late_rows_10_19_now"], 240)
        rows = comparison["per_row"]
        self.assertEqual(rows["15"], {"before": 111, "now": 138})
        for row in ("13", "14", "18", "19"):
            self.assertEqual(rows[row]["before"], 0 if row != "14" else 1, row)
            self.assertGreater(rows[row]["now"], 0, row)
        self.assertEqual(sum(row["now"] for row in rows.values()), 949)


if __name__ == "__main__":
    unittest.main()
