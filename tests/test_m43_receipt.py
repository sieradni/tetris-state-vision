"""Integrity controls for the M43 per-column step crossing receipt.

`docs/M43_COLUMN_STEP_CROSSING.md` fixed the bar, the ranking tolerance and the branch readings before the run,
and declared the arm as the winner's configuration with `occupancy_column_step_crossing` as the only varied
recorded fact. These tests bind the receipt to that declaration: the control reproduces the winner bit-exactly,
the verdict is the predeclared harmful branch rather than a post-hoc reading, the split sums, and the per-row
counts are the ones the document quotes - including the rows the rule emptied and the row it filled.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPE = json.loads((ROOT/"docs/evidence/m43-column-step-r1/scope.json").read_text())
DIAGNOSIS = json.loads((ROOT/"docs/evidence/m43-column-step-r1/diagnosis.json").read_text())


def manifest(run):
    return json.loads((ROOT/"runs"/run/"checkpoint/manifest.json").read_text())


def effective(config):
    return {key: (0 if value is None else value) for key, value in config.items()}


class IsolationTest(unittest.TestCase):
    def test_the_control_is_the_winner_bit_for_bit(self):
        control, winner = manifest("m43-control-winner-config"), manifest("m30-level-calibrate-r5")
        self.assertEqual(control["weights_sha256"], winner["weights_sha256"])
        self.assertEqual(control["weights_sha256"][:12], "eddec1fd4af2")
        self.assertEqual(control["parameters"], winner["parameters"])

    def test_the_arm_is_the_control_plus_one_recorded_field(self):
        arm, control = manifest("m43-column-step"), manifest("m43-control-winner-config")
        self.assertEqual(arm["parameters"], control["parameters"])
        self.assertEqual(arm["parameters"], 8023562)
        differing = sorted(key for key in set(arm["config"]) | set(control["config"])
                           if effective(arm["config"]).get(key) != effective(control["config"]).get(key))
        self.assertEqual(differing, ["occupancy_column_step_crossing"])
        self.assertEqual((arm["config"]["occupancy_column_step_crossing"],
                          control["config"]["occupancy_column_step_crossing"]), (1, 0))

    def test_the_arm_is_a_distinct_checkpoint_and_not_inert(self):
        arm = manifest("m43-column-step")
        self.assertNotEqual(arm["weights_sha256"], manifest("m43-control-winner-config")["weights_sha256"])
        self.assertEqual(SCOPE["arm"]["run"], "m43-column-step")
        self.assertEqual(SCOPE["arm"]["weights_sha256"], arm["weights_sha256"])


class DeclaredTest(unittest.TestCase):
    def test_the_bar_is_the_winners_rescored_readings(self):
        self.assertEqual(SCOPE["bar"], {"best": 0.689775, "f1_at_zero": 0.372881, "met": False})
        self.assertFalse(SCOPE["cleared"])
        self.assertEqual(SCOPE["gates"], {"attribution": True, "per_cell_f1": False, "ranking": False})

    def test_the_verdict_is_the_predeclared_harmful_branch(self):
        # harmful required dF1@0 <= -0.010 or dAUC < -0.005; both held, so it is not neutral
        self.assertLessEqual(SCOPE["gain_over_anchor_f1"], -0.010)
        self.assertLess(SCOPE["auc_delta_vs_anchor"], -0.005)
        self.assertAlmostEqual(SCOPE["gain_over_anchor_f1"], -0.102873, places=6)
        self.assertAlmostEqual(SCOPE["auc_delta_vs_anchor"], -0.201629, places=6)
        self.assertAlmostEqual(SCOPE["gain_over_anchor_best"], -0.417120, places=6)


class MeasuredTest(unittest.TestCase):
    def test_the_readings_are_the_ones_the_document_quotes(self):
        self.assertAlmostEqual(SCOPE["own"]["f1_at_zero"], 0.270008, places=6)
        self.assertAlmostEqual(SCOPE["own"]["auc"], 0.752899, places=6)
        self.assertAlmostEqual(SCOPE["own_best"], 0.272654, places=6)
        self.assertAlmostEqual(SCOPE["own"]["best"]["threshold"], -0.15, places=6)
        self.assertAlmostEqual(SCOPE["anchored_best"], 0.259563, places=6)

    def test_the_split_sums_to_the_measured_span(self):
        self.assertTrue(SCOPE["fractions_sum_to_one"])
        self.assertAlmostEqual(SCOPE["span_closed_fraction"]+SCOPE["placement_fraction"]+
                               SCOPE["representation_fraction"], 1.0, places=5)
        self.assertAlmostEqual(SCOPE["span_points"]+SCOPE["placement_points"]+
                               SCOPE["representation_points"], SCOPE["anchors"]["span"], places=5)
        self.assertEqual(SCOPE["short_stage"], "closure")

    def test_the_high_margin_late_row_mass_rose(self):
        comparison = SCOPE["comparison_to_m35"]
        self.assertEqual(comparison["anchor_false_positives"], 698)
        self.assertEqual(DIAGNOSIS["errors_at_zero"]["false_positive"]["count"], 856)
        self.assertEqual(comparison["late_rows_10_19_before"], 140)
        self.assertEqual(comparison["late_rows_10_19_now"], 173)
        rows = comparison["per_row"]
        for row in ("3", "4", "9", "16", "17"):
            self.assertEqual(rows[row]["now"], 0, row)
        self.assertEqual(rows["15"], {"before": 111, "now": 173})
        self.assertEqual(rows["0"], {"before": 0, "now": 43})

    def test_the_error_mix_is_the_one_recorded(self):
        errors = DIAGNOSIS["errors_at_zero"]
        self.assertEqual(errors["true_positive"]["count"], 167)
        self.assertEqual(errors["false_negative"]["count"], 47)
        self.assertAlmostEqual(DIAGNOSIS["precision_at_zero"], 0.163245, places=6)
        self.assertEqual(sum(row["now"] for row in SCOPE["comparison_to_m35"]["per_row"].values()), 856)
        self.assertAlmostEqual(DIAGNOSIS["recall_at_zero"], 0.780374, places=6)


if __name__ == "__main__":
    unittest.main()
