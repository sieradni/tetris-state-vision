"""Integrity controls for the M29 mechanism instrument and its gate.

`docs/M29_PIXEL_QUERY_FORM.md` declares the instrument; the receipts are
`docs/evidence/m29-pixel-form-r2/scope.json` and `docs/evidence/m29-amplitude-r1/scope.json`.
These tests bind what the pass claims: one fixed population, an amplitude read that excludes
starvation, per-variant gains that confirm exactly one mechanism, a split that sums to the span,
and a gate verdict that follows the predeclared rule.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPE = json.loads((ROOT/"docs/evidence/m29-pixel-form-r2/scope.json").read_text())
AMPLITUDE = json.loads((ROOT/"docs/evidence/m29-amplitude-r1/scope.json").read_text())


class PopulationTest(unittest.TestCase):
    def test_every_variant_is_scored_on_the_fixed_population(self):
        self.assertEqual(SCOPE["population"]["fixed_cells"], 2986)
        self.assertEqual(SCOPE["population"]["boards"], 28)
        self.assertEqual((SCOPE["population"]["occupied_identity"],
                          SCOPE["population"]["occupied_anchored"]), (214, 309))
        for name, item in SCOPE["variants"].items():
            self.assertEqual(item["own"]["cells"], 2986, name)
            self.assertEqual(item["anchored"]["cells"], 2986, name)

    def test_the_anchors_reproduce_the_committed_arms(self):
        anchors = SCOPE["anchors"]
        self.assertAlmostEqual(anchors["reference_own_best"], 0.543974, places=6)
        self.assertAlmostEqual(anchors["pixel_best"], 0.987179, places=6)
        self.assertAlmostEqual(anchors["control_best"], 0.219626, places=6)
        self.assertAlmostEqual(anchors["form_one_best"], 0.425703, places=6)


class AmplitudeTest(unittest.TestCase):
    def test_the_projection_is_not_starved(self):
        self.assertFalse(AMPLITUDE["starved"])
        self.assertEqual(AMPLITUDE["relative_gain"], 0.703541)
        self.assertGreater(AMPLITUDE["relative_gain"], AMPLITUDE["amplitude_floor"])
        self.assertEqual(AMPLITUDE["run"], "m28-pixelread-lever-r1")

    def test_the_amplitude_read_is_a_real_composition(self):
        self.assertAlmostEqual(AMPLITUDE["statistics_to_logit_gain_norm"],
                               SCOPE["amplitude"]["statistics_to_logit_gain_norm"], places=6)
        self.assertEqual(SCOPE["amplitude"]["tensors"]["projection"], "occupancy_pixel_query.weight")


class MechanismTest(unittest.TestCase):
    def test_only_the_hidden_layer_clears_the_gain_floor(self):
        variants = SCOPE["variants"]
        self.assertGreaterEqual(variants["hidden-r2"]["gain_over_form_one"], 0.05)
        self.assertLess(variants["centred-r3"]["gain_over_form_one"], 0.05)
        self.assertLess(variants["neighbours-r2"]["gain_over_form_one"], 0.05)

    def test_the_decision_names_the_confirmed_and_excluded_mechanisms(self):
        decision = SCOPE["decision"]
        self.assertEqual(decision["cause"], "form:hidden-r2")
        self.assertEqual(decision["confirmed"], ["hidden-r2"])
        self.assertEqual(sorted(decision["excluded"]), ["centred-r3", "neighbours-r2"])
        self.assertFalse(decision["cleared"])

    def test_the_winner_beats_both_the_form_one_arm_and_the_reference(self):
        hidden = SCOPE["variants"]["hidden-r2"]
        self.assertAlmostEqual(hidden["own_best"], 0.610101, places=6)
        self.assertGreater(hidden["own_best"], 0.543974)
        self.assertAlmostEqual(hidden["gain_over_control"], 0.390475, places=6)


class SplitTest(unittest.TestCase):
    def test_every_split_reproduces_the_span(self):
        span = SCOPE["anchors"]["span"]
        self.assertAlmostEqual(span, 0.443205, places=6)
        for name, item in SCOPE["variants"].items():
            self.assertTrue(item["fractions_sum_to_one"], name)
            self.assertAlmostEqual(item["span_closed_fraction"]+item["placement_fraction"]+
                                   item["representation_fraction"], 1.0, delta=2e-6)
            self.assertAlmostEqual(item["span_points"]+item["placement_points"]+
                                   item["representation_points"], span, places=5)
            self.assertAlmostEqual(item["span_points"], item["own_best"]-0.543974, places=5)

    def test_the_hidden_layer_form_is_the_first_positive_closure(self):
        hidden = SCOPE["variants"]["hidden-r2"]
        self.assertGreater(hidden["span_points"], 0.0)
        self.assertAlmostEqual(hidden["span_closed_fraction"], 0.149202, places=6)
        self.assertGreater(hidden["representation_fraction"],
                           abs(hidden["placement_fraction"]))


class GateTest(unittest.TestCase):
    def test_the_gate_fails_only_on_the_operating_point(self):
        gates = SCOPE["decision"]["gates"]
        self.assertEqual(gates, {"per_cell_f1": False, "ranking": False, "attribution": True})
        hidden = SCOPE["variants"]["hidden-r2"]
        self.assertGreaterEqual(hidden["own"]["auc"], 0.90)
        self.assertLess(hidden["own_best"], 0.70)
        self.assertLess(hidden["own"]["f1_at_zero"], 0.70)

    def test_the_study_peaks_stay_under_the_ceiling(self):
        for name, item in SCOPE["variants"].items():
            self.assertLessEqual(item["peak_bytes"], 1152*1024**2, name)


if __name__ == "__main__":
    unittest.main()
