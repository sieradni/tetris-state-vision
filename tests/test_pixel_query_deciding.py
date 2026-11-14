"""Integrity controls for the M28 deciding-population re-score.

`docs/M28_PIXEL_QUERY.md` declares the re-score and
`docs/evidence/m28-pixel-query-deciding-r2/scope.json` is its receipt. These tests bind the
receipt's three load-bearing claims: one fixed population, references that reproduce the
committed M27 endpoints, and an arithmetic split that actually sums to the span. They read
the committed evidence only, so they hold in a fresh checkout.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPE = json.loads((ROOT/"docs/evidence/m28-pixel-query-deciding-r2/scope.json").read_text())


class PopulationTest(unittest.TestCase):
    def test_the_population_is_the_fixed_committed_set(self):
        population = SCOPE["population"]
        self.assertEqual(population["fixed_cells"], 2986)
        self.assertEqual(population["boards"], 28)
        self.assertEqual(population["occupied_identity"], 214)
        self.assertEqual(population["occupied_anchored"], 309)

    def test_both_arms_are_scored_on_every_fixed_cell(self):
        for name, read in SCOPE["reads"].items():
            self.assertEqual(read["cells"], 2986, name)

    def test_the_arms_own_mappings_do_not_cover_the_population(self):
        coverage = SCOPE["population"]["reference_targets_covered_by_arm"]
        self.assertEqual(coverage, {"base": 1861, "lever": 1583})
        for value in coverage.values():
            self.assertLess(value, 2986)


class ReferenceTest(unittest.TestCase):
    def test_the_references_reproduce_the_committed_endpoints(self):
        references = SCOPE["references"]
        self.assertAlmostEqual(references["own_best"], 0.543974, places=6)
        self.assertAlmostEqual(references["pixel_best"], 0.987179, places=6)
        self.assertAlmostEqual(references["own_f1_at_zero"], 0.439024, places=6)
        self.assertAlmostEqual(references["anchored_best"], 0.762590, places=6)
        self.assertAlmostEqual(references["pixel_f1_at_zero"], 0.974763, places=6)

    def test_the_pixel_read_is_identical_for_both_arms(self):
        self.assertEqual(SCOPE["reads"]["pixel"]["best"]["f1"],
                         SCOPE["reads"]["pixel"]["best"]["f1"])
        self.assertEqual(SCOPE["arithmetic"]["arms"]["base"]["pixel_best"],
                         SCOPE["arithmetic"]["arms"]["lever"]["pixel_best"])


class ArithmeticTest(unittest.TestCase):
    def test_the_split_reproduces_the_span(self):
        span = SCOPE["arithmetic"]["span"]
        for name, values in SCOPE["arithmetic"]["arms"].items():
            self.assertTrue(values["fractions_sum_to_one"], name)
            # The receipt rounds each fraction to six places and asserts the exact sum before
            # writing, so the rounded triple can differ from one by at most a rounding unit.
            self.assertAlmostEqual(values["span_closed_fraction"]+values["placement_fraction"]+
                                   values["representation_fraction"], 1.0, delta=2e-6)
            self.assertAlmostEqual(values["span_points"]+values["placement_points"]+
                                   values["representation_points"], span, places=5)
            self.assertAlmostEqual(values["span_points"], values["own_best"]-0.543974, places=5)
            self.assertAlmostEqual(values["placement_points"],
                                   values["anchored_best"]-values["own_best"], places=6)
            self.assertAlmostEqual(values["representation_points"],
                                   0.987179-values["anchored_best"], places=6)

    def test_the_control_relative_closure_is_the_attribution_over_the_span(self):
        arithmetic = SCOPE["arithmetic"]
        self.assertAlmostEqual(arithmetic["control_relative_span_closed"],
                               arithmetic["attribution_best_delta"]/arithmetic["span"], places=6)

    def test_the_residual_is_representation_for_both_arms(self):
        for name, values in SCOPE["arithmetic"]["arms"].items():
            self.assertEqual(values["residual"], "representation", name)
            self.assertGreater(values["representation_fraction"], values["placement_fraction"])


class VerdictTest(unittest.TestCase):
    def test_the_verdict_follows_the_declared_gates(self):
        gates = SCOPE["gates"]["conditions"]
        self.assertEqual(gates, {"per_cell_f1": False, "ranking": False, "attribution": True})
        self.assertEqual(SCOPE["verdict"], "not-cleared")
        self.assertEqual(SCOPE["cause"], "representation-limited-partial-gain")

    def test_the_secondary_read_is_labelled_per_arm(self):
        self.assertEqual(SCOPE["secondary_population_cells"], 3316)
        self.assertIn("per-arm", SCOPE["secondary_note"])
        self.assertIn("not used for any verdict", SCOPE["secondary_note"])

    def test_the_study_peaks_stay_under_the_declared_ceiling(self):
        for name, peak in SCOPE["study_peaks"].items():
            self.assertLessEqual(peak, 1152*1024**2, name)


if __name__ == "__main__":
    unittest.main()
