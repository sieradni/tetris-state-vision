"""The power analysis: its definitions, its arithmetic and its refusal to pass a stale read."""
import math
import statistics
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]

import power_analysis as power


def clusters(*triples):
    return [{"seed": 20261101, "group_id": f"g{index}", "base128": tuple(triple[:3]),
             "budget192": tuple(triple[3:])} for index, triple in enumerate(triples)]


class Definitions(unittest.TestCase):
    def test_the_detectable_floor_is_the_two_sided_power_multiplier(self):
        multiplier = (statistics.NormalDist().inv_cdf(0.975) +
                      statistics.NormalDist().inv_cdf(0.80))
        self.assertAlmostEqual(power.minimum_detectable_effect(0.029549),
                               multiplier*0.029549, places=9)
        self.assertAlmostEqual(power.minimum_detectable_effect(1.0), 2.801585, places=5)

    def test_power_recovers_the_target_at_the_floor_and_rises_with_effect(self):
        error = 0.03
        self.assertAlmostEqual(power.power_at(power.minimum_detectable_effect(error), error),
                               0.80, places=4)
        self.assertAlmostEqual(power.power_at(0.0, error), 0.05, places=4)
        self.assertGreater(power.power_at(0.09, error), power.power_at(0.05, error))
        self.assertEqual(power.power_at(-0.09, error), power.power_at(0.09, error))

    def test_degenerate_inputs_are_refused(self):
        for value in (0.0, -1.0):
            with self.assertRaises(ValueError):
                power.minimum_detectable_effect(value)
        with self.assertRaises(ValueError):
            power.power_at(0.05, 0.0)
        with self.assertRaises(ValueError):
            power.seeds_needed(0.0, 0.01, 0.01)


class Influence(unittest.TestCase):
    def test_cluster_influence_telescopes_to_zero_and_matches_a_removal(self):
        rows = clusters((6, 4, 2, 5, 5, 2), (5, 5, 3, 6, 4, 3), (7, 3, 1, 6, 4, 1),
                        (4, 6, 4, 5, 5, 4), (6, 4, 2, 6, 4, 2), (5, 5, 3, 5, 5, 3))
        values = power.influence(rows, "budget192")
        self.assertAlmostEqual(sum(values), 0.0, places=9)
        # Dropping one cluster from the pooled read moves F1 by the linearisation to first order.
        full = power.pooled_f1(rows, "budget192")
        reduced = power.pooled_f1(rows[:-1], "budget192")
        weight = (len(rows)-1)/len(rows)
        self.assertLess(abs((full-reduced)-values[-1]*weight), 0.02)

    def test_a_paired_standard_error_is_finite_and_positive(self):
        rows = clusters((6, 4, 2, 5, 5, 2), (5, 5, 3, 6, 4, 3), (7, 3, 1, 6, 4, 1),
                        (4, 6, 4, 5, 5, 4), (6, 4, 2, 6, 4, 2), (5, 5, 3, 5, 5, 3))
        self.assertGreater(power.paired_standard_error(rows, "budget192"), 0.0)


class Decomposition(unittest.TestCase):
    def test_the_three_parts_add_to_the_out_of_fold_read(self):
        parts = power.decompose(0.50, 0.40, 0.45)
        self.assertAlmostEqual(sum(parts.values()), 0.45, places=12)
        self.assertAlmostEqual(parts["zero_boundary_cost"], -0.10, places=12)
        self.assertAlmostEqual(parts["transfer_cost"], 0.05, places=12)

    def test_a_boundary_cost_is_negative_when_zero_is_worse_than_the_best_boundary(self):
        self.assertLess(power.decompose(0.60, 0.35, 0.55)["zero_boundary_cost"], 0.0)


class Population(unittest.TestCase):
    def test_between_population_variance_is_corrected_and_clamped(self):
        variances = power.population_variance([0.10, 0.11, 0.09], [0.01, 0.01, 0.01])
        self.assertAlmostEqual(variances["raw"], 0.0001*2/3, places=12)
        self.assertAlmostEqual(variances["corrected"], 0.0, places=12)
        self.assertTrue(variances["clamped"])
        wide = power.population_variance([0.0, 0.0, -0.26], [0.002, 0.002, 0.002])
        self.assertGreater(wide["corrected"], 0.01)

    def test_seeds_needed_falls_as_the_effect_grows(self):
        small = power.seeds_needed(0.10, 0.02, 0.003)
        large = power.seeds_needed(0.30, 0.02, 0.003)
        self.assertGreater(small, large)
        self.assertEqual(large, math.ceil(
            (statistics.NormalDist().inv_cdf(0.975)+statistics.NormalDist().inv_cdf(0.80))**2
            *0.023/0.09))


if __name__ == "__main__":
    unittest.main()
