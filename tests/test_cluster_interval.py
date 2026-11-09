"""The clustering, design-effect and resampling helpers are exact and reproducible."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from cluster_interval import (cluster_bootstrap, clusters_of, design_effect, effective_size,
                              intra_cluster_correlation, percentile, rate_design_effect,
                              wilson_interval_with_design_effect)


class ClusteringTests(unittest.TestCase):
    def test_identical_cluster_means_give_no_design_effect(self):
        # Both clusters have mean .5: no between-cluster variance, so the ICC is at or below zero
        # and the design effect can never fall under 1.
        icc = intra_cluster_correlation([0, 1, 0, 1], ["a", "a", "b", "b"])
        self.assertLessEqual(icc, 0.0)
        self.assertEqual(design_effect(icc, 10), 1.0)
        self.assertEqual(design_effect(0.0, 10), 1.0)
        self.assertAlmostEqual(effective_size(5600, 1.0), 5600)

    def test_clusters_that_disagree_completely_give_icc_one(self):
        self.assertEqual(intra_cluster_correlation([0, 0, 0, 1, 1, 1],
                                                   ["a", "a", "a", "b", "b", "b"]), 1.0)

    def test_a_cluster_constant_outcome_has_zero_icc_not_negative(self):
        # All failures: no between-cluster variance is estimable.
        self.assertEqual(intra_cluster_correlation([0]*8, ["a"]*4+["b"]*4), 0.0)

    def test_cluster_constant_outcomes_give_a_large_design_effect(self):
        values = [1, 1, 1, 1, 0, 0, 0, 0]
        icc = intra_cluster_correlation(values, ["a"]*4+["b"]*4)
        self.assertGreater(icc, 0.5)
        self.assertGreater(design_effect(icc, 4), 2)

    def test_icc_is_reproducible_and_order_independent_for_equal_clusters(self):
        values, ids = [1, 1, 0, 1, 0, 0], ["a", "a", "a", "b", "b", "b"]
        self.assertAlmostEqual(intra_cluster_correlation(values, ids),
                               intra_cluster_correlation(values[::-1], ids[::-1]))

    def test_rate_design_effect_uses_between_cluster_dispersion(self):
        # Perfectly separated clusters: far above 1. Identical rates: exactly 1.
        self.assertGreater(rate_design_effect([(10, 10), (10, 0)]), 5)
        self.assertEqual(rate_design_effect([(10, 5), (10, 5)]), 1.0)

    def test_a_degenerate_rate_reports_no_fabricated_design_effect(self):
        self.assertEqual(rate_design_effect([(6, 0), (6, 0), (6, 0)]), 1.0)
        self.assertEqual(rate_design_effect([(6, 6), (6, 6)]), 1.0)

    def test_clusters_without_trials_carry_no_weight(self):
        self.assertEqual(rate_design_effect([(6, 3), (6, 3), (0, 0)]), 1.0)
        with self.assertRaises(ValueError):
            rate_design_effect([(6, 3), (0, 0)])

    def test_design_effect_reduced_wilson_widens_the_interval(self):
        plain = wilson_interval_with_design_effect(15, 33, 1.0)
        corrected = wilson_interval_with_design_effect(15, 33, 4.0)
        self.assertLess(corrected["low"], plain["low"])
        self.assertGreater(corrected["high"], plain["high"])
        self.assertAlmostEqual(corrected["estimate"], plain["estimate"])
        self.assertEqual(corrected["effective_trials"], 8.25)

    def test_the_bootstrap_is_seeded_sorted_and_skips_degenerate_draws(self):
        rows = [{"id": f"{index:02d}", "value": float(index)} for index in range(8)]
        clusters = clusters_of(list(reversed(rows)), "id")
        self.assertEqual([name for name, _ in clusters], sorted(row["id"] for row in rows))
        calls, statistic = [], None

        def statistic(sample):
            calls.append(sample)
            values = [row["value"] for cluster in sample for row in cluster]
            return None if len(sample) % 7 == 0 else {"mean": sum(values)/len(values)}
        first = cluster_bootstrap(statistic, clusters, ("mean",), draws=200, seed=7)
        second = cluster_bootstrap(statistic, clusters, ("mean",), draws=200, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["clusters"], 8)
        self.assertEqual(first["seed"], 7)
        self.assertLess(first["mean"]["low"], first["mean"]["high"])
        self.assertGreaterEqual(first["mean"]["valid_draws"], 100)

    def test_percentile_endpoints_and_interpolation(self):
        ordered = [0.0, 1.0, 2.0, 3.0]
        self.assertEqual(percentile(ordered, 0.0), 0.0)
        self.assertEqual(percentile(ordered, 1.0), 3.0)
        self.assertAlmostEqual(percentile(ordered, 0.5), 1.5)

    def test_malformed_inputs_are_refused(self):
        for call in (lambda: intra_cluster_correlation([], []),
                     lambda: intra_cluster_correlation([1, 2], ["a"]),
                     lambda: intra_cluster_correlation([1], ["a"]),
                     lambda: intra_cluster_correlation([1, 2, 3], ["a", "a", "a"]),
                     lambda: design_effect(0.5, 0),
                     lambda: effective_size(10, 0),
                     lambda: rate_design_effect([(1, 1)]),
                     lambda: wilson_interval_with_design_effect(1, 2, 0.5),
                     lambda: cluster_bootstrap(lambda sample: {"x": 1.0}, (), ("x",)),
                     lambda: cluster_bootstrap(lambda sample: None, (("a", [1]), ("b", [2])), ("x",), draws=100),
                     lambda: percentile([], 0.5)):
            with self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
