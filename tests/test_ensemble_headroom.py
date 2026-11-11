"""The ensemble-headroom rule and ensemble arithmetic are pinned to the declaration."""
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]/"scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ensemble_headroom import f1_at, headroom_verdict, mean_logits  # noqa: E402


def sweep(auc, best_f1, f1_zero=0.4000):
    return {"auc": auc, "best": {"f1": best_f1, "threshold": 1.05},
            "rows": [{"threshold": -0.15, "f1": 0.1}, {"threshold": 0.0, "f1": f1_zero}]}


class EnsembleArithmeticTests(unittest.TestCase):
    def test_mean_is_unweighted_per_cell(self):
        self.assertEqual(mean_logits([[1.0, 2.0], [3.0, 4.0]]), [2.0, 3.0])
        self.assertEqual(mean_logits([[0.1], [0.2]]), [0.15])
        self.assertEqual(mean_logits([[2.0, -4.0], [0.0, 0.0], [1.0, 2.0]]), [1.0, -0.666667])

    def test_one_arm_or_mismatched_widths_refused(self):
        for vectors in ([], [[1.0]], [[1.0], [1.0, 2.0]], [[1.0], ["x"]]):
            with self.assertRaises(ValueError):
                mean_logits(vectors)

    def test_missing_threshold_refused(self):
        with self.assertRaises(ValueError):
            f1_at({"rows": [{"threshold": 0.15, "f1": 0.2}]})


class HeadroomVerdictTests(unittest.TestCase):
    def test_bars_met_clears(self):
        verdict = headroom_verdict(sweep(0.9500, 0.7000), [sweep(0.8500, 0.5200)])
        self.assertEqual(verdict["label"], "ensemble_clears")

    def test_ordering_met_and_floor_met_is_variance(self):
        verdict = headroom_verdict(sweep(0.9200, 0.5500), [sweep(0.8500, 0.4900), sweep(0.8300, 0.4700)])
        self.assertEqual(verdict["label"], "variance_limited")
        self.assertAlmostEqual(verdict["best_delta"], 0.0600, places=6)

    def test_floor_boundary_is_inclusive(self):
        verdict = headroom_verdict(sweep(0.9000, 0.5400), [sweep(0.8500, 0.4900)])
        self.assertEqual(verdict["label"], "variance_limited")
        self.assertAlmostEqual(verdict["best_delta"], 0.0500, places=6)

    def test_small_gain_is_bias(self):
        verdict = headroom_verdict(sweep(0.9200, 0.5200), [sweep(0.8500, 0.4900)])
        self.assertEqual(verdict["label"], "bias_limited")

    def test_ordering_short_of_bar_stays_bias_even_with_large_gain(self):
        verdict = headroom_verdict(sweep(0.8999, 0.6000), [sweep(0.8500, 0.5000)])
        self.assertEqual(verdict["label"], "bias_limited")

    def test_malformed_input_refused(self):
        for ensemble, members in (({}, [sweep(0.8, 0.4)]), (sweep(0.8, 0.4), []),
                                  (sweep(0.8, 0.4), [{}])):
            with self.assertRaises(ValueError):
                headroom_verdict(ensemble, members)


if __name__ == "__main__":
    unittest.main()
