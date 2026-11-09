"""The reported intervals are closed form, bounded, and read by no verdict."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

import gate_power


class IntervalTests(unittest.TestCase):
    def test_wilson_interval_matches_the_closed_form(self):
        interval = gate_power.wilson_interval(0, 10)
        self.assertEqual(interval["low"], 0.0)
        self.assertAlmostEqual(interval["high"], 0.2775, places=4)
        self.assertEqual(interval["trials"], 10)
        self.assertEqual(interval["successes"], 0)

    def test_wilson_interval_brackets_its_estimate_and_widens_as_trials_fall(self):
        wide, narrow = gate_power.wilson_interval(67, 263), gate_power.wilson_interval(670, 2630)
        for interval in (wide, narrow):
            self.assertLessEqual(interval["low"], interval["estimate"])
            self.assertGreaterEqual(interval["high"], interval["estimate"])
            self.assertTrue(0 <= interval["low"] <= interval["high"] <= 1)
        self.assertGreater(wide["high"]-wide["low"], narrow["high"]-narrow["low"])

    def test_auc_interval_reproduces_the_published_standard_error(self):
        interval = gate_power.auc_interval(0.7491158395176253, 168, 1232)
        self.assertAlmostEqual(interval["standard_error"], 0.0226, places=4)
        self.assertAlmostEqual(interval["low"], 0.7049, places=3)
        self.assertAlmostEqual(interval["high"], 0.7934, places=3)
        self.assertEqual(interval["positives"], 168)
        self.assertEqual(interval["negatives"], 1232)

    def test_auc_interval_narrows_with_more_cells(self):
        small = gate_power.auc_interval(0.75, 168, 1232)
        large = gate_power.auc_interval(0.75, 600, 4400)
        self.assertGreater(small["standard_error"], large["standard_error"])

    def test_f1_and_its_gap_use_the_gate_formula(self):
        self.assertAlmostEqual(gate_power.f1_from_counts(67, 196, 101), 0.3109, places=4)
        self.assertAlmostEqual(gate_power.f1_gap(0.3470), 0.3530, places=4)
        self.assertAlmostEqual(gate_power.f1_from_counts(10, 0, 0), 1.0)

    def test_precision_and_recall_intervals_are_reported_for_the_composing_proportions(self):
        intervals = gate_power.precision_recall_interval(67, 196, 101)
        self.assertAlmostEqual(intervals["precision"]["estimate"], 67/263, places=6)
        self.assertAlmostEqual(intervals["recall"]["estimate"], 67/168, places=6)
        self.assertEqual(intervals["precision"]["trials"], 263)
        self.assertEqual(intervals["recall"]["trials"], 168)

    def test_malformed_inputs_are_refused(self):
        for call in (lambda: gate_power.wilson_interval(11, 10),
                     lambda: gate_power.wilson_interval(0, 0),
                     lambda: gate_power.wilson_interval(1.0, 10),
                     lambda: gate_power.wilson_interval(True, 10),
                     lambda: gate_power.auc_interval(1.2, 10, 10),
                     lambda: gate_power.auc_interval(0.5, 0, 10),
                     lambda: gate_power.auc_interval(0.5, 10.0, 10),
                     lambda: gate_power.precision_recall_interval(0, 10, 10),
                     lambda: gate_power.f1_from_counts(-1, 10, 10),
                     lambda: gate_power.f1_from_counts(0, 0, 0),
                     lambda: gate_power.f1_gap(1.5)):
            with self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
