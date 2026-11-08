"""The predeclared fit-gate rule, and the mechanism that pinned the measured negative."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from fit_gate import DECIDING_SPLIT, FIT_GATE_THRESHOLD, fit_gate_opens, occupancy_mechanism

EVIDENCE = ROOT/"docs/evidence/m3-fit-occupancy-r1-errors-r1/result.json"


def population(exact, true_positive, false_negative=0):
    return {"empty": {"images": 2, "exact_boards": 2, "known_cells": 400, "true_positive": 0,
                      "true_negative": 400, "false_positive": 0, "false_negative": 0},
            "nonempty": {"images": 7, "exact_boards": exact, "known_cells": 1400,
                         "true_positive": true_positive, "true_negative": 1173,
                         "false_positive": 59, "false_negative": false_negative}}


class FitGateRuleTests(unittest.TestCase):
    def test_the_threshold_is_the_predeclared_three_of_seven(self):
        self.assertEqual(FIT_GATE_THRESHOLD, 3)
        self.assertEqual(DECIDING_SPLIT, "training")
        self.assertFalse(fit_gate_opens(population(2, 40, 1)))
        self.assertTrue(fit_gate_opens(population(3, 40, 1)))

    def test_empty_board_successes_alone_never_open_the_gate(self):
        totals = {"empty": {"images": 2, "exact_boards": 2, "true_positive": 0, "false_positive": 0,
                             "false_negative": 0},
                  "nonempty": {"images": 7, "exact_boards": 0, "true_positive": 0, "false_positive": 23,
                               "false_negative": 168}}
        self.assertFalse(fit_gate_opens(totals))
        self.assertEqual(occupancy_mechanism(totals), "constant-empty")

    def test_the_two_mechanisms_are_separated(self):
        self.assertEqual(occupancy_mechanism(population(0, 0, 168)), "constant-empty")
        self.assertEqual(occupancy_mechanism(population(0, 34, 134)), "per-cell-error")

    def test_a_missing_population_is_refused_rather_than_read_as_fit(self):
        with self.assertRaisesRegex(ValueError, "nonempty locked population"):
            fit_gate_opens({"empty": {"images": 2, "exact_boards": 2}})
        with self.assertRaisesRegex(ValueError, "exact-board counts"):
            fit_gate_opens({"nonempty": {"images": 7}})

    def test_the_recorded_run_reproduces_a_standing_gate_that_is_not_gradient_starved(self):
        result = json.loads(EVIDENCE.read_text())
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["occupancy_factor"], 4)
        self.assertTrue(result["automatic_metrics_reproduced"])
        populations = result["totals"][DECIDING_SPLIT]["locked_populations"]
        self.assertEqual(populations["nonempty"]["images"], 7)
        self.assertEqual(populations["nonempty"]["exact_boards"], 0)
        self.assertFalse(fit_gate_opens(populations))
        self.assertGreater(populations["nonempty"]["true_positive"], 0)
        self.assertEqual(occupancy_mechanism(populations), "per-cell-error")


if __name__ == "__main__":
    unittest.main()
