"""The predeclared per-cell decision rule, and the mechanism label it can name."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from percell_gate import (CELL_F1_THRESHOLD, DECIDING_SPLIT, decision_mechanism, occupied_cell_f1,
                          percell_gate_opens)


def population(true_positive, false_positive, false_negative, exact=0, images=7):
    return {"empty": {"images": 2, "exact_boards": 2, "known_cells": 400, "true_positive": 0,
                      "true_negative": 400, "false_positive": 0, "false_negative": 0},
            "nonempty": {"images": images, "exact_boards": exact, "known_cells": 200*images,
                         "true_positive": true_positive, "true_negative": 0,
                         "false_positive": false_positive, "false_negative": false_negative}}


class PerCellGateRuleTests(unittest.TestCase):
    def test_the_threshold_is_the_predeclared_seventy_hundredths(self):
        self.assertEqual(CELL_F1_THRESHOLD, 0.70)
        self.assertEqual(DECIDING_SPLIT, "training")

    def test_the_quantity_is_the_occupied_cell_f1_not_an_exact_match_count(self):
        # 2TP/(2TP+FP+FN): a perfect decision scores 1, a constant-empty read-out scores 0.
        self.assertEqual(occupied_cell_f1(population(168, 0, 0)), 1.0)
        self.assertEqual(occupied_cell_f1(population(0, 0, 168)), 0.0)
        self.assertAlmostEqual(occupied_cell_f1(population(34, 59, 134)), 68/261)

    def test_an_exact_tie_opens_and_a_partial_fit_below_it_stands(self):
        self.assertTrue(percell_gate_opens(population(7, 0, 6)))       # 14/20 = 0.70
        self.assertFalse(percell_gate_opens(population(7, 0, 7)))      # 14/21 < 0.70
        self.assertFalse(percell_gate_opens(population(34, 59, 134)))

    def test_no_exact_match_count_can_open_or_close_the_gate_by_itself(self):
        # Seven exact boards and zero discrimination: stands.
        self.assertFalse(percell_gate_opens(population(0, 0, 168, exact=7)))
        # Every board wrong and the cells still discriminate: opens.
        self.assertTrue(percell_gate_opens(population(160, 2, 8, exact=0)))

    def test_the_mechanism_names_which_error_mass_pins_a_standing_decision(self):
        self.assertEqual(decision_mechanism(population(0, 0, 168)), "constant-empty")
        self.assertEqual(decision_mechanism(population(34, 59, 134)), "coverage-limited")
        self.assertEqual(decision_mechanism(population(34, 134, 59)), "specificity-limited")
        self.assertEqual(decision_mechanism(population(34, 134, 134)), "both-limited")
        self.assertEqual(decision_mechanism(population(0, 40, 0)), "specificity-limited")
        self.assertEqual(decision_mechanism(population(160, 2, 8)), "discriminating")

    def test_a_missing_population_is_refused_rather_than_read_as_fit(self):
        with self.assertRaisesRegex(ValueError, "nonempty locked population"):
            occupied_cell_f1({"empty": {"images": 2}})
        with self.assertRaisesRegex(ValueError, "nonempty cell counts"):
            occupied_cell_f1({"nonempty": {"true_positive": 1}})
        with self.assertRaisesRegex(ValueError, "F1 needs occupied or predicted cells"):
            occupied_cell_f1(population(0, 0, 0))


EVIDENCE = ROOT/"docs/evidence/m3-percell-balance-r1-errors-r1/result.json"


class PerCellGateEvidenceTests(unittest.TestCase):
    def test_the_recorded_run_reproduces_a_standing_reallocated_decision(self):
        import json
        result = json.loads(EVIDENCE.read_text())
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["occupancy_factor"], 4)
        self.assertEqual(result["occupancy_positive_weight"], 4)
        self.assertTrue(result["automatic_metrics_reproduced"])
        populations = result["totals"][DECIDING_SPLIT]["locked_populations"]
        nonempty = populations["nonempty"]
        self.assertEqual(nonempty["images"], 7)
        self.assertEqual((nonempty["true_positive"], nonempty["false_positive"],
                          nonempty["false_negative"]), (67, 196, 101))
        self.assertAlmostEqual(occupied_cell_f1(populations), 134/431, places=12)
        self.assertFalse(percell_gate_opens(populations))
        self.assertEqual(decision_mechanism(populations), "specificity-limited")
        self.assertEqual(nonempty["exact_boards"], 0)
        development = result["totals"]["development"]["locked_populations"]
        self.assertAlmostEqual(occupied_cell_f1(development), 550/1028, places=12)
        self.assertFalse(percell_gate_opens(development))


if __name__ == "__main__":
    unittest.main()
