"""The predeclared occupancy ranking rule: grid, AUC, curve and verdict."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from occupancy_sweep import (AUC_THRESHOLD, CHANCE_AUC, DECIDING_SPLIT, F1_THRESHOLD, evaluate,
                             pooled_auc, replay_keywords, sweep_verdict, threshold_grid)

EVIDENCE = ROOT/"docs/evidence/m3-sweep-occupancy-r1-ranking-r1/result.json"


def board(split, occupied, empty):
    return {"split": split, "sample_id": "synthetic", "group_id": "synthetic",
            "logits": list(occupied)+list(empty), "occupied": [1]*len(occupied)+[0]*len(empty)}


class SweepRuleTests(unittest.TestCase):
    def test_the_grid_is_fixed_before_any_receipt(self):
        grid = threshold_grid()
        self.assertEqual(len(grid), 81)
        self.assertEqual((grid[0], grid[-1], round(grid[1]-grid[0], 2)), (-6.0, 6.0, 0.15))
        self.assertIn(0.0, grid)
        self.assertEqual((AUC_THRESHOLD, CHANCE_AUC, F1_THRESHOLD), (0.90, 0.5, 0.70))
        self.assertEqual(DECIDING_SPLIT, "training")

    def test_the_local_replay_policy_cannot_drift(self):
        from m3_error_analysis import replay_keywords as frozen_policy
        for capacity, recipe in ((64, {"final_capture": True}), (0, {})):
            self.assertEqual(replay_keywords(recipe, capacity), frozen_policy(recipe, capacity))
        with self.assertRaisesRegex(ValueError, "probe capture policy differs"):
            replay_keywords({}, 64)
        with self.assertRaisesRegex(ValueError, "bounded component capacity"):
            replay_keywords({}, 65)

    def test_the_auc_counts_ties_as_one_half(self):
        self.assertEqual(pooled_auc([1.0, 1.0], [1.0, 1.0]), 0.5)
        self.assertEqual(pooled_auc([2.0, 3.0], [-1.0, 0.0]), 1.0)
        self.assertEqual(pooled_auc([-3.0, -2.0], [3.0, 4.0]), 0.0)
        self.assertAlmostEqual(pooled_auc([1.0, -1.0], [-1.0, 1.0]), 0.5)

    def test_a_logit_exactly_at_the_threshold_counts_as_empty(self):
        summary = evaluate([board("training", [1.0, 1.0], [1.0])], thresholds=(1.0,))
        row = summary["rows"][0]
        self.assertEqual((row["true_positive"], row["false_positive"], row["false_negative"]), (0, 0, 2))

    def test_counts_come_from_the_deciding_population_only(self):
        summary = evaluate([board("training", [2.0]*2, [-2.0]*2), board("training", [], [-3.0]*4)],
                           thresholds=(0.0, -2.5))
        zero, low = summary["rows"]
        self.assertEqual((zero["true_positive"], zero["false_positive"], zero["false_negative"],
                          zero["true_negative"]), (2, 0, 0, 2))
        self.assertEqual((low["true_positive"], low["false_positive"]), (2, 2))
        self.assertEqual(zero["exact_empty_boards"], 1)
        self.assertEqual(summary["empty_population"], {"images": 1, "known_cells": 4,
                                                       "false_positive": 0, "false_positive_rate": 0.0})

    def test_the_best_point_breaks_f1_ties_by_precision_then_lower_threshold(self):
        summary = evaluate([board("training", [3.0, 3.0], [-3.0])])
        self.assertEqual(summary["best"], max(summary["rows"],
                         key=lambda row: (row["f1"], row["precision"], -row["threshold"])))
        self.assertEqual(summary["best"]["f1"], 1.0)

    def test_a_perfect_ranking_is_informative(self):
        summary = evaluate([board("training", [5.0]*7, [-5.0]*9)])
        self.assertEqual(summary["auc"], 1.0)
        self.assertEqual(sweep_verdict(summary), "informative")
        self.assertEqual(summary["exact_boards_at_best"]["nonempty"], 1)

    def test_a_weak_ranking_is_uninformative(self):
        summary = evaluate([board("training", [0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0],
                                  [0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0])])
        self.assertLess(summary["auc"], AUC_THRESHOLD)
        self.assertEqual(sweep_verdict(summary), "uninformative")

    def test_both_conditions_are_required_and_boundary_ties_are_met(self):
        self.assertEqual(sweep_verdict({"auc": 0.90, "best": {"f1": 0.70}}), "informative")
        self.assertEqual(sweep_verdict({"auc": 0.95, "best": {"f1": 0.69}}), "uninformative")
        self.assertEqual(sweep_verdict({"auc": 0.89, "best": {"f1": 0.95}}), "uninformative")

    def test_a_missing_population_is_refused_rather_than_read_as_informative(self):
        with self.assertRaisesRegex(ValueError, "occupied and empty known-cell logits"):
            pooled_auc([1.0], [])
        with self.assertRaisesRegex(ValueError, "occupied and empty known-cell logits"):
            pooled_auc([], [1.0])
        with self.assertRaisesRegex(ValueError, "at least one saved board"):
            evaluate([])
        with self.assertRaisesRegex(ValueError, "sweep AUC and best operating point"):
            sweep_verdict({"auc": 0.95})
        with self.assertRaisesRegex(ValueError, "sweep AUC and best operating point"):
            sweep_verdict({"best": {"f1": 0.9}})

    def test_a_board_missing_its_labels_is_refused(self):
        with self.assertRaisesRegex(ValueError, "saved board logits and labels"):
            evaluate([{"split": "training", "sample_id": "x", "logits": [1.0], "occupied": []}])
        with self.assertRaisesRegex(ValueError, "saved board logits and labels"):
            evaluate([{"split": "training", "sample_id": "x", "logits": [1.0], "occupied": [2]}])
        with self.assertRaisesRegex(ValueError, "one split at a time"):
            evaluate([board("training", [1.0], [-1.0]), board("development", [1.0], [-1.0])])


if __name__ == "__main__":
    unittest.main()
