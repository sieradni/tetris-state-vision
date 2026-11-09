"""The conditioning rule, its boundaries and its tie conventions."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from board_conditioning import (ORACLE_BOARDS_MIN, THRESHOLD_SPREAD_BAR, board_counts, board_f1,
                                board_rows, classify, measure, oracle_board)
from occupancy_sweep import F1_THRESHOLD, threshold_grid

GRID = threshold_grid()


def board(name, cells):
    return {"board": name, "group": "g", "view": 0, "cells": cells}


class ConditioningRuleTests(unittest.TestCase):
    def test_the_grid_is_the_standing_eighty_one_points(self):
        self.assertEqual(len(GRID), 81)
        self.assertIn(0.0, GRID)
        self.assertEqual((GRID[0], GRID[-1]), (-6.0, 6.0))

    def test_a_perfectly_ranked_board_reaches_with_a_full_f1(self):
        entry = oracle_board(board("a", [(2.0, 1), (2.0, 1), (-2.0, 0), (-2.0, 0)]))
        self.assertEqual(entry["f1"], 1.0)
        self.assertEqual(entry["counts"], {"true_positive": 2, "false_positive": 0, "false_negative": 0})

    def test_an_inverted_board_never_reaches(self):
        # Two occupied cells against five empty ones: calling everything occupied still only
        # reaches 4/9, so no threshold on the grid clears the bar.
        entry = oracle_board(board("a", [(-3.0, 1), (-3.0, 1)]+[(3.0, 0)]*5))
        self.assertLess(entry["f1"], F1_THRESHOLD)
        self.assertAlmostEqual(entry["f1"], 4/9, places=6)

    def test_a_tie_keeps_the_lowest_threshold(self):
        # Every grid point strictly between the occupied and empty logits ties at F1 1.0, so the
        # declared convention must pick the lowest of them rather than an arbitrary one.
        cells = [(1.0, 1), (2.0, 1), (-1.0, 0), (-2.0, 0)]
        entry = oracle_board(board("a", cells))
        lowest = min(value for value in GRID if board_f1(board("a", cells), value) == 1.0)
        self.assertEqual(entry["f1"], 1.0)
        self.assertEqual(entry["threshold"], lowest)
        self.assertEqual(lowest, -0.9)

    def test_counts_and_f1_agree_with_the_definition(self):
        cells = [(2.0, 1), (-2.0, 1), (2.0, 0), (-2.0, 0)]
        counts = board_counts(board("a", cells), 0.0)
        self.assertEqual(counts, {"true_positive": 1, "false_positive": 1, "false_negative": 1})
        self.assertAlmostEqual(board_f1(board("a", cells), 0.0), 2/(2+1+1))

    def test_a_board_without_occupied_cells_has_no_f1(self):
        self.assertEqual(board_f1(board("a", [(1.0, 0), (-1.0, 0)]), 0.0), 0.0)

    def test_below_the_floor_is_representation_limited(self):
        oracle = [{"f1": 0.9, "threshold": -1.0} for _ in range(ORACLE_BOARDS_MIN-1)]
        oracle += [{"f1": 0.1, "threshold": 0.0} for _ in range(6)]
        verdict = classify(oracle)
        self.assertEqual(verdict["verdict"], "representation-limited")
        self.assertEqual(verdict["reaching"], ORACLE_BOARDS_MIN-1)

    def test_at_the_floor_with_disagreeing_thresholds_is_conditioned_boundary(self):
        oracle = [{"f1": 0.8, "threshold": -2.0}, {"f1": 0.8, "threshold": 0.0},
                  {"f1": 0.8, "threshold": 1.0}, {"f1": 0.8, "threshold": 2.0}]
        verdict = classify(oracle)
        self.assertEqual(verdict["verdict"], "conditioned-boundary")
        self.assertAlmostEqual(verdict["oracle_threshold_spread"], 4.0)

    def test_at_the_floor_with_agreeing_thresholds_is_inconsistent(self):
        oracle = [{"f1": 0.8, "threshold": -1.0}, {"f1": 0.9, "threshold": -0.9},
                  {"f1": 0.8, "threshold": -0.5}, {"f1": 0.8, "threshold": 0.0}]
        verdict = classify(oracle)
        self.assertEqual(verdict["verdict"], "inconsistent")
        self.assertLessEqual(verdict["oracle_threshold_spread"], THRESHOLD_SPREAD_BAR)

    def test_an_exact_bar_tie_reaches(self):
        oracle = [{"f1": F1_THRESHOLD, "threshold": 0.0} for _ in range(ORACLE_BOARDS_MIN)]
        self.assertEqual(classify(oracle)["reaching"], ORACLE_BOARDS_MIN)

    def test_board_rows_take_nonempty_deciding_boards_only(self):
        sweep = {"cases": [
            {"split": "training", "sample_id": "b", "group_id": "g", "view": 0,
             "logits": [1.0, -1.0], "occupied": [1, 0]},
            {"split": "training", "sample_id": "a", "group_id": "g", "view": 0,
             "logits": [-1.0], "occupied": [0]},
            {"split": "development", "sample_id": "d", "group_id": "g", "view": 0,
             "logits": [1.0], "occupied": [1]}]}
        rows = board_rows(sweep)
        self.assertEqual([row["board"] for row in rows], ["b"])

    def test_measure_reports_the_oracle_pooled_counts(self):
        sweep = {"cases": [
            {"split": "training", "sample_id": "a", "group_id": "g", "view": 0,
             "logits": [2.0, -2.0], "occupied": [1, 0]},
            {"split": "training", "sample_id": "b", "group_id": "g", "view": 0,
             "logits": [2.0, -2.0], "occupied": [1, 1]}]}
        result = measure(sweep)
        self.assertEqual(result["oracle_pooled"]["f1"], 1.0)
        self.assertEqual(result["classification"]["boards"], 2)

    def test_malformed_inputs_are_refused(self):
        for call in (lambda: board_rows({"cases": []}),
                     lambda: classify([]),
                     lambda: oracle_board(board("a", []), grid=())):
            with self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
