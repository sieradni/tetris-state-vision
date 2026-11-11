"""The predeclared trunk-width verdict rule and its config-derived read-out."""
import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]

from width_lever_read import lever_verdict, readout_statistics, visible_board_logits


def swept(auc, best, exact=0):
    return {"auc": auc, "best": {"f1": best},
            "exact_boards_at_best": {"nonempty": exact, "empty": 7-exact},
            "rows": [{"threshold": 0.0, "f1": round(best-0.05, 4)}, {"threshold": 0.5, "f1": best}]}


class Readout(unittest.TestCase):
    CONFIG = {"visible_rows": 20, "spawn_rows": 6, "columns": 10}

    def test_the_visible_read_is_the_first_rows_of_a_single_channel_map(self):
        occupancy = torch.arange(26*10, dtype=torch.float32).reshape(1, 26, 10)
        values = visible_board_logits(occupancy, self.CONFIG)
        self.assertEqual(tuple(values.shape), (200,))
        self.assertEqual(float(values[0]), 0.0)
        self.assertEqual(float(values[199]), 199.0)

    def test_the_slice_width_comes_from_the_config_not_a_literal(self):
        config = {"visible_rows": 26, "spawn_rows": 0, "columns": 10}
        self.assertEqual(visible_board_logits(torch.zeros((1, 26, 10)), config).numel(), 260)

    def test_a_map_that_does_not_match_the_declared_board_is_refused(self):
        for wrong in (torch.zeros((1, 26, 10, 1)), torch.zeros((1, 20, 10)), torch.zeros((2, 26, 10))):
            with self.assertRaises(ValueError):
                visible_board_logits(wrong, self.CONFIG)

    def test_the_recorded_statistics_separate_the_visible_block_from_the_spawn_margin(self):
        occupancy = torch.cat((torch.full((1, 20, 10), 2.0), torch.full((1, 6, 10), -1.0)), 1)
        stats = readout_statistics(occupancy, self.CONFIG)
        self.assertEqual(stats["shape"], [1, 26, 10])
        self.assertAlmostEqual(stats["visible_mean_abs"], 2.0, places=6)
        self.assertAlmostEqual(stats["visible_std"], 0.0, places=6)
        self.assertAlmostEqual(stats["spawn_mean_abs"], 1.0, places=6)


class Branches(unittest.TestCase):
    def test_clearing_attribution_with_the_auc_bar_helps(self):
        verdict = lever_verdict(swept(0.80, 0.40), swept(0.91, 0.50), flips=100, known_cells=667)
        self.assertEqual(verdict["label"], "capacity_helps")
        self.assertAlmostEqual(verdict["attribution"], 0.10, places=6)
        self.assertFalse(verdict["fit_gate_opened"])

    def test_opening_the_fit_gate_helps_even_below_the_auc_bar(self):
        verdict = lever_verdict(swept(0.80, 0.40), swept(0.85, 0.50, exact=3), flips=100, known_cells=667)
        self.assertEqual(verdict["label"], "capacity_helps")
        self.assertTrue(verdict["fit_gate_opened"])

    def test_below_the_attribution_floor_with_a_live_control_is_a_null(self):
        verdict = lever_verdict(swept(0.80, 0.40), swept(0.85, 0.42), flips=100, known_cells=667)
        self.assertEqual(verdict["label"], "capacity_null")

    def test_a_flagged_gain_that_decides_nothing_is_named_as_such(self):
        verdict = lever_verdict(swept(0.80, 0.40), swept(0.85, 0.50), flips=100, known_cells=667)
        self.assertEqual(verdict["label"], "capacity_fits_not_decides")

    def test_a_collapsed_control_cannot_speak_about_capacity(self):
        verdict = lever_verdict(swept(0.60, 0.40), swept(0.75, 0.55), flips=100, known_cells=667)
        self.assertEqual(verdict["label"], "floor_limited")

    def test_an_engagement_floor_that_fails_voids_the_read(self):
        verdict = lever_verdict(swept(0.80, 0.40), swept(0.95, 0.60), flips=10, known_cells=667)
        self.assertEqual(verdict["label"], "inactive")

    def test_the_rule_refuses_an_empty_population_or_a_malformed_arm(self):
        with self.assertRaises(ValueError):
            lever_verdict(swept(0.80, 0.40), swept(0.95, 0.60), flips=0, known_cells=0)
        with self.assertRaises(ValueError):
            lever_verdict({"auc": 0.8}, swept(0.95, 0.60), flips=1, known_cells=10)
        with self.assertRaises(ValueError):
            lever_verdict(swept(0.80, 0.40), swept(0.95, 0.60), flips=-1, known_cells=10)
