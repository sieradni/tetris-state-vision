"""Controls for the M30 level mechanisms.

`docs/M30_OPERATING_POINT.md` declares three single-change mechanisms that can only move the
read's level. These tests bind the properties the interpretation depends on: each is exactly
identity at step zero, the level freedom preserves ordering, the calibration is monotone inside a
board, and the data-derived class budget is a real prevalence ratio rather than a constant.
"""
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from tetris_vision.losses import DATA_DERIVED_BUDGET, balanced_occupancy  # noqa: E402
from tetris_vision.model import ModelConfig, TetrisStateNet  # noqa: E402

SIDE = 64


def model(**flags):
    torch.manual_seed(5)
    return TetrisStateNet(replace(ModelConfig.pilot(), occupancy_pixel_query=3, **flags))


def inputs():
    pixels = torch.rand(1, 3, SIDE, SIDE)
    valid = torch.ones(1, 1, SIDE, SIDE, dtype=torch.bool)
    return pixels, valid


class StepZeroTest(unittest.TestCase):
    def test_every_mechanism_is_identity_at_step_zero(self):
        pixels, valid = inputs()
        with torch.no_grad():
            anchor = model()(pixels, valid)["final"]["occupancy"]
        for flags in ({"pixel_query_level": 1}, {"occupancy_read_standardise": 1},
                      {"occupancy_balance": 1}):
            base, arm = model(), model(**flags)
            arm.load_state_dict(base.state_dict(), strict=False)
            with torch.no_grad():
                read = arm(pixels, valid)["final"]["occupancy"]
            self.assertTrue(torch.equal(anchor, read), flags)

    def test_the_level_parameters_start_at_identity(self):
        arm = model(pixel_query_level=1)
        self.assertEqual(float(arm.pixel_query_level.abs().sum()), 0.0)
        gate = model(occupancy_read_standardise=1)
        self.assertEqual(float(gate.read_calibration.abs().sum()), 0.0)

    def test_the_mechanisms_are_refused_without_the_pixel_read(self):
        with self.assertRaises(ValueError):
            replace(ModelConfig(), pixel_query_level=1)
        with self.assertRaises(ValueError):
            replace(ModelConfig(), occupancy_read_standardise=1)
        with self.assertRaises(ValueError):
            replace(ModelConfig(), occupancy_pixel_query=3, pixel_query_level=2)


class OrderingTest(unittest.TestCase):
    def test_the_level_freedom_cannot_reorder_cells(self):
        pixels, valid = inputs()
        arm = model(pixel_query_level=1)
        with torch.no_grad():
            before = arm(pixels, valid)["final"]["occupancy"]
            arm.pixel_query_level[0] = 0.7
            arm.pixel_query_level[1] = 0.4
            after = arm(pixels, valid)["final"]["occupancy"]
        self.assertFalse(torch.equal(before, after))
        self.assertTrue(torch.equal(before.argsort(dim=-1), after.argsort(dim=-1)))
        spread = float((after-before).std())
        self.assertGreater(spread, 0.0)

    def test_the_calibration_is_monotone_inside_a_board(self):
        pixels, valid = inputs()
        arm = model(occupancy_read_standardise=1)
        with torch.no_grad():
            before = arm(pixels, valid)["final"]["occupancy"]
            arm.read_calibration[0] = 0.5
            after = arm(pixels, valid)["final"]["occupancy"]
        self.assertFalse(torch.equal(before, after))
        visible = ModelConfig.pilot().visible_rows
        self.assertTrue(torch.equal(before[:, :visible].argsort(dim=-1),
                                    after[:, :visible].argsort(dim=-1)))


class BalanceTest(unittest.TestCase):
    def test_the_data_derived_budget_is_the_prevalence_ratio(self):
        logits = torch.randn(2, 20, 10)
        target = torch.zeros(2, 20, 10)
        target[0, :3] = 1.0
        target[1, :9] = 1.0
        mask = torch.ones(2, 20, 10, dtype=torch.bool)
        derived = balanced_occupancy(logits, target, mask, -1)
        self.assertTrue(torch.isfinite(derived) and derived > 0)
        # the two images carry different prevalence, so the derived budget cannot equal a constant one
        self.assertNotAlmostEqual(float(derived),
                                  float(balanced_occupancy(logits, target, mask, 1)), places=6)
        clamped = balanced_occupancy(logits, target, mask, -1)
        self.assertAlmostEqual(float(derived), float(clamped), places=6)

    def test_the_sentinel_is_declared(self):
        from tetris_vision.losses import POSITIVE_CELL_WEIGHTS, require_positive_weight
        self.assertEqual(DATA_DERIVED_BUDGET, -1)
        self.assertNotIn(DATA_DERIVED_BUDGET, POSITIVE_CELL_WEIGHTS)
        with self.assertRaises(ValueError):
            require_positive_weight(DATA_DERIVED_BUDGET, {"arguments": {}})
        require_positive_weight(DATA_DERIVED_BUDGET, {"arguments": {"occupancy_balance": 1},
                                                      "recovery_contract": {"occupancy_balance": 1}})


if __name__ == "__main__":
    unittest.main()
