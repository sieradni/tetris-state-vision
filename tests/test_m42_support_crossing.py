"""Controls for the M42 decision-side arm.

`docs/M42_SUPPORT_CROSSING.md` declares the arm as the winner's recorded configuration with exactly one
varied fact - `occupancy_support_crossing` - applied as a per-board constant inside the head. These tests pin
the isolation: the same parameters, the same sampled points, the same within-board order, an added level that
actually acts, and the declared boundary.
"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch
import tetris_vision.model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from tetris_vision.model import ModelConfig, TetrisStateNet  # noqa: E402


def winner(**kw):
    torch.manual_seed(29)
    return TetrisStateNet(replace(ModelConfig.pilot(), dimension=192, blocks=6,
                                  occupancy_pixel_query=3, occupancy_read_standardise=1, **kw))


def mask(top=8, bottom=56, left=6, right=58):
    valid = torch.zeros(1, 1, 64, 64, dtype=torch.bool)
    valid[:, :, top:bottom, left:right] = True
    return valid


class IsolationTest(unittest.TestCase):
    def test_the_arm_adds_no_parameter(self):
        self.assertEqual(sum(p.numel() for p in winner(occupancy_support_crossing=1).parameters()),
                         sum(p.numel() for p in winner().parameters()))

    def test_the_arm_moves_no_sampled_point(self):
        pixels = torch.rand(1, 3, 64, 64)
        base, arm = winner(), winner(occupancy_support_crossing=1)
        arm.load_state_dict(base.state_dict())
        with torch.no_grad():
            self.assertTrue(torch.equal(base(pixels, mask())["final"]["sample_points"],
                                        arm(pixels, mask())["final"]["sample_points"]))

    def test_the_rule_acts_as_one_constant_per_board(self):
        pixels = torch.rand(1, 3, 64, 64)
        base, arm = winner(), winner(occupancy_support_crossing=1)
        arm.load_state_dict(base.state_dict())
        with torch.no_grad():
            plain = base(pixels, mask())["final"]["occupancy"]
            shifted = arm(pixels, mask())["final"]["occupancy"]
        self.assertFalse(torch.equal(plain, shifted))
        difference = plain-shifted
        board = difference[0, :20]
        # a constant per board: every cell moves by the same amount, so the ordering inside a board is fixed
        self.assertTrue(torch.allclose(board, board.flatten()[0].expand_as(board), atol=1e-6))
        self.assertTrue(torch.equal(torch.argsort(plain[0].flatten()), torch.argsort(shifted[0].flatten())))

    def test_the_rule_is_applied_by_the_form_and_not_silently_dropped(self):
        pixels = torch.rand(1, 3, 64, 64)
        calls = {}
        real = tetris_vision.model.support_crossing_shift
        for flag in (0, 1):
            with mock.patch.object(tetris_vision.model, "support_crossing_shift",
                                   side_effect=real) as spy:
                winner(occupancy_support_crossing=flag)(pixels, mask())
                calls[flag] = spy.call_count
        self.assertGreaterEqual(calls[1], 1)
        self.assertEqual(calls[0], 0)

    def test_the_declared_boundary_is_where_the_predeclaration_says(self):
        replace(ModelConfig(), occupancy_support_crossing=1)
        with self.assertRaises(ValueError):
            replace(ModelConfig(), occupancy_support_crossing=2)


if __name__ == "__main__":
    unittest.main()
