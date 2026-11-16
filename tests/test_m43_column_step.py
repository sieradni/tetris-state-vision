"""Mechanical controls for the M43 per-column step crossing.

`docs/M43_COLUMN_STEP_CROSSING.md` declares the rule exactly: one constant per column, taken at the column's own
sharpest row-to-row descent, with no parameter, no annotation, no prevalence and no search grid. These tests pin
that declaration, including the two properties the arm's reading depends on - the cut lands on the step rather
than on the spurious upper blob, and the order of cells inside a column cannot move.
"""
import inspect
import unittest
from dataclasses import replace

import torch

from tetris_vision.model import ModelConfig, TetrisStateNet, column_step_crossing

# the winner's declared read, at one block so the mechanics stay cheap to exercise
WINNER = dict(dimension=192, blocks=2, occupancy_read_standardise=1, occupancy_read_level=1,
              occupancy_pixel_query=3)
GUARDED = ("label", "target", "occupied", "prevalence", "thresholds")


def board(logits):
    return logits.reshape(1, logits.shape[0], logits.shape[1])


class DeclaredRuleTest(unittest.TestCase):
    def test_the_cut_lands_on_the_step_and_not_the_blob_above_it(self):
        # rows 0-4 are the naively occupied run; rows 5-11 are the moderate upper blob the diagnosis found;
        # rows 12-19 are clearly lower. The largest descent is the true step (2.0 -> 0.3 beats 0.3 -> -1.5).
        column = torch.tensor([2.0]*5+[0.3]*7+[0.0]*8)
        out = column_step_crossing(board(torch.stack((column, column), -1)), 20)
        positive = (out[0, :, 0] > 0).tolist()
        self.assertEqual(positive, [True]*5+[False]*15)

    def test_the_cut_is_the_midpoint_of_the_step(self):
        column = torch.tensor([4.0, 4.0, 0.0, 0.0])
        out = column_step_crossing(board(torch.stack((column, column), -1)), 4)
        self.assertEqual(out[0, :, 0].tolist(), [2.0, 2.0, -2.0, -2.0])

    def test_a_column_without_a_step_is_untouched(self):
        flat = torch.ones(1, 20, 10)
        rising = torch.arange(20.0).reshape(1, 20, 1).expand(1, 20, 10).contiguous()
        self.assertTrue(torch.equal(column_step_crossing(flat, 20), flat))
        self.assertTrue(torch.equal(column_step_crossing(rising, 20), rising))

    def test_one_constant_per_column_so_the_order_inside_it_cannot_move(self):
        torch.manual_seed(4)
        logits = torch.randn(3, 20, 10)
        out = column_step_crossing(logits, 20)
        shift = logits-out
        for b in range(3):
            for c in range(10):
                self.assertLess(float(shift[b, :, c].std()), 1e-6)
        for b in range(3):
            for c in range(10):
                order = logits[b, :, c].argsort()
                self.assertTrue(torch.equal(order, out[b, :, c].argsort()))

    def test_the_rule_is_deterministic_on_ties(self):
        column = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
        first = column_step_crossing(board(torch.stack((column, column), -1)), 5)
        second = column_step_crossing(board(torch.stack((column, column), -1)), 5)
        self.assertTrue(torch.equal(first, second))

    def test_the_spawn_rows_pass_through(self):
        logits = torch.zeros(1, 24, 10)
        logits[0, :20] = torch.tensor([2.0]*10+[0.0]*10).reshape(20, 1)
        logits[0, 20:] = 7.0
        out = column_step_crossing(logits, 20)
        self.assertTrue(torch.equal(out[0, 20:], logits[0, 20:]))


class IsolationTest(unittest.TestCase):
    def test_the_flag_adds_no_parameter(self):
        base = sum(p.numel() for p in TetrisStateNet(ModelConfig(**WINNER)).parameters())
        arm = sum(p.numel() for p in TetrisStateNet(
            ModelConfig(**WINNER, occupancy_column_step_crossing=1)).parameters())
        self.assertEqual(base, arm)

    def test_the_sampled_points_are_identical(self):
        torch.manual_seed(11)
        pixels, mask = torch.rand(2, 3, 64, 128), torch.ones(2, 1, 64, 128, dtype=torch.bool)
        base = TetrisStateNet(ModelConfig(**WINNER))
        arm = TetrisStateNet(ModelConfig(**WINNER, occupancy_column_step_crossing=1))
        arm.load_state_dict(base.state_dict())
        base.eval(), arm.eval()
        with torch.no_grad():
            first = base(pixels, mask)["stages"][0]
            second = arm(pixels, mask)["stages"][0]
        self.assertTrue(torch.equal(first["sample_points"], second["sample_points"]))
        self.assertTrue(torch.equal(first["sample_offsets"], second["sample_offsets"]))

    def test_the_forward_applies_exactly_the_declared_rule(self):
        torch.manual_seed(12)
        pixels, mask = torch.rand(2, 3, 64, 128), torch.ones(2, 1, 64, 128, dtype=torch.bool)
        base = TetrisStateNet(ModelConfig(**WINNER))
        arm = TetrisStateNet(ModelConfig(**WINNER, occupancy_column_step_crossing=1))
        arm.load_state_dict(base.state_dict())
        base.eval(), arm.eval()
        with torch.no_grad():
            plain = base(pixels, mask)["stages"][0]["occupancy"]
            ruled = arm(pixels, mask)["stages"][0]["occupancy"]
        expected = column_step_crossing(plain, ModelConfig().visible_rows)
        self.assertTrue(torch.allclose(ruled, expected, atol=1e-6))

    def test_the_flag_is_validated(self):
        for value in (0, 1):
            replace(ModelConfig(), occupancy_column_step_crossing=value)
        for value in (2, -1, True, 0.5):
            with self.assertRaises(ValueError):
                replace(ModelConfig(), occupancy_column_step_crossing=value)


class LabelBlindnessTest(unittest.TestCase):
    def test_the_rule_takes_no_annotation_and_names_none(self):
        parameters = list(inspect.signature(column_step_crossing).parameters)
        self.assertEqual(parameters, ["logits", "visible_rows"])
        body = inspect.getsource(column_step_crossing).lower()
        for token in GUARDED:
            self.assertNotIn(token, body)


if __name__ == "__main__":
    unittest.main()
