"""Controls for the M33 fixed-lattice pixel read.

`docs/M33_FIXED_LATTICE.md` declares the sampling placement as a constant of the capture geometry and
nothing else. These tests bind that: the lattice is the uniform grid with the campaign's conventions,
it is a pure function of the board's shape, the form's step zero is the published read, and the form
adds no parameter to the winner's arm.
"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from tetris_vision.model import (ModelConfig, TetrisStateNet,  # noqa: E402
                                 cell_pixel_statistics, pixel_fixed_lattice)

SIDE = 64


def model(form, **flags):
    torch.manual_seed(11)
    return TetrisStateNet(replace(ModelConfig.pilot(), occupancy_pixel_query=form, **flags))


class LatticeTest(unittest.TestCase):
    def test_the_lattice_is_the_capture_divided_into_the_board(self):
        lattice = pixel_fixed_lattice(20, 10, "cpu", torch.float64)
        self.assertEqual(tuple(lattice.shape), (1, 3, 2))
        origin, column, row = (lattice[0, index].tolist() for index in range(3))
        self.assertEqual(origin, [0.0, 1.0])
        self.assertEqual(column, [0.1, 0.0])
        self.assertAlmostEqual(row[1], -0.05, places=12)

    def test_the_lattice_is_a_pure_function_of_the_board_shape(self):
        first = pixel_fixed_lattice(20, 10, "cpu", torch.float64)
        second = pixel_fixed_lattice(20, 10, "cpu", torch.float64)
        self.assertTrue(torch.equal(first, second))
        other = pixel_fixed_lattice(20, 10, "cpu", torch.float64)
        self.assertTrue(torch.equal(first, other))

    def test_the_fixed_lattice_does_not_reach_the_decoded_one(self):
        rows, columns = 26, 10
        pixels = torch.rand(1, 3, SIDE, SIDE)
        fixed = pixel_fixed_lattice(rows, columns, pixels.device, pixels.dtype)
        coordinates = torch.stack(torch.meshgrid(torch.arange(columns), torch.arange(rows),
                                                 indexing="ij"), -1).reshape(-1, 2)+.5
        offsets = torch.zeros(1, 1, 1, 2)
        first = cell_pixel_statistics(pixels, fixed, coordinates, offsets)
        moved = fixed.clone()
        moved[0, 0] = moved[0, 0]+torch.tensor([0.0, 0.0])
        second = cell_pixel_statistics(pixels, moved, coordinates, offsets)
        self.assertTrue(torch.equal(first, second))
        distinct = fixed.clone()
        distinct[0, 0] = distinct[0, 0]+torch.tensor([0.02, -0.02])
        third = cell_pixel_statistics(pixels, distinct, coordinates, offsets)
        self.assertFalse(torch.equal(first, third))


class FormTest(unittest.TestCase):
    def test_step_zero_is_the_published_read(self):
        pixels = torch.rand(1, 3, SIDE, SIDE)
        valid = torch.ones(1, 1, SIDE, SIDE, dtype=torch.bool)
        with torch.no_grad():
            published = model(0)(pixels, valid)["final"]["occupancy"]
        base, lever = model(0), model(6)
        lever.load_state_dict(base.state_dict(), strict=False)
        with torch.no_grad():
            read = lever(pixels, valid)["final"]["occupancy"]
        self.assertTrue(torch.equal(published, read))

    def test_the_form_adds_no_parameter_to_the_winner(self):
        winner = sum(p.numel() for p in model(3, occupancy_read_standardise=1).parameters())
        fixed = sum(p.numel() for p in model(6, occupancy_read_standardise=1).parameters())
        self.assertEqual(winner, fixed)

    def test_the_form_is_declared(self):
        with self.assertRaises(ValueError):
            replace(ModelConfig(), occupancy_pixel_query=10)


if __name__ == "__main__":
    unittest.main()
