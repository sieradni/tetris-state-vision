"""Controls for the M34 support-derived pixel read.

`docs/M34_SUPPORT_LATTICE.md` declares the placement as the bounding box of the capture's valid pixels
and nothing else. These tests bind that: the lattice maps the board onto the support exactly, it is
invariant to added padding, it refuses an empty mask, the form's step zero is the published read, and
the form adds no parameter.
"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from tetris_vision.model import ModelConfig, TetrisStateNet, pixel_support_lattice  # noqa: E402

ROWS, COLUMNS = 26, 10


def mask(height=64, width=64, top=8, bottom=56, left=6, right=58):
    valid = torch.zeros(1, 1, height, width, dtype=torch.bool)
    valid[:, :, top:bottom, left:right] = True
    return valid


def model(form, **flags):
    torch.manual_seed(7)
    return TetrisStateNet(replace(ModelConfig.pilot(), occupancy_pixel_query=form, **flags))


class LatticeTest(unittest.TestCase):
    def test_the_lattice_maps_the_board_onto_the_support(self):
        lattice = pixel_support_lattice(mask(), ROWS, COLUMNS, "cpu", torch.float64)
        origin, column, row = (lattice[0, index].tolist() for index in range(3))
        self.assertAlmostEqual(origin[0], 6/64, places=12)
        self.assertAlmostEqual(origin[1], 56/64, places=12)
        self.assertAlmostEqual(column[0], (58-6)/64/COLUMNS, places=12)
        self.assertAlmostEqual(column[1], 0.0, places=12)
        self.assertAlmostEqual(row[1], -(56-8)/64/ROWS, places=12)

    def test_added_padding_does_not_move_the_lattice(self):
        small = pixel_support_lattice(mask(), ROWS, COLUMNS, "cpu", torch.float64)
        large = pixel_support_lattice(mask(128, 128, 16, 112, 12, 116), ROWS, COLUMNS,
                                      "cpu", torch.float64)
        self.assertTrue(torch.allclose(small, large, atol=1e-12))

    def test_an_empty_mask_is_refused(self):
        with self.assertRaises(ValueError):
            pixel_support_lattice(torch.zeros(1, 1, 64, 64, dtype=torch.bool), ROWS, COLUMNS,
                                  "cpu", torch.float64)


class FormTest(unittest.TestCase):
    def test_step_zero_is_the_published_read(self):
        pixels = torch.rand(1, 3, 64, 64)
        valid = mask()
        with torch.no_grad():
            published = model(0)(pixels, valid)["final"]["occupancy"]
        base, lever = model(0), model(7)
        lever.load_state_dict(base.state_dict(), strict=False)
        with torch.no_grad():
            read = lever(pixels, valid)["final"]["occupancy"]
        self.assertTrue(torch.equal(published, read))

    def test_the_form_adds_no_parameter_to_the_winner(self):
        winner = sum(p.numel() for p in model(3, occupancy_read_standardise=1).parameters())
        support = sum(p.numel() for p in model(7, occupancy_read_standardise=1).parameters())
        self.assertEqual(winner, support)

    def test_the_form_is_declared(self):
        with self.assertRaises(ValueError):
            replace(ModelConfig(), occupancy_pixel_query=10)


if __name__ == "__main__":
    unittest.main()
