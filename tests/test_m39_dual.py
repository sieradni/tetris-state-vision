"""Controls for the M39 dual-placement read (form 13).

`docs/M39_DUAL_PLACEMENT.md` declares the arm as the same `30` statistics sampled at the predicted lattice
and at the support lattice, concatenated to `60` inputs through the one hidden projection. These tests bind
that: the width is `60`, the halves are genuinely different placements, the form adds only the hidden input
width, step zero is the published read, and the declared boundary stops where the predeclaration says.
"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from tetris_vision.model import (ModelConfig, TetrisStateNet, PIXEL_QUERY_FEATURES,  # noqa: E402
                                 cell_pixel_statistics, pixel_fixed_lattice, pixel_support_lattice)


def support_mask(top=8, bottom=56, left=6, right=58, height=64, width=64):
    valid = torch.zeros(1, 1, height, width, dtype=torch.bool)
    valid[:, :, top:bottom, left:right] = True
    return valid


def model(form, standardise=1):
    torch.manual_seed(17)
    return TetrisStateNet(replace(ModelConfig.pilot(), occupancy_pixel_query=form,
                                  occupancy_read_standardise=standardise))


class DeclaredWidthTest(unittest.TestCase):
    def test_the_form_reads_sixty_inputs(self):
        self.assertEqual(model(13).occupancy_pixel_query[0].in_features, 2*PIXEL_QUERY_FEATURES)

    def test_the_form_adds_only_the_hidden_input_width(self):
        sampled = sum(p.numel() for p in model(3).parameters())
        dual = sum(p.numel() for p in model(13).parameters())
        self.assertEqual(dual-sampled, PIXEL_QUERY_FEATURES*64)

    def test_the_declared_boundary_is_where_the_predeclaration_says(self):
        for form in (11, 12, 13):
            replace(ModelConfig(), occupancy_pixel_query=form)
        with self.assertRaises(ValueError):
            replace(ModelConfig(), occupancy_pixel_query=14)


class BothPlacementsTest(unittest.TestCase):
    def test_the_two_halves_are_different_placements_of_the_same_cells(self):
        valid = support_mask()
        predicted = pixel_fixed_lattice(26, 10, "cpu", torch.float64)
        supported = pixel_support_lattice(valid, 26, 10, "cpu", torch.float64)
        self.assertFalse(torch.allclose(predicted, supported))
        coordinates = model(0, standardise=0).coordinates
        pixels = torch.rand(1, 3, 64, 64, dtype=torch.float64)
        offsets = torch.zeros(1, 1, 9, 2, dtype=torch.float64)
        left = cell_pixel_statistics(pixels, predicted, coordinates, offsets)
        right = cell_pixel_statistics(pixels, supported, coordinates, offsets)
        self.assertEqual(tuple(left.shape)[-1], PIXEL_QUERY_FEATURES)
        self.assertFalse(torch.allclose(left, right))

    def test_the_halves_coincide_when_the_support_fills_the_capture(self):
        valid = torch.ones(1, 1, 64, 64, dtype=torch.bool)
        predicted = pixel_fixed_lattice(26, 10, "cpu", torch.float64)
        supported = pixel_support_lattice(valid, 26, 10, "cpu", torch.float64)
        self.assertTrue(torch.allclose(predicted, supported, atol=1e-12))


class StepZeroTest(unittest.TestCase):
    def test_the_dual_form_is_the_published_read_at_step_zero(self):
        pixels = torch.rand(1, 3, 64, 64)
        valid = support_mask()
        with torch.no_grad():
            published = model(0, standardise=0)(pixels, valid)["final"]["occupancy"]
        base, lever = model(0, standardise=0), model(13)
        lever.load_state_dict(base.state_dict(), strict=False)
        with torch.no_grad():
            read = lever(pixels, valid)["final"]["occupancy"]
        self.assertTrue(torch.equal(published, read))


if __name__ == "__main__":
    unittest.main()
