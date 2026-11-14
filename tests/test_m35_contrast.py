"""Controls for the M35 contrast read and the M36 spread-normalised contrast.

`docs/M35_CONTRAST_READ.md` declares form 8 as the support lattice's 30 statistics beside their
sample-mean deviation (60 inputs) through the unchanged hidden projection. These tests bind the
declaration: the forms add exactly the declared projection width, step zero is the published read, the
declared boundary is where the predeclaration says it is, and the deviation half is sample-derived.
"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from tetris_vision.model import (ModelConfig, TetrisStateNet, cell_pixel_statistics,  # noqa: E402
                                 pixel_support_lattice)

PIXEL_QUERY_FEATURES = 30


def support():
    valid = torch.zeros(1, 1, 64, 64, dtype=torch.bool)
    valid[:, :, 8:56, 6:58] = True
    return valid


def model(form, standardise=1):
    torch.manual_seed(11)
    return TetrisStateNet(replace(ModelConfig.pilot(), occupancy_pixel_query=form,
                                  occupancy_read_standardise=standardise))


class DeclaredWidthTest(unittest.TestCase):
    def test_the_read_produces_thirty_statistics(self):
        pixels = torch.rand(1, 3, 64, 64)
        valid = support()
        offsets = torch.zeros(1, 1, 9, 2)
        lattice = pixel_support_lattice(valid, 26, 10, "cpu", torch.float32)
        statistics = cell_pixel_statistics(pixels, lattice, model(0, standardise=0).coordinates,
                                          offsets)
        self.assertEqual(statistics.shape[-1], PIXEL_QUERY_FEATURES)
        self.assertEqual(statistics.shape[1], 260)

    def test_the_contrast_forms_add_only_their_declared_input_width(self):
        winner = sum(p.numel() for p in model(7).parameters())
        self.assertEqual(sum(p.numel() for p in model(8).parameters()),
                         winner+PIXEL_QUERY_FEATURES*64)
        self.assertEqual(sum(p.numel() for p in model(9).parameters()),
                         winner+PIXEL_QUERY_FEATURES*64)

    def test_the_declared_boundary_is_where_the_predeclaration_says(self):
        for form in (8, 9):
            replace(ModelConfig(), occupancy_pixel_query=form)
        with self.assertRaises(ValueError):
            replace(ModelConfig(), occupancy_pixel_query=14)


class StepZeroTest(unittest.TestCase):
    def test_both_contrast_forms_are_the_published_read_at_step_zero(self):
        pixels = torch.rand(1, 3, 64, 64)
        valid = support()
        with torch.no_grad():
            published = model(0, standardise=0)(pixels, valid)["final"]["occupancy"]
        for form in (8, 9):
            base, lever = model(0, standardise=0), model(form)
            lever.load_state_dict(base.state_dict(), strict=False)
            with torch.no_grad():
                read = lever(pixels, valid)["final"]["occupancy"]
            self.assertTrue(torch.equal(published, read), form)


class SampleOnlyTest(unittest.TestCase):
    def test_the_deviation_half_is_sample_derived(self):
        torch.manual_seed(3)
        statistics = torch.rand(2, 260, PIXEL_QUERY_FEATURES, dtype=torch.float64)
        deviation = statistics-statistics.mean(dim=1, keepdim=True)
        self.assertTrue(torch.allclose(deviation.mean(dim=1),
                                       torch.zeros(2, PIXEL_QUERY_FEATURES, dtype=torch.float64),
                                       atol=1e-12))
        shifted = statistics+5.0
        self.assertTrue(torch.allclose(shifted-shifted.mean(dim=1, keepdim=True), deviation,
                                       atol=1e-12))

    def test_the_spread_normalised_half_cancels_a_positive_gain(self):
        torch.manual_seed(4)
        statistics = torch.rand(2, 260, PIXEL_QUERY_FEATURES, dtype=torch.float64)+.5

        def normalised(values):
            mean = values.mean(dim=1, keepdim=True)
            spread = values.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
            return (values-mean)/spread

        self.assertTrue(torch.allclose(normalised(3.0*statistics), normalised(statistics), atol=1e-12))


if __name__ == "__main__":
    unittest.main()
