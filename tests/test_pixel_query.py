"""Mechanical controls for the M28 per-cell pixel-statistics read-out.

`docs/M28_PIXEL_QUERY.md` declares the lever: deterministic colour/contrast statistics
computed at the model's own predicted lattice and injected through a zero-initialized
projection. These tests bind the three properties the arm's whole interpretation rests on -
step zero is bit-identical to the published function, the features are a function of the
raster and the lattice only, and the head is additive and live once it moves.
"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from tetris_vision.model import (PIXEL_QUERY_FEATURES, ModelConfig,  # noqa: E402
                                 TetrisStateNet, cell_pixel_statistics)

SIDE = 64
COORDINATES = torch.stack(torch.meshgrid(torch.arange(10), torch.arange(20), indexing="ij"), -1) \
                     .reshape(-1, 2).flip(-1).to(torch.float32)+.5
LATTICE = torch.tensor([[[.05, .05], [.04, 0.], [0., .03]]], dtype=torch.float32)
OFFSETS = (torch.stack(torch.meshgrid(torch.tensor([-.3, 0., .3]), torch.tensor([-.3, 0., .3]),
                                      indexing="ij"), -1).reshape(1, 1, 9, 2)).to(torch.float32)


def model(flag):
    torch.manual_seed(11)
    return TetrisStateNet(replace(ModelConfig.pilot(), occupancy_pixel_query=flag))


def raster(value=.5):
    return torch.full((1, 3, SIDE, SIDE), value)


class ZeroInitTest(unittest.TestCase):
    def test_step_zero_is_bit_identical_to_the_published_function(self):
        base, lever = model(0), model(1)
        lever.load_state_dict(base.state_dict(), strict=False)
        pixels = torch.rand(1, 3, SIDE, SIDE)
        valid = torch.ones(1, 1, SIDE, SIDE, dtype=torch.bool)
        with torch.no_grad():
            published = base(pixels, valid)["final"]["occupancy"]
            arm = lever(pixels, valid)["final"]["occupancy"]
        self.assertTrue(torch.equal(published, arm))
        self.assertEqual(float(lever.occupancy_pixel_query.weight.abs().sum()), 0.0)
        self.assertEqual(float(lever.occupancy_pixel_query.bias.abs().sum()), 0.0)

    def test_the_projection_is_the_only_tensor_the_lever_adds(self):
        base, lever = model(0), model(1)
        added = set(lever.state_dict())-set(base.state_dict())
        self.assertEqual(added, {"occupancy_pixel_query.weight", "occupancy_pixel_query.bias"})
        self.assertEqual(lever.occupancy_pixel_query.weight.shape,
                         (ModelConfig.pilot().dimension, PIXEL_QUERY_FEATURES))

    def test_a_moved_projection_changes_the_read(self):
        base, lever = model(0), model(1)
        lever.load_state_dict(base.state_dict(), strict=False)
        torch.manual_seed(3)
        with torch.no_grad():
            lever.occupancy_pixel_query.weight.normal_(0., .05)
        pixels = torch.rand(1, 3, SIDE, SIDE)
        valid = torch.ones(1, 1, SIDE, SIDE, dtype=torch.bool)
        with torch.no_grad():
            published = base(pixels, valid)["final"]["occupancy"]
            arm = lever(pixels, valid)["final"]["occupancy"]
        self.assertFalse(torch.equal(published, arm))

    def test_the_configuration_refuses_an_undeclared_value(self):
        with self.assertRaises(ValueError):
            replace(ModelConfig(), occupancy_pixel_query=2)


class StatisticsTest(unittest.TestCase):
    def test_the_feature_matrix_has_the_declared_shape(self):
        statistics = cell_pixel_statistics(raster(), LATTICE, COORDINATES, OFFSETS)
        self.assertEqual(statistics.shape, (1, 200, PIXEL_QUERY_FEATURES))
        self.assertTrue(torch.isfinite(statistics).all())

    def test_the_statistics_are_deterministic(self):
        pixels = torch.rand(1, 3, SIDE, SIDE)
        first = cell_pixel_statistics(pixels, LATTICE, COORDINATES, OFFSETS)
        second = cell_pixel_statistics(pixels, LATTICE, COORDINATES, OFFSETS)
        self.assertTrue(torch.equal(first, second))

    def test_a_uniform_raster_has_no_contrast(self):
        statistics = cell_pixel_statistics(raster(.5), LATTICE, COORDINATES, OFFSETS)
        self.assertTrue(torch.allclose(statistics[..., :27], torch.full((1, 200, 27), .5,
                                                                        dtype=statistics.dtype),
                                       atol=1e-6))
        self.assertLess(float(statistics[..., 27:].abs().max()), 1e-6)

    def test_a_local_change_moves_the_cell_it_touches(self):
        pixels = raster(.5)
        plain = cell_pixel_statistics(pixels, LATTICE, COORDINATES, OFFSETS)
        changed = raster(.5)
        changed[0, :, 8, 12] = 1.0
        moved = cell_pixel_statistics(changed, LATTICE, COORDINATES, OFFSETS)
        difference = (moved-plain).abs().amax(-1)[0]
        self.assertGreater(float(difference.max()), 0.0)
        self.assertGreater(int((difference > 0).sum()), 0)
        self.assertLess(int((difference > 0).sum()), 200)

    def test_the_statistics_read_the_raster_not_the_model(self):
        base, lever = model(0), model(1)
        pixels = torch.rand(1, 3, SIDE, SIDE)
        first = cell_pixel_statistics(pixels, LATTICE, COORDINATES, OFFSETS)
        second = cell_pixel_statistics(pixels, LATTICE, COORDINATES, OFFSETS)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(hasattr(base, "occupancy_pixel_query") and
                         base.occupancy_pixel_query is not None)


if __name__ == "__main__":
    unittest.main()
