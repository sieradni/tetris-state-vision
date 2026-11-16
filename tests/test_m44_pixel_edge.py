"""Mechanical controls for the M44 pixel edge ring.

The declaration is narrow: four samples on the cell's own edge midpoints, half a cell out, through a
zero-initialised projection of exactly `12 x dimension + dimension` parameters. These tests pin the identity at
step zero, the size of the change, that the samples sit where the declaration says, and that the feature actually
carries a boundary signal rather than another interior summary.
"""
import unittest

import torch

from tetris_vision.model import (ModelConfig, PIXEL_QUERY_EDGE_FEATURES, PIXEL_QUERY_EDGE_SAMPLES,
                                 TetrisStateNet, pixel_fixed_lattice, sample_features)

WINNER = dict(dimension=192, blocks=2, occupancy_pixel_query=3, occupancy_read_standardise=1)


def boards(count=2, height=64, width=128):
    torch.manual_seed(7)
    return torch.rand(count, 3, height, width), torch.ones(count, 1, height, width, dtype=torch.bool)


class DeclaredChangeTest(unittest.TestCase):
    def test_the_ring_adds_exactly_its_own_projection(self):
        control = TetrisStateNet(ModelConfig(**WINNER))
        arm = TetrisStateNet(ModelConfig(**WINNER, occupancy_pixel_edge=1))
        base = sum(p.numel() for p in control.parameters())
        added = sum(p.numel() for p in arm.parameters())-base
        self.assertEqual(added, PIXEL_QUERY_EDGE_FEATURES*192+192)
        self.assertEqual(added, 2496)

    def test_step_zero_is_the_published_function(self):
        pixels, mask = boards()
        control = TetrisStateNet(ModelConfig(**WINNER))
        arm = TetrisStateNet(ModelConfig(**WINNER, occupancy_pixel_edge=1))
        arm.load_state_dict(control.state_dict(), strict=False)
        control.eval(), arm.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(control(pixels, mask)["stages"][0]["occupancy"],
                                        arm(pixels, mask)["stages"][0]["occupancy"]))

    def test_the_learned_projection_moves_the_read_once_it_is_not_zero(self):
        pixels, mask = boards()
        control = TetrisStateNet(ModelConfig(**WINNER))
        arm = TetrisStateNet(ModelConfig(**WINNER, occupancy_pixel_edge=1))
        arm.load_state_dict(control.state_dict(), strict=False)
        with torch.no_grad():
            arm.occupancy_pixel_edge.weight.normal_(0, 0.1)
        control.eval(), arm.eval()
        with torch.no_grad():
            self.assertFalse(torch.allclose(control(pixels, mask)["stages"][0]["occupancy"],
                                            arm(pixels, mask)["stages"][0]["occupancy"]))

    def test_the_sampled_cell_points_are_identical(self):
        pixels, mask = boards()
        control = TetrisStateNet(ModelConfig(**WINNER))
        arm = TetrisStateNet(ModelConfig(**WINNER, occupancy_pixel_edge=1))
        arm.load_state_dict(control.state_dict(), strict=False)
        control.eval(), arm.eval()
        with torch.no_grad():
            first = control(pixels, mask)["stages"][0]
            second = arm(pixels, mask)["stages"][0]
        self.assertTrue(torch.equal(first["sample_points"], second["sample_points"]))
        self.assertTrue(torch.equal(first["sample_offsets"], second["sample_offsets"]))

    def test_the_offsets_are_the_four_edge_midpoints_half_a_cell_out(self):
        arm = TetrisStateNet(ModelConfig(**WINNER, occupancy_pixel_edge=1))
        offsets = arm.pixel_query_edge_offsets.reshape(PIXEL_QUERY_EDGE_SAMPLES, 2)
        self.assertTrue(torch.equal(offsets.abs().sum(1), torch.full((4,), 0.5)))
        self.assertEqual(sorted(offsets.abs().tolist()),
                         [[0.0, 0.5], [0.0, 0.5], [0.5, 0.0], [0.5, 0.0]])


class BoundarySignalTest(unittest.TestCase):
    def raster(self):
        """Two boards: the left board's cell row is filled to its top edge, the right board's continues above."""
        pixels = torch.zeros(2, 3, 32, 20)
        pixels[0, :, 17:, :] = 1.0          # fill stops at a row that cuts through the cell grid
        pixels[1, :, :, :] = 1.0            # fill continues everywhere
        return pixels

    def test_the_edge_samples_separate_a_stopping_fill_from_a_continuing_one(self):
        pixels = self.raster()
        placement = pixel_fixed_lattice(20, 10, pixels.device, pixels.dtype).expand(2, 3, 2)
        coordinates = torch.tensor([[0.5+column, 0.5+row] for row in range(20) for column in range(10)])
        offsets = torch.tensor([[-.5, 0.], [.5, 0.], [0., -.5], [0., .5]]).reshape(1, 1, 4, 2)
        edges = sample_features(pixels, placement, coordinates, offsets)[0]
        self.assertEqual(tuple(edges.shape), (2, 200, PIXEL_QUERY_EDGE_SAMPLES, 3))
        # where the fill stops, cells whose own edges straddle the boundary disagree; where it continues,
        # only the capture's own border cells disagree, because their outer edge samples the zero pad. The
        # interior comparison is the signal the ring hands the read.
        stopping = edges[0].mean(dim=-1).std(dim=1).reshape(20, 10)[1:-1, 1:-1]
        continuing = edges[1].mean(dim=-1).std(dim=1).reshape(20, 10)[1:-1, 1:-1]
        self.assertGreater(float(stopping.max()), 0.25)
        self.assertGreater(int((stopping > 0.25).sum()), 0)
        self.assertEqual(int((continuing > 1e-6).sum()), 0)


class ValidationTest(unittest.TestCase):
    def test_the_flag_is_validated(self):
        ModelConfig(**dict(WINNER, occupancy_pixel_edge=1))
        for value in (2, -1, True, 0.5):
            with self.assertRaises(ValueError):
                ModelConfig(**dict(WINNER, occupancy_pixel_edge=value))

    def test_the_ring_needs_the_declared_pixel_read(self):
        with self.assertRaises(ValueError):
            ModelConfig(dimension=192, blocks=2, occupancy_pixel_edge=1)


if __name__ == "__main__":
    unittest.main()
