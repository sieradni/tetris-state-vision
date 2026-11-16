"""Mechanical controls for the M45 interior-contrast cell feature.

The declaration is narrow: the same four edge midpoints M44 used, each expressed against the cell's own interior
mean per channel, through a zero-initialised projection of `12 x dimension + dimension` parameters. These tests pin
the identity at step zero, the size of the change, and - the property the arm exists for - that a board-wide
background lift cancels in the feature while a genuine cell step does not.
"""
import unittest

import torch

from tetris_vision.model import (ModelConfig, PIXEL_QUERY_EDGE_FEATURES, TetrisStateNet,
                                 pixel_fixed_lattice, sample_features)

WINNER = dict(dimension=192, blocks=2, occupancy_pixel_query=3, occupancy_read_standardise=1)
RING = torch.tensor([[-.5, 0.], [.5, 0.], [0., -.5], [0., .5]]).reshape(1, 1, 4, 2)


def boards(count=2, height=64, width=128):
    torch.manual_seed(9)
    return torch.rand(count, 3, height, width), torch.ones(count, 1, height, width, dtype=torch.bool)


def contrast(pixels, rows=20, columns=10):
    placement = pixel_fixed_lattice(rows, columns, pixels.device, pixels.dtype).expand(
        pixels.shape[0], 3, 2)
    coordinates = torch.tensor([[0.5+column, 0.5+row] for row in range(rows) for column in range(columns)])
    interior_offsets = torch.stack(torch.meshgrid(torch.tensor([-.3, 0., .3]),
                                                  torch.tensor([-.3, 0., .3]),
                                                  indexing="ij"), -1).reshape(1, 1, 9, 2)
    interior = sample_features(pixels, placement, coordinates, interior_offsets)[0].mean(dim=2)
    ring = sample_features(pixels, placement, coordinates, RING)[0]
    return (ring-interior[:, :, None, :])


class DeclaredChangeTest(unittest.TestCase):
    def test_the_contrast_read_adds_exactly_its_own_projection(self):
        control = TetrisStateNet(ModelConfig(**WINNER))
        arm = TetrisStateNet(ModelConfig(**WINNER, occupancy_pixel_contrast=1))
        base = sum(p.numel() for p in control.parameters())
        added = sum(p.numel() for p in arm.parameters())-base
        self.assertEqual(added, PIXEL_QUERY_EDGE_FEATURES*192+192)
        self.assertEqual(added, 2496)

    def test_step_zero_is_the_published_function(self):
        pixels, mask = boards()
        control = TetrisStateNet(ModelConfig(**WINNER))
        arm = TetrisStateNet(ModelConfig(**WINNER, occupancy_pixel_contrast=1))
        arm.load_state_dict(control.state_dict(), strict=False)
        control.eval(), arm.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(control(pixels, mask)["stages"][0]["occupancy"],
                                        arm(pixels, mask)["stages"][0]["occupancy"]))

    def test_the_projection_changes_the_read_once_it_is_not_zero(self):
        pixels, mask = boards()
        control = TetrisStateNet(ModelConfig(**WINNER))
        arm = TetrisStateNet(ModelConfig(**WINNER, occupancy_pixel_contrast=1))
        arm.load_state_dict(control.state_dict(), strict=False)
        with torch.no_grad():
            arm.occupancy_pixel_contrast.weight.normal_(0, 0.1)
        control.eval(), arm.eval()
        with torch.no_grad():
            self.assertFalse(torch.allclose(control(pixels, mask)["stages"][0]["occupancy"],
                                            arm(pixels, mask)["stages"][0]["occupancy"]))

    def test_the_geometry_points_are_untouched(self):
        pixels, mask = boards()
        control = TetrisStateNet(ModelConfig(**WINNER))
        arm = TetrisStateNet(ModelConfig(**WINNER, occupancy_pixel_contrast=1))
        arm.load_state_dict(control.state_dict(), strict=False)
        control.eval(), arm.eval()
        with torch.no_grad():
            first = control(pixels, mask)["stages"][0]
            second = arm(pixels, mask)["stages"][0]
        self.assertTrue(torch.equal(first["sample_points"], second["sample_points"]))
        self.assertTrue(torch.equal(first["sample_offsets"], second["sample_offsets"]))


class CancellationTest(unittest.TestCase):
    def test_a_board_wide_lift_cancels_in_the_feature(self):
        pixels = torch.zeros(1, 3, 32, 20)
        pixels[0, :, 16:, :] = 1.0
        lifted = pixels+0.4
        # the capture's own border cells sample the zero pad, which no lift can change, so the property is
        # stated where the samples are inside the raster - the population the read is scored on.
        difference = (contrast(pixels)-contrast(lifted)).reshape(20, 10, 4, 3)[1:-1, 1:-1]
        self.assertLess(float(difference.abs().max()), 1e-5)

    def test_a_cell_step_does_not_cancel(self):
        pixels = torch.zeros(1, 3, 32, 20)
        pixels[0, :, 17:, :] = 1.0
        values = contrast(pixels)[0].mean(dim=-1).std(dim=1).reshape(20, 10)[1:-1, 1:-1]
        self.assertGreater(float(values.max()), 0.25)
        flat = torch.zeros(1, 3, 32, 20)
        flat_values = contrast(flat)[0].mean(dim=-1).std(dim=1).reshape(20, 10)[1:-1, 1:-1]
        self.assertLess(float(flat_values.max()), 1e-6)


class ValidationTest(unittest.TestCase):
    def test_the_flag_is_validated(self):
        ModelConfig(**dict(WINNER, occupancy_pixel_contrast=1))
        for value in (2, -1, True, 0.5):
            with self.assertRaises(ValueError):
                ModelConfig(**dict(WINNER, occupancy_pixel_contrast=value))

    def test_the_contrast_needs_the_declared_pixel_read(self):
        with self.assertRaises(ValueError):
            ModelConfig(dimension=192, blocks=2, occupancy_pixel_contrast=1)


if __name__ == "__main__":
    unittest.main()
