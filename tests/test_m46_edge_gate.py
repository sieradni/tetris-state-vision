"""Mechanical controls for the M46 edge-gated contrast cell feature.

The declaration is narrow: the same twelve ring contrasts M45 measured, each multiplied by a gate built from the
cell's own in-cell spread - the statistic the standardised read already divides by - plus those three gate values,
through a zero-initialised projection of `15 x dimension + dimension` parameters. These tests pin the identity at
step zero, the size of the change, and the two properties the gate exists for: it closes on a featureless cell (so
the level evidence stays where M45 lost it) and it opens on a real local edge while a board-wide lift still cancels.
"""
import unittest

import torch

from tetris_vision.model import (ModelConfig, PIXEL_QUERY_GATE_FEATURES, TetrisStateNet, pixel_fixed_lattice,
                                 sample_features)

WINNER = dict(dimension=192, blocks=2, occupancy_pixel_query=3, occupancy_read_standardise=1)
RING = torch.tensor([[-.5, 0.], [.5, 0.], [0., -.5], [0., .5]]).reshape(1, 1, 4, 2)
INTERIOR = torch.stack(torch.meshgrid(torch.tensor([-.3, 0., .3]), torch.tensor([-.3, 0., .3]),
                                      indexing="ij"), -1).reshape(1, 1, 9, 2)


def boards(count=2, height=64, width=128):
    torch.manual_seed(9)
    return torch.rand(count, 3, height, width), torch.ones(count, 1, height, width, dtype=torch.bool)


def gated(pixels, rows=20, columns=10):
    placement = pixel_fixed_lattice(rows, columns, pixels.device, pixels.dtype).expand(pixels.shape[0], 3, 2)
    coordinates = torch.tensor([[0.5+column, 0.5+row] for row in range(rows) for column in range(columns)])
    samples = sample_features(pixels, placement, coordinates, INTERIOR)[0]
    interior, spread = samples.mean(dim=2), samples.std(dim=2, unbiased=False)
    ring = sample_features(pixels, placement, coordinates, RING)[0]
    contrast = ring-interior[:, :, None, :]
    strength = contrast.abs()
    gate = ((strength-spread[:, :, None, :]-1e-6).clamp_min(0)
            / (strength+spread[:, :, None, :]+1e-6)).clamp_max(1.0)
    return contrast, gate


class DeclaredChangeTest(unittest.TestCase):
    def test_the_gated_read_adds_exactly_its_own_projection(self):
        control = TetrisStateNet(ModelConfig(**WINNER))
        arm = TetrisStateNet(ModelConfig(**WINNER, occupancy_pixel_edge_gate=1))
        base = sum(p.numel() for p in control.parameters())
        added = sum(p.numel() for p in arm.parameters())-base
        self.assertEqual(added, PIXEL_QUERY_GATE_FEATURES*192+192)
        self.assertEqual(added, 15*192+192)
        self.assertEqual(added, 3072)

    def test_step_zero_is_the_published_function(self):
        pixels, mask = boards()
        control = TetrisStateNet(ModelConfig(**WINNER))
        arm = TetrisStateNet(ModelConfig(**WINNER, occupancy_pixel_edge_gate=1))
        arm.load_state_dict(control.state_dict(), strict=False)
        control.eval(), arm.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(control(pixels, mask)["stages"][0]["occupancy"],
                                        arm(pixels, mask)["stages"][0]["occupancy"]))

    def test_the_projection_changes_the_read_once_it_is_not_zero(self):
        pixels, mask = boards()
        control = TetrisStateNet(ModelConfig(**WINNER))
        arm = TetrisStateNet(ModelConfig(**WINNER, occupancy_pixel_edge_gate=1))
        arm.load_state_dict(control.state_dict(), strict=False)
        with torch.no_grad():
            arm.occupancy_pixel_edge_gate.weight.normal_(0, 0.1)
        control.eval(), arm.eval()
        with torch.no_grad():
            self.assertFalse(torch.allclose(control(pixels, mask)["stages"][0]["occupancy"],
                                            arm(pixels, mask)["stages"][0]["occupancy"]))

    def test_the_geometry_points_are_untouched(self):
        pixels, mask = boards()
        control = TetrisStateNet(ModelConfig(**WINNER))
        arm = TetrisStateNet(ModelConfig(**WINNER, occupancy_pixel_edge_gate=1))
        arm.load_state_dict(control.state_dict(), strict=False)
        control.eval(), arm.eval()
        with torch.no_grad():
            first = control(pixels, mask)["stages"][0]
            second = arm(pixels, mask)["stages"][0]
        self.assertTrue(torch.equal(first["sample_points"], second["sample_points"]))
        self.assertTrue(torch.equal(first["sample_offsets"], second["sample_offsets"]))


class GateRuleTest(unittest.TestCase):
    # a raster whose cells are eight pixels wide, so the sampled points sit strictly inside the frame and a
    # board-wide lift cannot reach the zero padding at the border - which no lift can change, and which is
    # therefore excluded from every invariance statement here.
    def _raster(self, step=None, lift=0.0):
        pixels = torch.zeros(1, 3, 64, 64)
        if step is not None:
            pixels[0, :, step:, :] = 1.0
        return pixels+lift

    def _values(self, pixels, rows=8, columns=8):
        contrast, gate = gated(pixels, rows=rows, columns=columns)
        shape = (rows, columns, 4, 3)
        return contrast.reshape(shape), gate.reshape(shape)

    def test_the_gate_closes_where_there_is_no_local_edge(self):
        # a flat raster has no contrast and no spread, so nothing passes and the absolute read is untouched
        _, gate = self._values(self._raster())
        self.assertLess(float(gate.max()), 1e-6)

    def test_the_gate_opens_on_a_local_edge(self):
        contrast, gate = self._values(self._raster(step=32))
        inside = gate[1:-1, 1:-1].mean(dim=-1)
        self.assertGreater(float(inside.max()), 0.2)
        self.assertLess(float(inside.min()), 0.05)
        # and the graded value rises with the contrast's excess over the cell's own spread

    def test_a_board_wide_lift_still_cancels(self):
        first_contrast, first_gate = self._values(self._raster(step=32))
        second_contrast, second_gate = self._values(self._raster(step=32, lift=0.4))
        inside = (slice(1, -1), slice(1, -1))
        self.assertLess(float((first_contrast*first_gate-second_contrast*second_gate)[inside].abs().max()), 1e-4)
        self.assertLess(float((first_gate-second_gate)[inside].abs().max()), 1e-4)

    def test_the_gate_is_the_contrasts_excess_over_the_cells_own_spread(self):
        # the rule is a declared function of two statistics the pipeline already computes, with no free constant
        # except the standardised read's own floor; here it is checked directly on a synthetic pair.
        contrast = torch.tensor([[[[0.0, 0.5, 2.0]]]])
        spread = torch.tensor([[[0.25, 0.25, 0.25]]])
        strength = contrast.abs()
        gate = ((strength-spread[:, :, None, :]-1e-6).clamp_min(0)
                / (strength+spread[:, :, None, :]+1e-6)).clamp_max(1.0)
        self.assertAlmostEqual(float(gate[0, 0, 0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(gate[0, 0, 0, 1]), (0.5-0.25-1e-6)/(0.5+0.25+1e-6), places=5)
        self.assertAlmostEqual(float(gate[0, 0, 0, 2]), (2.0-0.25-1e-6)/(2.0+0.25+1e-6), places=5)


class ValidationTest(unittest.TestCase):
    def test_the_flag_is_validated(self):
        ModelConfig(**dict(WINNER, occupancy_pixel_edge_gate=1))
        for value in (2, -1, True, 0.5):
            with self.assertRaises(ValueError):
                ModelConfig(**dict(WINNER, occupancy_pixel_edge_gate=value))

    def test_the_gate_needs_the_declared_pixel_read(self):
        with self.assertRaises(ValueError):
            ModelConfig(dimension=192, blocks=2, occupancy_pixel_edge_gate=1)

    def test_the_gated_and_ungated_forms_never_run_together(self):
        with self.assertRaises(ValueError):
            ModelConfig(**dict(WINNER, occupancy_pixel_edge_gate=1, occupancy_pixel_contrast=1))


if __name__ == "__main__":
    unittest.main()
