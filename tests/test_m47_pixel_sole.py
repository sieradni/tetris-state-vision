"""Mechanical controls for the M47 pixel-sole occupancy read.

The declared change is a routing restriction with no new tensor: the occupancy head's input becomes the raster
projection itself, so no decoder-latent path reaches the decision. These tests pin the three things the declaration
promises - the parameter delta is exactly zero, the arm is not secretly the published function, and the head's input
really is the raster projection (captured from both layers in one forward pass, so the claim is structural rather
than a comment).
"""
import unittest

import torch

from tetris_vision.model import ModelConfig, TetrisStateNet

WINNER = dict(dimension=192, blocks=2, occupancy_pixel_query=3, occupancy_read_standardise=1)


def boards(count=2, height=64, width=128):
    torch.manual_seed(9)
    return torch.rand(count, 3, height, width), torch.ones(count, 1, height, width, dtype=torch.bool)


class DeclaredChangeTest(unittest.TestCase):
    def test_the_pixel_sole_read_adds_no_parameter(self):
        control = TetrisStateNet(ModelConfig(**WINNER))
        arm = TetrisStateNet(ModelConfig(**WINNER, occupancy_pixel_sole=1))
        self.assertEqual(sum(p.numel() for p in arm.parameters()), sum(p.numel() for p in control.parameters()))
        self.assertEqual(sorted(arm.state_dict()), sorted(control.state_dict()))
        for key in control.state_dict():
            self.assertEqual(arm.state_dict()[key].shape, control.state_dict()[key].shape, key)

    def test_the_arm_is_not_secretly_the_published_function(self):
        pixels, mask = boards()
        control = TetrisStateNet(ModelConfig(**WINNER))
        arm = TetrisStateNet(ModelConfig(**WINNER, occupancy_pixel_sole=1))
        arm.load_state_dict(control.state_dict())
        control.eval(), arm.eval()
        with torch.no_grad():
            first = control(pixels, mask)["stages"][0]["occupancy"]
            second = arm(pixels, mask)["stages"][0]["occupancy"]
        self.assertFalse(torch.allclose(first, second))

    def test_the_readout_is_live_on_the_raster(self):
        # the declared projection's last layer is zero-initialised, so a freshly built arm reads a constant until
        # a trained projection exists - which is exactly what the source checkpoint supplies. The test gives the
        # projection a declared non-zero value and then requires the raster to reach the decision.
        pixels, mask = boards()
        arm = TetrisStateNet(ModelConfig(**WINNER, occupancy_pixel_sole=1))
        with torch.no_grad():
            torch.manual_seed(3)
            arm.occupancy_pixel_query[-1].weight.normal_(0, 0.02)
        arm.eval()
        with torch.no_grad():
            base = arm(pixels, mask)["stages"][0]["occupancy"]
            moved = arm(pixels.flip(-1), mask)["stages"][0]["occupancy"]
        self.assertFalse(torch.allclose(base, moved))

    def test_the_head_input_is_the_raster_projection_and_nothing_else(self):
        # both hooks fire in one forward pass: the occupancy head's input must equal the raster projection's
        # output exactly, which is what "no latent path reaches the decision" means operationally.
        pixels, mask = boards()
        arm = TetrisStateNet(ModelConfig(**WINNER, occupancy_pixel_sole=1))
        with torch.no_grad():
            torch.manual_seed(3)
            arm.occupancy_pixel_query[-1].weight.normal_(0, 0.02)
        arm.eval()
        captured = {}
        arm.occupancy_pixel_query.register_forward_hook(
            lambda module, args, output: captured.__setitem__("projected", output.detach().clone()))
        arm.occupancy.register_forward_pre_hook(
            lambda module, args: captured.__setitem__("head", args[0].detach().clone()))
        with torch.no_grad():
            arm(pixels, mask)
        self.assertIn("projected", captured)
        self.assertIn("head", captured)
        self.assertTrue(torch.equal(captured["head"], captured["projected"]))

    def test_a_control_forward_feeds_the_head_more_than_the_projection(self):
        # the same probe on the control shows the latent path present, so the test above is not vacuous
        pixels, mask = boards()
        control = TetrisStateNet(ModelConfig(**WINNER))
        with torch.no_grad():
            torch.manual_seed(3)
            control.occupancy_pixel_query[-1].weight.normal_(0, 0.02)
        control.eval()
        captured = {}
        control.occupancy_pixel_query.register_forward_hook(
            lambda module, args, output: captured.__setitem__("projected", output.detach().clone()))
        control.occupancy.register_forward_pre_hook(
            lambda module, args: captured.__setitem__("head", args[0].detach().clone()))
        with torch.no_grad():
            control(pixels, mask)
        self.assertFalse(torch.equal(captured["head"], captured["projected"]))


class ValidationTest(unittest.TestCase):
    def test_the_flag_is_validated(self):
        ModelConfig(**dict(WINNER, occupancy_pixel_sole=1))
        for value in (2, -1, True, 0.5):
            with self.assertRaises(ValueError):
                ModelConfig(**dict(WINNER, occupancy_pixel_sole=value))

    def test_the_sole_read_needs_the_declared_pixel_read(self):
        with self.assertRaises(ValueError):
            ModelConfig(dimension=192, blocks=2, occupancy_pixel_sole=1)

    def test_no_additive_raster_term_runs_with_the_sole_read(self):
        for extra in ("occupancy_pixel_contrast", "occupancy_pixel_edge", "occupancy_pixel_edge_gate"):
            with self.assertRaises(ValueError):
                ModelConfig(**dict(WINNER, occupancy_pixel_sole=1, **{extra: 1}))


if __name__ == "__main__":
    unittest.main()
