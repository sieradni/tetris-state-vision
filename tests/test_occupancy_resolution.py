"""The declared resolution lever: a stride-1 source for the detail tap, identity at zero."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

import torch
from tetris_vision.losses import require_occupancy_resolution
from tetris_vision.model import ModelConfig, TetrisStateNet

from record_stream_run import OWNED_RUN_PATTERN


def pilot(**overrides):
    return ModelConfig(**{**ModelConfig.pilot().__dict__, **overrides})


def inputs(seed=20261026, size=32):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, size, size, generator=generator), torch.ones(1, 1, size, size, dtype=torch.bool)


def forward(config, seed=20261026):
    torch.manual_seed(seed)
    model = TetrisStateNet(config).eval()
    pixels, valid = inputs()
    with torch.no_grad():
        return model, model(pixels, valid)["final"]["occupancy"]


class OccupancyResolutionTests(unittest.TestCase):
    def test_the_default_is_the_published_configuration(self):
        self.assertEqual(ModelConfig().occupancy_resolution, 0)
        self.assertEqual(ModelConfig.pilot().occupancy_resolution, 0)

    def test_an_undeclared_value_is_refused(self):
        for value in (2, -1, True, 1.0):
            with self.assertRaises(ValueError):
                ModelConfig(occupancy_detail=1, occupancy_resolution=value)

    def test_the_resolution_lever_is_a_form_of_the_detail_tap_and_needs_it(self):
        with self.assertRaises(ValueError):
            ModelConfig(occupancy_detail=0, occupancy_resolution=1)
        model, _ = forward(pilot(occupancy_detail=1))
        self.assertIsNone(model.occupancy_resolution_branch)

    def test_the_branch_is_a_stride_one_convolution_of_the_image(self):
        model, _ = forward(pilot(occupancy_detail=1, occupancy_resolution=1))
        branch = model.occupancy_resolution_branch
        self.assertEqual((branch.in_channels, branch.out_channels), (3, pilot().widths[0]))
        self.assertEqual((branch.stride, branch.padding), ((1, 1), (1, 1)))
        self.assertEqual(model.occupancy_detail_tap.in_features, pilot().widths[0])

    def test_both_arms_are_exactly_the_published_function_at_step_zero(self):
        _, plain = forward(pilot())
        _, control = forward(pilot(occupancy_detail=1))
        _, lever = forward(pilot(occupancy_detail=1, occupancy_resolution=1))
        self.assertTrue(torch.equal(plain, control))
        self.assertTrue(torch.equal(plain, lever))

    def test_the_arms_share_every_tensor_but_the_branch(self):
        torch.manual_seed(20261026)
        control = TetrisStateNet(pilot(occupancy_detail=1))
        torch.manual_seed(20261026)
        lever = TetrisStateNet(pilot(occupancy_detail=1, occupancy_resolution=1))
        control_state, lever_state = control.state_dict(), lever.state_dict()
        added = set(lever_state)-set(control_state)
        self.assertTrue(added and all(name.startswith("occupancy_resolution_branch.") for name in added))
        self.assertFalse(set(control_state)-set(lever_state))
        for name, value in control_state.items():
            self.assertTrue(torch.equal(value, lever_state[name]), name)

    def test_a_trained_branch_changes_the_decision_only_through_the_projection(self):
        model, _ = forward(pilot(occupancy_detail=1, occupancy_resolution=1))
        pixels, valid = inputs()
        with torch.no_grad():
            model.occupancy_resolution_branch.weight.normal_(0, .5)
            inert = model(pixels, valid)["final"]["occupancy"]
            model.occupancy_detail_tap.weight.fill_(.5)
            active = model(pixels, valid)["final"]["occupancy"]
        _, plain = forward(pilot())
        self.assertTrue(torch.equal(plain, inert))
        self.assertFalse(torch.equal(inert, active))

    def test_a_hidden_branch_state_cannot_change_a_zero_projection(self):
        """Paired-raster equivalence: identical pixels, different hidden branch weights, same output."""
        torch.manual_seed(11)
        first = TetrisStateNet(pilot(occupancy_detail=1, occupancy_resolution=1)).eval()
        torch.manual_seed(11)
        second = TetrisStateNet(pilot(occupancy_detail=1, occupancy_resolution=1)).eval()
        with torch.no_grad():
            first.occupancy_resolution_branch.weight.normal_(0, .9)
            second.occupancy_resolution_branch.weight.normal_(0, .1)
        pixels, valid = inputs()
        with torch.no_grad():
            a = first(pixels, valid)["final"]["occupancy"]
            b = second(pixels, valid)["final"]["occupancy"]
        self.assertFalse(torch.equal(first.occupancy_resolution_branch.weight,
                                     second.occupancy_resolution_branch.weight))
        self.assertTrue(torch.equal(a, b))

    def test_the_admission_join_accepts_only_a_matching_declaration(self):
        summary = {"arguments": {"occupancy_resolution": 1}, "recovery_contract": {"occupancy_resolution": 1}}
        self.assertIsNone(require_occupancy_resolution(1, summary))
        require_occupancy_resolution(0, {"arguments": {}, "recovery_contract": {}})
        for resolution, record in ((1, {"arguments": {"occupancy_resolution": 0},
                                        "recovery_contract": {"occupancy_resolution": 1}}),
                                   (1, {"arguments": {"occupancy_resolution": 1}, "recovery_contract": {}}),
                                   (2, summary), (True, summary)):
            with self.assertRaises(ValueError):
                require_occupancy_resolution(resolution, record)

    def test_the_owned_run_family_is_declared(self):
        for name in ("m3-occres-lever-r1", "m3-occres-control-r1",
                     "m3-occres-lever-r1-sweep-r1", "m3-occres-control-r1-sweep-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        self.assertFalse(OWNED_RUN_PATTERN.fullmatch("m3-occres-"))


if __name__ == "__main__":
    unittest.main()
