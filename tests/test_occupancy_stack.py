"""The declared stacking lemma: the two read-out residuals add, they do not overwrite."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

import torch
from tetris_vision.losses import require_occupancy_context, require_occupancy_detail
from tetris_vision.model import ModelConfig, TetrisStateNet

from record_stream_run import OWNED_RUN_PATTERN


def pilot(**overrides):
    return ModelConfig(**{**ModelConfig.pilot().__dict__, **overrides})


def inputs(seed=20261026, size=32):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, size, size, generator=generator), torch.ones(1, 1, size, size, dtype=torch.bool)


def model(config, seed=20261026):
    torch.manual_seed(seed)
    return TetrisStateNet(config).eval()


def occupancy(net):
    pixels, valid = inputs()
    with torch.no_grad():
        return net(pixels, valid)["final"]["occupancy"]


class StackingTests(unittest.TestCase):
    def test_both_taps_off_is_the_published_configuration(self):
        self.assertEqual((ModelConfig().occupancy_context, ModelConfig().occupancy_detail), (0, 0))
        self.assertEqual((ModelConfig.pilot().occupancy_context, ModelConfig.pilot().occupancy_detail), (0, 0))

    def test_the_stack_adds_no_configuration_and_no_shared_tensor(self):
        control = model(pilot(occupancy_detail=1))
        lever = model(pilot(occupancy_detail=1, occupancy_context=1))
        added = set(lever.state_dict())-set(control.state_dict())
        self.assertTrue(added and all(name.startswith("occupancy_neighborhood.") for name in added))
        self.assertFalse(set(control.state_dict())-set(lever.state_dict()))
        for name, value in control.state_dict().items():
            self.assertTrue(torch.equal(value, lever.state_dict()[name]), name)

    def test_the_stack_is_exactly_the_published_function_at_step_zero(self):
        published = occupancy(model(pilot()))
        control = occupancy(model(pilot(occupancy_detail=1)))
        lever = occupancy(model(pilot(occupancy_detail=1, occupancy_context=1)))
        self.assertTrue(torch.equal(published, control))
        self.assertTrue(torch.equal(published, lever))

    def test_the_detail_tap_is_no_longer_overwritten_by_the_context_block(self):
        detail_only = model(pilot(occupancy_detail=1))
        stack = model(pilot(occupancy_detail=1, occupancy_context=1))
        with torch.no_grad():
            detail_only.occupancy_detail_tap.weight.fill_(.5)
            stack.occupancy_detail_tap.weight.fill_(.5)
            stack.occupancy_neighborhood[-1].weight.zero_()
            stack.occupancy_neighborhood[-1].bias.zero_()
        self.assertTrue(torch.equal(occupancy(detail_only), occupancy(stack)))
        with torch.no_grad():
            stack.occupancy_neighborhood[-1].weight.normal_(0, .3)
            stack.occupancy_neighborhood[-1].bias.normal_(0, .3)
        self.assertFalse(torch.equal(occupancy(detail_only), occupancy(stack)))

    def test_the_context_residual_alone_reproduces_its_own_arm(self):
        context_only = model(pilot(occupancy_context=1))
        stack = model(pilot(occupancy_detail=1, occupancy_context=1))
        with torch.no_grad():
            # The block's first convolution is drawn from the RNG stream the absent detail tap
            # would otherwise consume, so the whole block is aligned rather than one layer.
            for name, value in context_only.occupancy_neighborhood.state_dict().items():
                stack.occupancy_neighborhood.state_dict()[name].copy_(value)
        self.assertTrue(torch.equal(occupancy(context_only), occupancy(stack)))

    def test_the_stack_differs_from_each_tap_alone(self):
        stack = model(pilot(occupancy_detail=1, occupancy_context=1))
        detail_only = model(pilot(occupancy_detail=1))
        with torch.no_grad():
            stack.occupancy_detail_tap.weight.fill_(.5)
            stack.occupancy_neighborhood[-1].weight.normal_(0, .3)
            detail_only.occupancy_detail_tap.weight.fill_(.5)
        stacked, detail = occupancy(stack), occupancy(detail_only)
        self.assertFalse(torch.equal(stacked, detail))

    def test_the_admission_joins_still_cover_both_taps(self):
        summary = {"arguments": {"occupancy_context": 1, "occupancy_detail": 1},
                   "recovery_contract": {"occupancy_context": 1, "occupancy_detail": 1}}
        self.assertIsNone(require_occupancy_context(1, summary))
        self.assertIsNone(require_occupancy_detail(1, summary))
        for require, name in ((require_occupancy_context, "occupancy_context"),
                              (require_occupancy_detail, "occupancy_detail")):
            with self.assertRaises(ValueError):
                require(0, summary)
            broken = {"arguments": dict(summary["arguments"]), "recovery_contract": {}}
            with self.assertRaises(ValueError):
                require(1, broken)

    def test_the_owned_run_family_is_declared(self):
        for name in ("m3-occstack-lever-r1", "m3-occstack-control-r1",
                     "m3-occstack-lever-r1-sweep-r1", "m3-occstack-control-r1-sweep-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        self.assertFalse(OWNED_RUN_PATTERN.fullmatch("m3-occstack-"))


if __name__ == "__main__":
    unittest.main()
