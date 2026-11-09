"""The declared 2x2 in-cell stencil: same tap, four quadrant samples, identity at zero."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

import torch
from tetris_vision.losses import require_occupancy_detail_stencil
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


class DetailStencilTests(unittest.TestCase):
    def test_the_default_is_the_published_configuration(self):
        self.assertEqual(ModelConfig().occupancy_detail_stencil, 0)
        self.assertEqual(ModelConfig.pilot().occupancy_detail_stencil, 0)

    def test_an_undeclared_value_is_refused(self):
        for value in (2, -1, True, 1.0):
            with self.assertRaises(ValueError):
                ModelConfig(occupancy_detail=1, occupancy_detail_stencil=value)

    def test_the_stencil_is_a_form_of_the_detail_tap_and_needs_it(self):
        with self.assertRaises(ValueError):
            ModelConfig(occupancy_detail=0, occupancy_detail_stencil=1)
        self.assertIsNone(forward(pilot(occupancy_detail=0))[0].occupancy_detail_tap)

    def test_the_stencil_reads_four_samples_through_the_same_projection(self):
        model, _ = forward(pilot(occupancy_detail=1, occupancy_detail_stencil=1))
        self.assertIsNotNone(model.occupancy_detail_tap)
        self.assertEqual(model.occupancy_detail_tap.in_features, 4*pilot().widths[0])
        self.assertEqual(tuple(model.detail_stencil.shape), (1, 1, 4, 2))
        self.assertEqual(sorted(map(tuple, model.detail_stencil.reshape(4, 2).tolist())),
                         sorted(map(tuple, [[-.25, -.25], [.25, -.25], [-.25, .25], [.25, .25]])))
        single, _ = forward(pilot(occupancy_detail=1))
        self.assertEqual(single.occupancy_detail_tap.in_features, pilot().widths[0])
        for model in (single, model):
            self.assertNotIn("detail_stencil", model.state_dict())

    def test_each_arm_is_exactly_the_published_function_at_step_zero(self):
        _, plain = forward(pilot())
        for stencil in (0, 1):
            _, levered = forward(pilot(occupancy_detail=1, occupancy_detail_stencil=stencil))
            self.assertTrue(torch.equal(plain, levered), stencil)

    def test_the_arms_differ_only_in_the_tap_and_share_every_other_tensor(self):
        control, _ = forward(pilot(occupancy_detail=1))
        lever, _ = forward(pilot(occupancy_detail=1, occupancy_detail_stencil=1))
        control_keys, lever_keys = set(control.state_dict()), set(lever.state_dict())
        self.assertEqual(control_keys, lever_keys)
        self.assertNotEqual(tuple(control.state_dict()["occupancy_detail_tap.weight"].shape),
                            tuple(lever.state_dict()["occupancy_detail_tap.weight"].shape))
        for name, value in control.state_dict().items():
            if name.startswith("occupancy_detail_tap."):
                continue
            self.assertTrue(torch.equal(value, lever.state_dict()[name]), name)
        self.assertEqual(tuple(lever.state_dict()["occupancy_detail_tap.weight"].shape),
                         (pilot().dimension, 4*pilot().widths[0]))
        self.assertEqual(tuple(control.state_dict()["occupancy_detail_tap.weight"].shape),
                         (pilot().dimension, pilot().widths[0]))

    def test_a_trained_stencil_changes_the_decision(self):
        _, plain = forward(pilot())
        model, untrained = forward(pilot(occupancy_detail=1, occupancy_detail_stencil=1))
        self.assertTrue(torch.equal(plain, untrained))
        with torch.no_grad():
            model.occupancy_detail_tap.weight.fill_(0.25)
        pixels, valid = inputs()
        with torch.no_grad():
            changed = model(pixels, valid)["final"]["occupancy"]
        self.assertFalse(torch.equal(plain, changed))

    def test_relabelling_the_hidden_sample_order_cannot_change_the_decision(self):
        """Paired-raster equivalence: identical pixels, permuted hidden sample order."""
        torch.manual_seed(7)
        lever = TetrisStateNet(pilot(occupancy_detail=1, occupancy_detail_stencil=1)).eval()
        with torch.no_grad():
            lever.occupancy_detail_tap.weight.normal_(0, .5)
            lever.occupancy_detail_tap.bias.normal_(0, .1)
        permuted = deepcopy(lever)
        order = torch.tensor([2, 0, 3, 1])
        width = lever.occupancy_detail_tap.in_features//4
        with torch.no_grad():
            permuted.detail_stencil.copy_(lever.detail_stencil[:, :, order])
            weight = lever.occupancy_detail_tap.weight.view(-1, 4, width)[:, order]
            permuted.occupancy_detail_tap.weight.copy_(weight.reshape(-1, 4*width))
        pixels, valid = inputs()
        with torch.no_grad():
            first = lever(pixels, valid)["final"]["occupancy"]
            second = permuted(pixels, valid)["final"]["occupancy"]
        self.assertFalse(torch.equal(lever.detail_stencil, permuted.detail_stencil))
        self.assertTrue(torch.allclose(first, second, atol=1e-5, rtol=1e-4))

    def test_the_admission_join_accepts_only_a_matching_declaration(self):
        summary = {"arguments": {"occupancy_detail_stencil": 1},
                   "recovery_contract": {"occupancy_detail_stencil": 1}}
        self.assertIsNone(require_occupancy_detail_stencil(1, summary))
        require_occupancy_detail_stencil(0, {"arguments": {}, "recovery_contract": {}})
        for stencil, record in ((1, {"arguments": {"occupancy_detail_stencil": 0},
                                     "recovery_contract": {"occupancy_detail_stencil": 1}}),
                                (1, {"arguments": {"occupancy_detail_stencil": 1},
                                     "recovery_contract": {}}),
                                (2, summary), (True, {"arguments": {"occupancy_detail_stencil": 1},
                                                      "recovery_contract": {"occupancy_detail_stencil": 1}})):
            with self.assertRaises(ValueError):
                require_occupancy_detail_stencil(stencil, record)

    def test_the_owned_run_family_is_declared(self):
        for name in ("m3-occdstencil-lever-r1", "m3-occdstencil-control-r1",
                     "m3-occdstencil-lever-r1-sweep-r1", "m3-occdstencil-control-r1-sweep-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        self.assertFalse(OWNED_RUN_PATTERN.fullmatch("m3-occdstencil-"))


if __name__ == "__main__":
    unittest.main()
