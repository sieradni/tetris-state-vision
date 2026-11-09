"""The declared occupancy representation lever: identity at zero, additive only."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

import torch
from tetris_vision.losses import require_occupancy_context
from tetris_vision.model import ModelConfig, TetrisStateNet

from record_stream_run import OWNED_RUN_PATTERN


def pilot(**overrides):
    config = ModelConfig.pilot()
    return ModelConfig(**{**config.__dict__, **overrides}) if overrides else config


def forward(config, seed=20261026, size=32):
    torch.manual_seed(seed)
    model = TetrisStateNet(config).eval()
    generator = torch.Generator().manual_seed(seed)
    pixels = torch.rand(1, 3, size, size, generator=generator)
    valid = torch.ones(1, 1, size, size, dtype=torch.bool)
    with torch.no_grad():
        return model, model(pixels, valid)["final"]["occupancy"]


class OccupancyContextTests(unittest.TestCase):
    def test_the_default_is_the_published_configuration(self):
        self.assertEqual(ModelConfig().occupancy_context, 0)
        self.assertEqual(ModelConfig.pilot().occupancy_context, 0)

    def test_an_undeclared_value_is_refused(self):
        for value in (2, -1, True, 1.0):
            with self.assertRaises(ValueError):
                ModelConfig(occupancy_context=value)

    def test_flag_zero_adds_no_parameters_and_no_tensor_names(self):
        off, _ = forward(pilot())
        reference = [name for name in off.state_dict() if not name.startswith("occupancy_neighborhood.")]
        self.assertIsNone(off.occupancy_neighborhood)
        self.assertEqual(reference, list(off.state_dict()))

    def test_the_lever_is_exactly_the_published_function_at_step_zero(self):
        _, plain = forward(pilot())
        _, levered = forward(pilot(occupancy_context=1))
        self.assertTrue(torch.equal(plain, levered))

    def test_the_lever_adds_only_the_neighborhood_block(self):
        off, _ = forward(pilot())
        on, _ = forward(pilot(occupancy_context=1))
        added = set(on.state_dict())-set(off.state_dict())
        self.assertTrue(added)
        self.assertTrue(all(name.startswith("occupancy_neighborhood.") for name in added))
        self.assertFalse(set(off.state_dict())-set(on.state_dict()))
        self.assertIsNotNone(on.occupancy_neighborhood)

    def test_every_shared_tensor_is_byte_identical(self):
        off, _ = forward(pilot())
        on, _ = forward(pilot(occupancy_context=1))
        for name, value in off.state_dict().items():
            self.assertTrue(torch.equal(value, on.state_dict()[name]), name)

    def test_a_trained_neighborhood_changes_the_decision(self):
        _, plain = forward(pilot())
        model, untrained = forward(pilot(occupancy_context=1))
        self.assertTrue(torch.equal(plain, untrained))
        with torch.no_grad():
            model.occupancy_neighborhood[-1].weight.fill_(1.0)
            generator = torch.Generator().manual_seed(20261026)
            pixels = torch.rand(1, 3, 32, 32, generator=generator)
            valid = torch.ones(1, 1, 32, 32, dtype=torch.bool)
            changed = model(pixels, valid)["final"]["occupancy"]
        self.assertFalse(torch.equal(plain, changed))

    def test_the_admission_join_accepts_only_a_matching_declaration(self):
        summary = {"arguments": {"occupancy_context": 1}, "recovery_contract": {"occupancy_context": 1}}
        self.assertEqual(require_occupancy_context(1, summary), None)
        require_occupancy_context(0, {"arguments": {}, "recovery_contract": {}})
        for context, record in ((1, {"arguments": {"occupancy_context": 0}, "recovery_contract": {"occupancy_context": 1}}),
                                (1, {"arguments": {"occupancy_context": 1}, "recovery_contract": {}}),
                                (2, summary)):
            with self.assertRaises(ValueError):
                require_occupancy_context(context, record)

    def test_the_owned_run_family_is_declared(self):
        for name in ("m3-occctx-lever-r1", "m3-occctx-control-r1",
                     "m3-occctx-lever-r1-sweep-r1", "m3-occctx-control-r1-sweep-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        self.assertFalse(OWNED_RUN_PATTERN.fullmatch("m3-occctx-"))


if __name__ == "__main__":
    unittest.main()
