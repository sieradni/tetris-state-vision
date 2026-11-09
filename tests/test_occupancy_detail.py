"""The declared precision lever: a sharper sample, identity at zero, additive only."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

import torch
from tetris_vision.losses import require_occupancy_detail
from tetris_vision.model import ModelConfig, TetrisStateNet

from record_stream_run import OWNED_RUN_PATTERN


def pilot(**overrides):
    return ModelConfig(**{**ModelConfig.pilot().__dict__, **overrides})


def forward(config, seed=20261026, size=32):
    torch.manual_seed(seed)
    model = TetrisStateNet(config).eval()
    generator = torch.Generator().manual_seed(seed)
    pixels = torch.rand(1, 3, size, size, generator=generator)
    valid = torch.ones(1, 1, size, size, dtype=torch.bool)
    with torch.no_grad():
        return model, model(pixels, valid)["final"]["occupancy"]


class OccupancyDetailTests(unittest.TestCase):
    def test_the_default_is_the_published_configuration(self):
        self.assertEqual(ModelConfig().occupancy_detail, 0)
        self.assertEqual(ModelConfig.pilot().occupancy_detail, 0)

    def test_an_undeclared_value_is_refused(self):
        for value in (2, -1, True, 1.0):
            with self.assertRaises(ValueError):
                ModelConfig(occupancy_detail=value)

    def test_the_tap_reads_the_first_stage_features(self):
        model, _ = forward(pilot(occupancy_detail=1))
        self.assertIsNotNone(model.occupancy_detail_tap)
        self.assertEqual(model.occupancy_detail_tap.in_features, pilot().widths[0])

    def test_flag_zero_adds_no_parameters(self):
        off, _ = forward(pilot())
        self.assertIsNone(off.occupancy_detail_tap)
        self.assertFalse([name for name in off.state_dict() if name.startswith("occupancy_detail_tap.")])

    def test_the_lever_is_exactly_the_published_function_at_step_zero(self):
        _, plain = forward(pilot())
        _, levered = forward(pilot(occupancy_detail=1))
        self.assertTrue(torch.equal(plain, levered))

    def test_every_shared_tensor_is_byte_identical_and_only_the_tap_is_added(self):
        off, _ = forward(pilot())
        on, _ = forward(pilot(occupancy_detail=1))
        added = set(on.state_dict())-set(off.state_dict())
        self.assertTrue(added and all(name.startswith("occupancy_detail_tap.") for name in added))
        self.assertFalse(set(off.state_dict())-set(on.state_dict()))
        for name, value in off.state_dict().items():
            self.assertTrue(torch.equal(value, on.state_dict()[name]), name)

    def test_a_trained_tap_changes_the_decision(self):
        _, plain = forward(pilot())
        model, untrained = forward(pilot(occupancy_detail=1))
        self.assertTrue(torch.equal(plain, untrained))
        with torch.no_grad():
            model.occupancy_detail_tap.weight.fill_(0.5)
            generator = torch.Generator().manual_seed(20261026)
            pixels = torch.rand(1, 3, 32, 32, generator=generator)
            valid = torch.ones(1, 1, 32, 32, dtype=torch.bool)
            changed = model(pixels, valid)["final"]["occupancy"]
        self.assertFalse(torch.equal(plain, changed))

    def test_the_two_declared_levers_are_independent(self):
        model, _ = forward(pilot(occupancy_detail=1, occupancy_context=0))
        self.assertIsNone(model.occupancy_neighborhood)
        self.assertIsNotNone(model.occupancy_detail_tap)
        model, _ = forward(pilot(occupancy_detail=0, occupancy_context=1))
        self.assertIsNotNone(model.occupancy_neighborhood)
        self.assertIsNone(model.occupancy_detail_tap)

    def test_the_admission_join_accepts_only_a_matching_declaration(self):
        summary = {"arguments": {"occupancy_detail": 1}, "recovery_contract": {"occupancy_detail": 1}}
        self.assertEqual(require_occupancy_detail(1, summary), None)
        require_occupancy_detail(0, {"arguments": {}, "recovery_contract": {}})
        for detail, record in ((1, {"arguments": {"occupancy_detail": 0}, "recovery_contract": {"occupancy_detail": 1}}),
                               (1, {"arguments": {"occupancy_detail": 1}, "recovery_contract": {}}),
                               (2, summary)):
            with self.assertRaises(ValueError):
                require_occupancy_detail(detail, record)

    def test_the_owned_run_family_is_declared(self):
        for name in ("m3-occdetail-lever-r1", "m3-occdetail-control-r1",
                     "m3-occdetail-lever-r1-sweep-r1", "m3-occdetail-control-r1-sweep-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        self.assertFalse(OWNED_RUN_PATTERN.fullmatch("m3-occdetail-"))


if __name__ == "__main__":
    unittest.main()
