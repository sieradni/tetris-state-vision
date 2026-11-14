"""Controls for the M41 provenance-isolating read (form 15).

`docs/M41_SUPPORT_CONTRAST.md` declares form 15 as M35's `60` contrast features sampled at the support
lattice, with the head width, decision rule and every other recorded setting held fixed, so that against M35
the only varied fact is where the geometry comes from. These tests pin the inputs, the parameter identity
against form 8, the geometry actually used, the absence of the monotone rule, the declared boundary and step
zero.
"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch
import tetris_vision.model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from tetris_vision.model import ModelConfig, TetrisStateNet  # noqa: E402


def trained(form):
    torch.manual_seed(23)
    return TetrisStateNet(replace(ModelConfig.pilot(), dimension=192, blocks=6,
                                  occupancy_pixel_query=form, occupancy_read_standardise=1))


def mask(top=8, bottom=56, left=6, right=58):
    valid = torch.zeros(1, 1, 64, 64, dtype=torch.bool)
    valid[:, :, top:bottom, left:right] = True
    return valid


class FormTest(unittest.TestCase):
    def test_the_arm_reads_m35s_sixty_contrast_inputs(self):
        self.assertEqual(trained(15).occupancy_pixel_query[0].in_features, 60)

    def test_the_arm_matches_m35_parameter_for_parameter(self):
        # the same config, the same features, the same head: only the sampled geometry differs. the trained
        # checkpoint's absolute count (8,025,482) is asserted in the M41 receipt test, against the arm's own manifest.
        self.assertEqual(sum(p.numel() for p in trained(15).parameters()),
                         sum(p.numel() for p in trained(8).parameters()))

    def test_the_arm_samples_the_support_lattice_and_form_five_does_not(self):
        pixels = torch.rand(1, 3, 64, 64)
        calls = {}
        real = tetris_vision.model.pixel_support_lattice
        for form in (8, 15):
            with mock.patch.object(tetris_vision.model, "pixel_support_lattice",
                                   side_effect=real) as spy:
                trained(form)(pixels, mask())
                calls[form] = spy.call_count
        self.assertGreaterEqual(calls[15], 1)
        self.assertEqual(calls[8], 0)

    def test_the_arm_carries_no_monotone_decision_rule(self):
        pixels = torch.rand(1, 3, 64, 64)
        calls = {}
        real = tetris_vision.model.stack_contiguous
        for form in (10, 15):
            with mock.patch.object(tetris_vision.model, "stack_contiguous",
                                   side_effect=real) as spy:
                trained(form)(pixels, mask())
                calls[form] = spy.call_count
        self.assertEqual(calls[15], 0)
        self.assertGreaterEqual(calls[10], 1)

    def test_the_declared_boundary_is_where_the_predeclaration_says(self):
        replace(ModelConfig(), occupancy_pixel_query=15)
        with self.assertRaises(ValueError):
            replace(ModelConfig(), occupancy_pixel_query=16)

    def test_step_zero_is_the_published_read(self):
        pixels = torch.rand(1, 3, 64, 64)
        valid = mask()
        # form 8 is the same 60-input head fed the predicted lattice; at step zero the projection contributes
        # exactly zero, so the two must be bit-identical and any difference would be a wiring defect (V-166).
        base, arm = trained(8), trained(15)
        arm.load_state_dict(base.state_dict())
        with torch.no_grad():
            published = base(pixels, valid)["final"]["occupancy"]
            read = arm(pixels, valid)["final"]["occupancy"]
        self.assertTrue(torch.equal(published, read))


if __name__ == "__main__":
    unittest.main()
