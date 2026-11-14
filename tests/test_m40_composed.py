"""Controls for the M40 composed-alignment read (form 14).

`docs/M40_COMPOSED_ALIGNMENT.md` declares the sampled geometry as `(1-w)*predicted + w*aligned` with
`w = 0.5` fixed, the identity on a full support and neither endpoint on a padded one, and records that the
outright alignment of the predicted box onto the support box is provably M34's lattice. These tests pin
exactly that, plus the form's declared boundary, its unchanged `30` inputs and the zero new parameters.
"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from tetris_vision.model import (ModelConfig, TetrisStateNet, PIXEL_COMPOSED_WEIGHT,  # noqa: E402
                                 pixel_composed_lattice, pixel_fixed_lattice, pixel_support_lattice)


def mask(top=8, bottom=56, left=6, right=58, height=64, width=64):
    valid = torch.zeros(1, 1, height, width, dtype=torch.bool)
    valid[:, :, top:bottom, left:right] = True
    return valid


def model(form, standardise=1):
    torch.manual_seed(19)
    return TetrisStateNet(replace(ModelConfig.pilot(), occupancy_pixel_query=form,
                                  occupancy_read_standardise=standardise))


class GeometryTest(unittest.TestCase):
    def test_the_declared_weight_is_a_fixed_half(self):
        self.assertEqual(PIXEL_COMPOSED_WEIGHT, 0.5)

    def test_it_is_the_identity_when_the_support_fills_the_capture(self):
        predicted = pixel_fixed_lattice(26, 10, "cpu", torch.float64)
        composed = pixel_composed_lattice(predicted, torch.ones(1, 1, 64, 64, dtype=torch.bool),
                                          26, 10, "cpu", torch.float64)
        self.assertTrue(torch.allclose(composed, predicted, atol=1e-12))

    def test_the_outright_alignment_would_be_the_support_lattice(self):
        """The defect M40 records: full alignment collapses onto form 7, so the arm is a blend instead."""
        valid = mask()
        predicted = pixel_fixed_lattice(26, 10, "cpu", torch.float64)
        support = pixel_support_lattice(valid, 26, 10, "cpu", torch.float64)
        full = pixel_composed_lattice(predicted, valid, 26, 10, "cpu", torch.float64, weight=1.0)[0]
        self.assertTrue(torch.allclose(full, support, atol=1e-12))

    def test_the_declared_blend_is_neither_endpoint(self):
        valid = mask()
        predicted = pixel_fixed_lattice(26, 10, "cpu", torch.float64)
        support = pixel_support_lattice(valid, 26, 10, "cpu", torch.float64)
        blended = pixel_composed_lattice(predicted, valid, 26, 10, "cpu", torch.float64)[0]
        self.assertFalse(torch.allclose(blended, support, atol=1e-9))
        self.assertFalse(torch.allclose(blended, predicted[0], atol=1e-9))
        middle = 0.5*(predicted[0]+support)
        self.assertTrue(torch.allclose(blended, middle, atol=1e-12))

    def test_an_empty_mask_is_refused(self):
        with self.assertRaises(ValueError):
            pixel_composed_lattice(pixel_fixed_lattice(26, 10, "cpu", torch.float64),
                                   torch.zeros(1, 1, 64, 64, dtype=torch.bool),
                                   26, 10, "cpu", torch.float64)


class FormTest(unittest.TestCase):
    def test_the_form_adds_no_parameter(self):
        self.assertEqual(sum(p.numel() for p in model(14).parameters()),
                         sum(p.numel() for p in model(3).parameters()))

    def test_the_form_reads_the_unchanged_thirty_inputs(self):
        self.assertEqual(model(14).occupancy_pixel_query[0].in_features, 30)

    def test_the_declared_boundary_is_where_the_predeclaration_says(self):
        replace(ModelConfig(), occupancy_pixel_query=14)
        with self.assertRaises(ValueError):
            replace(ModelConfig(), occupancy_pixel_query=15)

    def test_step_zero_is_the_published_read(self):
        pixels = torch.rand(1, 3, 64, 64)
        valid = mask()
        with torch.no_grad():
            published = model(0, standardise=0)(pixels, valid)["final"]["occupancy"]
        base, lever = model(0, standardise=0), model(14)
        lever.load_state_dict(base.state_dict(), strict=False)
        with torch.no_grad():
            read = lever(pixels, valid)["final"]["occupancy"]
        self.assertTrue(torch.equal(published, read))


if __name__ == "__main__":
    unittest.main()
