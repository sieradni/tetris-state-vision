"""Controls for the M37 fused coarse-channel read (form 11).

`docs/M37_STENCIL_FUSION.md` declares the fusion as the `30` pixel statistics plus a fixed ten-number
summary of the coarse `stencil_contrast` taps, `40` inputs through the unchanged hidden projection. These
tests bind the declaration: the width is `40` and not some earlier document's number, the added parameters
are exactly the hidden input width, step zero is the published read, and the coarse half is invariant to a
board-wide gain the way the declared transform is.
"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from tetris_vision.model import (ModelConfig, TetrisStateNet, PIXEL_QUERY_FEATURES,  # noqa: E402
                                 PIXEL_QUERY_STENCIL_FEATURES, stencil_contrast)


def support():
    valid = torch.zeros(1, 1, 64, 64, dtype=torch.bool)
    valid[:, :, 8:56, 6:58] = True
    return valid


def model(form, standardise=1):
    torch.manual_seed(13)
    return TetrisStateNet(replace(ModelConfig.pilot(), occupancy_pixel_query=form,
                                  occupancy_read_standardise=standardise))


class DeclaredWidthTest(unittest.TestCase):
    def test_the_declared_widths_are_thirty_and_ten(self):
        self.assertEqual(PIXEL_QUERY_FEATURES, 30)
        self.assertEqual(PIXEL_QUERY_STENCIL_FEATURES, 10)
        hidden = model(11).occupancy_pixel_query[0]
        self.assertEqual(hidden.in_features, PIXEL_QUERY_FEATURES+PIXEL_QUERY_STENCIL_FEATURES)

    def test_the_form_adds_only_the_hidden_input_width(self):
        """Against the 30-statistic form the fusion differs by exactly its ten extra inputs."""
        sampled = sum(p.numel() for p in model(3).parameters())
        fusion = sum(p.numel() for p in model(11).parameters())
        self.assertEqual(fusion-sampled, PIXEL_QUERY_STENCIL_FEATURES*64)

    def test_the_declared_boundary_is_where_the_predeclaration_says(self):
        for form in (11, 12, 13, 14):
            replace(ModelConfig(), occupancy_pixel_query=form)
        with self.assertRaises(ValueError):
            replace(ModelConfig(), occupancy_pixel_query=15)

    def test_the_capacity_arm_doubles_only_the_hidden_width(self):
        fused, wide = model(11).occupancy_pixel_query, model(12).occupancy_pixel_query
        self.assertEqual(fused[0].in_features, wide[0].in_features)
        self.assertEqual(wide[0].out_features, 128)
        self.assertEqual(fused[-1].out_features, wide[-1].out_features)
        growth = (sum(p.numel() for p in model(12).parameters())
                  - sum(p.numel() for p in model(11).parameters()))
        inputs = PIXEL_QUERY_FEATURES+PIXEL_QUERY_STENCIL_FEATURES
        self.assertEqual(growth, inputs*64+64+64*64)


class StepZeroTest(unittest.TestCase):
    def test_the_fused_form_is_the_published_read_at_step_zero(self):
        pixels = torch.rand(1, 3, 64, 64)
        valid = support()
        with torch.no_grad():
            published = model(0, standardise=0)(pixels, valid)["final"]["occupancy"]
        base, lever = model(0, standardise=0), model(11)
        lever.load_state_dict(base.state_dict(), strict=False)
        with torch.no_grad():
            read = lever(pixels, valid)["final"]["occupancy"]
        self.assertTrue(torch.equal(published, read))


class CoarseHalfTest(unittest.TestCase):
    def test_the_coarse_summary_cancels_a_board_wide_gain(self):
        torch.manual_seed(5)
        grid = torch.rand(2, 8, 8, 8)+.5
        self.assertTrue(torch.allclose(stencil_contrast(3.0*grid), stencil_contrast(grid), atol=1e-5))

    def test_the_summary_is_five_means_and_five_spreads(self):
        torch.manual_seed(6)
        pool = torch.nn.functional.adaptive_avg_pool2d(torch.rand(1, 4, 16, 16), (6, 12))
        grid = stencil_contrast(pool)
        taps = (grid[:, :, 1:-1, 1:-1], grid[:, :, :-2, 1:-1], grid[:, :, 2:, 1:-1],
                grid[:, :, 1:-1, :-2], grid[:, :, 1:-1, 2:])
        summary = torch.stack([tap.mean(dim=1) for tap in taps] +
                              [tap.std(dim=1, unbiased=False) for tap in taps], dim=-1)
        self.assertEqual(tuple(summary.shape), (1, 4, 10, PIXEL_QUERY_STENCIL_FEATURES))


class SourceScanTest(unittest.TestCase):
    """Standing instrument for the leakage guard's slice, so a new form fails in its own file.

    The guard in `test_span_level`, `test_tap_span` and `test_two_path` scans the module source from the
    first `forward(self, pixels, valid)` to the end of the module. Twice now a new arm's comment has been
    written as if that slice were only the model's own forward, and the guard has - correctly - refused it.
    This test says which word and where, before the suite does.
    """

    def test_no_guarded_word_enters_the_reads_source_slice(self):
        import inspect
        from tetris_vision import model as module
        source = inspect.getsource(module)
        body = source[source.index("def forward(self, pixels, valid):"):]
        for token in ("label", "target", "occupied", "prevalence", "thresholds"):
            self.assertNotIn(token, body.lower(),
                             f"'{token}' sits in the slice the leakage guard scans; word it otherwise")


if __name__ == "__main__":
    unittest.main()
