"""Mechanical controls for the M52 lattice refinement.

M51 measured the residual as pitch: this arm localises each expected cell boundary to sub-pixel, drops missing
boundaries, fits pitch by Theil-Sen and anchors the origin to the box's own convention. These tests pin the machinery
on synthetic profiles with known pitch and origin (including sub-pixel offsets and a missing interior line), on a
synthetic capture, on the declared fallback, and on the declaration's own claims.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from lattice_fit import (CONSTANTS, MIN_LINES, MISSING_SHARE, OUTLIER_CELLS, SEARCH_SHARE,  # noqa: E402
                         fit_axis, localise, profile, refine, theil_sen)


def peaks(positions, size, height=1000.0, width=1.2):
    """An edge profile with Gaussian peaks at known sub-pixel positions."""
    profile = np.zeros(size)
    for position in positions:
        axis = np.arange(size)
        profile += height*np.exp(-0.5*((axis-position)/width)**2)
    return profile


def synthetic_grey(pitch=14.0, left=40.0, top=30.0, grid=True, missing=None, block=None):
    """A capture holding a ten-by-twenty well drawn at a known pitch and origin."""
    height, width = 460, 340
    grey = np.full((height, width), 30.0)
    for index in range(11):
        x = int(round(left+index*pitch))
        if missing is not None and index == missing:
            continue
        if grid:
            grey[int(top):int(top+20*pitch), x] = 150.0
    for index in range(21):
        y = int(round(top+index*pitch))
        if grid:
            grey[y, int(left):int(left+10*pitch)] = 150.0
    if block is not None:
        column, row = block
        grey[int(top+row*pitch)+1:int(top+(row+1)*pitch), int(left+column*pitch)+1:int(left+(column+1)*pitch)] = 200.0
    return grey


class LocalisationTest(unittest.TestCase):
    def test_a_boundary_is_localised_to_sub_pixel_accuracy(self):
        # the centroid of a three-sample window is sub-pixel but biased toward the nearest integer: the measured
        # bias on a 1.2-pixel peak is a sawtooth of at most 0.30 pixels, recorded as the estimator's limitation
        energy = peaks([40.4, 54.1, 68.2], 100)
        centre = localise(energy, 40.0, 14.0)
        self.assertAlmostEqual(centre, 40.16, delta=0.05)
        self.assertGreater(centre, 40.0)
        self.assertLess(abs(centre-40.4), 0.25)

    def test_the_localisers_bias_is_a_sawtooth_of_at_most_three_tenths_of_a_pixel(self):
        offsets = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        errors = [localise(peaks([40.0+offset], 100), 40.0, 14.0)-(40.0+offset) for offset in offsets]
        self.assertTrue(all(abs(error) <= 0.31 for error in errors))
        self.assertAlmostEqual(max(errors), 0.2389, delta=0.01)
        self.assertAlmostEqual(min(errors), -0.2997, delta=0.01)
        # the sign flips across the half-pixel boundary: the estimator is sub-pixel but not bias-free
        self.assertLess(min(errors), 0.0)
        self.assertGreater(max(errors), 0.0)

    def test_the_theil_sen_slope_recovers_a_known_pitch(self):
        index = np.arange(11.0)
        centres = 40.4+13.7*index
        slope, intercept = theil_sen(index, centres)
        self.assertAlmostEqual(slope, 13.7, places=6)
        self.assertAlmostEqual(intercept, 40.4, places=6)

    def test_one_displaced_line_cannot_move_the_theil_sen_slope(self):
        index = np.arange(11.0)
        centres = 40.0+14.0*index
        centres[4] += 3.0
        slope, _ = theil_sen(index, centres)
        self.assertAlmostEqual(slope, 14.0, delta=0.35)

    def test_a_missing_interior_line_is_named_and_the_fit_survives(self):
        drawn = [40.0+14.0*index for index in range(11)]
        present = [value for position, value in enumerate(drawn) if position != 5]
        energy = peaks(present, 220)
        fit = fit_axis(energy, 40.0, 14.0, 10)
        self.assertTrue(fit["kept"])
        self.assertAlmostEqual(fit["pitch"], 14.0, delta=0.05)
        self.assertAlmostEqual(fit["origin"], 40.0, delta=0.6)

    def test_a_featureless_axis_keeps_the_box(self):
        fit = fit_axis(np.zeros(200), 40.0, 14.0, 10)
        self.assertFalse(fit.get("kept"))
        self.assertIn("too few", fit["reason"])

    def test_the_declared_constants_are_the_declared_constants(self):
        self.assertEqual((SEARCH_SHARE, MISSING_SHARE, OUTLIER_CELLS), (0.35, 0.35, 0.25))
        self.assertEqual(MIN_LINES, {"x": 6, "y": 12})
        self.assertEqual(CONSTANTS["min_lines_x"], 6)


class RefineTest(unittest.TestCase):
    def box(self, pitch=14.0, left=40.0, top=30.0):
        return {"origin": [left, top+20*pitch], "u": [pitch, 0.0], "v": [0.0, -pitch],
                "frame": {"left": left, "right": left+10*pitch, "top": top, "bottom": top+20*pitch},
                "source": "walls"}

    def test_the_refined_lattice_lands_on_the_drawn_grid(self):
        grey = synthetic_grey(pitch=14.0, left=40.0, top=30.0)
        fitted = refine(grey[..., None], self.box())
        self.assertAlmostEqual(fitted["u"][0], 14.0, delta=0.05)
        self.assertAlmostEqual(-fitted["v"][1], 14.0, delta=0.05)
        self.assertAlmostEqual(fitted["frame"]["left"], 40.0, delta=0.6)
        self.assertAlmostEqual(fitted["frame"]["top"], 30.0, delta=0.6)
        self.assertTrue(fitted["fit"]["x"]["kept"] and fitted["fit"]["y"]["kept"])

    def test_a_sub_pixel_box_offset_is_corrected(self):
        # the box is a third of a pixel right and low; the fit must pull the lattice back onto the drawn lines
        grey = synthetic_grey(pitch=14.0, left=40.0, top=30.0)
        shaken = self.box()
        shaken["frame"]["left"] += 0.6
        shaken["origin"][0] += 0.6
        fitted = refine(grey[..., None], shaken)
        # the estimator's bias is worth about half a pixel, so the correction is right to within one pixel here
        self.assertAlmostEqual(fitted["frame"]["left"], 40.0, delta=1.0)
        self.assertLess(abs(fitted["frame"]["left"]-40.0), abs(shaken["frame"]["left"]-40.0))

    def test_a_pitch_one_cell_off_is_out_of_reach_and_the_box_is_kept(self):
        # the declared window is 0.35 of a cell, so a whole-cell pitch error cannot be reached by construction: this is
        # the limitation the arm reports, not a bug the test should hide
        grey = synthetic_grey(pitch=14.0, left=40.0, top=30.0)
        wrong = self.box(pitch=15.0)
        fitted = refine(grey[..., None], wrong)
        self.assertAlmostEqual(fitted["u"][0], 15.0, places=6)
        self.assertFalse(fitted["fit"]["x"]["kept"])

    def test_a_small_pitch_error_is_pulled_back(self):
        grey = synthetic_grey(pitch=14.0, left=40.0, top=30.0)
        wrong = self.box(pitch=14.25)
        fitted = refine(grey[..., None], wrong)
        self.assertAlmostEqual(fitted["u"][0], 14.0, delta=0.15)

    def test_a_gridless_capture_falls_back_to_the_box(self):
        grey = synthetic_grey(grid=False)
        box = self.box()
        fitted = refine(grey[..., None], box)
        self.assertAlmostEqual(fitted["u"][0], box["u"][0], places=6)
        self.assertAlmostEqual(fitted["frame"]["left"], box["frame"]["left"], places=6)
        self.assertFalse(fitted["fit"]["x"]["kept"])

    def test_a_block_on_the_board_supplies_edges_without_moving_the_fit(self):
        grey = synthetic_grey(block=(2, 18))
        fitted = refine(grey[..., None], self.box())
        self.assertAlmostEqual(fitted["u"][0], 14.0, delta=0.05)
        self.assertAlmostEqual(fitted["frame"]["left"], 40.0, delta=0.6)


class DeclarationTest(unittest.TestCase):
    def test_the_declaration_precedes_the_receipt_and_names_the_branches(self):
        declaration = (ROOT/"docs/M52_LATTICE_FIT.md").read_text()
        self.assertIn("predeclared before any scoring run", declaration)
        for branch in ("fit-general", "fit-under-specified", "fit-fails"):
            self.assertIn(branch, declaration)
        self.assertIn("greater than **`0.25` cells, or**", declaration)
        self.assertIn("AUC is below **`0.90`**", declaration)

    def test_the_declaration_discloses_the_motivation_and_the_withdrawn_stage(self):
        declaration = (ROOT/"docs/M52_LATTICE_FIT.md").read_text()
        self.assertIn("Motivation disclosure", declaration)
        self.assertIn("motivated by M51's measured residual", declaration)
        self.assertIn("withdrawn before this declaration", declaration)
        self.assertIn("full 28-board population", declaration)

    def test_the_declaration_fixes_the_no_regression_rule_for_the_named_pair(self):
        declaration = (ROOT/"docs/M52_LATTICE_FIT.md").read_text()
        self.assertIn("training-20261031-v1", declaration)
        self.assertIn("their **M50** values (`0.0588`)", declaration)

    def test_the_arm_adds_no_learned_parameter_and_no_training(self):
        declaration = (ROOT/"docs/M52_LATTICE_FIT.md").read_text()
        self.assertIn("no training", declaration)
        self.assertIn("no threshold fitting", declaration)
        self.assertIn("no new learned parameter", declaration)


if __name__ == "__main__":
    unittest.main()
