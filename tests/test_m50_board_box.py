"""Mechanical controls for the M50 board-box read.

The arm imports the campaign's committed per-cell read-out and applies it to the same thirty statistics sampled at a
lattice the capture's own well walls define. These tests pin what the declaration promises: the detector finds a
synthetic well's inner edges and ignores a short panel, its declared constants are the declared constants, the
retained per-fold transform *is* the committed estimator, the read is live and not a shortcut, the branch arithmetic
is the predeclared one at the declared thresholds, and the declaration precedes the receipt.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from m50_board_box_read import (ASPECT_SLACK, CLUSTER, CONTROL_FLOOR, DETECTION_GATE_CELLS, EDGE,  # noqa: E402
                               AUC_GATE, EOF_F1_GATE, MIN_WIDTH, WALL_SHARE, apply_folds, branch_for,
                               detect_box, fit_folds)
from pixel_ceiling import logistic_fit  # noqa: E402


def synthetic(seed=7, cells=240, groups=4):
    generator = np.random.default_rng(seed)
    features = generator.normal(size=(cells, 30))
    labels = (features[:, 0]+0.5*features[:, 5] > 0.3).astype(float)
    labels[:4] = [1.0, 0.0, 1.0, 0.0]
    return features, labels, np.array([f"group-{index % groups}" for index in range(cells)])


def frame(width=320, height=460, cell=14, left=40, top=30, panel=True):
    """A capture with a ten-by-twenty well and, optionally, a short panel beside it."""
    background = np.array([24.0, 29.0, 42.0])
    pixels = np.tile(background, (height, width, 1))
    for x in range(left, left+10*cell+1, cell):
        pixels[top:top+20*cell+1, x] = [150.0, 160.0, 170.0]
    for y in range(top, top+20*cell+1, cell):
        pixels[y, left:left+10*cell+1] = [150.0, 160.0, 170.0]
    if panel:
        for x in (left+10*cell+30, left+10*cell+90):
            pixels[top:top+5*cell, x] = [150.0, 160.0, 170.0]
        for y in (top, top+5*cell):
            pixels[y, left+10*cell+30:left+10*cell+91] = [150.0, 160.0, 170.0]
    return pixels


class DetectorTest(unittest.TestCase):
    def test_the_detector_reads_the_wells_inner_edges(self):
        box = detect_box(frame())
        self.assertEqual(box["origin"], [40.0, 30.0+20*14])
        self.assertAlmostEqual(box["u"][0], 14.0)
        self.assertAlmostEqual(box["v"][1], -14.0)

    def test_a_short_panel_beside_the_well_is_ignored(self):
        with_panel = detect_box(frame(panel=True))
        without = detect_box(frame(panel=False))
        self.assertEqual(with_panel["origin"], without["origin"])
        self.assertAlmostEqual(with_panel["u"][0], without["u"][0])

    def test_the_detector_scales_with_the_capture(self):
        small = detect_box(frame(width=260, height=400, cell=10, left=20, top=20))
        large = detect_box(frame(width=420, height=600, cell=18, left=60, top=40))
        self.assertAlmostEqual(small["u"][0], 10.0)
        self.assertAlmostEqual(large["u"][0], 18.0)
        self.assertAlmostEqual(large["v"][1], -18.0)

    def test_a_flat_capture_has_no_box_to_read(self):
        flat = np.tile(np.array([24.0, 29.0, 42.0]), (64, 64, 1))
        with self.assertRaises(Exception):
            detect_box(flat)

    def test_the_declared_constants_are_the_declared_constants(self):
        self.assertEqual((EDGE, WALL_SHARE, CLUSTER), (12.0, 0.5, 6))
        self.assertEqual(MIN_WIDTH, 60)
        self.assertAlmostEqual(ASPECT_SLACK, 0.06)
        self.assertEqual(DETECTION_GATE_CELLS, 0.25)


class ReadOutProvenanceTest(unittest.TestCase):
    def test_the_retained_transform_is_the_committed_estimator(self):
        features, labels, groups = synthetic()
        folds, logits = fit_folds(features, labels, groups)
        committed, _ = logistic_fit(features, labels, groups)
        self.assertLess(float(np.abs(logits-committed).max()), 1e-10)
        self.assertEqual(len(folds), len(set(groups.tolist())))

    def test_applying_the_folds_to_their_own_features_reproduces_their_logits(self):
        features, labels, groups = synthetic()
        folds, logits = fit_folds(features, labels, groups)
        self.assertLess(float(np.abs(apply_folds(folds, features, groups)-logits).max()), 1e-10)

    def test_perturbing_the_statistics_changes_the_readout(self):
        features, labels, groups = synthetic()
        folds, logits = fit_folds(features, labels, groups)
        moved = features.copy()
        moved[:, 0] += 1.5
        self.assertGreater(float(np.abs(apply_folds(folds, moved, groups)-logits).max()), 0.05)

    def test_the_readout_is_not_a_shortcut_that_ignores_its_sampling(self):
        features, labels, groups = synthetic()
        folds, logits = fit_folds(features, labels, groups)
        shifted = features+0.4*np.random.default_rng(11).normal(size=features.shape)
        self.assertGreater(float(np.abs(apply_folds(folds, shifted, groups)-logits).max()), 0.05)


class BranchSelectionTest(unittest.TestCase):
    def test_the_predeclared_thresholds_select_the_predeclared_branches(self):
        self.assertEqual(DETECTION_GATE_CELLS, 0.25)
        self.assertEqual(AUC_GATE, 0.90)
        self.assertEqual(EOF_F1_GATE, 0.70)
        # arguments are (displacement in cells, F1 at 0.0, AUC)
        self.assertEqual(branch_for(0.061328, 0.95, 0.974), "box-recovers")
        self.assertEqual(branch_for(0.25, 0.70, 0.90), "box-recovers")
        self.assertEqual(branch_for(0.250001, 0.90, 0.95), "box-partial")
        self.assertEqual(branch_for(0.80, 0.10, 0.86), "box-partial")
        self.assertEqual(branch_for(0.80, 0.05, 0.50), "box-fails")

    def test_m49s_reading_lands_in_the_fail_branch(self):
        self.assertEqual(branch_for(8.256949, 0.0, 0.505163), "box-fails")


class DeclarationTest(unittest.TestCase):
    def test_the_declaration_precedes_the_receipt_and_states_the_substitution(self):
        declaration = (ROOT/"docs/M50_BOARD_BOX.md").read_text()
        self.assertIn("predeclared before any scoring run", declaration)
        self.assertIn("out-of-fold logits", declaration)
        self.assertIn("reproduces the committed logits to `1e-6`", declaration)
        for branch in ("box-recovers", "box-partial", "box-fails"):
            self.assertIn(branch, declaration)

    def test_the_declaration_fixes_the_falsifier_and_the_instrument_check(self):
        declaration = (ROOT/"docs/M50_BOARD_BOX.md").read_text()
        self.assertIn("displacement `<= 0.25` cells", declaration)
        self.assertIn("`0.802227`", declaration)
        self.assertIn("`0.061328`", declaration)
        self.assertIn("`3.325530`x", declaration)

    def test_the_arm_adds_no_learned_parameter_and_no_training(self):
        declaration = (ROOT/"docs/M50_BOARD_BOX.md").read_text()
        self.assertIn("**no training**", declaration)
        self.assertIn("no threshold fitting", declaration)
        self.assertIn("no new learned parameter", declaration)


if __name__ == "__main__":
    unittest.main()
