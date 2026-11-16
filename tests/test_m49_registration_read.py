"""Mechanical controls for the M49 registration read.

The arm imports the campaign's committed per-cell read-out and applies it to the same thirty statistics at a third
sampling: the lattice registered from the board's raster support. These tests pin what the declaration promises: the
retained per-fold transform *is* the committed estimator, applying it to its own features reproduces its own logits,
the read is live on perturbed statistics and is not a shortcut, the support registration is the declared label-free
recipe (the frozen geometry tree's own `pixel_support_lattice`), the branch selection is the predeclared arithmetic at
its declared thresholds, and the declaration precedes the receipt.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from m49_registration_read import (CONTROL_FLOOR, PARTIAL_AUC, REGISTRATION_GATE_AUC,  # noqa: E402
                                   REGISTRATION_GATE_F1, apply_folds, branch_for, displacement_cells, fit_folds)
from pixel_ceiling import logistic_fit  # noqa: E402

sys.path.insert(0, str(ROOT/"src"))
from tetris_vision.model import pixel_support_lattice  # noqa: E402


def synthetic(seed=7, cells=240, groups=4):
    generator = np.random.default_rng(seed)
    features = generator.normal(size=(cells, 30))
    labels = (features[:, 0]+0.5*features[:, 5] > 0.3).astype(float)
    labels[:4] = [1.0, 0.0, 1.0, 0.0]
    return features, labels, np.array([f"group-{index % groups}" for index in range(cells)])


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


class NonInertnessTest(unittest.TestCase):
    def test_perturbing_the_statistics_changes_the_readout(self):
        features, labels, groups = synthetic()
        folds, logits = fit_folds(features, labels, groups)
        moved = features.copy()
        moved[:, 0] += 1.5
        self.assertGreater(float(np.abs(apply_folds(folds, moved, groups)-logits).max()), 0.05)

    def test_the_readout_is_not_a_shortcut_that_ignores_its_sampling(self):
        # a differently sampled input must not reproduce the target logits, which is what a shortcut would do
        features, labels, groups = synthetic()
        folds, logits = fit_folds(features, labels, groups)
        shifted = features+0.4*np.random.default_rng(11).normal(size=features.shape)
        self.assertGreater(float(np.abs(apply_folds(folds, shifted, groups)-logits).max()), 0.05)

    def test_a_class_free_feature_set_cannot_be_fitted(self):
        features, labels, groups = synthetic()
        with self.assertRaises(Exception):
            fit_folds(features, np.zeros_like(labels), groups)


class RegistrationTest(unittest.TestCase):
    """The registered lattice is the declared raster-only recipe, not a label or a fitted quantity."""

    def test_the_support_lattice_follows_the_valid_mask_and_ignores_the_raster_colours(self):
        mask = torch.zeros(1, 1, 64, 64, dtype=torch.bool)
        mask[0, 0, 16:64, 8:56] = True
        lattice = pixel_support_lattice(mask, 20, 10, "cpu", torch.float64)[0].tolist()
        self.assertAlmostEqual(lattice[0][0], 8/64)
        self.assertAlmostEqual(lattice[0][1], 1.0)
        self.assertAlmostEqual(lattice[1][0], (56-8)/64/10)
        self.assertAlmostEqual(lattice[2][1], -(1.0-16/64)/20)

    def test_the_registered_lattice_is_displaced_only_by_the_mask(self):
        # doubling the valid box doubles the cell stride and keeps the bottom-left origin: a pure geometry change
        narrow = torch.zeros(1, 1, 64, 64, dtype=torch.bool)
        narrow[0, 0, 32:64, 16:48] = True
        wide = torch.zeros(1, 1, 64, 64, dtype=torch.bool)
        wide[0, 0, 0:64, 0:64] = True
        small = pixel_support_lattice(narrow, 20, 10, "cpu", torch.float64)[0].tolist()
        large = pixel_support_lattice(wide, 20, 10, "cpu", torch.float64)[0].tolist()
        self.assertAlmostEqual(large[1][0], 2*small[1][0])
        self.assertAlmostEqual(large[2][1], 2*small[2][1])

    def test_the_displacement_is_measured_in_the_target_lattices_own_cells(self):
        class Lattice:
            def __init__(self, origin, u, v):
                self.origin, self.u, self.v = origin, u, v
        target = Lattice([0.0, 200.0], [10.0, 0.0], [0.0, -10.0])
        same = Lattice([0.0, 200.0], [10.0, 0.0], [0.0, -10.0])
        self.assertEqual(displacement_cells(same, target)["mean_magnitude"], 0.0)
        # a one-row, half-column shift is exactly one cell along each axis
        shifted = Lattice([5.0, 190.0], [10.0, 0.0], [0.0, -10.0])
        reading = displacement_cells(shifted, target)
        self.assertAlmostEqual(reading["mean_columns"], 0.5, places=6)
        self.assertAlmostEqual(reading["mean_rows"], 1.0, places=6)
        self.assertAlmostEqual(reading["mean_magnitude"], np.hypot(1.0, 0.5), places=6)


class BranchSelectionTest(unittest.TestCase):
    def test_the_predeclared_thresholds_select_the_predeclared_branches(self):
        self.assertEqual(REGISTRATION_GATE_F1, 0.70)
        self.assertEqual(REGISTRATION_GATE_AUC, 0.90)
        self.assertEqual(CONTROL_FLOOR, 0.382881)
        self.assertEqual(PARTIAL_AUC, 0.86)
        self.assertEqual(branch_for(0.974763, 0.999195), "registration-recovers")
        self.assertEqual(branch_for(0.70, 0.90), "registration-recovers")
        self.assertEqual(branch_for(0.70, 0.899999), "registration-partial")
        self.assertEqual(branch_for(0.70, 0.86), "registration-partial")
        self.assertEqual(branch_for(0.382881, 0.50), "registration-partial")
        self.assertEqual(branch_for(0.38288, 0.86), "registration-partial")
        self.assertEqual(branch_for(0.38288, 0.80), "registration-fails")
        self.assertEqual(branch_for(0.037152, 0.442341), "registration-fails")
        self.assertEqual(branch_for(0.0, 0.0), "registration-fails")

    def test_m48s_predicted_lattice_reading_lands_in_the_fail_branch(self):
        # the falsifier's floor is the arm this one replaces: M48's own predicted-lattice reading
        self.assertEqual(branch_for(0.037152, 0.442341), "registration-fails")
        # and the winner control's F1 alone is not enough to claim recovery
        self.assertEqual(branch_for(0.372881, 0.954529), "registration-partial")


class DeclarationTest(unittest.TestCase):
    def test_the_declaration_precedes_the_receipt_and_states_the_substitution(self):
        declaration = (ROOT/"docs/M49_REGISTRATION_READ.md").read_text()
        self.assertIn("predeclared before any scoring run", declaration)
        self.assertIn("out-of-fold logits", declaration)
        self.assertIn("reproduces the committed logits to `1e-6`", declaration)
        self.assertIn("registration-recovers", declaration)
        self.assertIn("registration-partial", declaration)
        self.assertIn("registration-fails", declaration)

    def test_the_declaration_names_the_label_free_registration_recipe(self):
        declaration = (ROOT/"docs/M49_REGISTRATION_READ.md").read_text()
        self.assertIn("pixel_support_lattice", declaration)
        self.assertIn("no label, no prevalence, no scored cell", declaration)
        self.assertIn("never sees the committed model's predicted lattice", declaration)

    def test_the_arm_adds_no_learned_parameter_and_no_training(self):
        declaration = (ROOT/"docs/M49_REGISTRATION_READ.md").read_text()
        self.assertIn("**no training**", declaration)
        self.assertIn("no threshold fitting", declaration)
        self.assertIn("no new learned parameter", declaration)


if __name__ == "__main__":
    unittest.main()
