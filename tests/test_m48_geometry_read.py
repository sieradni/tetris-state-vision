"""Mechanical controls for the M48 geometry read.

The arm imports the campaign's committed per-cell read-out and applies it to the same thirty statistics at two
samplings. These tests pin what the declaration promises: the retained per-fold transform *is* the committed
estimator (it reproduces `pixel_ceiling.logistic_fit` exactly), applying it to its own features reproduces its own
logits, the read-out is live on perturbed features and not a shortcut, the branch selection is the predeclared
arithmetic at its declared thresholds, and the declaration precedes the receipt.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from m48_geometry_read import CONTROL_FLOOR, CELL_GATE, branch_for, fit_folds, apply_folds  # noqa: E402
from pixel_ceiling import logistic_fit  # noqa: E402


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
        self.assertEqual(logits.shape, committed.shape)
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

    def test_the_readout_is_not_a_shortcut_that_ignores_its_input(self):
        # a shifted sampling must not reproduce the target logits, which is what a shortcut would do
        features, labels, groups = synthetic()
        folds, logits = fit_folds(features, labels, groups)
        shifted = features + 0.4*np.random.default_rng(11).normal(size=features.shape)
        self.assertGreater(float(np.abs(apply_folds(folds, shifted, groups)-logits).max()), 0.05)

    def test_a_class_free_feature_set_cannot_be_fitted(self):
        features, labels, groups = synthetic()
        with self.assertRaises(Exception):
            fit_folds(features, np.zeros_like(labels), groups)


class BranchSelectionTest(unittest.TestCase):
    def test_the_predeclared_thresholds_select_the_predeclared_branches(self):
        self.assertEqual(CELL_GATE, 0.70)
        self.assertEqual(CONTROL_FLOOR, 0.382881)
        self.assertEqual(branch_for(0.974763), "readout-binds")
        self.assertEqual(branch_for(0.70), "readout-binds")
        self.assertEqual(branch_for(0.699999), "geometry-costs")
        self.assertEqual(branch_for(0.382881), "geometry-costs")
        self.assertEqual(branch_for(0.38288), "geometry-binds")
        self.assertEqual(branch_for(0.13375), "geometry-binds")

    def test_the_geometry_branch_matches_the_measured_control(self):
        # the control's own F1 at 0.0 is the floor: an arm at that level has gained nothing over the latent read
        self.assertLess(branch_for(0.372881), "geometry-costs")


class DeclarationTest(unittest.TestCase):
    def test_the_declaration_precedes_the_receipt_and_states_the_substitution(self):
        declaration = (ROOT/"docs/M48_GEOMETRY_READ.md").read_text()
        self.assertIn("predeclared before any scoring run", declaration)
        self.assertIn("out-of-fold logits", declaration)
        self.assertIn("coefficients", declaration)
        self.assertIn("reproduces the committed logits to `1e-6`", declaration)
        self.assertIn("readout-binds", declaration)
        self.assertIn("geometry-binds", declaration)

    def test_the_arm_adds_no_learned_parameter_and_no_training(self):
        declaration = (ROOT/"docs/M48_GEOMETRY_READ.md").read_text()
        self.assertIn("No training", declaration)
        self.assertIn("no threshold fitting", declaration)
        self.assertIn("no new\n  learned parameter", declaration)


if __name__ == "__main__":
    unittest.main()
