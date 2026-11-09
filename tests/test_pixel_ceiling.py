"""The declared pixel-ceiling rule: count-based F1, fold-mean AUC, a collapsing control."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from pixel_ceiling import (AUC_THRESHOLD, F1_THRESHOLD, cell_features, ceiling_verdict, counts,
                           evaluate, f1, logistic_fit, raster, shuffled_labels, sweep,
                           tie_aware_auc)
from record_pixel_ceiling import require_deciding
from record_stream_run import OWNED_RUN_PATTERN


def groups_and_labels(size=200, groups=6, rate=.12, seed=7):
    generator = np.random.default_rng(seed)
    group = np.repeat([f"ancestry-{index}" for index in range(groups)], size)
    labels = (generator.random(group.size) < rate).astype(int)
    return group, labels


class RankingTests(unittest.TestCase):
    def test_constant_scores_are_exactly_chance(self):
        labels = np.array([1, 0, 1, 0])
        self.assertEqual(tie_aware_auc(np.zeros(4), labels), .5)

    def test_perfect_separation_is_one(self):
        self.assertEqual(tie_aware_auc(np.array([.1, .2, .8, .9]), np.array([0, 0, 1, 1])), 1.)

    def test_ties_are_averaged_rather_than_broken_by_order(self):
        # Two occupied and two empty cells; each class shares one score value.
        self.assertEqual(tie_aware_auc(np.array([0., 0., 1., 1.]), np.array([0, 1, 0, 1])), .5)
        self.assertAlmostEqual(tie_aware_auc(np.array([0., 0., 1., 1.]), np.array([0, 0, 1, 1])), 1.)


class RuleTests(unittest.TestCase):
    def measured(self, operating_f1, fold_auc, pooled=0.0):
        return {"operating": {"f1": operating_f1}, "fold_mean_auc": fold_auc, "pooled_auc": pooled}

    def test_both_bars_are_required(self):
        control = self.measured(.0, .5)
        self.assertEqual(ceiling_verdict(self.measured(.9, .95), control)["status"], "separable")
        self.assertEqual(ceiling_verdict(self.measured(.69, .95), control)["status"], "not-separable")
        self.assertEqual(ceiling_verdict(self.measured(.9, .89), control)["status"], "not-separable")

    def test_no_verdict_reads_the_pooled_auc(self):
        control = self.measured(.0, .5)
        passing = ceiling_verdict(self.measured(.9, .95, pooled=0.0), control)
        self.assertEqual(passing["status"], "separable")

    def test_a_control_that_does_not_collapse_invalidates_the_instrument(self):
        verdict = ceiling_verdict(self.measured(.9, .95), self.measured(.8, .95))
        self.assertEqual(verdict["status"], "instrument-invalid")

    def test_the_bars_are_the_standing_gates(self):
        self.assertEqual((F1_THRESHOLD, AUC_THRESHOLD), (.70, .90))


class PerfoldTests(unittest.TestCase):
    def test_a_per_fold_constant_score_inflates_pooled_auc_but_not_the_fold_mean(self):
        # V-066: four folds, each with both classes, constant inside a fold and different
        # between folds. Fold AUCs are chance; the pooled AUC orders by fold frequency.
        scores, labels, folds = [], [], []
        for index, (rate, score) in enumerate(((.02, -3.), (.05, -2.), (.3, 1.), (.5, 2.))):
            count = 100
            fold_labels = np.zeros(count, dtype=int)
            fold_labels[:int(rate*count)] = 1
            labels.extend(fold_labels)
            scores.extend([score]*count)
            folds.extend([f"ancestry-{index}"]*count)
        pooled = tie_aware_auc(scores, labels)
        per_fold = [tie_aware_auc(np.array([score]*100), np.array(labels[index*100:(index+1)*100]))
                    for index, score in enumerate((-3., -2., 1., 2.))]
        self.assertGreater(pooled, .75)
        self.assertLess(pooled, .95)
        self.assertTrue(all(value == .5 for value in per_fold))

    def test_fit_reports_the_fold_mean_that_the_rule_reads(self):
        group, labels = groups_and_labels()
        features = np.array([[value, 0.] for value in labels], dtype=float)+np.random.default_rng(3).normal(0, .01, (len(labels), 2))
        logits, per_fold = logistic_fit(features, labels, group)
        self.assertEqual(len(per_fold), 6)
        self.assertEqual(len(logits), len(labels))


class CeilingBehaviourTests(unittest.TestCase):
    def test_separable_pixels_clear_the_bar_and_random_pixels_do_not(self):
        group, labels = groups_and_labels(size=300, groups=6, rate=.2)
        signal = np.c_[labels*4+np.random.default_rng(11).normal(0, .1, len(labels)),
                       np.zeros(len(labels))]
        measured = evaluate(signal, labels, group)
        control = evaluate(signal, shuffled_labels(labels, group), group)
        self.assertEqual(ceiling_verdict(measured, control)["status"], "separable")
        noise = np.random.default_rng(13).normal(0, 1, (len(labels), 2))
        self.assertEqual(ceiling_verdict(evaluate(noise, labels, group), control)["status"],
                         "not-separable")

    def test_the_control_permutation_keeps_the_class_budget_and_breaks_the_mapping(self):
        group, labels = groups_and_labels()
        control = shuffled_labels(labels, group)
        self.assertEqual(int(control.sum()), int(labels.sum()))
        self.assertEqual(sorted(control.tolist()), sorted(labels.tolist()))
        self.assertFalse(np.array_equal(control, labels))

    def test_counts_and_sweep_use_the_gate_identity(self):
        logits, labels = np.array([2., -2., 1., -1.]), np.array([1, 0, 0, 1])
        position = counts(logits, labels, 0.)
        self.assertEqual((position["true_positive"], position["false_positive"],
                          position["false_negative"]), (1, 1, 1))
        self.assertAlmostEqual(f1(position), .5)
        # The best swept point is the threshold that keeps both occupied cells.
        self.assertEqual(sweep(logits, labels)["f1"], .8)


class FeatureTests(unittest.TestCase):
    def lattice(self, cell=8):
        from tetris_vision.layout import Lattice
        return Lattice(origin=(4., 164.), u=(cell, 0.), v=(0., -cell))

    def test_features_are_deterministic_and_pixel_only(self):
        raster_pixels = np.full((176, 96, 3), 30., dtype=np.float64)
        before = cell_features(raster_pixels, self.lattice())
        self.assertEqual(before.shape, (200, 30))
        self.assertTrue(np.array_equal(before, cell_features(raster_pixels.copy(), self.lattice())))
        changed = raster_pixels.copy()
        # Row 0 is the bottom row; fill part of its first cell interior only.
        changed[158:164, 6:12] = 220.
        after = cell_features(changed, self.lattice())
        self.assertFalse(np.allclose(after[0], before[0]))
        self.assertTrue(np.allclose(after[1], before[1]))

    def test_a_subpixel_lattice_is_refused(self):
        with self.assertRaises(ValueError):
            cell_features(np.zeros((176, 96, 3)), self.lattice(cell=.5))

    def test_a_non_rgb_raster_is_refused(self):
        with self.assertRaises(ValueError):
            cell_features(np.zeros((176, 96)), self.lattice())

    def test_raster_converts_an_image(self):
        from PIL import Image
        image = Image.new("RGBA", (8, 8), (1, 2, 3, 255))
        self.assertEqual(raster(image).shape, (8, 8, 3))


class CuratorTests(unittest.TestCase):
    def population(self):
        return {"nonempty_boards": 28, "empty_boards": 8, "occupied_cells": 667, "empty_cells": 4933}

    def cases(self):
        rows = []
        for index in range(36):
            occupied = 0 if index < 8 else 0
            rows.append({"known_cells": 200, "occupied": occupied})
        rows[8]["occupied"] = 667
        return rows

    def test_the_declared_arithmetic_is_required(self):
        control = {"operating": {"f1": .1}, "fold_mean_auc": .5}
        require_deciding(self.population(), self.cases(), control)
        broken = dict(self.population(), occupied_cells=666)
        with self.assertRaises(ValueError):
            require_deciding(broken, self.cases(), control)

    def test_a_control_that_clears_the_bar_is_refused(self):
        with self.assertRaises(ValueError):
            require_deciding(self.population(), self.cases(),
                             {"operating": {"f1": .95}, "fold_mean_auc": .5})

    def test_the_owned_run_family_is_declared(self):
        for name in ("m3-occpix-ceiling-r1", "m3-occpix-ceiling-r1-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)


if __name__ == "__main__":
    unittest.main()
