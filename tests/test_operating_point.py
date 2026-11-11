"""The predeclared operating-point audit: invariance, no leakage, the state table and its guards."""
import copy
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]

import operating_point as audit


def board(group, labels, scores):
    return {"group_id": group, "split": "training", "logits": list(scores),
            "occupied": list(labels)}


def separable_cases(offset=0.0, groups=4, boards=2):
    """Perfectly separable boards: occupied logit 1.0, empty logit -1.0, shifted by `offset`."""
    cases = []
    for group in range(groups):
        for index in range(boards):
            labels = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
            scores = [(1.0 if label else -1.0)+offset for label in labels]
            cases.append(board(group, labels, scores))
    return cases


class Invariance(unittest.TestCase):
    def test_a_shifted_boundary_changes_the_standing_read_and_not_the_out_of_fold_one(self):
        control = audit.arm_read(separable_cases())
        shifted = audit.arm_read(separable_cases(offset=3.0))
        self.assertAlmostEqual(control["auc"], 1.0, places=12)
        self.assertAlmostEqual(shifted["auc"], 1.0, places=12)
        # The same ranking, decided at 0.0: the shifted arm predicts nearly every cell occupied.
        self.assertAlmostEqual(control["f1_at_zero"], 1.0, places=12)
        self.assertLess(shifted["f1_at_zero"], 0.5)
        # The out-of-fold boundary travels with the offset, so the held-out decision is identical.
        self.assertAlmostEqual(control["cv"]["f1"], 1.0, places=12)
        self.assertAlmostEqual(shifted["cv"]["f1"], 1.0, places=12)
        self.assertEqual(control["cv"]["counts"], shifted["cv"]["counts"])
        self.assertAlmostEqual(control["logit_offset"], 2.0, places=12)
        self.assertAlmostEqual(shifted["logit_offset"], 2.0, places=12)
        self.assertGreater(shifted["median_occupied"]-control["median_occupied"], 2.9)
        self.assertEqual(audit.pair_state(shifted["f1_at_zero"]-control["f1_at_zero"],
                                          shifted["cv"]["f1"]-control["cv"]["f1"]), "shifted")

    def test_equal_arms_are_aligned_at_every_floor(self):
        cases = separable_cases()
        read = audit.arm_read(cases)
        self.assertEqual(audit.pair_state(0.0, 0.0), "aligned")
        self.assertEqual(read["exact_at_zero"], read["cv"]["exact_boards"])
        self.assertEqual(read["exact_at_zero"], len(cases))


class NoLeakage(unittest.TestCase):
    def test_a_held_out_groups_labels_cannot_move_its_own_fold_boundary(self):
        cases = []
        for group in range(4):
            for index in range(2):
                labels = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
                scores = [0.4, -0.2, 1.1, -0.9, -0.1, 0.8, -1.3, -0.4, 0.2, -0.7]
                cases.append(board(group, labels, [value+group*0.35 for value in scores]))
        base = audit.cross_validated(cases)
        for fold_index, fold in enumerate(base["folds"]):
            corrupt = copy.deepcopy(cases)
            for case in corrupt:
                if case["group_id"] == fold["group_id"]:
                    case["occupied"] = [1-label for label in case["occupied"]]
            moved = audit.cross_validated(corrupt)
            self.assertEqual(moved["folds"][fold_index]["threshold"], fold["threshold"])
            held = [case for case in corrupt if case["group_id"] == fold["group_id"]]
            self.assertEqual((moved["folds"][fold_index]["true_positive"],
                              moved["folds"][fold_index]["false_positive"],
                              moved["folds"][fold_index]["false_negative"]),
                             audit.counts(held, fold["threshold"]))

    def test_the_fold_boundary_is_the_complements_optimum_not_the_pooled_one(self):
        cases = separable_cases(offset=0.0, groups=3)
        # A fourth group whose scale is 30 units away would drag any pooled choice off the others.
        for index in range(2):
            labels = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
            cases.append(board(3, labels, [(1.0 if label else -1.0)+30.0 for label in labels]))
        folds = audit.cross_validated(cases)["folds"]
        self.assertEqual(len(folds), 4)
        for fold in folds:
            complement = [case for case in cases if case["group_id"] != fold["group_id"]]
            self.assertEqual(fold["threshold"], audit.select_threshold(complement))


class Decisions(unittest.TestCase):
    def test_a_logit_exactly_at_the_threshold_counts_as_empty(self):
        cases = [board(0, [1, 0], [0.0, 0.0]), board(1, [1, 0], [2.0, -2.0]),
                 board(2, [1, 0], [1.0, -1.0]), board(3, [0, 1], [1.0, -1.0])]
        # The 0.0 logit on board 0 is *not* predicted, so its occupied cell is a miss.
        self.assertEqual(audit.counts(cases, 0.0), (2, 1, 2))
        self.assertAlmostEqual(audit.f1_from_counts(2, 1, 2), 4/7, places=12)

    def test_the_selection_rule_is_the_sweeps_highest_f1_then_precision_then_lower_boundary(self):
        cases = [board(0, [1, 0], [1.4, -1.1]), board(1, [1, 0], [1.6, -1.3]),
                 board(2, [1, 0], [1.2, -1.0])]
        chosen = audit.select_threshold(cases, thresholds=(-2.0, 0.0, 0.5, 1.5))
        self.assertEqual(chosen, 0.0)

    def test_an_empty_board_contributes_no_cells_to_the_shared_curve(self):
        cases = separable_cases(groups=3)
        polluted = cases + [board(0, [0]*10, [5.0]*10)]
        self.assertEqual(audit.counts(audit.nonempty(polluted), 0.0),
                         audit.counts(audit.nonempty(cases), 0.0))
        self.assertEqual(len(audit.nonempty(polluted)), len(cases))
        self.assertEqual(audit.arm_read(polluted)["empty_boards"], 1)

    def test_f1_refuses_an_empty_denominator(self):
        with self.assertRaises(ValueError):
            audit.f1_from_counts(0, 0, 0)
        with self.assertRaises(ValueError):
            audit.f1_from_counts(-1, 0, 2)


class StateTable(unittest.TestCase):
    def test_every_branch_of_the_predeclared_table(self):
        self.assertEqual(audit.pair_state(0.00, 0.050), "refusal_reversed")
        self.assertEqual(audit.pair_state(-0.16, 0.12), "refusal_reversed")
        self.assertEqual(audit.pair_state(0.07, -0.01), "gain_reversed")
        self.assertEqual(audit.pair_state(0.06, 0.005), "shifted")
        self.assertEqual(audit.pair_state(0.08, 0.01), "shifted")
        self.assertEqual(audit.pair_state(0.02, 0.03), "aligned")
        self.assertEqual(audit.pair_state(0.05, 0.05), "aligned")

    def test_the_overall_verdict_takes_its_first_matching_branch(self):
        def pairs(*states):
            return [{"state": state} for state in states]
        self.assertEqual(audit.audit_verdict(pairs("aligned", "shifted"))["label"],
                         "operating_point_matters")
        self.assertEqual(audit.audit_verdict(pairs("aligned", "refusal_reversed", "shifted"))["label"],
                         "standing_read_confounded")
        self.assertEqual(audit.audit_verdict(pairs("aligned", "aligned"))["label"],
                         "standing_read_sufficient")
        with self.assertRaises(ValueError):
            audit.audit_verdict(pairs("invented"))
        with self.assertRaises(ValueError):
            audit.audit_verdict([])


class Guards(unittest.TestCase):
    def test_the_declared_population_and_floor_are_pinned(self):
        self.assertEqual((audit.DECIDING_CASES, audit.DECIDING_GROUPS,
                          audit.OCCUPIED_CELLS, audit.KNOWN_CELLS), (36, 12, 667, 200))
        self.assertAlmostEqual(audit.ATTRIBUTION_FLOOR, 0.05, places=12)
        self.assertIn("leave-one-group-out", audit.cross_validated(separable_cases())["selection"])

    def test_a_receipt_of_another_population_is_refused(self):
        cases = separable_cases(groups=3)
        with self.assertRaises(ValueError):
            audit.require_population(cases)
        with self.assertRaises(ValueError):
            audit.require_population([])

    def test_a_board_with_unpaired_logits_and_labels_is_refused(self):
        for broken in ({"group_id": 0, "logits": [0.1, 0.2], "occupied": [1]},
                       {"group_id": 0, "logits": [0.1], "occupied": [2]},
                       {"group_id": 0, "logits": [float("nan")], "occupied": [1]},
                       {"group_id": 0, "logits": [0.1], "occupied": [True]}):
            with self.assertRaises(ValueError):
                audit.require_cases([broken])

    def test_a_paired_interval_is_zero_for_identical_arms_and_positive_for_a_consistent_gain(self):
        base = [{"group_id": group, "true_positive": 8, "false_positive": 2, "false_negative": 2}
                for group in range(4)]
        identical = audit.paired_delta_interval(base, copy.deepcopy(base))
        self.assertEqual((identical["low"], identical["high"]), (0.0, 0.0))
        self.assertFalse(identical["excludes_zero"])
        better = [{"group_id": group, "true_positive": 10, "false_positive": 0, "false_negative": 0}
                  for group in range(4)]
        gain = audit.paired_delta_interval(base, better)
        self.assertGreater(gain["low"], 0.0)
        self.assertTrue(gain["excludes_zero"])
        with self.assertRaises(ValueError):
            audit.paired_delta_interval(base, better[:3])
        with self.assertRaises(ValueError):
            audit.paired_delta_interval(base, list(reversed(better)))

    def test_a_bootstrap_over_folds_needs_three_folds_and_returns_an_ordered_interval(self):
        folds = [{"true_positive": 10, "false_positive": 0, "false_negative": 0},
                 {"true_positive": 9, "false_positive": 1, "false_negative": 1},
                 {"true_positive": 8, "false_positive": 2, "false_negative": 2}]
        interval = audit.bootstrap_cv_interval(folds)
        self.assertLessEqual(interval["low"], interval["median"])
        self.assertLessEqual(interval["median"], interval["high"])
        perfect = [{"true_positive": 10, "false_positive": 0, "false_negative": 0}]*3
        self.assertEqual(audit.bootstrap_cv_interval(perfect)["low"], 1.0)
        with self.assertRaises(ValueError):
            audit.bootstrap_cv_interval(folds[:2])


if __name__ == "__main__":
    unittest.main()
