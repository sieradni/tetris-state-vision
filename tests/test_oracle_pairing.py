"""Mechanical controls for the M27 three-factor decomposition instrument.

`docs/M27_THREE_FACTOR.md` declares the reads and the reading rule. These tests bind the
pairing arithmetic that produces them: the standing read uses each cell's own label, the
oracle read uses the label of the target cell the model cell covers, a model cell that maps
outside the board or onto an unknown cell is dropped rather than scored as empty, and the
declared branches of the decomposition are reachable and ordered.
"""
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from oracle_pairing_read import (CELLS, counts_at, decomposition, matched_rows,  # noqa: E402
                                 pairs, require_close, score)

GROUPS = ("a", "b")


def board(sample_id, labels, *, logits=None, mapping=None, known=None):
    """One synthetic deciding board in the receipt's own shape."""
    if len(labels) != CELLS:
        raise ValueError("fixture must carry 200 cells")
    return {"sample_id": sample_id, "group_id": GROUPS[0],
            "logits": list(logits) if logits is not None else [0.0]*CELLS,
            "occupied": [int(value) for value in labels],
            "known": list(known) if known is not None else [True]*CELLS,
            "mapping": list(mapping) if mapping is not None else list(range(CELLS)),
            "known_cells": CELLS}


def pixel_board(sample_id, labels, logits):
    return {"sample_id": sample_id, "group_id": GROUPS[0], "known_cells": CELLS,
            "occupied": [int(value) for value in labels], "pixel_logits": list(logits)}


def labels_with(first, count=CELLS):
    labels = [False]*count
    for index in first:
        labels[index] = True
    return labels


class PairingTest(unittest.TestCase):
    def test_as_is_reads_each_cells_own_label(self):
        labels = labels_with((3, 7))
        case = board("s", labels, mapping=list(range(CELLS))[::-1])
        rows = pairs([case], pairing="as_is")
        self.assertEqual(len(rows), CELLS)
        self.assertEqual([label for _, label in rows], [bool(value) for value in labels])

    def test_oracle_read_reads_the_covered_target_label(self):
        labels = labels_with((3, 7))
        mapping = list(range(CELLS))
        mapping[0], mapping[1] = 3, 7
        case = board("s", labels, logits=[10.0, 10.0]+[0.0]*198, mapping=mapping)
        rows = pairs([case], pairing="oracle")
        self.assertEqual(len(rows), CELLS)
        self.assertTrue(rows[0][1])
        self.assertTrue(rows[1][1])
        self.assertFalse(rows[2][1])

    def test_unmapped_model_cells_are_dropped_not_scored_empty(self):
        labels = labels_with((0,))
        mapping = [None]*CELLS
        mapping[5] = 0
        case = board("s", labels, mapping=mapping)
        rows = pairs([case], pairing="oracle")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], (0.0, True))

    def test_unknown_target_cells_are_dropped(self):
        labels = labels_with((0, 1))
        known = [True]*CELLS
        known[0] = False
        case = board("s", labels, known=known)
        rows = pairs([case], pairing="as_is")
        self.assertEqual(len(rows), CELLS-1)

    def test_empty_boards_never_enter_the_pairing(self):
        empty = board("s", [False]*CELLS)
        self.assertEqual(pairs([empty], pairing="as_is"), [])
        self.assertEqual(len(pairs([empty, board("t", labels_with((0,)))], pairing="as_is")),
                         CELLS)

    def test_pixel_pairing_reads_the_pixel_logits_at_the_target_cell(self):
        labels = labels_with((4,))
        mapping = [None]*CELLS
        mapping[9] = 4
        model = board("s", labels, mapping=mapping)
        values = [0.0]*CELLS
        values[4] = 5.0
        pixel = pixel_board("s", labels, values)
        rows = pairs([pixel], pairing="as_is", pixel=True)
        self.assertEqual(rows[4], (5.0, True))
        model_rows, pixel_rows = matched_rows([model], [pixel], pairing="oracle")
        self.assertEqual(model_rows, [(0.0, True)])
        self.assertEqual(pixel_rows, [(5.0, True)])


class ScoreTest(unittest.TestCase):
    def test_perfect_separation_reaches_both_ceilings(self):
        rows = [(5.0, True), (4.0, True), (-1.0, False), (-2.0, False)]
        measured = score(rows)
        self.assertEqual(measured["auc"], 1.0)
        self.assertEqual(measured["f1_at_zero"], 1.0)
        self.assertEqual(measured["best"]["f1"], 1.0)

    def test_a_constant_read_is_chance_and_declares_nothing(self):
        rows = [(0.5, True), (0.5, False)]
        measured = score(rows)
        self.assertEqual(measured["auc"], 0.5)

    def test_the_best_point_is_a_grid_point_and_its_f1_is_attained_there(self):
        rows = [(3.0, True), (1.0, True), (0.5, False), (-3.0, False)]
        measured = score(rows)
        best = measured["best"]
        self.assertEqual(counts_at(rows, best["threshold"])["f1"], best["f1"])
        self.assertGreaterEqual(best["f1"], measured["f1_at_zero"])
        self.assertLessEqual(best["threshold"], 6.0)
        self.assertGreaterEqual(best["threshold"], -6.0)

    def test_a_missing_class_is_refused(self):
        with self.assertRaises(ValueError):
            score([(1.0, True), (2.0, True)])


class ReadingTest(unittest.TestCase):
    @staticmethod
    def reads(registration, representation, boundary=0.0):
        zero = {"best": {"f1": 0.5}, "f1_at_zero": 0.5}
        return {"model_matched_as_is": {"best": {"f1": 0.5},
                                        "f1_at_zero": 0.5},
                "model_oracle": {"best": {"f1": 0.5+registration},
                                 "f1_at_zero": 0.5+registration-boundary},
                "pixel_matched": {"best": {"f1": 0.5+registration+representation}},
                "model_full_as_is": zero, "pixel_full": zero}

    def test_both_terms_above_floor_read_as_two_factor(self):
        reading = decomposition(self.reads(0.20, 0.20))
        self.assertEqual(reading["verdict"], "two-factor-first-order")

    def test_one_term_above_floor_names_that_term(self):
        self.assertEqual(decomposition(self.reads(0.20, 0.01))["verdict"],
                         "registration-first-order")
        self.assertEqual(decomposition(self.reads(0.01, 0.20))["verdict"],
                         "representation-first-order")

    def test_neither_term_reaches_the_floor_keeps_the_single_deficit_reading(self):
        self.assertEqual(decomposition(self.reads(0.09, 0.09))["verdict"], "single-deficit")

    def test_the_boundary_term_cannot_move_the_verdict(self):
        one = decomposition(self.reads(0.20, 0.20, boundary=0.0))
        two = decomposition(self.reads(0.20, 0.20, boundary=0.4))
        self.assertEqual(one["verdict"], two["verdict"])
        self.assertNotAlmostEqual(one["boundary_gap"], two["boundary_gap"])


class JoinTest(unittest.TestCase):
    def test_a_join_within_tolerance_is_admitted(self):
        self.assertEqual(require_close("x", 0.5, 0.5+1e-9), 0.5)

    def test_a_join_outside_tolerance_is_refused(self):
        with self.assertRaises(ValueError):
            require_close("x", 0.5, 0.5+1e-3)

    def test_a_non_finite_join_is_refused(self):
        with self.assertRaises(ValueError):
            require_close("x", math.nan, 0.5)


if __name__ == "__main__":
    unittest.main()
