"""V-139: the exact-state metric must count a board exact only when the board really is exact."""
import importlib
import inspect
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

ROWS, COLUMNS = 20, 10
ROOT = Path(__file__).resolve().parents[1]


def case(truth_columns, decided_columns, stray=None):
    """A synthetic sweep case: `decided_columns` says which rows the read decides occupied, `truth_columns` the labels.

    `stray` is a (row, column) cell decided occupied in a column that is truly empty - the shape the earlier lenient
    comparison silently dropped.
    """
    logits, occupied = [], []
    for row in range(ROWS):
        for column in range(COLUMNS):
            truth = 1 if (column in truth_columns and row in truth_columns[column]) else 0
            predicted = column in decided_columns and row in decided_columns[column]
            if stray and (row, column) == stray:
                predicted = True
            logits.append(1.0 if predicted else -1.0)
            occupied.append(truth)
    return {"split": "training", "logits": logits, "occupied": occupied}


def result(*cases):
    return {"config": {"visible_rows": ROWS, "columns": COLUMNS}, "cases": list(cases)}


TRUTH = {0: (0, 1, 2), 3: (0,), 7: (0, 1, 2, 3, 4)}
IDENTICAL = dict(TRUTH)


class ExactStateTests(unittest.TestCase):
    def test_a_perfectly_decided_board_is_exact(self):
        from record_crossing import exact_boards_at_zero
        scored = exact_boards_at_zero(result(case(TRUTH, IDENTICAL)))
        self.assertEqual((scored["boards"], scored["exact"], scored["boundary_exact"]), (1, 1, 1))

    def test_one_wrong_cell_makes_the_board_inexact(self):
        from record_crossing import exact_boards_at_zero
        wrong = {0: (0, 1), 3: (0,), 7: (0, 1, 2, 3, 4)}          # one missed cell in column 0
        scored = exact_boards_at_zero(result(case(TRUTH, wrong)))
        self.assertEqual((scored["exact"], scored["boundary_exact"]), (0, 0))

    def test_a_filled_but_truly_empty_column_can_never_be_exact(self):
        # The V-139 shape: every occupied column's top row is right, and one empty column is filled. The previous
        # comparison dropped the column where either side was empty, so this board scored as exact.
        from record_crossing import exact_boards_at_zero
        scored = exact_boards_at_zero(result(case(TRUTH, IDENTICAL, stray=(0, 5))))
        self.assertEqual((scored["exact"], scored["boundary_exact"]), (0, 0))

    def test_an_empty_column_decided_empty_stays_exact(self):
        from record_crossing import exact_boards_at_zero
        scored = exact_boards_at_zero(result(case(TRUTH, IDENTICAL)))
        self.assertEqual(scored["exact"], 1)

    def test_the_cell_definition_agrees_with_the_sweep(self):
        from occupancy_sweep import _counts
        from record_crossing import exact_boards_at_zero
        for stray in (None, (0, 5), (3, 9)):
            built = case(TRUTH, IDENTICAL, stray=stray)
            sweep = _counts(built, 0.0)[4]
            curated = exact_boards_at_zero(result(built))["exact"]
            self.assertEqual(bool(sweep), bool(curated), stray)

    def test_a_population_without_a_board_is_refused(self):
        from record_crossing import exact_boards_at_zero
        with self.assertRaises(ValueError):
            exact_boards_at_zero(result(case({}, {})))

    def test_every_curator_carries_the_strict_comparison(self):
        for name in ("record_centring", "record_contiguity", "record_crossing", "record_stack_centred",
                     "record_dense_hinge", "record_midpoint_anchor"):
            module = importlib.import_module(name)
            source = inspect.getsource(module.exact_boards_at_zero)
            self.assertNotIn("if got is not None and want is not None", source, name)
            self.assertIn("value > 0.0", source, name)
            self.assertIn("zip(decided, occupied)", source, name)


class RetainedEvidenceTests(unittest.TestCase):
    def test_no_retained_arm_has_an_exact_board(self):
        # The corrected figure over real receipts: every arm ever retained is 0 of 28 under the strict definition.
        from record_crossing import exact_boards_at_zero
        receipts = sorted(ROOT.glob("runs/*-sweep-r1/result.json"))
        self.assertGreater(len(receipts), 40)
        checked = 0
        for path in receipts:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("status") != "passed":
                continue
            try:
                scored = exact_boards_at_zero(data)
            except ValueError:
                continue
            checked += 1
            self.assertEqual(scored["exact"], 0, path.parent.name)
            self.assertEqual(scored["boundary_exact"], 0, path.parent.name)
        self.assertGreater(checked, 40)


if __name__ == "__main__":
    unittest.main()
