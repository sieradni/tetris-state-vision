from dataclasses import replace
import unittest

from tetris_vision.evaluate import Example, evaluate, exact_state
from tetris_vision.render import Appearance, render, sample_scene
from tetris_vision.state import Cell, Icon, Queue


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.state = render(sample_scene(31)).state
        self.example = Example("one", "group-one", "clean", self.state)

    def test_missing_prediction_is_failure_and_stays_in_denominator(self):
        report = evaluate([self.example], {})
        self.assertEqual(report["readable"], 1)
        self.assertEqual(report["exact_rate"], 0)
        self.assertEqual(report["abstained_readable"], 1)

    def test_one_wrong_cell_fails_whole_state(self):
        rows = [list(row) for row in self.state.board.rows]
        old = rows[0][0]
        rows[0][0] = Cell(not old.occupied)
        wrong = replace(self.state, board=replace(self.state.board, rows=tuple(map(tuple, rows))))
        self.assertFalse(exact_state(self.state, wrong))
        report = evaluate([self.example], {"one": wrong})
        self.assertEqual(report["accepted_wrong_readable"], 1)

    def test_unknown_slot_is_not_removed_to_match_queue(self):
        wrong = replace(self.state, queue=Queue((Icon("unknown"), *self.state.queue.items[1:])),
                        decision="needs_review")
        self.assertFalse(exact_state(self.state, wrong))

    def test_one_queue_item_fails_whole_state(self):
        wrong = replace(self.state, queue=replace(self.state.queue, items=(Icon("observed", "Z"),
                                                                  *self.state.queue.items[1:])))
        self.assertFalse(exact_state(self.state, wrong))

    def test_colored_state_stricter_than_occupancy_state(self):
        rows = [list(row) for row in self.state.board.rows]
        c, r = next((c, r) for r, row in enumerate(rows) for c, cell in enumerate(row) if cell.occupied)
        rows[r][c] = replace(rows[r][c], semantic="I" if rows[r][c].semantic != "I" else "O")
        wrong = replace(self.state, board=replace(self.state.board, rows=tuple(map(tuple, rows))))
        self.assertTrue(exact_state(self.state, wrong))
        self.assertFalse(exact_state(self.state, wrong, colored=True))

    def test_ambiguous_inputs_are_reported_separately(self):
        target = render(sample_scene(31), Appearance(locked_alpha=0)).state
        report = evaluate([Example("two", "group-two", "hidden", target)], {"two": self.state})
        self.assertIsNone(report["exact_rate"])
        self.assertEqual(report["accepted_ambiguous"], 1)

    def test_duplicate_and_extra_ids_refuse(self):
        for examples, predictions in (([self.example, self.example], {}),
                                       ([self.example], {"bogus": self.state})):
            with self.assertRaises(ValueError):
                evaluate(examples, predictions)

    def test_success_and_interval_have_no_fake_release_claim(self):
        report = evaluate([self.example], {"one": self.state})
        self.assertEqual(report["exact_rate"], 1)
        self.assertLess(report["image_wilson_95"][0], .3)
        self.assertIn("no cluster-aware release claim", report["limitation"])


if __name__ == "__main__":
    unittest.main()
