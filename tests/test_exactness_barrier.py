"""The M24 exactness barrier: the instrument is controlled, and the record's claim is held to the evidence.

`docs/M24_EXACTNESS_BARRIER.md` states that no level and no projection the record contains can make any board exact,
because the miss is ordering rather than level. A negative claim of that shape is only worth the instrument behind it, so
this file first proves the checker finds exactness when it exists and loses it when a single pair is inverted, and only
then holds the committed decomposition to its own numbers.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT/"src", ROOT/"scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from decompose_exactness import exact_level_search, projected_exactness

SCOPE = ROOT/"docs/evidence/m24-exactness-r1/scope.json"
ARMS = ("stencil", "centred", "norm", "norm-cross", "envelope", "control")
ROWS, COLUMNS = 4, 3


def board(values):
    return [values[index*COLUMNS:(index+1)*COLUMNS] for index in range(ROWS)]


def labels_of(values):
    return [value > 0 for value in values]


class CheckerControl(unittest.TestCase):
    """The checker must find exactness when a level exists, and lose it for one inverted pair."""

    def separable(self):
        """Two occupied cells and the rest empty, with every occupied cell above every empty one."""
        values = [-2.0, -1.5, -1.0,
                  -0.8, -0.4, -0.2,
                  0.3, 0.9, 1.4,
                  1.8, 2.2, 2.6]
        return values

    def test_a_separable_board_is_exact_at_some_level(self):
        values = self.separable()
        result = exact_level_search(board(values), board(labels_of(values)))
        self.assertTrue(result["exact_at_any_level"])
        self.assertEqual(result["fewest_wrong_cells"], 0)

    def test_a_single_inverted_pair_makes_exactness_impossible(self):
        """One empty cell above one occupied cell: no level can separate them, so no level is exact."""
        values = self.separable()
        labels = board(labels_of(values))                  # labels stay what they were; only the read moves
        values[10], values[1] = values[1], values[10]      # an occupied cell sinks below an empty one
        result = exact_level_search(board(values), labels)
        self.assertFalse(result["exact_at_any_level"])
        self.assertGreaterEqual(result["fewest_wrong_cells"], 1)

    def test_a_constant_offset_changes_no_verdict(self):
        values = self.separable()
        for offset in (-100.0, 0.0, 250.0):
            shifted = [value+offset for value in values]
            result = exact_level_search(board(shifted), board(labels_of(shifted)))
            self.assertTrue(result["exact_at_any_level"], offset)

    def test_the_search_covers_every_level_that_can_change_a_decision(self):
        """Distinct values plus one cut either side is exactly the set of outcomes, so the search is complete."""
        values = self.separable()
        result = exact_level_search(board(values), board(labels_of(values)))
        self.assertEqual(result["levels_searched"], len(set(values))+1)

    def test_a_projection_can_move_a_cell_but_cannot_invent_separation(self):
        """The monotone envelope keeps the inversion, so the projected read is still not exact at any level."""
        values = self.separable()
        labels = board(labels_of(values))
        values[10], values[1] = values[1], values[10]
        projected = projected_exactness(board(values), labels, ROWS, COLUMNS)
        self.assertFalse(projected["raw"]["exact_at_any_level"])
        self.assertFalse(projected["monotone"]["exact_at_any_level"])
        self.assertFalse(projected["monotone_then_crossing"]["exact_at_any_level"])


@unittest.skipUnless(SCOPE.exists(), "the M24 decomposition has not run yet")
class RecordTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = json.loads(SCOPE.read_text(encoding="utf-8"))

    def test_every_arm_was_decomposed(self):
        self.assertEqual(sorted(self.scope["arms"]), sorted(ARMS))
        for name in ARMS:
            block = self.scope["arms"][name]
            self.assertEqual(block["boards"], 28, name)
            self.assertEqual(len(block["per_board"]), 28, name)

    def test_no_board_is_exact_at_any_level_in_any_arm(self):
        for name in ARMS:
            self.assertEqual(self.scope["arms"][name]["boards_exact_at_some_exact_level"], 0, name)
            for record in self.scope["arms"][name]["per_board"]:
                self.assertFalse(record["exact_level_search"]["exact_at_any_level"], (name, record["sample_id"]))
                self.assertGreaterEqual(record["exact_level_search"]["fewest_wrong_cells"], 1)

    def test_no_projection_makes_a_board_exact(self):
        for name in ARMS:
            projections = self.scope["arms"][name]["boards_exact_after_a_projection"]
            self.assertEqual(projections, {"raw": 0, "monotone": 0, "monotone_then_crossing": 0}, name)

    def test_the_miss_is_ordering_and_the_deficit_is_recorded(self):
        for name in ARMS:
            block = self.scope["arms"][name]
            self.assertGreater(block["median_inversions"], 0.0, name)
            self.assertGreater(block["mean_inversion_rate"], 0.1, name)
            self.assertLess(block["mean_auc_within_board"], 0.9, name)
            for record in block["per_board"]:
                self.assertGreater(record["inversions"], 0, (name, record["sample_id"]))

    def test_the_miss_is_dominated_by_false_positives(self):
        for name in ARMS:
            block = self.scope["arms"][name]
            self.assertGreater(block["mean_false_positive"], 10*max(block["mean_false_negative"], 0.5), name)

    def test_the_decomposition_agrees_with_the_curated_counts(self):
        """The two instruments must agree on the record: 0 exact boards is the curator's number too."""
        catalogue = json.loads((ROOT/"docs/evidence/m23-contrast-r1/scope.json").read_text(encoding="utf-8"))
        for name in ARMS:
            self.assertEqual(catalogue["arms"][name]["exact_boards_at_zero"]["exact"],
                             self.scope["arms"][name]["exact_boards_at_zero"], name)


if __name__ == "__main__":
    unittest.main()
