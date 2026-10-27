from collections import Counter
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from m2_interleaved_order import cycle_order, distinct_groups, interleaved_spec


class InterleavedOrderTests(unittest.TestCase):
    def test_every_admitted_cycle_has_identical_complete_pair_population(self):
        expected = {(index, view) for index in range(16) for view in (0, 1)}
        for cycle in range(128):
            order = cycle_order(cycle)
            self.assertEqual(len(order), 32)
            self.assertEqual(set(order), expected)
            self.assertEqual(Counter(index // 2 for index, _ in order), {label: 4 for label in range(8)})
            self.assertEqual(Counter(index for index, _ in order), {index: 2 for index in range(16)})

    def test_every_eight_position_round_has_every_class_and_absence(self):
        for cycle in range(128):
            order = cycle_order(cycle)
            for start in (0, 8, 16, 24):
                self.assertEqual({index // 2 for index, _ in order[start:start + 8]}, set(range(8)))

    def test_arbitrary_resume_cursors_reproduce_global_order(self):
        expected = tuple(pair for cycle in range(128) for pair in cycle_order(cycle))
        actual = tuple(interleaved_spec(step) for step in range(1, 4097))
        self.assertEqual(actual, expected)
        for cursor in (0, 1, 7, 8, 31, 32, 63, 64, 127, 128, 640, 1023, 1024, 4095, 4096):
            self.assertEqual(actual[:cursor] + tuple(interleaved_spec(step) for step in range(cursor + 1, 4097)), expected)

    def test_repeated_queries_consume_no_global_randomness_and_are_not_one_fixed_cycle(self):
        state = random.getstate()
        first = cycle_order(0)
        _ = cycle_order(1), cycle_order(127), interleaved_spec(1024)
        self.assertEqual(cycle_order(0), first)
        self.assertEqual(random.getstate(), state)
        self.assertNotEqual(cycle_order(0), cycle_order(1))
        self.assertGreater(len({cycle_order(cycle)[-1][0] // 2 for cycle in range(32)}), 1)

    def test_ancestry_count_uses_actual_prefix_without_inventing_diversity(self):
        all_pairs = tuple(pair for cycle in range(128) for pair in cycle_order(cycle))
        for cursor in (*range(65), 128, 640, 1024, 4096):
            self.assertEqual(distinct_groups(cursor), len({index for index, _ in all_pairs[:cursor]}))

    def test_invalid_boolean_fractional_or_unbounded_cursors_refuse(self):
        for function, values in ((cycle_order, (True, -1, 128, 1.0, "0", None)),
                                 (interleaved_spec, (True, 0, -1, 4097, 1.0, "1", None)),
                                 (distinct_groups, (True, -1, 4097, 1.0, "0", None))):
            for value in values:
                with self.subTest(function=function.__name__, value=value), self.assertRaises(ValueError):
                    function(value)

    def test_reordered_pairs_retain_actual_pixels_public_labels_and_recipes(self):
        from m2_positive_control_data import sample_control
        baseline = {(index, view): sample_control(index, view).record for index in range(16) for view in (0, 1)}
        for cycle in (0, 31, 127):
            replay = {pair: sample_control(*pair).record for pair in cycle_order(cycle)}
            self.assertEqual(replay, baseline)


if __name__ == "__main__":
    unittest.main()
