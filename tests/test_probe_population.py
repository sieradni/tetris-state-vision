"""The widened deciding population is declared once and cannot drift."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

import probe_population


class PopulationTests(unittest.TestCase):
    def test_only_the_recorded_or_canonical_population_is_admissible(self):
        self.assertEqual(probe_population.require_training_population(3, 3), 3)
        self.assertEqual(probe_population.require_training_population(3, 12), 12)
        for recorded, evaluated in ((3, 5), (3, 13), (2, 12), (3, 0), (3.0, 12), (3, True)):
            with self.assertRaises(ValueError):
                probe_population.require_training_population(recorded, evaluated)

    def test_the_extension_is_the_next_twenty_seven_stream_drawings(self):
        specs = probe_population.training_extension_specs(20261026)
        self.assertEqual(len(specs), 27)
        self.assertEqual((specs[0]["group_index"], specs[0]["view"]), (3, 0))
        self.assertEqual((specs[-1]["group_index"], specs[-1]["view"]), (11, 2))
        self.assertEqual({spec["group_index"] for spec in specs}, set(range(3, 12)))
        self.assertEqual({spec["view"] for spec in specs}, {0, 1, 2})
        self.assertEqual({spec["base_seed"] for spec in specs}, {20261026})

    def test_the_extension_is_disjoint_from_the_recorded_groups(self):
        specs = probe_population.training_extension_specs(20261026)
        recorded = {(group, view) for group in range(3) for view in range(3)}
        extended = {(spec["group_index"], spec["view"]) for spec in specs}
        self.assertFalse(recorded & extended)
        self.assertEqual(len(recorded | extended), 36)

    def test_the_recorded_population_is_a_prefix_and_reports_no_extension(self):
        self.assertEqual(probe_population.training_extension_specs(1, 3, 3), ())
        self.assertEqual(len(probe_population.training_extension_specs(1)), 27)

    def test_malformed_seeds_and_bounds_are_refused(self):
        for call in (lambda: probe_population.training_extension_specs(-1),
                     lambda: probe_population.training_extension_specs(2**31),
                     lambda: probe_population.training_extension_specs(1.0),
                     lambda: probe_population.training_extension_specs(True)):
            with self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
