"""The declared placement measurement: offsets, scale, decomposition and the four-condition rule."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from placement_error import (ATTRIBUTION_FLOOR, PLACEMENT_IMPROVEMENT, decomposition, relative_improvement,
                             cell_offsets, score, stats, summarize, verdict)
from registration_attribution import map_cells


def lattice(origin=(0.1, 0.9), u=(0.08, 0.0), v=(0.0, -0.042)):
    return {"origin": origin, "u": u, "v": v}


def shifted(dx=0.0, dy=0.0, *, scale=(1.0, 1.0), origin=(0.1, 0.9)):
    base = lattice()
    u = (base["u"][0]*scale[0], 0.0)
    v = (0.0, base["v"][1]*scale[1])
    return {"origin": (origin[0]+dx*base["u"][0], origin[1]+dy*base["v"][1]), "u": u, "v": v}


def case(predicted, target, *, occupied=None, blank=False):
    cells = 200
    occupied = [0]*cells if blank else (occupied or [1 if index % 2 else 0 for index in range(cells)])
    return {"sample_id": "training-20261026-v0", "predicted": predicted, "target": target,
            "offsets": [list(pair) for pair in cell_offsets(predicted, target)],
            "mapping": map_cells(predicted, target)[0], "known": [True]*cells,
            "occupied": occupied,
            "logits": [1.0 if value else -1.0 for value in occupied]}


class OffsetTests(unittest.TestCase):
    def test_identical_lattices_have_no_offset(self):
        offsets = cell_offsets(lattice(), lattice())
        self.assertTrue(all(abs(a) < 1e-9 and abs(b) < 1e-9 for a, b in offsets))

    def test_a_one_cell_right_shift_is_a_positive_column_offset(self):
        offsets = cell_offsets(shifted(dx=1.0), lattice())
        self.assertTrue(all(abs(a-1.0) < 1e-9 and abs(b) < 1e-9 for a, b in offsets))

    def test_a_one_cell_down_shift_is_a_positive_row_offset(self):
        offsets = cell_offsets(shifted(dy=1.0), lattice())
        self.assertTrue(all(abs(a) < 1e-9 and abs(b-1.0) < 1e-9 for a, b in offsets))

    def test_scale_error_grows_towards_the_far_edge(self):
        offsets = cell_offsets(shifted(scale=(1.1, 1.1)), lattice())
        self.assertAlmostEqual(offsets[0][0], 0.05, places=6)
        self.assertGreater(offsets[-1][0], 0.95)

    def test_stats_separate_translation_from_scale(self):
        translated = stats([case(shifted(dx=1.0), lattice())])
        self.assertAlmostEqual(translated["signed_column_offset_cells"], 1.0, places=6)
        self.assertAlmostEqual(translated["mean_scale_ratio_u"], 1.0, places=6)
        scaled = stats([case(shifted(scale=(1.1, 1.0)), lattice())])
        self.assertAlmostEqual(scaled["mean_scale_ratio_u"], 1.1, places=6)
        self.assertAlmostEqual(scaled["signed_column_offset_cells"], 0.5, places=6)


class PopulationTests(unittest.TestCase):
    def test_the_full_population_is_scored_not_a_co_scoreable_subset(self):
        good = case(lattice(), lattice())
        blank = case(lattice(), lattice(), blank=True)
        measured = score([good, blank])
        self.assertEqual((measured["images"], measured["empty_images"]), (1, 1))
        self.assertEqual(measured["known_cells"], 200)
        self.assertEqual(measured["occupied_cells"], 100)
        self.assertEqual(measured["f1"], 1.0)

    def test_summarize_reports_full_population_and_placement(self):
        measured = summarize([case(shifted(dx=1.0), lattice())])
        self.assertEqual(measured["occupancy"]["known_cells"], 200)
        self.assertEqual(measured["ranking"], "informative")
        # The metric means over both cell axes, so a pure one-column shift is half a cell.
        self.assertAlmostEqual(measured["placement"]["mean_abs_offset_cells"], 0.5, places=6)
        # A one-cell shift leaves the last column off the board, so 9 of 10 columns move.
        self.assertAlmostEqual(measured["placement"]["shifted_share"], 0.9, places=6)


class RuleTests(unittest.TestCase):
    def test_relative_improvement_signs(self):
        self.assertAlmostEqual(relative_improvement(2.0, 1.5), .25)
        self.assertLess(relative_improvement(2.0, 2.5), 0)
        with self.assertRaises(ValueError):
            relative_improvement(0.0, 1.0)

    def test_all_four_conditions_are_required(self):
        occupancy = {"f1": .75}
        cleared = verdict(occupancy, "informative", .08, .30)
        self.assertEqual(cleared["verdict"], "cleared")
        self.assertFalse(verdict(occupancy, "informative", .08, .24)["conditions"]["placement"])
        self.assertFalse(verdict(occupancy, "informative", .04, .30)["conditions"]["attribution"])
        self.assertFalse(verdict(occupancy, "uninformative", .08, .30)["conditions"]["ranking"])
        self.assertFalse(verdict({"f1": .69}, "informative", .08, .30)["conditions"]["per_cell_f1"])

    def test_boundaries_count_as_met(self):
        cleared = verdict({"f1": .70}, "informative", ATTRIBUTION_FLOOR, PLACEMENT_IMPROVEMENT)
        self.assertEqual(cleared["verdict"], "cleared")

    def test_decomposition_precedence(self):
        base = {"signed_column_offset_cells": 0.1, "signed_row_offset_cells": 0.1,
                "mean_scale_ratio_u": 1.0, "mean_scale_ratio_v": 1.0,
                "board_error_spread_cells": 0.2}
        self.assertEqual(decomposition(base, 0.0)["rule"], "geometry_capacity")
        self.assertEqual(decomposition(dict(base, board_error_spread_cells=2.0), 0.0)["rule"], "dispersion")
        self.assertEqual(decomposition(dict(base, mean_scale_ratio_u=1.2), 0.0)["rule"], "scale")
        self.assertEqual(decomposition(dict(base, signed_row_offset_cells=1.2), 0.0)["rule"], "translation")
        self.assertEqual(decomposition(base, 0.4)["rule"], "read-out_inheritance")
        self.assertEqual(decomposition(dict(base, signed_column_offset_cells=-1.5), 0.9)["rule"],
                         "translation")

    def test_every_rule_names_a_next_suspect(self):
        base = {"signed_column_offset_cells": 0.0, "signed_row_offset_cells": 0.0,
                "mean_scale_ratio_u": 1.0, "mean_scale_ratio_v": 1.0,
                "board_error_spread_cells": 0.0}
        for placement, delta in ((base, 0.0), (base, 0.5),
                                 (dict(base, board_error_spread_cells=3.0), 0.0),
                                 (dict(base, mean_scale_ratio_v=0.85), 0.0),
                                 (dict(base, signed_column_offset_cells=1.1), 0.0)):
            self.assertTrue(decomposition(placement, delta)["next_suspect"])


if __name__ == "__main__":
    unittest.main()
