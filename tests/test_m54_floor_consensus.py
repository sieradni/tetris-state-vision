"""Synthetic controls for M54's floor-consensus lattice.

`docs/M54_FLOOR_CONSENSUS.md` declares the instrument: keep M53's anchors, decide the row correction from a bottom row
that at least `MIN_FLOOR_BLOCKS` components share within `FLOOR_AGREEMENT` cells, apply it **one-sided** (only when the
pinned row is below the box's bottom), and fall back to M51's box otherwise. These tests pin each clause on rasters whose
pitch, origin and floor are known exactly, so the arm is judged on geometry rather than on the population it came from.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from floor_consensus import (  # noqa: E402
    CONSTANTS, FLOOR_AGREEMENT, MIN_FLOOR_BLOCKS, anchor_lattice, floor_consensus, row_correction,
)

PITCH = 17.0
LEFT = 40.0
TOP = 30.0
WIDTH, HEIGHT = 260, 430


def raster(cells, pitch=PITCH, left=LEFT, top=TOP, shift=0.0):
    """Board-sized image with the given `(column, row)` cells drawn flush to their cell boundaries."""
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for column, row in cells:
        x0 = int(round(left + column*pitch))
        x1 = int(round(left + (column+1)*pitch))
        y0 = int(round(top + row*pitch + shift))
        y1 = int(round(top + (row+1)*pitch + shift))
        canvas[y0:y1, x0:x1] = (255, 40, 40)
    return canvas


def box(pitch=PITCH, left=LEFT, top=TOP, bottom=None):
    bottom = top + 20*pitch if bottom is None else bottom
    return {"origin": [left, bottom], "u": [pitch, 0.0], "v": [0.0, -pitch],
            "frame": {"left": left, "right": left + 10*pitch, "top": top, "bottom": bottom}}


# staggered heights, so the vertical edges pin three row boundaries and the floor is shared by both components
TWO_COLUMNS_FULL = [(0, row) for row in (16, 17, 18, 19)] + [(3, row) for row in (17, 18, 19)]
TWO_COLUMNS_SHORT = [(0, row) for row in (15, 16, 17, 18)] + [(3, row) for row in (16, 17, 18)]
STAGGERED = [(0, row) for row in (16, 17, 18, 19)] + [(3, row) for row in (15, 16, 17, 18)]


class ConsensusTest(unittest.TestCase):
    def test_two_components_at_one_bottom_are_pinned(self):
        found = [{"bottom": 400.0}, {"bottom": 400.4}, {"bottom": 350.0}]
        floor = floor_consensus(found, PITCH)
        self.assertTrue(floor["kept"])
        self.assertEqual(floor["members"], 2)
        self.assertAlmostEqual(floor["bottom"], 400.2, places=6)

    def test_one_component_is_not_a_floor(self):
        floor = floor_consensus([{"bottom": 400.0}], PITCH)
        self.assertFalse(floor["kept"])
        self.assertEqual(floor["members"], 1)

    def test_components_beyond_the_agreement_band_disagree(self):
        # 4 px apart on a 17 px pitch is 0.235 cells: beyond the declared 0.15 band, so nothing is pinned
        self.assertGreater(4.0/PITCH, FLOOR_AGREEMENT)
        floor = floor_consensus([{"bottom": 400.0}, {"bottom": 404.0}], PITCH)
        self.assertFalse(floor["kept"])
        self.assertEqual(floor["clusters"], [1, 1])

    def test_within_the_agreement_band_they_agree(self):
        self.assertLess(2.0/PITCH, FLOOR_AGREEMENT)
        floor = floor_consensus([{"bottom": 400.0}, {"bottom": 402.0}], PITCH)
        self.assertTrue(floor["kept"])

    def test_the_lowest_pinned_row_wins_not_the_most_populated(self):
        found = [{"bottom": 300.0}, {"bottom": 300.0}, {"bottom": 300.0}, {"bottom": 400.0}, {"bottom": 400.0}]
        floor = floor_consensus(found, PITCH)
        self.assertTrue(floor["kept"])
        self.assertAlmostEqual(floor["bottom"], 400.0, places=6)

    def test_no_blocks_is_a_declared_fallback(self):
        floor = floor_consensus([], PITCH)
        self.assertFalse(floor["kept"])
        self.assertEqual(floor["reason"], "no visible blocks")
        self.assertEqual((MIN_FLOOR_BLOCKS, FLOOR_AGREEMENT), (2, 0.15))
        self.assertIn("min_floor_blocks", CONSTANTS)


class CorrectionTest(unittest.TestCase):
    def test_pinned_ink_below_the_box_moves_the_frame_down(self):
        floor = {"kept": True, "bottom": TOP + 20*PITCH}
        rows, why = row_correction(floor, TOP + 19*PITCH, PITCH)
        self.assertEqual(rows, 1)
        self.assertIsNone(why)

    def test_agreement_moves_nothing(self):
        floor = {"kept": True, "bottom": TOP + 20*PITCH}
        self.assertEqual(row_correction(floor, TOP + 20*PITCH, PITCH)[0], 0)

    def test_a_pinned_row_above_the_box_never_raises_the_frame(self):
        floor = {"kept": True, "bottom": TOP + 19*PITCH}
        rows, why = row_correction(floor, TOP + 20*PITCH, PITCH)
        self.assertEqual(rows, 0)
        self.assertIn("bottom row is empty", why)

    def test_an_unpinned_floor_moves_nothing(self):
        rows, why = row_correction({"kept": False, "reason": "no row is pinned"}, 400.0, PITCH)
        self.assertEqual(rows, 0)
        self.assertEqual(why, "no row is pinned")


class AnchoredLatticeTest(unittest.TestCase):
    def test_a_pinned_floor_below_a_high_box_is_corrected(self):
        result = anchor_lattice(raster(TWO_COLUMNS_FULL), box(bottom=TOP + 19*PITCH))
        self.assertEqual(result["fit"]["y"]["rows_corrected"], 1)
        self.assertAlmostEqual(result["origin"][1], TOP + 20*PITCH, delta=0.5)
        self.assertTrue(result["fit"]["y"]["correction"]["applied"])

    def test_one_block_per_row_is_a_fallback(self):
        # two components whose floors disagree by a whole row: no row is pinned twice, so nothing moves
        source = box(bottom=TOP + 20*PITCH)
        result = anchor_lattice(raster(STAGGERED), source)
        self.assertFalse(result["fit"]["y"]["floor"]["kept"])
        self.assertEqual(result["fit"]["y"]["rows_corrected"], 0)
        self.assertAlmostEqual(result["origin"][1], source["origin"][1], delta=1e-9)

    def test_a_single_block_falls_back_to_the_box(self):
        # one component pins no pitch pairs at all, which is the earlier, declared fallback
        source = box()
        result = anchor_lattice(raster([(0, 19)]), source)
        self.assertFalse(result["fit"]["y"]["kept"])
        self.assertEqual(result["fit"]["y"]["rows_corrected"], 0)
        self.assertAlmostEqual(result["origin"][1], source["origin"][1], delta=1e-9)

    def test_an_empty_bottom_row_is_not_a_correction(self):
        source = box()
        result = anchor_lattice(raster(TWO_COLUMNS_SHORT), source)
        self.assertTrue(result["fit"]["y"]["floor"]["kept"])
        self.assertEqual(result["fit"]["y"]["rows_corrected"], 0)
        self.assertAlmostEqual(result["origin"][1], source["origin"][1], delta=1e-9)
        self.assertIn("bottom row is empty", result["fit"]["y"]["correction"]["reason"])

    def test_gridless_input_keeps_the_box_on_both_axes(self):
        source = box()
        result = anchor_lattice(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8), source)
        self.assertFalse(result["fit"]["x"]["kept"])
        self.assertFalse(result["fit"]["y"]["kept"])
        self.assertEqual(result["u"], [PITCH, 0.0])
        self.assertEqual(result["origin"], source["origin"])
        self.assertEqual(result["source"], "floor-consensus")

    def test_the_horizontal_axis_is_m53s_unchanged(self):
        from stack_anchor import anchor_lattice as m53_lattice
        image = raster(TWO_COLUMNS_FULL)
        source = box(bottom=TOP + 19*PITCH)
        ours = anchor_lattice(image, source)
        theirs = m53_lattice(image, source)
        self.assertAlmostEqual(ours["u"][0], theirs["u"][0], places=9)
        self.assertAlmostEqual(ours["origin"][0], theirs["origin"][0], places=9)


if __name__ == "__main__":
    unittest.main()
