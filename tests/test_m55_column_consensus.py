"""Synthetic controls for M55's column certification and pitch-span rule.

`docs/M55_COLUMN_CONSENSUS.md` declares the instrument: the pitch is used only where the anchors span at least
`MIN_PITCH_SPAN_CELLS` cells, the columns are *certified* when at least `MIN_COLUMN_BLOCKS` components agree on both edge
phases within `COLUMN_TOLERANCE` and inside the drawing band `COLUMN_INSET_MAX`, and the horizontal frame is never
shifted. These tests pin each clause on rasters of known pitch, inset and origin, and assert that the vertical axis is
M54's unchanged.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from column_consensus import (  # noqa: E402
    COLUMN_INSET_MAX, COLUMN_TOLERANCE, CONSTANTS, MIN_COLUMN_BLOCKS, MIN_PITCH_SPAN_CELLS, anchor_lattice,
    column_consensus,
)

PITCH = 17.0
LEFT = 40.0
TOP = 30.0
WIDTH, HEIGHT = 260, 430


def raster(cells, pitch=PITCH, left=LEFT, top=TOP, inset=2.0):
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for column, row in cells:
        x0 = int(round(left + column*pitch + inset))
        x1 = int(round(left + (column+1)*pitch - inset))
        y0 = int(round(top + row*pitch))
        y1 = int(round(top + (row+1)*pitch))
        canvas[y0:y1, x0:x1] = (255, 40, 40)
    return canvas


def box(pitch=PITCH, left=LEFT, top=TOP, bottom=None):
    bottom = top + 20*pitch if bottom is None else bottom
    return {"origin": [left, bottom], "u": [pitch, 0.0], "v": [0.0, -pitch],
            "frame": {"left": left, "right": left + 10*pitch, "top": top, "bottom": bottom}}


# four separated columns, floor shared by all of them: the stack both certifies the columns and pins the floor
WIDE = ([(0, row) for row in (16, 17, 18, 19)] + [(2, row) for row in (17, 18, 19)]
        + [(4, row) for row in (15, 16, 17, 18, 19)] + [(7, row) for row in (18, 19)])
# one merged 2x2 block: one component spanning two cells, below both the consensus and the pitch-span rule
NARROW = [(0, 18), (1, 18), (0, 19), (1, 19)]


class CertificationTest(unittest.TestCase):
    def test_components_agreeing_inside_the_inset_certify_the_columns(self):
        found = [{"left": LEFT + 2.0, "right": LEFT + 3*PITCH - 2.0},
                 {"left": LEFT + 4*PITCH + 2.0, "right": LEFT + 5*PITCH - 2.0},
                 {"left": LEFT + 7*PITCH + 2.0, "right": LEFT + 8*PITCH - 2.0}]
        report = column_consensus(found, PITCH, LEFT)
        self.assertTrue(report["kept"])
        self.assertEqual(report["shift"], 0)
        self.assertLessEqual(report["left_phase"], COLUMN_INSET_MAX)
        self.assertGreaterEqual(report["right_phase"], 1.0-COLUMN_INSET_MAX)

    def test_one_component_cannot_certify(self):
        found = [{"left": LEFT + 2.0, "right": LEFT + PITCH - 2.0}]
        report = column_consensus(found, PITCH, LEFT)
        self.assertFalse(report["kept"])
        self.assertEqual(report["shift"], 0)
        self.assertIn("too few components", report["reason"])

    def test_phases_disagreeing_beyond_tolerance_fall_back(self):
        found = [{"left": LEFT + 1.0, "right": LEFT + PITCH - 1.0},
                 {"left": LEFT + 3*PITCH + 0.5*PITCH, "right": LEFT + 4*PITCH + 0.5*PITCH},
                 {"left": LEFT + 6*PITCH + 0.9*PITCH, "right": LEFT + 7*PITCH + 0.9*PITCH}]
        report = column_consensus(found, PITCH, LEFT)
        self.assertFalse(report["kept"])
        self.assertGreater(report["left_spread"], COLUMN_TOLERANCE)
        self.assertEqual(report["shift"], 0)

    def test_a_phase_outside_the_drawing_band_is_not_resolvable(self):
        # both edges a fifth of a cell into the cell from the left: the stack is between columns, so no whole-column
        # decision either keeps or moves this frame - the band must refuse it
        found = [{"left": LEFT + 0.40*PITCH, "right": LEFT + 1.40*PITCH},
                 {"left": LEFT + 3.40*PITCH, "right": LEFT + 4.40*PITCH}]
        report = column_consensus(found, PITCH, LEFT)
        self.assertFalse(report["kept"])
        self.assertIn("outside the drawing band", report["reason"])

    def test_no_blocks_is_the_declared_fallback(self):
        report = column_consensus([], PITCH, LEFT)
        self.assertFalse(report["kept"])
        self.assertEqual(report["reason"], "no visible blocks")
        self.assertEqual((MIN_COLUMN_BLOCKS, COLUMN_TOLERANCE, COLUMN_INSET_MAX), (2, 0.15, 0.20))
        self.assertIn("min_pitch_span_cells", CONSTANTS)


class LatticeTest(unittest.TestCase):
    def test_the_horizontal_frame_is_m51s_unchanged(self):
        source = box()
        result = anchor_lattice(raster(WIDE), source)
        self.assertEqual(result["origin"][0], source["frame"]["left"])
        self.assertEqual(result["u"][0], PITCH)
        self.assertEqual(result["fit"]["x"]["correction"]["columns"], 0)
        self.assertFalse(result["fit"]["x"]["correction"]["applied"])
        self.assertTrue(result["fit"]["x"]["columns"]["kept"])

    def test_a_narrow_stack_cannot_pin_the_pitch(self):
        result = anchor_lattice(raster(NARROW), box())
        self.assertFalse(result["fit"]["pitch_pinned"]["x"])
        self.assertLess(result["fit"]["span_cells"]["x"], MIN_PITCH_SPAN_CELLS)
        self.assertEqual(result["u"][0], PITCH)
        self.assertEqual(result["origin"][0], LEFT)

    def test_a_wide_stack_pins_the_pitch_it_fits(self):
        result = anchor_lattice(raster(WIDE), box())
        self.assertTrue(result["fit"]["pitch_pinned"]["x"])
        self.assertAlmostEqual(result["u"][0], PITCH, delta=0.25)

    def test_gridless_input_keeps_the_box_on_both_axes(self):
        source = box()
        result = anchor_lattice(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8), source)
        self.assertFalse(result["fit"]["x"]["kept"])
        self.assertFalse(result["fit"]["y"]["kept"])
        self.assertEqual(result["u"], [PITCH, 0.0])
        self.assertEqual(result["origin"], source["origin"])
        self.assertEqual(result["source"], "column-consensus")

    def test_one_block_falls_back_on_both_axes(self):
        source = box()
        result = anchor_lattice(raster([(2, 19)]), source)
        self.assertFalse(result["fit"]["x"]["columns"]["kept"])
        self.assertFalse(result["fit"]["y"]["kept"])
        self.assertEqual(result["origin"], source["origin"])

    def test_the_vertical_axis_is_m54s_unchanged(self):
        from floor_consensus import anchor_lattice as m54_lattice
        image = raster(WIDE)
        source = box(bottom=TOP + 19*PITCH)   # a box one row high: M54's consensus floor must still correct it
        ours = anchor_lattice(image, source)
        theirs = m54_lattice(image, source)
        self.assertEqual(ours["fit"]["y"]["rows_corrected"], 1)
        self.assertEqual(ours["fit"]["y"]["rows_corrected"], theirs["fit"]["y"]["rows_corrected"])
        self.assertAlmostEqual(ours["origin"][1], theirs["origin"][1], places=9)
        self.assertAlmostEqual(ours["origin"][1], TOP + 20*PITCH, delta=0.5)

    def test_an_unpinned_floor_still_keeps_m51s_bottom(self):
        source = box()
        result = anchor_lattice(raster([(0, 12), (5, 15)]), source)
        self.assertFalse(result["fit"]["y"]["floor"]["kept"])
        self.assertEqual(result["fit"]["y"]["rows_corrected"], 0)
        self.assertAlmostEqual(result["origin"][1], source["origin"][1], delta=1e-9)


if __name__ == "__main__":
    unittest.main()
