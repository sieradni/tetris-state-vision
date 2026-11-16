"""Synthetic-stack controls for M53's stack-anchored lattice.

`docs/M53_STACK_ANCHOR.md` declares the instrument: block edges are absolute cell boundaries, the pitch is a median over
edge spacings, the vertical origin is the stack's floor moved by whole cells, the horizontal origin is the box's left
snapped to the anchors' phase, and any axis with too few anchors keeps M51's box. These tests pin each clause with a
raster whose pitch and origin are known exactly, so the instrument is judged on numbers, not on the population it was
tuned against.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from stack_anchor import (  # noqa: E402
    CONSTANTS, FLOOR_TOLERANCE, MIN_ANCHORS, MIN_PAIRS, anchor_lattice, anchors, blocks, circular_median, components,
    pitch_from,
)

PITCH = 17.0
LEFT = 40.0
TOP = 30.0
INSET = 0.0
WIDTH, HEIGHT = 260, 430


def raster(cells, pitch=PITCH, left=LEFT, top=TOP, inset=INSET, jitter=None, colour=(255, 40, 40)):
    """A board-sized image with the given `(column, row)` cells drawn as inset rectangles."""
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for index, (column, row) in enumerate(cells):
        shift_x, shift_y = (0.0, 0.0) if jitter is None else jitter[index]
        x0 = int(round(left + column*pitch + inset + shift_x))
        x1 = int(round(left + (column+1)*pitch - inset + shift_x))
        y0 = int(round(top + row*pitch + inset + shift_y))
        y1 = int(round(top + (row+1)*pitch - inset + shift_y))
        canvas[y0:y1, x0:x1] = colour
    return canvas


def box(pitch=PITCH, left=LEFT, top=TOP, bottom=None):
    """M51's detected-box shape, by default exactly the true lattice."""
    bottom = top + 20*pitch if bottom is None else bottom
    return {"origin": [left, bottom], "u": [pitch, 0.0], "v": [0.0, -pitch],
            "frame": {"left": left, "right": left + 10*pitch, "top": top, "bottom": bottom}}


# two stack heights, so the vertical edges pin three row boundaries and the floor is the lowest of them
STACK = ([(column, row) for row in (16, 17, 18, 19) for column in (0, 1)]
         + [(column, row) for row in (18, 19) for column in (7, 8)])


class SegmentsTest(unittest.TestCase):
    def test_a_rectangle_is_one_component_with_cell_edges(self):
        found = components(raster([(2, 18)], inset=2.0)[:, :, 0] > 100)
        self.assertEqual(len(found), 1)
        x0, x1, y0, y1 = found[0]
        self.assertAlmostEqual(x0, LEFT + 2*PITCH + 2.0, delta=1.0)
        self.assertAlmostEqual(x1 + 1, LEFT + 3*PITCH - 2.0, delta=1.0)

    def test_touching_blocks_merge_and_are_still_kept(self):
        image = raster([(3, 18), (3, 19), (4, 18), (4, 19)], inset=0.0)
        found = blocks(image, PITCH, PITCH, box()["frame"])
        self.assertEqual(len(found), 1, "a merged 2x2 block must not be dropped for being large")
        self.assertGreater(found[0]["bottom"] - found[0]["top"], 1.5*PITCH)

    def test_components_outside_the_box_are_not_blocks(self):
        image = raster([(0, 18)])
        frame = box()["frame"]
        frame = {"left": frame["left"] + 6*PITCH, "right": frame["right"], "top": frame["top"],
                 "bottom": frame["bottom"]}
        self.assertEqual(blocks(image, PITCH, PITCH, frame), [])


class PitchTest(unittest.TestCase):
    def test_the_theil_sen_median_recovers_the_pitch(self):
        positions = [LEFT + i*PITCH for i in range(6)]
        fitted, pairs = pitch_from(positions, PITCH)
        self.assertAlmostEqual(fitted, PITCH, delta=1e-6)
        self.assertGreaterEqual(pairs, MIN_PAIRS)

    def test_one_displaced_anchor_cannot_move_the_median(self):
        positions = [LEFT + i*PITCH for i in range(6)] + [LEFT + 2*PITCH + 6.0]
        fitted, _ = pitch_from(positions, PITCH)
        self.assertAlmostEqual(fitted, PITCH, delta=0.2)

    def test_too_few_pairs_yields_no_pitch(self):
        fitted, pairs = pitch_from([100.0], PITCH)
        self.assertIsNone(fitted)
        self.assertLess(pairs, MIN_PAIRS)

    def test_the_circular_median_folds_onto_the_period(self):
        phase = circular_median([100.4, 117.2, 134.1], PITCH)
        self.assertAlmostEqual(phase, 100.4 % PITCH, delta=0.5)


class AnchoredLatticeTest(unittest.TestCase):
    def test_a_stack_on_the_floor_reproduces_pitch_and_origin(self):
        result = anchor_lattice(raster(STACK), box())
        self.assertAlmostEqual(result["u"][0], PITCH, delta=0.2)
        self.assertAlmostEqual(-result["v"][1], PITCH, delta=0.2)
        self.assertAlmostEqual(result["origin"][1], TOP + 20*PITCH, delta=1.5)
        self.assertEqual(result["fit"]["y"]["rows_corrected"], 0)
        self.assertTrue(result["fit"]["y"]["floor_consistent"])
        self.assertTrue(result["fit"]["x"]["kept"])

    def test_a_box_one_row_high_is_corrected_by_the_floor(self):
        high = box(bottom=TOP + 19*PITCH)
        result = anchor_lattice(raster(STACK), high)
        self.assertEqual(result["fit"]["y"]["rows_corrected"], 1)
        self.assertAlmostEqual(result["origin"][1], TOP + 20*PITCH, delta=1.5)

    def test_subpixel_jitter_keeps_the_pitch(self):
        jitter = [((index % 3 - 1)*0.55, (index % 2)*0.45) for index in range(len(STACK))]
        result = anchor_lattice(raster(STACK, jitter=jitter), box())
        self.assertAlmostEqual(result["u"][0], PITCH, delta=0.25)
        self.assertAlmostEqual(-result["v"][1], PITCH, delta=0.25)

    def test_a_floating_stack_does_not_move_the_floor(self):
        floating = ([(0, 10), (1, 10), (0, 11), (1, 11)]
                    + [(3, 12), (4, 12), (3, 13), (4, 13)])
        result = anchor_lattice(raster(floating), box())
        self.assertFalse(result["fit"]["y"]["floor_consistent"])
        self.assertEqual(result["fit"]["y"]["rows_corrected"], 0)
        self.assertAlmostEqual(result["origin"][1], TOP + 20*PITCH, delta=1e-9)

    def test_the_floor_guard_is_the_declared_tolerance(self):
        self.assertEqual(FLOOR_TOLERANCE, 1.0)
        self.assertIn("floor_tolerance", CONSTANTS)
        # one pitch above the floor is inside the tolerance band, five pitches is not
        self.assertLessEqual(1.0*PITCH, FLOOR_TOLERANCE*PITCH)
        self.assertGreater(5.0*PITCH, FLOOR_TOLERANCE*PITCH)


class FallbackTest(unittest.TestCase):
    def test_gridless_input_keeps_the_box_on_both_axes(self):
        source = box()
        result = anchor_lattice(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8), source)
        self.assertEqual(result["fit"]["blocks"], 0)
        self.assertFalse(result["fit"]["x"]["kept"])
        self.assertFalse(result["fit"]["y"]["kept"])
        self.assertEqual(result["u"], [PITCH, 0.0])
        self.assertEqual(result["origin"], source["origin"])

    def test_one_block_is_anchor_starved_on_both_axes(self):
        source = box()
        result = anchor_lattice(raster([(4, 19)]), source)
        self.assertLess(result["fit"]["blocks"], MIN_ANCHORS + 1)
        self.assertFalse(result["fit"]["x"]["kept"])
        self.assertFalse(result["fit"]["y"]["kept"])
        self.assertEqual(result["origin"], [LEFT, TOP + 20*PITCH])
        self.assertAlmostEqual(result["u"][0], PITCH, delta=1e-9)

    def test_the_fallback_is_per_axis(self):
        # four separated columns pin the horizontal pitch, while two shared row boundaries pin no vertical pitch
        cells = [(column, row) for column in (0, 2, 4, 6) for row in (18, 19)]
        result = anchor_lattice(raster(cells), box())
        self.assertTrue(result["fit"]["x"]["kept"])
        self.assertFalse(result["fit"]["y"]["kept"])
        self.assertAlmostEqual(result["u"][0], PITCH, delta=0.25)
        self.assertAlmostEqual(result["origin"][1], TOP + 20*PITCH, delta=1e-9)

    def test_anchor_edges_are_sorted_and_unique(self):
        found = blocks(raster(STACK), PITCH, PITCH, box()["frame"])
        x_edges, y_edges = anchors(found)
        self.assertEqual(x_edges, sorted(set(x_edges)))
        self.assertEqual(y_edges, sorted(set(y_edges)))
        self.assertTrue(all(edge >= 0 for edge in x_edges + y_edges))


if __name__ == "__main__":
    unittest.main()
