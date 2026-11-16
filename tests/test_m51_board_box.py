"""Mechanical controls for the M51 detector-repair read.

M50's wall-run detector needed a full-height wall pair and misplaced two named captures. M51 replaces it with one
declared objective over label-free wall and cell-grid candidates (`scripts/box_search.py`). These tests pin what the
declaration promises: the candidate search finds a well drawn with grid lines and one drawn without them, it does not
follow a panel of the same cell pitch, it snaps to the capture's scale, it refuses a structureless frame, its
objective arithmetic is the declared one, the branch selection is the predeclared one, and the declaration precedes
the receipt with the leakage disclosed.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from box_search import (CLUSTER, COLUMNS, EDGE, MIN_WIDTH, ROWS,  # noqa: E402
                        STACK_WEIGHT, detect_box, grid_candidate, score_box, wall_candidate)
from m51_board_box_read import (AUC_GATE, CONTROL_FLOOR, DETECTION_GATE_CELLS,  # noqa: E402
                                DETECTION_SLACK, EOF_F1_GATE, branch_for)


def capture(width=320, height=460, cell=14, left=40, top=30, grid=True, panel=True, block=False):
    """A capture with a ten-by-twenty well, optionally drawn without cell lines and with a panel beside it."""
    background = np.array([24.0, 29.0, 42.0])
    pixels = np.tile(background, (height, width, 1))
    pixels[top:top+20*cell+1, left:left+10*cell+1] = [13.0, 15.0, 20.0]
    border = [150.0, 160.0, 170.0]
    pixels[top:top+20*cell+1, left] = border
    pixels[top:top+20*cell+1, left+10*cell] = border
    pixels[top, left:left+10*cell+1] = border
    pixels[top+20*cell, left:left+10*cell+1] = border
    if grid:
        for x in range(left, left+10*cell+1, cell):
            pixels[top:top+20*cell+1, x] = [70.0, 80.0, 95.0]
        for y in range(top, top+20*cell+1, cell):
            pixels[y, left:left+10*cell+1] = [70.0, 80.0, 95.0]
        pixels[top:top+20*cell+1, left] = border
    if panel:
        for x in (left+10*cell+30, left+10*cell+90):
            pixels[top:top+5*cell, x] = border
        for y in (top, top+5*cell):
            pixels[y, left+10*cell+30:left+10*cell+91] = border
        for index in range(5):
            pixels[top+2*cell+index*cell:top+2*cell+index*cell+cell,
                   left+10*cell+40:left+10*cell+40+2*cell] = [200.0, 60.0, 60.0]
    if block:
        pixels[top+19*cell:top+20*cell, left+cell:left+2*cell] = [200.0, 60.0, 60.0]
    return pixels


class DetectorTest(unittest.TestCase):
    def test_the_search_reads_a_well_drawn_with_grid_lines(self):
        box = detect_box(capture())
        self.assertAlmostEqual(box["u"][0], 14.0, delta=0.4)
        self.assertAlmostEqual(box["v"][1], -14.0, delta=0.4)
        self.assertAlmostEqual(box["frame"]["left"], 40.0, delta=2.0)
        self.assertAlmostEqual(box["frame"]["bottom"], 310.0, delta=2.0)

    def test_the_search_reads_a_well_drawn_without_grid_lines(self):
        # the fallback the arm exists for: no cell lines to key on, only the border and the fill. The box is found,
        # and - exactly the failure mode this arm measured on real captures - the pitch is allowed to drift there.
        box = detect_box(capture(grid=False, panel=False))
        frame = box["frame"]
        self.assertAlmostEqual(frame["right"]-frame["left"], 140.0, delta=20.0)
        # the phase, not the pitch, is what a gridless well leaves ambiguous: require overlap with the true well
        overlap = (max(0.0, min(frame["right"], 180.0)-max(frame["left"], 40.0))
                   * max(0.0, min(frame["bottom"], 310.0)-max(frame["top"], 30.0)))
        union = ((frame["right"]-frame["left"])*(frame["bottom"]-frame["top"])
                 + 140.0*280.0-overlap)
        self.assertGreater(overlap/union, 0.6)

    def test_a_panel_of_the_same_cell_pitch_does_not_move_the_box(self):
        with_panel = detect_box(capture(panel=True))
        without = detect_box(capture(panel=False))
        self.assertAlmostEqual(with_panel["frame"]["left"], without["frame"]["left"], delta=2.0)
        self.assertAlmostEqual(with_panel["u"][0], without["u"][0], delta=0.4)

    def test_the_search_snaps_to_the_captures_scale(self):
        small = detect_box(capture(width=260, height=380, cell=10, left=20, top=20))
        large = detect_box(capture(width=460, height=620, cell=18, left=60, top=40))
        self.assertAlmostEqual(small["u"][0], 10.0, delta=0.4)
        self.assertAlmostEqual(large["u"][0], 18.0, delta=0.5)
        self.assertAlmostEqual(large["v"][1], -18.0, delta=0.5)

    def test_a_structureless_frame_has_no_box_to_read(self):
        flat = np.tile(np.array([24.0, 29.0, 42.0]), (64, 64, 1))
        with self.assertRaises(Exception):
            detect_box(flat)

    def test_every_candidate_is_scored_and_recorded(self):
        box = detect_box(capture())
        self.assertGreaterEqual(len(box["candidates"]), 1)
        self.assertEqual(box["source"], box["candidates"][0]["source"])
        for candidate in box["candidates"]:
            for key in ("score", "wall_support", "grid_alignment", "cell_uniformity", "stack_containment"):
                self.assertIn(key, candidate["support"])

    def test_the_candidates_are_label_free_generators(self):
        pixels = capture()
        grey = pixels.mean(2)
        column_runs = np.array([0]*pixels.shape[1])
        box = wall_candidate(np.ones(pixels.shape[1], dtype=np.int64)*40,
                             np.ones(pixels.shape[0], dtype=np.int64)*40)
        self.assertEqual(box["source"], "walls")
        edge_x = np.abs(np.diff(grey, axis=1)).sum(0)
        edge_y = np.abs(np.diff(grey, axis=0)).sum(1)
        grid = grid_candidate(edge_x, edge_y)
        self.assertEqual(grid["source"], "grid")
        self.assertGreaterEqual(column_runs.size, 1)


class ObjectiveTest(unittest.TestCase):
    def test_the_objective_is_the_declared_sum(self):
        pixels = capture()
        grey = pixels.mean(2)
        box = detect_box(pixels)
        reading = score_box(pixels, grey, box)
        expected = (reading["wall_support"]+reading["grid_alignment"]+(1.0-(1.0-reading["cell_uniformity"]))
                    +STACK_WEIGHT*reading["stack_containment"])
        self.assertAlmostEqual(reading["score"], expected, places=5)

    def test_a_box_on_the_stack_scores_above_a_box_on_empty_background(self):
        pixels = capture(block=True)
        grey = pixels.mean(2)
        on_board = {"origin": [40.0, 310.0], "u": [14.0, 0.0], "v": [0.0, -14.0],
                    "frame": {"left": 40.0, "right": 180.0, "top": 30.0, "bottom": 310.0}}
        off_board = {"origin": [200.0, 450.0], "u": [14.0, 0.0], "v": [0.0, -14.0],
                     "frame": {"left": 200.0, "right": 300.0, "top": 200.0, "bottom": 450.0}}
        self.assertGreater(score_box(pixels, grey, on_board)["score"],
                           score_box(pixels, grey, off_board)["score"])

    def test_the_declared_constants_are_the_declared_constants(self):
        self.assertEqual((EDGE, CLUSTER, MIN_WIDTH, COLUMNS, ROWS), (12.0, 6, 60, 10, 20))
        self.assertEqual(DETECTION_GATE_CELLS, 0.25)
        self.assertAlmostEqual(DETECTION_SLACK, 0.10)
        self.assertAlmostEqual(CONTROL_FLOOR, 0.382881)


class BranchSelectionTest(unittest.TestCase):
    def test_the_predeclared_thresholds_select_the_predeclared_branches(self):
        # arguments are (mean displacement in cells, median of per-board means, F1 at 0.0, AUC)
        self.assertEqual(branch_for(0.20, 0.06, 0.97, 0.99), "repair-general")
        self.assertEqual(branch_for(0.25, 0.25, 0.90, 0.90), "repair-general")
        self.assertEqual(branch_for(0.383358, 0.066438, 0.904376, 0.933275), "repair-under-specified")
        self.assertEqual(branch_for(0.30, 0.05, 0.90, 0.95), "repair-under-specified")
        self.assertEqual(branch_for(0.80, 0.60, 0.10, 0.60), "repair-fails")
        # a placed board whose read does not separate is not "most boards misplaced": it stays under-specified
        self.assertEqual(branch_for(0.10, 0.05, 0.10, 0.80), "repair-under-specified")

    def test_m50s_population_lands_where_the_declaration_says_it_does(self):
        # M50: 0.802227 mean, 0.061328 median-of-medians, 0.979200 F1, 0.999150 AUC
        self.assertEqual(branch_for(0.802227, 0.061328, 0.9792, 0.99915), "repair-under-specified")

    def test_the_displacement_gate_is_the_inherited_one(self):
        self.assertEqual(DETECTION_GATE_CELLS, 0.25)
        self.assertEqual(AUC_GATE, 0.90)
        self.assertEqual(EOF_F1_GATE, 0.70)


class DeclarationTest(unittest.TestCase):
    def test_the_declaration_precedes_the_receipt_and_fixes_the_falsifier(self):
        declaration = (ROOT/"docs/M51_DETECTOR_REPAIR.md").read_text()
        self.assertIn("predeclared before any scoring run", declaration)
        self.assertIn("displacement is greater than **`0.25` cells, or**", declaration)
        self.assertIn("AUC is below **`0.90`**", declaration)
        for branch in ("repair-general", "repair-under-specified", "repair-fails"):
            self.assertIn(branch, declaration)

    def test_the_declaration_discloses_the_leakage_and_the_prediction(self):
        declaration = (ROOT/"docs/M51_DETECTOR_REPAIR.md").read_text()
        self.assertIn("Leakage disclosure", declaration)
        self.assertIn("motivated by the two named outlier captures", declaration)
        self.assertIn("judged on the **full 28-board population**", declaration)
        self.assertIn("Prediction, standing as written whatever the outcome", declaration)

    def test_the_arm_adds_no_learned_parameter_and_no_training(self):
        declaration = (ROOT/"docs/M51_DETECTOR_REPAIR.md").read_text()
        self.assertIn("no training", declaration)
        self.assertIn("no threshold fitting", declaration)
        self.assertIn("no new learned parameter", declaration)


if __name__ == "__main__":
    unittest.main()
