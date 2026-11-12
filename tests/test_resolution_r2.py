"""The re-declared resolution pass: envelope constants, out-of-fold boundary, owned family."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"scripts"), str(ROOT/"src")]

import record_resolution_lever_r2 as recorder
from occupancy_sweep import threshold_grid
from record_stream_run import OWNED_RUN_PATTERN


def board(logits, occupied, sample_id):
    return {"sample_id": sample_id, "logits": logits, "occupied": occupied}


class DeclarationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = (ROOT/"docs/M3_OCCUPANCY_RESOLUTION_R2.md").read_text(encoding="utf-8")

    def test_the_recorder_binds_the_declared_envelope(self):
        self.assertEqual(recorder.CEILING_BYTES, 1792*1024**2)
        self.assertEqual(recorder.CEILING_BYTES, 1879048192)
        self.assertEqual((recorder.STEPS, recorder.FORWARDS), (64, 54))
        self.assertEqual(recorder.ATTRIBUTION_DELTA, 0.05)
        self.assertEqual(recorder.FAMILY, "m3-occres2")

    def test_the_declaration_states_those_numbers_and_the_refused_peak(self):
        for figure in ("1,792 MiB", "1,879,048,192", "1,231,880,192", "647,168,000",
                       "64 further steps", "m3-occres2", "23,920,640"):
            self.assertIn(figure, self.text)
        self.assertIn("NOT ADMISSIBLE AS DECLARED", self.text)
        self.assertIn("Reported, never read", self.text)

    def test_the_envelope_is_inside_the_admission_rule(self):
        # min(physical/3, 8 GiB) on the host that runs this pass is 8 GiB.
        self.assertLess(recorder.CEILING_BYTES, 8*1024**3)
        self.assertGreaterEqual(recorder.CEILING_BYTES, 1231880192)

    def test_the_owned_run_family_is_declared(self):
        for name in ("m3-occres2-lever-r1", "m3-occres2-control-r1",
                     "m3-occres2-lever-r1-sweep-r1", "m3-occres2-control-r1-sweep-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        self.assertFalse(OWNED_RUN_PATTERN.fullmatch("m3-occres2-"))


class OutOfFoldTests(unittest.TestCase):
    """The predeclared boundary: chosen on the other groups, never on the cell it scores."""

    def setUp(self):
        from occupancy_sweep import threshold_grid as grid
        self.grid = tuple(grid())

    def test_the_boundary_never_reads_the_held_out_group(self):
        """Two clusters whose own best boundaries disagree: the fold must use the other one's."""
        separated = board([3.0, -3.0], [True, False], "training-1")
        overlapping = board([0.2, -0.2], [True, False], "training-2")
        curves = recorder.curves_of([separated, overlapping])
        result = recorder.out_of_fold(curves, self.grid)
        # Cluster 1 read on its own would separate at -3.0; the fold instead takes cluster 2's gap,
        # whose lowest tied grid point is -0.15, and vice versa. Neither boundary sees its own group.
        self.assertEqual(recorder.best_threshold(curves[0:1], self.grid), -3.0)
        self.assertEqual(recorder.best_threshold(curves[1:2], self.grid), -0.15)
        self.assertEqual(result["thresholds"], [-0.15, -3.0])
        self.assertEqual(result["counts"]["true_positive"]
                         + result["counts"]["false_negative"], 2)

    def test_a_monotone_shift_of_an_arm_cannot_move_the_out_of_fold_read(self):
        rows = [board([2.0, 0.4, -1.0, -2.5], [True, True, False, False], "training-1"),
                board([1.1, -0.3, -0.7, -1.9], [True, True, False, False], "training-2"),
                board([0.6, -0.2, -1.4, -2.0], [True, False, False, False], "training-3")]
        shifted = [board([logit+0.3 for logit in row["logits"]], row["occupied"], row["sample_id"])
                   for row in rows]
        first = recorder.out_of_fold(recorder.curves_of(rows), self.grid)
        second = recorder.out_of_fold(recorder.curves_of(shifted), self.grid)
        self.assertEqual(first["f1"], second["f1"])
        for before, after in zip(first["thresholds"], second["thresholds"]):
            self.assertAlmostEqual(after-before, 0.3, places=9)

    def test_the_fixed_boundary_is_not_shift_invariant(self):
        rows = [board([0.4, -0.1], [True, False], "training-1"),
                board([0.3, -0.2], [True, False], "training-2")]
        shifted = [board([logit-0.6 for logit in row["logits"]], row["occupied"], row["sample_id"])
                   for row in rows]
        self.assertNotEqual(recorder.f1_at_threshold(rows), recorder.f1_at_threshold(shifted))

    def test_the_components_decompose_the_primary_exactly(self):
        rows = [board([2.0, 0.4, -1.0, -2.5], [True, True, False, False], "training-1"),
                board([1.1, -0.3, -0.7, -1.9], [True, True, False, False], "training-2")]
        block = recorder.components_of(rows, self.grid)
        self.assertAlmostEqual(block["zero_cost"], block["zero_f1"]-block["best_f1"], places=9)
        self.assertAlmostEqual(block["transfer_cost"], block["out_of_fold_f1"]-block["zero_f1"], places=9)
        self.assertGreaterEqual(block["best_f1"], block["zero_f1"])

    def test_an_empty_board_is_not_part_of_the_deciding_population(self):
        """V-097: the components must read the 28 nonempty boards the sweep's rows read."""
        first = board([2.0, -2.0], [True, False], "training-1")
        second = board([1.5, -1.5], [True, False], "training-2")
        empty = board([0.1]*8, [False]*8, "training-3")
        self.assertEqual(recorder.components_of([first, second], self.grid),
                         recorder.components_of([first, second, empty], self.grid))

    def test_a_short_grid_or_a_single_cluster_is_refused(self):
        with self.assertRaises(ValueError):
            recorder.require_grid(tuple(self.grid[:-1]))
        with self.assertRaises(ValueError):
            recorder.out_of_fold(recorder.curves_of(
                [board([1.0, -1.0], [True, False], "training-1")]), self.grid)

    def test_each_reported_interval_names_its_own_statistic(self):
        """V-098: the fixed-boundary delta carries the resampled interval, the fitted one none."""
        source = (ROOT/"scripts/record_resolution_lever_r2.py").read_text(encoding="utf-8")
        self.assertIn('"best_f1": {"delta": round(delta_best, 6),', source)
        self.assertIn('"interval": None', source)
        self.assertIn('"zero_boundary_f1"', source)
        self.assertIn("the F1 interval does not belong to the fixed-boundary delta", source)

    def test_the_detectable_floor_is_the_declared_multiple(self):
        interval = {"standard_error": 0.02}
        self.assertEqual(recorder.detectable(interval), round(2.802*0.02, 6))


if __name__ == "__main__":
    unittest.main()
