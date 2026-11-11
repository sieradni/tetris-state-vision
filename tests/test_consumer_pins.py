"""Consumer-level pins for the derived board window (V-089).

Each test drives the converted consumer's own module-level entry point - not the shared helper - with a
declaration whose visible rows are not the published 20, then with a map that cannot cover it. A literal row
slice reintroduced in that file fails its own named test, not only the generic scan.
"""
import re
import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]

TALLER = {"visible_rows": 26, "spawn_rows": 0, "columns": 10}
LITERAL = re.compile(r"\[\s*(?:0\s*,\s*)?:\s*20\s*\]")


class ConsumerPins(unittest.TestCase):
    def assert_converted(self, filename):
        source = (ROOT/"scripts"/filename).read_text(encoding="utf-8")
        self.assertNotRegex(source, LITERAL, filename+" reintroduced a literal cell-map slice")
        self.assertIn("board_window", source, filename+" no longer uses the derived window")

    def test_ensemble_headroom_reads_the_declared_board(self):
        from ensemble_headroom import visible_units
        values = visible_units(torch.arange(260.0).reshape(1, 26, 10), TALLER)
        self.assertEqual(values.numel(), 260)
        self.assertEqual(float(values[-1]), 259.0)
        with self.assertRaises(ValueError):
            visible_units(torch.zeros(1, 20, 10), TALLER)
        self.assert_converted("ensemble_headroom.py")

    def test_sweep_stream_occupancy_reads_the_declared_board(self):
        from sweep_stream_occupancy import sweep_logits
        values = sweep_logits(torch.arange(260.0).reshape(1, 26, 10), TALLER)
        self.assertEqual(values.numel(), 260)
        self.assertEqual(float(values[-1]), 259.0)
        with self.assertRaises(ValueError):
            sweep_logits(torch.zeros(1, 26, 9), TALLER)
        self.assert_converted("sweep_stream_occupancy.py")

    def test_probe_local_reading_reads_the_declared_board(self):
        from probe_local_reading import reading_map
        logits = reading_map({"occupancy": torch.arange(260.0).reshape(1, 26, 10)}, TALLER)
        self.assertEqual(tuple(logits.shape), (26, 10))
        self.assertEqual(float(logits[25, 9]), 259.0)
        with self.assertRaises(ValueError):
            reading_map({"occupancy": torch.zeros(1, 20, 10)}, TALLER)
        self.assert_converted("probe_local_reading.py")

    def test_analyze_m2_pilot_reads_the_declared_board(self):
        from analyze_m2_pilot import pilot_maps
        final = {"occupancy": torch.arange(260.0).reshape(1, 26, 10),
                 "semantic": torch.zeros(1, 26, 10, 8)}
        occupancy, semantic = pilot_maps(final, TALLER)
        self.assertEqual(tuple(occupancy.shape), (26, 10))
        self.assertEqual(tuple(semantic.shape), (26, 10, 8))
        with self.assertRaises(ValueError):
            pilot_maps({"occupancy": torch.zeros(1, 20, 10), "semantic": torch.zeros(1, 20, 10, 8)}, TALLER)
        self.assert_converted("analyze_m2_pilot.py")

    def test_probe_stream_errors_packs_the_labelled_board(self):
        from probe_stream_errors import pack
        final = {"occupancy": torch.arange(260.0).reshape(1, 26, 10),
                 "observability": torch.zeros(1, 26, 10), "semantic": torch.zeros(1, 26, 10, 8),
                 "lattice": torch.zeros(1, 3, 2), "registration": torch.zeros(1),
                 "layer_known": torch.zeros(1, 2), "layer_type": torch.zeros(1, 2, 8),
                 "pose_rotation": torch.zeros(1, 2, 4), "pose_column": torch.zeros(1, 2, 10),
                 "pose_row": torch.zeros(1, 2, 26)}
        target = {"known": torch.ones(1, 26, 10, dtype=torch.bool),
                  "typed": torch.zeros(1, 26, 10, dtype=torch.bool),
                  "semantic": torch.zeros(1, 26, 10, dtype=torch.long),
                  "lattice": torch.zeros(1, 3, 2), "registered": torch.zeros(1, dtype=torch.bool),
                  "occupied": torch.zeros(1, 26, 10),
                  "layer_known": torch.zeros(1, 2, dtype=torch.bool),
                  "pose_known": torch.zeros(1, 2, dtype=torch.bool),
                  "layer_type": torch.zeros(1, 2, dtype=torch.long),
                  "column": torch.zeros(1, 2, dtype=torch.long), "row": torch.zeros(1, 2, dtype=torch.long),
                  "rotations": torch.zeros(1, 2, 4, dtype=torch.bool)}
        prediction, truth = pack(final, target)
        self.assertEqual(len(prediction["occupancy_logits"]), 26)
        self.assertEqual(len(prediction["occupancy_logits"][0]), 10)
        self.assertEqual(float(prediction["occupancy_logits"][25][9]), 259.0)
        self.assertEqual(len(truth["occupied"]), 26)
        short = dict(final, occupancy=torch.zeros(1, 20, 10))
        with self.assertRaises(ValueError):
            pack(short, target)
        self.assert_converted("probe_stream_errors.py")

    def test_trainer_fit_read_uses_the_labelled_board(self):
        from train_m2_stream import fit_correct
        target = {"semantic": torch.zeros(1, 26, 10, dtype=torch.long),
                  "occupied": torch.zeros(1, 26, 10)}
        correct = fit_correct(torch.zeros(1, 26, 10, 8), target)
        self.assertEqual(tuple(correct.shape), (1, 26, 10))
        with self.assertRaises(ValueError):
            fit_correct(torch.zeros(1, 20, 10, 8), target)
        self.assert_converted("train_m2_stream.py")


class FrozenFixture(unittest.TestCase):
    """`tests/fixtures/m2_original/` is excluded from the scan because it is frozen evidence."""

    def test_the_frozen_m2_reference_fixture_keeps_its_original_literal(self):
        path = ROOT/"tests/fixtures/m2_original/losses.py"
        self.assertTrue(path.exists(), "the frozen M2 reference fixture is missing")
        source = path.read_text(encoding="utf-8")
        self.assertRegex(source, re.compile(r"\[:, :20\]"),
                         "the frozen M2 reference lost its original slicing; the scan exclusion no longer holds")
        self.assertIn("def stage_loss", source)
