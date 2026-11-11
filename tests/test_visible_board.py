"""Every occupancy consumer derives its visible-board window (V-089)."""
import re
import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]

from board_window import board_window, visible_board, visible_board_for_target
from tetris_vision.losses import diagnostic_metrics, visible_window
from tetris_vision.model import ModelConfig

PUBLISHED = {"visible_rows": 20, "spawn_rows": 6, "columns": 10}


class Window(unittest.TestCase):
    def test_the_window_is_the_first_rows_of_a_single_channel_map(self):
        occupancy = torch.arange(26*10, dtype=torch.float32).reshape(1, 26, 10)
        window = visible_board(occupancy, PUBLISHED, "occupancy")
        self.assertEqual(tuple(window.shape), (1, 20, 10))
        self.assertTrue(torch.equal(window[0], occupancy[0, :20]))

    def test_a_model_config_object_works_without_a_dict(self):
        self.assertEqual(board_window(ModelConfig()), (20, 6, 10))
        self.assertEqual(tuple(visible_board(torch.zeros(1, 26, 10), ModelConfig()).shape), (1, 20, 10))

    def test_the_window_follows_the_config_rather_than_a_literal(self):
        taller = {"visible_rows": 26, "spawn_rows": 0, "columns": 10}
        self.assertEqual(tuple(visible_board(torch.zeros(1, 26, 10), taller).shape), (1, 26, 10))
        short = {"visible_rows": 5, "spawn_rows": 6, "columns": 10}
        self.assertEqual(tuple(visible_board(torch.zeros(1, 11, 10), short).shape), (1, 5, 10))

    def test_a_map_that_does_not_match_the_declared_board_is_refused(self):
        for wrong in (torch.zeros(1, 20, 10), torch.zeros(1, 26, 9), torch.zeros(2, 26, 10),
                      torch.zeros(26, 10)):
            with self.assertRaises(ValueError):
                visible_board(wrong, PUBLISHED, "occupancy")

    def test_a_class_map_keeps_its_class_axis(self):
        semantic = torch.zeros(1, 26, 10, 8)
        self.assertEqual(tuple(visible_board(semantic, PUBLISHED, "semantic").shape), (1, 20, 10, 8))

    def test_a_declaration_without_a_board_window_is_refused(self):
        with self.assertRaises(ValueError):
            board_window({"visible_rows": 20})
        with self.assertRaises(ValueError):
            board_window({"visible_rows": "20", "spawn_rows": 6, "columns": 10})


class TargetWindow(unittest.TestCase):
    def test_the_window_comes_from_the_labelled_target(self):
        tensor = torch.zeros(1, 26, 10)
        target = {"occupied": torch.zeros(20, 10)}
        self.assertEqual(tuple(visible_board_for_target(tensor, target).shape), (1, 20, 10))

    def test_a_map_that_cannot_cover_the_label_is_refused(self):
        target = {"occupied": torch.zeros(20, 10)}
        for wrong in (torch.zeros(1, 19, 10), torch.zeros(1, 26, 9)):
            with self.assertRaises(ValueError):
                visible_board_for_target(wrong, target)


class Losses(unittest.TestCase):
    def test_the_loss_window_is_identical_to_the_published_slice(self):
        tensor = torch.randn(1, 26, 10)
        target = {"occupied": torch.zeros(20, 10)}
        self.assertTrue(torch.equal(visible_window(tensor, target, "occupancy"), tensor[:, :20]))
        self.assertTrue(torch.equal(visible_window(tensor[..., None], target, "semantic"), tensor[..., None][:, :20]))

    def test_the_loss_window_refuses_a_map_that_cannot_cover_the_label(self):
        target = {"occupied": torch.zeros(20, 10)}
        for wrong in (torch.zeros(1, 10, 10), torch.zeros(1, 26, 9), torch.zeros(26, 10)):
            with self.assertRaises(ValueError):
                visible_window(wrong, target, "occupancy")

    def test_diagnostic_metrics_reads_the_labelled_board_of_a_taller_map(self):
        occupancy = torch.zeros(1, 26, 10)
        occupancy[0, :20] = 4.0
        outputs = {"final": {"occupancy": occupancy, "lattice": torch.zeros(1, 3, 2)}}
        target = {"known": torch.ones(20, 10, dtype=torch.bool), "occupied": torch.ones(20, 10),
                  "lattice": torch.zeros(1, 3, 2), "registered": torch.zeros(1, dtype=torch.bool)}
        metrics = diagnostic_metrics(outputs, target)
        self.assertEqual(metrics["cell_accuracy"], 1.0)
        self.assertTrue(metrics["exact_locked_board"])


class NoLiteralSlices(unittest.TestCase):
    """A standing guard: the literal row slice must not come back to any consumer."""

    PATTERN = re.compile(r"\[\s*(?:0\s*,\s*)?:\s*20\s*\]")

    def test_no_source_file_slices_a_cell_map_with_a_literal(self):
        offenders = []
        for folder in ("src", "scripts"):
            for path in sorted((ROOT/folder).rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if self.PATTERN.search(line):
                        offenders.append("%s:%d: %s" % (path.relative_to(ROOT), number, line.strip()))
        self.assertEqual(offenders, [], "literal cell-map slices returned:\n" + "\n".join(offenders))


if __name__ == "__main__":
    unittest.main()
