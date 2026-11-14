"""The M25 within-board pair-order term: the mechanism, held to the properties it claims.

`docs/M25_PAIR_ORDER.md` builds the one lever M24's decomposition left standing - exactness needs every occupied cell on
a board to outrank every empty one, so the read is charged for its pairs rather than for its boundary. These tests hold
the term to what it promises: exactly zero on a board separated by the declared margin, positive on an inverted board,
complete and order-free over its pairs, invariant to a board's common shift (no board-level component at all), blind to
unknown cells and to spawn rows, non-zero gradient at the zero read so the first optimizer step leaves the fixed point,
and unchanged parameters and read at step zero, because the lever adds no tensor to the model.
"""
import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT/"src", ROOT/"scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from tetris_vision.losses import (OCCUPANCY_PAIR_RANK_ARMS, PAIR_RANK_MARGIN, PAIR_RANK_WEIGHT,
                                  joint_loss, pair_rank_term, require_occupancy_pair_rank, stage_loss)

SCOPE = ROOT/"docs/evidence/m25-rank-r1/scope.json"
DOC = ROOT/"docs/M25_PAIR_ORDER.md"
ROWS, COLUMNS = 20, 10


def arms(values, occupied, known=None):
    """A one-board output/target pair from flat logits and labels."""
    logits = torch.tensor(values, dtype=torch.float32).reshape(1, *values.shape)
    labels = torch.tensor(occupied, dtype=torch.float32).reshape(1, *occupied.shape)
    mask = (torch.ones_like(labels) if known is None else
            torch.tensor(known, dtype=torch.float32).reshape(1, *known.shape))
    return {"occupancy": logits}, {"occupied": labels, "known": mask}


def three_cells(occupied_values, empty_values):
    """A 2x3 board laid out row by row: occupied first, then empty, so the labels follow the values."""
    values = torch.tensor([occupied_values+empty_values]).reshape(1, 2, 3)
    labels = (values > 0).float()
    return {"occupancy": values}, {"occupied": labels, "known": torch.ones_like(labels)}


class DeclarationTest(unittest.TestCase):
    def test_the_lever_is_a_declared_pair_of_arms(self):
        self.assertEqual(OCCUPANCY_PAIR_RANK_ARMS, (0, 1))

    def test_the_constants_were_fixed_before_any_receipt(self):
        self.assertEqual(PAIR_RANK_MARGIN, 1.0)
        self.assertEqual(PAIR_RANK_WEIGHT, .5)

    def test_the_admission_join_refuses_a_disagreement_on_both_sides(self):
        for value in (0, 1):
            require_occupancy_pair_rank(value, {"arguments": {"occupancy_pair_rank": value},
                                                "recovery_contract": {"occupancy_pair_rank": value}})
        with self.assertRaises(ValueError):
            require_occupancy_pair_rank(1, {"arguments": {"occupancy_pair_rank": 0},
                                            "recovery_contract": {"occupancy_pair_rank": 1}})
        with self.assertRaises(ValueError):
            require_occupancy_pair_rank(1, {"arguments": {"occupancy_pair_rank": 1},
                                            "recovery_contract": {"occupancy_pair_rank": 0}})
        with self.assertRaises(ValueError):
            require_occupancy_pair_rank(2, {"arguments": {"occupancy_pair_rank": 2},
                                            "recovery_contract": {"occupancy_pair_rank": 2}})

    def test_the_parser_carries_it_and_refuses_it_without_the_free_channel(self):
        from train_m2_stream import arguments
        base = ["--output", "runs/m25-test", "--model", "accuracy", "--steps", "2", "--profile-square",
                "--occupancy-detail", "1"]
        parsed = arguments(base+["--occupancy-lattice-free", "2", "--occupancy-pair-rank", "1"])
        self.assertEqual(parsed.occupancy_pair_rank, 1)
        self.assertEqual(arguments(base).occupancy_pair_rank, 0)
        with self.assertRaises(SystemExit):
            arguments(base+["--occupancy-pair-rank", "1"])
        with self.assertRaises(SystemExit):
            arguments(base+["--occupancy-lattice-free", "2", "--occupancy-pair-rank", "1",
                            "--occupancy-midpoint", "1"])
        with self.assertRaises(SystemExit):
            arguments(base+["--occupancy-lattice-free", "2", "--occupancy-pair-rank", "2"])


class TermTest(unittest.TestCase):
    def test_a_board_separated_by_the_margin_is_exactly_zero(self):
        output, target = arms(torch.tensor([[PAIR_RANK_MARGIN+2., -2.]]), torch.tensor([[1., 0.]]))
        self.assertEqual(float(pair_rank_term(output, target)), 0.0)

    def test_an_inverted_board_is_positive_and_the_margin_sets_the_charge(self):
        """One inverted pair is charged exactly its overlap with the declared margin."""
        output, target = arms(torch.tensor([[0.5, 0.2]]), torch.tensor([[1., 0.]]))
        loss = float(pair_rank_term(output, target))
        self.assertAlmostEqual(loss, PAIR_RANK_MARGIN-(0.5-0.2), places=5)

    def test_the_term_is_the_complete_cross_class_mean(self):
        """Every occupied/empty pair, each once: no sampling, no double counting, no class weighting."""
        values = torch.tensor([[2.0, 0.4, -0.3, -1.2, 1.1, -0.9]]).reshape(1, 2, 3)
        labels = (values > 0).float()
        output = {"occupancy": values}
        target = {"occupied": labels, "known": torch.ones_like(labels)}
        high = values[labels > 0].flatten()
        low = values[labels == 0].flatten()
        manual = torch.relu(PAIR_RANK_MARGIN-(high[:, None]-low[None, :])).mean()
        self.assertAlmostEqual(float(pair_rank_term(output, target)), float(manual), places=6)
        self.assertEqual(high.numel()*low.numel(), 3*3)

    def test_pair_order_permutation_leaves_the_term_unchanged(self):
        generator = torch.Generator().manual_seed(131)
        values = torch.randn(1, 4, 5, generator=generator)
        labels = (values > .2).float()
        first = pair_rank_term({"occupancy": values}, {"occupied": labels, "known": torch.ones_like(labels)})
        index = torch.randperm(values.numel(), generator=generator)
        shuffled = values.flatten()[index].reshape(values.shape)
        permuted = labels.flatten()[index].reshape(labels.shape)
        second = pair_rank_term({"occupancy": shuffled},
                                {"occupied": permuted, "known": torch.ones_like(permuted)})
        self.assertAlmostEqual(float(first), float(second), places=6)

    def test_a_board_level_shift_is_invisible_to_the_term(self):
        """Zero board-level component: the term has no way to move the read's level, by construction."""
        values = torch.tensor([[2.0, 0.4, -0.3, -1.2, 1.1, -0.9]]).reshape(1, 2, 3)
        labels = (values > 0).float()
        target = {"occupied": labels, "known": torch.ones_like(labels)}
        base = pair_rank_term({"occupancy": values}, target)
        for offset in (-12.5, 0.0, 3.25, 40.0):
            self.assertAlmostEqual(float(pair_rank_term({"occupancy": values+offset}, target)), float(base),
                                   places=6)

    def test_unknown_cells_are_excluded_from_both_sides(self):
        values = torch.tensor([[2.0, 0.4, -0.3, -1.2]]).reshape(1, 2, 2)
        labels = torch.tensor([[1., 1.], [0., 0.]])
        known = torch.tensor([[1., 0.], [1., 1.]])
        masked = pair_rank_term({"occupancy": values}, {"occupied": labels, "known": known})
        kept = torch.tensor([[2.0, -0.3, -1.2]]).reshape(1, 1, 3)
        self.assertAlmostEqual(float(masked), float(pair_rank_term(
            {"occupancy": kept}, {"occupied": torch.tensor([[1., 0., 0.]]),
                                  "known": torch.ones(1, 1, 3)})), places=6)

    def test_spawn_rows_are_never_scored(self):
        """The window is the labelled visible board: junk above it must not reach the term."""
        values = torch.zeros(1, ROWS+4, COLUMNS)
        values[0, :2, :2] = 2.0
        values[0, ROWS:, :] = -50.0
        labels = torch.zeros(1, ROWS+4, COLUMNS)
        labels[0, :2, :2] = 1.0
        wide = pair_rank_term({"occupancy": values}, {"occupied": labels, "known": torch.ones_like(labels)})
        narrow = pair_rank_term({"occupancy": values[:, :ROWS]},
                                {"occupied": labels[:, :ROWS], "known": torch.ones(1, ROWS, COLUMNS)})
        self.assertAlmostEqual(float(wide), float(narrow), places=6)

    def test_a_single_class_board_is_zero_and_finite(self):
        for values, labels in ((torch.zeros(1, 2, 2), torch.zeros(1, 2, 2)),
                               (torch.ones(1, 2, 2), torch.ones(1, 2, 2))):
            term = pair_rank_term({"occupancy": values}, {"occupied": labels, "known": torch.ones_like(labels)})
            self.assertEqual(float(term), 0.0)
            self.assertTrue(bool(torch.isfinite(term)))

    def test_a_genuine_change_still_moves_the_read_term(self):
        output, target = arms(torch.tensor([[0.4, 0.2]]), torch.tensor([[1., 0.]]))
        before = float(pair_rank_term(output, target))
        output["occupancy"] = output["occupancy"].clone()
        output["occupancy"][0, 0] += 2.0
        self.assertGreater(float(pair_rank_term(output, target)), 0.0)
        self.assertLess(float(pair_rank_term(output, target)), before)

    def test_the_zero_read_leaves_the_fixed_point_on_the_first_step(self):
        """At an all-zero read the term is active, so its gradient is non-zero and one step moves the read."""
        logits = torch.zeros(1, 2, 3, requires_grad=True)
        labels = torch.tensor([[[1., 1., 0.], [0., 0., 0.]]])
        loss = pair_rank_term({"occupancy": logits}, {"occupied": labels, "known": torch.ones_like(labels)})
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)
        occupied_gradient = float(logits.grad[labels > 0].abs().sum())
        empty_gradient = float(logits.grad[labels == 0].abs().sum())
        self.assertGreater(occupied_gradient, 0.0)
        self.assertGreater(empty_gradient, 0.0)
        with torch.no_grad():
            before = logits.detach().clone()
            logits -= .1*logits.grad
        self.assertGreater(float((logits.detach()-before).abs().sum()), 0.0)

    def test_the_gate_is_absent_unless_declared(self):
        output = {"occupancy": torch.zeros(1, 2, 2), "lattice": torch.tensor([[[0., 0.], [1., 0.], [0., 1.]]]),
                  "semantic": torch.zeros(1, 2, 2, 4), "observability": torch.zeros(1, 2, 2),
                  "registration": torch.zeros(1), "layer_type": torch.zeros(1, 2, 2),
                  "layer_known": torch.zeros(1, 2), "pose_rotation": torch.zeros(1, 2, 4),
                  "pose_column": torch.zeros(1, 2, 2), "pose_row": torch.zeros(1, 2, 2)}
        target = {"occupied": torch.tensor([[[1., 0.], [0., 0.]]]),
                  "known": torch.ones(1, 2, 2, dtype=torch.bool),
                  "lattice": torch.tensor([[[0., 0.], [1., 0.], [0., 1.]]]),
                  "registered": torch.ones(1), "semantic": torch.zeros(1, 2, 2, dtype=torch.long),
                  "typed": torch.zeros(1, 2, 2), "rotations": torch.zeros(1, 2, 4, dtype=torch.bool),
                  "column": torch.zeros(1, 2, dtype=torch.long), "row": torch.zeros(1, 2, dtype=torch.long),
                  "pose_known": torch.zeros(1, 2), "layer_type": torch.zeros(1, 2, dtype=torch.long),
                  "layer_known": torch.zeros(1, 2)}
        plain = stage_loss(output, target)
        declared = stage_loss(output, target, occupancy_pair_rank=1)
        self.assertNotIn("pair", plain)
        self.assertIn("pair", declared)
        self.assertGreater(float(declared["pair"]), 0.0)


class ConstructionTest(unittest.TestCase):
    """The lever is an objective term, so it must add no tensor and leave the read bitwise identical."""

    def test_it_adds_no_parameter_and_leaves_the_read_identical(self):
        from tetris_vision.model import ModelConfig, TetrisStateNet
        torch.manual_seed(5)
        model = TetrisStateNet(ModelConfig(**{**ModelConfig.pilot().__dict__, "occupancy_detail": 1,
                                              "occupancy_lattice_free": 2}))
        names = sorted(name for name, _ in model.named_parameters())
        self.assertFalse(any("pair" in name for name in names))
        generator = torch.Generator().manual_seed(19)
        pixels = torch.rand(1, 3, 96, 96, generator=generator)
        valid = torch.ones(1, 1, 96, 96, dtype=torch.bool)
        with torch.no_grad():
            first = model(pixels, valid)["final"]["occupancy"].clone()
        occupied = torch.zeros(1, *first.shape[-2:])
        occupied[0, 0, 0] = 1.0
        pair_rank_term({"occupancy": first},
                       {"occupied": occupied, "known": torch.ones_like(occupied, dtype=torch.bool)})
        with torch.no_grad():
            second = model(pixels, valid)["final"]["occupancy"]
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(names, sorted(name for name, _ in model.named_parameters()))

    def test_the_term_gradient_reaches_the_real_read(self):
        """The term is not decorative: its gradient flows into the model's own occupancy emission."""
        from tetris_vision.model import ModelConfig, TetrisStateNet
        torch.manual_seed(5)
        model = TetrisStateNet(ModelConfig(**{**ModelConfig.pilot().__dict__, "occupancy_detail": 1,
                                              "occupancy_lattice_free": 2}))
        generator = torch.Generator().manual_seed(19)
        pixels = torch.rand(1, 3, 96, 96, generator=generator)
        valid = torch.ones(1, 1, 96, 96, dtype=torch.bool)
        output = model(pixels, valid)["final"]
        occupied = torch.zeros(1, *output["occupancy"].shape[-2:])
        occupied[0, :2, :3] = 1.0
        loss = pair_rank_term(output, {"occupied": occupied,
                                       "known": torch.ones_like(occupied, dtype=torch.bool)})
        loss.backward()
        self.assertGreater(float(model.occupancy_lattice_free[2].weight.grad.abs().sum()), 0.0)


class RecordTest(unittest.TestCase):
    """The catalogued arms, when the block has run."""

    @unittest.skipUnless(SCOPE.exists() and DOC.exists(), "the M25 block has not run yet")
    def test_every_arm_is_recorded_with_its_declarations(self):
        scope = json.loads(SCOPE.read_text(encoding="utf-8"))
        self.assertEqual(scope["instrument"], "m25-rank-read-r1")
        self.assertEqual(sorted(scope["arms"]), ["rank", "rank-stencil", "repro"])
        for name, block in scope["arms"].items():
            self.assertEqual(block["declared"]["occupancy_pair_rank"], 0 if name == "repro" else 1, name)
        self.assertEqual(scope["arms"]["rank"]["declared"]["occupancy_stack_monotone"], 0)
        self.assertEqual(scope["arms"]["rank-stencil"]["declared"]["occupancy_free_stencil"], 1)

    @unittest.skipUnless(SCOPE.exists() and DOC.exists(), "the M25 block has not run yet")
    def test_the_reproduction_arm_matches_the_published_control(self):
        scope = json.loads(SCOPE.read_text(encoding="utf-8"))
        self.assertAlmostEqual(scope["arms"]["repro"]["auc"], 0.8226433610683003, places=9)

    @unittest.skipUnless(SCOPE.exists() and DOC.exists(), "the M25 block has not run yet")
    def test_the_doc_records_each_arm_at_six_places(self):
        doc = DOC.read_text(encoding="utf-8")
        scope = json.loads(SCOPE.read_text(encoding="utf-8"))
        for name, block in scope["arms"].items():
            self.assertIn("%.6f" % block["auc"], doc, name)
            self.assertIn("%.6f" % block["f1_at_zero"], doc, name)


if __name__ == "__main__":
    unittest.main()
