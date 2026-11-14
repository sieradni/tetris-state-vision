"""The M26 symmetric pair margin: the mechanism, held to the properties it claims - and to the one it cannot claim.

M25 charged one direction of every occupied/empty pair and was paid by lifting empties (99.93% positive share, precision
0.119). M26 charges the pair by the closer of its two directional violations, so it cannot be paid by collapsing the two
modes together. That symmetry has a price this file states rather than hides: a charge that is the same whichever class is
called occupied is blind to orientation, so an inverted-but-separated board is free. The tests below prove the claimed
properties and pin the blind spot, so the record's verdict rests on a mechanism that was measured, not on one that was
assumed.
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

from tetris_vision.losses import (OCCUPANCY_PAIR_GAP_ARMS, PAIR_GAP_MARGIN, PAIR_GAP_WEIGHT,
                                  joint_loss, pair_gap_term, pair_rank_term,
                                  require_occupancy_pair_gap, stage_loss)

SCOPE = ROOT/"docs/evidence/m26-gap-r1/scope.json"
DOC = ROOT/"docs/M26_GAP_MARGIN.md"
ROWS, COLUMNS = 20, 10


def board(values, labels):
    values = torch.tensor(values, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    return ({"occupancy": values.reshape(1, *values.shape)},
            {"occupied": labels.reshape(1, *labels.shape), "known": torch.ones_like(labels).reshape(1, *labels.shape)})


def charge(values, labels):
    output, target = board(values, labels)
    return float(pair_gap_term(output, target))


class DeclarationTest(unittest.TestCase):
    def test_the_lever_is_a_declared_pair_of_arms(self):
        self.assertEqual(OCCUPANCY_PAIR_GAP_ARMS, (0, 1))

    def test_the_constants_were_fixed_before_any_receipt(self):
        self.assertEqual(PAIR_GAP_MARGIN, 1.0)
        self.assertEqual(PAIR_GAP_WEIGHT, .5)

    def test_the_admission_join_refuses_a_disagreement_on_both_sides(self):
        for value in (0, 1):
            require_occupancy_pair_gap(value, {"arguments": {"occupancy_pair_gap": value},
                                               "recovery_contract": {"occupancy_pair_gap": value}})
        with self.assertRaises(ValueError):
            require_occupancy_pair_gap(1, {"arguments": {"occupancy_pair_gap": 0},
                                           "recovery_contract": {"occupancy_pair_gap": 1}})
        with self.assertRaises(ValueError):
            require_occupancy_pair_gap(1, {"arguments": {"occupancy_pair_gap": 1},
                                           "recovery_contract": {"occupancy_pair_gap": 0}})
        with self.assertRaises(ValueError):
            require_occupancy_pair_gap(2, {"arguments": {"occupancy_pair_gap": 2},
                                           "recovery_contract": {"occupancy_pair_gap": 2}})

    def test_the_parser_carries_it_and_refuses_it_without_the_free_channel(self):
        from train_m2_stream import arguments
        base = ["--output", "runs/m26-test", "--model", "accuracy", "--steps", "2", "--profile-square",
                "--occupancy-detail", "1"]
        parsed = arguments(base+["--occupancy-lattice-free", "2", "--occupancy-pair-gap", "1"])
        self.assertEqual(parsed.occupancy_pair_gap, 1)
        self.assertEqual(arguments(base).occupancy_pair_gap, 0)
        with self.assertRaises(SystemExit):
            arguments(base+["--occupancy-pair-gap", "1"])
        with self.assertRaises(SystemExit):
            arguments(base+["--occupancy-lattice-free", "2", "--occupancy-pair-gap", "1",
                            "--occupancy-midpoint", "1"])
        with self.assertRaises(SystemExit):
            arguments(base+["--occupancy-lattice-free", "2", "--occupancy-pair-gap", "1",
                            "--occupancy-pair-rank", "1"])
        with self.assertRaises(SystemExit):
            arguments(base+["--occupancy-lattice-free", "2", "--occupancy-pair-gap", "2"])


class TermTest(unittest.TestCase):
    def test_a_margin_separated_board_is_exactly_zero(self):
        self.assertEqual(charge([[3., 2.], [0., -1.]], [[1., 1.], [0., 0.]]), 0.0)

    def test_a_collapsed_board_is_charged_exactly_the_margin_on_every_pair(self):
        """The M25 failure mode: all reads equal. Every pair carries exactly the declared margin."""
        self.assertAlmostEqual(charge([[0., 0.], [0., 0.]], [[1., 1.], [0., 0.]]), PAIR_GAP_MARGIN, places=6)

    def test_a_partly_collapsed_board_is_charged_per_pair(self):
        """Each pair carries the margin minus its own gap, so the charge is the pair product, not a class mean."""
        values = [[0.4, 0.4], [-0.2, -0.9]]
        labels = [[1., 1.], [0., 0.]]
        pairs = [max(0.0, PAIR_GAP_MARGIN-abs(high-low)) for high in values[0] for low in values[1]]
        self.assertAlmostEqual(charge(values, labels), sum(pairs)/len(pairs), places=6)

    def test_the_charge_is_the_same_whichever_class_is_called_occupied(self):
        """Two-sided: swapping the two classes leaves the value unchanged, which the M25 term did not."""
        values, labels = [[0.7, -0.1], [0.2, -0.4]], [[1., 1.], [0., 0.]]
        swapped = [[0.2, -0.4], [0.7, -0.1]]
        self.assertAlmostEqual(charge(values, labels), charge(swapped, labels), places=6)
        output, target = board(values, labels)
        mirrored = {"occupancy": -output["occupancy"]}
        self.assertAlmostEqual(float(pair_gap_term(mirrored, target)),
                               float(pair_gap_term(output, target)), places=6)

    def test_pair_iteration_order_does_not_matter(self):
        generator = torch.Generator().manual_seed(77)
        values = torch.randn(1, 3, 4, generator=generator)
        labels = (values > .1).float()
        target = {"occupied": labels, "known": torch.ones_like(labels)}
        first = pair_gap_term({"occupancy": values}, target)
        index = torch.randperm(values.numel(), generator=generator)
        shuffled = values.flatten()[index].reshape(values.shape)
        permuted = labels.flatten()[index].reshape(labels.shape)
        second = pair_gap_term({"occupancy": shuffled}, {"occupied": permuted, "known": torch.ones_like(permuted)})
        self.assertAlmostEqual(float(first), float(second), places=6)

    def test_a_global_shift_is_invisible(self):
        values, labels = [[0.4, 0.4], [-0.2, -0.9]], [[1., 1.], [0., 0.]]
        base = charge(values, labels)
        self.assertGreater(base, 0.0)
        for offset in (-40.0, -1.5, 3.25, 60.0):
            shifted = [[value+offset for value in row] for row in values]
            self.assertAlmostEqual(charge(shifted, labels), base, places=5)

    def test_unknown_cells_are_excluded_from_both_sides(self):
        output, target = board([[2.0, 0.4], [-0.3, -1.2]], [[1., 1.], [0., 0.]])
        target["known"] = torch.tensor([[[1., 0.], [1., 1.]]])
        masked = pair_gap_term(output, target)
        kept = pair_gap_term({"occupancy": torch.tensor([[[2.0, -0.3, -1.2]]])},
                             {"occupied": torch.tensor([[[1., 0., 0.]]]),
                              "known": torch.ones(1, 1, 3)})
        self.assertAlmostEqual(float(masked), float(kept), places=6)

    def test_spawn_rows_are_never_scored(self):
        values = torch.zeros(1, ROWS+4, COLUMNS)
        values[0, :2, :2] = 1.0
        labels = torch.zeros(1, ROWS+4, COLUMNS)
        labels[0, :2, :2] = 1.0
        values[0, ROWS:, :] = -80.0
        wide = pair_gap_term({"occupancy": values},
                             {"occupied": labels, "known": torch.ones_like(labels)})
        narrow = pair_gap_term({"occupancy": values[:, :ROWS]},
                               {"occupied": labels[:, :ROWS], "known": torch.ones(1, ROWS, COLUMNS)})
        self.assertAlmostEqual(float(wide), float(narrow), places=6)

    def test_the_exact_tie_is_a_symmetric_saddle_and_is_charged_the_margin(self):
        """At a fully tied read every pair carries exactly the margin, and the slope there is exactly zero.

        This is a theorem rather than a defect of the implementation: the charge is even in the pair gap (that is what
        makes it symmetric in the two classes), and the derivative of an even function at zero is zero. A charge that is
        the same whichever class is called occupied therefore cannot also have a non-zero slope at the tie - the two
        properties are exclusive, and the record states this instead of claiming both.
        """
        logits = torch.zeros(1, 2, 3, requires_grad=True)
        labels = torch.tensor([[[1., 1., 0.], [0., 0., 0.]]])
        loss = pair_gap_term({"occupancy": logits}, {"occupied": labels, "known": torch.ones_like(labels)})
        self.assertAlmostEqual(float(loss.detach()), PAIR_GAP_MARGIN, places=6)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(float(logits.grad.abs().sum()), 0.0)

    def test_a_read_off_the_tie_has_a_live_gradient_and_one_step_moves_it(self):
        """Every read a model actually emits is off the exact tie, and there the gradient is live, on both classes."""
        logits = torch.tensor([[[0.3, 0.2, -0.1], [-0.2, -0.3, -0.4]]], requires_grad=True)
        labels = torch.tensor([[[1., 1., 0.], [0., 0., 0.]]])
        loss = pair_gap_term({"occupancy": logits}, {"occupied": labels, "known": torch.ones_like(labels)})
        loss.backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)
        self.assertGreater(float(logits.grad[labels > 0].abs().sum()), 0.0)
        self.assertGreater(float(logits.grad[labels == 0].abs().sum()), 0.0)
        with torch.no_grad():
            before = logits.detach().clone()
            logits -= .05*logits.grad
        self.assertGreater(float((logits.detach()-before).abs().sum()), 0.0)

    def test_a_genuine_change_still_moves_the_charge(self):
        output, target = board([[0.4, 0.4], [-0.2, -0.9]], [[1., 1.], [0., 0.]])
        before = float(pair_gap_term(output, target))
        moved = {"occupancy": output["occupancy"].clone()}
        moved["occupancy"][0, 0, 0] += 2.0
        self.assertLess(float(pair_gap_term(moved, target)), before)

    def test_the_symmetric_charge_is_blind_to_an_inverted_but_separated_board(self):
        """The price of symmetry, pinned: an inverted-but-separated board is free, so this is a collapse guard.

        A charge that is the same whichever class is occupied cannot also teach orientation. This is recorded so the
        record states what the mechanism is instead of discovering it in the numbers.
        """
        self.assertEqual(charge([[-3., -2.], [0., 1.]], [[1., 1.], [0., 0.]]), 0.0)

    def test_the_two_terms_differ_exactly_on_inverted_pairs(self):
        """The one-sided term charges an inversion by the margin plus the overlap; the symmetric one by the overlap.

        On a board with no inverted pair the two agree, which is why the symmetric form can only change an arm through
        the inversions and the collapse, and that is what the measured arms are read against.
        """
        inverted = ([[0.4, 0.4], [0.9, -0.9]], [[1., 1.], [0., 0.]])
        output, target = board(*inverted)
        self.assertAlmostEqual(float(pair_gap_term(output, target)), .5*(1.0-.5)/1, places=2)
        self.assertGreater(float(pair_rank_term(output, target)), float(pair_gap_term(output, target)))
        upright = ([[0.4, 0.4], [-0.2, -0.9]], [[1., 1.], [0., 0.]])
        upright_output, upright_target = board(*upright)
        self.assertAlmostEqual(float(pair_gap_term(upright_output, upright_target)),
                               float(pair_rank_term(upright_output, upright_target)), places=6)

    def test_the_gate_is_absent_unless_declared(self):
        output = {"occupancy": torch.zeros(1, 2, 2),
                  "lattice": torch.tensor([[[0., 0.], [1., 0.], [0., 1.]]]),
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
        declared = stage_loss(output, target, occupancy_pair_gap=1)
        self.assertNotIn("gap", plain)
        self.assertIn("gap", declared)
        self.assertGreater(float(declared["gap"]), 0.0)


class ConstructionTest(unittest.TestCase):
    def test_it_adds_no_parameter_and_leaves_the_read_identical(self):
        from tetris_vision.model import ModelConfig, TetrisStateNet
        torch.manual_seed(5)
        model = TetrisStateNet(ModelConfig(**{**ModelConfig.pilot().__dict__, "occupancy_detail": 1,
                                              "occupancy_lattice_free": 2}))
        names = sorted(name for name, _ in model.named_parameters())
        self.assertFalse(any("gap" in name for name in names))
        generator = torch.Generator().manual_seed(23)
        pixels = torch.rand(1, 3, 96, 96, generator=generator)
        valid = torch.ones(1, 1, 96, 96, dtype=torch.bool)
        with torch.no_grad():
            first = model(pixels, valid)["final"]["occupancy"].clone()
        occupied = torch.zeros(1, *first.shape[-2:])
        occupied[0, :2, :2] = 1.0
        pair_gap_term({"occupancy": first}, {"occupied": occupied,
                                             "known": torch.ones_like(occupied, dtype=torch.bool)})
        with torch.no_grad():
            second = model(pixels, valid)["final"]["occupancy"]
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(names, sorted(name for name, _ in model.named_parameters()))

    def test_the_gradient_reaches_the_real_read(self):
        from tetris_vision.model import ModelConfig, TetrisStateNet
        torch.manual_seed(5)
        model = TetrisStateNet(ModelConfig(**{**ModelConfig.pilot().__dict__, "occupancy_detail": 1,
                                              "occupancy_lattice_free": 2}))
        generator = torch.Generator().manual_seed(23)
        output = model(torch.rand(1, 3, 96, 96, generator=generator),
                       torch.ones(1, 1, 96, 96, dtype=torch.bool))["final"]
        occupied = torch.zeros(1, *output["occupancy"].shape[-2:])
        occupied[0, :3, :3] = 1.0
        loss = pair_gap_term(output, {"occupied": occupied,
                                      "known": torch.ones_like(occupied, dtype=torch.bool)})
        loss.backward()
        self.assertGreater(float(model.occupancy_lattice_free[2].weight.grad.abs().sum()), 0.0)


class RecordTest(unittest.TestCase):
    @unittest.skipUnless(SCOPE.exists() and DOC.exists(), "the M26 block has not run yet")
    def test_every_arm_is_recorded_with_its_declarations(self):
        scope = json.loads(SCOPE.read_text(encoding="utf-8"))
        self.assertEqual(scope["instrument"], "m26-gap-read-r1")
        self.assertEqual(sorted(scope["arms"]), ["gap", "gap-stencil", "repro"])
        for name, block in scope["arms"].items():
            self.assertEqual(block["declared"]["occupancy_pair_gap"], 0 if name == "repro" else 1, name)
            self.assertEqual(block["declared"]["occupancy_pair_rank"], 0, name)
        self.assertEqual(scope["arms"]["gap"]["declared"]["occupancy_stack_monotone"], 0)
        self.assertEqual(scope["arms"]["gap-stencil"]["declared"]["occupancy_free_stencil"], 1)

    @unittest.skipUnless(SCOPE.exists() and DOC.exists(), "the M26 block has not run yet")
    def test_the_reproduction_arm_matches_the_published_control(self):
        scope = json.loads(SCOPE.read_text(encoding="utf-8"))
        self.assertAlmostEqual(scope["arms"]["repro"]["auc"], 0.8226433610683003, places=9)

    @unittest.skipUnless(SCOPE.exists() and DOC.exists(), "the M26 block has not run yet")
    def test_the_doc_records_each_arm_at_six_places(self):
        doc = DOC.read_text(encoding="utf-8")
        scope = json.loads(SCOPE.read_text(encoding="utf-8"))
        for name, block in scope["arms"].items():
            self.assertIn("%.6f" % block["auc"], doc, name)
            self.assertIn("%.6f" % block["f1_at_zero"], doc, name)


if __name__ == "__main__":
    unittest.main()
