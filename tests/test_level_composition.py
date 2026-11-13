"""The M21 composition: the two-path read under the monotone envelope and the crossing rule.

`docs/M21_LEVEL_COMPOSITION.md` declares four arms and the bar each is judged by. These tests hold the mechanism the
block depends on: the order of the transforms, that the crossing rule decides each board at a level derived from that
board alone, that the sum is measurable on its own, that every declared combination reaches the real parser, and that the
curated record says what the predeclaration said it would.
"""
import inspect
import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT/"src", ROOT/"scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from tetris_vision.model import ModelConfig, TetrisStateNet, otsu_board, stack_contiguous

ROWS = 20
SCOPE = ROOT/"docs/evidence/m21-level-r1/scope.json"

STANDARD = ("occupancy_lattice_free", "occupancy_stack_monotone", "occupancy_crossing_threshold")


def config(**overrides):
    """Every arm carries the published detail tap; the declared fields are the only difference."""
    return ModelConfig(**{**ModelConfig.pilot().__dict__, "occupancy_detail": 1, **overrides})


def trained(model, free=True):
    """Fill the published tap (and the free channel where it exists) from a local generator, so these weights do not
    depend on how much RNG the construction happened to consume - which differs between arms."""
    generator = torch.Generator().manual_seed(5_021)
    with torch.no_grad():
        for parameter in (model.occupancy_detail_tap.weight, model.occupancy_detail_tap.bias):
            parameter.copy_(torch.randn(parameter.shape, generator=generator)
                            * (.4 if parameter.dim() > 1 else .1))
        if free and model.occupancy_lattice_free is not None:
            for parameter in (model.occupancy_lattice_free[2].weight,
                              model.occupancy_lattice_free[2].bias):
                parameter.copy_(torch.randn(parameter.shape, generator=generator)
                                * (.3 if parameter.dim() > 1 else .1))
    return model


def build(free=True, **overrides):
    torch.manual_seed(21_021)
    model = trained(TetrisStateNet(config(**overrides)), free=free)
    pixels, valid = inputs()
    with torch.no_grad():
        return model, model(pixels, valid)["final"]


def inputs():
    torch.manual_seed(7)
    return torch.rand(2, 3, 96, 96), torch.ones(2, 1, 96, 96, dtype=torch.bool)


class CompositionMechanismTests(unittest.TestCase):

    def test_the_sum_is_admitted_without_the_envelope(self):
        model, final = build(occupancy_lattice_free=2)
        self.assertEqual(model.config.occupancy_stack_monotone, 0)
        self.assertEqual(model.config.occupancy_crossing_threshold, 0)
        self.assertEqual(sorted(final["occupancy_parts"]), ["free", "lattice"])
        summed = final["occupancy_parts"]["lattice"]+final["occupancy_parts"]["free"]
        self.assertTrue(torch.equal(final["occupancy"], summed))

    def test_the_bare_sum_at_step_zero_is_the_published_read(self):
        """The free channel's output projection starts at zero, so `2` alone is the published function at step zero."""
        # At step zero the free head's output projection is exactly zero, so the sum is the published read.
        torch.manual_seed(21_021)
        plain = trained(TetrisStateNet(config()), free=False)
        torch.manual_seed(21_021)
        model = trained(TetrisStateNet(config(occupancy_lattice_free=2)), free=False)
        pixels, valid = inputs()
        with torch.no_grad():
            left = plain(pixels, valid)["final"]["occupancy"]
            right = model(pixels, valid)["final"]["occupancy"]
        self.assertTrue(torch.equal(left, right))

    def test_the_composition_applies_the_envelope_then_the_crossing(self):
        model, final = build(occupancy_lattice_free=2, occupancy_stack_monotone=1,
                             occupancy_crossing_threshold=1)
        parts = final["occupancy_parts"]
        rebuilt = otsu_board(stack_contiguous(parts["lattice"]+parts["free"], ROWS), ROWS)
        self.assertTrue(torch.equal(final["occupancy"], rebuilt))

    def test_the_envelope_comes_before_the_crossing_not_after(self):
        """Order is a claim of the arm: crossing the raw sum first is a different mechanism."""
        model, final = build(occupancy_lattice_free=2, occupancy_stack_monotone=1,
                             occupancy_crossing_threshold=1)
        parts = final["occupancy_parts"]
        other = stack_contiguous(otsu_board(parts["lattice"]+parts["free"], ROWS), ROWS)
        self.assertFalse(torch.equal(final["occupancy"], other))

    def test_the_crossing_rule_subtracts_one_level_per_board(self):
        torch.manual_seed(11)
        logits = torch.randn(4, 24, 10)
        moved = otsu_board(logits, ROWS)
        level = logits-moved
        self.assertTrue(torch.allclose(level, level[:, :1, :1].expand_as(level), atol=1e-6))

    def test_the_crossing_rule_is_shift_equivariant(self):
        """It reads the board's own split, so adding a constant to a board's read changes nothing."""
        torch.manual_seed(12)
        logits = torch.randn(4, 24, 10)
        shift = torch.tensor([-2.0, 0.0, .5, 3.0]).reshape(4, 1, 1)
        self.assertTrue(torch.allclose(otsu_board(logits+shift, ROWS), otsu_board(logits, ROWS), atol=1e-5))

    def test_the_envelope_never_lowers_a_read(self):
        torch.manual_seed(13)
        logits = torch.randn(4, 24, 10)
        self.assertTrue(torch.all(stack_contiguous(logits, ROWS) >= logits))

    def test_the_composition_is_deterministic(self):
        model, first = build(occupancy_lattice_free=2, occupancy_stack_monotone=1,
                             occupancy_crossing_threshold=1)
        pixels, valid = inputs()
        with torch.no_grad():
            second = model(pixels, valid)["final"]["occupancy"]
        self.assertTrue(torch.equal(first["occupancy"], second))

    def test_neither_transform_reads_a_label(self):
        for function in (otsu_board, stack_contiguous):
            # The docstring is prose about the rule; the executable body is what must not touch a label.
            body = inspect.getsource(function).split('"""')[-1]
            for word in ("occupied", "label", "target", "prevalence", "truth"):
                self.assertNotIn(word, body, function.__name__)


class CompositionConfigTests(unittest.TestCase):

    def test_the_three_declared_fields_are_independent_values(self):
        alone = config(occupancy_lattice_free=2)
        self.assertEqual((alone.occupancy_lattice_free, alone.occupancy_stack_monotone,
                          alone.occupancy_crossing_threshold), (2, 0, 0))
        composed = config(occupancy_lattice_free=2, occupancy_stack_monotone=1,
                          occupancy_crossing_threshold=1)
        self.assertEqual(composed.occupancy_crossing_threshold, 1)

    def test_the_crossing_still_refuses_the_normalised_output(self):
        with self.assertRaises(ValueError):
            config(occupancy_crossing_threshold=1, occupancy_normalise=1)

    def test_an_undeclared_value_is_still_refused(self):
        for field in STANDARD:
            with self.subTest(field=field), self.assertRaises(ValueError):
                config(**{field: 3})


class CompositionPlumbingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from train_m2_stream import arguments
        cls.arguments = staticmethod(arguments)
        cls.base = ["--output", "runs/m21-level-refusal-probe", "--model", "accuracy", "--steps", "2",
                    "--profile-square"]

    def test_every_declared_arm_reaches_the_real_parser(self):
        for arm, expected in (((""), (0, 0, 0)),
                              (("--occupancy-lattice-free", "2"), (0, 2, 0)),
                              (("--occupancy-lattice-free", "2", "--occupancy-stack-monotone", "1"), (1, 2, 0)),
                              (("--occupancy-lattice-free", "2", "--occupancy-stack-monotone", "1",
                                "--occupancy-crossing-threshold", "1"), (1, 2, 1))):
            with self.subTest(arm=arm):
                parsed = self.arguments(self.base+["--occupancy-detail", "1"]+list(arm))
                self.assertEqual((parsed.occupancy_stack_monotone, parsed.occupancy_lattice_free,
                                  parsed.occupancy_crossing_threshold), expected)

    def test_the_single_free_channel_is_still_its_own_arm(self):
        import contextlib
        import io
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            with self.assertRaises(SystemExit):
                self.arguments(self.base+["--occupancy-detail", "1", "--occupancy-lattice-free", "1",
                                          "--occupancy-stack-monotone", "1"])
        self.assertIn("stack-contiguous read", stream.getvalue())


@unittest.skipUnless(SCOPE.exists(), "the M21 curation has not run yet")
class CompositionRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = json.loads(SCOPE.read_text(encoding="utf-8"))
        cls.document = (ROOT/"docs/M21_LEVEL_COMPOSITION.md").read_text(encoding="utf-8")

    def test_the_four_declared_arms_are_recorded(self):
        self.assertEqual(sorted(self.scope["arms"]), ["bare", "control", "crossing", "envelope"])
        for name, arm in self.scope["arms"].items():
            self.assertEqual(arm["declared"],
                             {key: arm["config"][key] for key in STANDARD}, name)
            self.assertEqual(arm["exact_boards_at_zero"]["boards"], 28, name)

    def test_the_control_reproduces_the_recorded_published_read(self):
        control = self.scope["arms"]["control"]
        self.assertEqual(control["auc"], 0.8226433610683003)
        self.assertEqual(control["f1_at_zero"], 0.381640)

    def test_every_arm_differs_from_the_control_in_its_declared_fields(self):
        self.assertEqual(self.scope["aliasing_guard"]["arms_differ_in"], list(STANDARD))
        self.assertTrue(self.scope["aliasing_guard"]["weights_all_distinct"])
        # The arm whose declared config equals M20's reproduces that checkpoint bit-for-bit, which is the
        # block's determinism check; every other arm must be a different checkpoint.
        reproduced = self.scope["aliasing_guard"]["reproduces_the_m20_checkpoint"]
        self.assertEqual(reproduced, ["envelope"])
        self.assertEqual(sorted(self.scope["arms"]), sorted(set(self.scope["arms"])))
        self.assertEqual(len({arm["weights_sha256"] for arm in self.scope["arms"].values()}), 4)

    def test_the_stage_decomposition_reproduces_each_arm(self):
        for name, decomposition in self.scope["decomposition"].items():
            self.assertTrue(decomposition["final_stage_matches_the_arm"], name)
            self.assertEqual(decomposition["stages"][0]["stage"], "raw_sum", name)
            shares = [stage["share"] for stage in decomposition["stages"]]
            self.assertEqual(len(shares), len(decomposition["chain"])+1, name)

    def test_the_raising_stage_is_named_for_the_requested_arm(self):
        diagnosis = self.scope["diagnosis"]
        self.assertIn(diagnosis["raising_stage"], ("monotone", "crossing", "centred"))
        self.assertIsNotNone(diagnosis["monotone_stage"])
        self.assertIsNotNone(diagnosis["crossing_stage"])

    def test_the_channel_split_covers_every_arm_that_carries_the_free_channel(self):
        self.assertEqual(sorted(self.scope["channels"]), ["bare", "crossing", "envelope"])
        for name, split in self.scope["channels"].items():
            self.assertEqual(sorted(key for key in split if key.endswith(("_only", "summed"))),
                             ["free_only", "lattice_only", "summed"], name)

    def test_the_reading_follows_the_declared_bar(self):
        control = self.scope["arms"]["control"]
        arm = self.scope["arms"][self.scope["diagnosis"]["requested_arm"]]
        delta = round(arm["f1_at_zero"]-control["f1_at_zero"], 6)
        auc_delta = round(arm["auc"]-control["auc"], 6)
        expected = ("level-passes" if delta > 0 and auc_delta >= -0.02 else
                    "level-without-ranking" if delta > 0 else "level-fails")
        self.assertEqual(self.scope["reading"], expected)
        decoupled = self.scope["arms"]["bare"]
        decoupled_delta = round(decoupled["f1_at_zero"]-control["f1_at_zero"], 6)
        decoupled_auc = round(decoupled["auc"]-control["auc"], 6)
        expected = ("decoupling-passes" if decoupled_delta > 0 and decoupled_auc >= -0.02 else
                    "decoupling-with-ranking-price" if decoupled_delta > 0 else "decoupling-fails")
        self.assertEqual(self.scope["decoupled_reading"], expected)

    def test_the_document_quotes_the_curated_numbers(self):
        for name, arm in self.scope["arms"].items():
            self.assertIn(f"{round(arm['auc'], 6):,.6f}", self.document, name)
            self.assertIn(f"{round(arm['f1_at_zero'], 6):,.6f}", self.document, name)
        self.assertIn(self.scope["reading"], self.document)
        self.assertIn(self.scope["decoupled_reading"], self.document)


if __name__ == "__main__":
    unittest.main()
