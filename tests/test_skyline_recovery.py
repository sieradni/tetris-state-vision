"""Tests for the M17 joint skyline decision: recovery, determinism, leakage, refusals, record."""
import contextlib
import inspect
import io
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from tetris_vision.losses import require_occupancy_skyline_recovery
from tetris_vision.model import (ModelConfig, SKYLINE_PENALTY, centre_board, skyline_shift,
                                stack_contiguous)

ROOT = Path(__file__).resolve().parents[1]
ROWS, COLUMNS, BATCH = 20, 10, 2


def board(heights, low=-2.0, high=2.0, spawn=0.0):
    """A read whose columns are an occupied run of `heights[column]` cells over empty ones."""
    columns = [[high]*h+[low]*(ROWS-h) for h in heights]
    visible = torch.tensor(columns, dtype=torch.float32).transpose(0, 1)[None].repeat(BATCH, 1, 1)
    return torch.cat((visible, torch.full((BATCH, 6, COLUMNS), spawn)), 1)


def decided(logits, rows=ROWS):
    return ((logits[:, :rows] > 0).sum(dim=1)).tolist()


class JointDecisionTests(unittest.TestCase):
    def test_a_clean_board_is_recovered_exactly(self):
        heights = [0, 1, 2, 3, 3, 2, 1, 0, 0, 1]
        ruled = skyline_shift(board(heights), ROWS)
        self.assertEqual(decided(ruled)[0], heights)
        for index in range(BATCH):
            self.assertEqual(decided(ruled)[index], heights)

    def test_the_joint_decision_repairs_a_column_whose_own_read_is_marginal(self):
        # Column 4's read is one row too tall on its own, while its neighbours and its own lower rows agree. A joint
        # decision must prefer the coherent skyline; an independent per-column one cannot see the disagreement.
        heights = [2, 2, 3, 3, 3, 3, 2, 2, 1, 1]
        latent = board(heights)
        latent[:, 3, 4] = 0.05                  # the marginal cell that makes column 4 look four tall
        latent[:, 2, 4] = 2.0
        alone = decided(latent)[0]
        self.assertEqual(alone[4], 4, "the per-column decision should be fooled")
        self.assertEqual(decided(skyline_shift(latent, ROWS))[0], [2, 2, 3, 3, 3, 3, 2, 2, 1, 1])

    def test_the_decision_is_deterministic(self):
        latent = board([1, 0, 2, 5, 5, 0, 1, 3, 0, 0])
        first = skyline_shift(latent, ROWS)
        for _ in range(4):
            self.assertTrue(torch.equal(skyline_shift(latent, ROWS), first))

    def test_the_decision_is_one_skyline_not_ten_crossings(self):
        # The decided state is monotone per column and contiguous from the floor, whatever the read looks like.
        latent = torch.randn(BATCH, ROWS+6, COLUMNS)*2.0
        ruled = skyline_shift(latent, ROWS)
        heights = decided(ruled)[0]
        expected = (torch.arange(ROWS)[:, None] < torch.tensor(heights)[None, :]).float()
        self.assertTrue(torch.equal((ruled[0, :ROWS] > 0).float(), expected))

    def test_the_spawn_rows_are_untouched_and_the_board_is_per_board(self):
        latent = board([1, 2, 3, 2, 1, 0, 0, 1, 2, 3])
        latent[1, :, :] += 4.0
        ruled = skyline_shift(latent, ROWS)
        self.assertTrue(torch.equal(ruled[:, ROWS:], latent[:, ROWS:]))
        for index in range(BATCH):
            self.assertTrue(torch.allclose(ruled[index:index+1], skyline_shift(latent[index:index+1], ROWS),
                                           atol=1e-6, rtol=0))

    def test_a_penalty_is_the_only_declared_constant(self):
        self.assertEqual(SKYLINE_PENALTY, 1.0)
        signature = inspect.signature(skyline_shift)
        self.assertEqual(list(signature.parameters), ["logits", "visible_rows", "penalty"])
        self.assertEqual(signature.parameters["penalty"].default, SKYLINE_PENALTY)

    def test_the_recovery_composes_after_the_two_structural_reads(self):
        latent = board([2, 2, 3, 3, 2, 2, 1, 1, 0, 0])
        envelope = stack_contiguous(latent, ROWS)
        centred = centre_board(envelope, ROWS)
        ruled = skyline_shift(centred, ROWS)
        self.assertEqual(decided(ruled)[0], [2, 2, 3, 3, 2, 2, 1, 1, 0, 0])
        self.assertEqual(decided(centred)[0], [2, 2, 3, 3, 2, 2, 1, 1, 0, 0])


class LeakageGuardTests(unittest.TestCase):
    def test_the_recovery_reads_no_labels(self):
        self.assertEqual(list(inspect.signature(skyline_shift).parameters),
                         ["logits", "visible_rows", "penalty"])
        latent = torch.randn(BATCH, ROWS+6, COLUMNS)*2.0
        decided_once = skyline_shift(latent, ROWS)
        labels = (torch.rand(BATCH, ROWS, COLUMNS) > .9).float()
        for permutation in (labels.flip(1), labels.flip(2), 1.0-labels, torch.zeros_like(labels),
                            torch.ones_like(labels)):
            self.assertTrue(torch.equal(skyline_shift(latent, ROWS), decided_once))
            self.assertTrue(permutation is not None)
        source = Path(inspect.getsourcefile(skyline_shift)).read_text(encoding="utf-8")
        body = source[source.index("def skyline_shift"):source.index("def stack_contiguous")]
        body = body.split('"""')[-1]
        for forbidden in ("label", "target", "occupied", "prevalence", "threshold_grid", "best_threshold"):
            self.assertNotIn(forbidden, body.lower(), forbidden)


class DeclaredConfigTests(unittest.TestCase):
    def test_the_published_decision_is_the_default(self):
        self.assertEqual(ModelConfig().occupancy_skyline_recovery, 0)
        self.assertEqual(ModelConfig(occupancy_skyline_recovery=1).occupancy_skyline_recovery, 1)

    def test_an_undeclared_option_is_refused(self):
        for value in (2, -1, True, 1.0, "1"):
            with self.assertRaises(ValueError):
                ModelConfig(occupancy_skyline_recovery=value)

    def test_the_recovery_cannot_be_normalised(self):
        with self.assertRaises(ValueError):
            ModelConfig(occupancy_skyline_recovery=1, occupancy_normalise=1)

    def test_the_declared_composition_is_admitted(self):
        config = ModelConfig(occupancy_skyline_recovery=1, occupancy_crossing_threshold=1,
                             occupancy_stack_monotone=1, occupancy_centre=1)
        self.assertEqual((config.occupancy_skyline_recovery, config.occupancy_crossing_threshold,
                          config.occupancy_stack_monotone, config.occupancy_centre), (1, 1, 1, 1))

    def test_the_admission_join_refuses_a_mismatch(self):
        summary = {"arguments": {"occupancy_skyline_recovery": 1},
                   "recovery_contract": {"occupancy_skyline_recovery": 1},
                   "config": {"occupancy_skyline_recovery": 1}}
        self.assertIsNone(require_occupancy_skyline_recovery(1, summary))
        for value, wrong in ((0, summary), (2, summary),
                             (1, {"arguments": {"occupancy_skyline_recovery": 0},
                                  "recovery_contract": {"occupancy_skyline_recovery": 1},
                                  "config": {"occupancy_skyline_recovery": 1}}),
                             (1, {"arguments": {}, "recovery_contract": {}, "config": {}})):
            with self.assertRaises(ValueError):
                require_occupancy_skyline_recovery(value, wrong)


class PlumbingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from train_m2_stream import arguments, validate_admission, main
        cls.source = Path(inspect.getsourcefile(validate_admission)).read_text(encoding="utf-8")
        cls.driver = inspect.getsource(main)
        cls.refusals = inspect.getsource(arguments)
        cls.signature = inspect.signature(validate_admission)
        cls.arguments = staticmethod(arguments)
        cls.base = ["--output", "runs/m17-skyline-refusal-probe", "--model", "accuracy", "--steps", "2"]

    def test_the_driver_writes_the_option_into_the_config(self):
        start = self.driver.index("replace(config,")
        block = self.driver[start:self.driver.index("config_json", start)]
        self.assertIn("occupancy_skyline_recovery=args.occupancy_skyline_recovery", block)

    def test_the_driver_records_the_option_in_the_recovery_contract(self):
        self.assertIn('recovery_contract["occupancy_skyline_recovery"] = args.occupancy_skyline_recovery',
                      self.driver)

    def test_the_admission_checks_the_option_it_receives(self):
        self.assertIn("occupancy_skyline_recovery", self.signature.parameters)
        self.assertEqual(self.signature.parameters["occupancy_skyline_recovery"].default, 0)
        self.assertIn("require_occupancy_skyline_recovery(occupancy_skyline_recovery, summary)", self.source)

    def test_the_trainer_refuses_every_term_and_lever_beside_the_recovery(self):
        self.assertIn("if args.occupancy_skyline_recovery and (args.occupancy_normalise", self.refusals)
        for lever in (["--occupancy-midpoint", "1"], ["--occupancy-dense-hinge", "1"],
                      ["--occupancy-quantile-anchor", "1"], ["--occupancy-anchor", "1"],
                      ["--occupancy-margin-hinge", "1"], ["--occupancy-normalise", "1"],
                      ["--occupancy-deform", "2", "--occupancy-detail", "1"],
                      ["--occupancy-offset-rows", "1", "--occupancy-detail", "1"],
                      ["--occupancy-detail-stencil", "1", "--occupancy-detail", "1"],
                      ["--occupancy-resolution", "1"], ["--occupancy-context", "1"],
                      ["--occupancy-column", "1"], ["--occupancy-lattice-free", "1"]):
            stream = io.StringIO()
            with self.subTest(lever=lever), contextlib.redirect_stderr(stream):
                with self.assertRaises(SystemExit):
                    self.arguments(self.base+["--occupancy-skyline-recovery", "1"]+lever)
            self.assertIn("the skyline recovery decides the board's structure jointly",
                          stream.getvalue(), lever)

    def test_an_undeclared_value_is_refused_by_the_parser(self):
        for value in ("2", "-1", "true"):
            stream = io.StringIO()
            with self.subTest(value=value), contextlib.redirect_stderr(stream):
                with self.assertRaises(SystemExit):
                    self.arguments(self.base+["--occupancy-skyline-recovery", value])
            self.assertIn("invalid", stream.getvalue(), value)

    def test_both_settings_are_admitted_and_zero_is_the_published_default(self):
        self.assertEqual(self.arguments(self.base+["--profile-square", "--occupancy-skyline-recovery", "1"])
                         .occupancy_skyline_recovery, 1)
        self.assertEqual(self.arguments(self.base+["--profile-square"]).occupancy_skyline_recovery, 0)

    def test_the_recovery_is_admitted_after_the_three_declared_read_options(self):
        parsed = self.arguments(self.base+["--profile-square", "--occupancy-skyline-recovery", "1",
                                           "--occupancy-crossing-threshold", "1",
                                           "--occupancy-stack-monotone", "1", "--occupancy-centre", "1"])
        self.assertEqual((parsed.occupancy_skyline_recovery, parsed.occupancy_crossing_threshold,
                          parsed.occupancy_stack_monotone, parsed.occupancy_centre), (1, 1, 1, 1))

    def test_the_head_applies_the_recovery_last(self):
        source = Path(inspect.getsourcefile(ModelConfig)).read_text(encoding="utf-8")
        applied = source.index("skyline_shift(occupancy_logits, self.config.visible_rows)")
        for earlier in ("stack_contiguous(occupancy_logits", "centre_board(occupancy_logits",
                        "otsu_board(occupancy_logits"):
            self.assertLess(source.index(earlier), applied, earlier)
        self.assertGreater(source.index("stages.append(", applied), applied)

    def test_the_run_family_is_owned(self):
        from record_stream_run import OWNED_RUN_PATTERN
        for name in ("m17-skyline-control-r1", "m17-skyline-recovered-r1",
                     "m17-skyline-recovered-r1-sweep-r1-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        for name in ("m17-skyline-", "m17-skyline", "m17-skyliner-r1"):
            self.assertFalse(OWNED_RUN_PATTERN.fullmatch(name), name)
        for name in ("m16-crossing-ruled-r1", "m15-stack-combined-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)


class SkylineRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        scope = ROOT/"docs/evidence/m17-skyline-recovery-r1/scope.json"
        if not scope.exists():
            raise unittest.SkipTest("the skyline record has not been curated yet")
        cls.scope = json.loads(scope.read_text(encoding="utf-8"))
        cls.neighbour = json.loads((ROOT/"docs/evidence/m15-stack-combined-r1/scope.json")
                                   .read_text(encoding="utf-8"))
        cls.text = " ".join((ROOT/"docs/M17_SKYLINE_RECOVERY.md").read_text(encoding="utf-8").split())

    def test_the_arms_differ_in_exactly_the_declared_fields(self):
        self.assertEqual(sorted(self.scope["arms"]), ["control", "recovered"])
        self.assertEqual(self.scope["aliasing_guard"]["arms_differ_in"],
                         ["occupancy_centre", "occupancy_crossing_threshold", "occupancy_skyline_recovery",
                          "occupancy_stack_monotone"])
        self.assertEqual({name: (arm["occupancy_skyline_recovery"], arm["occupancy_crossing_threshold"],
                                 arm["occupancy_stack_monotone"], arm["occupancy_centre"])
                          for name, arm in self.scope["arms"].items()},
                         {"control": (0, 0, 0, 0), "recovered": (1, 1, 1, 1)})
        self.assertTrue(self.scope["aliasing_guard"]["weights_differ"])
        self.assertTrue(self.scope["aliasing_guard"]["not_the_previous_block"])

    def test_the_control_reproduces_and_the_neighbour_is_excluded(self):
        self.assertEqual(self.scope["arms"]["control"]["auc"], 0.8226433610683003)
        self.assertEqual(self.scope["arms"]["control"]["f1_at_zero"], 0.38164)
        self.assertNotEqual(self.scope["arms"]["recovered"]["weights_sha256"],
                            self.neighbour["arms"]["combined"]["weights_sha256"])

    def test_the_strict_exact_metric_is_reported_beside_the_boundary_one(self):
        for name, arm in self.scope["arms"].items():
            gate = arm["exact_boards_at_zero"]
            self.assertEqual(gate["boards"], 28, name)
            self.assertIn("boundary_exact", gate, name)
            self.assertEqual(self.scope["fit_gate"]["boundary_exact_nonempty_boards"],
                             self.scope["arms"]["recovered"]["exact_boards_at_zero"]["boundary_exact"])

    def test_the_option_adds_no_term(self):
        mechanism = self.scope["mechanism"]
        self.assertEqual(mechanism["loss_terms_added"], 0)
        self.assertEqual(mechanism["parameters_added"], 0)
        self.assertIsNone(mechanism["weight"])
        self.assertTrue(mechanism["label_free"])
        self.assertTrue(mechanism["declared_before_receipts"])

    def test_the_readings_follow_the_printed_numbers(self):
        arm, control = self.scope["arms"]["recovered"], self.scope["arms"]["control"]
        delta = round(arm["f1_at_zero"]-control["f1_at_zero"], 6)
        expected = ("skyline-clears-the-ceiling" if delta >= .05 else
                    "skyline-moves-without-clearing" if delta > 0 else "skyline-fails")
        self.assertEqual(self.scope["reading"], expected)
        self.assertEqual(self.scope["secondary_reading"],
                         "ranking-preserved" if round(arm["auc"]-control["auc"], 6) >= -.02 else "ranking-degraded")
        self.assertEqual(self.scope["gate_moved"],
                         arm["exact_boards_at_zero"]["exact"] > 0)

    def test_the_document_quotes_the_published_scope(self):
        for name, arm in self.scope["arms"].items():
            self.assertIn(f"{round(arm['auc'], 6):,.6f}", self.text, name)
            self.assertIn(f"{round(arm['f1_at_zero'], 6):,.6f}", self.text, name)
        self.assertIn(self.scope["reading"], self.text)
        rows = {name: arm for name, arm in self.scope["reference"].items() if isinstance(arm, dict)}
        self.assertGreaterEqual(len(rows), 8)
        for name, arm in rows.items():
            self.assertIn(f"{round(arm['auc'], 6):,.6f}", self.text, name)


if __name__ == "__main__":
    unittest.main()
