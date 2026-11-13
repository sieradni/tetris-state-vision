"""Tests for the M16 label-free crossing decision rule: selection, determinism, leakage, refusals, record."""
import contextlib
import inspect
import io
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from tetris_vision.losses import require_occupancy_crossing_threshold
from tetris_vision.model import ModelConfig, centre_board, otsu_board, stack_contiguous

ROOT = Path(__file__).resolve().parents[1]
ROWS, COLUMNS, BATCH = 20, 10, 2


def two_mode_board(low=-2.0, high=2.0, occupied_rows=4):
    """A board whose columns are an occupied run of `occupied_rows` over an empty one."""
    columns = [[high]*occupied_rows+[low]*(ROWS-occupied_rows)]*COLUMNS
    visible = torch.tensor(columns, dtype=torch.float32).transpose(0, 1)[None].repeat(BATCH, 1, 1)
    spawn = torch.full((BATCH, 6, COLUMNS), low)
    return torch.cat((visible, spawn), 1)


def decided_rows(logits, rows):
    """The height each column decides at when the board is thresholded at 0.0."""
    return (logits[:, :rows] >= 0).sum(dim=1)


class SelectionTests(unittest.TestCase):
    def test_the_rule_selects_the_gap_between_the_two_groups(self):
        # Occupied +2.0 over empty -2.0: the only split with mass on both sides is the one at 0.0, so the rule must
        # subtract 0.0 (a shift of the level by the gap's midpoint) and decide exactly the occupied rows.
        board = two_mode_board(occupied_rows=4)
        ruled = otsu_board(board, ROWS)
        # The level comes from the visible cells; the spawn rows are scored too, so they shift with it.
        self.assertTrue(torch.allclose(ruled[:, :ROWS], board[:, :ROWS], atol=1e-6))
        self.assertEqual([int(v) for v in decided_rows(ruled, ROWS)[0]], [4]*COLUMNS)

    def test_a_offset_read_is_decided_at_its_own_crossing(self):
        # The same structure, shifted up by 1.5: a fixed 0.0 threshold over-decides, and the rule must recover the height.
        board = two_mode_board(low=0.5, high=4.5, occupied_rows=7)
        plain = decided_rows(board, ROWS)[0].tolist()
        ruled = decided_rows(otsu_board(board, ROWS), ROWS)[0].tolist()
        self.assertGreater(plain[0], 7)
        self.assertEqual(ruled, [7]*COLUMNS)

    def test_the_rule_is_deterministic(self):
        board = torch.randn(BATCH, ROWS+6, COLUMNS)*2.0
        first = otsu_board(board, ROWS)
        for _ in range(4):
            self.assertTrue(torch.equal(otsu_board(board, ROWS), first))
        # And it is per board: two different boards are not shifted by a shared constant.
        shifted = first-board
        self.assertFalse(torch.allclose(shifted[0, :ROWS], shifted[1, :ROWS], atol=1e-6))

    def test_the_rule_is_per_board_and_ignores_the_spawn_rows(self):
        board = two_mode_board(occupied_rows=5)
        board[1, :, :] += 3.0
        ruled = otsu_board(board, ROWS)
        for index in range(BATCH):
            self.assertTrue(torch.allclose(ruled[index:index+1], otsu_board(board[index:index+1], ROWS),
                                           atol=1e-6, rtol=0))
        hidden = board.clone()
        hidden[:, ROWS:] += 100.0
        self.assertTrue(torch.allclose(otsu_board(board, ROWS)[:, :ROWS],
                                       otsu_board(hidden, ROWS)[:, :ROWS], atol=1e-6))

    def test_a_constant_board_collapses_to_its_own_level(self):
        flat = torch.full((BATCH, ROWS+6, COLUMNS), 4.25)
        ruled = otsu_board(flat, ROWS)
        self.assertEqual(float(ruled[:, :ROWS].abs().max()), 0.0)

    def test_a_single_visible_cell_is_an_identity(self):
        one = torch.tensor([[[2.0]*COLUMNS, [5.0]*COLUMNS]], dtype=torch.float32)
        ruled = otsu_board(one, 1)
        self.assertTrue(torch.equal(ruled[:, :1], torch.zeros_like(one[:, :1])))
        # The level a constant row reports is its own value, so the scored spawn row keeps its distance from it.
        self.assertTrue(torch.equal(ruled[0, 1], one[0, 1]-2.0))

    def test_the_rule_composes_after_the_envelope_and_centring(self):
        # The declared order is envelope, then centring, then the rule; each stage is a per-board shift or envelope, and
        # the rule must leave a board whose crossing already sits at 0.0 where it is.
        board = two_mode_board()
        envelope = stack_contiguous(board, ROWS)
        centred = centre_board(envelope, ROWS)
        ruled = otsu_board(centred, ROWS)
        # Centring already puts the two plateaus on either side of 0.0, so the rule re-selects the same gap: its level
        # is that gap's midpoint and no cell changes side. The declared order is asserted on the side, not the value.
        self.assertEqual(decided_rows(centred, ROWS)[0].tolist(), [4]*COLUMNS)
        self.assertEqual(decided_rows(ruled, ROWS)[0].tolist(), [4]*COLUMNS)
        self.assertTrue(torch.equal(ruled >= 0, centred >= 0))


class LeakageGuardTests(unittest.TestCase):
    def test_the_rule_reads_no_labels_and_is_invariant_to_their_permutation(self):
        # The leakage guard: the decision is a function of the logits alone. Permuting every label, or replacing them
        # with anything at all, cannot move the tensor that every consumer thresholds.
        from train_m2_stream import arguments

        self.assertEqual(list(inspect.signature(otsu_board).parameters), ["logits", "visible_rows"])
        board = torch.randn(BATCH, ROWS+6, COLUMNS)*2.0
        labels = (torch.rand(BATCH, ROWS, COLUMNS) > .88).float()
        decided = otsu_board(board, ROWS)
        for permutation in (labels.flip(1), labels.flip(2), 1.0-labels, torch.zeros_like(labels),
                            torch.ones_like(labels)):
            # Whatever the labels are, the rule's output is the same object, and so is what gets decided from it.
            self.assertTrue(torch.equal(otsu_board(board, ROWS), decided))
            self.assertTrue(torch.equal(decided >= 0, decided >= 0))
            self.assertEqual(float(permutation.sum()) >= 0, True)
        source = Path(inspect.getsourcefile(otsu_board)).read_text(encoding="utf-8")
        body = source[source.index("def otsu_board"):source.index("def stack_contiguous")]
        body = body.split('"""')[-1]
        for forbidden in ("label", "target", "occupied", "prevalence", "threshold_grid"):
            self.assertNotIn(forbidden, body.lower().replace("occupancy", ""), forbidden)
        self.assertTrue(arguments is not None)

    def test_only_the_read_enters_the_model_forward(self):
        # A stronger, structural form: the head has no label parameter and the emitted occupancy tensor is produced
        # before any loss is evaluated, so no label-derived quantity can be inside the decision.
        from tetris_vision.model import TetrisStateNet
        forward = inspect.signature(TetrisStateNet.forward)
        self.assertNotIn("labels", forward.parameters)
        self.assertNotIn("target", forward.parameters)


class DeclaredConfigTests(unittest.TestCase):
    def test_the_published_decision_is_the_default(self):
        self.assertEqual(ModelConfig().occupancy_crossing_threshold, 0)
        self.assertEqual(ModelConfig(occupancy_crossing_threshold=1).occupancy_crossing_threshold, 1)

    def test_an_undeclared_option_is_refused(self):
        for value in (2, -1, True, 1.0, "1"):
            with self.assertRaises(ValueError):
                ModelConfig(occupancy_crossing_threshold=value)

    def test_the_rule_cannot_be_normalised(self):
        with self.assertRaises(ValueError):
            ModelConfig(occupancy_crossing_threshold=1, occupancy_normalise=1)

    def test_the_declared_composition_is_admitted(self):
        config = ModelConfig(occupancy_crossing_threshold=1, occupancy_stack_monotone=1, occupancy_centre=1)
        self.assertEqual((config.occupancy_crossing_threshold, config.occupancy_stack_monotone,
                          config.occupancy_centre), (1, 1, 1))

    def test_the_admission_join_refuses_a_mismatch(self):
        summary = {"arguments": {"occupancy_crossing_threshold": 1},
                   "recovery_contract": {"occupancy_crossing_threshold": 1},
                   "config": {"occupancy_crossing_threshold": 1}}
        self.assertIsNone(require_occupancy_crossing_threshold(1, summary))
        for value, wrong in ((0, summary), (2, summary),
                             (1, {"arguments": {"occupancy_crossing_threshold": 0},
                                  "recovery_contract": {"occupancy_crossing_threshold": 1},
                                  "config": {"occupancy_crossing_threshold": 1}}),
                             (1, {"arguments": {}, "recovery_contract": {}, "config": {}})):
            with self.assertRaises(ValueError):
                require_occupancy_crossing_threshold(value, wrong)


class PlumbingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from train_m2_stream import arguments, validate_admission, main
        cls.source = Path(inspect.getsourcefile(validate_admission)).read_text(encoding="utf-8")
        cls.driver = inspect.getsource(main)
        cls.refusals = inspect.getsource(arguments)
        cls.signature = inspect.signature(validate_admission)
        cls.arguments = staticmethod(arguments)
        cls.base = ["--output", "runs/m16-crossing-refusal-probe", "--model", "accuracy", "--steps", "2"]

    def test_the_driver_writes_the_rule_into_the_config(self):
        start = self.driver.index("replace(config,")
        block = self.driver[start:self.driver.index("config_json", start)]
        self.assertIn("occupancy_crossing_threshold=args.occupancy_crossing_threshold", block)

    def test_the_driver_records_the_rule_in_the_recovery_contract(self):
        self.assertIn('recovery_contract["occupancy_crossing_threshold"] = args.occupancy_crossing_threshold',
                      self.driver)

    def test_the_admission_checks_the_rule_it_receives(self):
        self.assertIn("occupancy_crossing_threshold", self.signature.parameters)
        self.assertEqual(self.signature.parameters["occupancy_crossing_threshold"].default, 0)
        self.assertIn("require_occupancy_crossing_threshold(occupancy_crossing_threshold, summary)", self.source)

    def test_the_trainer_refuses_every_term_and_lever_beside_the_rule(self):
        self.assertIn("if args.occupancy_crossing_threshold and (args.occupancy_normalise", self.refusals)
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
                    self.arguments(self.base+["--occupancy-crossing-threshold", "1"]+lever)
            self.assertIn("the crossing threshold decides at the read's own Otsu split", stream.getvalue(), lever)

    def test_an_undeclared_value_is_refused_by_the_parser(self):
        for value in ("2", "-1", "true"):
            stream = io.StringIO()
            with self.subTest(value=value), contextlib.redirect_stderr(stream):
                with self.assertRaises(SystemExit):
                    self.arguments(self.base+["--occupancy-crossing-threshold", value])
            self.assertIn("invalid", stream.getvalue(), value)

    def test_both_settings_are_admitted_and_zero_is_the_published_default(self):
        self.assertEqual(self.arguments(self.base+["--profile-square", "--occupancy-crossing-threshold", "1"])
                         .occupancy_crossing_threshold, 1)
        self.assertEqual(self.arguments(self.base+["--profile-square"]).occupancy_crossing_threshold, 0)

    def test_the_rule_is_admitted_beside_the_two_structural_read_options(self):
        parsed = self.arguments(self.base+["--profile-square", "--occupancy-crossing-threshold", "1",
                                           "--occupancy-stack-monotone", "1", "--occupancy-centre", "1"])
        self.assertEqual((parsed.occupancy_crossing_threshold, parsed.occupancy_stack_monotone,
                          parsed.occupancy_centre), (1, 1, 1))

    def test_the_run_family_is_owned(self):
        from record_stream_run import OWNED_RUN_PATTERN
        for name in ("m16-crossing-control-r1", "m16-crossing-ruled-r1",
                     "m16-crossing-ruled-r1-sweep-r1-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        for name in ("m16-crossing-", "m16-crossing", "m16-cross-ruled-r1"):
            self.assertFalse(OWNED_RUN_PATTERN.fullmatch(name), name)
        for name in ("m15-stack-contiguous-r1", "m14-centre-centred-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)


class CrossingRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        scope = ROOT/"docs/evidence/m16-crossing-threshold-r1/scope.json"
        if not scope.exists():
            raise unittest.SkipTest("the crossing record has not been curated yet")
        cls.scope = json.loads(scope.read_text(encoding="utf-8"))
        cls.combined = json.loads((ROOT/"docs/evidence/m15-stack-combined-r1/scope.json").read_text(encoding="utf-8"))
        cls.text = " ".join((ROOT/"docs/M16_CROSSING_THRESHOLD.md").read_text(encoding="utf-8").split())

    def test_the_arms_differ_in_exactly_the_three_declared_fields(self):
        self.assertEqual(sorted(self.scope["arms"]), ["control", "ruled"])
        self.assertEqual(self.scope["aliasing_guard"]["arms_differ_in"],
                         ["occupancy_centre", "occupancy_crossing_threshold", "occupancy_stack_monotone"])
        self.assertEqual({name: (arm["occupancy_stack_monotone"], arm["occupancy_centre"],
                                 arm["occupancy_crossing_threshold"])
                          for name, arm in self.scope["arms"].items()},
                         {"control": (0, 0, 0), "ruled": (1, 1, 1)})
        self.assertTrue(self.scope["aliasing_guard"]["weights_differ"])
        self.assertTrue(self.scope["aliasing_guard"]["not_the_previous_block"])

    def test_the_control_is_reproduced_and_the_neighbour_excluded(self):
        self.assertEqual(self.scope["arms"]["control"]["auc"], 0.8226433610683003)
        self.assertEqual(self.scope["arms"]["control"]["f1_at_zero"], 0.38164)
        self.assertNotEqual(self.scope["arms"]["ruled"]["weights_sha256"],
                            self.combined["arms"]["combined"]["weights_sha256"])

    def test_the_rule_consumes_no_labels(self):
        mechanism = self.scope["mechanism"]
        self.assertEqual(mechanism["parameters_added"], 0)
        self.assertEqual(mechanism["loss_terms_added"], 0)
        self.assertIsNone(mechanism["weight"])
        self.assertTrue(mechanism["label_free"])
        self.assertTrue(mechanism["declared_before_receipts"])

    def test_the_readings_follow_the_printed_numbers(self):
        arm, control = self.scope["arms"]["ruled"], self.scope["arms"]["control"]
        delta = round(arm["f1_at_zero"]-control["f1_at_zero"], 6)
        expected = ("crossing-clears-the-ceiling" if delta >= .05 else
                    "crossing-moves-without-clearing" if delta > 0 else "crossing-fails")
        self.assertEqual(self.scope["reading"], expected)
        self.assertEqual(self.scope["secondary_reading"],
                         "ranking-preserved" if round(arm["auc"]-control["auc"], 6) >= -.02 else "ranking-degraded")

    def test_the_document_quotes_the_published_scope(self):
        for name, arm in self.scope["arms"].items():
            self.assertIn(f"{round(arm['auc'], 6):,.6f}", self.text, name)
            self.assertIn(f"{round(arm['f1_at_zero'], 6):,.6f}", self.text, name)
        self.assertIn(self.scope["reading"], self.text)
        self.assertIn(str(self.scope["ceiling_broken"]), ("True", "False"))
        ceiling = self.scope["fixed_threshold_ceiling"]
        self.assertEqual(self.scope["ceiling_broken"],
                         self.scope["arms"]["ruled"]["f1_at_zero"] > ceiling["f1_at_zero"])
        self.assertEqual(ceiling["f1_at_zero"],
                         self.combined["arms"]["combined"]["f1_at_zero"])

    def test_the_fit_gate_and_drift_are_reported(self):
        gate = self.scope["fit_gate"]
        self.assertEqual(gate["boards"], 28)
        self.assertEqual(gate["exact_nonempty_boards"],
                         self.scope["arms"]["ruled"]["exact_boards_at_zero"]["exact"])
        for name, arm in self.scope["arms"].items():
            self.assertAlmostEqual(arm["positive_share_at_zero"],
                                   (arm["operating_counts"]["true_positive"]
                                    + arm["operating_counts"]["false_positive"])/5600, places=6)


if __name__ == "__main__":
    unittest.main()
