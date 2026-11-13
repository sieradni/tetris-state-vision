"""Per-column stack contiguity of the occupancy read: the structure places the crossing, not a term."""
import contextlib
import inspect
import io
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from tetris_vision.losses import require_occupancy_stack_monotone
from tetris_vision.model import ModelConfig, centre_board, stack_contiguous

ROOT = Path(__file__).resolve().parents[1]
ROWS, COLUMNS, BATCH = 20, 10, 2
# A column that already is a stack: occupied below row 10, empty above it, in bottom-first order.
STEP = [2.5]*10+[-1.5]*10


def board(columns=None):
    """A synthetic read: 20 visible rows plus six spawn rows, as the model produces."""
    if columns is None:
        columns = [STEP]*COLUMNS
    visible = torch.tensor(columns, dtype=torch.float32).transpose(0, 1)[None].repeat(BATCH, 1, 1)
    spawn = torch.linspace(-2.0, 3.0, BATCH*6*COLUMNS).reshape(BATCH, 6, COLUMNS)
    return torch.cat((visible, spawn), 1)


def crossings(column):
    """Sign changes along a bottom-first column, counting a zero as its predecessor's side."""
    signs = [value >= 0 for value in column]
    return sum(1 for left, right in zip(signs, signs[1:]) if left != right)


class StackContiguousTests(unittest.TestCase):
    def test_each_column_becomes_one_step_with_a_single_crossing(self):
        latent = torch.tensor([[[3.0, -1.0], [-1.0, 2.0], [2.0, -2.0], [-3.0, 1.0]]])
        envelope = stack_contiguous(latent, 4)
        for index in range(2):
            column = [float(value) for value in envelope[0, :, index]]
            self.assertLessEqual(crossings(column), 1, column)
            self.assertGreaterEqual(column[0], column[-1])
        # The envelope is a maximum, so it never lowers a read and raises the cells under a high one.
        self.assertTrue(bool(torch.all(envelope >= latent)))
        # Suffix maximum in bottom-first order: every cell reads the highest material at or above it.
        self.assertEqual([float(value) for value in envelope[0, :, 0]], [3.0, 2.0, 2.0, -3.0])
        self.assertEqual([float(value) for value in envelope[0, :, 1]], [2.0, 2.0, 1.0, 1.0])

    def test_a_hole_is_raised_to_the_material_above_it(self):
        hole = list(STEP)
        hole[5] = -3.0
        envelope = stack_contiguous(board([hole]+[STEP]*(COLUMNS-1)), ROWS)
        self.assertEqual(float(envelope[0, 5, 0]), 2.5)
        self.assertTrue(bool(torch.all(envelope[0, :10, 0] >= 2.5)))

    def test_a_floating_overhang_is_projected_downward(self):
        # The declared failure mode, asserted as behaviour rather than discovered later: a monotone column cannot
        # represent material above empty space, so the space under it is filled.
        overhang = [-3.0]*5+[2.0]*5+[-1.5]*10
        envelope = stack_contiguous(board([overhang]+[STEP]*(COLUMNS-1)), ROWS)
        self.assertEqual([float(value) for value in envelope[0, :5, 0]], [2.0]*5)
        self.assertEqual([float(value) for value in envelope[0, 10:ROWS, 0]], [-1.5]*10)

    def test_an_already_monotone_column_is_unchanged_exactly(self):
        latent = board()
        self.assertTrue(torch.equal(stack_contiguous(latent, ROWS)[:, :ROWS], latent[:, :ROWS]))

    def test_a_constant_column_is_unchanged(self):
        flat = torch.full((BATCH, ROWS+6, COLUMNS), 4.25)
        self.assertTrue(torch.equal(stack_contiguous(flat, ROWS)[:, :ROWS], flat[:, :ROWS]))

    def test_a_single_visible_cell_is_an_identity(self):
        one = torch.tensor([[[2.0]*COLUMNS, [5.0]*COLUMNS]], dtype=torch.float32)
        self.assertTrue(torch.equal(stack_contiguous(one, 1)[:, :1], one[:, :1]))

    def test_the_envelope_is_equivariant_to_a_board_shift(self):
        # Unlike centring this transform does not remove the level - it places the crossing - so the honest property
        # is equivariance: a shift of the whole board shifts the envelope by the same constant and nothing else.
        latent = board()
        for shift in (7.5, -3.25, 0.0):
            self.assertTrue(torch.allclose(stack_contiguous(latent+shift, ROWS),
                                           stack_contiguous(latent, ROWS)+shift, atol=1e-5, rtol=0))
        # And the crossing really does move with the level: that is the freedom this arm leaves open.
        shifted = stack_contiguous(latent+4.0, ROWS)
        self.assertGreater(int((shifted[:, :ROWS] >= 0).sum()), int((stack_contiguous(latent, ROWS)[:, :ROWS] >= 0).sum()))

    def test_the_envelope_is_per_column_and_per_board(self):
        latent = board()
        envelope = stack_contiguous(latent, ROWS)
        for index in range(BATCH):
            self.assertTrue(torch.equal(envelope[index:index+1], stack_contiguous(latent[index:index+1], ROWS)))
        # Rewriting one board's column 0 cannot move any other board or column.
        changed = latent.clone()
        changed[1, :, 0] = 9.0
        moved = stack_contiguous(changed, ROWS)
        self.assertTrue(torch.equal(moved[0], envelope[0]))
        self.assertTrue(torch.equal(moved[:, :, 1:], envelope[:, :, 1:]))

    def test_the_spawn_rows_pass_through_untouched(self):
        latent = board()
        envelope = stack_contiguous(latent, ROWS)
        self.assertTrue(torch.equal(envelope[:, ROWS:], latent[:, ROWS:]))
        # Nothing above the visible window may raise a visible cell: the envelope reads only the board.
        hidden = latent.clone()
        hidden[:, ROWS:] += 100.0
        self.assertTrue(torch.equal(stack_contiguous(hidden, ROWS)[:, :ROWS], envelope[:, :ROWS]))


class DeclaredConfigTests(unittest.TestCase):
    def test_the_published_function_is_the_default(self):
        self.assertEqual(ModelConfig().occupancy_stack_monotone, 0)
        self.assertEqual(ModelConfig(occupancy_stack_monotone=1).occupancy_stack_monotone, 1)

    def test_an_undeclared_option_is_refused(self):
        for value in (2, -1, True, 1.0, "1"):
            with self.assertRaises(ValueError):
                ModelConfig(occupancy_stack_monotone=value)

    def test_the_monotone_read_cannot_be_normalised(self):
        with self.assertRaises(ValueError):
            ModelConfig(occupancy_stack_monotone=1, occupancy_normalise=1)

    def test_the_monotone_read_composes_with_centring_as_the_declared_follow_on(self):
        config = ModelConfig(occupancy_stack_monotone=1, occupancy_centre=1)
        self.assertEqual((config.occupancy_stack_monotone, config.occupancy_centre), (1, 1))

    def test_the_admission_join_refuses_a_mismatch(self):
        summary = {"arguments": {"occupancy_stack_monotone": 1},
                   "recovery_contract": {"occupancy_stack_monotone": 1},
                   "config": {"occupancy_stack_monotone": 1}}
        self.assertIsNone(require_occupancy_stack_monotone(1, summary))
        for value, wrong in ((0, summary), (2, summary),
                             (1, {"arguments": {"occupancy_stack_monotone": 0},
                                  "recovery_contract": {"occupancy_stack_monotone": 1},
                                  "config": {"occupancy_stack_monotone": 1}}),
                             (1, {"arguments": {"occupancy_stack_monotone": 1},
                                  "recovery_contract": {"occupancy_stack_monotone": 0},
                                  "config": {"occupancy_stack_monotone": 1}}),
                             (1, {"arguments": {}, "recovery_contract": {}, "config": {}})):
            with self.assertRaises(ValueError):
                require_occupancy_stack_monotone(value, wrong)


class HeadPlacementTests(unittest.TestCase):
    def test_the_head_applies_the_envelope_before_the_stages_are_emitted(self):
        source = Path(inspect.getsourcefile(ModelConfig)).read_text(encoding="utf-8")
        guard = source.index("if self.config.occupancy_stack_monotone:")
        applied = source.index("stack_contiguous(occupancy_logits, self.config.visible_rows)")
        emitted = source.index("stages.append(", applied)
        self.assertGreater(applied, guard)
        self.assertGreater(emitted, applied)
        # The envelope places the crossing and centring then moves the level, so the declared order is fixed.
        self.assertLess(applied, source.index("centre_board(occupancy_logits", applied))

    def test_the_envelope_is_not_a_loss_term(self):
        from tetris_vision import losses
        term_source = Path(inspect.getsourcefile(losses)).read_text(encoding="utf-8")
        self.assertNotIn("def stack_term", term_source)
        self.assertNotIn('"stack_monotone":', term_source)
        self.assertIn("def require_occupancy_stack_monotone", term_source)

    def test_centring_after_the_envelope_shifts_but_does_not_reorder(self):
        latent = board([[-3.0]*5+[2.0]*5+[-1.5]*10]+[STEP]*(COLUMNS-1))
        envelope = stack_contiguous(latent, ROWS)
        combined = centre_board(stack_contiguous(latent, ROWS), ROWS)
        for index in range(BATCH):
            self.assertAlmostEqual(float(combined[index, :ROWS].mean()), 0.0, places=5)
            pairs = envelope[index, :ROWS].flatten()[:, None]-envelope[index, :ROWS].flatten()[None, :]
            kept = combined[index, :ROWS].flatten()[:, None]-combined[index, :ROWS].flatten()[None, :]
            self.assertTrue(torch.allclose(kept, pairs, atol=1e-5))


class PlumbingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from train_m2_stream import arguments, validate_admission, main
        cls.source = Path(inspect.getsourcefile(validate_admission)).read_text(encoding="utf-8")
        cls.driver = inspect.getsource(main)
        cls.refusals = inspect.getsource(arguments)
        cls.signature = inspect.signature(validate_admission)
        cls.arguments = staticmethod(arguments)
        cls.base = ["--output", "runs/m15-stack-refusal-probe", "--model", "accuracy", "--steps", "2"]

    def test_the_driver_writes_the_lever_into_the_config(self):
        start = self.driver.index("replace(config,")
        block = self.driver[start:self.driver.index("config_json", start)]
        self.assertIn("occupancy_stack_monotone=args.occupancy_stack_monotone", block)

    def test_the_driver_records_the_lever_in_the_recovery_contract(self):
        self.assertIn('recovery_contract["occupancy_stack_monotone"] = args.occupancy_stack_monotone', self.driver)

    def test_the_admission_checks_the_lever_it_receives(self):
        self.assertIn("occupancy_stack_monotone", self.signature.parameters)
        self.assertEqual(self.signature.parameters["occupancy_stack_monotone"].default, 0)
        self.assertIn("require_occupancy_stack_monotone(occupancy_stack_monotone, summary)", self.source)

    def test_the_trainer_refuses_every_term_and_lever_beside_the_monotone_read(self):
        self.assertIn("if args.occupancy_stack_monotone and (args.occupancy_normalise", self.refusals)
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
                    self.arguments(self.base+["--occupancy-stack-monotone", "1"]+lever)
            self.assertIn("the stack-contiguous read places the crossing by column structure",
                          stream.getvalue(), lever)

    def test_an_undeclared_value_is_refused_by_the_parser(self):
        for value in ("2", "-1", "true"):
            stream = io.StringIO()
            with self.subTest(value=value), contextlib.redirect_stderr(stream):
                with self.assertRaises(SystemExit):
                    self.arguments(self.base+["--occupancy-stack-monotone", value])
            self.assertIn("invalid", stream.getvalue(), value)

    def test_both_settings_are_admitted_and_zero_is_the_published_default(self):
        parsed = self.arguments(self.base+["--profile-square", "--occupancy-stack-monotone", "1"])
        self.assertEqual(parsed.occupancy_stack_monotone, 1)
        self.assertEqual(self.arguments(self.base+["--profile-square"]).occupancy_stack_monotone, 0)
        self.assertEqual(self.arguments(self.base+["--profile-square", "--occupancy-stack-monotone", "0"])
                         .occupancy_stack_monotone, 0)

    def test_the_monotone_read_is_admitted_beside_centring(self):
        parsed = self.arguments(self.base+["--profile-square", "--occupancy-stack-monotone", "1",
                                           "--occupancy-centre", "1"])
        self.assertEqual((parsed.occupancy_stack_monotone, parsed.occupancy_centre), (1, 1))

    def test_the_run_family_is_owned(self):
        from record_stream_run import OWNED_RUN_PATTERN
        for name in ("m15-stack-control-r1", "m15-stack-contiguous-r1",
                     "m15-stack-contiguous-r1-sweep-r1-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        for name in ("m15-stack-", "m15-stack", "m15-stacked-arm-r1"):
            self.assertFalse(OWNED_RUN_PATTERN.fullmatch(name), name)
        for name in ("m14-centre-control-r1", "m13-midpoint-control-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)


class ContiguityRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        scope = ROOT/"docs/evidence/m15-stack-contiguity-r1/scope.json"
        if not scope.exists():
            raise unittest.SkipTest("the contiguity record has not been curated yet")
        cls.scope = json.loads(scope.read_text(encoding="utf-8"))
        cls.prior = json.loads((ROOT/"docs/evidence/m10-supervised-anchor-r1/scope.json")
                               .read_text(encoding="utf-8"))
        cls.centring = json.loads((ROOT/"docs/evidence/m14-read-centring-r1/scope.json")
                                  .read_text(encoding="utf-8"))
        cls.text = " ".join((ROOT/"docs/M15_STACK_CONTIGUITY.md").read_text(encoding="utf-8").split())

    def test_the_arms_and_the_single_declared_difference(self):
        self.assertEqual(sorted(self.scope["arms"]), ["contiguous", "control"])
        self.assertEqual({name: arm["occupancy_stack_monotone"] for name, arm in self.scope["arms"].items()},
                         {"control": 0, "contiguous": 1})
        self.assertEqual(self.scope["aliasing_guard"]["arms_differ_in"], ["occupancy_stack_monotone"])
        self.assertTrue(self.scope["aliasing_guard"]["weights_differ"])
        self.assertTrue(self.scope["aliasing_guard"]["not_the_previous_block"])

    def test_control_reproduces_the_recorded_control(self):
        self.assertEqual(self.scope["arms"]["control"]["auc"], self.prior["arms"]["control"]["auc"])
        self.assertEqual(self.scope["arms"]["control"]["f1_at_zero"],
                         self.prior["arms"]["control"]["f1_at_zero"])

    def test_the_structural_option_adds_no_term(self):
        mechanism = self.scope["mechanism"]
        self.assertEqual(mechanism["loss_terms_added"], 0)
        self.assertIsNone(mechanism["weight"])
        self.assertIsNone(mechanism["margin"])
        self.assertEqual(mechanism["parameters_added"], 0)
        self.assertTrue(mechanism["declared_before_receipts"])

    def test_the_readings_follow_the_printed_numbers(self):
        arm, control = self.scope["arms"]["contiguous"], self.scope["arms"]["control"]
        delta = round(arm["f1_at_zero"]-control["f1_at_zero"], 6)
        expected = ("stack-clears-the-crossing" if delta >= .05 else
                    "stack-moves-without-clearing" if delta > 0 else "stack-fails")
        self.assertEqual(self.scope["reading"], expected)
        self.assertEqual(self.scope["secondary_reading"],
                         "ranking-preserved" if round(arm["auc"]-control["auc"], 6) >= -.02 else "ranking-degraded")

    def test_the_document_quotes_the_published_scope(self):
        for name, arm in self.scope["arms"].items():
            self.assertIn(f"{round(arm['auc'], 6):,.6f}", self.text, name)
            self.assertIn(f"{round(arm['f1_at_zero'], 6):,.6f}", self.text, name)
        self.assertIn(self.scope["reading"], self.text)
        self.assertIn(self.scope["secondary_reading"], self.text)

    def test_every_reference_row_is_quoted(self):
        rows = {name: arm for name, arm in self.scope["reference"].items() if isinstance(arm, dict)}
        self.assertEqual(len(rows), 6)
        for name, arm in rows.items():
            self.assertIn(f"{round(arm['auc'], 6):,.6f}", self.text, name)

    def test_the_fit_gate_is_reported_from_the_treated_arm(self):
        gate = self.scope["fit_gate"]
        self.assertEqual(gate["boards"], 28)
        self.assertEqual(gate["exact_nonempty_boards"],
                         self.scope["arms"]["contiguous"]["exact_boards_at_zero"]["exact"])


class CombinedRecordTests(unittest.TestCase):
    """The declared follow-on: two structural fields, one control, its own checkpoint."""

    @classmethod
    def setUpClass(cls):
        import json
        scope = ROOT/"docs/evidence/m15-stack-combined-r1/scope.json"
        if not scope.exists():
            raise unittest.SkipTest("the combined record has not been curated yet")
        cls.scope = json.loads(scope.read_text(encoding="utf-8"))
        cls.envelope = json.loads((ROOT/"docs/evidence/m15-stack-contiguity-r1/scope.json")
                                  .read_text(encoding="utf-8"))
        cls.text = " ".join((ROOT/"docs/M15_STACK_CONTIGUITY.md").read_text(encoding="utf-8").split())

    def test_the_arms_differ_in_exactly_the_two_declared_fields(self):
        self.assertEqual(sorted(self.scope["arms"]), ["combined", "control"])
        self.assertEqual(self.scope["aliasing_guard"]["arms_differ_in"],
                         ["occupancy_centre", "occupancy_stack_monotone"])
        self.assertEqual({name: (arm["occupancy_stack_monotone"], arm["occupancy_centre"])
                          for name, arm in self.scope["arms"].items()},
                         {"control": (0, 0), "combined": (1, 1)})
        self.assertTrue(self.scope["aliasing_guard"]["weights_differ"])
        self.assertTrue(self.scope["aliasing_guard"]["not_the_previous_block"])

    def test_the_control_is_the_shared_one_and_the_envelope_arm_is_excluded_by_name(self):
        self.assertEqual(self.scope["arms"]["control"]["weights_sha256"],
                         self.envelope["arms"]["control"]["weights_sha256"])
        self.assertNotEqual(self.scope["arms"]["combined"]["weights_sha256"],
                            self.envelope["arms"]["contiguous"]["weights_sha256"])

    def test_the_readings_follow_the_printed_numbers(self):
        arm, control = self.scope["arms"]["combined"], self.scope["arms"]["control"]
        delta = round(arm["f1_at_zero"]-control["f1_at_zero"], 6)
        expected = ("stack-clears-the-crossing" if delta >= .05 else
                    "stack-moves-without-clearing" if delta > 0 else "stack-fails")
        self.assertEqual(self.scope["reading"], expected)
        self.assertEqual(self.scope["secondary_reading"],
                         "ranking-preserved" if round(arm["auc"]-control["auc"], 6) >= -.02 else "ranking-degraded")

    def test_the_document_quotes_the_combined_scope(self):
        for name, arm in self.scope["arms"].items():
            self.assertIn(f"{round(arm['auc'], 6):,.6f}", self.text, name)
            self.assertIn(f"{round(arm['f1_at_zero'], 6):,.6f}", self.text, name)
        rows = {name: arm for name, arm in self.scope["reference"].items() if isinstance(arm, dict)}
        # envelope arm plus the five supervised references; the M14 centring row belongs to the M15
        # block's own scope and is quoted in the earlier table of the same document.
        self.assertEqual(len(rows), 6)
        for name, arm in rows.items():
            self.assertIn(f"{round(arm['auc'], 6):,.6f}", self.text, name)
        self.assertIn(self.scope["reading"], self.text)

    def test_the_fit_gate_is_reported_from_the_combined_arm(self):
        gate = self.scope["fit_gate"]
        self.assertEqual(gate["boards"], 28)
        self.assertEqual(gate["exact_nonempty_boards"],
                         self.scope["arms"]["combined"]["exact_boards_at_zero"]["exact"])


if __name__ == "__main__":
    unittest.main()
