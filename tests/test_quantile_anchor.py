"""The class-quantile anchor: supervision of the empty mass rather than two cells of each class."""
import contextlib
import inspect
import io
import json
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from tetris_vision.losses import (ANCHOR_EMPTY_QUANTILE, ANCHOR_MARGIN, ANCHOR_OCCUPIED_QUANTILE, ANCHOR_WEIGHT,
                                  QUANTILE_INTERPOLATION, anchor_term, quantile_anchor_term,
                                  require_occupancy_quantile_anchor)

ROOT = Path(__file__).resolve().parents[1]
ROWS, COLUMNS, BATCH = 20, 10, 2


def board(occupied_values=(), empty_values=(), *, unknown_at=None):
    """A board whose known cells are exactly the named ones: everything else is unknown, not empty."""
    logits = torch.full((BATCH, ROWS+6, COLUMNS), -4.0)
    occupied = torch.zeros(BATCH, ROWS, COLUMNS, dtype=torch.long)
    known = torch.zeros(BATCH, ROWS, COLUMNS, dtype=torch.bool)
    cells = [(row, column) for row in range(ROWS) for column in range(COLUMNS)]
    for (row, column), value in zip(cells[:len(occupied_values)], occupied_values):
        logits[0, row, column] = value
        occupied[0, row, column] = 1
        known[0, row, column] = True
    for (row, column), value in zip(cells[len(occupied_values):], empty_values):
        logits[0, row, column] = value
        known[0, row, column] = True
    if unknown_at is not None:
        logits[0, unknown_at[0], unknown_at[1]] = 9.0
    return {"occupancy": logits}, {"occupied": occupied, "known": known}


def quantile(output, target):
    return float(quantile_anchor_term(output, target))


class QuantileAnchorTests(unittest.TestCase):
    def test_the_declared_constants_are_the_documented_ones(self):
        self.assertEqual((ANCHOR_MARGIN, ANCHOR_WEIGHT), (1.0, .5))
        self.assertEqual((ANCHOR_OCCUPIED_QUANTILE, ANCHOR_EMPTY_QUANTILE), (.10, .90))
        self.assertEqual(QUANTILE_INTERPOLATION, "linear")

    def test_a_correctly_anchored_board_is_free(self):
        self.assertEqual(quantile(*board([2.0]*10, [-2.0]*10)), 0.0)
        self.assertEqual(quantile(*board([5.0]*10, [-9.0]*10)), 0.0)

    def test_a_shifted_bulk_pays_both_halves(self):
        value = quantile(*board([.5]*10, [.5]*10))
        self.assertAlmostEqual(value, (ANCHOR_MARGIN-.5)+(ANCHOR_MARGIN+.5), places=5)

    def test_the_interpolation_convention_is_linear(self):
        # Ten occupied values [-4, 0, ..., 0]: h=(n-1)q=0.9, so q10 is -4+0.9*(0-(-4)) = -0.4. The `lower`
        # convention would return -4 (charging 5.0) and `higher` 0 (charging 1.0); this pins `linear`.
        value = quantile(*board([-4.0]+[0.0]*9, [-4.0]*10))
        self.assertAlmostEqual(value, ANCHOR_MARGIN-(-0.4), places=5)
        self.assertNotAlmostEqual(value, ANCHOR_MARGIN-(-4.0), places=3)

    def test_an_outlier_extreme_costs_the_extremal_anchor_and_not_this_one(self):
        # One occupied outlier at -3 and one empty outlier at +3: the extremal anchor pays 8.0 for the two cells,
        # while the tenth and ninetieth percentiles sit with the bulk and this term is free.
        output, target = board([-3.0]+[3.0]*9, [3.0]+[-2.0]*9)
        self.assertEqual(quantile(output, target), 0.0)
        self.assertAlmostEqual(float(anchor_term(output, target)), 2*(ANCHOR_MARGIN+3.0), places=5)

    def test_the_quantile_term_is_never_larger_than_the_extremal_term(self):
        # q10 >= min and q90 <= max, so on any single board this term charges at most what the extremal anchor
        # charges - the converse cannot occur. What differs is where the pressure lands: an extremal term can be
        # driven down by one sacrificed cell, a quantile term only by moving the body of the class.
        generator = torch.Generator().manual_seed(20261112)
        for _ in range(20):
            keys = board(torch.randn(12, generator=generator).tolist(),
                         torch.randn(12, generator=generator).tolist())
            self.assertLessEqual(quantile(*keys), float(anchor_term(*keys))+1e-6)

    def test_an_unknown_cell_is_never_treated_as_empty(self):
        named = ([2.0]*10, [-2.0]*10)
        without = quantile(*board(*named))
        with_unknown = quantile(*board(*named, unknown_at=(9, 9)))
        self.assertEqual(without, 0.0)
        self.assertEqual(with_unknown, 0.0)

    def test_a_board_without_one_of_the_classes_is_not_counted(self):
        output, target = board([.5]*10, [.5]*10)
        # Board 1 carries occupied cells only: it has no empty known cell, so it has no boundary to supervise.
        for column in range(COLUMNS):
            output["occupancy"][1, 0, column] = .5
            target["occupied"][1, 0, column] = 1
            target["known"][1, 0, column] = True
        expected = quantile(*board([.5]*10, [.5]*10))
        self.assertAlmostEqual(quantile(output, target), expected, places=6)
        self.assertGreater(expected, 0.0)

    def test_a_batch_with_nothing_to_judge_is_zero_and_still_differentiable(self):
        logits = torch.zeros(BATCH, ROWS+6, COLUMNS, requires_grad=True)
        value = quantile_anchor_term({"occupancy": logits},
                                     {"occupied": torch.zeros(BATCH, ROWS, COLUMNS, dtype=torch.long),
                                      "known": torch.zeros(BATCH, ROWS, COLUMNS, dtype=torch.bool)})
        self.assertEqual(float(value.detach()), 0.0)
        self.assertTrue(value.requires_grad)

    def test_a_binding_quantile_carries_gradient_to_the_body(self):
        output, target = board([.5]*10, [.5]*10)
        output["occupancy"].requires_grad_(True)
        quantile_anchor_term(output, target).backward()
        gradient = output["occupancy"].grad
        self.assertIsNotNone(gradient)
        self.assertGreater(int((gradient.abs() > 0).sum()), 1)

    def test_the_term_is_pooled_at_its_declared_weight(self):
        from tetris_vision.losses import joint_loss, stage_loss
        source = Path(inspect.getsourcefile(joint_loss)).read_text(encoding="utf-8")
        self.assertIn('"quantile": ANCHOR_WEIGHT*occupancy_quantile_anchor', source)
        for function in (joint_loss, stage_loss):
            self.assertIn("occupancy_quantile_anchor", inspect.signature(function).parameters)

    def test_zero_leaves_the_published_objective_untouched(self):
        # The flag adds a term and never scales anything, and its weight is exactly 0.0 at 0, so the published
        # objective is unchanged by construction; the training-path tests in this suite (test_model, test_components,
        # test_layer_evidence) run joint_loss on that exact default every time and would break if it were not.
        from tetris_vision.losses import joint_loss, stage_loss
        self.assertEqual(inspect.signature(stage_loss).parameters["occupancy_quantile_anchor"].default, 0)
        self.assertEqual(inspect.signature(joint_loss).parameters["occupancy_quantile_anchor"].default, 0)
        self.assertEqual(ANCHOR_WEIGHT*0, 0.0)
        source = Path(inspect.getsourcefile(stage_loss)).read_text(encoding="utf-8")
        guard = source.index("if occupancy_quantile_anchor:")
        assigned = source.index('result["quantile"]')
        self.assertGreater(assigned, guard)
        self.assertNotIn("result[\"quantile\"] *", source)
        self.assertNotIn("quantile", "".join(line for line in source.splitlines()
                                              if "result[\"quantile\"]" in line and "*" in line))

    def test_the_admission_join_refuses_a_mismatch(self):
        summary = {"arguments": {"occupancy_quantile_anchor": 1},
                   "recovery_contract": {"occupancy_quantile_anchor": 1}}
        self.assertIsNone(require_occupancy_quantile_anchor(1, summary))
        for value, wrong in ((0, summary), (2, summary),
                             (1, {"arguments": {"occupancy_quantile_anchor": 0},
                                  "recovery_contract": {"occupancy_quantile_anchor": 1}}),
                             (1, {"arguments": {}, "recovery_contract": {}})):
            with self.assertRaises(ValueError):
                require_occupancy_quantile_anchor(value, wrong)


class PlumbingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from train_m2_stream import arguments, validate_admission, main
        cls.source = Path(inspect.getsourcefile(validate_admission)).read_text(encoding="utf-8")
        cls.driver = inspect.getsource(main)
        cls.refusals = inspect.getsource(arguments)
        cls.signature = inspect.signature(validate_admission)
        cls.arguments = staticmethod(arguments)

    def test_the_driver_passes_the_lever_to_the_objective(self):
        self.assertEqual(self.driver.count(
            "occupancy_quantile_anchor=args.occupancy_quantile_anchor"), 4)

    def test_the_driver_records_the_lever_in_the_recovery_contract(self):
        self.assertIn('recovery_contract["occupancy_quantile_anchor"] = args.occupancy_quantile_anchor',
                      self.driver)

    def test_the_admission_checks_the_lever_it_receives(self):
        self.assertIn("occupancy_quantile_anchor", self.signature.parameters)
        self.assertEqual(self.signature.parameters["occupancy_quantile_anchor"].default, 0)
        self.assertIn("require_occupancy_quantile_anchor(occupancy_quantile_anchor, summary)", self.source)

    def test_the_quantile_anchor_refuses_to_stack_with_both_extremal_terms(self):
        self.assertIn("args.occupancy_quantile_anchor and (args.occupancy_anchor or "
                      "args.occupancy_margin_hinge", self.refusals)
        self.assertIn("two-sided anchor and the one-sided hinge", self.refusals)

    def test_the_trainer_refuses_every_arm_that_moves_the_read_or_zero(self):
        base = ["--output", "runs/m11-quantile-anchor-refusal-probe", "--model", "accuracy",
                "--steps", "2"]
        for lever in (["--occupancy-anchor", "1"], ["--occupancy-margin-hinge", "1"],
                      ["--occupancy-normalise", "1"], ["--occupancy-deform", "2", "--occupancy-detail", "1"],
                      ["--occupancy-offset-rows", "1", "--occupancy-detail", "1"],
                      ["--occupancy-detail-stencil", "1", "--occupancy-detail", "1"],
                      ["--occupancy-resolution", "1"], ["--occupancy-context", "1"],
                      ["--occupancy-column", "1"], ["--occupancy-lattice-free", "1"]):
            stream = io.StringIO()
            with self.subTest(lever=lever), contextlib.redirect_stderr(stream):
                with self.assertRaises(SystemExit):
                    self.arguments(base+["--occupancy-quantile-anchor", "1"]+lever)
            # The refusal that fired must be the declared one, not a later unrelated validation.
            self.assertIn("quantile anchor supervises", stream.getvalue(), lever)

    def test_the_quantile_anchor_alone_is_admitted(self):
        base = ["--output", "runs/m11-quantile-anchor-refusal-probe", "--model", "accuracy",
                "--steps", "2", "--profile-square"]
        parsed = self.arguments(base+["--occupancy-quantile-anchor", "1"])
        self.assertEqual(parsed.occupancy_quantile_anchor, 1)
        self.assertEqual(self.arguments(base).occupancy_quantile_anchor, 0)

    def test_the_run_family_is_owned(self):
        from record_stream_run import OWNED_RUN_PATTERN
        for name in ("m11-quantile-anchor-control-r1", "m11-quantile-anchor-bulk-r1",
                     "m11-quantile-anchor-bulk-r1-sweep-r1-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        for name in ("m11-quantile-anchor-", "m11-quantile-anchor", "m11-quantile-anchored-arm-r1"):
            self.assertFalse(OWNED_RUN_PATTERN.fullmatch(name), name)
        for name in ("m9-margin-control-r1", "m10-anchor-control-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)


class QuantileAnchorRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scope = ROOT/"docs/evidence/m11-quantile-anchor-r1/scope.json"
        if not scope.exists():
            raise unittest.SkipTest("the quantile-anchor record has not been curated yet")
        cls.scope = json.loads(scope.read_text(encoding="utf-8"))
        cls.prior = json.loads((ROOT/"docs/evidence/m10-supervised-anchor-r1/scope.json")
                               .read_text(encoding="utf-8"))
        cls.hinge = json.loads((ROOT/"docs/evidence/m9-supervised-margin-r1/scope.json")
                               .read_text(encoding="utf-8"))
        cls.text = " ".join((ROOT/"docs/M11_QUANTILE_ANCHOR.md").read_text(encoding="utf-8").split())

    def test_the_arms_and_their_supervision(self):
        self.assertEqual(sorted(self.scope["arms"]), ["bulk", "control"])
        self.assertEqual({name: arm["occupancy_quantile_anchor"]
                          for name, arm in self.scope["arms"].items()},
                         {"control": 0, "bulk": 1})
        for arm in self.scope["arms"].values():
            self.assertEqual(arm["occupancy_anchor"], 0)
            self.assertEqual(arm["occupancy_margin_hinge"], 0)

    def test_control_reproduces_the_recorded_control(self):
        self.assertEqual(self.scope["arms"]["control"]["auc"], self.prior["arms"]["control"]["auc"])
        self.assertEqual(self.scope["arms"]["control"]["f1_at_zero"],
                         self.prior["arms"]["control"]["f1_at_zero"])

    def test_the_declared_constants_are_the_ones_recorded(self):
        mechanism = self.scope["mechanism"]
        self.assertEqual((mechanism["margin"], mechanism["weight"]), (ANCHOR_MARGIN, ANCHOR_WEIGHT))
        self.assertEqual((mechanism["occupied_quantile"], mechanism["empty_quantile"]),
                         (ANCHOR_OCCUPIED_QUANTILE, ANCHOR_EMPTY_QUANTILE))
        self.assertEqual(mechanism["interpolation"], QUANTILE_INTERPOLATION)
        self.assertTrue(mechanism["declared_before_receipts"])
        self.assertIn("0.682", self.text)

    def test_the_readings_follow_the_printed_numbers(self):
        arm, control = self.scope["arms"]["bulk"], self.scope["arms"]["control"]
        delta = round(arm["f1_at_zero"]-control["f1_at_zero"], 6)
        expected = ("bulk-clears-the-boundary" if delta >= .05 else
                    "bulk-moves-without-clearing" if delta > 0 else "bulk-fails")
        self.assertEqual(self.scope["reading"], expected)
        auc_delta = round(arm["auc"]-control["auc"], 6)
        self.assertEqual(self.scope["secondary_reading"],
                         "ranking-preserved" if auc_delta >= -.02 else "ranking-degraded")

    def test_the_document_quotes_the_published_scope(self):
        for name, arm in self.scope["arms"].items():
            self.assertIn(f"{round(arm['auc'], 6):,.6f}", self.text, name)
            self.assertIn(f"{round(arm['f1_at_zero'], 6):,.6f}", self.text, name)
        self.assertIn(self.scope["reading"], self.text)
        self.assertIn(self.scope["secondary_reading"], self.text)

    def test_both_reference_rows_are_quoted_and_not_reverdicted(self):
        extremal = self.scope["reference"]["extremal_anchor"]
        hinge = self.scope["reference"]["hinge"]
        self.assertEqual(extremal["auc"], self.prior["arms"]["anchor"]["auc"])
        self.assertEqual(extremal["f1_at_zero"], self.prior["arms"]["anchor"]["f1_at_zero"])
        self.assertEqual(hinge["auc"], self.hinge["arms"]["hinge"]["auc"])
        self.assertIn("reference", self.scope["reference"]["role"])
        self.assertIn(f"{round(extremal['auc'], 6):,.6f}", self.text)
        self.assertIn(f"{round(hinge['auc'], 6):,.6f}", self.text)

    def test_the_paid_term_is_reported_from_the_study_log(self):
        paid = self.scope["arms"]["bulk"]["training_quantile"]
        self.assertEqual(paid["steps"], 64)
        self.assertTrue(paid["in_sample"])
        self.assertGreaterEqual(paid["mean"], 0.0)

    def test_the_mass_diagnostic_is_reported_for_both_arms(self):
        for name, arm in self.scope["arms"].items():
            self.assertEqual(arm["positives_at_zero"],
                             arm["operating_counts"]["true_positive"]
                             + arm["operating_counts"]["false_positive"], name)
            self.assertAlmostEqual(arm["positive_share_at_zero"],
                                   arm["positives_at_zero"]/5600, places=6)

    def test_the_fit_gate_is_reported_on_exact_boards(self):
        self.assertEqual(self.scope["fit_gate"]["boards"], 28)
        self.assertEqual(self.scope["fit_gate"]["cleared"],
                         self.scope["fit_gate"]["exact_nonempty_boards"] == 28)


if __name__ == "__main__":
    unittest.main()
