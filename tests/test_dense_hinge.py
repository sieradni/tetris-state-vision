"""The dense class-balanced hinge: every known cell charged inside the band, both classes equally."""
import contextlib
import inspect
import io
import json
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from tetris_vision.losses import (ANCHOR_MARGIN, ANCHOR_WEIGHT, DENSE_EMPTY_BALANCE, DENSE_OCCUPIED_BALANCE,
                                  dense_hinge_term, quantile_anchor_term, require_occupancy_dense_hinge)

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


def dense(output, target):
    return float(dense_hinge_term(output, target))


class DenseHingeTests(unittest.TestCase):
    def test_the_declared_constants_are_the_documented_ones(self):
        self.assertEqual((ANCHOR_MARGIN, ANCHOR_WEIGHT), (1.0, .5))
        self.assertEqual((DENSE_OCCUPIED_BALANCE, DENSE_EMPTY_BALANCE), (.5, .5))

    def test_a_correctly_banded_board_is_free(self):
        self.assertEqual(dense(*board([2.0]*10, [-2.0]*10)), 0.0)
        self.assertEqual(dense(*board([1.0]*10, [-1.0]*10)), 0.0)

    def test_each_class_halves_the_charge(self):
        # Occupied at +0.5 falls 0.5 short of the band; empty at +0.5 falls 1.5 short plus the margin.
        both = dense(*board([.5]*10, [.5]*10))
        self.assertAlmostEqual(both, DENSE_OCCUPIED_BALANCE*(ANCHOR_MARGIN-.5)
                               + DENSE_EMPTY_BALANCE*(ANCHOR_MARGIN+.5), places=6)
        only_occupied = dense(*board([.5]*10, [-2.0]*10))
        only_empty = dense(*board([2.0]*10, [.5]*10))
        self.assertAlmostEqual(only_occupied, DENSE_OCCUPIED_BALANCE*(ANCHOR_MARGIN-.5), places=6)
        self.assertAlmostEqual(only_empty, DENSE_EMPTY_BALANCE*(ANCHOR_MARGIN+.5), places=6)
        self.assertAlmostEqual(both, only_occupied+only_empty, places=6)

    def test_the_term_matches_an_independent_reference(self):
        # Standing instrument (V-133): an expectation written from the implementation's own algebra cannot catch a
        # missing divisor or a wrong balance. This recomputes the declared formula in plain Python from the
        # declaration's words - explicit per-class means with explicit counts - and compares on random boards.
        generator = torch.Generator().manual_seed(20261113)
        for _ in range(6):
            occupied_values = torch.randn(9, generator=generator).tolist()
            empty_values = torch.randn(11, generator=generator).tolist()
            value = dense(*board(occupied_values, empty_values))
            occupied_means = sum(max(0.0, ANCHOR_MARGIN-v) for v in occupied_values)/len(occupied_values)
            empty_means = sum(max(0.0, ANCHOR_MARGIN+v) for v in empty_values)/len(empty_values)
            self.assertAlmostEqual(value, .5*occupied_means+.5*empty_means, places=5)

    def test_the_class_balance_is_a_weight_not_a_count(self):
        # Two occupied cells against eighteen empty ones charges exactly what ten against ten does.
        skewed = dense(*board([.5]*2, [.5]*18))
        even = dense(*board([.5]*10, [.5]*10))
        self.assertAlmostEqual(skewed, even, places=6)
        self.assertGreater(skewed, 0.0)

    def test_the_band_charges_cells_the_quantile_anchor_lets_through(self):
        # One occupied outlier at -3 and one empty outlier at +3: the quantile anchor is free (its tenth and
        # ninetieth percentiles sit with the bulk), while every cell inside the band is charged here.
        output, target = board([-3.0]+[3.0]*9, [3.0]+[-2.0]*9)
        self.assertEqual(float(quantile_anchor_term(output, target)), 0.0)
        # One cell of ten at 4.0 logits inside the band, so each class contributes a tenth of its charge.
        self.assertAlmostEqual(dense(output, target), DENSE_OCCUPIED_BALANCE*(ANCHOR_MARGIN+3.0)/10
                               + DENSE_EMPTY_BALANCE*(ANCHOR_MARGIN+3.0)/10, places=6)
        self.assertAlmostEqual(dense(output, target), .4, places=6)

    def test_a_board_one_cell_inside_the_band_is_not_free(self):
        self.assertGreater(dense(*board([.999]*10, [-2.0]*10)), 0.0)
        self.assertLess(dense(*board([.999]*10, [-2.0]*10)), .01)

    def test_an_unknown_cell_is_never_charged_or_counted_as_empty(self):
        named = ([2.0]*10, [-2.0]*10)
        self.assertEqual(dense(*board(*named, unknown_at=(9, 9))), dense(*board(*named)))
        # An unknown cell sitting above zero must not enter the empty mean either.
        self.assertEqual(dense(*board([.5]*10, [.5]*10, unknown_at=(9, 9))),
                         dense(*board([.5]*10, [.5]*10)))

    def test_a_board_without_one_of_the_classes_is_not_counted(self):
        output, target = board([.5]*10, [.5]*10)
        for column in range(COLUMNS):
            output["occupancy"][1, 0, column] = .5
            target["occupied"][1, 0, column] = 1
            target["known"][1, 0, column] = True
        self.assertAlmostEqual(dense(output, target), dense(*board([.5]*10, [.5]*10)), places=6)
        self.assertGreater(dense(output, target), 0.0)

    def test_a_batch_with_nothing_to_judge_is_zero_and_still_differentiable(self):
        logits = torch.zeros(BATCH, ROWS+6, COLUMNS, requires_grad=True)
        value = dense_hinge_term({"occupancy": logits},
                                 {"occupied": torch.zeros(BATCH, ROWS, COLUMNS, dtype=torch.long),
                                  "known": torch.zeros(BATCH, ROWS, COLUMNS, dtype=torch.bool)})
        self.assertEqual(float(value.detach()), 0.0)
        self.assertTrue(value.requires_grad)

    def test_the_gradient_reaches_the_whole_body_of_the_board(self):
        output, target = board([-2.0]*10, [2.0]*10)
        output["occupancy"].requires_grad_(True)
        dense_hinge_term(output, target).backward()
        gradient = output["occupancy"].grad
        self.assertIsNotNone(gradient)
        # Every one of the twenty banded cells carries gradient, not one summary cell.
        self.assertEqual(int((gradient.abs() > 0).sum()), 20)

    def test_the_term_is_pooled_at_its_declared_weight(self):
        from tetris_vision.losses import joint_loss, stage_loss
        source = Path(inspect.getsourcefile(joint_loss)).read_text(encoding="utf-8")
        self.assertIn('"dense": ANCHOR_WEIGHT*occupancy_dense_hinge', source)
        for function in (joint_loss, stage_loss):
            self.assertIn("occupancy_dense_hinge", inspect.signature(function).parameters)

    def test_zero_leaves_the_published_objective_untouched(self):
        from tetris_vision.losses import joint_loss, stage_loss
        self.assertEqual(inspect.signature(stage_loss).parameters["occupancy_dense_hinge"].default, 0)
        self.assertEqual(inspect.signature(joint_loss).parameters["occupancy_dense_hinge"].default, 0)
        self.assertEqual(ANCHOR_WEIGHT*0, 0.0)
        source = Path(inspect.getsourcefile(stage_loss)).read_text(encoding="utf-8")
        self.assertGreater(source.index('result["dense"]'), source.index("if occupancy_dense_hinge:"))

    def test_the_admission_join_refuses_a_mismatch(self):
        summary = {"arguments": {"occupancy_dense_hinge": 1},
                   "recovery_contract": {"occupancy_dense_hinge": 1}}
        self.assertIsNone(require_occupancy_dense_hinge(1, summary))
        for value, wrong in ((0, summary), (2, summary),
                             (1, {"arguments": {"occupancy_dense_hinge": 0},
                                  "recovery_contract": {"occupancy_dense_hinge": 1}}),
                             (1, {"arguments": {}, "recovery_contract": {}})):
            with self.assertRaises(ValueError):
                require_occupancy_dense_hinge(value, wrong)


class PlumbingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from train_m2_stream import arguments, validate_admission, main
        cls.source = Path(inspect.getsourcefile(validate_admission)).read_text(encoding="utf-8")
        cls.driver = inspect.getsource(main)
        cls.refusals = inspect.getsource(arguments)
        cls.signature = inspect.signature(validate_admission)
        cls.arguments = staticmethod(arguments)
        cls.base = ["--output", "runs/m12-dense-hinge-refusal-probe", "--model", "accuracy", "--steps", "2"]

    def test_the_driver_passes_the_lever_to_the_objective(self):
        self.assertEqual(self.driver.count("occupancy_dense_hinge=args.occupancy_dense_hinge"), 4)

    def test_the_driver_records_the_lever_in_the_recovery_contract(self):
        self.assertIn('recovery_contract["occupancy_dense_hinge"] = args.occupancy_dense_hinge', self.driver)

    def test_the_admission_checks_the_lever_it_receives(self):
        self.assertIn("occupancy_dense_hinge", self.signature.parameters)
        self.assertEqual(self.signature.parameters["occupancy_dense_hinge"].default, 0)
        self.assertIn("require_occupancy_dense_hinge(occupancy_dense_hinge, summary)", self.source)

    def test_the_dense_hinge_refuses_to_stack_with_all_three_summaries(self):
        self.assertIn("args.occupancy_dense_hinge and (args.occupancy_quantile_anchor or args.occupancy_anchor or",
                      self.refusals)
        self.assertIn("args.occupancy_margin_hinge", self.refusals)
        self.assertIn("quantile anchor, the two-sided anchor and the one-sided hinge", self.refusals)

    def test_the_trainer_refuses_every_arm_that_moves_the_read_or_zero(self):
        for lever in (["--occupancy-quantile-anchor", "1"], ["--occupancy-anchor", "1"],
                      ["--occupancy-margin-hinge", "1"], ["--occupancy-normalise", "1"],
                      ["--occupancy-deform", "2", "--occupancy-detail", "1"],
                      ["--occupancy-offset-rows", "1", "--occupancy-detail", "1"],
                      ["--occupancy-detail-stencil", "1", "--occupancy-detail", "1"],
                      ["--occupancy-resolution", "1"], ["--occupancy-context", "1"],
                      ["--occupancy-column", "1"], ["--occupancy-lattice-free", "1"]):
            stream = io.StringIO()
            with self.subTest(lever=lever), contextlib.redirect_stderr(stream):
                with self.assertRaises(SystemExit):
                    self.arguments(self.base+["--occupancy-dense-hinge", "1"]+lever)
            self.assertIn("dense hinge supervises", stream.getvalue(), lever)

    def test_the_dense_hinge_alone_is_admitted(self):
        parsed = self.arguments(self.base+["--profile-square", "--occupancy-dense-hinge", "1"])
        self.assertEqual(parsed.occupancy_dense_hinge, 1)
        self.assertEqual(self.arguments(self.base+["--profile-square"]).occupancy_dense_hinge, 0)

    def test_the_run_family_is_owned(self):
        from record_stream_run import OWNED_RUN_PATTERN
        for name in ("m12-dense-hinge-control-r1", "m12-dense-hinge-banded-r1",
                     "m12-dense-hinge-banded-r1-sweep-r1-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        for name in ("m12-dense-hinge-", "m12-dense-hinge", "m12-dense-hinged-arm-r1"):
            self.assertFalse(OWNED_RUN_PATTERN.fullmatch(name), name)
        for name in ("m9-margin-control-r1", "m10-anchor-control-r1", "m11-quantile-anchor-control-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)


class DenseHingeRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scope = ROOT/"docs/evidence/m12-dense-hinge-r1/scope.json"
        if not scope.exists():
            raise unittest.SkipTest("the dense-hinge record has not been curated yet")
        cls.scope = json.loads(scope.read_text(encoding="utf-8"))
        cls.prior = json.loads((ROOT/"docs/evidence/m10-supervised-anchor-r1/scope.json")
                               .read_text(encoding="utf-8"))
        cls.quantile = json.loads((ROOT/"docs/evidence/m11-quantile-anchor-r1/scope.json")
                                  .read_text(encoding="utf-8"))
        cls.hinge = json.loads((ROOT/"docs/evidence/m9-supervised-margin-r1/scope.json")
                               .read_text(encoding="utf-8"))
        cls.text = " ".join((ROOT/"docs/M12_DENSE_HINGE.md").read_text(encoding="utf-8").split())

    def test_the_arms_and_their_supervision(self):
        self.assertEqual(sorted(self.scope["arms"]), ["banded", "control"])
        self.assertEqual({name: arm["occupancy_dense_hinge"]
                          for name, arm in self.scope["arms"].items()},
                         {"control": 0, "banded": 1})
        for arm in self.scope["arms"].values():
            self.assertEqual(arm["occupancy_quantile_anchor"], 0)
            self.assertEqual(arm["occupancy_anchor"], 0)
            self.assertEqual(arm["occupancy_margin_hinge"], 0)

    def test_control_reproduces_the_recorded_control(self):
        self.assertEqual(self.scope["arms"]["control"]["auc"], self.prior["arms"]["control"]["auc"])
        self.assertEqual(self.scope["arms"]["control"]["f1_at_zero"],
                         self.prior["arms"]["control"]["f1_at_zero"])

    def test_the_declared_constants_are_the_ones_recorded(self):
        mechanism = self.scope["mechanism"]
        self.assertEqual((mechanism["margin"], mechanism["weight"]), (ANCHOR_MARGIN, ANCHOR_WEIGHT))
        self.assertEqual((mechanism["occupied_balance"], mechanism["empty_balance"]),
                         (DENSE_OCCUPIED_BALANCE, DENSE_EMPTY_BALANCE))
        self.assertTrue(mechanism["declared_before_receipts"])
        self.assertEqual(mechanism["expected_reading"], "dense-clears-the-boundary")

    def test_the_readings_follow_the_printed_numbers(self):
        arm, control = self.scope["arms"]["banded"], self.scope["arms"]["control"]
        delta = round(arm["f1_at_zero"]-control["f1_at_zero"], 6)
        expected = ("dense-clears-the-boundary" if delta >= .05 else
                    "dense-moves-without-clearing" if delta > 0 else "dense-fails")
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

    def test_all_reference_rows_are_quoted_and_not_reverdicted(self):
        extremal = self.scope["reference"]["extremal_anchor"]
        quantile = self.scope["reference"]["quantile_anchor"]
        hinge = self.scope["reference"]["hinge"]
        self.assertEqual(extremal["auc"], self.prior["arms"]["anchor"]["auc"])
        self.assertEqual(quantile["auc"], self.quantile["arms"]["bulk"]["auc"])
        self.assertEqual(hinge["auc"], self.hinge["arms"]["hinge"]["auc"])
        self.assertIn("reference", self.scope["reference"]["role"])
        for arm in (extremal, quantile, hinge):
            self.assertIn(f"{round(arm['auc'], 6):,.6f}", self.text)

    def test_the_fit_gate_is_the_zero_boundary_implication(self):
        gate = self.scope["fit_gate"]
        self.assertEqual(gate["boundary"], 0.0)
        self.assertEqual(gate["measured_on"], "banded")
        self.assertEqual(gate["boards"], 28)
        self.assertEqual(gate["exact_nonempty_boards"],
                         self.scope["arms"]["banded"]["exact_boards_at_zero"]["exact"])
        self.assertEqual(gate["cleared"], gate["exact_nonempty_boards"] == 28)

    def test_the_paid_charge_is_reported_from_the_study_log(self):
        paid = self.scope["arms"]["banded"]["training_dense"]
        self.assertEqual(paid["steps"], 64)
        self.assertTrue(paid["in_sample"])
        self.assertGreaterEqual(paid["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
