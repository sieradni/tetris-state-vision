"""The midpoint anchor: where the boundary sits, with no separation claim."""
import contextlib
import inspect
import io
import json
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from tetris_vision.losses import (ANCHOR_WEIGHT, MIDPOINT_EMPTY_BALANCE, MIDPOINT_OCCUPIED_BALANCE, MIDPOINT_TARGET,
                                  dense_hinge_term, midpoint_term, require_occupancy_midpoint)

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


def midpoint(output, target):
    return float(midpoint_term(output, target))


class MidpointTermTests(unittest.TestCase):
    def test_the_declared_constants_are_the_documented_ones(self):
        self.assertEqual(ANCHOR_WEIGHT, .5)
        self.assertEqual(MIDPOINT_TARGET, 0.0)
        self.assertEqual((MIDPOINT_OCCUPIED_BALANCE, MIDPOINT_EMPTY_BALANCE), (.5, .5))

    def test_a_symmetric_board_is_free(self):
        self.assertEqual(midpoint(*board([2.0]*10, [-2.0]*10)), 0.0)
        self.assertEqual(midpoint(*board([-3.0]*10, [3.0]*10)), 0.0)

    def test_the_charge_is_the_distance_of_the_midpoint(self):
        self.assertAlmostEqual(midpoint(*board([2.0]*10, [2.0]*10)), 2.0, places=6)
        self.assertAlmostEqual(midpoint(*board([-2.0]*10, [-2.0]*10)), 2.0, places=6)
        self.assertAlmostEqual(midpoint(*board([4.0]*10, [-2.0]*10)), 1.0, places=6)
        self.assertAlmostEqual(midpoint(*board([.5]*10, [.5]*10)), .5, places=6)

    def test_the_two_means_carry_equal_weight(self):
        # 2 occupied against 18 empty must charge what 10 against 10 does: the balance is a weight, not a count.
        skewed = midpoint(*board([4.0]*2, [-2.0]*18))
        even = midpoint(*board([4.0]*10, [-2.0]*10))
        self.assertAlmostEqual(skewed, even, places=6)
        self.assertAlmostEqual(skewed, 1.0, places=6)
        self.assertAlmostEqual(midpoint(*board([4.0]*4, [-2.0]*16)), 1.0, places=6)

    def test_the_location_term_and_the_band_disagree_in_both_directions(self):
        # The band is free while the location is not: both classes sit outside the band, but their midpoint is +1.
        located = board([4.0]*10, [-2.0]*10)
        self.assertEqual(float(dense_hinge_term(*located)), 0.0)
        self.assertAlmostEqual(midpoint(*located), 1.0, places=6)
        # The location is free while the band is not: symmetric, but both classes sit inside the band.
        inside = board([.5]*10, [-.5]*10)
        self.assertEqual(midpoint(*inside), 0.0)
        self.assertAlmostEqual(float(dense_hinge_term(*inside)), .5, places=6)

    def test_the_term_matches_an_independent_reference(self):
        # Standing instrument (V-133): evaluate the declared formula in plain Python, with explicit per-class
        # divisors and the declared equal weight, instead of rearranging the implementation.
        generator = torch.Generator().manual_seed(20261114)
        for _ in range(6):
            occupied_values = torch.randn(9, generator=generator).tolist()
            empty_values = torch.randn(11, generator=generator).tolist()
            value = midpoint(*board(occupied_values, empty_values))
            occupied_mean = sum(occupied_values)/len(occupied_values)
            empty_mean = sum(empty_values)/len(empty_values)
            self.assertAlmostEqual(value, abs(.5*occupied_mean+.5*empty_mean-MIDPOINT_TARGET), places=5)

    def test_an_unknown_cell_is_never_counted_as_empty(self):
        named = ([4.0]*10, [-2.0]*10)
        self.assertAlmostEqual(midpoint(*board(*named, unknown_at=(9, 9))), midpoint(*board(*named)), places=6)
        # A hidden cell above zero must not move the midpoint either.
        self.assertEqual(midpoint(*board([2.0]*10, [-2.0]*10, unknown_at=(9, 9))), 0.0)

    def test_a_board_without_one_of_the_classes_is_not_counted(self):
        output, target = board([4.0]*10, [-2.0]*10)
        for column in range(COLUMNS):
            output["occupancy"][1, 0, column] = 5.0
            target["occupied"][1, 0, column] = 1
            target["known"][1, 0, column] = True
        self.assertAlmostEqual(midpoint(output, target), 1.0, places=6)

    def test_a_batch_with_nothing_to_judge_is_zero_and_still_differentiable(self):
        logits = torch.zeros(BATCH, ROWS+6, COLUMNS, requires_grad=True)
        value = midpoint_term({"occupancy": logits},
                              {"occupied": torch.zeros(BATCH, ROWS, COLUMNS, dtype=torch.long),
                               "known": torch.zeros(BATCH, ROWS, COLUMNS, dtype=torch.bool)})
        self.assertEqual(float(value.detach()), 0.0)
        self.assertTrue(value.requires_grad)

    def test_the_gradient_moves_both_classes_together(self):
        # A location term, not a separation term: when the midpoint sits above the target, both classes are pushed
        # the same way (down); below the target, both are pushed the other way. The signs must never oppose.
        for value, expected in ((2.0, 1), (-2.0, -1)):
            output, target = board([value]*10, [value]*10)
            output["occupancy"].requires_grad_(True)
            midpoint_term(output, target).backward()
            gradient = output["occupancy"].grad
            self.assertEqual(int((gradient.abs() > 0).sum()), 20)
            self.assertTrue(bool((torch.sign(gradient[0, 0, :10]) == expected).all()))
            self.assertTrue(bool((torch.sign(gradient[0, 1, :10]) == expected).all()))

    def test_the_term_is_pooled_at_its_declared_weight(self):
        from tetris_vision.losses import joint_loss, stage_loss
        source = Path(inspect.getsourcefile(joint_loss)).read_text(encoding="utf-8")
        self.assertIn('"midpoint": ANCHOR_WEIGHT*occupancy_midpoint', source)
        for function in (joint_loss, stage_loss):
            self.assertIn("occupancy_midpoint", inspect.signature(function).parameters)

    def test_zero_leaves_the_published_objective_untouched(self):
        from tetris_vision.losses import joint_loss, stage_loss
        self.assertEqual(inspect.signature(stage_loss).parameters["occupancy_midpoint"].default, 0)
        self.assertEqual(inspect.signature(joint_loss).parameters["occupancy_midpoint"].default, 0)
        self.assertEqual(ANCHOR_WEIGHT*0, 0.0)
        source = Path(inspect.getsourcefile(stage_loss)).read_text(encoding="utf-8")
        self.assertGreater(source.index('result["midpoint"]'), source.index("if occupancy_midpoint:"))

    def test_the_admission_join_refuses_a_mismatch(self):
        summary = {"arguments": {"occupancy_midpoint": 1}, "recovery_contract": {"occupancy_midpoint": 1}}
        self.assertIsNone(require_occupancy_midpoint(1, summary))
        for value, wrong in ((0, summary), (2, summary),
                             (1, {"arguments": {"occupancy_midpoint": 0},
                                  "recovery_contract": {"occupancy_midpoint": 1}}),
                             (1, {"arguments": {}, "recovery_contract": {}})):
            with self.assertRaises(ValueError):
                require_occupancy_midpoint(value, wrong)


class PlumbingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from train_m2_stream import arguments, validate_admission, main
        cls.source = Path(inspect.getsourcefile(validate_admission)).read_text(encoding="utf-8")
        cls.driver = inspect.getsource(main)
        cls.refusals = inspect.getsource(arguments)
        cls.signature = inspect.signature(validate_admission)
        cls.arguments = staticmethod(arguments)
        cls.base = ["--output", "runs/m13-midpoint-refusal-probe", "--model", "accuracy", "--steps", "2"]

    def test_the_driver_passes_the_lever_to_the_objective(self):
        self.assertEqual(self.driver.count("occupancy_midpoint=args.occupancy_midpoint"), 4)

    def test_the_driver_records_the_lever_in_the_recovery_contract(self):
        self.assertIn('recovery_contract["occupancy_midpoint"] = args.occupancy_midpoint', self.driver)

    def test_the_admission_checks_the_lever_it_receives(self):
        self.assertIn("occupancy_midpoint", self.signature.parameters)
        self.assertEqual(self.signature.parameters["occupancy_midpoint"].default, 0)
        self.assertIn("require_occupancy_midpoint(occupancy_midpoint, summary)", self.source)

    def test_the_midpoint_anchor_refuses_to_stack_with_all_four_terms(self):
        self.assertIn("args.occupancy_midpoint and (args.occupancy_dense_hinge or args.occupancy_quantile_anchor or",
                      self.refusals)
        self.assertIn("args.occupancy_anchor or args.occupancy_margin_hinge", self.refusals)
        self.assertIn("the quantile anchor, the two-sided anchor and the one-sided hinge", self.refusals)

    def test_the_trainer_refuses_every_arm_that_moves_the_read_or_zero(self):
        for lever in (["--occupancy-dense-hinge", "1"], ["--occupancy-quantile-anchor", "1"],
                      ["--occupancy-anchor", "1"], ["--occupancy-margin-hinge", "1"],
                      ["--occupancy-normalise", "1"], ["--occupancy-deform", "2", "--occupancy-detail", "1"],
                      ["--occupancy-offset-rows", "1", "--occupancy-detail", "1"],
                      ["--occupancy-detail-stencil", "1", "--occupancy-detail", "1"],
                      ["--occupancy-resolution", "1"], ["--occupancy-context", "1"],
                      ["--occupancy-column", "1"], ["--occupancy-lattice-free", "1"]):
            stream = io.StringIO()
            with self.subTest(lever=lever), contextlib.redirect_stderr(stream):
                with self.assertRaises(SystemExit):
                    self.arguments(self.base+["--occupancy-midpoint", "1"]+lever)
            self.assertIn("midpoint anchor supervises", stream.getvalue(), lever)

    def test_the_midpoint_anchor_alone_is_admitted(self):
        parsed = self.arguments(self.base+["--profile-square", "--occupancy-midpoint", "1"])
        self.assertEqual(parsed.occupancy_midpoint, 1)
        self.assertEqual(self.arguments(self.base+["--profile-square"]).occupancy_midpoint, 0)

    def test_the_run_family_is_owned(self):
        from record_stream_run import OWNED_RUN_PATTERN
        for name in ("m13-midpoint-control-r1", "m13-midpoint-level-r1",
                     "m13-midpoint-level-r1-sweep-r1-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        for name in ("m13-midpoint-", "m13-midpoint", "m13-midpoints-arm-r1"):
            self.assertFalse(OWNED_RUN_PATTERN.fullmatch(name), name)
        for name in ("m9-margin-control-r1", "m10-anchor-control-r1", "m11-quantile-anchor-control-r1",
                     "m12-dense-hinge-control-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)


class MidpointRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scope = ROOT/"docs/evidence/m13-midpoint-r1/scope.json"
        if not scope.exists():
            raise unittest.SkipTest("the midpoint record has not been curated yet")
        cls.scope = json.loads(scope.read_text(encoding="utf-8"))
        cls.prior = json.loads((ROOT/"docs/evidence/m10-supervised-anchor-r1/scope.json")
                               .read_text(encoding="utf-8"))
        cls.dense = json.loads((ROOT/"docs/evidence/m12-dense-hinge-r1/scope.json")
                               .read_text(encoding="utf-8"))
        cls.text = " ".join((ROOT/"docs/M13_MIDPOINT_ANCHOR.md").read_text(encoding="utf-8").split())

    def test_the_arms_and_their_supervision(self):
        self.assertEqual(sorted(self.scope["arms"]), ["control", "level"])
        self.assertEqual({name: arm["occupancy_midpoint"] for name, arm in self.scope["arms"].items()},
                         {"control": 0, "level": 1})
        for arm in self.scope["arms"].values():
            for key in ("occupancy_dense_hinge", "occupancy_quantile_anchor", "occupancy_anchor",
                        "occupancy_margin_hinge"):
                self.assertEqual(arm[key], 0)

    def test_control_reproduces_the_recorded_control(self):
        self.assertEqual(self.scope["arms"]["control"]["auc"], self.prior["arms"]["control"]["auc"])
        self.assertEqual(self.scope["arms"]["control"]["f1_at_zero"],
                         self.prior["arms"]["control"]["f1_at_zero"])

    def test_the_declared_constants_are_the_ones_recorded(self):
        mechanism = self.scope["mechanism"]
        self.assertEqual((mechanism["weight"], mechanism["target"]), (ANCHOR_WEIGHT, MIDPOINT_TARGET))
        self.assertIsNone(mechanism["margin"])
        self.assertEqual((mechanism["occupied_balance"], mechanism["empty_balance"]),
                         (MIDPOINT_OCCUPIED_BALANCE, MIDPOINT_EMPTY_BALANCE))
        self.assertTrue(mechanism["declared_before_receipts"])

    def test_the_readings_follow_the_printed_numbers_and_the_override_rule(self):
        arm, control = self.scope["arms"]["level"], self.scope["arms"]["control"]
        delta = round(arm["f1_at_zero"]-control["f1_at_zero"], 6)
        primary = ("midpoint-clears-the-boundary" if delta >= .05 else
                   "midpoint-moves-without-clearing" if delta > 0 else "midpoint-fails")
        auc_delta = round(arm["auc"]-control["auc"], 6)
        degraded = auc_delta < -.02
        self.assertEqual(self.scope["secondary_reading"],
                         "ranking-degraded" if degraded else "ranking-preserved")
        expected = "midpoint-fails" if degraded else primary
        self.assertEqual(self.scope["reading"], expected)
        if degraded:
            self.assertEqual(self.scope["reading_override"]["primary_reading"], primary)
            self.assertEqual(self.scope["reading_override"]["forced_reading"], "midpoint-fails")
        else:
            self.assertIsNone(self.scope["reading_override"])

    def test_the_document_quotes_the_published_scope(self):
        for name, arm in self.scope["arms"].items():
            self.assertIn(f"{round(arm['auc'], 6):,.6f}", self.text, name)
            self.assertIn(f"{round(arm['f1_at_zero'], 6):,.6f}", self.text, name)
        self.assertIn(self.scope["reading"], self.text)
        self.assertIn(self.scope["secondary_reading"], self.text)

    def test_all_reference_rows_are_quoted_and_not_reverdicted(self):
        dense = self.scope["reference"]["dense_hinge"]
        self.assertEqual(dense["auc"], self.dense["arms"]["banded"]["auc"])
        self.assertEqual(self.scope["reference"]["extremal_anchor"]["auc"], self.prior["arms"]["anchor"]["auc"])
        for arm in self.scope["reference"].values():
            if isinstance(arm, dict) and "auc" in arm:
                self.assertIn(f"{round(arm['auc'], 6):,.6f}", self.text)

    def test_the_fit_gate_and_the_level_drift_are_reported(self):
        gate = self.scope["fit_gate"]
        self.assertEqual(gate["boards"], 28)
        self.assertEqual(gate["exact_nonempty_boards"],
                         self.scope["arms"]["level"]["exact_boards_at_zero"]["exact"])
        self.assertEqual(gate["cleared"], gate["exact_nonempty_boards"] == 28)
        for name, arm in self.scope["arms"].items():
            self.assertAlmostEqual(arm["positive_share_at_zero"],
                                   (arm["operating_counts"]["true_positive"]
                                    + arm["operating_counts"]["false_positive"])/5600, places=6)

    def test_the_paid_charge_is_reported_from_the_study_log(self):
        paid = self.scope["arms"]["level"]["training_midpoint"]
        self.assertEqual(paid["steps"], 64)
        self.assertTrue(paid["in_sample"])


if __name__ == "__main__":
    unittest.main()
