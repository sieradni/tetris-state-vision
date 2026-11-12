"""The supervised operating point: a two-sided margin anchored to the boundary a consumer applies."""
import json
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from tetris_vision.losses import (ANCHOR_MARGIN, ANCHOR_WEIGHT, anchor_term,
                                  require_occupancy_anchor)

ROOT = Path(__file__).resolve().parents[1]
ROWS, COLUMNS, BATCH = 20, 10, 2


def anchor(high=None, low=None, *, known=None, occupied_rows=((5, 3),)):
    # Every other visible cell is known and empty, so the board's floor is what "highest empty" finds when the
    # named cell is not the highest; -4 keeps a clean case clean instead of tripping the anchor on the floor.
    logits = torch.full((BATCH, ROWS+6, COLUMNS), -4.0)
    if high is not None:
        logits[0, 5, 3] = high
    if low is not None:
        logits[0, 7, 4] = low
    occupied = torch.zeros(BATCH, ROWS, COLUMNS, dtype=torch.long)
    for row, column in occupied_rows:
        occupied[0, row, column] = 1
    mask = (torch.ones(BATCH, ROWS, COLUMNS, dtype=torch.bool) if known is None else known)
    return anchor_term({"occupancy": logits}, {"occupied": occupied, "known": mask})


class AnchorTermTests(unittest.TestCase):
    def test_a_correctly_anchored_board_is_free(self):
        self.assertEqual(float(anchor(1.0, -1.0)), 0.0)
        self.assertEqual(float(anchor(5.0, -9.0)), 0.0)

    def test_an_occupied_mode_too_low_pays_the_shortfall(self):
        self.assertAlmostEqual(float(anchor(0.4, -1.0)), ANCHOR_MARGIN-0.4, places=5)

    def test_an_empty_mode_too_high_pays_the_shortfall(self):
        self.assertAlmostEqual(float(anchor(3.0, 0.5)), ANCHOR_MARGIN+0.5, places=5)

    def test_a_separated_but_shifted_board_is_no_longer_free(self):
        # The M9 hinge was satisfied by this board (a 1.5-logit gap above zero); the anchor is not, which is
        # the whole point of the two-sided form.
        hinge_satisfied = (2.5, 1.0)
        self.assertAlmostEqual(float(anchor(*hinge_satisfied)), ANCHOR_MARGIN+1.0, places=5)

    def test_the_two_halves_are_independent(self):
        both_bad = float(anchor(0.0, 0.0))
        self.assertAlmostEqual(both_bad, 2*ANCHOR_MARGIN, places=5)
        self.assertAlmostEqual(both_bad, float(anchor(0.0, -1.0))+float(anchor(1.0, 0.0)), places=5)

    def test_an_unknown_cell_is_never_treated_as_empty(self):
        known = torch.ones(BATCH, ROWS, COLUMNS, dtype=torch.bool)
        known[0, 9, 9] = False
        logits = torch.full((BATCH, ROWS+6, COLUMNS), -4.0)
        logits[0, 5, 3], logits[0, 7, 4], logits[0, 9, 9] = 2.0, -2.0, 9.0
        occupied = torch.zeros(BATCH, ROWS, COLUMNS, dtype=torch.long)
        occupied[0, 5, 3] = 1
        value = anchor_term({"occupancy": logits}, {"occupied": occupied, "known": known})
        self.assertEqual(float(value), 0.0)

    def test_a_board_without_one_of_the_classes_is_not_counted(self):
        logits = torch.full((BATCH, ROWS+6, COLUMNS), -4.0)
        logits[0, 5, 3], logits[0, 7, 4] = 2.0, -2.0
        logits[1, 5, 3] = 0.0
        occupied = torch.zeros(BATCH, ROWS, COLUMNS, dtype=torch.long)
        occupied[0, 5, 3] = 1
        value = anchor_term({"occupancy": logits},
                            {"occupied": occupied,
                             "known": torch.ones(BATCH, ROWS, COLUMNS, dtype=torch.bool)})
        self.assertEqual(float(value), 0.0)

    def test_a_batch_with_nothing_to_judge_is_zero_and_still_differentiable(self):
        empty = torch.zeros(BATCH, ROWS, COLUMNS, dtype=torch.long)
        unknown = torch.zeros(BATCH, ROWS, COLUMNS, dtype=torch.bool)
        logits = torch.zeros(BATCH, ROWS+6, COLUMNS, requires_grad=True)
        value = anchor_term({"occupancy": logits}, {"occupied": empty, "known": unknown})
        self.assertEqual(float(value.detach()), 0.0)
        self.assertTrue(value.requires_grad)

    def test_the_declared_constants_are_the_documented_ones(self):
        self.assertEqual((ANCHOR_MARGIN, ANCHOR_WEIGHT), (1.0, .5))

    def test_the_term_is_pooled_at_its_declared_weight(self):
        from tetris_vision.losses import joint_loss
        source = Path(__import__("inspect").getsourcefile(joint_loss)).read_text(encoding="utf-8")
        self.assertIn('"anchor": ANCHOR_WEIGHT*occupancy_anchor', source)
        self.assertIn("occupancy_anchor", __import__("inspect").signature(joint_loss).parameters)
        self.assertIn("occupancy_anchor", __import__("inspect").signature(
            __import__("tetris_vision.losses", fromlist=["stage_loss"]).stage_loss).parameters)

    def test_the_admission_join_refuses_a_mismatch(self):
        summary = {"arguments": {"occupancy_anchor": 1}, "recovery_contract": {"occupancy_anchor": 1}}
        self.assertIsNone(require_occupancy_anchor(1, summary))
        for value, wrong in ((0, summary), (2, summary),
                             (1, {"arguments": {"occupancy_anchor": 0},
                                  "recovery_contract": {"occupancy_anchor": 1}}),
                             (1, {"arguments": {}, "recovery_contract": {}})):
            with self.assertRaises(ValueError):
                require_occupancy_anchor(value, wrong)


class PlumbingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import inspect
        from train_m2_stream import arguments, validate_admission, main
        cls.source = Path(inspect.getsourcefile(validate_admission)).read_text(encoding="utf-8")
        cls.driver = inspect.getsource(main)
        cls.refusals = inspect.getsource(arguments)
        cls.signature = inspect.signature(validate_admission)

    def test_the_driver_passes_the_lever_to_the_objective(self):
        self.assertEqual(self.driver.count("occupancy_anchor=args.occupancy_anchor"), 4)

    def test_the_driver_records_the_lever_in_the_recovery_contract(self):
        self.assertIn('recovery_contract["occupancy_anchor"] = args.occupancy_anchor', self.driver)

    def test_the_admission_checks_the_lever_it_receives(self):
        self.assertIn("occupancy_anchor", self.signature.parameters)
        self.assertEqual(self.signature.parameters["occupancy_anchor"].default, 0)
        self.assertIn("require_occupancy_anchor(occupancy_anchor, summary)", self.source)

    def test_the_anchor_refuses_to_stack_with_the_hinge(self):
        # The declared refusal must name both levers: stacking two margins on one boundary is not this arm.
        self.assertIn("args.occupancy_anchor and (args.occupancy_margin_hinge", self.refusals)
        self.assertIn("one-sided hinge, so neither it nor any lever", self.refusals)
        self.assertIn("args.occupancy_normalise", self.refusals)
        self.assertIn("args.occupancy_offset_rows", self.refusals)

    def test_the_run_family_is_owned(self):
        from record_stream_run import OWNED_RUN_PATTERN
        for name in ("m10-anchor-control-r1", "m10-anchor-two-sided-r1-sweep-r1",
                     "m10-anchor-two-sided-r1-sweep-r1-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        for name in ("m10-anchor-", "m10-anchor", "m10-anchored-arm-r1"):
            self.assertFalse(OWNED_RUN_PATTERN.fullmatch(name), name)


class SupervisedAnchorRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scope = ROOT/"docs/evidence/m10-supervised-anchor-r1/scope.json"
        if not scope.exists():
            raise unittest.SkipTest("the supervised-anchor record has not been curated yet")
        cls.scope = json.loads(scope.read_text(encoding="utf-8"))
        cls.prior = json.loads((ROOT/"docs/evidence/m9-supervised-margin-r1/scope.json")
                               .read_text(encoding="utf-8"))
        cls.text = " ".join((ROOT/"docs/M10_SUPERVISED_ANCHOR.md").read_text(encoding="utf-8").split())

    def test_the_arms_and_their_supervision(self):
        self.assertEqual(sorted(self.scope["arms"]), ["anchor", "control"])
        self.assertEqual({name: arm["occupancy_anchor"] for name, arm in self.scope["arms"].items()},
                         {"control": 0, "anchor": 1})

    def test_control_reproduces_the_recorded_control(self):
        self.assertEqual(self.scope["arms"]["control"]["auc"], self.prior["arms"]["control"]["auc"])
        self.assertEqual(self.scope["arms"]["control"]["f1_at_zero"],
                         self.prior["arms"]["control"]["f1_at_zero"])

    def test_the_readings_follow_the_printed_numbers(self):
        arm, control = self.scope["arms"]["anchor"], self.scope["arms"]["control"]
        delta = round(arm["f1_at_zero"]-control["f1_at_zero"], 6)
        expected = ("anchor-clears-the-boundary" if delta >= .05 else
                    "anchor-moves-without-clearing" if delta > 0 else "anchor-fails")
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

    def test_the_hinge_reference_is_quoted_and_not_reverdict(self):
        hinge = self.scope["reference"]["hinge"]
        self.assertEqual(hinge["auc"], self.prior["arms"]["hinge"]["auc"])
        self.assertIn("reference", self.scope["reference"]["role"])

    def test_the_fit_gate_is_reported_on_exact_boards(self):
        self.assertEqual(self.scope["fit_gate"]["boards"], 28)
        self.assertEqual(self.scope["fit_gate"]["cleared"],
                         self.scope["fit_gate"]["exact_nonempty_boards"] == 28)


if __name__ == "__main__":
    unittest.main()
