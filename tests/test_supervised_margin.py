"""The supervised boundary: a within-board margin hinge on the occupancy read."""
import json
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from tetris_vision.losses import (MARGIN_HINGE_MARGIN, MARGIN_HINGE_WEIGHT,
                                  margin_hinge_term, require_occupancy_margin_hinge)

ROOT = Path(__file__).resolve().parents[1]
ROWS, COLUMNS, BATCH = 20, 10, 2


def hinge(high=None, low=None, *, known=None, occupied_rows=((5, 3),)):
    """One synthetic board carrying an occupied cell, an empty cell and an unknown region."""
    logits = torch.zeros(BATCH, ROWS+6, COLUMNS)
    if high is not None:
        logits[0, 5, 3] = high
    if low is not None:
        logits[0, 7, 4] = low
    occupied = torch.zeros(BATCH, ROWS, COLUMNS, dtype=torch.long)
    for row, column in occupied_rows:
        occupied[0, row, column] = 1
    mask = (torch.ones(BATCH, ROWS, COLUMNS, dtype=torch.bool) if known is None else known)
    return margin_hinge_term({"occupancy": logits}, {"occupied": occupied, "known": mask})


class MarginHingeTests(unittest.TestCase):
    def test_a_separated_board_is_free(self):
        self.assertEqual(float(hinge(3.0, 0.5)), 0.0)

    def test_an_overlapping_board_pays_the_shortfall(self):
        self.assertAlmostEqual(float(hinge(0.2, 0.5)), MARGIN_HINGE_MARGIN+0.3, places=5)

    def test_the_margin_is_the_declared_constant(self):
        # Exactly at the margin costs nothing; one unit less costs one unit.
        self.assertEqual(float(hinge(MARGIN_HINGE_MARGIN, 0.0)), 0.0)
        self.assertAlmostEqual(float(hinge(MARGIN_HINGE_MARGIN-1.0, 0.0)), 1.0, places=5)

    def test_an_unknown_cell_is_never_treated_as_empty(self):
        # The highest *empty* logit is 0.5; a larger logit at an unknown cell must not be read as empty.
        known = torch.ones(BATCH, ROWS, COLUMNS, dtype=torch.bool)
        known[0, 9, 9] = False
        logits = torch.zeros(BATCH, ROWS+6, COLUMNS)
        logits[0, 5, 3], logits[0, 7, 4], logits[0, 9, 9] = 3.0, 0.5, 9.0
        occupied = torch.zeros(BATCH, ROWS, COLUMNS, dtype=torch.long)
        occupied[0, 5, 3] = 1
        value = margin_hinge_term({"occupancy": logits}, {"occupied": occupied, "known": known})
        self.assertEqual(float(value), 0.0)

    def test_a_board_without_one_of_the_classes_is_not_counted(self):
        # Board 0 is cleanly separated; board 1 has an empty cell but no occupied cell, so the mean is
        # board 0's value alone rather than being diluted by a board that cannot be judged.
        logits = torch.zeros(BATCH, ROWS+6, COLUMNS)
        logits[0, 5, 3], logits[0, 7, 4], logits[1, 7, 4] = 0.5, 1.5, 0.0
        occupied = torch.zeros(BATCH, ROWS, COLUMNS, dtype=torch.long)
        occupied[0, 5, 3] = 1
        value = margin_hinge_term({"occupancy": logits},
                                  {"occupied": occupied,
                                   "known": torch.ones(BATCH, ROWS, COLUMNS, dtype=torch.bool)})
        self.assertAlmostEqual(float(value), MARGIN_HINGE_MARGIN+1.0, places=5)

    def test_a_batch_with_nothing_to_judge_is_zero_and_still_differentiable(self):
        empty = torch.zeros(BATCH, ROWS, COLUMNS, dtype=torch.long)
        unknown = torch.zeros(BATCH, ROWS, COLUMNS, dtype=torch.bool)
        logits = torch.zeros(BATCH, ROWS+6, COLUMNS, requires_grad=True)
        value = margin_hinge_term({"occupancy": logits}, {"occupied": empty, "known": unknown})
        self.assertEqual(float(value.detach()), 0.0)
        self.assertTrue(value.requires_grad)

    def test_the_declared_constants_are_the_documented_ones(self):
        self.assertEqual((MARGIN_HINGE_MARGIN, MARGIN_HINGE_WEIGHT), (1.0, .5))

    def test_the_term_is_added_to_the_objective_at_its_declared_weight(self):
        from tetris_vision.losses import joint_loss
        source = Path(__import__("inspect").getsourcefile(joint_loss)).read_text(encoding="utf-8")
        self.assertIn('"margin": MARGIN_HINGE_WEIGHT*occupancy_margin_hinge', source)

    def test_the_admission_join_refuses_a_mismatch(self):
        summary = {"arguments": {"occupancy_margin_hinge": 1},
                   "recovery_contract": {"occupancy_margin_hinge": 1}}
        self.assertIsNone(require_occupancy_margin_hinge(1, summary))
        for value, wrong in ((0, summary), (2, summary),
                             (1, {"arguments": {"occupancy_margin_hinge": 0},
                                  "recovery_contract": {"occupancy_margin_hinge": 1}}),
                             (1, {"arguments": {}, "recovery_contract": {}})):
            with self.assertRaises(ValueError):
                require_occupancy_margin_hinge(value, wrong)


class PlumbingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import inspect
        from train_m2_stream import validate_admission, main
        cls.source = Path(inspect.getsourcefile(validate_admission)).read_text(encoding="utf-8")
        cls.driver = inspect.getsource(main)
        cls.signature = inspect.signature(validate_admission)

    def test_the_driver_passes_the_lever_to_the_objective(self):
        self.assertEqual(self.driver.count("occupancy_margin_hinge=args.occupancy_margin_hinge"), 4)

    def test_the_driver_records_the_lever_in_the_recovery_contract(self):
        self.assertIn('recovery_contract["occupancy_margin_hinge"] = args.occupancy_margin_hinge',
                      self.driver)

    def test_the_admission_checks_the_lever_it_receives(self):
        self.assertIn("occupancy_margin_hinge", self.signature.parameters)
        self.assertEqual(self.signature.parameters["occupancy_margin_hinge"].default, 0)
        self.assertIn("require_occupancy_margin_hinge(occupancy_margin_hinge, summary)", self.source)

    def test_the_run_family_is_owned(self):
        from record_stream_run import OWNED_RUN_PATTERN
        for name in ("m9-margin-control-r1", "m9-margin-hinge-r1-sweep-r1",
                     "m9-margin-hinge-r1-sweep-r1-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        for name in ("m9-margin-", "m9-margin", "m9-hinge-arm-r1"):
            self.assertFalse(OWNED_RUN_PATTERN.fullmatch(name), name)


class SupervisedMarginRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scope = ROOT/"docs/evidence/m9-supervised-margin-r1/scope.json"
        if not scope.exists():
            raise unittest.SkipTest("the supervised-margin record has not been curated yet")
        cls.scope = json.loads(scope.read_text(encoding="utf-8"))
        cls.prior = json.loads((ROOT/"docs/evidence/m8-normalised-boundary-r1/scope.json")
                               .read_text(encoding="utf-8"))
        cls.text = " ".join((ROOT/"docs/M9_SUPERVISED_MARGIN.md").read_text(encoding="utf-8").split())

    def test_the_arms_and_their_supervision(self):
        self.assertEqual(sorted(self.scope["arms"]), ["control", "hinge"])
        self.assertEqual({name: arm["occupancy_margin_hinge"] for name, arm in self.scope["arms"].items()},
                         {"control": 0, "hinge": 1})

    def test_control_reproduces_the_recorded_control(self):
        self.assertEqual(self.scope["arms"]["control"]["auc"], self.prior["arms"]["control"]["auc"])
        self.assertEqual(self.scope["arms"]["control"]["f1_at_zero"],
                         self.prior["arms"]["control"]["f1_at_zero"])

    def test_the_readings_follow_the_printed_numbers(self):
        arm, control = self.scope["arms"]["hinge"], self.scope["arms"]["control"]
        delta = round(arm["f1_at_zero"]-control["f1_at_zero"], 6)
        expected = ("margin-clears-the-boundary" if delta >= .05 else
                    "margin-moves-without-clearing" if delta > 0 else "margin-fails")
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

    def test_the_fit_gate_is_reported_on_exact_boards(self):
        self.assertEqual(self.scope["fit_gate"]["boards"], 28)
        self.assertEqual(self.scope["fit_gate"]["cleared"],
                         self.scope["fit_gate"]["exact_nonempty_boards"] == 28)


if __name__ == "__main__":
    unittest.main()
