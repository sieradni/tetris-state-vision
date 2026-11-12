"""The declared sampling offset: a constant geometry in rows, identity at zero."""
import inspect
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

import torch
import tetris_vision.losses as losses_module
from tetris_vision.losses import require_occupancy_offset_rows
from tetris_vision.model import ModelConfig, TetrisStateNet

import measure_row_error
from record_stream_run import OWNED_RUN_PATTERN


def pilot(**overrides):
    return ModelConfig(**{**ModelConfig.pilot().__dict__, **overrides})


def inputs(seed=20261026, size=32):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, size, size, generator=generator), torch.ones(1, 1, size, size, dtype=torch.bool)


def forward(config, seed=20261026):
    torch.manual_seed(seed)
    model = TetrisStateNet(config).eval()
    pixels, valid = inputs()
    with torch.no_grad():
        return model, model(pixels, valid)


class SamplingOffsetTests(unittest.TestCase):
    def test_the_default_is_the_published_sampling_point(self):
        self.assertEqual(ModelConfig().occupancy_offset_rows, 0)
        self.assertEqual(ModelConfig.pilot().occupancy_offset_rows, 0)

    def test_an_undeclared_value_is_refused(self):
        for value in (3, -3, True, 1.0):
            with self.assertRaises(ValueError):
                ModelConfig(occupancy_detail=1, occupancy_offset_rows=value)

    def test_the_offset_is_a_form_of_the_detail_tap_and_needs_it(self):
        with self.assertRaises(ValueError):
            ModelConfig(occupancy_detail=0, occupancy_offset_rows=1)
        self.assertEqual(ModelConfig(occupancy_detail=1, occupancy_offset_rows=-2).occupancy_offset_rows, -2)

    def test_the_offset_adds_no_parameter_and_no_state_key(self):
        plain, _ = forward(pilot(occupancy_detail=1))
        levered, _ = forward(pilot(occupancy_detail=1, occupancy_offset_rows=1))
        self.assertEqual(sorted(plain.state_dict()), sorted(levered.state_dict()))

    def test_zero_is_exactly_the_published_function(self):
        _, control = forward(pilot(occupancy_detail=1))
        _, zero = forward(pilot(occupancy_detail=1, occupancy_offset_rows=0))
        self.assertTrue(torch.equal(control["final"]["occupancy"], zero["final"]["occupancy"]))
        self.assertTrue(torch.equal(control["final"]["occupancy_applied_offsets"],
                                    zero["final"]["occupancy_applied_offsets"]))

    def test_the_declared_row_is_the_offset_every_cell_carries(self):
        for rows in (1, -1, 2):
            model, output = forward(pilot(occupancy_detail=1, occupancy_offset_rows=rows))
            applied = output["final"]["occupancy_applied_offsets"]
            self.assertEqual(tuple(applied.shape), (1, 260, 2))
            self.assertEqual(float(applied[..., 0].abs().max()), 0.0)
            self.assertEqual(float(applied[..., 1].min()), float(rows))
            self.assertEqual(float(applied[..., 1].max()), float(rows))
            self.assertNotIn("tap_offset", model.state_dict())

    def test_the_offset_changes_what_is_read_once_the_tap_is_trained(self):
        _, control = forward(pilot(occupancy_detail=1))
        _, up = forward(pilot(occupancy_detail=1, occupancy_offset_rows=1))
        # The published tap is zero-initialized, so the arm is the published function at step zero
        # whatever the geometry; what the offset changes is where a trained tap reads from.
        self.assertTrue(torch.equal(control["final"]["occupancy"], up["final"]["occupancy"]))
        self.assertFalse(torch.equal(control["final"]["occupancy_applied_offsets"],
                                     up["final"]["occupancy_applied_offsets"]))
        generator = torch.Generator().manual_seed(11)
        decisions = {}
        for rows in (0, 1):
            torch.manual_seed(20261026)
            model = TetrisStateNet(pilot(occupancy_detail=1, occupancy_offset_rows=rows)).eval()
            with torch.no_grad():
                model.occupancy_detail_tap.weight.normal_(generator=generator)
            pixels, valid = inputs()
            with torch.no_grad():
                decisions[rows] = model(pixels, valid)["final"]["occupancy"]
        self.assertFalse(torch.equal(decisions[0], decisions[1]))

    def test_paired_rasters_give_the_same_decision(self):
        torch.manual_seed(20261026)
        model = TetrisStateNet(pilot(occupancy_detail=1, occupancy_offset_rows=1)).eval()
        pixels, valid = inputs()
        with torch.no_grad():
            first = model(pixels, valid)["final"]["occupancy"]
            second = model(pixels, valid)["final"]["occupancy"]
        self.assertTrue(torch.equal(first, second))

    def test_the_declared_constant_is_the_validation_allowlist(self):
        from train_m2_stream import OCCUPANCY_OFFSET_ROWS
        self.assertEqual(tuple(OCCUPANCY_OFFSET_ROWS), tuple(losses_module.OCCUPANCY_OFFSET_ROWS))

    def test_the_admission_join_refuses_a_mismatched_declaration(self):
        recorded = {"arguments": {"occupancy_offset_rows": 1},
                    "recovery_contract": {"occupancy_offset_rows": 1}}
        require_occupancy_offset_rows(1, recorded)
        require_occupancy_offset_rows(0, {"arguments": {}, "recovery_contract": {}})
        for value in (1, -1):
            with self.assertRaises(ValueError):
                require_occupancy_offset_rows(value, {"arguments": {"occupancy_offset_rows": -value},
                                                     "recovery_contract": {"occupancy_offset_rows": -value}})
        with self.assertRaises(ValueError):
            require_occupancy_offset_rows(3, recorded)


class SamplingOffsetDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from train_m2_stream import validate_admission, main
        cls.source = inspect.getsource(validate_admission)
        cls.driver = inspect.getsource(main)

    def test_the_recovery_contract_and_config_carry_the_lever(self):
        self.assertIn('recovery_contract["occupancy_offset_rows"] = args.occupancy_offset_rows',
                      self.driver)
        start = self.driver.index("replace(config,")
        block = self.driver[start:self.driver.index("config_json", start)]
        self.assertIn("occupancy_offset_rows=args.occupancy_offset_rows", block)

    def test_the_admission_checks_the_lever_it_receives(self):
        self.assertIn("require_occupancy_offset_rows(occupancy_offset_rows, summary)", self.source)
        self.assertIn("occupancy_offset_rows=args.occupancy_offset_rows", self.driver)
        self.assertIn("occupancy_offset_rows=0", self.source)

    def test_the_run_family_is_owned(self):
        self.assertTrue(OWNED_RUN_PATTERN.match("m6-rowoffset-up-r1"))
        self.assertFalse(OWNED_RUN_PATTERN.match("m6-row-offset-up-r1"))


class RowErrorDiagnosticTests(unittest.TestCase):
    """The diagnostic's own arithmetic, before any of its numbers are believed."""

    def setUp(self):
        self.rows, self.columns = 6, 2

    def grid(self, labels_by_column, decided_by_column):
        labels = [0]*(self.rows*self.columns)
        logits = [0.0]*(self.rows*self.columns)
        for column, rows in enumerate(labels_by_column):
            for row in rows:
                labels[row*self.columns+column] = 1
        for column, rows in enumerate(decided_by_column):
            for row in rows:
                logits[row*self.columns+column] = 2.0
        return logits, labels

    def test_the_top_boundary_is_the_highest_decided_and_occupied_row(self):
        logits, labels = self.grid([[0, 1, 2], []], [[0, 1, 2, 3], []])
        decided, occupied = measure_row_error.boundaries(logits, labels, self.rows, self.columns, 0.0)
        self.assertEqual(decided, [3, None])
        self.assertEqual(occupied, [2, None])

    def test_a_column_with_no_label_and_no_decision_is_neither(self):
        logits, labels = self.grid([[], []], [[], []])
        decided, occupied = measure_row_error.boundaries(logits, labels, self.rows, self.columns, 0.0)
        self.assertEqual((decided, occupied), ([None, None], [None, None]))

    def test_the_threshold_moves_the_boundary_downwards(self):
        logits, labels = self.grid([[0]], [[0, 4]])
        decided, _ = measure_row_error.boundaries(logits, labels, self.rows, self.columns, 0.0)
        self.assertEqual(decided[0], 4)
        logits[4*self.columns+0] = 0.5
        decided, _ = measure_row_error.boundaries(logits, labels, self.rows, self.columns, 1.0)
        self.assertEqual(decided[0], 0)

    def test_the_summary_reports_the_mean_and_the_spread(self):
        block = measure_row_error.summarise([-1, 0, 1, 2])
        self.assertEqual(block["mean"], 0.5)
        self.assertEqual(block["min"], -1)
        self.assertEqual(block["max"], 2)
        self.assertEqual(block["cells"], 4)


class RowOffsetRecordTests(unittest.TestCase):
    """The published record, once curation has written it."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        scope = cls.root/"docs/evidence/m6-row-offset-r1/scope.json"
        if not scope.exists():
            raise unittest.SkipTest("the row-offset record has not been curated yet")
        cls.scope = json.loads(scope.read_text(encoding="utf-8"))
        cls.prior = json.loads((cls.root/"docs/evidence/m5-column-readout-r1/scope.json")
                               .read_text(encoding="utf-8"))
        cls.text = " ".join((cls.root/"docs/M6_ROW_OFFSET.md").read_text(encoding="utf-8").split())

    def test_the_arms_are_the_declared_geometry(self):
        self.assertEqual(sorted(self.scope["arms"]), ["control", "down", "up", "wide"])
        self.assertEqual({name: arm["occupancy_offset_rows"] for name, arm in self.scope["arms"].items()},
                         {"control": 0, "wide": 0, "up": 1, "down": -1})

    def test_control_and_wide_reproduce_the_previous_record(self):
        for name in ("control", "wide"):
            self.assertEqual(self.scope["arms"][name]["auc"], self.prior["arms"][name]["auc"])
            self.assertEqual(self.scope["arms"][name]["best"]["f1"],
                             self.prior["arms"][name]["best"]["f1"])

    def test_the_reading_follows_the_printed_deltas(self):
        delta = self.scope["contrasts"]["up_control"]
        moved = self.scope["row_error"]["arms"]["control"]["mean"] - \
            self.scope["row_error"]["arms"]["up"]["mean"]
        if delta["best_f1_delta"] >= .05 and delta["auc_interval"]["low"] > 0 and moved >= .5:
            expected = "offset-is-the-lever"
        elif moved >= .5:
            expected = "offset-moves-without-ranking"
        else:
            expected = "offset-class-fails"
        self.assertEqual(self.scope["offset"]["reading"], expected)

    def test_the_document_quotes_the_published_scope(self):
        for name, arm in self.scope["arms"].items():
            self.assertIn(f"{round(arm['auc'], 6):,.6f}", self.text, name)
            self.assertIn(f"{round(arm['best']['f1'], 6):,.6f}", self.text, name)
        for key, delta in self.scope["contrasts"].items():
            self.assertIn(f"{round(delta['best_f1_delta'], 6):,.6f}", self.text, key)
        self.assertIn(self.scope["offset"]["reading"], self.text)


if __name__ == "__main__":
    unittest.main()
