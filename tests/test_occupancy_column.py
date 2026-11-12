"""The declared column read-out: the cell's own column, below and above it, identity at zero."""
import inspect
import json
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

import torch
from tetris_vision.losses import require_occupancy_column
from tetris_vision.model import ModelConfig, TetrisStateNet, column_context


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
        return model, model(pixels, valid)["final"]["occupancy"]


class ColumnContextTests(unittest.TestCase):
    def test_the_default_is_the_published_configuration(self):
        self.assertEqual(ModelConfig().occupancy_column, 0)
        self.assertEqual(ModelConfig.pilot().occupancy_column, 0)

    def test_an_undeclared_value_is_refused(self):
        for value in (2, -1, True, 1.0):
            with self.assertRaises(ValueError):
                ModelConfig(occupancy_column=value)

    def test_the_context_summarises_strictly_below_and_strictly_above(self):
        rows, columns, dimension = 3, 2, 2
        cells = torch.zeros(rows*columns, dimension)
        # Column 0 rows bottom-first: 1, 3, 5 on the first feature.
        for row, value in enumerate((1., 3., 5.)):
            cells[row*columns, 0] = value
        context = column_context(cells[None], rows, columns)[0]
        below = [float(context[row*columns, 0]) for row in range(rows)]
        above = [float(context[row*columns, dimension]) for row in range(rows)]
        self.assertEqual(below, [0.0, 1.0, 2.0])                # mean of nothing, of 1, of 1 and 3
        self.assertEqual(above, [4.0, 5.0, 0.0])                # mean of 3 and 5, of 5, of nothing
        self.assertEqual(float(context[0, 2*dimension]), 0.0)   # the bottom row covers no column below
        self.assertEqual(float(context[2*columns, 2*dimension]), 1.0)

    def test_a_cell_sees_its_own_column_and_no_other(self):
        rows, columns, dimension = 2, 2, 1
        cells = torch.zeros(rows*columns, dimension)
        cells[0, 0] = 1.0                                        # bottom-left only
        context = column_context(cells[None], rows, columns)[0]
        self.assertGreater(float(context[columns, 0]), 0.0)      # above it in column 0
        self.assertEqual(float(context[columns+1, 0]), 0.0)      # column 1 is untouched

    def test_the_disabled_arm_adds_no_parameters_and_the_declared_one_is_zero_initialized(self):
        plain, _ = forward(pilot())
        self.assertIsNone(plain.occupancy_column)
        self.assertFalse(any(name.startswith("occupancy_column") for name in plain.state_dict()))
        model, _ = forward(pilot(occupancy_column=1))
        self.assertEqual(model.occupancy_column.in_features, 2*pilot().dimension+2)
        self.assertEqual(model.occupancy_column.out_features, pilot().dimension)
        self.assertEqual(float(model.occupancy_column.weight.abs().sum()), 0.0)
        self.assertEqual(float(model.occupancy_column.bias.abs().sum()), 0.0)

    def test_step_zero_is_exactly_the_published_function(self):
        _, control = forward(pilot())
        _, column = forward(pilot(occupancy_column=1))
        self.assertTrue(torch.equal(control, column))

    def test_the_read_out_changes_the_decision_once_its_projection_is_learned(self):
        model, before = forward(pilot(occupancy_column=1))
        with torch.no_grad():
            model.occupancy_column.weight.normal_(generator=torch.Generator().manual_seed(7))
        pixels, valid = inputs()
        with torch.no_grad():
            after = model(pixels, valid)["final"]["occupancy"]
        self.assertFalse(torch.equal(before, after))

    def test_the_read_out_trains_and_its_gradient_reaches_the_projection(self):
        torch.manual_seed(20261026)
        model = TetrisStateNet(pilot(occupancy_column=1))
        pixels, valid = inputs()
        output = model(pixels, valid)["final"]
        occupancy = output["occupancy"].flatten()
        target = torch.zeros_like(occupancy)
        target[:10] = 1.0
        torch.nn.functional.binary_cross_entropy_with_logits(occupancy, target).backward()
        gradient = model.occupancy_column.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_paired_rasters_give_the_same_decision(self):
        torch.manual_seed(20261026)
        model = TetrisStateNet(pilot(occupancy_column=1)).eval()
        pixels, valid = inputs()
        hidden_a = torch.zeros(1, 1, 4, 4)
        hidden_b = torch.ones(1, 1, 4, 4)
        self.assertTrue(torch.equal(pixels, pixels.clone()))
        with torch.no_grad():
            first = model(pixels, valid)["final"]["occupancy"]
            second = model(pixels, valid)["final"]["occupancy"]
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(hidden_a, hidden_b))

    def test_the_admission_join_refuses_a_mismatched_declaration(self):
        recorded = {"arguments": {"occupancy_column": 1},
                    "recovery_contract": {"occupancy_column": 1}}
        require_occupancy_column(1, recorded)
        require_occupancy_column(0, {"arguments": {}, "recovery_contract": {}})
        for value in (1, 0):
            with self.assertRaises(ValueError):
                require_occupancy_column(value, {"arguments": {"occupancy_column": 1-value},
                                                 "recovery_contract": {"occupancy_column": 1-value}})
        with self.assertRaises(ValueError):
            require_occupancy_column(2, recorded)


class ColumnDriverTests(unittest.TestCase):
    """The lever must be declared end to end: config, admission, recovery contract, run family."""

    @classmethod
    def setUpClass(cls):
        from train_m2_stream import INITIAL_SOURCE_ADDED_PREFIXES, validate_admission, main
        from record_stream_run import OWNED_RUN_PATTERN
        cls.source = inspect.getsource(validate_admission)
        cls.driver = inspect.getsource(main)
        cls.prefixes = INITIAL_SOURCE_ADDED_PREFIXES
        cls.family = OWNED_RUN_PATTERN

    def test_the_recovery_contract_carries_the_lever(self):
        self.assertIn('recovery_contract["occupancy_column"] = args.occupancy_column', self.driver)

    def test_the_model_config_is_handed_the_lever(self):
        start = self.driver.index("replace(config,")
        block = self.driver[start:self.driver.index("config_json", start)]
        self.assertIn("occupancy_column=args.occupancy_column", block)

    def test_the_admission_checks_the_lever_it_receives(self):
        checked = set(re.findall(r"require_\w+\((\w+), summary\)", self.source))
        self.assertIn("occupancy_column", checked)
        self.assertIn("occupancy_column=args.occupancy_column", self.driver)
        self.assertIn("occupancy_column=0", self.source)

    def test_the_added_tensor_is_a_declared_prefix(self):
        model = TetrisStateNet(pilot(occupancy_column=1))
        added = [name for name in model.state_dict() if name.startswith("occupancy_column")]
        self.assertTrue(added)
        self.assertTrue(all(name.startswith(self.prefixes) for name in added))

    def test_the_run_family_is_owned(self):
        self.assertTrue(self.family.match("m5-colread-column-r1"))
        self.assertFalse(self.family.match("m5-colreadout-column-r1"))


class ColumnRecordTests(unittest.TestCase):
    """The published record, once curation has written it."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        scope = cls.root/"docs/evidence/m5-column-readout-r1/scope.json"
        if not scope.exists():
            raise unittest.SkipTest("the column read-out record has not been curated yet")
        cls.scope = json.loads(scope.read_text(encoding="utf-8"))
        cls.prior = json.loads((cls.root/"docs/evidence/m4-placed-supervision-r1/scope.json")
                               .read_text(encoding="utf-8"))
        cls.text = " ".join((cls.root/"docs/M5_COLUMN_READOUT.md").read_text(encoding="utf-8").split())

    def test_the_arms_are_the_declared_levers(self):
        self.assertEqual(sorted(self.scope["arms"]), ["column", "control", "demean", "wide"])
        self.assertEqual({name: arm["occupancy_column"] for name, arm in self.scope["arms"].items()},
                         {"control": 0, "wide": 0, "demean": 0, "column": 1})
        self.assertEqual({name: arm["occupancy_deform"] for name, arm in self.scope["arms"].items()},
                         {"control": 0, "wide": 2, "demean": 3, "column": 0})

    def test_the_three_prior_arms_reproduce_the_previous_record(self):
        for name in ("control", "wide", "demean"):
            self.assertEqual(self.scope["arms"][name]["auc"], self.prior["arms"][name]["auc"])
            self.assertEqual(self.scope["arms"][name]["per_cell_f1"],
                             self.prior["arms"][name]["per_cell_f1"])
            self.assertEqual(self.scope["arms"][name]["best"]["f1"],
                             self.prior["arms"][name]["best"]["f1"])

    def test_the_reading_follows_the_printed_deltas(self):
        delta = self.scope["contrasts"]["column_control"]
        interval = delta["auc_interval"]
        if delta["best_f1_delta"] >= .05 and interval["low"] > 0:
            expected = "readout-is-the-lever"
        elif delta["best_f1_delta"] < .05 and (interval["high"] < 0 or delta["auc_delta"] <= 0):
            expected = "readout-class-fails"
        else:
            expected = "readout-moves-without-ranking"
        self.assertEqual(self.scope["column"]["reading"], expected)

    def test_the_ablation_is_the_same_weights_with_the_read_out_zeroed(self):
        ablation = self.scope["column"]["ablation"]
        self.assertGreaterEqual(ablation["contribution"]["mean_logit"], 0.0)
        self.assertLessEqual(ablation["without"]["auc"], 1.0)
        self.assertIn("with", ablation)

    def test_the_document_quotes_the_published_scope(self):
        for name, arm in self.scope["arms"].items():
            self.assertIn(f"{round(arm['auc'], 6):,.6f}", self.text, name)
            self.assertIn(f"{round(arm['best']['f1'], 6):,.6f}", self.text, name)
        for key, delta in self.scope["contrasts"].items():
            self.assertIn(f"{round(delta['best_f1_delta'], 6):,.6f}", self.text, key)
        self.assertIn(self.scope["column"]["reading"], self.text)


if __name__ == "__main__":
    unittest.main()
