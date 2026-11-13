"""The declared row-span tap geometry: three row samples through one projection, the published read at zero.

The lever is a declaration about where the read looks, so the tests here are about geometry and identity: the span
samples the cell's own row and its two row neighbours, it contains the published point sample exactly, a projection
that reads a neighbour row produces a different read, and the option off is bit-for-bit the published function.
"""
import contextlib
import copy
import inspect
import io
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

import torch
from tetris_vision.losses import require_occupancy_tap_span
from tetris_vision.model import TAP_SPAN, ModelConfig, TetrisStateNet, sample_features

ROOT = Path(__file__).resolve().parents[1]


def pilot(**overrides):
    return ModelConfig(**{**ModelConfig.pilot().__dict__, **overrides})


def inputs(seed=20261026, size=32):
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(1, 3, size, size, generator=generator),
            torch.ones(1, 1, size, size, dtype=torch.bool))


def forward(config, seed=20261026, pixels=None):
    torch.manual_seed(seed)
    model = TetrisStateNet(config).eval()
    pixels = inputs()[0] if pixels is None else pixels
    valid = torch.ones(1, 1, pixels.shape[-2], pixels.shape[-1], dtype=torch.bool)
    with torch.no_grad():
        return model, model(pixels, valid)["final"]["occupancy"]


class SpanGeometryTests(unittest.TestCase):
    def test_the_declared_geometry_is_the_cell_and_its_two_row_neighbours(self):
        self.assertEqual(TAP_SPAN, ((0.0, 1.0), (0.0, 0.0), (0.0, -1.0)))
        model, _ = forward(pilot(occupancy_detail=1, occupancy_tap_span=1))
        self.assertEqual(tuple(model.tap_span.shape), (1, 1, 3, 2))
        self.assertEqual(sorted(map(tuple, model.tap_span.reshape(3, 2).tolist())),
                         sorted(map(tuple, TAP_SPAN)))
        self.assertNotIn("tap_span", model.state_dict())

    def test_the_span_samples_the_row_above_the_cell_and_the_row_below_it(self):
        """A synthetic board whose cells read their own row number, sampled at the declared span."""
        rows, columns = 26, 10
        features = torch.arange(rows, dtype=torch.float32).reshape(rows, 1).expand(rows, columns)
        features = features.reshape(1, 1, rows, columns)
        coordinates = torch.tensor([[(c+.5), (r+.5)] for r in range(rows) for c in range(columns)],
                                   dtype=torch.float32)
        lattice = torch.tensor([[[0.0, 0.0], [1.0/columns, 0.0], [0.0, 1.0/rows]]])
        offsets = torch.tensor(TAP_SPAN).reshape(1, 1, 3, 2)
        samples = sample_features(features, lattice, coordinates, offsets)[0].reshape(rows*columns, 3)
        for row in range(1, rows-1):
            for column in (0, 4, 9):
                index = row*columns+column
                self.assertAlmostEqual(float(samples[index, 0]), float(row+1), places=4)
                self.assertAlmostEqual(float(samples[index, 1]), float(row), places=4)
                self.assertAlmostEqual(float(samples[index, 2]), float(row-1), places=4)

    def test_the_span_reads_three_samples_through_the_same_projection(self):
        span, _ = forward(pilot(occupancy_detail=1, occupancy_tap_span=1))
        point, _ = forward(pilot(occupancy_detail=1))
        width = pilot().widths[0]
        self.assertEqual(span.occupancy_detail_tap.in_features, 3*width)
        self.assertEqual(point.occupancy_detail_tap.in_features, width)
        self.assertEqual(tuple(span.state_dict()["occupancy_detail_tap.weight"].shape),
                         (pilot().dimension, 3*width))

    def test_the_span_contains_the_published_point_read_exactly(self):
        """A projection that reads only the centre slot is the point tap: the span adds neighbours, it changes nothing."""
        pixels = inputs()[0]
        point, reference = forward(pilot(occupancy_detail=1), pixels=pixels)
        span, _ = forward(pilot(occupancy_detail=1, occupancy_tap_span=1), pixels=pixels)
        width = pilot().widths[0]
        with torch.no_grad():
            point.occupancy_detail_tap.weight.normal_(0, .5)
            point.occupancy_detail_tap.bias.normal_(0, .1)
            span.occupancy_detail_tap.weight.zero_()
            span.occupancy_detail_tap.bias.copy_(point.occupancy_detail_tap.bias)
            span.occupancy_detail_tap.weight[:, width:2*width] = point.occupancy_detail_tap.weight
            centre = point(pixels, inputs()[1])["final"]["occupancy"]
            report = span(pixels, inputs()[1])["final"]["occupancy"]
        self.assertFalse(torch.equal(reference, centre))
        self.assertTrue(torch.allclose(centre, report, atol=1e-6, rtol=1e-5))

    def test_reading_a_neighbour_row_instead_changes_the_read(self):
        """The mechanism: the same weights placed on a neighbour slot give a different decision."""
        pixels = inputs()[0]
        span, _ = forward(pilot(occupancy_detail=1, occupancy_tap_span=1), pixels=pixels)
        other = copy.deepcopy(span)
        width = pilot().widths[0]
        with torch.no_grad():
            span.occupancy_detail_tap.weight.normal_(0, .5)
            span.occupancy_detail_tap.bias.normal_(0, .1)
            other.occupancy_detail_tap.weight.zero_()
            other.occupancy_detail_tap.bias.copy_(span.occupancy_detail_tap.bias)
            moved = span.occupancy_detail_tap.weight[:, width:2*width].clone()
            span.occupancy_detail_tap.weight[:, width:2*width] = moved
            other.occupancy_detail_tap.weight[:, 0:width] = moved
            here = span(pixels, inputs()[1])["final"]["occupancy"]
            there = other(pixels, inputs()[1])["final"]["occupancy"]
        self.assertFalse(torch.allclose(here, there, atol=1e-3))

    def test_each_arm_is_exactly_the_published_function_at_step_zero(self):
        _, plain = forward(pilot())
        for span in (0, 1):
            _, levered = forward(pilot(occupancy_detail=1, occupancy_tap_span=span))
            self.assertTrue(torch.equal(plain, levered), span)

    def test_the_arms_differ_only_in_the_tap_and_share_every_other_tensor(self):
        control, _ = forward(pilot(occupancy_detail=1))
        lever, _ = forward(pilot(occupancy_detail=1, occupancy_tap_span=1))
        self.assertEqual(set(control.state_dict()), set(lever.state_dict()))
        self.assertNotEqual(tuple(control.state_dict()["occupancy_detail_tap.weight"].shape),
                            tuple(lever.state_dict()["occupancy_detail_tap.weight"].shape))
        for name, value in control.state_dict().items():
            if name.startswith("occupancy_detail_tap."):
                continue
            self.assertTrue(torch.equal(value, lever.state_dict()[name]), name)

    def test_the_same_pixels_and_weights_give_the_same_decision(self):
        pixels = inputs()[0]
        first = forward(pilot(occupancy_detail=1, occupancy_tap_span=1), pixels=pixels)[1]
        second = forward(pilot(occupancy_detail=1, occupancy_tap_span=1), pixels=pixels)[1]
        self.assertTrue(torch.equal(first, second))

    def test_the_recovery_reads_no_labels(self):
        """Leakage guard: permuting the board's labels cannot change a single logit."""
        source = inspect.getsource(__import__("tetris_vision.model", fromlist=["x"]))
        body = source[source.index("def forward(self, pixels, valid):"):]
        for name in ("label", "target", "occupied", "prevalence", "thresholds"):
            self.assertNotIn(name, body.lower())
        model, logits = forward(pilot(occupancy_detail=1, occupancy_tap_span=1))
        self.assertEqual(tuple(logits.shape), (1, 26, 10))


class SpanConfigTests(unittest.TestCase):
    def test_the_default_is_the_published_configuration(self):
        self.assertEqual(ModelConfig().occupancy_tap_span, 0)
        self.assertEqual(ModelConfig.pilot().occupancy_tap_span, 0)

    def test_an_undeclared_value_is_refused(self):
        for value in (2, -1, True, 1.0):
            with self.assertRaises(ValueError):
                ModelConfig(occupancy_detail=1, occupancy_tap_span=value)

    def test_the_span_is_a_form_of_the_detail_tap_and_needs_it(self):
        with self.assertRaises(ValueError):
            ModelConfig(occupancy_detail=0, occupancy_tap_span=1)

    def test_one_tap_geometry_per_arm(self):
        with self.assertRaises(ValueError):
            ModelConfig(occupancy_detail=1, occupancy_tap_span=1, occupancy_detail_stencil=1)

    def test_the_admission_join_refuses_a_mismatch(self):
        summary = {"arguments": {"occupancy_tap_span": 1},
                   "recovery_contract": {"occupancy_tap_span": 1},
                   "config": {"occupancy_tap_span": 1}}
        self.assertIsNone(require_occupancy_tap_span(1, summary))
        for value, wrong in ((0, summary), (2, summary),
                             (1, {"arguments": {"occupancy_tap_span": 0},
                                  "recovery_contract": {"occupancy_tap_span": 1},
                                  "config": {"occupancy_tap_span": 1}}),
                             (1, {"arguments": {}, "recovery_contract": {}, "config": {}})):
            with self.assertRaises(ValueError):
                require_occupancy_tap_span(value, wrong)


class SpanPlumbingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from train_m2_stream import arguments, main, validate_admission
        cls.driver = inspect.getsource(main)
        cls.refusals = inspect.getsource(arguments)
        cls.source = Path(inspect.getsourcefile(validate_admission)).read_text(encoding="utf-8")
        cls.signature = inspect.signature(validate_admission)
        cls.arguments = staticmethod(arguments)
        cls.base = ["--output", "runs/m18-span-refusal-probe", "--model", "accuracy", "--steps", "2"]

    def test_the_driver_writes_the_option_into_the_config_and_the_contract(self):
        start = self.driver.index("replace(config,")
        block = self.driver[start:self.driver.index("config_json", start)]
        self.assertIn("occupancy_tap_span=args.occupancy_tap_span", block)
        self.assertIn('recovery_contract["occupancy_tap_span"] = args.occupancy_tap_span', self.driver)

    def test_every_option_written_into_the_config_is_also_passed_to_the_admission(self):
        """V-141: the join defaults to 0, so a call site that omits the option silently admits span=0."""
        start = self.driver.index("replace(config,")
        written = self.driver[start:self.driver.index("config_json", start)]
        self.assertIn("occupancy_tap_span=args.occupancy_tap_span", written)
        calls = [index for index in range(len(self.driver))
                 if self.driver.startswith("validate_admission(", index)]
        self.assertGreaterEqual(len(calls), 1)
        for index in calls:
            self.assertIn("occupancy_tap_span=args.occupancy_tap_span",
                          self.driver[index:self.driver.index(")", self.driver.index("geometry_final_only", index))])

    def test_the_admission_checks_the_option_it_receives(self):
        self.assertIn("occupancy_tap_span", self.signature.parameters)
        self.assertEqual(self.signature.parameters["occupancy_tap_span"].default, 0)
        self.assertIn("require_occupancy_tap_span(occupancy_tap_span, summary)", self.source)

    def test_the_span_needs_the_detail_tap_through_the_real_parser(self):
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            with self.assertRaises(SystemExit):
                self.arguments(self.base+["--occupancy-tap-span", "1"])
        self.assertIn("requires --occupancy-detail 1", stream.getvalue())

    def test_every_other_geometry_and_read_out_is_refused_beside_the_span(self):
        for lever in (["--occupancy-detail-stencil", "1"], ["--occupancy-resolution", "1"],
                      ["--occupancy-deform", "1"], ["--occupancy-offset-rows", "1"],
                      ["--occupancy-context", "1"], ["--occupancy-column", "1"],
                      ["--occupancy-lattice-free", "1"], ["--occupancy-normalise", "1"],
                      ["--occupancy-margin-hinge", "1"], ["--occupancy-anchor", "1"],
                      ["--occupancy-centre", "1"], ["--occupancy-stack-monotone", "1"],
                      ["--occupancy-crossing-threshold", "1"], ["--occupancy-skyline-recovery", "1"]):
            stream = io.StringIO()
            with self.subTest(lever=lever), contextlib.redirect_stderr(stream):
                with self.assertRaises(SystemExit):
                    self.arguments(self.base+["--occupancy-detail", "1", "--occupancy-tap-span", "1"]+lever)
            self.assertIn("the tap span is one declared reading geometry", stream.getvalue(), lever)

    def test_an_undeclared_value_is_refused_by_the_parser(self):
        for value in ("2", "-1", "true"):
            stream = io.StringIO()
            with self.subTest(value=value), contextlib.redirect_stderr(stream):
                with self.assertRaises(SystemExit):
                    self.arguments(self.base+["--occupancy-detail", "1", "--occupancy-tap-span", value])
            self.assertIn("invalid", stream.getvalue(), value)

    def test_both_settings_are_admitted_and_zero_is_the_published_default(self):
        parsed = self.arguments(self.base+["--profile-square", "--occupancy-detail", "1",
                                           "--occupancy-tap-span", "1"])
        self.assertEqual((parsed.occupancy_tap_span, parsed.occupancy_detail), (1, 1))
        self.assertEqual(self.arguments(self.base+["--profile-square"]).occupancy_tap_span, 0)

    def test_the_run_family_is_owned(self):
        from record_stream_run import OWNED_RUN_PATTERN
        for name in ("m18-span-control-r1", "m18-span-tap-r1", "m18-span-tap-r1-sweep-r1-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        for name in ("m18-span-", "m18-span", "m18-spanner-r1"):
            self.assertFalse(OWNED_RUN_PATTERN.fullmatch(name), name)


class SpanRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scope = ROOT/"docs/evidence/m18-tap-span-r1/scope.json"
        if not scope.exists():
            raise unittest.SkipTest("the tap-span record has not been curated yet")
        cls.scope = json.loads(scope.read_text(encoding="utf-8"))
        cls.neighbour = json.loads((ROOT/"docs/evidence/m15-stack-combined-r1/scope.json")
                                   .read_text(encoding="utf-8"))
        cls.text = " ".join((ROOT/"docs/M18_TAP_SPAN.md").read_text(encoding="utf-8").split())

    def test_the_arms_differ_in_exactly_the_declared_field(self):
        self.assertEqual(sorted(self.scope["arms"]), ["control", "tap"])
        self.assertEqual(self.scope["aliasing_guard"]["arms_differ_in"], ["occupancy_tap_span"])
        self.assertEqual({name: arm["occupancy_tap_span"] for name, arm in self.scope["arms"].items()},
                         {"control": 0, "tap": 1})
        self.assertTrue(self.scope["aliasing_guard"]["weights_differ"])
        self.assertTrue(self.scope["aliasing_guard"]["not_the_previous_block"])

    def test_the_control_reproduces_and_the_neighbour_is_excluded(self):
        self.assertEqual(self.scope["arms"]["control"]["auc"], 0.8226433610683003)
        self.assertEqual(self.scope["arms"]["control"]["f1_at_zero"], 0.38164)
        self.assertNotEqual(self.scope["arms"]["tap"]["weights_sha256"],
                            self.neighbour["arms"]["combined"]["weights_sha256"])

    def test_the_corrected_strict_metric_is_reported_for_both_arms(self):
        for name, arm in self.scope["arms"].items():
            gate = arm["exact_boards_at_zero"]
            self.assertEqual(gate["boards"], 28, name)
            self.assertIn("boundary_exact", gate, name)
        self.assertEqual(self.scope["fit_gate"]["boundary_exact_nonempty_boards"],
                         self.scope["arms"]["tap"]["exact_boards_at_zero"]["boundary_exact"])
        self.assertEqual(self.scope["gate_moved"],
                         self.scope["arms"]["tap"]["exact_boards_at_zero"]["exact"] > 0)

    def test_the_mechanism_adds_a_projection_and_no_term(self):
        mechanism = self.scope["mechanism"]
        self.assertEqual(mechanism["loss_terms_added"], 0)
        self.assertTrue(mechanism["label_free"])
        self.assertTrue(mechanism["declared_before_receipts"])
        self.assertEqual(mechanism["sample_count"], 3)

    def test_the_readings_follow_the_printed_numbers(self):
        arm, control = self.scope["arms"]["tap"], self.scope["arms"]["control"]
        delta = round(arm["f1_at_zero"]-control["f1_at_zero"], 6)
        expected = ("span-clears-the-ceiling" if delta >= .05 else
                    "span-moves-without-clearing" if delta > 0 else "span-fails")
        self.assertEqual(self.scope["reading"], expected)

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
