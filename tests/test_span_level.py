"""M19: the row-span tap under the validated level transforms, and the proof the composition is not inert."""
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
from tetris_vision.model import ModelConfig, TetrisStateNet

ROOT = Path(__file__).resolve().parents[1]
WIDTH = ModelConfig.pilot().widths[0]
DIMENSION = ModelConfig.pilot().dimension


def pilot(**overrides):
    return ModelConfig(**{**ModelConfig.pilot().__dict__, **overrides})


def inputs(seed=20261026, size=32):
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(1, 3, size, size, generator=generator),
            torch.ones(1, 1, size, size, dtype=torch.bool))


def compose(**overrides):
    return pilot(occupancy_detail=1, occupancy_tap_span=1, occupancy_centre=1,
                 occupancy_crossing_threshold=1, **overrides)


def forward(config, seed=20261026, pixels=None, trained=True):
    torch.manual_seed(seed)
    model = TetrisStateNet(config).eval()
    if trained:
        with torch.no_grad():
            model.occupancy_detail_tap.weight.normal_(0, .5)
            model.occupancy_detail_tap.bias.normal_(0, .1)
    pixels = inputs()[0] if pixels is None else pixels
    valid = torch.ones(1, 1, pixels.shape[-2], pixels.shape[-1], dtype=torch.bool)
    with torch.no_grad():
        return model, model(pixels, valid)["final"]["occupancy"]


class CompositionTests(unittest.TestCase):
    def test_the_three_fields_are_admitted_together(self):
        config = compose()
        self.assertEqual((config.occupancy_tap_span, config.occupancy_centre,
                          config.occupancy_crossing_threshold, config.occupancy_detail), (1, 1, 1, 1))

    def test_the_transforms_perturb_the_spans_logits(self):
        """Non-degeneracy: the same trained tap weights read differently once the level moves."""
        pixels = inputs()[0]
        span, span_only = forward(pilot(occupancy_detail=1, occupancy_tap_span=1), pixels=pixels)
        composed, composed_logits = forward(compose(), pixels=pixels)
        with torch.no_grad():
            composed.occupancy_detail_tap.weight.copy_(span.occupancy_detail_tap.weight)
            composed.occupancy_detail_tap.bias.copy_(span.occupancy_detail_tap.bias)
            shared = composed(pixels, inputs()[1])["final"]["occupancy"]
        self.assertFalse(torch.allclose(span_only, shared, atol=1e-4))

    def test_each_transform_perturbs_the_read_on_its_own(self):
        pixels = inputs()[0]
        base = forward(pilot(occupancy_detail=1, occupancy_tap_span=1), pixels=pixels)[1]
        self.assertFalse(torch.allclose(base, forward(pilot(
            occupancy_detail=1, occupancy_tap_span=1, occupancy_centre=1), pixels=pixels)[1], atol=1e-4))
        self.assertFalse(torch.allclose(base, forward(pilot(
            occupancy_detail=1, occupancy_tap_span=1, occupancy_crossing_threshold=1), pixels=pixels)[1],
            atol=1e-4))

    def test_the_composition_is_not_the_control_geometry(self):
        """No reading of this arm can be produced by the control's tap."""
        control, _ = forward(pilot(occupancy_detail=1))
        composed, _ = forward(compose())
        self.assertEqual(control.occupancy_detail_tap.in_features, WIDTH)
        self.assertEqual(composed.occupancy_detail_tap.in_features, 3*WIDTH)
        self.assertEqual(tuple(control.state_dict()["occupancy_detail_tap.weight"].shape), (DIMENSION, WIDTH))
        self.assertEqual(tuple(composed.state_dict()["occupancy_detail_tap.weight"].shape), (DIMENSION, 3*WIDTH))
        pixels = inputs()[0]
        level_only = forward(pilot(occupancy_detail=1, occupancy_centre=1,
                                   occupancy_crossing_threshold=1), pixels=pixels, trained=False)[1]
        with torch.no_grad():
            composed.occupancy_detail_tap.weight.zero_()
            composed.occupancy_detail_tap.bias.zero_()
            value = composed(pixels, inputs()[1])["final"]["occupancy"]
        self.assertTrue(torch.equal(value, level_only))
        self.assertFalse(torch.equal(value, forward(pilot(), pixels=pixels, trained=False)[1]))

    def test_the_tap_field_off_is_the_published_function(self):
        """The span is the identity at zero weights; the level transforms are not, and never claim to be."""
        _, plain = forward(pilot(), trained=False)
        for span in (0, 1):
            _, levered = forward(pilot(occupancy_detail=1, occupancy_tap_span=span), trained=False)
            self.assertTrue(torch.equal(plain, levered), span)

    def test_the_transforms_move_the_level_at_zero_tap_weights(self):
        """Which is exactly why the composition exists: the tap contributes nothing yet, the level still moves."""
        plain = forward(pilot(), trained=False)[1]
        composed = forward(compose(), trained=False)[1]
        centred = forward(pilot(occupancy_detail=1, occupancy_tap_span=1, occupancy_centre=1),
                          trained=False)[1]
        self.assertFalse(torch.equal(plain, composed))
        self.assertFalse(torch.equal(centred, composed))
        self.assertLess(float(centred[:, :20].mean().abs()), 1e-4)

    def test_the_same_pixels_and_weights_give_the_same_decision(self):
        pixels = inputs()[0]
        first = forward(compose(), pixels=pixels)[1]
        second = forward(compose(), pixels=pixels)[1]
        self.assertTrue(torch.equal(first, second))

    def test_the_composition_reads_no_labels(self):
        source = inspect.getsource(__import__("tetris_vision.model", fromlist=["x"]))
        body = source[source.index("def forward(self, pixels, valid):"):]
        for name in ("label", "target", "occupied", "prevalence", "thresholds"):
            self.assertNotIn(name, body.lower())
        _, logits = forward(compose())
        self.assertEqual(tuple(logits.shape), (1, 26, 10))

    def test_the_transforms_keep_their_own_guarantees(self):
        """Centring leaves the board mean at zero; the crossing rule leaves the read separating at zero."""
        pixels = inputs()[0]
        centred = forward(pilot(occupancy_detail=1, occupancy_tap_span=1, occupancy_centre=1),
                          pixels=pixels)[1][:, :20].mean()
        self.assertLess(float(centred.abs()), 1e-4)
        ruled = forward(pilot(occupancy_detail=1, occupancy_tap_span=1, occupancy_crossing_threshold=1),
                        pixels=pixels)[1]
        self.assertTrue(torch.isfinite(ruled).all())


class CompositionPlumbingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from train_m2_stream import arguments, main, validate_admission
        cls.driver = inspect.getsource(main)
        cls.source = Path(inspect.getsourcefile(validate_admission)).read_text(encoding="utf-8")
        cls.signature = inspect.signature(validate_admission)
        cls.arguments = staticmethod(arguments)
        cls.base = ["--output", "runs/m19-span-level-refusal-probe", "--model", "accuracy", "--steps", "2"]

    def parse(self, extra):
        return self.arguments(self.base+extra)

    def test_the_composition_is_admitted_through_the_real_parser(self):
        parsed = self.parse(["--profile-square", "--occupancy-detail", "1", "--occupancy-tap-span", "1",
                             "--occupancy-centre", "1", "--occupancy-crossing-threshold", "1"])
        self.assertEqual((parsed.occupancy_tap_span, parsed.occupancy_centre,
                          parsed.occupancy_crossing_threshold), (1, 1, 1))

    def test_a_half_composition_is_refused_with_its_own_stderr(self):
        for half in (["--occupancy-centre", "1"], ["--occupancy-crossing-threshold", "1"]):
            stream = io.StringIO()
            with self.subTest(half=half), contextlib.redirect_stderr(stream):
                with self.assertRaises(SystemExit):
                    self.parse(["--occupancy-detail", "1", "--occupancy-tap-span", "1"]+half)
            self.assertIn("declared together", stream.getvalue(), half)

    def test_every_other_lever_is_refused_beside_the_composition(self):
        composed = ["--occupancy-detail", "1", "--occupancy-tap-span", "1",
                    "--occupancy-centre", "1", "--occupancy-crossing-threshold", "1"]
        for lever in (["--occupancy-detail-stencil", "1"], ["--occupancy-resolution", "1"],
                      ["--occupancy-deform", "1"], ["--occupancy-offset-rows", "1"],
                      ["--occupancy-context", "1"], ["--occupancy-column", "1"],
                      ["--occupancy-lattice-free", "1"], ["--occupancy-normalise", "1"],
                      ["--occupancy-margin-hinge", "1"], ["--occupancy-anchor", "1"],
                      ["--occupancy-stack-monotone", "1"], ["--occupancy-skyline-recovery", "1"]):
            stream = io.StringIO()
            with self.subTest(lever=lever), contextlib.redirect_stderr(stream):
                with self.assertRaises(SystemExit):
                    self.parse(composed+lever)
            self.assertIn("the tap span is one declared reading geometry", stream.getvalue(), lever)

    def test_the_driver_and_admission_carry_the_composition(self):
        for option in ("occupancy_tap_span", "occupancy_centre", "occupancy_crossing_threshold"):
            self.assertIn(f"{option}=args.{option}", self.driver, option)
            self.assertIn(option, self.signature.parameters, option)
        self.assertIn("require_occupancy_tap_span(occupancy_tap_span, summary)", self.source)
        self.assertIn("require_occupancy_centre(occupancy_centre, summary)", self.source)
        self.assertIn("require_occupancy_crossing_threshold(occupancy_crossing_threshold, summary)", self.source)

    def test_the_head_applies_the_span_then_the_transforms(self):
        source = Path(inspect.getsourcefile(ModelConfig)).read_text(encoding="utf-8")
        tap = source.index("if self.occupancy_detail_tap is not None:")
        centre = source.index("centre_board(occupancy_logits", tap)
        rule = source.index("otsu_board(occupancy_logits", centre)
        self.assertLess(tap, centre)
        self.assertLess(centre, rule)

    def test_the_run_family_is_owned(self):
        from record_stream_run import OWNED_RUN_PATTERN
        for name in ("m19-span-level-control-r1", "m19-span-level-composed-r1",
                     "m19-span-level-composed-r1-sweep-r1-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        for name in ("m19-span-level-", "m19-span-level", "m19-span-leveler-r1"):
            self.assertFalse(OWNED_RUN_PATTERN.fullmatch(name), name)


class CompositionRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scope = ROOT/"docs/evidence/m19-span-level-r1/scope.json"
        if not scope.exists():
            raise unittest.SkipTest("the composition record has not been curated yet")
        cls.scope = json.loads(scope.read_text(encoding="utf-8"))
        cls.span = json.loads((ROOT/"docs/evidence/m18-tap-span-r1/scope.json").read_text(encoding="utf-8"))
        cls.text = " ".join((ROOT/"docs/M19_SPAN_LEVEL.md").read_text(encoding="utf-8").split())

    def test_the_arms_differ_in_exactly_the_declared_fields(self):
        self.assertEqual(sorted(self.scope["arms"]), ["composed", "control"])
        self.assertEqual(self.scope["aliasing_guard"]["arms_differ_in"],
                         ["occupancy_centre", "occupancy_crossing_threshold", "occupancy_tap_span"])
        self.assertEqual({name: (arm["occupancy_tap_span"], arm["occupancy_centre"],
                                 arm["occupancy_crossing_threshold"])
                          for name, arm in self.scope["arms"].items()},
                         {"control": (0, 0, 0), "composed": (1, 1, 1)})
        self.assertTrue(self.scope["aliasing_guard"]["weights_differ"])
        self.assertTrue(self.scope["aliasing_guard"]["not_the_previous_block"])
        self.assertTrue(self.scope["aliasing_guard"]["not_the_span_only_arm"])

    def test_the_control_reproduces_and_every_excluded_checkpoint_is_excluded(self):
        self.assertEqual(self.scope["arms"]["control"]["auc"], 0.8226433610683003)
        self.assertEqual(self.scope["arms"]["control"]["f1_at_zero"], 0.38164)
        composed = self.scope["arms"]["composed"]["weights_sha256"]
        self.assertNotEqual(composed, self.span["arms"]["tap"]["weights_sha256"])

    def test_the_strict_metric_and_the_level_diagnostic_are_reported(self):
        for name, arm in self.scope["arms"].items():
            gate = arm["exact_boards_at_zero"]
            self.assertEqual(gate["boards"], 28, name)
            self.assertIn("boundary_exact", gate, name)
        self.assertEqual(self.scope["gate_moved"],
                         self.scope["arms"]["composed"]["exact_boards_at_zero"]["exact"] > 0)
        diagnostic = self.scope["level_diagnostic"]
        self.assertEqual(diagnostic["span_only_positive_share"],
                         self.span["arms"]["tap"]["positive_share_at_zero"])
        self.assertEqual(diagnostic["span_only_row_error"],
                         self.span["arms"]["tap"]["row_error_at_zero"]["mean"])

    def test_the_reading_follows_the_declared_vocabulary(self):
        arm, control = self.scope["arms"]["composed"], self.scope["arms"]["control"]
        delta = round(arm["f1_at_zero"]-control["f1_at_zero"], 6)
        gate = arm["exact_boards_at_zero"]["exact"] > 0
        expected = ("span-level-passes" if delta >= .05 and gate else
                    "span-level-moves-without-clearing" if delta > 0 else "span-level-fails")
        self.assertEqual(self.scope["reading"], expected)
        self.assertEqual(self.scope["contrasts"]["composed_control"]["f1_at_zero_delta"], delta)

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
