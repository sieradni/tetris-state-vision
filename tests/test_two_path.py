"""M20: the summed two-path read, its non-degeneracy, and the channel split that says which failure it was."""
import contextlib
import inspect
import io
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

import torch
from tetris_vision.model import ModelConfig, TetrisStateNet, stack_contiguous

ROOT = Path(__file__).resolve().parents[1]
ROWS = 20


def pilot(**overrides):
    return ModelConfig(**{**ModelConfig.pilot().__dict__, **overrides})


def arm(**overrides):
    return pilot(occupancy_detail=1, occupancy_stack_monotone=1, occupancy_lattice_free=2, **overrides)


def inputs(seed=20261026, size=32):
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(1, 3, size, size, generator=generator),
            torch.ones(1, 1, size, size, dtype=torch.bool))


def build(config, seed=20261026, trained=True):
    torch.manual_seed(seed)
    model = TetrisStateNet(config).eval()
    if trained:
        with torch.no_grad():
            model.occupancy_detail_tap.weight.normal_(0, .5)
            model.occupancy_detail_tap.bias.normal_(0, .1)
            if model.occupancy_lattice_free is not None:
                for layer in (model.occupancy_lattice_free[0], model.occupancy_lattice_free[2]):
                    layer.weight.normal_(0, .5)
                    layer.bias.normal_(0, .1)
    pixels = inputs()[0]
    with torch.no_grad():
        return model, model(pixels, inputs()[1])["final"]


class TwoPathMechanismTests(unittest.TestCase):
    def test_the_arm_sums_both_channels_and_keeps_its_parts(self):
        model, final = build(arm())
        self.assertEqual(model.config.occupancy_lattice_free, 2)
        self.assertEqual(sorted(final["occupancy_parts"]), ["free", "lattice"])
        summed = final["occupancy_parts"]["lattice"]+final["occupancy_parts"]["free"]
        self.assertTrue(torch.allclose(stack_contiguous(summed, ROWS), final["occupancy"], atol=1e-6))

    def test_step_zero_is_the_envelope_of_the_published_read(self):
        published, final = build(pilot(occupancy_stack_monotone=1), trained=False)
        untrained, arm_final = build(arm(), trained=False)
        self.assertIsNone(untrained.occupancy_lattice_gate, "the summed form builds no gate (V-143)")
        self.assertTrue(torch.equal(arm_final["occupancy"], final["occupancy"]))
        self.assertTrue(torch.equal(arm_final["occupancy_parts"]["free"],
                                    torch.zeros_like(arm_final["occupancy_parts"]["free"])))

    def test_zeroing_the_channel_reduces_the_sum_exactly_to_the_lattice_read(self):
        """The sum is additive: silencing the free channel's output leaves the lattice read untouched."""
        model, final = build(arm())
        self.assertFalse(torch.equal(final["occupancy"],
                                     stack_contiguous(final["occupancy_parts"]["lattice"], ROWS)))
        with torch.no_grad():
            model.occupancy_lattice_free[2].weight.zero_()
            model.occupancy_lattice_free[2].bias.zero_()
            zeroed = model(inputs()[0], inputs()[1])["final"]
        self.assertTrue(torch.equal(zeroed["occupancy"],
                                    stack_contiguous(zeroed["occupancy_parts"]["lattice"], ROWS)))

    def test_the_sum_builds_no_gate_to_get_stuck_on(self):
        """V-143: a gate at zero is a training fixed point, so the summed form has no gate at all."""
        model, _ = build(arm())
        self.assertIsNone(model.occupancy_lattice_gate)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        single, _ = build(pilot(occupancy_lattice_free=1), trained=False)
        self.assertIsNotNone(single.occupancy_lattice_gate)

    def test_zeroing_the_read_out_reduces_the_sum_to_the_free_channel(self):
        """The free channel is a real read, not a decoration: alone it can carry the score."""
        model, _ = build(arm())
        with torch.no_grad():
            model.occupancy.weight.zero_()
            model.occupancy.bias.zero_()
            final = model(inputs()[0], inputs()[1])["final"]
        self.assertTrue(torch.allclose(stack_contiguous(final["occupancy_parts"]["free"], ROWS),
                                       final["occupancy"], atol=1e-5))
        self.assertGreater(float(final["occupancy_parts"]["free"].abs().max()), 0.0)

    def test_each_channel_moves_the_summed_read(self):
        model, base = build(arm())
        with torch.no_grad():
            model.occupancy_lattice_free[2].weight.add_(0.5)
            moved_free = model(inputs()[0], inputs()[1])["final"]["occupancy"]
        self.assertFalse(torch.allclose(base["occupancy"], moved_free, atol=1e-4))
        model, base = build(arm())
        with torch.no_grad():
            model.occupancy.weight.add_(0.5)
            moved_lattice = model(inputs()[0], inputs()[1])["final"]["occupancy"]
        self.assertFalse(torch.allclose(base["occupancy"], moved_lattice, atol=1e-4))

    def test_the_arm_cannot_produce_the_controls_geometry(self):
        control, control_final = build(pilot(occupancy_detail=1, occupancy_stack_monotone=1))
        two_path, arm_final = build(arm())
        self.assertIsNone(control.occupancy_lattice_free)
        self.assertIsNotNone(two_path.occupancy_lattice_free)
        self.assertNotEqual(two_path.state_dict().keys(), control.state_dict().keys())
        self.assertFalse(torch.allclose(control_final["occupancy"], arm_final["occupancy"], atol=1e-4))
        self.assertNotIn("occupancy_parts", control_final)

    def test_the_same_pixels_and_weights_give_the_same_decision(self):
        first = build(arm())[1]["occupancy"]
        second = build(arm())[1]["occupancy"]
        self.assertTrue(torch.equal(first, second))

    def test_the_composition_reads_no_labels(self):
        source = inspect.getsource(__import__("tetris_vision.model", fromlist=["x"]))
        body = source[source.index("def forward(self, pixels, valid):"):]
        for name in ("label", "target", "occupied", "prevalence", "thresholds"):
            self.assertNotIn(name, body.lower())
        self.assertEqual(tuple(build(arm())[1]["occupancy"].shape), (1, 26, 10))


class TwoPathConfigTests(unittest.TestCase):
    def test_the_default_is_the_published_configuration(self):
        self.assertEqual(ModelConfig().occupancy_lattice_free, 0)
        self.assertEqual(ModelConfig.pilot().occupancy_lattice_free, 0)

    def test_the_sum_mode_is_a_declared_value_only(self):
        for value in (3, -1, True, 1.0):
            with self.assertRaises(ValueError):
                ModelConfig(occupancy_stack_monotone=1, occupancy_lattice_free=value)

    def test_the_sum_is_declared_on_its_own_since_m21(self):
        """M20 coupled the sum to the envelope; the M21 decomposition separated them, so the sum stands alone."""
        alone = ModelConfig(occupancy_lattice_free=2)
        self.assertEqual((alone.occupancy_lattice_free, alone.occupancy_stack_monotone), (2, 0))
        self.assertEqual(arm().occupancy_lattice_free, 2)

    def test_the_admission_join_admits_the_sum_mode(self):
        """V-142: the join's declared value set must carry the arm's value, not only the published ones."""
        from tetris_vision.losses import require_occupancy_lattice_free
        summary = {"arguments": {"occupancy_lattice_free": 2},
                   "recovery_contract": {"occupancy_lattice_free": 2}}
        self.assertIsNone(require_occupancy_lattice_free(2, summary))
        with self.assertRaises(ValueError):
            require_occupancy_lattice_free(3, summary)

    def test_the_single_channel_forms_are_unchanged(self):
        self.assertEqual(ModelConfig(occupancy_lattice_free=1).occupancy_lattice_free, 1)
        self.assertEqual(ModelConfig(occupancy_stack_monotone=1).occupancy_stack_monotone, 1)


class TwoPathPlumbingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from train_m2_stream import arguments, main, validate_admission
        cls.driver = inspect.getsource(main)
        cls.source = Path(inspect.getsourcefile(validate_admission)).read_text(encoding="utf-8")
        cls.signature = inspect.signature(validate_admission)
        cls.arguments = staticmethod(arguments)
        cls.base = ["--output", "runs/m20-two-path-refusal-probe", "--model", "accuracy", "--steps", "2",
                    "--profile-square"]

    def test_the_arm_is_admitted_through_the_real_parser(self):
        parsed = self.arguments(self.base+["--occupancy-detail", "1", "--occupancy-stack-monotone", "1",
                                           "--occupancy-lattice-free", "2"])
        self.assertEqual((parsed.occupancy_stack_monotone, parsed.occupancy_lattice_free), (1, 2))

    def test_the_sum_is_admitted_without_the_envelope_since_m21(self):
        """M21: the envelope is a separate mechanism, so the sum must be measurable on its own."""
        parsed = self.arguments(self.base+["--occupancy-detail", "1", "--occupancy-lattice-free", "2"])
        self.assertEqual((parsed.occupancy_stack_monotone, parsed.occupancy_lattice_free), (0, 2))
        parsed = self.arguments(self.base+["--occupancy-detail", "1", "--occupancy-lattice-free", "2",
                                          "--occupancy-stack-monotone", "1", "--occupancy-crossing-threshold", "1"])
        self.assertEqual((parsed.occupancy_stack_monotone, parsed.occupancy_lattice_free,
                         parsed.occupancy_crossing_threshold), (1, 2, 1))

    def test_the_single_channel_beside_the_envelope_stays_refused(self):
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            with self.assertRaises(SystemExit):
                self.arguments(self.base+["--occupancy-detail", "1", "--occupancy-stack-monotone", "1",
                                          "--occupancy-lattice-free", "1"])
        self.assertIn("stack-contiguous read", stream.getvalue())

    def test_every_other_lever_is_refused_beside_the_two_path_read(self):
        composed = ["--occupancy-detail", "1", "--occupancy-stack-monotone", "1", "--occupancy-lattice-free", "2"]
        # The crossing rule left this list in M21: it composes with the sum by declaration, because it decides the
        # score the sum forms rather than changing what is read or where zero sits.
        for lever in (["--occupancy-centre", "1"],
                      ["--occupancy-normalise", "1"], ["--occupancy-skyline-recovery", "1"],
                      ["--occupancy-tap-span", "1"], ["--occupancy-context", "1"], ["--occupancy-column", "1"],
                      ["--occupancy-margin-hinge", "1"], ["--occupancy-anchor", "1"],
                      ["--occupancy-resolution", "1"], ["--occupancy-deform", "1"]):
            stream = io.StringIO()
            with self.subTest(lever=lever), contextlib.redirect_stderr(stream):
                with self.assertRaises(SystemExit):
                    self.arguments(self.base+composed+lever)
            self.assertTrue(stream.getvalue(), lever)

    def test_the_driver_and_admission_carry_the_channel(self):
        for option in ("occupancy_lattice_free", "occupancy_stack_monotone"):
            self.assertIn(f"{option}=args.{option}", self.driver, option)
            self.assertIn(option, self.signature.parameters, option)
        self.assertIn("require_occupancy_lattice_free(occupancy_lattice_free, summary)", self.source)

    def test_the_run_family_is_owned(self):
        from record_stream_run import OWNED_RUN_PATTERN
        for name in ("m20-two-path-control-r1", "m20-two-path-two-path-r1",
                     "m20-two-path-two-path-r1-sweep-r1-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        for name in ("m20-two-path-", "m20-two-path", "m20-two-paths-r1"):
            self.assertFalse(OWNED_RUN_PATTERN.fullmatch(name), name)


class TwoPathRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scope = ROOT/"docs/evidence/m20-two-path-r1/scope.json"
        if not scope.exists():
            raise unittest.SkipTest("the two-path record has not been curated yet")
        cls.scope = json.loads(scope.read_text(encoding="utf-8"))
        cls.text = " ".join((ROOT/"docs/M20_TWO_PATH_READ.md").read_text(encoding="utf-8").split())

    def test_the_arms_differ_in_exactly_the_declared_fields(self):
        self.assertEqual(sorted(self.scope["arms"]), ["control", "two-path"])
        self.assertEqual(self.scope["aliasing_guard"]["arms_differ_in"],
                         ["occupancy_lattice_free", "occupancy_stack_monotone"])
        self.assertEqual({name: (data["occupancy_lattice_free"], data["occupancy_stack_monotone"])
                          for name, data in self.scope["arms"].items()},
                         {"control": (0, 0), "two-path": (2, 1)})

    def test_the_control_reproduces_and_every_excluded_checkpoint_is_excluded(self):
        self.assertEqual(self.scope["arms"]["control"]["auc"], 0.8226433610683003)
        self.assertEqual(self.scope["arms"]["control"]["f1_at_zero"], 0.38164)
        guard = self.scope["aliasing_guard"]
        self.assertTrue(guard["weights_differ"])
        self.assertTrue(guard["not_the_previous_block"])
        self.assertTrue(guard["not_the_span_only_arm"])
        self.assertTrue(guard["not_the_span_under_level_arm"])

    def test_the_channel_split_scores_the_sum_and_both_parts_on_one_fold(self):
        channels = self.scope["channels"]
        self.assertEqual(sorted(key for key in channels if key.endswith(("_only", "summed"))),
                         ["free_only", "lattice_only", "summed"])
        self.assertEqual(round(channels["summed"]["auc"], 6), round(self.scope["arms"]["two-path"]["auc"], 6))
        self.assertEqual(channels["gain_over_lattice_auc"],
                         round(channels["summed"]["auc"]-channels["lattice_only"]["auc"], 6))
        self.assertIn(channels["verdict"], ("complementary", "one-channel-noise-dominates",
                                            "no-complementary-signal"))

    def test_the_reading_and_the_channel_verdict_follow_the_declared_bars(self):
        arm, control = self.scope["arms"]["two-path"], self.scope["arms"]["control"]
        delta = round(arm["f1_at_zero"]-control["f1_at_zero"], 6)
        gate = arm["exact_boards_at_zero"]["exact"] > 0
        expected = ("two-path-passes" if delta >= .05 and gate else
                    "two-path-moves-without-clearing" if delta > 0 else "two-path-fails")
        self.assertEqual(self.scope["reading"], expected)
        channels = self.scope["channels"]
        gain = channels["summed"]["auc"]-channels["lattice_only"]["auc"]
        gap = channels["lattice_only"]["auc"]-channels["free_only"]["auc"]
        verdict = ("one-channel-noise-dominates" if gap >= .01 and gain < .005 else
                   "complementary" if gain >= .005 else "no-complementary-signal")
        self.assertEqual(channels["verdict"], verdict)

    def test_the_strict_metric_and_the_incremental_reading_are_reported(self):
        for name, arm in self.scope["arms"].items():
            gate = arm["exact_boards_at_zero"]
            self.assertEqual(gate["boards"], 28, name)
            self.assertIn("boundary_exact", gate, name)
        self.assertIn("incremental_over_envelope", self.scope["contrasts"])
        self.assertEqual(self.scope["gate_moved"],
                         self.scope["arms"]["two-path"]["exact_boards_at_zero"]["exact"] > 0)

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
