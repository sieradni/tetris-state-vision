"""The declared registration lever: a bounded per-cell field moving the detail tap's sample."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

import torch
import tetris_vision.model as model_module
from tetris_vision.losses import require_occupancy_deform
from tetris_vision.model import DEFORM_LIMIT, ModelConfig, TetrisStateNet, deform_offsets

from record_registration_lever import mechanism_of
from record_stream_run import OWNED_RUN_PATTERN


def pilot(**overrides):
    return ModelConfig(**{**ModelConfig.pilot().__dict__, **overrides})


def inputs(seed=20261026, size=32):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, size, size, generator=generator), torch.ones(1, 1, size, size, dtype=torch.bool)


def occupancy(config, seed=20261026):
    torch.manual_seed(seed)
    model = TetrisStateNet(config).eval()
    pixels, valid = inputs()
    with torch.no_grad():
        return model, model(pixels, valid)["final"]["occupancy"]


class FieldTests(unittest.TestCase):
    def test_the_default_is_the_published_configuration(self):
        self.assertEqual(ModelConfig().occupancy_deform, 0)
        self.assertEqual(ModelConfig.pilot().occupancy_deform, 0)

    def test_an_undeclared_value_is_refused(self):
        for value in (2, -1, True, 1.0):
            with self.assertRaises(ValueError):
                ModelConfig(occupancy_detail=1, occupancy_deform=value)

    def test_the_lever_moves_the_detail_taps_sample_and_needs_it(self):
        with self.assertRaises(ValueError):
            ModelConfig(occupancy_detail=0, occupancy_deform=1)
        model, _ = occupancy(pilot(occupancy_detail=1))
        self.assertIsNone(model.occupancy_deform)

    def test_the_field_is_a_zero_initialized_convolution_of_the_tap_source(self):
        model, _ = occupancy(pilot(occupancy_detail=1, occupancy_deform=1))
        field = model.occupancy_deform
        self.assertIsNotNone(field)
        self.assertEqual((field.in_channels, field.out_channels), (pilot().widths[0], 2))
        self.assertEqual((field.kernel_size, field.padding), ((3, 3), (1, 1)))
        self.assertFalse(bool(field.weight.any()))
        self.assertFalse(bool(field.bias.any()))

    def test_the_declared_limit_is_one_cell_and_the_field_cannot_exceed_it(self):
        self.assertEqual(DEFORM_LIMIT, 1.0)
        samples = torch.tensor([[[-40.0, 40.0]]])
        bounded = deform_offsets(samples)
        self.assertTrue(torch.all(bounded.abs() <= DEFORM_LIMIT))
        self.assertTrue(all(abs(value) <= DEFORM_LIMIT for value in bounded.flatten().tolist()))
        self.assertTrue(torch.equal(deform_offsets(torch.zeros(1, 1, 2)), torch.zeros(1, 1, 2)))
        with self.assertRaises(ValueError):
            deform_offsets(samples, limit=2.0)

    def test_both_arms_are_exactly_the_published_function_at_step_zero(self):
        _, plain = occupancy(pilot())
        _, control = occupancy(pilot(occupancy_detail=1))
        _, lever = occupancy(pilot(occupancy_detail=1, occupancy_deform=1))
        self.assertTrue(torch.equal(plain, control))
        self.assertTrue(torch.equal(plain, lever))

    def test_the_arms_share_every_tensor_but_the_field(self):
        torch.manual_seed(20261026)
        control = TetrisStateNet(pilot(occupancy_detail=1))
        torch.manual_seed(20261026)
        lever = TetrisStateNet(pilot(occupancy_detail=1, occupancy_deform=1))
        control_state, lever_state = control.state_dict(), lever.state_dict()
        added = set(lever_state)-set(control_state)
        self.assertTrue(added and all(name.startswith("occupancy_deform.") for name in added))
        self.assertFalse(set(control_state)-set(lever_state))
        for name, value in control_state.items():
            self.assertTrue(torch.equal(value, lever_state[name]), name)

    def test_a_trained_field_changes_the_decision_only_through_the_tap_projection(self):
        """Zero the projection and the field is invisible; train it and the field is the difference."""
        torch.manual_seed(5)
        lever = TetrisStateNet(pilot(occupancy_detail=1, occupancy_deform=1)).eval()
        torch.manual_seed(5)
        control = TetrisStateNet(pilot(occupancy_detail=1)).eval()
        with torch.no_grad():
            control.occupancy_detail_tap.weight.copy_(lever.occupancy_detail_tap.weight)
            control.occupancy_detail_tap.bias.copy_(lever.occupancy_detail_tap.bias)
            lever.occupancy_deform.weight.normal_(0, 1.0)
        pixels, valid = inputs()
        with torch.no_grad():
            inert = lever(pixels, valid)["final"]["occupancy"]
            reference = control(pixels, valid)["final"]["occupancy"]
            control.occupancy_detail_tap.weight.fill_(.5)
            control.occupancy_detail_tap.bias.fill_(.1)
            lever.occupancy_detail_tap.weight.fill_(.5)
            lever.occupancy_detail_tap.bias.fill_(.1)
            moved = lever(pixels, valid)["final"]["occupancy"]
            lever.occupancy_deform.weight.zero_()
            zeroed = lever(pixels, valid)["final"]["occupancy"]
            reference_active = control(pixels, valid)["final"]["occupancy"]
        self.assertTrue(torch.equal(inert, reference))
        self.assertFalse(torch.equal(moved, reference_active))
        self.assertTrue(torch.equal(zeroed, reference_active))

    def test_a_hidden_field_cannot_change_a_zero_projection(self):
        """Paired-raster equivalence: identical pixels, different hidden field, same output."""
        torch.manual_seed(11)
        first = TetrisStateNet(pilot(occupancy_detail=1, occupancy_deform=1)).eval()
        torch.manual_seed(11)
        second = TetrisStateNet(pilot(occupancy_detail=1, occupancy_deform=1)).eval()
        with torch.no_grad():
            first.occupancy_deform.weight.normal_(0, .9)
            second.occupancy_deform.weight.normal_(0, .1)
        pixels, valid = inputs()
        with torch.no_grad():
            a = first(pixels, valid)["final"]["occupancy"]
            b = second(pixels, valid)["final"]["occupancy"]
        self.assertFalse(torch.equal(first.occupancy_deform.weight,
                                     second.occupancy_deform.weight))
        self.assertTrue(torch.equal(a, b))

    def test_the_admission_join_accepts_only_a_matching_declaration(self):
        summary = {"arguments": {"occupancy_deform": 1}, "recovery_contract": {"occupancy_deform": 1}}
        self.assertIsNone(require_occupancy_deform(1, summary))
        require_occupancy_deform(0, {"arguments": {}, "recovery_contract": {}})
        for deform, record in ((1, {"arguments": {"occupancy_deform": 0},
                                    "recovery_contract": {"occupancy_deform": 1}}),
                               (1, {"arguments": {"occupancy_deform": 1}, "recovery_contract": {}}),
                               (2, summary), (True, summary)):
            with self.assertRaises(ValueError):
                require_occupancy_deform(deform, record)

    def test_the_owned_run_family_is_declared(self):
        for name in ("m3-occdeform-lever-r1", "m3-occdeform-control-r1",
                     "m3-occdeform-lever-r1-sweep-r1", "m3-occdeform-control-r1-sweep-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        self.assertFalse(OWNED_RUN_PATTERN.fullmatch("m3-occdeform-"))


class MechanismTests(unittest.TestCase):
    @staticmethod
    def measured(best_lever, best_control, zero_lever, zero_control):
        return {"lever": {"best": {"precision": best_lever}, "precision": zero_lever},
                "control": {"best": {"precision": best_control}, "precision": zero_control}}

    def test_the_mechanism_reads_the_arms_own_boundary_and_keeps_the_fixed_cost_beside_it(self):
        interval = {"median": -.01, "standard_error": .004, "low": -.02, "high": 0.0,
                    "valid_draws": 400}
        block = mechanism_of(self.measured(.45, .42, .30, .31), {"precision_delta": interval})
        self.assertAlmostEqual(block["best_precision_delta"], .03)
        self.assertFalse(block["best_precision_improved"])
        self.assertEqual(block["read"], "precision at the arm's own best boundary")
        self.assertAlmostEqual(block["zero_boundary_precision_delta"]["delta"], -.01)
        self.assertAlmostEqual(block["zero_boundary_precision_delta"]["minimum_detectable_effect"],
                               round(2.802*.004, 6))
        self.assertIs(block["zero_boundary_precision_delta"]["interval"], interval)

    def test_a_mislabelled_precision_interval_is_refused(self):
        with self.assertRaises(ValueError):
            mechanism_of(self.measured(.45, .42, .30, .31),
                         {"precision_delta": {"median": -.5, "standard_error": .004}})


class DeclarationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        text = (Path(__file__).resolve().parents[1]/"docs/M3_OCCUPANCY_REGISTRATION.md").read_text(
            encoding="utf-8")
        cls.text = " ".join(text.split())

    def test_the_declaration_names_its_bound_and_its_population(self):
        for phrase in ("1,792 MiB", "64 further steps", "m3-occdeform", "28 nonempty boards",
                       "667 occupied", "twelve ancestry", "minimum detectable effect"):
            self.assertIn(phrase, self.text)

    def test_the_outcome_figures_are_the_published_scope(self):
        """The measured section is bound to the evidence it quotes, not typed beside it."""
        scope = json.loads((Path(__file__).resolve().parents[1]/
                            "docs/evidence/m3-occupancy-deformation-r1/scope.json").read_text(
            encoding="utf-8"))
        lever, control = scope["arms"]["lever"], scope["arms"]["control"]
        self.assertEqual(scope["verdict"]["verdict"], "not-cleared")
        self.assertTrue(scope["verdict"]["conditions"]["attribution"])
        for arm, other in ((lever, control), (control, lever)):
            self.assertNotEqual(arm["auc"], other["auc"])
        self.assertEqual(scope["population"]["nonempty_boards"], 28)
        self.assertEqual(scope["population"]["occupied_cells"], 667)
        for value in (lever["auc"], lever["per_cell_f1"], lever["best"]["f1"],
                      control["auc"], control["per_cell_f1"], control["best"]["f1"],
                      lever["components"]["out_of_fold_f1"],
                      control["components"]["out_of_fold_f1"],
                      lever["components"]["zero_cost"], control["components"]["zero_cost"],
                      scope["attribution"]["best_f1_delta"], scope["attribution"]["auc_delta"]):
            self.assertIn(f"{round(value, 6):,.6f}", self.text)
        for value in (lever["study_peak_commit_bytes"], control["study_peak_commit_bytes"],
                      lever["sweep_peak_commit_bytes"], control["sweep_peak_commit_bytes"]):
            self.assertIn(f"{value:,}", self.text)
        self.assertIn(f"{lever['study_peak_commit_bytes']-control['study_peak_commit_bytes']:,}",
                      self.text)
        at_best = sum(1 for value in lever["components"]["out_of_fold_thresholds"]
                      if abs(value-lever["best"]["threshold"]) < 1e-9)
        self.assertEqual(at_best, 10)
        self.assertIn(("ten" if at_best == 10 else str(at_best))+" of twelve groups", self.text)

    def test_the_recorder_binds_the_declared_envelope(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
        import record_registration_lever as recorder
        self.assertEqual(recorder.CEILING_BYTES, 1792*1024**2)
        self.assertEqual((recorder.STEPS, recorder.FORWARDS), (64, 54))
        self.assertEqual(recorder.FAMILY, "m3-occdeform")
        self.assertEqual(recorder.ARMS, {"lever": 1, "control": 0})


if __name__ == "__main__":
    unittest.main()
