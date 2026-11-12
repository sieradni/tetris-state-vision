"""The declared registration lever: a bounded per-cell field moving the detail tap's sample."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

import torch
import tetris_vision.model as model_module
from tetris_vision.losses import (OCCUPANCY_PLACED_WEIGHTS, PLACED_TARGET_WEIGHT, joint_loss,
                                 placed_displacement_term, placed_target_displacement,
                                 require_occupancy_deform, require_occupancy_placed, targets_for)
from tetris_vision.preprocess import prepare_image
from tetris_vision.render import render, sample_scene
from tetris_vision.model import (DEFORM_DEMEAN_REACH, DEFORM_LIMIT, DEFORM_REACH, DEFORM_WIDE_LIMIT,
                                 ModelConfig, TetrisStateNet, deform_offsets)

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
        for value in (4, -1, True, 1.0):
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

    def test_every_arm_is_exactly_the_published_function_at_step_zero(self):
        _, plain = occupancy(pilot())
        _, control = occupancy(pilot(occupancy_detail=1))
        _, lever = occupancy(pilot(occupancy_detail=1, occupancy_deform=1))
        _, wide = occupancy(pilot(occupancy_detail=1, occupancy_deform=2))
        _, demeaned = occupancy(pilot(occupancy_detail=1, occupancy_deform=3))
        self.assertTrue(torch.equal(plain, control))
        self.assertTrue(torch.equal(plain, lever))
        self.assertTrue(torch.equal(plain, wide))
        self.assertTrue(torch.equal(plain, demeaned))

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


class WideFieldTests(unittest.TestCase):
    """The coarse-to-fine arm: the same fine level plus a wider-reach coarse one."""

    CELLS = (ModelConfig.pilot().visible_rows+ModelConfig.pilot().spawn_rows)*ModelConfig.pilot().columns

    @staticmethod
    def final(config, seed=20261026):
        torch.manual_seed(seed)
        model = TetrisStateNet(config).eval()
        pixels, valid = inputs()
        with torch.no_grad():
            return model, model(pixels, valid)["final"]

    def test_the_widened_arm_adds_one_coarse_level_over_the_fine_one(self):
        torch.manual_seed(3)
        wide = TetrisStateNet(pilot(occupancy_detail=1, occupancy_deform=2))
        self.assertIsNotNone(wide.occupancy_deform)
        self.assertIsNotNone(wide.occupancy_deform_coarse)
        self.assertEqual((wide.occupancy_deform_coarse.in_channels,
                          wide.occupancy_deform_coarse.out_channels), (pilot().widths[0], 2))
        self.assertEqual((wide.occupancy_deform_coarse.kernel_size,
                          wide.occupancy_deform_coarse.padding), ((3, 3), (1, 1)))
        self.assertFalse(bool(wide.occupancy_deform_coarse.weight.any()))
        self.assertFalse(bool(wide.occupancy_deform_coarse.bias.any()))
        torch.manual_seed(3)
        fine = TetrisStateNet(pilot(occupancy_detail=1, occupancy_deform=1))
        self.assertIsNone(fine.occupancy_deform_coarse)
        for name, value in fine.state_dict().items():
            self.assertTrue(torch.equal(value, wide.state_dict()[name]), name)

    def test_each_level_keeps_its_declared_bound(self):
        self.assertEqual((DEFORM_LIMIT, DEFORM_WIDE_LIMIT, DEFORM_REACH), (1.0, 3.0, 4.0))
        sample = torch.tensor([[[40.0, -40.0]]])
        self.assertTrue(torch.allclose(deform_offsets(sample, DEFORM_WIDE_LIMIT),
                                       torch.tensor([[[3.0, -3.0]]])))
        self.assertTrue(torch.allclose(deform_offsets(sample), torch.tensor([[[1.0, -1.0]]])))
        with self.assertRaises(ValueError):
            deform_offsets(sample, 2.0)

    def test_the_widened_field_reaches_past_one_cell_and_stays_inside_its_reach(self):
        model, _ = self.final(pilot(occupancy_detail=1, occupancy_deform=2))
        with torch.no_grad():
            model.occupancy_deform.weight.zero_()
            model.occupancy_deform.bias.zero_()
            model.occupancy_deform_coarse.weight.fill_(1e3)
        pixels, valid = inputs()
        with torch.no_grad():
            offsets = model(pixels, valid)["final"]["occupancy_deform_offsets"]
        self.assertEqual(tuple(offsets.shape), (1, self.CELLS, 2))
        reach = float(offsets.abs().max())
        self.assertGreater(reach, 2.9)
        self.assertLessEqual(reach, DEFORM_REACH)
        self.assertGreater(reach, DEFORM_LIMIT)

    def test_the_one_cell_arm_cannot_reach_past_its_own_bound(self):
        model, _ = self.final(pilot(occupancy_detail=1, occupancy_deform=1))
        with torch.no_grad():
            model.occupancy_deform.weight.fill_(1e3)
        pixels, valid = inputs()
        with torch.no_grad():
            offsets = model(pixels, valid)["final"]["occupancy_deform_offsets"]
        self.assertLessEqual(float(offsets.abs().max()), DEFORM_LIMIT)
        self.assertGreater(float(offsets.abs().max()), .9)

    def test_the_applied_offsets_are_exposed_and_exactly_zero_at_step_zero(self):
        for deform in (0, 1, 2):
            _, final = self.final(pilot(occupancy_detail=1, occupancy_deform=deform))
            if deform == 0:
                self.assertNotIn("occupancy_deform_offsets", final)
            else:
                offsets = final["occupancy_deform_offsets"]
                self.assertEqual(tuple(offsets.shape), (1, self.CELLS, 2))
                self.assertFalse(bool(offsets.any()))

    def test_the_widened_arm_shares_every_published_tensor(self):
        torch.manual_seed(9)
        control = TetrisStateNet(pilot(occupancy_detail=1))
        torch.manual_seed(9)
        wide = TetrisStateNet(pilot(occupancy_detail=1, occupancy_deform=2))
        added = set(wide.state_dict())-set(control.state_dict())
        self.assertEqual(len(added), 4)
        self.assertTrue(all(name.startswith(("occupancy_deform.", "occupancy_deform_coarse."))
                            for name in added))
        for name, value in control.state_dict().items():
            self.assertTrue(torch.equal(value, wide.state_dict()[name]), name)

    def test_paired_raster_equivalence_holds_for_the_widened_arm(self):
        torch.manual_seed(13)
        first = TetrisStateNet(pilot(occupancy_detail=1, occupancy_deform=2)).eval()
        torch.manual_seed(13)
        second = TetrisStateNet(pilot(occupancy_detail=1, occupancy_deform=2)).eval()
        with torch.no_grad():
            first.occupancy_deform_coarse.weight.normal_(0, .9)
            second.occupancy_deform_coarse.weight.normal_(0, .05)
        pixels, valid = inputs()
        with torch.no_grad():
            a = first(pixels, valid)["final"]["occupancy"]
            b = second(pixels, valid)["final"]["occupancy"]
        self.assertFalse(torch.equal(first.occupancy_deform_coarse.weight,
                                    second.occupancy_deform_coarse.weight))
        self.assertTrue(torch.equal(a, b))

    def test_the_widened_field_changes_the_decision_only_through_the_projection(self):
        torch.manual_seed(21)
        wide = TetrisStateNet(pilot(occupancy_detail=1, occupancy_deform=2)).eval()
        torch.manual_seed(21)
        control = TetrisStateNet(pilot(occupancy_detail=1)).eval()
        with torch.no_grad():
            control.occupancy_detail_tap.weight.copy_(wide.occupancy_detail_tap.weight)
            control.occupancy_detail_tap.bias.copy_(wide.occupancy_detail_tap.bias)
            wide.occupancy_deform_coarse.weight.normal_(0, 1.5)
        pixels, valid = inputs()
        with torch.no_grad():
            inert = wide(pixels, valid)["final"]["occupancy"]
            reference = control(pixels, valid)["final"]["occupancy"]
            for model in (wide, control):
                model.occupancy_detail_tap.weight.fill_(.5)
                model.occupancy_detail_tap.bias.fill_(.1)
            moved = wide(pixels, valid)["final"]["occupancy"]
            reference_active = control(pixels, valid)["final"]["occupancy"]
            wide.occupancy_deform_coarse.weight.zero_()
            zeroed = wide(pixels, valid)["final"]["occupancy"]
        self.assertTrue(torch.equal(inert, reference))
        self.assertFalse(torch.equal(moved, reference_active))
        self.assertTrue(torch.equal(zeroed, reference_active))

    def test_the_admission_binds_the_widened_declaration(self):
        summary = {"arguments": {"occupancy_deform": 2},
                   "recovery_contract": {"occupancy_deform": 2}}
        self.assertIsNone(require_occupancy_deform(2, summary))
        with self.assertRaises(ValueError):
            require_occupancy_deform(1, summary)

    def test_the_owned_widened_family_is_declared(self):
        for name in ("m3-occdeform2-wide-r1", "m3-occdeform2-control-r1",
                     "m3-occdeform2-wide-r1-sweep-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)


class DemeanedFieldTests(unittest.TestCase):
    """The de-meaned arm: identical parameters to the widened one, constant mode barred."""

    @staticmethod
    def model(config, seed=20261026):
        torch.manual_seed(seed)
        return TetrisStateNet(config).eval()

    @staticmethod
    def offsets(model):
        pixels, valid = inputs()
        with torch.no_grad():
            return model(pixels, valid)["final"]["occupancy_deform_offsets"]

    def test_the_demeaned_arm_adds_no_parameters_and_shares_every_tensor_with_the_widened_arm(self):
        torch.manual_seed(17)
        wide = TetrisStateNet(pilot(occupancy_detail=1, occupancy_deform=2))
        torch.manual_seed(17)
        demeaned = TetrisStateNet(pilot(occupancy_detail=1, occupancy_deform=3))
        self.assertIsNotNone(demeaned.occupancy_deform_coarse)
        self.assertEqual(set(wide.state_dict()), set(demeaned.state_dict()))
        for name, value in wide.state_dict().items():
            self.assertTrue(torch.equal(value, demeaned.state_dict()[name]), name)
        self.assertFalse(bool(demeaned.occupancy_deform_coarse.weight.any()))

    def test_the_board_mean_is_zero_by_construction_while_the_free_arm_carries_one(self):
        means = {}
        for deform in (2, 3):
            model = self.model(pilot(occupancy_detail=1, occupancy_deform=deform))
            with torch.no_grad():
                model.occupancy_deform.weight.zero_()
                model.occupancy_deform.bias.fill_(3.0)
                model.occupancy_deform_coarse.weight.zero_()
                model.occupancy_deform_coarse.bias.fill_(3.0)
            means[deform] = float(self.offsets(model).mean(dim=1).abs().max())
        self.assertGreater(means[2], 2.0)
        self.assertLessEqual(means[3], 1e-6)

    def test_the_demeaned_field_is_bounded_and_its_board_mean_is_zero(self):
        model = self.model(pilot(occupancy_detail=1, occupancy_deform=3))
        with torch.no_grad():
            model.occupancy_deform.weight.normal_(0, 5.0)
            model.occupancy_deform_coarse.weight.normal_(0, 5.0)
        offsets = self.offsets(model)
        self.assertLessEqual(float(offsets.abs().max()), DEFORM_DEMEAN_REACH)
        self.assertGreater(float(offsets.abs().max()), .5)
        self.assertLessEqual(float(offsets.mean(dim=1).abs().max()), 1e-6)
        self.assertEqual(DEFORM_DEMEAN_REACH, 2*DEFORM_REACH)

    def test_paired_raster_equivalence_holds_for_the_demeaned_arm(self):
        first = self.model(pilot(occupancy_detail=1, occupancy_deform=3), seed=23)
        second = self.model(pilot(occupancy_detail=1, occupancy_deform=3), seed=23)
        with torch.no_grad():
            first.occupancy_deform_coarse.weight.normal_(0, .9)
            second.occupancy_deform_coarse.weight.normal_(0, .05)
        pixels, valid = inputs()
        with torch.no_grad():
            a = first(pixels, valid)["final"]["occupancy"]
            b = second(pixels, valid)["final"]["occupancy"]
        self.assertFalse(torch.equal(first.occupancy_deform_coarse.weight,
                                    second.occupancy_deform_coarse.weight))
        self.assertTrue(torch.equal(a, b))

    def test_the_demeaned_field_changes_the_decision_only_through_the_projection(self):
        demeaning = self.model(pilot(occupancy_detail=1, occupancy_deform=3), seed=29)
        control = self.model(pilot(occupancy_detail=1), seed=29)
        with torch.no_grad():
            control.occupancy_detail_tap.weight.copy_(demeaning.occupancy_detail_tap.weight)
            control.occupancy_detail_tap.bias.copy_(demeaning.occupancy_detail_tap.bias)
            demeaning.occupancy_deform_coarse.weight.normal_(0, 2.0)
        pixels, valid = inputs()
        with torch.no_grad():
            inert = demeaning(pixels, valid)["final"]["occupancy"]
            reference = control(pixels, valid)["final"]["occupancy"]
            for model in (demeaning, control):
                model.occupancy_detail_tap.weight.fill_(.5)
                model.occupancy_detail_tap.bias.fill_(.1)
            moved = demeaning(pixels, valid)["final"]["occupancy"]
            reference_active = control(pixels, valid)["final"]["occupancy"]
            demeaning.occupancy_deform_coarse.weight.zero_()
            zeroed = demeaning(pixels, valid)["final"]["occupancy"]
        self.assertTrue(torch.equal(inert, reference))
        self.assertFalse(torch.equal(moved, reference_active))
        self.assertTrue(torch.equal(zeroed, reference_active))

    def test_the_admission_binds_the_demeaned_declaration(self):
        summary = {"arguments": {"occupancy_deform": 3},
                   "recovery_contract": {"occupancy_deform": 3}}
        self.assertIsNone(require_occupancy_deform(3, summary))
        with self.assertRaises(ValueError):
            require_occupancy_deform(2, summary)

    def test_the_owned_demeaned_family_is_declared(self):
        for name in ("m3-occdeform3-demean-r1", "m3-occdeform3-control-r1",
                     "m3-occdeform3-demean-r1-sweep-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)


class WidenedRecordTests(unittest.TestCase):
    """The three-arm record: what it published and what the document says about it."""

    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.scope = json.loads((root/"docs/evidence/m3-occupancy-deformation-r2/scope.json").read_text(
            encoding="utf-8"))
        cls.prior = json.loads((root/"docs/evidence/m3-occupancy-deformation-r1/scope.json").read_text(
            encoding="utf-8"))
        cls.text = " ".join((root/"docs/M3_OCCUPANCY_REGISTRATION.md").read_text(
            encoding="utf-8").split())

    def test_the_three_arms_and_the_primary_verdict_are_mechanical(self):
        self.assertEqual(sorted(self.scope["arms"]), ["control", "lever", "wide"])
        self.assertEqual({name: arm["occupancy_deform"] for name, arm in self.scope["arms"].items()},
                         {"control": 0, "lever": 1, "wide": 2})
        self.assertEqual({name: arm["reach_cells"] for name, arm in self.scope["arms"].items()},
                         {"control": 0.0, "lever": 1.0, "wide": 4.0})
        self.assertEqual(self.scope["verdict"]["verdict"], "not-cleared")
        self.assertTrue(self.scope["verdict"]["conditions"]["attribution"])
        self.assertEqual(self.scope["population"]["nonempty_boards"], 28)
        self.assertEqual(self.scope["population"]["occupied_cells"], 667)

    def test_the_one_cell_arm_reproduces_the_previous_pass_exactly(self):
        for key in ("auc", "per_cell_f1", "precision"):
            self.assertEqual(self.scope["arms"]["lever"][key], self.prior["arms"]["lever"][key])
        self.assertEqual(self.scope["arms"]["lever"]["best"]["f1"],
                         self.prior["arms"]["lever"]["best"]["f1"])

    def test_the_measured_field_is_mostly_one_constant_vector_and_the_reading_says_so(self):
        wide, lever = self.scope["reach"]["field"]["wide"], self.scope["reach"]["field"]["lever"]
        self.assertGreater(wide["constant_share"], .95)
        self.assertGreater(lever["constant_share"], .85)
        self.assertGreater(wide["magnitude_cells"]["mean"], 3.0)
        self.assertLessEqual(lever["magnitude_cells"]["mean"], 1.0)
        self.assertGreaterEqual(wide["at_bound_fraction"], self.scope["reach"]["saturation_floor"])
        self.assertEqual(self.scope["reach"]["reading"], "reach-exceeded")
        self.assertEqual(self.scope["reach"]["oracle_statistic"]["mean_placement_offset_cells"], 2.57)
        self.assertEqual(wide["arm"]["bound_cells"], 4.0)
        self.assertEqual(lever["arm"]["bound_cells"], 1.0)

    def test_every_added_tensor_of_every_arm_is_on_the_declared_allowlist(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
        from train_m2_stream import INITIAL_SOURCE_ADDED_PREFIXES
        torch.manual_seed(4)
        control = TetrisStateNet(pilot(occupancy_detail=1))
        baseline = set(control.state_dict())
        for deform in (1, 2, 3):
            torch.manual_seed(4)
            arm = TetrisStateNet(pilot(occupancy_detail=1, occupancy_deform=deform))
            added = set(arm.state_dict())-baseline
            self.assertTrue(added)
            for name in added:
                self.assertTrue(name.startswith(INITIAL_SOURCE_ADDED_PREFIXES), name)

    def test_the_document_quotes_the_published_scope(self):
        wide, control = self.scope["arms"]["wide"], self.scope["arms"]["control"]
        lever = self.scope["arms"]["lever"]
        field = self.scope["reach"]["field"]["wide"]
        for value in (wide["auc"], control["auc"], wide["per_cell_f1"], control["per_cell_f1"],
                      wide["best"]["f1"], lever["best"]["f1"],
                      self.scope["contrasts"]["wide_control"]["best_f1_delta"],
                      self.scope["contrasts"]["wide_control"]["auc_delta"],
                      self.scope["contrasts"]["wide_lever"]["per_cell_f1_delta"]):
            self.assertIn(f"{round(value, 6):,.6f}", self.text)
        self.assertIn(f"{field['constant_share']:.3f}", self.text)
        self.assertIn(f"{field['magnitude_cells']['mean']:.3f}", self.text)
        self.assertIn(f"{100*field['at_bound_fraction']:.1f}%", self.text)
        self.assertIn(f"{100*self.scope['reach']['field']['lever']['at_bound_fraction']:.1f}%", self.text)
        for arm in self.scope["arms"].values():
            self.assertIn(f"{arm['study_peak_commit_bytes']:,}", self.text)


class PlacedTargetTests(unittest.TestCase):
    """The placed-target lever: supervise where the tap samples, not what it finds there."""

    @staticmethod
    def wide(seed=20261026):
        torch.manual_seed(seed)
        return TetrisStateNet(pilot(occupancy_detail=1, occupancy_deform=2)).eval()

    @staticmethod
    def target(lattice, rows=20, columns=10, known=None):
        return {"occupied": torch.zeros(1, rows, columns),
                "known": (torch.ones(1, rows, columns, dtype=torch.bool) if known is None else known),
                "lattice": lattice, "registered": torch.ones(1, dtype=torch.bool)}

    def stage(self, model):
        pixels, valid = inputs()
        with torch.no_grad():
            return model(pixels, valid)["final"]

    def test_a_field_that_already_sits_on_the_target_costs_nothing(self):
        stage = self.stage(self.wide())
        lattice = stage["lattice"]
        shifted = lattice.clone()
        shifted[:, 0] = lattice[:, 0]+2*lattice[:, 1]
        wanted = placed_target_displacement(lattice, shifted, 20, 10, DEFORM_REACH)
        self.assertAlmostEqual(float(wanted[..., 0].mean()), 2.0, places=4)
        self.assertAlmostEqual(float(wanted[..., 1].abs().max()), 0.0, places=5)
        # The displacement is two cells along the column axis and nothing along the row axis, so a
        # field that already sits there pays exactly nothing.
        placed = torch.tensor([[2.0, 0.0]]).expand(1, 200, 2)
        output = {"lattice": lattice, "occupancy_deform_offsets": placed}
        self.assertAlmostEqual(float(placed_displacement_term(output, self.target(shifted))), 0.0,
                               places=6)
        behind = {"lattice": lattice, "occupancy_deform_offsets": torch.zeros(1, 200, 2)}
        self.assertGreater(float(placed_displacement_term(behind, self.target(shifted))), .5)

    def test_an_unreachable_target_is_clipped_to_the_arms_own_reach(self):
        stage = self.stage(self.wide())
        lattice = stage["lattice"]
        far = lattice.clone()
        far[:, 0] = lattice[:, 0]+6*lattice[:, 1]
        wanted = placed_target_displacement(lattice, far, 20, 10, DEFORM_REACH)
        self.assertAlmostEqual(float(wanted[..., 0].max()), DEFORM_REACH, places=5)
        self.assertLessEqual(float(wanted.abs().max()), DEFORM_REACH)

    def test_the_target_depends_on_where_the_cells_are_not_on_what_they_contain(self):
        stage = self.stage(self.wide())
        lattice = stage["lattice"]
        shifted = lattice.clone()
        shifted[:, 0] = lattice[:, 0]+1.5*lattice[:, 2]
        output = {"lattice": lattice, "occupancy_deform_offsets": torch.zeros(1, 200, 2)}
        target = self.target(shifted)
        before = float(placed_displacement_term(output, target))
        target["occupied"] = 1-target["occupied"]
        self.assertEqual(float(placed_displacement_term(output, target)), before)

    def test_unknown_cells_are_not_scored(self):
        stage = self.stage(self.wide())
        lattice = stage["lattice"]
        shifted = lattice.clone()
        shifted[:, 0] = lattice[:, 0]+2*lattice[:, 1]
        output = {"lattice": lattice, "occupancy_deform_offsets": torch.zeros(1, 200, 2)}
        known = torch.zeros(1, 20, 10, dtype=torch.bool)
        self.assertEqual(float(placed_displacement_term(output, self.target(shifted, known=known))),
                         0.0)
        known[:, :, :5] = True
        # A zero field against a two-cell column displacement: 1.5 on that axis, 0 on the other.
        self.assertAlmostEqual(float(placed_displacement_term(
            output, self.target(shifted, known=known))), .75, places=5)

    def test_a_field_that_cannot_cover_the_board_is_refused(self):
        stage = self.stage(self.wide())
        with self.assertRaises(ValueError):
            placed_displacement_term({"lattice": stage["lattice"],
                                      "occupancy_deform_offsets": torch.zeros(1, 150, 2)},
                                     self.target(stage["lattice"]))
        with self.assertRaises(ValueError):
            placed_displacement_term({"lattice": stage["lattice"],
                                      "occupancy_deform_offsets": None},
                                     self.target(stage["lattice"]))

    def test_the_placed_target_is_not_supervised_and_the_field_is(self):
        model = self.wide()
        result = render(sample_scene(4))
        prepared = prepare_image(result.image, max_side=383)
        target = targets_for(result.state, prepared)
        model.train()
        output = model(prepared.pixels[None], prepared.valid[None])
        term = placed_displacement_term(output["final"], target)
        wanted = placed_target_displacement(output["final"]["lattice"], target["lattice"], 20, 10,
                                            DEFORM_REACH)
        # The box and the occupancy labels are inputs, never targets: no gradient reaches them.
        self.assertFalse(wanted.requires_grad)
        self.assertFalse(target["lattice"].requires_grad)
        for parameter in (model.occupancy_deform.weight, model.occupancy_deform_coarse.weight):
            gradient = torch.autograd.grad(term, parameter, retain_graph=True)[0]
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_the_term_is_absent_at_zero_and_priced_at_the_declared_weight(self):
        model = self.wide()
        result = render(sample_scene(4))
        prepared = prepare_image(result.image, max_side=383)
        target = targets_for(result.state, prepared)
        with torch.no_grad():
            output = model(prepared.pixels[None], prepared.valid[None])
        off, terms_off = joint_loss(output, target, occupancy_placed=0)
        on, terms_on = joint_loss(output, target, occupancy_placed=1)
        self.assertNotIn("placed", terms_off)
        self.assertGreater(float(terms_on["placed"]), 0.0)
        self.assertAlmostEqual(float(on-off), PLACED_TARGET_WEIGHT*float(terms_on["placed"]),
                               places=6)
        self.assertEqual(PLACED_TARGET_WEIGHT, .25)
        self.assertEqual(OCCUPANCY_PLACED_WEIGHTS, (0, 1))
        from train_m2_stream import OCCUPANCY_PLACED_WEIGHTS as MIRROR
        self.assertEqual(MIRROR, OCCUPANCY_PLACED_WEIGHTS)
        require_occupancy_placed(0, {"arguments": {}, "recovery_contract": {}})
        with self.assertRaises(ValueError):
            require_occupancy_placed(1, {"arguments": {}, "recovery_contract": {}})

    def test_the_placed_arm_adds_no_tensors_and_its_step_zero_is_the_widened_arm(self):
        torch.manual_seed(5)
        plain = TetrisStateNet(pilot(occupancy_detail=1, occupancy_deform=2))
        torch.manual_seed(5)
        placed = TetrisStateNet(pilot(occupancy_detail=1, occupancy_deform=2))
        self.assertEqual(set(plain.state_dict()), set(placed.state_dict()))
        for name, value in plain.state_dict().items():
            self.assertTrue(torch.equal(value, placed.state_dict()[name]), name)
        pixels, valid = inputs()
        with torch.no_grad():
            first, second = plain(pixels, valid)["final"], placed(pixels, valid)["final"]
        self.assertTrue(torch.equal(first["occupancy"], second["occupancy"]))
        self.assertTrue(torch.equal(first["occupancy_deform_offsets"],
                                    second["occupancy_deform_offsets"]))
        self.assertEqual(float(first["occupancy_deform_offsets"].abs().max()), 0.0)

    def test_the_offsets_written_are_the_offsets_applied(self):
        model = self.wide()
        with torch.no_grad():
            model.occupancy_deform.weight.normal_(0, 1.0)
            model.occupancy_deform_coarse.weight.normal_(0, 1.0)
        pixels, valid = inputs()
        with torch.no_grad():
            stage = model(pixels, valid)["final"]
        applied = stage["occupancy_applied_offsets"]
        written = stage["occupancy_deform_offsets"]
        self.assertEqual(applied.shape, written.shape)
        # The declared fine arm samples at the cell centre, so the applied offset is the field.
        self.assertTrue(torch.equal(applied, written))
        self.assertLessEqual(float(written.abs().max()), DEFORM_REACH)

    def test_the_placed_arm_reads_pixels_and_placement_not_hidden_labels(self):
        model = self.wide()
        with torch.no_grad():
            model.occupancy_deform.weight.normal_(0, 1.0)
        pixels, valid = inputs()
        with torch.no_grad():
            left = model(pixels, valid)["final"]
            right = model(pixels, valid)["final"]
        self.assertTrue(torch.equal(left["occupancy"], right["occupancy"]))
        target = self.target(left["lattice"])
        output = {"lattice": left["lattice"],
                  "occupancy_deform_offsets": left["occupancy_deform_offsets"]}
        before = float(placed_displacement_term(output, target))
        target["occupied"] = 1-target["occupied"]
        flipped = {"lattice": right["lattice"],
                   "occupancy_deform_offsets": right["occupancy_deform_offsets"]}
        self.assertEqual(float(placed_displacement_term(flipped, target)), before)


class DemeanedRecordTests(unittest.TestCase):
    """The four-arm record: the constant-mode bar, its reproduction and what the documents claim."""

    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.root = root
        cls.scope = json.loads((root/"docs/evidence/m3-occupancy-deformation-r3/scope.json").read_text(
            encoding="utf-8"))
        cls.prior = json.loads((root/"docs/evidence/m3-occupancy-deformation-r2/scope.json").read_text(
            encoding="utf-8"))
        cls.text = " ".join((root/"docs/M3_OCCUPANCY_REGISTRATION.md").read_text(
            encoding="utf-8").split())

    def test_the_four_arms_and_the_verdict_are_mechanical(self):
        self.assertEqual(sorted(self.scope["arms"]), ["control", "demean", "lever", "wide"])
        self.assertEqual({name: arm["occupancy_deform"] for name, arm in self.scope["arms"].items()},
                         {"control": 0, "lever": 1, "wide": 2, "demean": 3})
        self.assertEqual({name: arm["declared_bound_cells"]
                          for name, arm in self.scope["arms"].items()},
                         {"control": 0.0, "lever": 1.0, "wide": 4.0, "demean": 8.0})
        self.assertEqual(self.scope["verdict"]["verdict"], "not-cleared")
        self.assertFalse(self.scope["verdict"]["conditions"]["attribution"])
        self.assertEqual(self.scope["population"]["nonempty_boards"], 28)
        self.assertEqual(self.scope["population"]["occupied_cells"], 667)
        self.assertEqual(self.scope["population"]["contexts"], 12)
        for arm in self.scope["arms"].values():
            self.assertEqual(arm["best"]["exact_nonempty_boards"], 0)

    def test_control_wide_and_one_cell_reproduce_the_previous_block_exactly(self):
        for name in ("control", "lever", "wide"):
            for key in ("auc", "per_cell_f1"):
                self.assertEqual(self.scope["arms"][name][key], self.prior["arms"][name][key], key)
            self.assertEqual(self.scope["arms"][name]["best"]["f1"],
                             self.prior["arms"][name]["best"]["f1"])
            self.assertEqual(self.scope["reproduction"][name]["auc"], self.prior["arms"][name]["auc"])

    def test_the_constant_mode_is_gone_and_per_cell_structure_remains(self):
        comparison = self.scope["constant_mode"]["comparison"]
        self.assertLess(comparison["constant_share"]["demean"], .5)
        self.assertGreater(comparison["constant_share"]["wide"], .95)
        self.assertGreater(comparison["mean_abs_offset_cells"]["demean"], .25)
        self.assertGreater(comparison["occupied_mean_abs_offset_cells"]["demean"], 1.0)
        self.assertEqual(comparison["at_bound_fraction"]["demean"], 0.0)
        self.assertGreater(comparison["outside_own_cell_fraction"]["demean"], .25)
        for axis in comparison["board_mean_of_applied_field"]["demean"]:
            self.assertLessEqual(abs(axis), .05)
        self.assertEqual(self.scope["constant_mode"]["reading"], "registration-without-signal")
        self.assertTrue(self.scope["constant_mode"]["structure_found"])
        self.assertFalse(self.scope["constant_mode"]["ranking_moves"])

    def test_every_constant_share_re_derives_from_its_own_mean_vector(self):
        """No vacuous formula can ship again: the share must follow from the stored moments."""
        for name, field in self.scope["constant_mode"]["field"].items():
            mean_x, mean_y = field["mean_vector_cells"]
            derived = (mean_x*mean_x+mean_y*mean_y)/field["mean_square_cells"]
            self.assertAlmostEqual(derived, field["constant_share"], places=5, msg=name)

    def test_the_demeaned_field_is_its_own_study_and_never_reaches_its_bound(self):
        field = self.scope["constant_mode"]["field"]["demean"]
        self.assertEqual(field["arm"]["occupancy_deform"], 3)
        self.assertEqual(field["arm"]["bound_cells"], DEFORM_DEMEAN_REACH)
        self.assertEqual(DEFORM_DEMEAN_REACH, 2*DEFORM_REACH)
        self.assertLess(field["magnitude_cells"]["max"], DEFORM_DEMEAN_REACH)

    def test_no_gate_bound_ceiling_or_population_was_altered(self):
        sys.path.insert(0, str(self.root/"scripts"))
        from occupancy_sweep import AUC_THRESHOLD, F1_THRESHOLD
        from percell_gate import CELL_F1_THRESHOLD
        envelope = self.scope["envelope"]
        self.assertEqual(envelope["ceiling_bytes"], 1792*1024**2)
        self.assertEqual(envelope["sweep_ceiling_bytes"], 640*1024**2)
        self.assertEqual(envelope["field_ceiling_bytes"], 640*1024**2)
        self.assertEqual(envelope["reach_cells"], {"control": 0.0, "lever": 1.0, "wide": 4.0,
                                                   "demean": 8.0})
        self.assertEqual(envelope["attribution_floor"], .05)
        self.assertEqual(self.scope["gates"], {"per_cell_f1": CELL_F1_THRESHOLD,
                                               "ranking_auc": AUC_THRESHOLD,
                                               "ranking_f1": F1_THRESHOLD,
                                               "attribution_floor": .05, "grid_points": 81})
        self.assertEqual(self.scope["unchanged"]["thresholds"],
                         {"per_cell_f1": CELL_F1_THRESHOLD, "ranking_auc": AUC_THRESHOLD,
                          "ranking_f1": F1_THRESHOLD})

    def test_the_document_quotes_the_published_scope(self):
        demean, wide = self.scope["arms"]["demean"], self.scope["arms"]["wide"]
        comparison = self.scope["constant_mode"]["comparison"]
        deltas = self.scope["contrasts"]
        for value in (demean["auc"], wide["auc"], self.scope["arms"]["control"]["auc"],
                      demean["per_cell_f1"], demean["best"]["f1"], demean["precision"],
                      demean["components"]["out_of_fold_f1"],
                      deltas["demean_control"]["best_f1_delta"],
                      deltas["demean_control"]["auc_delta"],
                      deltas["demean_wide"]["auc_delta"],
                      deltas["wide_control"]["best_f1_delta"]):
            self.assertIn(f"{round(value, 6):,.6f}", self.text)
        self.assertIn(f"{comparison['constant_share']['demean']:.6f}", self.text)
        self.assertIn(f"{comparison['constant_share']['wide']:.3f}", self.text)
        self.assertIn(f"{comparison['mean_abs_offset_cells']['demean']:.3f}", self.text)
        self.assertIn(f"{comparison['mean_abs_offset_cells']['wide']:.3f}", self.text)
        self.assertIn(f"{comparison['occupied_mean_abs_offset_cells']['demean']:.3f}", self.text)
        self.assertIn(f"{100*comparison['outside_own_cell_fraction']['demean']:.1f}%", self.text)
        for arm in self.scope["arms"].values():
            self.assertIn(f"{arm['study_peak_commit_bytes']:,}", self.text)
        self.assertIn(self.scope["constant_mode"]["reading"], self.text)


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
