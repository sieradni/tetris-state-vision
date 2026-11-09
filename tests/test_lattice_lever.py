"""The declared box-origin lever: admission join, zero-init identity, ownership and the rule."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from placement_error import (SIGNED_REDUCTION, signed_offsets_moved, signed_reduction,
                             translation_decomposition, translation_verdict)
from record_lattice_lever import STEPS, arm_matches
from record_stream_run import OWNED_RUN_PATTERN
from tetris_vision.losses import LATTICE_SUPPORT_LEVERS, require_lattice_support


def summary(flag=1, *, steps=STEPS, detail=1, stencil=0, context=0, factor=4, weight=4,
            geometry=0):
    return {"status": "complete", "steps": steps, "requested_steps": steps,
            "checkpoint_round_trip_exact": True,
            "arguments": {"lattice_support": flag, "geometry_final_only": geometry,
                          "occupancy_detail": detail, "occupancy_detail_stencil": stencil,
                          "occupancy_context": context, "steps": steps, "seed": 20261026,
                          "capture_max_side": 512, "occupancy_factor": factor,
                          "occupancy_positive_weight": weight},
            "recovery_contract": {"lattice_support": flag, "geometry_final_only": geometry}}


def job(steps=STEPS, ceiling=1152*1024**2, status="passed"):
    return {"status": status, "exit_code": 0 if status == "passed" else 1,
            "work_items": steps, "memory_bytes": ceiling}


def placement(column, row, *, spread=1.0, ratio_u=1.0, ratio_v=1.0):
    return {"signed_column_offset_cells": column, "signed_row_offset_cells": row,
            "board_error_spread_cells": spread, "mean_scale_ratio_u": ratio_u,
            "mean_scale_ratio_v": ratio_v}


class AdmissionTests(unittest.TestCase):
    def test_both_policies_are_admitted_only_for_their_own_flag(self):
        self.assertTrue(arm_matches(summary(1), job(), 1))
        self.assertTrue(arm_matches(summary(0), job(), 0))
        with self.assertRaises(ValueError):
            arm_matches(summary(1), job(), 0)

    def test_the_previous_geometry_policy_is_refused_here(self):
        with self.assertRaises(ValueError):
            arm_matches(summary(1, geometry=1), job(), 1)

    def test_a_wrong_budget_ceiling_or_receipt_is_refused(self):
        with self.assertRaises(ValueError):
            arm_matches(summary(1, steps=64), job(steps=64), 1)
        with self.assertRaises(ValueError):
            arm_matches(summary(1), job(ceiling=640*1024**2), 1)
        with self.assertRaises(ValueError):
            arm_matches(summary(1), job(status="failed"), 1)

    def test_another_read_out_in_the_arm_is_refused(self):
        for override in ({"detail": 0}, {"stencil": 1}, {"context": 1}, {"factor": 1}, {"weight": 1}):
            with self.assertRaises(ValueError):
                arm_matches(summary(1, **override), job(), 1)

    def test_the_lever_is_a_two_value_declaration(self):
        self.assertEqual(LATTICE_SUPPORT_LEVERS, (0, 1))
        self.assertIsNone(require_lattice_support(1, summary(1)))
        self.assertIsNone(require_lattice_support(0, summary(0)))
        with self.assertRaises(ValueError):
            require_lattice_support(1, summary(0))
        with self.assertRaises(ValueError):
            require_lattice_support(True, summary(1))

    def test_the_owned_family_is_declared(self):
        for name in ("m3-occorigin-lever-r1", "m3-occorigin-baseline-r1",
                     "m3-occorigin-lever-r1-sweep-r1", "m3-occorigin-baseline-r1-probe-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)


class RuleTests(unittest.TestCase):
    def test_the_signed_condition_requires_both_axes_halved(self):
        baseline = placement(2.3, -2.4)
        moved, reduction = signed_offsets_moved(baseline, placement(1.0, -1.0))
        self.assertTrue(moved)
        self.assertGreater(min(reduction.values()), .5)
        moved, _ = signed_offsets_moved(baseline, placement(1.5, -1.0))
        self.assertFalse(moved)

    def test_a_reversed_sign_is_not_a_reduction(self):
        moved, reduction = signed_offsets_moved(placement(2.0, -2.0), placement(-2.0, 2.0))
        self.assertFalse(moved)
        self.assertEqual(reduction["signed_column_offset_cells"], 0.0)

    def test_the_verdict_needs_the_three_gates_and_the_offsets(self):
        baseline, lever = placement(2.3, -2.4), placement(1.0, -1.0)
        cleared = translation_verdict({"f1": .75}, "informative", .08, baseline, lever)
        self.assertEqual(cleared["verdict"], "cleared")
        self.assertIn("signed_offsets",
                      translation_verdict({"f1": .75}, "informative", .08, baseline,
                                          placement(1.5, -1.0))["failing"])
        self.assertIn("attribution",
                      translation_verdict({"f1": .75}, "informative", .04, baseline, lever)["failing"])

    def test_the_decomposition_is_non_circular(self):
        baseline = placement(2.3, -2.4)
        unmoved = translation_verdict({"f1": .5}, "uninformative", .0, baseline, placement(2.2, -2.3))
        self.assertEqual(translation_decomposition(baseline, placement(2.2, -2.3), unmoved)["rule"],
                         "extent")
        moved = translation_verdict({"f1": .5}, "uninformative", .0, baseline, placement(1.0, -1.0))
        self.assertEqual(translation_decomposition(baseline, placement(1.0, -1.0), moved)["rule"],
                         "read-out_inheritance")
        halved = translation_verdict({"f1": .5}, "uninformative", .0, baseline, placement(1.6, -1.7))
        self.assertEqual(translation_decomposition(baseline, placement(1.6, -1.7, spread=2.0),
                                                   halved)["rule"], "dispersion")

    def test_the_signed_floor_is_the_declared_half(self):
        self.assertEqual(SIGNED_REDUCTION, 0.5)
        self.assertEqual(signed_reduction(placement(2.0, -4.0), placement(1.0, -2.0)),
                         {"signed_column_offset_cells": .5, "signed_row_offset_cells": .5})


class ModelTests(unittest.TestCase):
    def test_the_support_channel_starts_as_the_published_function(self):
        import torch
        from tetris_vision.model import ModelConfig, TetrisStateNet
        config = ModelConfig(**{**ModelConfig.pilot().__dict__, "lattice_support": 1})
        model = TetrisStateNet(config)
        self.assertIsNotNone(model.support_box)
        self.assertTrue(torch.equal(model.support_box.weight, torch.zeros_like(model.support_box.weight)))
        self.assertTrue(torch.equal(model.support_box.bias, torch.zeros_like(model.support_box.bias)))

    def test_the_published_config_has_no_support_parameters(self):
        from tetris_vision.model import ModelConfig, TetrisStateNet
        model = TetrisStateNet(ModelConfig.pilot())
        self.assertIsNone(model.support_box)
        self.assertFalse(any(name.startswith("support_box.") for name in model.state_dict()))

    def test_moments_are_normalized_and_finite(self):
        import torch
        from tetris_vision.model import TetrisStateNet
        coarse = torch.zeros(2, 8, 6, 6)
        coarse[0, :, 1:3, 2:4] = 1.0
        moments = TetrisStateNet.support_moments(coarse)
        self.assertEqual(tuple(moments.shape), (2, 5))
        self.assertTrue(torch.isfinite(moments).all())
        self.assertTrue(all(-1 <= value <= 1 for value in moments[0, :4].tolist()))
        self.assertGreater(moments[0, 4].item(), 0.0)
        self.assertEqual(moments[1, 4].item(), 0.0)

    def test_the_config_rejects_undeclared_values(self):
        from tetris_vision.model import ModelConfig
        with self.assertRaises(ValueError):
            ModelConfig(**{**ModelConfig.pilot().__dict__, "lattice_support": 2})


if __name__ == "__main__":
    unittest.main()
