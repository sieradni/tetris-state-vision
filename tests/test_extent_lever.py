"""The declared box-extent lever: admission, the shared matched start, the rule and the objective."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from matched_start import matched_start, step_zero_identical
from placement_error import (EXTENT_REDUCTION, extent_decomposition, extent_offsets_moved,
                             extent_reductions, extent_verdict)
from record_extent_lever import STEPS, arm_matches
from record_stream_run import OWNED_RUN_PATTERN
from tetris_vision.losses import BOX_EXTENT_WEIGHTS, extent_error, require_box_extent_weight


def summary(weight=1.0, *, steps=STEPS, detail=1, stencil=0, context=0, factor=4, positive=4,
            geometry=0, lattice=0):
    return {"status": "complete", "steps": steps, "requested_steps": steps,
            "checkpoint_round_trip_exact": True,
            "arguments": {"box_extent_weight": weight, "geometry_final_only": geometry,
                          "lattice_support": lattice, "occupancy_detail": detail,
                          "occupancy_detail_stencil": stencil, "occupancy_context": context,
                          "steps": steps, "seed": 20261026, "capture_max_side": 512,
                          "occupancy_factor": factor, "occupancy_positive_weight": positive},
            "recovery_contract": {"box_extent_weight": weight, "geometry_final_only": geometry,
                                  "lattice_support": lattice}}


def job(steps=STEPS, ceiling=1152*1024**2, status="passed"):
    return {"status": status, "exit_code": 0 if status == "passed" else 1,
            "work_items": steps, "memory_bytes": ceiling}


def placement(*, ratio_u=1.0, ratio_v=1.0, spread=1.0, column=1.0, row=-1.0):
    return {"mean_scale_ratio_u": ratio_u, "mean_scale_ratio_v": ratio_v,
            "board_error_spread_cells": spread, "signed_column_offset_cells": column,
            "signed_row_offset_cells": row}


class AdmissionTests(unittest.TestCase):
    def test_both_policies_are_admitted_only_for_their_own_weight(self):
        self.assertTrue(arm_matches(summary(1.0), job(), 1.0))
        self.assertTrue(arm_matches(summary(0.0), job(), 0.0))
        with self.assertRaises(ValueError):
            arm_matches(summary(1.0), job(), 0.0)

    def test_the_closed_head_levers_are_refused_here(self):
        with self.assertRaises(ValueError):
            arm_matches(summary(1.0, lattice=1), job(), 1.0)
        with self.assertRaises(ValueError):
            arm_matches(summary(1.0, geometry=1), job(), 1.0)

    def test_a_wrong_budget_ceiling_or_receipt_is_refused(self):
        with self.assertRaises(ValueError):
            arm_matches(summary(1.0, steps=64), job(steps=64), 1.0)
        with self.assertRaises(ValueError):
            arm_matches(summary(1.0), job(ceiling=640*1024**2), 1.0)
        with self.assertRaises(ValueError):
            arm_matches(summary(1.0), job(status="failed"), 1.0)

    def test_another_read_out_in_the_arm_is_refused(self):
        for override in ({"detail": 0}, {"stencil": 1}, {"context": 1}, {"factor": 1}, {"positive": 1}):
            with self.assertRaises(ValueError):
                arm_matches(summary(1.0, **override), job(), 1.0)

    def test_the_weight_is_a_two_value_declaration(self):
        self.assertEqual(BOX_EXTENT_WEIGHTS, (0.0, 1.0))
        self.assertIsNone(require_box_extent_weight(1.0, summary(1.0)))
        self.assertIsNone(require_box_extent_weight(0.0, summary(0.0)))
        for weight, record in ((1.0, summary(0.0)), (0.0, summary(1.0)),
                               (1, summary(1.0)), (2.0, summary(2.0))):
            with self.assertRaises(ValueError):
                require_box_extent_weight(weight, record)

    def test_the_driver_mirrors_the_declaration(self):
        from train_m2_stream import BOX_EXTENT_WEIGHTS as DRIVER_WEIGHTS
        self.assertEqual(tuple(DRIVER_WEIGHTS), tuple(BOX_EXTENT_WEIGHTS))

    def test_the_owned_family_is_declared(self):
        for name in ("m3-occextent-lever-r1", "m3-occextent-baseline-r1",
                     "m3-occextent-lever-r1-sweep-r1", "m3-occextent-baseline-r1-probe-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)


class MatchedStartTests(unittest.TestCase):
    """V-071 generalized: the contract must accept both kinds of lever and refuse a moved start."""

    def setUp(self):
        import torch
        self.torch = torch
        self.shared = {"encoder.weight": torch.tensor([1.0, -2.0, 3.5])}

    def test_identical_states_are_a_matched_start_without_an_added_parameter(self):
        self.assertTrue(matched_start(dict(self.shared), dict(self.shared), added=()))

    def test_a_parameter_adding_lever_is_accepted_at_its_declared_init(self):
        lever = {**self.shared, "support_box.weight": self.torch.zeros(4, 5),
                 "support_box.bias": self.torch.zeros(4)}
        self.assertTrue(matched_start(lever, dict(self.shared), added=("support_box.weight",
                                                                      "support_box.bias")))
        self.assertTrue(matched_start(lever, dict(self.shared),
                                      added={"support_box.weight": self.torch.zeros(4, 5),
                                             "support_box.bias": self.torch.zeros(4)}))

    def test_a_nonzero_added_tensor_or_a_mismatched_declared_init_is_refused(self):
        lever = {**self.shared, "support_box.weight": self.torch.zeros(4, 5),
                 "support_box.bias": self.torch.ones(4)}
        with self.assertRaises(ValueError):
            matched_start(lever, dict(self.shared), added=("support_box.weight", "support_box.bias"))
        with self.assertRaises(ValueError):
            matched_start({**self.shared, "support_box.bias": self.torch.zeros(4)},
                          dict(self.shared),
                          added={"support_box.bias": self.torch.ones(4)})

    def test_a_moved_shared_tensor_any_extra_or_any_missing_tensor_is_refused(self):
        moved = {"encoder.weight": self.torch.tensor([1.0, -2.0, 3.6])}
        with self.assertRaises(ValueError):
            matched_start(dict(self.shared), moved, added=())
        with self.assertRaises(ValueError):
            matched_start({**self.shared, "decoder.weight": self.torch.zeros(2)}, dict(self.shared))
        with self.assertRaises(ValueError):
            matched_start(dict(self.shared),
                          {**self.shared, "extra.weight": self.torch.zeros(1)}, added=())

    def test_step_zero_must_be_the_same_measured_function_not_the_same_objective_value(self):
        probe = {"initial_training_probe": {"loss": 12.28, "cell_accuracy": .795,
                                            "cases": [{"id": "a", "loss": 10.19, "cell_accuracy": .815}]},
                 "initial_development": {"loss": 10.75, "cell_accuracy": .8297,
                                         "cases": [{"id": "b", "loss": 9.99, "cell_accuracy": .83}]}}
        self.assertTrue(step_zero_identical(dict(probe), dict(probe)))
        # V-072: an objective lever changes the step-0 loss - top level and per case - by construction.
        relossed = json.loads(json.dumps(probe))
        relossed["initial_training_probe"]["loss"] = 12.41
        relossed["initial_training_probe"]["cases"][0]["loss"] = 10.32
        self.assertTrue(step_zero_identical(relossed, dict(probe)))
        # Any prediction field that differs is a real mismatch.
        moved = {**probe, "initial_development": {**probe["initial_development"],
                                                  "cell_accuracy": .831}}
        with self.assertRaises(ValueError):
            step_zero_identical(moved, dict(probe))
        with self.assertRaises(ValueError):
            step_zero_identical({"initial_training_probe": probe["initial_training_probe"]},
                                dict(probe))


class ExtentRuleTests(unittest.TestCase):
    def test_the_declared_floor_is_half_of_each_log_ratio(self):
        self.assertEqual(EXTENT_REDUCTION, 0.5)
        reduction = extent_reductions(placement(ratio_u=1.225688, ratio_v=1.180353),
                                      placement(ratio_u=1.10, ratio_v=1.08))
        self.assertGreater(min(reduction.values()), 0.5)

    def test_both_axes_must_move(self):
        moved, reduction = extent_offsets_moved(placement(ratio_u=1.225688, ratio_v=1.180353),
                                                placement(ratio_u=1.10, ratio_v=1.19))
        self.assertFalse(moved)
        self.assertLess(reduction["mean_scale_ratio_v"], 0.5)

    def test_a_ratio_crossing_one_counts_only_by_its_log_magnitude(self):
        moved, reduction = extent_offsets_moved(placement(ratio_u=1.4, ratio_v=1.4),
                                                placement(ratio_u=0.85, ratio_v=0.85))
        self.assertTrue(moved)
        self.assertAlmostEqual(reduction["mean_scale_ratio_u"], 1-abs(__import__("math").log(0.85))/abs(__import__("math").log(1.4)))

    def test_the_verdict_needs_the_three_gates_and_the_extent(self):
        baseline = placement(ratio_u=1.225688, ratio_v=1.180353)
        lever = placement(ratio_u=1.10, ratio_v=1.08)
        cleared = extent_verdict({"f1": .75}, "informative", .08, baseline, lever)
        self.assertEqual(cleared["verdict"], "cleared")
        self.assertIn("extent", extent_verdict({"f1": .75}, "informative", .08, baseline,
                                               placement(ratio_u=1.10, ratio_v=1.19))["failing"])
        self.assertIn("attribution", extent_verdict({"f1": .75}, "informative", .04, baseline,
                                                    lever)["failing"])
        self.assertIn("per_cell_f1", extent_verdict({"f1": .5}, "informative", .08, baseline,
                                                    lever)["failing"])

    def test_the_decomposition_is_non_circular(self):
        baseline = placement(ratio_u=1.225688, ratio_v=1.180353)
        stalled = extent_verdict({"f1": .5}, "uninformative", 0.0, baseline,
                                 placement(ratio_u=1.21, ratio_v=1.17))
        self.assertEqual(extent_decomposition(baseline, placement(ratio_u=1.21, ratio_v=1.17),
                                              stalled)["rule"], "geometry_capacity")
        moved = extent_verdict({"f1": .5}, "uninformative", 0.0, baseline,
                               placement(ratio_u=1.05, ratio_v=1.04))
        self.assertEqual(extent_decomposition(baseline, placement(ratio_u=1.05, ratio_v=1.04),
                                              moved)["rule"], "read_out_inheritance")
        # Extent moved and the occupancy gates are met, so the rule falls through to dispersion.
        spread = extent_verdict({"f1": .75}, "informative", .04, baseline,
                                placement(ratio_u=1.05, ratio_v=1.04, spread=2.0))
        self.assertEqual(extent_decomposition(baseline,
                                              placement(ratio_u=1.05, ratio_v=1.04, spread=2.0),
                                              spread)["rule"], "dispersion")
        compact = extent_verdict({"f1": .75}, "informative", .04, baseline,
                                 placement(ratio_u=1.05, ratio_v=1.04, spread=1.0))
        self.assertEqual(extent_decomposition(baseline,
                                              placement(ratio_u=1.05, ratio_v=1.04, spread=1.0),
                                              compact)["rule"], "scaling")


class ObjectiveTests(unittest.TestCase):
    def test_the_extent_error_is_the_relative_log_scale_per_axis(self):
        import torch
        lattice = torch.tensor([[[0., 0.], [.1, 0.], [0., -.05]]])
        self.assertTrue(torch.equal(extent_error(lattice, lattice.clone()), torch.zeros(1, 2)))
        wider = torch.tensor([[[0., 0.], [.12, 0.], [0., -.05]]])
        error = extent_error(wider, lattice)
        self.assertAlmostEqual(float(error[0, 0]), float(torch.log(torch.tensor(1.2))), places=5)
        self.assertAlmostEqual(float(error[0, 1]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
