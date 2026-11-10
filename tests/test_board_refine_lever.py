import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from placement_error import (BOARD_REFINE_ACTIVATION_FLOOR, BOARD_REFINE_SCALE_FLOOR,
                             board_refinement_decomposition, placement_reduction)
from probe_placement import activation_summary
from record_board_refine_lever import (ADDED_PARAMETERS, ADDED_TENSORS, ARMS, CURATED_CONTROL, FAMILY,
                                       PLACEMENT_IMPROVEMENT, activation_engaged, arm_matches,
                                       receipts_after, require_control, require_initial_state)
from record_stream_run import OWNED_RUN_PATTERN
from tetris_vision.losses import (LATTICE_BOARD_REFINE_LEVERS, require_lattice_board_refine)
from tetris_vision.model import ModelConfig, TetrisStateNet, apply_lattice_residual


def probe(occupancy_f1=.75, ranking="informative", attribution=.08, placement_improvement=.30,
          spread=1.0, activation=False):
    placement = {"mean_abs_offset_cells": 1.5, "signed_column_offset_cells": .5,
                 "signed_row_offset_cells": -.5, "mean_scale_ratio_u": 1.05,
                 "mean_scale_ratio_v": 1.02, "board_error_spread_cells": spread}
    baseline = {**placement, "mean_abs_offset_cells": 2.0}
    decision = {"conditions": {"per_cell_f1": occupancy_f1 >= .7, "ranking": ranking == "informative",
                              "attribution": attribution >= .05,
                              "placement": placement_improvement >= PLACEMENT_IMPROVEMENT,
                              "activation": activation},
                "placement_improvement": placement_improvement,
                "placement_floor": PLACEMENT_IMPROVEMENT}
    return baseline, placement, decision


def tiny_config(**overrides):
    config = ModelConfig(widths=(8, 8, 8, 8), dimension=16, heads=1, blocks=1, refinements=0)
    return replace(config, **overrides) if overrides else config


def paired_models():
    torch.manual_seed(31)
    published = TetrisStateNet(tiny_config()).eval()
    torch.manual_seed(31)
    lever = TetrisStateNet(tiny_config(lattice_board_refine=1)).eval()
    added = {name: torch.zeros_like(value) for name, value in lever.state_dict().items()
             if name not in published.state_dict()}
    lever.load_state_dict({**published.state_dict(), **added}, strict=True)
    return published, lever


class BoardRefineContractTests(unittest.TestCase):
    def test_the_flag_is_a_two_value_join(self):
        self.assertEqual(LATTICE_BOARD_REFINE_LEVERS, (0, 1))
        summary = {"arguments": {"lattice_board_refine": 1},
                   "recovery_contract": {"lattice_board_refine": 1}}
        self.assertIsNone(require_lattice_board_refine(1, summary))
        for bad, joined in ((True, summary), (2, summary),
                            (1, {"arguments": {}, "recovery_contract": {}}),
                            (1, {"arguments": {"lattice_board_refine": 1.0}, "recovery_contract": {}})):
            with self.assertRaises(ValueError):
                require_lattice_board_refine(bad, joined)

    def test_the_declared_addition_is_exactly_the_model_difference(self):
        published = TetrisStateNet(tiny_config()).state_dict()
        model = TetrisStateNet(tiny_config(lattice_board_refine=1))
        added = set(model.state_dict())-set(published)
        self.assertEqual(added, set(ADDED_TENSORS))
        self.assertTrue(all(int(torch.count_nonzero(model.state_dict()[name])) == 0 for name in added))
        tiny = sum(model.state_dict()[name].numel() for name in added)
        self.assertEqual(tiny, 16*64+64+64*4+4)
        full = TetrisStateNet(ModelConfig(lattice_board_refine=1))
        self.assertEqual(set(full.state_dict())-set(TetrisStateNet(ModelConfig()).state_dict()),
                         set(ADDED_TENSORS))
        self.assertEqual(ADDED_PARAMETERS, 192*64+64+64*4+4)

    def test_zero_initialization_reproduces_the_published_lattice_exactly(self):
        published, lever = paired_models()
        pixels = torch.rand(1, 3, 64, 64)
        valid = torch.ones(1, 1, 64, 64, dtype=torch.bool)
        with torch.no_grad():
            left, right = published(pixels, valid), lever(pixels, valid)
        for index in range(len(left["stages"])):
            self.assertTrue(torch.equal(left["stages"][index]["lattice"],
                                        right["stages"][index]["lattice"]))
        self.assertTrue(torch.equal(left["final"]["occupancy"], right["final"]["occupancy"]))
        self.assertEqual(tuple(right["final"]["board_refinement"].shape), (1, 4))

    def test_one_board_vector_moves_every_stage_identically(self):
        published, lever = paired_models()
        state = dict(lever.state_dict())
        state["lattice_board_refine.2.bias"] = torch.tensor([1.0, -0.5, 0.25, -0.25])
        lever.load_state_dict(state, strict=True)
        pixels = torch.rand(1, 3, 64, 64)
        valid = torch.ones(1, 1, 64, 64, dtype=torch.bool)
        with torch.no_grad():
            reference = published(pixels, valid)["stages"]
            moved = lever(pixels, valid)["stages"]
        first = reference[0]["lattice"][0]
        shifts = []
        for stage in range(len(moved)):
            delta = (moved[stage]["lattice"][0][0]-reference[stage]["lattice"][0][0])
            basis = torch.stack((first[1], first[2]), 1)
            cells = torch.linalg.lstsq(basis, delta.unsqueeze(1)).solution.flatten()
            shifts.append(cells)
        for stage in shifts[1:]:
            self.assertTrue(torch.allclose(stage, shifts[0], atol=1e-5))
        self.assertGreater(float(shifts[0].abs().sum()), 1e-4)

    def test_the_activation_summary_reports_applied_magnitudes(self):
        self.assertEqual(activation_summary([]), {"available": False})
        zero = activation_summary([[0.0, 0.0, 0.0, 0.0]])
        self.assertTrue(zero["available"])
        self.assertEqual(zero["mean_abs_origin_cells"], 0.0)
        expected = .25*math.tanh(2.0)
        reading = activation_summary([[2.0, -2.0, 2.0, 2.0]])
        self.assertAlmostEqual(reading["mean_abs_origin_cells"], expected, places=6)
        self.assertAlmostEqual(reading["mean_abs_log_scale"], abs(math.exp(expected)-1), places=6)
        # A signed pair averages the two applied magnitudes, which is what the probe reports.
        signed = activation_summary([[2.0, -2.0, 2.0, -2.0]])
        self.assertAlmostEqual(signed["mean_abs_log_scale"],
                               (abs(math.exp(expected)-1)+abs(math.exp(-expected)-1))/2, places=6)


class BoardRefineActivationTests(unittest.TestCase):
    def test_the_declared_floor_decides_engagement(self):
        engaged = {"available": True, "mean_abs_origin_cells": BOARD_REFINE_ACTIVATION_FLOOR,
                   "mean_abs_log_scale": BOARD_REFINE_SCALE_FLOOR}
        self.assertTrue(activation_engaged(engaged))
        for weak in ({"available": True, "mean_abs_origin_cells": 0.09,
                      "mean_abs_log_scale": 0.5},
                     {"available": True, "mean_abs_origin_cells": 0.5,
                      "mean_abs_log_scale": 0.019}):
            self.assertFalse(activation_engaged(weak))
        with self.assertRaises(ValueError):
            activation_engaged({"available": False})
        with self.assertRaises(ValueError):
            activation_engaged({})

    def test_an_unengaged_channel_cannot_rank_a_channel_suspect(self):
        baseline, lever, decision = probe(activation=False, spread=1.0)
        self.assertEqual(board_refinement_decomposition(baseline, lever, decision)["rule"], "fit_gate")
        baseline, lever, decision = probe(activation=True, spread=2.0)
        self.assertEqual(board_refinement_decomposition(baseline, lever, decision)["rule"], "fit_gate")
        baseline, lever, decision = probe(activation=True, spread=1.0)
        self.assertEqual(board_refinement_decomposition(baseline, lever, decision)["rule"], "scaling")

    def test_the_rule_is_non_circular(self):
        spent = ("translation", "scale", "geometry_capacity", "placement_capacity", "lattice_residual",
                 "lattice_free")
        for activation in (True, False):
            for spread in (0.0, 1.49, 1.5, 5.0):
                for improvement in (-.4, 0.0, .3, 5.0):
                    baseline, lever, decision = probe(activation=activation, spread=spread,
                                                      placement_improvement=improvement)
                    self.assertNotIn(board_refinement_decomposition(baseline, lever, decision)["rule"],
                                     spent)


class CuratorJoinTests(unittest.TestCase):
    def summary(self, flag=1, **overrides):
        arguments = {"lattice_board_refine": flag, "lattice_residual": 0,
                     "occupancy_lattice_free": 0, "support_anchor_weight": 0.0,
                     "box_extent_weight": 0.0, "lattice_support": 0, "geometry_final_only": 0,
                     "occupancy_detail": 1, "occupancy_detail_stencil": 0, "occupancy_context": 0,
                     "steps": 256, "seed": 20261026, "capture_max_side": 512,
                     "occupancy_factor": 4, "occupancy_positive_weight": 4}
        summary = {"status": "complete", "steps": 256, "requested_steps": 256,
                   "checkpoint_round_trip_exact": True, "arguments": dict(arguments),
                   "recovery_contract": dict(arguments)}
        summary.update(overrides)
        return summary

    def job(self, **overrides):
        job = {"status": "passed", "exit_code": 0, "work_items": 256, "memory_bytes": 1152*1024**2}
        job.update(overrides)
        return job

    def test_both_arms_are_admitted_only_for_their_own_flag(self):
        self.assertTrue(arm_matches(self.summary(1), self.job(), 1))
        self.assertTrue(arm_matches(self.summary(0), self.job(), 0))
        self.assertEqual(ARMS, {"lever": 1, "baseline": 0})
        for bad_flag, joined in ((1, 0), (0, 1)):
            with self.assertRaises(ValueError):
                arm_matches(self.summary(bad_flag), self.job(), joined)

    def test_the_join_refuses_foreign_configuration_or_envelope(self):
        for weight, job, summary in (
                (1, self.job(status="failed"), self.summary(1)),
                (1, self.job(work_items=2), self.summary(1)),
                (1, self.job(memory_bytes=640*1024**2), self.summary(1)),
                (1, self.job(), self.summary(1, steps=128)),
        ):
            with self.assertRaises(ValueError):
                arm_matches(summary, job, weight)
        for key, value in (("lattice_residual", 1), ("occupancy_lattice_free", 1),
                           ("geometry_final_only", 1), ("lattice_support", 1),
                           ("box_extent_weight", 1.0), ("support_anchor_weight", 1.0)):
            foreign = self.summary(1)
            foreign["arguments"][key] = value
            foreign["recovery_contract"][key] = value
            with self.assertRaises(ValueError):
                arm_matches(foreign, self.job(), 1)

    def test_receipts_must_postdate_the_predeclaration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"receipt.json"
            path.write_text("{}")
            stamp = int(path.stat().st_mtime)
            self.assertTrue(receipts_after(stamp-60, [path]))
            with self.assertRaises(ValueError):
                receipts_after(stamp+60, [path])
            with self.assertRaises(ValueError):
                receipts_after(stamp, [Path(directory)/"absent.json"])

    def test_the_baseline_control_drift_is_refused(self):
        placement = {"mean_abs_offset_cells": CURATED_CONTROL["mean_abs_offset_cells"],
                     "mean_scale_ratio_u": CURATED_CONTROL["mean_scale_ratio_u"],
                     "mean_scale_ratio_v": CURATED_CONTROL["mean_scale_ratio_v"],
                     "board_error_spread_cells": CURATED_CONTROL["board_error_spread_cells"]}
        occupancy = {"f1": CURATED_CONTROL["f1"], "auc": CURATED_CONTROL["auc"]}
        best = {"f1": CURATED_CONTROL["best_f1"], "threshold": CURATED_CONTROL["best_threshold"]}
        self.assertTrue(require_control(placement, occupancy, best))
        with self.assertRaises(ValueError):
            require_control({**placement, "mean_abs_offset_cells": 1.99}, occupancy, best)
        with self.assertRaises(ValueError):
            require_control(placement, {**occupancy, "auc": .83}, best)

    def test_the_matched_start_accepts_the_declared_addition_only(self):
        torch.manual_seed(32)
        baseline = TetrisStateNet(tiny_config()).state_dict()
        lever = {**baseline,
                 **{name: torch.zeros_like(value) for name, value in
                    TetrisStateNet(tiny_config(lattice_board_refine=1)).state_dict().items()
                    if name not in baseline}}
        self.assertTrue(require_initial_state(lever, baseline))
        nonzero = dict(lever)
        nonzero["lattice_board_refine.2.bias"] = torch.ones_like(nonzero["lattice_board_refine.2.bias"])
        with self.assertRaises(ValueError):
            require_initial_state(nonzero, baseline)
        extra = {**lever, "lattice_board_refine.9.weight": torch.zeros(1, 1)}
        with self.assertRaises(ValueError):
            require_initial_state(extra, baseline)

    def test_the_family_is_owned_without_dropping_earlier_families(self):
        self.assertEqual(FAMILY, "m3-occbref")
        for name in ("m3-occbref-lever-r1", "m3-occbref-baseline-r1-sweep-r1",
                     "m3-occlatfree-lever-r1", "m3-occlatres-lever-r1", "m3-occgeocap-lever-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        self.assertFalse(OWNED_RUN_PATTERN.fullmatch("m3-unknown-lever-r1"))

    def test_the_refinement_map_is_the_declared_bounded_correction(self):
        lattice = torch.tensor([[[.2, .2], [.05, 0.], [0., -.04]]])
        self.assertTrue(torch.equal(apply_lattice_residual(lattice, torch.zeros(1, 4)), lattice))
        moved = apply_lattice_residual(lattice, torch.tensor([[5.0, -5.0, 5.0, -5.0]]))
        self.assertLessEqual(float((moved[:, 0]-lattice[:, 0]).abs().max()), .25*2**.5*.05+.25*.06)


if __name__ == "__main__":
    unittest.main()
