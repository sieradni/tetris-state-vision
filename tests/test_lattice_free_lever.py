import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from placement_error import lattice_free_decomposition, placement_reduction
from record_lattice_free_lever import (ADDED_PARAMETERS, ADDED_TENSORS, ARMS, CURATED_CONTROL,
                                       FAMILY, PLACEMENT_IMPROVEMENT, arm_matches, gate_reading,
                                       receipts_after, require_control, require_initial_state)
from record_stream_run import OWNED_RUN_PATTERN
from tetris_vision.losses import (OCCUPANCY_LATTICE_FREE_LEVERS, require_occupancy_lattice_free)
from tetris_vision.model import ModelConfig, TetrisStateNet


def probe(occupancy_f1=.75, ranking="informative", attribution=.08, placement_improvement=.30,
          spread=1.0):
    placement = {"mean_abs_offset_cells": 1.5, "signed_column_offset_cells": .5,
                 "signed_row_offset_cells": -.5, "mean_scale_ratio_u": 1.05,
                 "mean_scale_ratio_v": 1.02, "board_error_spread_cells": spread}
    baseline = {**placement, "mean_abs_offset_cells": 2.0}
    decision = {"conditions": {"per_cell_f1": occupancy_f1 >= .7, "ranking": ranking == "informative",
                              "attribution": attribution >= .05,
                              "placement": placement_improvement >= PLACEMENT_IMPROVEMENT},
                "placement_improvement": placement_improvement,
                "placement_floor": PLACEMENT_IMPROVEMENT}
    return baseline, placement, decision


def tiny_config(**overrides):
    config = ModelConfig(widths=(8, 8, 8, 8), dimension=16, heads=1, blocks=1, refinements=0)
    return replace(config, **overrides) if overrides else config


def paired_models():
    torch.manual_seed(21)
    published = TetrisStateNet(tiny_config()).eval()
    torch.manual_seed(21)
    lever = TetrisStateNet(tiny_config(occupancy_lattice_free=1)).eval()
    added = {name: torch.zeros_like(value) for name, value in lever.state_dict().items()
             if name not in published.state_dict()}
    lever.load_state_dict({**published.state_dict(), **added}, strict=True)
    return published, lever


class LatticeFreeContractTests(unittest.TestCase):
    def test_the_flag_is_a_two_value_join(self):
        # The constant grew with M20: 2 is the summed two-path read, and 0/1 keep their meanings.
        self.assertEqual(OCCUPANCY_LATTICE_FREE_LEVERS, (0, 1, 2))
        summary = {"arguments": {"occupancy_lattice_free": 1},
                   "recovery_contract": {"occupancy_lattice_free": 1}}
        self.assertIsNone(require_occupancy_lattice_free(1, summary))
        for bad, joined in ((True, summary), (2, summary),
                            (1, {"arguments": {}, "recovery_contract": {}}),
                            (1, {"arguments": {"occupancy_lattice_free": 1.0}, "recovery_contract": {}})):
            with self.assertRaises(ValueError):
                require_occupancy_lattice_free(bad, joined)

    def test_the_declared_addition_is_exactly_the_model_difference(self):
        published = TetrisStateNet(tiny_config()).state_dict()
        model = TetrisStateNet(tiny_config(occupancy_lattice_free=1))
        added = set(model.state_dict())-set(published)
        self.assertEqual(added, set(ADDED_TENSORS))
        # Only the output projection and the gate are zeroed (V-143): the hidden layer keeps its random
        # initialization, because a zeroed hidden layer can only ever learn a constant. Identity at step
        # zero therefore comes from the output projection, which the next test checks on the read-out.
        zeroed = ("occupancy_lattice_free.2.weight", "occupancy_lattice_free.2.bias", "occupancy_lattice_gate")
        self.assertTrue(all(int(torch.count_nonzero(model.state_dict()[name])) == 0 for name in zeroed))
        self.assertGreater(int(torch.count_nonzero(
            model.state_dict()["occupancy_lattice_free.0.weight"])), 0)
        tiny = sum(model.state_dict()[name].numel() for name in added)
        self.assertEqual(tiny, 16*16+16+16+1+1)  # tiny config: dimension 16
        full_added = set(TetrisStateNet(ModelConfig(occupancy_lattice_free=1)).state_dict())-set(
            TetrisStateNet(ModelConfig()).state_dict())
        self.assertEqual(full_added, set(ADDED_TENSORS))
        self.assertEqual(ADDED_PARAMETERS, 64*64+64+64+1+1)

    def test_zero_initialization_reproduces_the_published_readout_exactly(self):
        published, lever = paired_models()
        pixels = torch.rand(1, 3, 64, 64)
        valid = torch.ones(1, 1, 64, 64, dtype=torch.bool)
        with torch.no_grad():
            left, right = published(pixels, valid), lever(pixels, valid)
        self.assertTrue(torch.equal(left["final"]["occupancy"], right["final"]["occupancy"]))
        self.assertTrue(torch.equal(left["final"]["lattice"], right["final"]["lattice"]))

    def test_a_closed_gate_replaces_the_lattice_specific_path(self):
        published, lever = paired_models()
        pixels = torch.rand(1, 3, 64, 64)
        valid = torch.ones(1, 1, 64, 64, dtype=torch.bool)
        state = dict(lever.state_dict())
        state["occupancy_lattice_gate"] = torch.tensor([20.])  # tanh -> 1, channel fully replaces
        state["occupancy_lattice_free.2.weight"] = torch.full_like(state["occupancy_lattice_free.2.weight"], .1)
        state["occupancy_lattice_free.0.weight"] = torch.full_like(state["occupancy_lattice_free.0.weight"], .05)
        lever.load_state_dict(state, strict=True)
        with torch.no_grad():
            first = lever(pixels, valid)["final"]["occupancy"].clone()
            # The published lattice-sampled head is now suppressed: perturbing it must not matter.
            for name in ("occupancy.weight", "occupancy.bias"):
                lever.state_dict()[name].add_(1.)
            second = lever(pixels, valid)["final"]["occupancy"]
            lever.load_state_dict(state, strict=True)
            self.assertTrue(torch.allclose(first, second, atol=1e-5))
            self.assertFalse(torch.allclose(published(pixels, valid)["final"]["occupancy"], first))

    def test_the_gate_reading_reports_the_declared_scalar(self):
        reading = gate_reading({"occupancy_lattice_gate": torch.zeros(1)})
        self.assertTrue(reading["available"])
        self.assertEqual(reading["raw"], 0.0)
        self.assertEqual(reading["abs_tanh"], 0.0)
        closed = gate_reading({"occupancy_lattice_gate": torch.tensor([2.])})
        self.assertAlmostEqual(closed["tanh"], float(torch.tanh(torch.tensor(2.))), places=6)
        self.assertFalse(gate_reading({})["available"])


class LatticeFreeRuleTests(unittest.TestCase):
    def test_the_verdict_needs_all_four_conditions(self):
        baseline, lever, decision = probe()
        self.assertEqual(placement_reduction(baseline, lever), .25)
        _, _, tight = probe(placement_improvement=.24)
        self.assertFalse(tight["conditions"]["placement"])

    def test_the_rule_is_non_circular_and_spread_decided(self):
        baseline, lever, decision = probe(spread=2.0)
        self.assertEqual(lattice_free_decomposition(baseline, lever, decision)["rule"], "dispersion")
        baseline, lever, decision = probe(spread=1.0)
        self.assertEqual(lattice_free_decomposition(baseline, lever, decision)["rule"], "scaling")
        for spread, improvement in ((.0, 5.0), (1.0, .0), (2.5, .3), (1.49, -.4)):
            baseline, lever, decision = probe(spread=spread, placement_improvement=improvement)
            self.assertNotIn(lattice_free_decomposition(baseline, lever, decision)["rule"],
                             ("translation", "scale", "geometry_capacity", "placement_capacity",
                              "lattice_residual"))


class CuratorJoinTests(unittest.TestCase):
    def summary(self, flag=1, **overrides):
        arguments = {"occupancy_lattice_free": flag, "lattice_residual": 0,
                     "support_anchor_weight": 0.0, "box_extent_weight": 0.0, "lattice_support": 0,
                     "geometry_final_only": 0, "occupancy_detail": 1, "occupancy_detail_stencil": 0,
                     "occupancy_context": 0, "steps": 256, "seed": 20261026,
                     "capture_max_side": 512, "occupancy_factor": 4, "occupancy_positive_weight": 4}
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
        for key, value in (("lattice_residual", 1), ("geometry_final_only", 1), ("lattice_support", 1),
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
        torch.manual_seed(22)
        baseline = TetrisStateNet(tiny_config()).state_dict()
        lever = {**baseline,
                 **{name: torch.zeros_like(value) for name, value in
                    TetrisStateNet(tiny_config(occupancy_lattice_free=1)).state_dict().items()
                    if name not in baseline}}
        self.assertTrue(require_initial_state(lever, baseline))
        nonzero = dict(lever)
        nonzero["occupancy_lattice_gate"] = torch.ones_like(nonzero["occupancy_lattice_gate"])
        with self.assertRaises(ValueError):
            require_initial_state(nonzero, baseline)
        extra = {**lever, "occupancy_lattice_free.9.weight": torch.zeros(1, 1)}
        with self.assertRaises(ValueError):
            require_initial_state(extra, baseline)

    def test_the_family_is_owned_without_dropping_earlier_families(self):
        self.assertEqual(FAMILY, "m3-occlatfree")
        for name in ("m3-occlatfree-lever-r1", "m3-occlatfree-baseline-r1-sweep-r1",
                     "m3-occlatres-lever-r1", "m3-occgeocap-lever-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        self.assertFalse(OWNED_RUN_PATTERN.fullmatch("m3-unknown-lever-r1"))


if __name__ == "__main__":
    unittest.main()
