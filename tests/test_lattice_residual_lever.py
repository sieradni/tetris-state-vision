import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from placement_error import (lattice_residual_decomposition, placement_reduction)
from record_lattice_residual_lever import (ADDED_PARAMETERS, ADDED_TENSORS, ARMS, CURATED_CONTROL,
                                           FAMILY, PLACEMENT_IMPROVEMENT, arm_matches,
                                           receipts_after, require_control, require_initial_state)
from record_stream_run import OWNED_RUN_PATTERN
from tetris_vision.losses import LATTICE_RESIDUAL_LEVERS, require_lattice_residual
from tetris_vision.model import (LATTICE_RESIDUAL_SCALE, ModelConfig, TetrisStateNet,
                                 apply_lattice_residual)

BASE_LATTICE = torch.tensor([[[.1, .9], [.08, 0.], [0., -.04]]])


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


class ResidualContractTests(unittest.TestCase):
    def test_the_residual_is_the_identity_at_zero(self):
        for lattice in (BASE_LATTICE, BASE_LATTICE.clone()*1.7):
            zero = torch.zeros(lattice.shape[0], 4)
            self.assertTrue(torch.equal(apply_lattice_residual(lattice, zero), lattice))

    def test_the_residual_moves_only_its_declared_degrees(self):
        shifted = apply_lattice_residual(BASE_LATTICE, torch.tensor([[1., 0., 0., 0.]]))
        self.assertFalse(torch.equal(shifted[:, 0], BASE_LATTICE[:, 0]))
        self.assertTrue(torch.equal(shifted[:, 1], BASE_LATTICE[:, 1]))
        self.assertTrue(torch.equal(shifted[:, 2], BASE_LATTICE[:, 2]))
        scaled = apply_lattice_residual(BASE_LATTICE, torch.tensor([[0., 0., 1., 0.]]))
        self.assertTrue(torch.equal(scaled[:, 0], BASE_LATTICE[:, 0]))
        self.assertFalse(torch.equal(scaled[:, 1], BASE_LATTICE[:, 1]))
        self.assertTrue(torch.equal(scaled[:, 2], BASE_LATTICE[:, 2]))

    def test_the_residual_is_bounded_by_the_declared_scale(self):
        extreme = torch.full((1, 4), 50.)
        moved = apply_lattice_residual(BASE_LATTICE, extreme)
        reach = float(LATTICE_RESIDUAL_SCALE)  # tanh saturates, so |value| <= the declared scale
        expected = BASE_LATTICE[:, 0]+reach*(BASE_LATTICE[:, 1]+BASE_LATTICE[:, 2])
        self.assertTrue(torch.allclose(moved[:, 0], expected, atol=1e-7))
        for axis in (1, 2):
            self.assertTrue(torch.allclose(moved[:, axis], BASE_LATTICE[:, axis]*torch.exp(torch.tensor(reach)),
                                          atol=1e-7))

    def test_the_flag_is_a_two_value_join(self):
        self.assertEqual(LATTICE_RESIDUAL_LEVERS, (0, 1))
        summary = {"arguments": {"lattice_residual": 1}, "recovery_contract": {"lattice_residual": 1}}
        self.assertIsNone(require_lattice_residual(1, summary))
        for bad, joined in ((True, summary), (2, summary), (1, {"arguments": {}, "recovery_contract": {}}),
                            (1, {"arguments": {"lattice_residual": 1.0}, "recovery_contract": {}})):
            with self.assertRaises(ValueError):
                require_lattice_residual(bad, joined)

    def test_the_declared_tensor_set_is_exactly_the_model_difference(self):
        published = TetrisStateNet(tiny_config()).state_dict()
        lever_model = TetrisStateNet(tiny_config(lattice_residual=1))
        lever_model.load_state_dict({**published, **{name: torch.zeros_like(value)
                                                     for name, value in lever_model.state_dict().items()
                                                     if name not in published}}, strict=True)
        lever = lever_model.state_dict()
        added = set(lever)-set(published)
        self.assertEqual(added, set(ADDED_TENSORS) if len(ADDED_TENSORS) == 2 else added)
        self.assertEqual(len(added), 2)  # one declared stage in this tiny config
        self.assertTrue(all(name.startswith("lattice_residual.") for name in added))
        self.assertTrue(all(int(torch.count_nonzero(lever[name])) == 0 for name in added))
        self.assertTrue(all(torch.equal(published[name], lever[name]) for name in published))
        full = TetrisStateNet(ModelConfig())
        declared = TetrisStateNet(ModelConfig(lattice_residual=1))
        self.assertEqual(len(set(declared.state_dict())-set(full.state_dict())), len(ADDED_TENSORS))
        self.assertEqual(len(ADDED_TENSORS), 12)
        self.assertEqual(ADDED_PARAMETERS, 120)

    def test_a_zero_residual_reproduces_the_published_forward_exactly(self):
        torch.manual_seed(11)
        published = TetrisStateNet(tiny_config()).eval()
        torch.manual_seed(11)
        lever = TetrisStateNet(tiny_config(lattice_residual=1)).eval()
        lever.load_state_dict({**published.state_dict(),
                               **{name: torch.zeros_like(value) for name, value in lever.state_dict().items()
                                  if name not in published.state_dict()}}, strict=True)
        pixels = torch.rand(1, 3, 64, 64)
        valid = torch.ones(1, 1, 64, 64, dtype=torch.bool)
        with torch.no_grad():
            left, right = published(pixels, valid), lever(pixels, valid)
        self.assertTrue(torch.equal(left["final"]["lattice"], right["final"]["lattice"]))
        self.assertTrue(torch.equal(left["final"]["occupancy"], right["final"]["occupancy"]))
        self.assertTrue(torch.equal(left["final"]["sample_points"], right["final"]["sample_points"]))

    def test_a_trained_residual_changes_the_lattice(self):
        torch.manual_seed(12)
        published = TetrisStateNet(tiny_config()).eval()
        lever = TetrisStateNet(tiny_config(lattice_residual=1)).eval()
        state = dict(published.state_dict())
        state.update({name: value for name, value in lever.state_dict().items()
                      if name not in published.state_dict()})
        state["lattice_residual.0.weight"] = torch.full_like(state["lattice_residual.0.weight"], .5)
        lever.load_state_dict(state, strict=True)
        pixels = torch.rand(1, 3, 64, 64)
        valid = torch.ones(1, 1, 64, 64, dtype=torch.bool)
        with torch.no_grad():
            left, right = published(pixels, valid), lever(pixels, valid)
        self.assertFalse(torch.equal(left["final"]["lattice"], right["final"]["lattice"]))
        self.assertEqual(asdict(tiny_config(lattice_residual=1))["lattice_residual"], 1)


class MatchedStartTests(unittest.TestCase):
    def states(self):
        torch.manual_seed(14)
        baseline = TetrisStateNet(tiny_config()).state_dict()
        lever = {**baseline,
                 **{name: (torch.zeros(4, 4) if name.endswith("weight") else torch.zeros(4))
                    for name in ADDED_TENSORS}}
        return lever, baseline

    def test_the_declared_addition_is_accepted(self):
        lever, baseline = self.states()
        self.assertTrue(require_initial_state(lever, baseline))

    def test_a_moved_shared_tensor_is_refused(self):
        lever, baseline = self.states()
        moved = dict(lever)
        moved["occupancy.weight"] = moved["occupancy.weight"]+1
        with self.assertRaises(ValueError):
            require_initial_state(moved, baseline)

    def test_a_nonzero_or_extra_addition_is_refused(self):
        lever, baseline = self.states()
        nonzero = {**lever, "lattice_residual.0.bias": torch.ones_like(lever["lattice_residual.0.bias"])}
        with self.assertRaises(ValueError):
            require_initial_state(nonzero, baseline)
        extra = {**lever, "lattice_residual.9.weight": torch.zeros_like(lever["lattice_residual.0.weight"])}
        with self.assertRaises(ValueError):
            require_initial_state(extra, baseline)


class LatticeResidualRuleTests(unittest.TestCase):
    def test_the_verdict_needs_all_four_and_the_placement_condition(self):
        baseline, lever, decision = probe()
        self.assertEqual(placement_reduction(baseline, lever), .25)
        self.assertTrue(decision["conditions"]["placement"])
        _, _, tight = probe(placement_improvement=.24)
        self.assertFalse(tight["conditions"]["placement"])

    def test_the_decomposition_is_non_circular(self):
        baseline, lever, decision = probe(placement_improvement=.02)
        self.assertEqual(lattice_residual_decomposition(baseline, lever, decision)["rule"],
                         "read_out_inheritance")
        baseline, lever, decision = probe(attribution=.01, spread=2.0)
        self.assertEqual(lattice_residual_decomposition(baseline, lever, decision)["rule"], "dispersion")
        baseline, lever, decision = probe(attribution=.01, spread=1.0)
        self.assertEqual(lattice_residual_decomposition(baseline, lever, decision)["rule"], "scaling")
        for improvement, spread in ((.0, 5.0), (.5, 5.0), (.02, 1.0), (.3, 1.0)):
            baseline, lever, decision = probe(placement_improvement=improvement, spread=spread)
            self.assertNotIn(lattice_residual_decomposition(baseline, lever, decision)["rule"],
                             ("translation", "scale", "geometry_capacity", "placement_capacity"))

    def test_the_declared_floors_are_the_campaign_criteria(self):
        self.assertEqual(PLACEMENT_IMPROVEMENT, .25)
        self.assertEqual(CURATED_CONTROL["f1"], 0.4146485528647372)


class CuratorJoinTests(unittest.TestCase):
    def summary(self, residual=1, **overrides):
        arguments = {"lattice_residual": residual, "support_anchor_weight": 0.0,
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

    def test_the_join_refuses_a_foreign_configuration_or_envelope(self):
        for weight, job, summary in (
                (1, self.job(status="failed"), self.summary(1)),
                (1, self.job(work_items=2), self.summary(1)),
                (1, self.job(memory_bytes=640*1024**2), self.summary(1)),
                (1, self.job(), self.summary(1, steps=128)),
        ):
            with self.assertRaises(ValueError):
                arm_matches(summary, job, weight)
        for key, value in (("geometry_final_only", 1), ("lattice_support", 1),
                           ("box_extent_weight", 1.0), ("support_anchor_weight", 1.0),
                           ("occupancy_detail", 0)):
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
            require_control(placement, {**occupancy, "f1": .42}, best)

    def test_the_family_is_owned(self):
        self.assertEqual(FAMILY, "m3-occlatres")
        for name in ("m3-occlatres-lever-r1", "m3-occlatres-baseline-r1-sweep-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name))
        self.assertFalse(OWNED_RUN_PATTERN.fullmatch("m3-unknown-lever-r1"))


if __name__ == "__main__":
    unittest.main()
