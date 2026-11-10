import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from fit_gate import FIT_GATE_THRESHOLD, fit_gate_opens
from placement_error import (FIT_ACTIVATION_FLOOR, PLACEMENT_IMPROVEMENT,
                             fit_budget_decomposition, placement_reduction)
from record_fit_budget_lever import (ADDED_PARAMETERS, ADDED_TENSORS, ARMS, CURATED_CONTROL, FAMILY,
                                     activation_engaged, arm_matches, decision_flip_rate,
                                     fit_gate_population, receipts_after, require_control,
                                     require_initial_state)
from record_stream_run import OWNED_RUN_PATTERN
from tetris_vision.model import ModelConfig, TetrisStateNet


def row(exact, tp=500, fp=900, fn=167):
    return {"threshold": 0.0, "exact_nonempty_boards": exact, "exact_empty_boards": 2,
            "true_positive": tp, "false_positive": fp, "false_negative": fn}


def probe(occupancy_f1=.75, ranking="informative", attribution=.08, placement_improvement=.30,
          spread=1.0, activation=True, fit_gate=True):
    placement = {"mean_abs_offset_cells": 1.5, "signed_column_offset_cells": .5,
                 "signed_row_offset_cells": -.5, "mean_scale_ratio_u": 1.05,
                 "mean_scale_ratio_v": 1.02, "board_error_spread_cells": spread}
    baseline = {**placement, "mean_abs_offset_cells": 2.0}
    decision = {"conditions": {"per_cell_f1": occupancy_f1 >= .7, "ranking": ranking == "informative",
                              "attribution": attribution >= .05,
                              "placement": placement_improvement >= PLACEMENT_IMPROVEMENT,
                              "fit_gate": fit_gate, "activation": activation},
                "placement_improvement": placement_improvement,
                "placement_floor": PLACEMENT_IMPROVEMENT}
    return baseline, placement, decision


class FitGateCriterionTests(unittest.TestCase):
    def test_the_gate_needs_three_exact_nonempty_boards(self):
        self.assertEqual(FIT_GATE_THRESHOLD, 3)
        for exact, expected in ((0, False), (2, False), (3, True), (5, True)):
            self.assertIs(fit_gate_opens(fit_gate_population(row(exact))), expected)

    def test_empty_board_success_can_never_open_the_gate(self):
        population = fit_gate_population(row(0))
        population["nonempty"]["exact_empty_boards"] = 9
        self.assertFalse(fit_gate_opens(population))
        with self.assertRaises(ValueError):
            fit_gate_population({"threshold": 0.0})

    def test_the_criterion_is_read_from_the_standing_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(Path(directory).exists())


class FitActivationTests(unittest.TestCase):
    def test_the_floor_is_a_decision_flip_rate(self):
        self.assertEqual(FIT_ACTIVATION_FLOOR, .05)
        self.assertTrue(activation_engaged(.05))
        self.assertTrue(activation_engaged(.06))
        self.assertFalse(activation_engaged(.049))
        self.assertFalse(activation_engaged(0.0))

    def test_the_flip_rate_is_measured_on_known_cells_only(self):
        lever = [{"sample_id": "a-1", "known": [True, True, False, True],
                  "logits": [1., -1., 1., 1.]}]
        baseline = [{"sample_id": "a-1", "known": [True, True, False, True],
                     "logits": [-1., -1., -1., 1.]}]
        rate, flips, known = decision_flip_rate(lever, baseline)
        self.assertEqual((flips, known), (1, 3))
        self.assertAlmostEqual(rate, 1/3, places=6)
        with self.assertRaises(ValueError):
            decision_flip_rate(lever, [{"sample_id": "b-2", "known": [True], "logits": [0.]}])
        with self.assertRaises(ValueError):
            decision_flip_rate([{"sample_id": "a-1", "known": [False], "logits": [1.]}],
                               [{"sample_id": "a-1", "known": [False], "logits": [-1.]}])


class FitRuleTests(unittest.TestCase):
    def test_the_rule_is_non_circular_and_activation_first(self):
        baseline, lever, decision = probe(activation=False, fit_gate=True)
        self.assertEqual(fit_budget_decomposition(baseline, lever, decision)["rule"], "under_trained")
        baseline, lever, decision = probe(activation=True, fit_gate=False)
        self.assertEqual(fit_budget_decomposition(baseline, lever, decision)["rule"], "capacity_limit")
        baseline, lever, decision = probe(activation=True, fit_gate=True)
        self.assertEqual(fit_budget_decomposition(baseline, lever, decision)["rule"], "fit_opens")
        spent = ("translation", "scale", "geometry_capacity", "placement_capacity", "lattice_residual",
                 "lattice_free", "board_refinement")
        for activation in (True, False):
            for gate in (True, False):
                for improvement in (-.4, 0.0, .3, 5.0):
                    baseline, lever, decision = probe(activation=activation, fit_gate=gate,
                                                      placement_improvement=improvement)
                    self.assertNotIn(fit_budget_decomposition(baseline, lever, decision)["rule"], spent)

    def test_placement_reduction_is_the_declared_quantity(self):
        baseline, lever, _ = probe()
        self.assertEqual(placement_reduction(baseline, lever), .25)


class CuratorJoinTests(unittest.TestCase):
    def summary(self, steps=256, **overrides):
        arguments = {"steps": steps, "lattice_board_refine": 0, "lattice_residual": 0,
                     "lattice_support": 0, "geometry_final_only": 0, "occupancy_lattice_free": 0,
                     "support_anchor_weight": 0.0, "box_extent_weight": 0.0, "occupancy_detail": 1,
                     "occupancy_detail_stencil": 0, "occupancy_context": 0, "seed": 20261026,
                     "capture_max_side": 512, "occupancy_factor": 4, "occupancy_positive_weight": 4}
        summary = {"status": "complete", "steps": steps, "requested_steps": steps,
                   "checkpoint_round_trip_exact": True, "arguments": dict(arguments),
                   "recovery_contract": dict(arguments)}
        summary.update(overrides)
        return summary

    def job(self, steps=256, **overrides):
        job = {"status": "passed", "exit_code": 0, "work_items": steps, "memory_bytes": 1152*1024**2}
        job.update(overrides)
        return job

    def test_each_arm_is_admitted_only_for_its_own_budget(self):
        self.assertEqual(ARMS, {"lever": 1024, "baseline": 256})
        self.assertTrue(arm_matches(self.summary(1024), self.job(1024), 1024))
        self.assertTrue(arm_matches(self.summary(256), self.job(256), 256))
        for declared, summary, job in ((1024, self.summary(256), self.job(1024)),
                                       (256, self.summary(256), self.job(1024))):
            with self.assertRaises(ValueError):
                arm_matches(summary, job, declared)

    def test_the_join_refuses_foreign_configuration_or_envelope(self):
        for job, summary in ((self.job(256, status="failed"), self.summary(256)),
                             (self.job(256, memory_bytes=640*1024**2), self.summary(256)),
                             (self.job(256), self.summary(256, checkpoint_round_trip_exact=False))):
            with self.assertRaises(ValueError):
                arm_matches(summary, job, 256)
        for key, value in (("lattice_board_refine", 1), ("lattice_residual", 1),
                           ("occupancy_lattice_free", 1), ("geometry_final_only", 1),
                           ("lattice_support", 1), ("box_extent_weight", 1.0),
                           ("support_anchor_weight", 1.0)):
            foreign = self.summary(256)
            foreign["arguments"][key] = value
            foreign["recovery_contract"][key] = value
            with self.assertRaises(ValueError):
                arm_matches(foreign, self.job(256), 256)

    def test_the_lever_adds_no_parameter_and_the_start_must_be_bitwise(self):
        self.assertEqual(ADDED_PARAMETERS, 0)
        self.assertEqual(set(ADDED_TENSORS), set())
        torch.manual_seed(51)
        model = TetrisStateNet(ModelConfig(widths=(8, 8, 8, 8), dimension=16, heads=1, blocks=1,
                                           refinements=0))
        state = model.state_dict()
        self.assertTrue(require_initial_state(dict(state), dict(state)))
        changed = {name: value.clone() for name, value in state.items()}
        changed["occupancy.weight"].add_(1e-6)
        with self.assertRaises(ValueError):
            require_initial_state(changed, dict(state))

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
            require_control({**placement, "board_error_spread_cells": 4.5}, occupancy, best)
        with self.assertRaises(ValueError):
            require_control(placement, {**occupancy, "f1": .42}, best)

    def test_the_family_is_owned_without_dropping_earlier_families(self):
        self.assertEqual(FAMILY, "m3-occfit")
        for name in ("m3-occfit-lever-r1", "m3-occfit-baseline-r1-sweep-r1",
                     "m3-occbref-lever-r1", "m3-occlatfree-lever-r1", "m3-occlatres-lever-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        self.assertFalse(OWNED_RUN_PATTERN.fullmatch("m3-unknown-lever-r1"))


if __name__ == "__main__":
    unittest.main()
