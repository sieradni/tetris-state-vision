import tempfile
import unittest
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from placement_error import capacity_decomposition, placement_reduction
from record_capacity_lever import (ARMS, CAPACITY_MOVED_FLOOR, CURATED_CONTROL, FAMILY,
                                   PLACEMENT_IMPROVEMENT, arm_matches, receipts_after,
                                   require_control)
from record_stream_run import OWNED_RUN_PATTERN
from tetris_vision.losses import (SUPPORT_ANCHORS, SUPPORT_ANCHOR_WEIGHTS, require_support_anchor_weight,
                                  support_anchor_error, support_anchor_term)

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


class SupportAnchorRuleTests(unittest.TestCase):
    def test_the_declared_weight_is_a_two_value_join(self):
        self.assertEqual(SUPPORT_ANCHOR_WEIGHTS, (0.0, 1.0))
        summary = {"arguments": {"support_anchor_weight": 1.0},
                   "recovery_contract": {"support_anchor_weight": 1.0}}
        self.assertIsNone(require_support_anchor_weight(1.0, summary))
        for bad, joined in ((1, summary), (2.0, summary), (1.0, {"arguments": {}, "recovery_contract": {}}),
                            (1.0, {"arguments": {"support_anchor_weight": 1}, "recovery_contract": {}})):
            with self.assertRaises(ValueError):
                require_support_anchor_weight(bad, joined)

    def test_the_anchor_set_is_the_eight_boundary_points(self):
        self.assertEqual(len(SUPPORT_ANCHORS), 8)
        self.assertEqual(set(SUPPORT_ANCHORS), {(0., 0.), (5., 0.), (10., 0.), (0., 10.),
                                                (10., 10.), (0., 20.), (5., 20.), (10., 20.)})
        for column, row in SUPPORT_ANCHORS:
            self.assertTrue(column in (0., 5., 10.) and row in (0., 10., 20.))
            self.assertTrue(column in (0., 10.) or row in (0., 20.))

    def test_the_anchor_error_is_zero_only_when_the_box_is_exact(self):
        error = support_anchor_error(BASE_LATTICE, BASE_LATTICE.clone())
        self.assertEqual(tuple(error.shape), (1, 8, 2))
        self.assertEqual(float(error.abs().max()), 0.0)
        shifted = BASE_LATTICE.clone()
        shifted[:, 0, 0] += .02
        self.assertGreater(float(support_anchor_error(shifted, BASE_LATTICE).abs().mean()), 0)
        wider = BASE_LATTICE.clone()
        wider[:, 1, 0] *= 1.2
        self.assertGreater(float(support_anchor_error(wider, BASE_LATTICE).abs().mean()), 0)

    def test_the_term_weights_only_registered_boards(self):
        registered = torch.tensor([True])
        self.assertEqual(float(support_anchor_term(BASE_LATTICE, BASE_LATTICE.clone(), registered)), 0.0)
        shifted = BASE_LATTICE.clone()
        shifted[:, 0, 1] += .03
        self.assertGreater(float(support_anchor_term(shifted, BASE_LATTICE, registered)), 0)
        self.assertEqual(float(support_anchor_term(shifted, BASE_LATTICE, torch.tensor([False]))), 0.0)


class CapacityVerdictTests(unittest.TestCase):
    def test_the_verdict_needs_all_four_and_the_placement_condition(self):
        baseline, lever, decision = probe()
        self.assertEqual(placement_reduction(baseline, lever), .25)
        self.assertTrue(decision["conditions"]["placement"])
        _, _, tight = probe(placement_improvement=.24)
        self.assertFalse(tight["conditions"]["placement"])

    def test_the_decomposition_is_non_circular(self):
        baseline, lever, decision = probe(placement_improvement=.02)
        self.assertEqual(capacity_decomposition(baseline, lever, decision)["rule"], "placement_capacity")
        baseline, lever, decision = probe(occupancy_f1=.5, ranking="uninformative")
        self.assertEqual(capacity_decomposition(baseline, lever, decision)["rule"],
                         "read_out_inheritance")
        baseline, lever, decision = probe(attribution=.01, spread=2.0)
        self.assertEqual(capacity_decomposition(baseline, lever, decision)["rule"], "dispersion")
        baseline, lever, decision = probe(attribution=.01, spread=1.0)
        rule = capacity_decomposition(baseline, lever, decision)["rule"]
        self.assertEqual(rule, "scaling")
        # None of the three spent levers can be returned again.
        for improvement, spread in ((.0, 5.0), (.5, 5.0), (.02, 1.0)):
            baseline, lever, decision = probe(placement_improvement=improvement, spread=spread)
            self.assertNotIn(capacity_decomposition(baseline, lever, decision)["rule"],
                             ("translation", "scale", "geometry_capacity"))

    def test_the_improvement_floor_is_the_geometry_pass_placement_condition(self):
        self.assertEqual(PLACEMENT_IMPROVEMENT, .25)
        self.assertEqual(CAPACITY_MOVED_FLOOR, .05)
        self.assertEqual(CURATED_CONTROL["f1"], 0.4146485528647372)


class CuratorJoinTests(unittest.TestCase):
    def summary(self, weight=1.0, **overrides):
        arguments = {"support_anchor_weight": weight, "box_extent_weight": 0.0,
                     "lattice_support": 0, "geometry_final_only": 0, "occupancy_detail": 1,
                     "occupancy_detail_stencil": 0, "occupancy_context": 0, "steps": 256,
                     "seed": 20261026, "capture_max_side": 512, "occupancy_factor": 4,
                     "occupancy_positive_weight": 4}
        summary = {"status": "complete", "steps": 256, "requested_steps": 256,
                   "checkpoint_round_trip_exact": True, "arguments": dict(arguments),
                   "recovery_contract": dict(arguments)}
        summary.update(overrides)
        return summary

    def job(self, **overrides):
        job = {"status": "passed", "exit_code": 0, "work_items": 256, "memory_bytes": 1152*1024**2}
        job.update(overrides)
        return job

    def test_both_arms_are_admitted_only_for_their_own_weight(self):
        self.assertTrue(arm_matches(self.summary(1.0), self.job(), 1.0))
        self.assertTrue(arm_matches(self.summary(0.0), self.job(), 0.0))
        self.assertEqual(ARMS, {"lever": 1.0, "baseline": 0.0})
        for bad_weight, joined in ((1.0, 0.0), (0.0, 1.0)):
            with self.assertRaises(ValueError):
                arm_matches(self.summary(bad_weight), self.job(), joined)

    def test_the_join_refuses_a_foreign_configuration_or_envelope(self):
        for weight, job, summary in (
                (1.0, self.job(status="failed"), self.summary(1.0)),
                (1.0, self.job(work_items=2), self.summary(1.0)),
                (1.0, self.job(memory_bytes=640*1024**2), self.summary(1.0)),
                (1.0, self.job(), self.summary(1.0, steps=128)),
        ):
            with self.assertRaises(ValueError):
                arm_matches(summary, job, weight)
        foreign = self.summary(1.0)
        foreign["arguments"]["geometry_final_only"] = 1
        with self.assertRaises(ValueError):
            arm_matches(foreign, self.job(), 1.0)
        extent = self.summary(1.0)
        extent["arguments"]["box_extent_weight"] = 1.0
        with self.assertRaises(ValueError):
            arm_matches(extent, self.job(), 1.0)

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
        self.assertEqual(FAMILY, "m3-occgeocap")
        for name in ("m3-occgeocap-lever-r1", "m3-occgeocap-baseline-r1-sweep-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name))
        self.assertFalse(OWNED_RUN_PATTERN.fullmatch("m3-unknown-lever-r1"))


if __name__ == "__main__":
    unittest.main()
