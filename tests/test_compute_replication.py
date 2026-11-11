"""The declared replicate block: its rule, its pairing, its guards and its declared constants."""
import copy
import os
import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]

import compute_replication as block
import run_compute_replication as driver


def folds(arm_counts):
    return [{"group_id": f"g{index}", "true_positive": tp, "false_positive": fp,
             "false_negative": fn} for index, (tp, fp, fn) in enumerate(arm_counts)]


class Rule(unittest.TestCase):
    def test_a_clear_point_resolvable_interval_and_unanimous_seeds_confirm(self):
        verdict = block.replication_verdict({1: 0.80, 2: 0.81, 3: 0.79},
                                            {1: 0.10, 2: 0.02, 3: 0.08}, 0.067,
                                            {"low": 0.004, "high": 0.13})
        self.assertEqual(verdict["label"], "confirmed")

    def test_each_failure_reason_is_named_and_never_becomes_a_pass(self):
        point_low = block.replication_verdict({1: 0.8, 2: 0.8, 3: 0.8},
                                              {1: 0.04, 2: 0.03, 3: 0.02}, 0.03,
                                              {"low": 0.001, "high": 0.06})
        self.assertEqual(point_low["label"], "unconfirmed")
        self.assertIn("point_below_floor", point_low["reason"])
        touching = block.replication_verdict({1: 0.8, 2: 0.8, 3: 0.8},
                                             {1: 0.06, 2: 0.05, 3: 0.07}, 0.06,
                                             {"low": 0.0, "high": 0.12})
        self.assertEqual(touching["label"], "unconfirmed")
        self.assertIn("interval_touches_zero", touching["reason"])
        reversing = block.replication_verdict({1: 0.8, 2: 0.8, 3: 0.8},
                                              {1: 0.12, 2: -0.01, 3: 0.09}, 0.07,
                                              {"low": 0.01, "high": 0.15})
        self.assertEqual(reversing["label"], "unconfirmed")
        self.assertIn("seed_reverses", reversing["reason"])

    def test_a_control_below_the_auc_floor_voids_the_block(self):
        verdict = block.replication_verdict({1: 0.8, 2: 0.55, 3: 0.8},
                                            {1: 0.2, 2: 0.2, 3: 0.2}, 0.2,
                                            {"low": 0.1, "high": 0.3})
        self.assertEqual(verdict["label"], "floor_limited")
        self.assertEqual(verdict["seeds"], [2])

    def test_the_rule_refuses_unpaired_or_malformed_inputs(self):
        with self.assertRaises(ValueError):
            block.replication_verdict({1: 0.8}, {2: 0.1}, 0.1, {"low": 0.0, "high": 0.2})
        with self.assertRaises(ValueError):
            block.replication_verdict({1: 0.8}, {}, 0.1, {"low": 0.0, "high": 0.2})
        with self.assertRaises(ValueError):
            block.replication_verdict({1: 0.8}, {1: 0.1}, 0.1, {"high": 0.2})


class Pairing(unittest.TestCase):
    def clusters(self):
        return [{"seed": 1, "group_id": "g0", "base128": (5, 5, 1), "budget192": (6, 4, 0),
                 "cosine128": (5, 5, 1)},
                {"seed": 1, "group_id": "g1", "base128": (4, 6, 2), "budget192": (6, 4, 0),
                 "cosine128": (4, 6, 2)},
                {"seed": 2, "group_id": "g0", "base128": (3, 7, 3), "budget192": (5, 5, 1),
                 "cosine128": (3, 7, 3)},
                {"seed": 2, "group_id": "g1", "base128": (2, 8, 4), "budget192": (4, 6, 2),
                 "cosine128": (2, 8, 4)},
                {"seed": 3, "group_id": "g0", "base128": (5, 5, 1), "budget192": (6, 4, 0),
                 "cosine128": (5, 5, 1)},
                {"seed": 3, "group_id": "g1", "base128": (4, 6, 2), "budget192": (6, 4, 0),
                 "cosine128": (4, 6, 2)}]

    def test_pooled_f1_uses_the_block_wide_counts(self):
        clusters = self.clusters()
        self.assertAlmostEqual(block.pooled_f1(clusters, "base128"),
                               block.f1_from_counts(23, 37, 13), places=12)
        self.assertGreater(block.pooled_f1(clusters, "budget192"),
                           block.pooled_f1(clusters, "base128"))

    def test_a_constant_improvement_gives_an_interval_above_zero(self):
        interval = block.paired_delta_interval(self.clusters(), "budget192")
        self.assertGreater(interval["low"], 0.0)
        self.assertTrue(interval["excludes_zero"])

    def test_identical_arms_give_a_degenerate_zero_interval(self):
        interval = block.paired_delta_interval(self.clusters(), "cosine128")
        self.assertEqual((interval["low"], interval["high"]), (0.0, 0.0))
        self.assertFalse(interval["excludes_zero"])

    def test_clusters_pair_every_arm_on_the_same_group(self):
        reads = {}
        for seed in block.SEEDS:
            for arm in block.ARMS:
                reads[(seed, arm)] = folds([(5, 5, 1), (4, 6, 2)] +
                                           [(3, 7, 3)]*(block.DECIDING_GROUPS-2))
        clusters = block.block_clusters(reads)
        self.assertEqual(len(clusters), len(block.SEEDS)*block.DECIDING_GROUPS)
        self.assertEqual({cluster["seed"] for cluster in clusters}, set(block.SEEDS))
        self.assertEqual(set(clusters[0]), {"seed", "group_id", *block.ARMS})
        mismatched = dict(reads)
        shifted = folds([(5, 5, 1)] + [(4, 6, 2)]*(block.DECIDING_GROUPS-1))
        shifted[-1]["group_id"] = "g-other"
        mismatched[(block.SEEDS[0], "base128")] = shifted
        with self.assertRaises(ValueError):
            block.block_clusters(mismatched)


class Guards(unittest.TestCase):
    def test_the_declared_block_is_pinned(self):
        self.assertEqual(block.SEEDS, (20261101, 20261102, 20261103))
        self.assertEqual(block.ARMS, {"base128": 128, "budget192": 192, "cosine128": 128})
        self.assertEqual(block.ARMS["budget192"]/block.ARMS["base128"], 1.5)
        self.assertAlmostEqual(block.FLOOR, 0.05, places=12)
        self.assertAlmostEqual(block.AUC_FLOOR, 0.70, places=12)
        self.assertEqual((block.DECIDING_CASES, block.DECIDING_GROUPS), (36, 12))

    def test_arms_that_scored_different_boards_are_refused(self):
        def case(identifier, labels):
            return {"sample_id": identifier, "occupied": labels}
        base = {"cases": [case("a", [1, 0]), case("b", [0, 0])]}
        same = {"cases": [copy.deepcopy(row) for row in base["cases"]]}
        block.require_one_population({"base128": base, "budget192": same, "cosine128": same}, 1)
        relabelled = {"cases": [case("a", [0, 1]), case("b", [0, 0])]}
        with self.assertRaises(ValueError):
            block.require_one_population({"base128": base, "budget192": relabelled,
                                          "cosine128": same}, 1)


class SeedLock(unittest.TestCase):
    """V-093: one block per seed, enforced rather than assumed."""

    def test_a_live_holder_refuses_a_second_block_and_a_dead_one_is_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/".repl-block.lock"
            self.assertEqual(driver.acquire_lock(path, os.getpid()), "acquired")
            with self.assertRaises(SystemExit):
                driver.acquire_lock(path, os.getpid()+1)
            self.assertEqual(path.read_text().strip(), str(os.getpid()))
            path.write_text("999999999")
            self.assertEqual(driver.acquire_lock(path, os.getpid()), "replaced_stale")
            self.assertEqual(path.read_text().strip(), str(os.getpid()))


if __name__ == "__main__":
    unittest.main()
