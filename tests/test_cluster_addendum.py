"""The clustering addendum's numbers re-derive from the committed artifacts, not from prose."""
from bisect import bisect_left, bisect_right
import json
from math import sqrt
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

import cluster_interval
import record_cluster_addendum as addendum

AUC_DRAWS = cluster_interval.AUC_DRAWS
COUNT_DRAWS = cluster_interval.COUNT_DRAWS
ADDENDUM = ROOT/"docs/evidence/m3-widened-deciding-population-clustering-r1/scope.json"


class ClusterAddendumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = json.loads((addendum.EVIDENCE/"scope.json").read_text())
        cls.addendum = json.loads(ADDENDUM.read_text())
        cls.sweep = addendum.read(addendum.INPUTS["sweep"])
        cls.boards = addendum.boards_of(cls.sweep)
        cls.cells = addendum.cells_of(cls.boards)

    def test_the_measured_inputs_are_the_published_bytes(self):
        for name, expected in self.addendum["input_sha256"].items():
            self.assertEqual(addendum.digest(addendum.INPUTS[name]), expected, name)
        for name, expected in self.scope["artifact_sha256"].items():
            self.assertEqual(addendum.digest(addendum.EVIDENCE/name), expected, name)

    def test_the_population_is_the_widened_deciding_one(self):
        population = self.addendum["population"]
        self.assertEqual((population["boards"], population["ancestry_groups"]), (28, 12))
        self.assertEqual((population["occupied_cells"], population["empty_cells"]), (667, 4933))
        self.assertEqual(population["known_cells"], len(self.cells))

    def test_the_auc_point_estimate_re_derives_independently(self):
        positive = [cell["logit"] for cell in self.cells if cell["label"]]
        negative = sorted(cell["logit"] for cell in self.cells if not cell["label"])
        score = 0.0
        for value in positive:
            low, high = bisect_left(negative, value), bisect_right(negative, value)
            score += low+0.5*(high-low)
        self.assertAlmostEqual(score/(len(positive)*len(negative)),
                               self.addendum["occupancy"]["ranking_auc"]["point"], places=9)
        self.assertAlmostEqual(self.addendum["occupancy"]["ranking_auc"]["point"],
                               self.scope["occupancy"]["ranking"]["auc"], places=12)

    def test_the_design_effect_is_re_derived_from_the_cell_errors(self):
        for key, threshold in (("operating_point", 0.0), ("best_point", -0.9)):
            published = self.addendum["clustering"][key]
            self.assertEqual(published, addendum.icc_block(self.cells, threshold), key)
            for level in ("board", "group"):
                block = published[level]
                self.assertGreaterEqual(block["design_effect"], 1.0)
                self.assertLessEqual(block["effective_cells"], published["cells"])
                self.assertLessEqual(block["effective_occupied"], published["occupied_cells"])
                self.assertGreater(block["clusters"], 1)

    def test_the_auc_intervals_re_derive_from_the_committed_cases(self):
        ranking = self.addendum["occupancy"]["ranking_auc"]
        self.assertEqual(ranking["cell_level"], self.scope["occupancy"]["ranking"]["auc_interval"])
        for level in ("board", "group"):
            recomputed = cluster_interval.cluster_bootstrap(
                addendum.auc_statistic, cluster_interval.clusters_of(self.boards, level),
                ("auc",), draws=AUC_DRAWS)["auc"]
            self.assertEqual(ranking[level+"_cluster"], recomputed, level)
            self.assertLess(recomputed["low"], recomputed["high"])
            self.assertGreaterEqual(recomputed["high"], round(ranking["point"], 6)-1e-6)
            self.assertLessEqual(recomputed["low"], round(ranking["point"], 6)+1e-6)

    def test_the_count_intervals_re_derive_at_both_operating_points(self):
        for key, threshold in (("operating_point", 0.0), ("best_point", -0.9)):
            block = self.addendum["occupancy"][key]
            statistic = addendum.counts_statistic(threshold)
            for level in ("board", "group"):
                recomputed = cluster_interval.cluster_bootstrap(
                    statistic, cluster_interval.clusters_of(self.boards, level),
                    ("precision", "recall", "f1"), draws=COUNT_DRAWS)
                self.assertEqual(block[level+"_cluster"], recomputed, (key, level))
            self.assertEqual(block["cell_level"]["precision"]["successes"], block["true_positive"])
            self.assertEqual(block["cell_level"]["recall"]["trials"],
                             block["true_positive"]+block["false_negative"])

    def test_the_per_proportion_design_effects_are_recorded(self):
        for key in ("operating_point", "best_point"):
            block = self.addendum["occupancy"][key]
            for name in ("precision", "recall"):
                for level in ("board", "group"):
                    record = block["standard_error_ratio"][name][level]
                    self.assertGreater(record["binomial_standard_error"], 0)
                    self.assertGreater(record["design_effect"], 0)
                    # The recorded standard errors are rounded to six decimals while the design
                    # effect is squared from the unrounded ones, so the ratio agrees to relative
                    # precision rather than absolutely.
                    expected = (block[level+"_cluster"][name]["standard_error"]/record["binomial_standard_error"])**2
                    self.assertAlmostEqual(record["design_effect"], expected,
                                           delta=max(1e-3, abs(expected)*2e-3))
        # The measured asymmetry: precision is the clustered quantity, recall is not.
        operating = self.addendum["occupancy"]["operating_point"]["standard_error_ratio"]
        self.assertGreater(operating["precision"]["group"]["design_effect"], 10)
        self.assertLess(operating["recall"]["group"]["design_effect"], 2)

    def test_the_identity_rates_re_derive_and_the_degenerate_one_is_declared(self):
        identity = addendum.read(addendum.INPUTS["identity"])
        block = addendum.identity_block(identity, self.scope)
        self.assertEqual(block, self.addendum["identity"])
        self.assertEqual(block["queue"]["design_effect"], 1.0)
        self.assertEqual(block["queue"]["corrected"]["high"], block["queue"]["cell_level"]["high"])
        self.assertEqual((block["hold"]["successes"], block["hold"]["trials"]), (15, 33))
        self.assertGreater(block["hold"]["design_effect_group"], 1.0)
        self.assertGreater(block["hold"]["corrected"]["high"]-block["hold"]["corrected"]["low"],
                           block["hold"]["cell_level"]["high"]-block["hold"]["cell_level"]["low"])

    def test_the_quantifiers_re_derive_from_the_stored_intervals(self):
        quantifiers = self.addendum["quantifiers"]
        for label, key in (("0.0", "operating_point"), ("-0.90", "best_point")):
            block = self.addendum["occupancy"][key]
            for level in ("board", "group"):
                record = quantifiers["per_point"][f"{label}/{level}"]
                design = block["standard_error_ratio"]["precision"][level]["design_effect"]
                clustered = block[level+"_cluster"]["precision"]
                cell = block["cell_level"]["precision"]
                self.assertAlmostEqual(record["precision_design_effect"], design, places=6)
                self.assertAlmostEqual(record["precision_standard_error_inflation"], sqrt(design), places=4)
                self.assertAlmostEqual(record["precision_width_ratio"],
                                       (clustered["high"]-clustered["low"]) / (cell["high"]-cell["low"]),
                                       places=4)
                self.assertAlmostEqual(record["precision_standard_error_inflation"],
                                       record["precision_width_ratio"], delta=0.25)
        self.assertEqual(quantifiers["precision_design_effect_range"]["low"],
                         min(block["standard_error_ratio"]["precision"][level]["design_effect"]
                             for key in ("operating_point", "best_point")
                             for level in ("board", "group")
                             for block in [self.addendum["occupancy"][key]]))
        self.assertAlmostEqual(quantifiers["precision_standard_error_inflation_range"]["low"],
                               sqrt(14.436382), places=3)
        self.assertAlmostEqual(quantifiers["precision_standard_error_inflation_range"]["high"],
                               sqrt(76.018109), places=3)
        # The corrected range must be the root of the design effects, not the design effects' ends.
        self.assertAlmostEqual(quantifiers["precision_standard_error_inflation_range"]["high"], 8.72, delta=0.01)
        self.assertLess(quantifiers["precision_standard_error_inflation_range"]["low"], 3.81)

    def test_the_published_qualifiers_still_match_the_measurement(self):
        report = addendum.check_published(self.addendum)
        self.assertEqual(report["documents"], 4)
        self.assertGreaterEqual(report["strings"], 60)
        mutated = json.loads(json.dumps(self.addendum))
        mutated["quantifiers"]["precision_design_effect_range"]["high"] = 6.7
        with self.assertRaises(ValueError):
            addendum.check_published(mutated)
        drifted = json.loads(json.dumps(self.addendum))
        drifted["quantifiers"]["precision_standard_error_inflation_range"]["high"] = 6.7
        with self.assertRaises(ValueError):
            addendum.check_published(drifted)

    def test_the_old_wrong_multiplier_is_gone_from_the_record(self):
        for name in ("docs/M3_WIDENED_DECIDING_POPULATION.md", "docs/CONTINUATION_STATE.md",
                     "docs/BUG_LOG.md"):
            text = (ROOT/name).read_text(encoding="utf-8")
            self.assertNotIn("over-confident by 3.4", text.replace("\r\n", "\n"), name)
            self.assertNotIn("widens precision by 3.4x-6.7x", text.replace("\r\n", "\n"), name)

    def test_nothing_that_decides_a_verdict_was_changed(self):
        self.assertEqual(self.addendum["unchanged"]["thresholds"], self.scope["unchanged_thresholds"])
        self.assertEqual(self.addendum["unchanged"]["thresholds"]["ranking_auc"], 0.9)
        self.assertEqual(self.addendum["unchanged"]["thresholds"]["per_cell_f1"], 0.7)
        self.assertEqual(self.addendum["unchanged"]["thresholds"]["fit_gate_exact_nonempty_boards"], 3)
        self.assertEqual(self.addendum["unchanged"]["thresholds"]["sweep_grid_points"], 81)
        self.assertEqual(self.addendum["unchanged"]["point_estimates"], True)
        self.assertEqual(self.addendum["status"], "passed")

    def test_the_corrected_intervals_still_sit_below_every_bar(self):
        ranking = self.addendum["occupancy"]["ranking_auc"]
        for level in ("board", "group"):
            self.assertLess(ranking[level+"_cluster"]["high"], ranking["bar"])
        for key in ("operating_point", "best_point"):
            block = self.addendum["occupancy"][key]
            for level in ("board", "group"):
                self.assertLess(block[level+"_cluster"]["f1"]["high"], 0.70, (key, level))


if __name__ == "__main__":
    unittest.main()
