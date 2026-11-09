"""The conditioning record re-derives from the committed widened bytes."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

import board_conditioning as rule
import record_board_conditioning as recorder

EVIDENCE = ROOT/"docs/evidence/m3-occupancy-conditioning-r1/scope.json"


class BoardConditioningArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        cls.widened = json.loads((recorder.WIDENED/"scope.json").read_text(encoding="utf-8"))
        cls.sweep = recorder.read(recorder.INPUTS["sweep"])
        cls.replay = recorder.read(recorder.INPUTS["replay"])
        cls.measured = rule.measure(cls.sweep)

    def test_the_inputs_are_the_published_bytes(self):
        for name, expected in self.record["input_sha256"].items():
            self.assertEqual(recorder.digest(recorder.INPUTS[name]), expected, name)
        for name, expected in self.widened["artifact_sha256"].items():
            self.assertEqual(recorder.digest(recorder.WIDENED/name), expected, name)

    def test_the_classification_re_derives(self):
        self.assertEqual(self.record["classification"], self.measured["classification"])
        classification = self.record["classification"]
        self.assertEqual(classification["verdict"], "representation-limited")
        self.assertEqual((classification["reaching"], classification["boards"]), (0, 28))
        self.assertEqual(classification["oracle_threshold_spread"], 0.0)
        self.assertEqual(classification["reaching_thresholds"], [])
        self.assertEqual(classification["bar"], 0.70)

    def test_the_board_table_re_derives(self):
        self.assertEqual(self.record["boards"], self.measured["boards"])
        self.assertEqual(len(self.record["boards"]), 28)
        for row in self.record["boards"]:
            self.assertLess(row["oracle_f1"], 0.70)
            self.assertNotIn(row["oracle_threshold"], ())

    def test_the_oracle_and_auc_blocks_re_derive(self):
        self.assertEqual(self.record["oracle"]["pooled"], self.measured["oracle_pooled"])
        self.assertEqual(self.record["oracle"]["pooled_f1_median"], self.measured["pooled_f1_median"])
        self.assertEqual(self.record["auc"]["per_board"], self.measured["auc_distribution"])
        self.assertAlmostEqual(self.record["auc"]["pooled"],
                               self.widened["occupancy"]["ranking"]["auc"], places=12)

    def test_the_population_is_the_widened_deciding_one(self):
        population = self.record["population"]
        self.assertEqual((population["boards"], population["occupied_cells"]), (28, 667))
        self.assertEqual((population["known_cells"], population["ancestry_groups"]), (5600, 12))
        self.assertEqual(population["split"], "training")

    def test_the_density_block_re_derives(self):
        expected = recorder.density_summary(self.measured["boards"],
                                            recorder.density_of(self.replay))
        self.assertEqual(self.record["density"], expected)
        if self.record["density"]["available"]:
            self.assertGreater(len(self.record["density"]["labels"]), 1)

    def test_nothing_that_decides_a_verdict_was_touched(self):
        self.assertEqual(self.record["unchanged"]["thresholds"], self.widened["unchanged_thresholds"])
        self.assertEqual(self.record["unchanged"]["ranking_verdict"], "uninformative")
        self.assertEqual(self.record["unchanged"]["point_estimates"], True)
        self.assertEqual(self.record["status"], "passed")

    def test_the_decomposition_would_report_the_other_branches_when_they_hold(self):
        reaching = [{"f1": 0.8, "threshold": value} for value in (-2.0, 0.0, 1.0, 2.0)]
        self.assertEqual(rule.classify(reaching)["verdict"], "conditioned-boundary")
        agreeing = [{"f1": 0.8, "threshold": value} for value in (-1.0, -0.9, -0.5, 0.0)]
        self.assertEqual(rule.classify(agreeing)["verdict"], "inconsistent")


if __name__ == "__main__":
    unittest.main()
