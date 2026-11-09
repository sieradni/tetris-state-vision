"""The representation-lever record re-derives from its committed receipts."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

import record_representation_lever as curator
from occupancy_sweep import sweep_verdict

EVIDENCE = ROOT/"docs/evidence/m3-occupancy-representation-r1/scope.json"


class RepresentationLeverRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        cls.sweeps = {}
        for arm in curator.ARMS:
            name = f"m3-occctx-{arm}-r1-sweep-r1"
            cls.sweeps[arm] = json.loads((ROOT/"runs"/name/"result.json").read_text(encoding="utf-8"))

    def training(self, arm):
        return [row for row in self.sweeps[arm]["cases"] if row["split"] == "training"]

    def test_the_record_is_a_passed_measurement_of_the_declared_instrument(self):
        self.assertEqual(self.record["status"], "passed")
        self.assertEqual(self.record["instrument"], "m3-occupancy-representation-r1")
        self.assertEqual(self.record["predeclaration"], "docs/M3_OCCUPANCY_REPRESENTATION.md")

    def test_the_population_is_the_declared_deciding_one(self):
        population = self.record["population"]
        self.assertEqual(population["split"], "training")
        self.assertEqual(population["images"], 36)
        self.assertEqual((population["occupied_cells"], population["empty_cells"]), (667, 4933))
        self.assertEqual(population["nonempty_boards"], 28)
        self.assertEqual(population["contexts"], 12)

    def test_each_arm_re_derives_from_its_own_sweep(self):
        for arm in curator.ARMS:
            measured = curator.arm_summary(self.sweeps[arm]["deciding"])
            for key in ("auc", "per_cell_f1", "per_cell_gap", "best_gap", "exact_boards_at_best",
                        "operating_counts", "best", "verdict", "auc_interval", "operating_intervals"):
                self.assertEqual(self.record["arms"][arm][key], measured[key], (arm, key))

    def test_the_sweeps_are_the_widened_population_on_the_standing_grid(self):
        for arm, sweep in self.sweeps.items():
            self.assertEqual(sweep["forwards"], 54, arm)
            self.assertEqual(len(self.training(arm)), 36, arm)
            self.assertEqual(sweep["training_population"]["evaluated_groups"], 12, arm)
            self.assertEqual(len(sweep["deciding"]["thresholds"]), 81, arm)
            self.assertEqual(sweep["verdict"], sweep_verdict(sweep["deciding"]), arm)

    def test_the_paired_bootstrap_re_derives_over_twelve_clusters(self):
        expected = curator.paired_bootstrap(self.training("lever"), self.training("control"))
        self.assertEqual(self.record["attribution"]["cluster_bootstrap"], expected)
        self.assertEqual(expected["clusters"], 12)
        for key in ("auc_lever", "f1_lever", "auc_control", "f1_control", "f1_delta", "auc_delta"):
            self.assertIn(key, expected)

    def test_the_verdict_is_the_predeclared_rule_on_the_recorded_numbers(self):
        arms = self.record["arms"]
        expected = curator.lever_verdict(arms["lever"]["per_cell_f1"], arms["lever"]["verdict"],
                                         arms["lever"]["best"]["f1"]-arms["control"]["best"]["f1"])
        self.assertEqual(self.record["verdict"], expected)
        self.assertEqual(self.record["verdict"]["verdict"], "not-cleared")
        self.assertEqual(self.record["verdict"]["failing"],
                         ["per_cell_f1", "ranking", "attribution"])

    def test_each_condition_is_independently_required(self):
        self.assertEqual(curator.lever_verdict(0.71, "informative", 0.06)["verdict"], "cleared")
        self.assertEqual(curator.lever_verdict(0.699, "informative", 0.06)["verdict"], "not-cleared")
        self.assertEqual(curator.lever_verdict(0.71, "uninformative", 0.06)["verdict"], "not-cleared")
        self.assertEqual(curator.lever_verdict(0.71, "informative", 0.049)["verdict"], "not-cleared")

    def test_the_envelope_and_start_are_matched(self):
        studies = {arm: json.loads((ROOT/"runs"/f"m3-occctx-{arm}-r1"/"summary.json").read_text())
                   for arm in curator.ARMS}
        origins = {arm: studies[arm]["initial_weights_from"] for arm in curator.ARMS}
        self.assertEqual(origins["lever"]["weights_sha256"], origins["control"]["weights_sha256"])
        for arm in curator.ARMS:
            self.assertEqual(studies[arm]["steps"], 64)
            self.assertEqual(studies[arm]["status"], "complete")
            self.assertEqual(studies[arm]["arguments"]["occupancy_context"], curator.ARMS[arm])
            self.assertEqual(studies[arm]["arguments"]["seed"], 20261026)
        self.assertEqual(self.record["envelope"]["attribution_floor"], 0.05)
        self.assertEqual(self.record["matched"]["config_difference"], ["occupancy_context"])


if __name__ == "__main__":
    unittest.main()
