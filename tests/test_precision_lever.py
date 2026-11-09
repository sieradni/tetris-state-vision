"""The precision-lever record re-derives from its committed receipts."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

import record_precision_lever as curator
from occupancy_sweep import sweep_verdict

EVIDENCE = ROOT/"docs/evidence/m3-occupancy-precision-r1/scope.json"


class PrecisionLeverRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        cls.sweeps = {}
        cls.studies = {}
        for arm in curator.ARMS:
            cls.sweeps[arm] = json.loads((ROOT/"runs"/f"m3-occdetail-{arm}-r1-sweep-r1"/"result.json").read_text(encoding="utf-8"))
            cls.studies[arm] = json.loads((ROOT/"runs"/f"m3-occdetail-{arm}-r1"/"summary.json").read_text(encoding="utf-8"))

    def training(self, arm):
        return [row for row in self.sweeps[arm]["cases"] if row["split"] == "training"]

    def test_the_record_is_a_passed_measurement_of_the_declared_instrument(self):
        self.assertEqual(self.record["status"], "passed")
        self.assertEqual(self.record["instrument"], "m3-occupancy-precision-r1")
        self.assertEqual(self.record["predeclaration"], "docs/M3_OCCUPANCY_PRECISION.md")

    def test_the_population_and_work_arithmetic_are_the_declared_ones(self):
        population = self.record["population"]
        self.assertEqual((population["images"], population["nonempty_boards"]), (36, 28))
        self.assertEqual((population["occupied_cells"], population["empty_cells"]), (667, 4933))
        self.assertEqual(population["contexts"], 12)
        self.assertEqual(self.record["envelope"]["work_items"],
                         {"profile": 2, "study": 64, "sweep_profile": 2, "sweep": 54, "curation": 4})
        for arm, sweep in self.sweeps.items():
            self.assertEqual(sweep["forwards"], 54, arm)

    def test_each_arm_re_derives_from_its_own_sweep(self):
        for arm in curator.ARMS:
            measured = curator.arm_summary(self.sweeps[arm]["deciding"])
            measured["precision"] = round(curator.precision_of(measured["operating_counts"]), 6)
            for key in ("auc", "per_cell_f1", "best", "verdict", "precision",
                        "operating_counts", "exact_boards_at_best"):
                self.assertEqual(self.record["arms"][arm][key], measured[key], (arm, key))

    def test_the_sweeps_are_the_widened_population_on_the_standing_grid(self):
        for arm, sweep in self.sweeps.items():
            self.assertEqual(len(self.training(arm)), 36, arm)
            self.assertEqual(sweep["training_population"]["evaluated_groups"], 12, arm)
            self.assertEqual(len(sweep["deciding"]["thresholds"]), 81, arm)
            self.assertEqual(sweep["verdict"], sweep_verdict(sweep["deciding"]), arm)

    def test_the_paired_bootstrap_re_derives_over_twelve_clusters(self):
        expected = curator.paired_bootstrap(self.training("lever"), self.training("control"))
        self.assertEqual(self.record["attribution"]["cluster_bootstrap"], expected)
        self.assertEqual(expected["clusters"], 12)

    def test_the_verdict_and_mechanism_are_the_predeclared_rules(self):
        arms = self.record["arms"]
        verdict = curator.lever_verdict(arms["lever"]["per_cell_f1"], arms["lever"]["verdict"],
                                        arms["lever"]["best"]["f1"]-arms["control"]["best"]["f1"])
        self.assertEqual(self.record["verdict"], verdict)
        mechanism = curator.mechanism(arms["lever"]["precision"]-arms["control"]["precision"],
                                      self.record["attribution"]["cluster_bootstrap"])
        self.assertEqual(self.record["mechanism"], mechanism)
        self.assertEqual(mechanism["precision_improved"],
                         mechanism["precision_delta"] >= 0.05 and
                         mechanism["precision_delta_interval"]["low"] > 0)

    def test_the_published_auc_gap_prose_matches_the_evidence(self):
        precision_gap = round(0.90-self.record["arms"]["lever"]["auc"], 3)
        best_gap = round(0.70-self.record["arms"]["lever"]["best"]["f1"], 3)
        self.assertEqual((precision_gap, best_gap), (0.077, 0.288))
        text = (ROOT/"docs/M3_OCCUPANCY_PRECISION.md").read_text(encoding="utf-8")
        self.assertIn("0.077 short of 0.90", text)
        self.assertIn("0.288 short of 0.70", text)
        for name in ("docs/M3_OCCUPANCY_REPRESENTATION.md", "docs/M3_OCCUPANCY_RANKING.md",
                     "docs/ROADMAP.md", "docs/CONTINUATION_STATE.md"):
            self.assertNotIn("0.176 short", (ROOT/name).read_text(encoding="utf-8"), name)

    def test_the_pair_is_matched_and_single_lever(self):
        self.assertEqual(len({self.studies[arm]["catalogue_sha256"] for arm in curator.ARMS}), 1)
        self.assertEqual(self.studies["lever"]["arguments"]["occupancy_detail"], 1)
        self.assertEqual(self.studies["control"]["arguments"]["occupancy_detail"], 0)
        for arm in curator.ARMS:
            self.assertEqual(self.studies[arm]["arguments"]["occupancy_context"], 0)
            self.assertEqual(self.studies[arm]["steps"], 64)
            self.assertEqual(self.studies[arm]["status"], "complete")
            self.assertEqual(self.studies[arm]["initial_weights_from"]["weights_sha256"],
                             self.studies["lever"]["initial_weights_from"]["weights_sha256"])
        self.assertEqual(self.record["matched"]["config_difference"], ["occupancy_detail"])


if __name__ == "__main__":
    unittest.main()
