"""Both verdicts re-derive from the committed widened-population evidence bytes."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from fit_gate import FIT_GATE_THRESHOLD, fit_gate_opens
from gate_power import auc_interval, f1_from_counts, f1_gap, wilson_interval
from identity_gate import IDENTITY_CLASS_THRESHOLD, identity_verdict
from occupancy_sweep import AUC_THRESHOLD, F1_THRESHOLD, sweep_verdict
from percell_gate import CELL_F1_THRESHOLD, percell_gate_opens, occupied_cell_f1

EVIDENCE = ROOT/"docs/evidence/m3-widened-deciding-population-r1"
SCOPE = EVIDENCE/"scope.json"


class WidenedVerdictTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCOPE.is_file(), "the widened-population evidence directory must be committed")
        self.scope = json.loads(SCOPE.read_text())
        self.point = self.scope["occupancy"]["replay_operating_point"]
        self.ranking = self.scope["occupancy"]["ranking"]

    def labels(self):
        return {"nonempty": {"images": self.point["nonempty_images"],
                             "exact_boards": self.point["exact_nonempty_boards"],
                             "true_positive": self.point["true_positive"],
                             "false_positive": self.point["false_positive"],
                             "false_negative": self.point["false_negative"]}}

    def test_the_population_is_the_canonical_twelve_groups(self):
        after = self.scope["populations"]["after"]
        self.assertEqual((after["groups"], after["images"]), (12, 36))
        self.assertEqual(self.scope["populations"]["before"]["images"], 9)
        self.assertEqual(self.scope["populations"]["development"]["images"], 18)
        self.assertGreater(after["occupied_cells"], self.scope["populations"]["before"]["occupied_cells"])

    def test_every_threshold_is_the_published_one(self):
        thresholds = self.scope["unchanged_thresholds"]
        self.assertEqual(thresholds["fit_gate_exact_nonempty_boards"], FIT_GATE_THRESHOLD)
        self.assertEqual(thresholds["per_cell_f1"], CELL_F1_THRESHOLD)
        self.assertEqual(thresholds["ranking_auc"], AUC_THRESHOLD)
        self.assertEqual(thresholds["ranking_f1"], F1_THRESHOLD)
        self.assertEqual(thresholds["identity_distinct_classes"], IDENTITY_CLASS_THRESHOLD)
        self.assertEqual(thresholds["sweep_grid_points"], 81)

    def test_the_occupancy_gates_re_derive_from_the_committed_counts(self):
        occupancy = self.scope["occupancy"]
        self.assertAlmostEqual(occupied_cell_f1(self.labels()), self.point["f1"], places=6)
        self.assertEqual(percell_gate_opens(self.labels()), occupancy["per_cell_gate"]["opens"])
        self.assertAlmostEqual(f1_gap(self.point["f1"], CELL_F1_THRESHOLD), self.point["f1_gap_to_bar"], places=6)
        self.assertEqual(fit_gate_opens(self.labels()), occupancy["fit_gate"]["opens"])
        self.assertEqual(self.point["exact_nonempty_boards"], occupancy["fit_gate"]["exact"])
        self.assertFalse(occupancy["fit_gate"]["opens"])

    def test_the_reported_intervals_re_derive_and_stay_below_their_bars(self):
        interval = auc_interval(self.ranking["auc"], self.ranking["positives"], self.ranking["negatives"])
        self.assertEqual(interval, self.ranking["auc_interval"])
        self.assertLess(interval["high"], self.ranking["bar"])
        self.assertEqual(sweep_verdict({"auc": self.ranking["auc"], "best": self.ranking["best"]}),
                         self.ranking["verdict"])
        best = self.ranking["best"]
        self.assertAlmostEqual(f1_from_counts(best["true_positive"], best["false_positive"],
                                              best["false_negative"]), best["f1"], places=6)
        self.assertEqual(wilson_interval(best["true_positive"], best["true_positive"]+best["false_positive"]),
                         self.ranking["best_intervals"]["precision"])
        self.assertEqual(wilson_interval(best["true_positive"], best["true_positive"]+best["false_negative"]),
                         self.ranking["best_intervals"]["recall"])
        self.assertLess(best["f1"], F1_THRESHOLD)

    def test_the_identity_verdict_re_derives_and_still_stands(self):
        identity = self.scope["identity"]
        self.assertEqual(identity_verdict(identity["queue"], identity["hold"]), identity["verdict"])
        self.assertEqual(identity["verdict"], "stands")
        self.assertEqual(identity["queue"]["distinct_predicted_classes"], 1)
        self.assertTrue(identity["queue"]["constant_predictor"])
        self.assertTrue(identity["hold"]["constant_predictor"])
        self.assertEqual(sum(identity["queue"]["predicted_classes"].values()), 216)
        self.assertEqual(sum(identity["hold"]["predicted_classes"].values()), 33)
        self.assertEqual(wilson_interval(0, 216), identity["queue_exact_interval"])
        self.assertEqual(identity["queue"]["images"], 36)

    def test_superseded_numbers_match_the_earlier_committed_evidence(self):
        superseded = self.scope["superseded_without_invalidation"]
        old = json.loads((ROOT/"docs/evidence/m3-percell-balance-r1-errors-r1/result.json").read_text())
        nonempty = old["totals"]["training"]["locked_populations"]["nonempty"]
        self.assertEqual(superseded["occupancy_counts"],
                         {key: nonempty[key] for key in ("true_positive", "false_positive", "false_negative")})
        self.assertEqual(superseded["occupancy_exact_nonempty_boards"], nonempty["exact_boards"])
        old_identity = json.loads((ROOT/"docs/evidence/m3-identity-factor-length-r1-errors-r1/result.json").read_text())
        self.assertEqual(superseded["identity_queue_classes"],
                         old_identity["totals"]["training"]["components"]["queue"]["predicted_classes"])
        self.assertEqual(superseded["identity_hold_classes"],
                         old_identity["totals"]["training"]["components"]["hold"]["predicted_classes"])

    def test_every_copied_artifact_still_hashes_to_its_record(self):
        self.assertTrue(self.scope["artifact_sha256"])
        for name, expected in self.scope["artifact_sha256"].items():
            self.assertEqual(hashlib.sha256((EVIDENCE/name).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
