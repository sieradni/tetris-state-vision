"""The label-free operating-point calibration: a per-board threshold from the read's own logits."""
import inspect
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

import measure_calibrated_operating_point as calibration


class OtsuRuleTests(unittest.TestCase):
    def test_the_split_of_two_equal_groups_is_their_midpoint(self):
        self.assertEqual(calibration.otsu([0.0]*50+[10.0]*50), 5.0)

    def test_the_rule_is_translation_equivariant(self):
        values = [-2.0, -1.5, -1.0, 0.5, 3.0, 4.0, 4.5, 9.0]
        base = calibration.otsu(values)
        for shift in (1.0, -3.5, 12.0):
            self.assertAlmostEqual(calibration.otsu([value+shift for value in values]),
                                   base+shift, places=9)

    def test_a_constant_board_decides_nothing(self):
        threshold = calibration.otsu([2.5]*200)
        self.assertEqual(threshold, 2.5)
        block = calibration.counts([2.5]*200, [1]*200, threshold)
        self.assertEqual(block["true_positive"], 0)

    def test_the_rule_separates_the_dense_side_of_a_skewed_board(self):
        values = [5.0, 5.4, 5.2, 4.9, 5.1]+[-3.0, -2.4, -3.3, -2.8]*10
        threshold = calibration.otsu(values)
        self.assertGreater(threshold, 0.0)
        self.assertLess(threshold, 4.9)
        decided = [value > threshold for value in values]
        self.assertEqual(sum(decided), 5)

    def test_the_rule_reads_only_the_logits(self):
        self.assertEqual(tuple(inspect.signature(calibration.otsu).parameters), ("values",))

    def test_the_counts_and_scores_are_arithmetic(self):
        logits = [1.0, 0.5, -0.5, -1.0]
        labels = [1, 1, 0, 0]
        block = calibration.counts(logits, labels, 0.0)
        self.assertEqual(block, {"true_positive": 2, "false_positive": 0,
                                 "false_negative": 0, "true_negative": 2})
        scores = calibration.scores(block)
        self.assertEqual((scores["precision"], scores["recall"], scores["f1"]), (1.0, 1.0, 1.0))
        mixed_counts = calibration.counts(logits, labels, .75)
        self.assertEqual(mixed_counts["true_positive"], 1)
        self.assertEqual(mixed_counts["false_negative"], 1)
        self.assertAlmostEqual(calibration.scores(mixed_counts)["f1"], 2/3, places=6)

    def test_a_degenerate_score_is_zero_not_an_error(self):
        block = {"true_positive": 0, "false_positive": 0, "false_negative": 4, "true_negative": 4}
        self.assertEqual(calibration.scores(block)["f1"], 0.0)


class CalibrationRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        scope = cls.root/"docs/evidence/m7-calibration-r1/scope.json"
        if not scope.exists():
            raise unittest.SkipTest("the calibration record has not been curated yet")
        cls.scope = json.loads(scope.read_text(encoding="utf-8"))
        cls.prior = json.loads((cls.root/"docs/evidence/m6-row-offset-r1/scope.json")
                               .read_text(encoding="utf-8"))
        cls.text = " ".join((cls.root/"docs/M7_CALIBRATION.md").read_text(encoding="utf-8").split())

    def test_the_arms_are_the_declared_ones(self):
        self.assertEqual(sorted(self.scope["arms"]), ["control", "down", "wide"])
        self.assertEqual({name: arm["occupancy_offset_rows"] for name, arm in self.scope["arms"].items()},
                         {"control": 0, "wide": 0, "down": -1})

    def test_control_and_wide_reproduce_the_published_record(self):
        self.assertEqual(self.scope["arms"]["control"]["auc"], 0.8226433610683003)
        self.assertEqual(self.scope["arms"]["wide"]["auc"], 0.8822623454135491)
        for name in ("control", "wide", "down"):
            self.assertEqual(self.scope["arms"][name]["auc"], self.prior["arms"][name]["auc"])

    def test_the_calibration_rule_is_labelfree_and_reported_as_such(self):
        for name in ("control", "wide", "down"):
            self.assertTrue(self.scope["arms"][name]["calibrated"]["label_free"])
            self.assertEqual(self.scope["arms"][name]["calibrated"]["rule"], "otsu-logits")

    def test_the_primary_reading_follows_the_printed_numbers(self):
        block = self.scope["arms"]["down"]
        if (block["calibrated"]["f1"] >= .381640+.05 and
                block["calibrated"]["f1"] >= .85*.469259):
            expected = "calibration-recovers-the-operating-point"
        elif block["calibrated"]["f1"] > .381640:
            expected = "calibration-moves-without-recovering"
        else:
            expected = "calibration-fails"
        self.assertEqual(self.scope["reading"], expected)

    def test_the_secondary_reading_follows_the_printed_delta(self):
        delta = self.scope["contrasts"]["down_calibrated_control_calibrated"]["f1_delta"]
        self.assertEqual(self.scope["secondary_reading"],
                         "offset-advantage-survives-calibration" if delta >= .05 else
                         "offset-advantage-was-a-threshold-effect")

    def test_the_document_quotes_the_published_scope(self):
        for name, arm in self.scope["arms"].items():
            self.assertIn(f"{round(arm['auc'], 6):,.6f}", self.text, name)
            self.assertIn(f"{round(arm['calibrated']['f1'], 6):,.6f}", self.text, name)
        self.assertIn(self.scope["reading"], self.text)
        self.assertIn(self.scope["secondary_reading"], self.text)


class CalibrationFamilyTests(unittest.TestCase):
    """V-126: the arm family must be owned by the run gate before the block is launched."""

    def test_the_family_is_declared_and_narrow(self):
        from record_stream_run import OWNED_RUN_PATTERN
        for name in ("m7-calibration-control-r1", "m7-calibration-wide-r1-sweep-r1",
                     "m7-calibration-down-r1-sweep-r1-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        for name in ("m7-calibration-", "m7-calibration", "m7-calib-control-r1"):
            self.assertFalse(OWNED_RUN_PATTERN.fullmatch(name), name)


if __name__ == "__main__":
    unittest.main()
