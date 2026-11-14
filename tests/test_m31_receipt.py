"""Integrity controls for the M31 boundary receipt and its follow-up.

`docs/M31_DECISION_RULE.md` predeclares three boundaries on one frozen arm, an oracle that is
excluded from every verdict, and the branch's structural follow-up. These tests bind what the
receipts claim: the global read joins the M30 record, the label-free rule is label-free and loses,
the oracle is an upper bound kept out of the verdict, the branch follows the declared rule, and the
follow-up's measured result is the one the docs quote.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECISION = json.loads((ROOT/"docs/evidence/m31-decision-rule-r2/scope.json").read_text())
FOLLOW = json.loads((ROOT/"docs/evidence/m31-followup-r1/scope.json").read_text())
M30 = json.loads((ROOT/"docs/evidence/m30-operating-point-r1/scope.json").read_text())


class JoinTest(unittest.TestCase):
    def test_the_global_read_is_the_m30_record_unchanged(self):
        identity = DECISION["reads"]["identity"]
        self.assertEqual(identity["global_f1_at_zero"],
                         round(M30["variants"]["calibrate"]["own"]["f1_at_zero"], 6))
        self.assertEqual(identity["global_best"], M30["variants"]["calibrate"]["own_best"])
        self.assertEqual(identity["global"]["cells"], 2986)
        self.assertEqual((DECISION["population"]["occupied_identity"],
                          DECISION["population"]["occupied_anchored"]), (214, 309))

    def test_the_gate_on_the_global_read_is_the_standing_one(self):
        self.assertEqual(DECISION["gate"], {"per_cell_f1": False, "ranking": False,
                                            "attribution": True})
        self.assertFalse(DECISION["cleared"])
        identity = DECISION["reads"]["identity"]
        self.assertLess(identity["global_f1_at_zero"], 0.70)
        self.assertGreaterEqual(identity["global"]["auc"], 0.90)
        self.assertLess(identity["global_best"], 0.70)


class BoundaryTest(unittest.TestCase):
    def test_the_label_free_rule_loses_to_the_global_boundary(self):
        for pairing in ("identity", "anchored"):
            read = DECISION["reads"][pairing]
            self.assertLess(read["label_free_f1_at_zero"], read["global_f1_at_zero"], pairing)
            self.assertLess(read["label_free_best"], read["global_best"], pairing)

    def test_the_branch_follows_the_declared_rule(self):
        identity = DECISION["reads"]["identity"]
        expected = ("decision-rule" if identity["label_free_f1_at_zero"] >= DECISION["decision"]["cell_gate"]
                    else "representation")
        self.assertEqual(DECISION["decision"]["branch"], expected)
        self.assertEqual(expected, "representation")

    def test_the_oracle_is_an_upper_bound_and_is_excluded(self):
        identity = DECISION["reads"]["identity"]
        self.assertGreaterEqual(identity["oracle_f1_at_zero"], identity["global_f1_at_zero"])
        self.assertGreaterEqual(identity["oracle_f1_at_zero"], identity["label_free_f1_at_zero"])
        self.assertGreaterEqual(identity["oracle_f1_at_zero"], 0.70)
        self.assertTrue(DECISION["decision"]["oracle_excluded"])
        self.assertIn("excluded", DECISION["decision"]["oracle_role"])
        self.assertNotIn("oracle", json.dumps(DECISION["gate"]))
        self.assertIn("oracle", json.dumps(DECISION["reads"]))


class FollowupTest(unittest.TestCase):
    def test_the_follow_up_is_the_declared_context_form(self):
        self.assertEqual(FOLLOW["branch"], "representation")
        self.assertEqual(FOLLOW["arm"]["run"], "m31-context-r5")
        self.assertEqual(FOLLOW["predeclaration"], "docs/M31_DECISION_RULE.md")

    def test_the_follow_up_result_is_the_measured_refusal(self):
        self.assertAlmostEqual(FOLLOW["own"]["f1_at_zero"], 0.255139, places=6)
        self.assertAlmostEqual(FOLLOW["own_best"], 0.518657, places=6)
        self.assertLess(FOLLOW["gain_over_anchor_best"], 0.0)
        self.assertAlmostEqual(FOLLOW["gain_over_anchor_best"], -0.171118, places=6)
        self.assertFalse(FOLLOW["cleared"])
        self.assertEqual(FOLLOW["gates"], {"per_cell_f1": False, "ranking": False,
                                           "attribution": True})

    def test_the_follow_up_split_reproduces_the_span(self):
        span = FOLLOW["anchors"]["span"]
        self.assertAlmostEqual(FOLLOW["span_closed_fraction"]+FOLLOW["placement_fraction"]+
                               FOLLOW["representation_fraction"], 1.0, delta=2e-6)
        self.assertAlmostEqual(FOLLOW["span_points"]+FOLLOW["placement_points"]+
                               FOLLOW["representation_points"], span, places=5)
        self.assertAlmostEqual(FOLLOW["span_points"], FOLLOW["own_best"]-0.543974, places=5)
        self.assertLessEqual(FOLLOW["peak_bytes"], 1152*1024**2)


if __name__ == "__main__":
    unittest.main()
