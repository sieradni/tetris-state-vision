"""Integrity controls for the M40 composed-alignment receipt.

`docs/M40_COMPOSED_ALIGNMENT.md` fixes the bar (`0.372881` / `0.689775`) before the run, declares the arm as
a half-strength blend that adds no parameter, and pre-commits the reading of a refusal. These tests bind the
receipt to those commitments: the width is `30`, the standing gate block is reported, the deltas are the
winner's re-scored numbers, the split sums, and the diagnosis is the swamped reading the document records.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPE = json.loads((ROOT/"docs/evidence/m40-composed-alignment-r6/scope.json").read_text())
DIAGNOSIS = json.loads((ROOT/"docs/evidence/m40-composed-alignment-r6/diagnosis.json").read_text())
M34 = json.loads((ROOT/"docs/evidence/m34-support-lattice-r1/scope.json").read_text())
M35 = json.loads((ROOT/"docs/evidence/m35-contrast-lattice-r1/scope.json").read_text())


class DeclaredTest(unittest.TestCase):
    def test_the_arm_is_the_declared_thirty_input_form(self):
        self.assertEqual(SCOPE["arm"]["inputs"], 30)
        self.assertEqual(SCOPE["arm"]["features"], 30)
        self.assertEqual(SCOPE["predeclaration"], "docs/M40_COMPOSED_ALIGNMENT.md")

    def test_the_bar_is_the_winners_rescored_numbers(self):
        self.assertEqual(SCOPE["bar"]["f1_at_zero"], 0.372881)
        self.assertEqual(SCOPE["bar"]["best"], 0.689775)
        self.assertFalse(SCOPE["bar"]["met"])
        self.assertAlmostEqual(SCOPE["anchors"]["anchor_auc"], 0.954529, places=6)


class MeasuredTest(unittest.TestCase):
    def test_the_receipt_is_the_one_the_document_quotes(self):
        self.assertAlmostEqual(SCOPE["own"]["f1_at_zero"], 0.260599, places=6)
        self.assertAlmostEqual(SCOPE["own"]["auc"], 0.897918, places=6)
        self.assertAlmostEqual(SCOPE["own_best"], 0.439535, places=6)
        self.assertAlmostEqual(SCOPE["own"]["best"]["threshold"], 1.65, places=6)

    def test_the_deltas_are_the_winners_own_numbers(self):
        self.assertAlmostEqual(SCOPE["gain_over_anchor_f1"], -0.112283, places=6)
        self.assertAlmostEqual(SCOPE["auc_delta_vs_anchor"], -0.056611, places=6)
        self.assertAlmostEqual(SCOPE["gain_over_anchor_best"], -0.250240, places=6)

    def test_the_standing_gate_is_reported(self):
        self.assertEqual(SCOPE["gates"], {"per_cell_f1": False, "ranking": False, "attribution": True})
        self.assertEqual(SCOPE["gates_anchored"], SCOPE["gates"])
        self.assertFalse(SCOPE["cleared"])

    def test_the_split_sums_to_the_measured_span(self):
        self.assertTrue(SCOPE["fractions_sum_to_one"])
        self.assertAlmostEqual(SCOPE["span_closed_fraction"]+SCOPE["placement_fraction"]+
                               SCOPE["representation_fraction"], 1.0, places=6)
        # the components are rounded to six places in the receipt, so their printed sum can differ from the
        # printed span by one unit in the sixth place; the decomposition is asserted, not re-derived
        self.assertAlmostEqual(SCOPE["span_points"]+SCOPE["placement_points"]+
                               SCOPE["representation_points"], SCOPE["anchors"]["span"], places=5)
        self.assertEqual(SCOPE["short_stage"], "closure")


class BetweenTheEndpointsTest(unittest.TestCase):
    """The dose-response the document reads: ranking falls with the share of replaced geometry."""

    def test_the_arm_sits_between_the_two_measured_endpoints(self):
        winner = SCOPE["anchors"]["anchor_auc"]
        support = M34["own"]["auc"]
        self.assertGreater(SCOPE["own"]["auc"], support)
        self.assertLess(SCOPE["own"]["auc"], winner)
        self.assertGreater(SCOPE["own_best"], M34["own_best"])
        self.assertLess(SCOPE["own_best"], SCOPE["anchors"]["anchor_best"])

    def test_the_losing_family_is_the_one_that_moves_the_sampled_geometry(self):
        # arms that keep the predicted lattice sampled land above every arm that moves off it
        keeping = [M35["own"]["auc"]]
        moving = [M34["own"]["auc"], SCOPE["own"]["auc"]]
        self.assertGreater(min(keeping), max(moving))
        self.assertGreater(min([M35["own_best"]]), max([M34["own_best"], SCOPE["own_best"]]))

    def test_the_swamped_signature_is_the_one_the_document_records(self):
        errors = DIAGNOSIS["errors_at_zero"]
        self.assertEqual(errors["false_positive"]["count"], 1181)
        self.assertAlmostEqual(errors["false_positive"]["margin_median"], 1.233860, places=6)
        self.assertLess(errors["false_positive"]["margin_median"],
                        errors["true_positive"]["margin_median"])
        bands = DIAGNOSIS["false_positive_bands"]
        self.assertEqual(bands["rows_0_4"]["false_positives"]+bands["rows_5_9"]["false_positives"], 910)
        self.assertEqual(bands["rows_15_19"]["false_positives"], 136)
        self.assertAlmostEqual(DIAGNOSIS["errors_at_zero"]["true_positive"]["margin_low"], 0.168971,
                               places=6)

    def test_the_textured_empty_reading_matches_the_document(self):
        comparison = SCOPE["comparison_to_m35"]
        self.assertFalse(comparison["textured_empty_separated"])
        self.assertEqual(comparison["false_positives_before"], 1088)
        self.assertEqual(comparison["false_positives_now"], 1181)


if __name__ == "__main__":
    unittest.main()


class ClosureConsistencyTest(unittest.TestCase):
    """The closure statement rebuilt from the per-arm receipts, so the prose cannot drift from the table.

    V-167: three maintained documents asserted the two arm families "do not overlap on either ranking
    measure" (retracted wording, quoted only as this instrument's symptom). They overlap on both, and one arm that moves the sampled geometry outranks two arms that keep
    it. These tests derive the ranges, the overlap and the signs from the receipts themselves and then check
    that the documents still claim the supported reading.
    """

    ARMS = {
        "M34": "m34-support-lattice-r1",
        "M35": "m35-contrast-lattice-r1",
        "M36": "m36-monotone-contrast-r1",
        "M37": "m37-stencil-fusion-r2",
        "M38": "m38-capacity-r1",
        "M39": "m39-dual-placement-r1",
        "M40": "m40-composed-alignment-r6",
        "M41": "m41-support-contrast-r1",
        "M42": "m42-support-crossing-r1",
        "M43": "m43-column-step-r1",
    }
    KEEPS = ("M35", "M37", "M38", "M36")
    MOVES = ("M34", "M40", "M39", "M41")
    DOCUMENTS = ("docs/M40_COMPOSED_ALIGNMENT.md", "docs/ROADMAP.md", "docs/CONTINUATION_STATE.md")

    @classmethod
    def setUpClass(cls):
        cls.metric = {}
        for name, directory in cls.ARMS.items():
            scope = json.loads((ROOT/"docs/evidence"/directory/"scope.json").read_text())
            cls.metric[name] = (scope["own"]["auc"], scope["own_best"])

    def auc(self, name):
        return self.metric[name][0]

    def best(self, name):
        return self.metric[name][1]

    def test_the_winner_is_above_every_arm_on_both_measures(self):
        winner_auc, winner_best = SCOPE["anchors"]["anchor_auc"], SCOPE["anchors"]["anchor_best"]
        self.assertLess(max(self.auc(name) for name in self.ARMS), winner_auc)
        self.assertLess(max(self.best(name) for name in self.ARMS), winner_best)

    def test_the_strength_axis_is_monotone_in_both_measures(self):
        # same read, same head, only the support's share of the sampled geometry changes
        axis = (SCOPE["anchors"]["anchor_auc"], self.auc("M40"), self.auc("M34"))
        bests = (SCOPE["anchors"]["anchor_best"], self.best("M40"), self.best("M34"))
        self.assertEqual(axis, tuple(sorted(axis, reverse=True)))
        self.assertEqual(bests, tuple(sorted(bests, reverse=True)))

    def test_the_families_overlap_and_the_overlap_holds_the_named_arms(self):
        keeping_auc = (min(self.auc(n) for n in self.KEEPS), max(self.auc(n) for n in self.KEEPS))
        moving_auc = (min(self.auc(n) for n in self.MOVES), max(self.auc(n) for n in self.MOVES))
        keeping_best = (min(self.best(n) for n in self.KEEPS), max(self.best(n) for n in self.KEEPS))
        moving_best = (min(self.best(n) for n in self.MOVES), max(self.best(n) for n in self.MOVES))
        overlap_auc = (max(keeping_auc[0], moving_auc[0]), min(keeping_auc[1], moving_auc[1]))
        overlap_best = (max(keeping_best[0], moving_best[0]), min(keeping_best[1], moving_best[1]))
        # the ranges overlap on both measures, which is what the documents now say
        self.assertLess(overlap_auc[0], overlap_auc[1])
        self.assertLess(overlap_best[0], overlap_best[1])
        inside_auc = sorted(n for n in self.ARMS if overlap_auc[0] <= self.auc(n) <= overlap_auc[1])
        inside_both = sorted(n for n in self.ARMS if overlap_auc[0] <= self.auc(n) <= overlap_auc[1]
                             and overlap_best[0] <= self.best(n) <= overlap_best[1])
        self.assertEqual(inside_auc, ["M36", "M38", "M40"])
        self.assertEqual(inside_both, ["M36", "M40"])

    def test_a_moving_arm_outranks_a_keeping_arm_inside_the_overlap(self):
        # the gap the corrected documents state explicitly: provenance alone does not order these arms
        self.assertGreater(self.auc("M40"), self.auc("M36"))
        self.assertGreater(self.best("M40"), self.best("M36"))
        self.assertGreater(self.auc("M40"), self.auc("M38"))
        self.assertLess(self.best("M40"), self.best("M38"))
        self.assertLess(self.auc("M34"), self.auc("M36"))
        self.assertLess(self.best("M34"), self.best("M36"))

    FAMILY_TOKENS = ("famil", "keeping", "moves the")
    SEPARATION = ("do not overlap", "don't overlap", "does not overlap", "no overlap", "are disjoint",
                  "are separate", "strictly higher", "strictly lower", "strictly separate",
                  "separate bands", "in a different band", "never overlaps", "disjoint bands")

    def phrases_claiming_separation(self, document):
        """Sentences that put a family and a separation word together - the retracted claim's shape."""
        body = (ROOT/document).read_text(encoding="utf-8")
        flat = " ".join(body.split())
        found = []
        for sentence in flat.replace(". ", ".\n").splitlines():
            low = sentence.lower()
            if any(token in low for token in self.FAMILY_TOKENS) and \
                    any(phrase in low for phrase in self.SEPARATION):
                found.append(sentence.strip())
        return found

    def test_the_documents_state_the_overlap_rather_than_the_separation(self):
        for document in self.DOCUMENTS:
            body = (ROOT/document).read_text(encoding="utf-8")
            self.assertIn("overlap", body.lower(), f"{document} must state the overlap")
            for sentence in self.phrases_claiming_separation(document):
                self.fail(f"{document} asserts a separation the receipts do not show: {sentence!r} - "
                          "the two families overlap on AUC 0.886350-0.897918 and best swept "
                          "0.420935-0.439535 (M36/M38/M40 and M36/M40), so the sentence must say the "
                          "ranges overlap and name the arms inside the overlap")

    def test_the_documents_keep_the_negative_and_the_gradient(self):
        body = (ROOT/"docs/M40_COMPOSED_ALIGNMENT.md").read_text(encoding="utf-8")
        for number in ("0.954529", "0.897918", "0.883260", "0.689775", "0.439535", "0.385925",
                       "0.011568", "0.018600", "0.002877", "0.024016"):
            self.assertIn(number, body)
        self.assertIn("not learned", body.lower())
