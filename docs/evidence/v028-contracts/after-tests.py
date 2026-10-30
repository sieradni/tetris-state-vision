"""Adversarial analysis-only checks; no model imports or training."""
import copy
import json
import unittest

from layer_feature_analysis import analyze


def row(name, group, label, value, **changes):
    result = {"sample_id": name, "group_id": group, "role": "active", "split": "training",
              "target_type": label, "known": True, "features": [value], "logits": [0.] * 8}
    result["logits"][label] = 1.
    return {**result, **changes}


class FeatureAnalysisTests(unittest.TestCase):
    def test_all_sibling_views_are_excluded_not_just_query_row(self):
        rows = [row("query", "a", 1, 0.), row("sibling", "a", 1, 0.),
                row("other-pose", "b", 1, 10.), row("other-class", "c", 2, .1)]
        result = analyze(rows)["roles"]["active"]["training"]
        query = result["cases"][0]
        self.assertEqual(query["squared_distances"], {"1": 100., "2": .010000000000000002})
        self.assertEqual(query["prototype_type"], 2)
        self.assertEqual(result["native_correct"], 4)

    def test_missing_reference_class_is_ineligible(self):
        result = analyze([row("a", "a", 1, 0), row("b", "b", 2, 1)])["roles"]["active"]["training"]
        self.assertEqual(result["prototype_eligible"], 0)
        self.assertEqual(result["prototype_correct"], 0)
        self.assertTrue(all(case["prototype_type"] is None for case in result["cases"]))

    def test_exact_ties_abstain(self):
        rows = [row("a", "a", 1, 0), row("b", "b", 1, 1), row("c", "c", 2, 1)]
        result = analyze(rows)["roles"]["active"]["training"]
        self.assertIsNone(result["cases"][0]["prototype_type"])
        self.assertEqual(result["prototype_ties"], 1)

    def test_development_never_becomes_a_reference(self):
        rows = [row("a", "a", 1, 1), row("b", "b", 2, 2),
                row("dev", "dev", 1, 2, split="development"),
                row("dev-sibling", "dev", 1, 2, split="development")]
        result = analyze(rows)["roles"]["active"]["development"]
        self.assertEqual(result["prototype_eligible"], 2)
        self.assertEqual(result["prototype_correct"], 0)

    def test_unknown_label_value_has_no_effect_and_roles_stay_separate(self):
        rows = [row("a", "a", 1, 1), row("b", "b", 1, 1), row("u", "u", 0, 100, known=False),
                row("a", "a", 2, 2, role="ghost"), row("b", "b", 2, 2, role="ghost")]
        before = analyze(rows)
        self.assertEqual(before, json.loads(json.dumps(before)))
        rows[2]["target_type"] = 7
        self.assertEqual(before, analyze(rows))
        self.assertEqual(before["roles"]["active"]["training"]["unknown_excluded"], 1)
        self.assertEqual(before["roles"]["ghost"]["training"]["prototype_correct"], 2)

    def test_duplicate_nonfinite_and_mixed_dimensions_refuse(self):
        base = [row("a", "a", 1, 0), row("b", "b", 1, 1)]
        for rows in (base + [base[0]], [dict(base[0], features=[float("nan")])],
                     [dict(base[0], logits=[float("inf")] * 8)],
                     [base[0], dict(base[1], features=[1, 2])]):
            with self.assertRaises(ValueError):
                analyze(copy.deepcopy(rows))


if __name__ == "__main__":
    unittest.main()
