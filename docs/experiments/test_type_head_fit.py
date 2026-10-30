"""Finite corpus/label/metric tripwires without importing Torch."""
import copy
import json
import math
import unittest

from type_head_fit_contract import metrics, training_indices, validate_rows


def fixture():
    rows = []
    for label in range(8):
        for pose in range(2):
            for view in range(2):
                for role in ("active", "ghost"):
                    logits = [-1.] * 8
                    logits[label] = 1.
                    rows.append({"split": "training", "sample_id": f"t{label}-{pose}-{view}",
                                 "group_id": f"g{label}-{pose}", "role": role, "known": True,
                                 "target_type": label, "features": [float(label), float(view)], "logits": logits})
    for index in range(9):
        for role in ("active", "ghost"):
            logits = [-1.] * 8
            logits[index % 8] = 1.
            rows.append({"split": "development", "sample_id": f"d{index}", "group_id": f"d{index//3}",
                         "role": role, "known": True, "target_type": index % 8,
                         "features": [float(index), 3.], "logits": logits})
    return rows


class TypeHeadContractTests(unittest.TestCase):
    def test_balanced_pairs_have_exact_expected_counts(self):
        rows = fixture()
        validate_rows(rows, 2)
        result = metrics(rows, [row["logits"] for row in rows])
        self.assertEqual(result["training"]["correct"], 64)
        self.assertEqual(result["training"]["ancestries"], 16)
        self.assertEqual(result["development"]["correct"], 18)

    def test_development_and_unknowns_cannot_supply_training_indices(self):
        rows = fixture()
        expected = list(range(64))
        self.assertEqual(training_indices(rows), expected)
        for row in rows[64:]:
            row.update(features=[1e50, -1e50], target_type=7, known=False)
        self.assertEqual(training_indices(rows), expected)
        rows[0]["known"] = False
        self.assertNotIn(0, training_indices(rows))
        with self.assertRaises(ValueError):
            validate_rows(rows, 2)

    def test_unknown_development_label_invariance(self):
        rows = fixture()
        rows[-1]["known"] = False
        logits = [row["logits"] for row in rows]
        before = metrics(rows, logits)
        rows[-1]["target_type"] = 7
        self.assertEqual(before, metrics(rows, logits))
        self.assertEqual(before["development"]["unknown_excluded"], 1)

    def test_duplicate_unpaired_unbalanced_and_ancestry_collapse_refuse(self):
        original = fixture()
        mutations = []
        rows = copy.deepcopy(original); rows[1] = rows[0]; mutations.append(rows)
        rows = copy.deepcopy(original); rows[1]["sample_id"] = "wrong"; mutations.append(rows)
        rows = copy.deepcopy(original); rows[0]["target_type"] = 7; mutations.append(rows)
        rows = copy.deepcopy(original)
        for row in rows:
            if row["split"] == "training" and row["target_type"] == 0:
                row["group_id"] = "same"
        mutations.append(rows)
        for rows in mutations:
            with self.assertRaises(ValueError):
                validate_rows(rows, 2)

    def test_nonfinite_shape_and_boolean_label_refuse(self):
        original = fixture()
        for change in ({"features": [float("nan"), 0.]}, {"logits": [float("inf")] * 8},
                       {"features": [0.]}, {"known": 1}, {"target_type": True}):
            rows = copy.deepcopy(original)
            rows[0].update(change)
            with self.assertRaises(ValueError):
                validate_rows(rows, 2)

    def test_stable_cross_entropy_and_json_roundtrip(self):
        rows = fixture()
        result = metrics(rows, [[1000.] * 8 for _ in rows])
        self.assertAlmostEqual(result["training"]["mean_cross_entropy"], math.log(8), places=11)
        self.assertEqual(result["training"]["correct"], 8)
        self.assertEqual(result, json.loads(json.dumps(result)))


if __name__ == "__main__":
    unittest.main()
