from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from m2_fit_contract import CLASSES, summarize_fit


def perfect_records():
    records = []
    for index in range(32):
        label = CLASSES[index // 4]
        records.append({"sample_id": f"case-{index}", "group_id": f"group-{index // 2}",
                        "exact_locked_board": True, "known_cells": 200,
                        "mean_geometry_error_cells": .1, "known_layers": 2, "exact_known_layers": 2,
                        "fit_case": {"semantic_known": 8 if index // 2 % 2 == 0 else 0,
                                     "semantic_correct": 8 if index // 2 % 2 == 0 else 0,
                                     "roles": {role: {"class": label, "known": True, "observed": label != "absent", "exact": True}
                                               for role in ("active", "ghost")}}})
    return records


class FitContractTests(unittest.TestCase):
    def test_perfect_declared_fit_has_explicit_observed_and_absent_denominators(self):
        result = summarize_fit(perfect_records())
        self.assertTrue(result["passed"])
        self.assertEqual((result["exact_observed_layers"], result["exact_absent_layers"], result["known_semantic_cells"]), (56, 8, 128))
        self.assertEqual(result["roles_by_target_class"]["active"]["I"], {"known": 4, "exact": 4})

    def test_any_board_pose_colour_or_geometry_miss_fails_fit(self):
        for failure in ("board", "role", "semantic", "geometry"):
            records = perfect_records()
            if failure == "board":
                records[0]["exact_locked_board"] = False
            elif failure == "role":
                records[0]["fit_case"]["roles"]["ghost"]["exact"] = False
                records[0]["exact_known_layers"] = 1
            elif failure == "semantic":
                records[0]["fit_case"]["semantic_correct"] = 7
            else:
                for row in records:
                    row["mean_geometry_error_cells"] = .25
            with self.subTest(failure=failure):
                self.assertFalse(summarize_fit(records)["passed"])

    def test_missing_duplicate_unknown_or_falsely_aggregated_support_refuses(self):
        changes = [lambda r: r.pop(), lambda r: r[0].update(sample_id=r[1]["sample_id"]),
                   lambda r: r[0].update(group_id=r[2]["group_id"]),
                   lambda r: r[0]["fit_case"]["roles"]["active"].update(known=False),
                   lambda r: r[0]["fit_case"]["roles"]["active"].update(observed=False),
                   lambda r: r[0].update(exact_known_layers=1),
                   lambda r: r[0]["fit_case"].update(semantic_known=7, semantic_correct=7),
                   lambda r: r[0].update(mean_geometry_error_cells=float("nan"))]
        for mutate in changes:
            records = perfect_records()
            mutate(records)
            with self.assertRaises(ValueError):
                summarize_fit(records)

    def test_absence_cannot_replace_observed_classes_or_zero_semantic_support(self):
        records = perfect_records()
        for row in records:
            for item in row["fit_case"]["roles"].values():
                item.update({"class": "absent", "observed": False})
        with self.assertRaises(ValueError):
            summarize_fit(records)
        records = perfect_records()
        for row in records:
            row["fit_case"].update(semantic_known=0, semantic_correct=0)
        with self.assertRaises(ValueError):
            summarize_fit(records)


if __name__ == "__main__":
    unittest.main()
