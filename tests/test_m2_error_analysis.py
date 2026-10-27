"""Adversarial reporting contracts, independent of Torch/model implementation."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("m2_error_analysis", Path(__file__).resolve().parents[1]/"scripts/m2_error_analysis.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture():
    matrix = lambda value: [[value for _ in range(10)] for _ in range(20)]
    lattice = [[.1, .9], [.04, 0.], [0., -.04]]
    prediction = {"occupancy_logits": matrix(-2.), "observability_logits": matrix(2.), "semantic": matrix(0),
                  "lattice": copy.deepcopy(lattice), "registration_logit": 2.,
                  "layers": [{"known_logit": 2., "type": 0, "column": 0, "row": 0, "rotation": 0} for _ in range(2)]}
    target = {"occupied": matrix(False), "known": matrix(True), "typed": matrix(False), "semantic": matrix(0),
              "lattice": copy.deepcopy(lattice), "registered": True,
              "layers": [{"known": True, "pose_known": False, "type": 0, "column": 0, "row": 0,
                          "rotations": [False]*4} for _ in range(2)]}
    return prediction, target


class M2ErrorAnalysisTests(unittest.TestCase):
    def test_fit_colour_and_role_scores_come_from_raw_supported_decomposition(self):
        pred, target = fixture()
        target["occupied"][0][0] = target["typed"][0][0] = True
        target["semantic"][0][0] = 2
        target["layers"][0].update(type=1, pose_known=True, column=3, row=2, rotations=[True, False, True, False])
        pred["layers"][0].update(type=1, column=3, row=2, rotation=2)
        metrics = analysis.fit_metrics_from_diagnostics(analysis.analyze_case(pred, target))
        self.assertEqual((metrics["semantic_known"], metrics["semantic_correct"]), (1, 0))
        self.assertEqual(metrics["roles"]["active"], {"class": "I", "known": True, "observed": True, "exact": True})
        self.assertEqual(metrics["roles"]["ghost"], {"class": "absent", "known": True, "observed": False, "exact": True})
        pred["layers"][0]["type"] = 0
        pred["semantic"][0][0] = 2
        changed = analysis.fit_metrics_from_diagnostics(analysis.analyze_case(pred, target))
        self.assertEqual(changed["semantic_correct"], 1)
        self.assertFalse(changed["roles"]["active"]["exact"])
        target["layers"][0].update(known=False, pose_known=False)
        unknown = analysis.fit_metrics_from_diagnostics(analysis.analyze_case(pred, target))["roles"]["active"]
        self.assertEqual(unknown["class"], "unknown")
        self.assertIsNone(unknown["exact"])

    def test_fit_strata_are_explicit_and_legacy_metadata_is_unchanged(self):
        self.assertEqual(analysis.recipe_strata({"density": "dense", "skin": "no_grid"}), ("dense", "no_grid"))
        base = {"curriculum": "all-class-fit-v1", "split": "fit", "index": 14, "view": 1}
        self.assertEqual(analysis.recipe_strata(base), ("sparse", "no_grid"))
        self.assertEqual(analysis.recipe_strata({**base, "curriculum": "all-class-interleaved-v1"}), ("sparse", "no_grid"))
        self.assertEqual(analysis.recipe_strata({**base, "index": 15, "view": 0}), ("empty", "clean"))
        for change in ({"curriculum": "other"}, {"index": True}, {"index": 16}, {"view": False}, {"split": "training"}):
            with self.assertRaises(ValueError):
                analysis.recipe_strata({**base, **change})

    def test_larger_fit_report_bound_is_explicit_and_stays_finite(self):
        pred, target = fixture()
        value = analysis.analyze_case(pred, target)
        rows = [{"sample_id": str(i), "group_id": "one-parent", "split": "training", "diagnostics": value} for i in range(50)]
        with self.assertRaises(ValueError):
            analysis.summarize(rows)
        self.assertEqual(analysis.summarize(rows, max_cases=68)["training"]["images"], 50)
        for maximum in (True, 1000):
            with self.assertRaises(ValueError):
                analysis.summarize(rows, max_cases=maximum)
    def test_empty_cell_accuracy_does_not_hide_missed_block(self):
        pred, target = fixture()
        target["occupied"][0][3] = True
        result = analysis.analyze_case(pred, target)["locked"]
        self.assertEqual(result["cell_accuracy"], .995)
        self.assertFalse(result["exact_board"])
        self.assertEqual((result["false_negative"], result["false_positive"]), (1, 0))
        self.assertEqual(result["population"], "nonempty")
        self.assertEqual(result["recall"], 0)
        self.assertIsNone(result["precision"])

    def test_false_positive_and_bottom_first_error_coordinates(self):
        pred, target = fixture()
        pred["occupancy_logits"][19][7] = 3.
        result = analysis.analyze_case(pred, target)["locked"]
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["row_errors_bottom_first"], [0]*19+[1])
        self.assertEqual((result["worst_errors"][0]["column"], result["worst_errors"][0]["row"]), (7, 19))

    def test_unknown_cells_are_excluded_even_with_changed_latent_values(self):
        pred, target = fixture()
        target["known"][0][3] = False
        first = analysis.analyze_case(pred, target)
        target["occupied"][0][3] = True
        target["semantic"][0][3] = 7
        second = analysis.analyze_case(pred, target)
        self.assertEqual(first, second)
        self.assertEqual(first["locked"]["known_cells"], 199)
        self.assertIsNone(first["locked"]["exact_board"])
        self.assertTrue(first["locked"]["exact_supported_cells"])

    def test_no_supported_cells_have_no_accuracy_or_perfect_claim(self):
        pred, target = fixture()
        target["known"] = [[False]*10 for _ in range(20)]
        target["registered"] = False
        result = analysis.analyze_case(pred, target)
        self.assertIsNone(result["locked"]["cell_accuracy"])
        self.assertIsNone(result["locked"]["exact_supported_cells"])
        self.assertEqual(result["geometry"], {"supported": False})

    def test_translation_and_scale_have_separate_analytic_signatures(self):
        pred, target = fixture()
        pred["lattice"][0][0] += .04
        shifted = analysis.analyze_case(pred, target)["geometry"]
        self.assertAlmostEqual(shifted["mean_cells"], 1.)
        self.assertAlmostEqual(shifted["origin_error_cells"], 1.)
        self.assertEqual(shifted["per_column_basis_error_cells"], 0)
        pred, target = fixture()
        pred["lattice"][1][0] += .004
        scaled = analysis.analyze_case(pred, target)["geometry"]
        self.assertAlmostEqual(scaled["origin_error_cells"], 0)
        self.assertAlmostEqual(scaled["per_column_basis_error_cells"], .1)
        self.assertAlmostEqual(scaled["mean_cells"], .5)
        self.assertAlmostEqual(scaled["max_cells"], 1.)

    def test_rotation_equivalence_is_preserved(self):
        pred, target = fixture()
        pred["layers"][0].update(type=2, rotation=3, column=3, row=8)
        target["layers"][0].update(type=2, pose_known=True, column=3, row=8, rotations=[True]*4)
        self.assertTrue(analysis.analyze_case(pred, target)["layers"]["active"]["exact"])
        target["layers"][0]["rotations"] = [True, False, True, False]
        item = analysis.analyze_case(pred, target)["layers"]["active"]
        self.assertFalse(item["rotation_correct"])
        self.assertFalse(item["exact"])

    def test_active_and_ghost_errors_remain_independent(self):
        pred, target = fixture()
        target["layers"][0].update(type=3, pose_known=True, column=4, row=7, rotations=[True, False, False, False])
        pred["layers"][0].update(type=3, column=4, row=9)
        result = analysis.analyze_case(pred, target)
        self.assertFalse(result["layers"]["active"]["row_correct"])
        self.assertTrue(result["layers"]["ghost"]["exact"])
        self.assertTrue(result["locked"]["exact_board"])

    def test_unknown_layers_have_no_type_or_pose_score(self):
        pred, target = fixture()
        target["layers"][1]["known"] = False
        first = analysis.analyze_case(pred, target)["layers"]["ghost"]
        target["layers"][1]["type"] = 7
        target["layers"][1]["row"] = 25
        second = analysis.analyze_case(pred, target)["layers"]["ghost"]
        self.assertEqual(first, second)
        self.assertIsNone(second["exact"])

    def test_right_type_cannot_hide_missing_occupancy(self):
        pred, target = fixture()
        target["occupied"][0][0] = target["typed"][0][0] = True
        target["semantic"][0][0] = pred["semantic"][0][0] = 3
        result = analysis.analyze_case(pred, target)["locked"]
        self.assertEqual(result["correct_type_head_cells"], 1)
        self.assertEqual(result["correct_joint_occupied_type_cells"], 0)

    def test_zero_logit_is_negative_and_not_confidently_known(self):
        pred, target = fixture()
        pred["occupancy_logits"][0][0] = 0.
        pred["layers"][0]["known_logit"] = 0.
        result = analysis.analyze_case(pred, target)
        self.assertTrue(result["locked"]["exact_board"])
        self.assertFalse(result["layers"]["active"]["exact"])
        self.assertEqual(result["locked"]["minimum_correct_class_probability"], .5)

    def test_wrong_shape_nonfinite_and_inconsistent_masks_refuse(self):
        for mutation in (lambda p, t: p["occupancy_logits"].pop(),
                         lambda p, t: p["lattice"][0].__setitem__(0, float("nan")),
                         lambda p, t: t["typed"][0].__setitem__(0, True),
                         lambda p, t: t.__setitem__("registered", False),
                         lambda p, t: t["layers"][0].__setitem__("pose_known", True)):
            pred, target = fixture()
            mutation(pred, target)
            with self.assertRaises(ValueError):
                analysis.analyze_case(pred, target)

    def test_aggregate_keeps_empty_nonempty_and_dependent_ancestry_separate(self):
        pred, target = fixture()
        first = analysis.analyze_case(pred, target)
        target["occupied"][0][0] = True
        second = analysis.analyze_case(pred, target)
        rows = [{"sample_id": str(i), "group_id": "same-parent", "split": "development", "diagnostics": value}
                for i, value in enumerate((first, second))]
        result = analysis.summarize(rows)["development"]
        self.assertEqual((result["images"], result["ancestry_groups"]), (2, 1))
        self.assertEqual(result["locked_populations"]["empty"]["exact_boards"], 1)
        self.assertEqual(result["locked_populations"]["nonempty"]["exact_boards"], 0)
        with self.assertRaises(ValueError):
            analysis.summarize([rows[0], rows[0]])
        with self.assertRaises(ValueError):
            analysis.summarize(rows*23)


if __name__ == "__main__":
    unittest.main()
