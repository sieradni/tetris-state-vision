import copy
import unittest

from type_forgetting_analysis import analyze


def fixture():
    cases = []
    for i in range(32):
        label = i // 4
        roles = {role: {"target_known": True, "target_type": label, "target_pose_known": label != 0, "type_correct": label == 0}
                 for role in ("active", "ghost")}
        cases.append({"sample_id": str(i), "group_id": str(i//2), "features": [[0.0]*64 for _ in range(2)],
                      "logits": [[0.0]*8 for _ in range(2)], "roles": roles})
    return [{"step": step, "cases": copy.deepcopy(cases),
             "head": {"weight": [[0.0]*64 for _ in range(8)], "bias": [0.0]*8}}
            for step in range(0, 1025, 64)]


class ForgettingAnalysisTest(unittest.TestCase):
    def test_equal_logits_have_expected_class_cosine(self):
        result = analyze(fixture())
        self.assertEqual(result["timeline"][0]["correct_types"], 8)
        self.assertEqual(result["timeline"][0]["negative_class_pairs"], 28)
        self.assertAlmostEqual(result["timeline"][0]["class_gradient_cosines"][0][1], -1/7)
        self.assertTrue(all(t["correct_to_wrong"] == t["wrong_to_correct"] == 0 for t in result["transitions"]))

    def test_transition_and_unknown_refusal(self):
        points = fixture()
        for point in points[1:]:
            point["cases"][4]["logits"][0][1] = 1.0
            point["cases"][4]["roles"]["active"]["type_correct"] = True
        result = analyze(points)
        self.assertEqual(result["transitions"][0]["wrong_to_correct"], 1)
        self.assertEqual(result["timeline"][1]["correct_types"], 9)
        points[0]["cases"][0]["roles"]["active"]["target_known"] = False
        with self.assertRaises(ValueError):
            analyze(points)


if __name__ == "__main__":
    unittest.main()
