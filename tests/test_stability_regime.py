"""Tests for the read-only clip-saturation census (V-086)."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "stability_regime", Path(__file__).resolve().parents[1]/"scripts"/"stability_regime.py")
regime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(regime)


def write_run(root, name, records):
    path = Path(root)/name
    path.mkdir(parents=True, exist_ok=True)
    (path/"curve.jsonl").write_text("".join(json.dumps(record)+"\n" for record in records),
                                    encoding="utf-8")


class CurveNormTests(unittest.TestCase):
    def test_reads_recorded_norms_in_order(self):
        with tempfile.TemporaryDirectory() as root:
            write_run(root, "run", [{"step": 1, "gradient_norm": 118.29}, {"step": 2, "gradient_norm": 1.657}])
            values = regime.curve_norms(Path(root)/"run"/"curve.jsonl")
        self.assertEqual(values, [118.29, 1.657])

    def test_skips_malformed_and_non_finite_records(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"curve.jsonl"
            path.write_text("\n".join([
                "not json at all",
                "{broken",
                json.dumps(["a", "list"]),
                json.dumps({"step": 1}),
                json.dumps({"step": 2, "gradient_norm": None}),
                json.dumps({"step": 3, "gradient_norm": True}),
                json.dumps({"step": 4, "gradient_norm": "9.0"}),
                json.dumps({"step": 5, "gradient_norm": 4.5}),
            ])+"\n", encoding="utf-8")
            self.assertEqual(regime.curve_norms(path), [4.5])


class CensusTests(unittest.TestCase):
    def test_saturation_and_fully_clipped_runs(self):
        summary = regime.census({"a": [2.0, 30.0], "b": [0.5, 12.0]})
        self.assertEqual(summary["runs"], 2)
        self.assertEqual(summary["steps"], 4)
        self.assertEqual(summary["clipped_steps"], 3)
        self.assertAlmostEqual(summary["saturation"], 0.75)
        self.assertEqual(summary["fully_clipped_runs"], 1)
        self.assertEqual(summary["unclipped_runs"], 1)
        self.assertEqual((summary["min_norm"], summary["max_norm"]), (0.5, 30.0))

    def test_exactly_at_the_clip_is_not_clipped(self):
        summary = regime.census({"a": [regime.CLIP_NORM]})
        self.assertEqual(summary["clipped_steps"], 0)
        self.assertEqual(summary["saturation"], 0.0)

    def test_empty_census_does_not_divide_by_zero(self):
        summary = regime.census({})
        self.assertEqual(summary["steps"], 0)
        self.assertIsNone(summary["saturation"])
        self.assertEqual(summary["fully_clipped_runs"], 0)


class CollectTests(unittest.TestCase):
    def test_groups_by_run_directory_and_ignores_empty_files(self):
        with tempfile.TemporaryDirectory() as root:
            write_run(root, "alpha", [{"gradient_norm": 5.0}])
            write_run(root, "beta", [{"gradient_norm": 0.25}, {"gradient_norm": 9.0}])
            write_run(root, "empty", [{"step": 1}])
            summary = regime.report(root)
        self.assertEqual(set(summary["per_run"]), {"alpha", "beta"})
        self.assertEqual(summary["steps"], 3)
        self.assertEqual(summary["fully_clipped_runs"], 1)
        self.assertEqual(summary["root"], str(root).replace("\\", "/"))

    def test_report_is_json_serialisable(self):
        with tempfile.TemporaryDirectory() as root:
            write_run(root, "alpha", [{"gradient_norm": 5.0}])
            json.dumps(regime.report(root), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
