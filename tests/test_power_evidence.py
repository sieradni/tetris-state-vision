"""Every quoted power figure must equal what the analysis recomputes from the receipts.

This is the guard V-096 asks for: the method document's figure block and the headline figures in the
roadmap and continuation state are compared against a fresh `power_analysis.analyse()` run, so a hand
edit to a table, a stale evidence directory, or a drifted number fails the suite instead of surviving
as prose.
"""
import json
import math
import statistics
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]

import power_analysis as power

EVIDENCE = ROOT/"docs/evidence/m3-power-mde-r3/result.json"
DOC = ROOT/"docs/M3_MINIMUM_DETECTABLE_EFFECT.md"
ROADMAP = ROOT/"docs/ROADMAP.md"
STATE = ROOT/"docs/CONTINUATION_STATE.md"


def walk(actual, expected, path="result"):
    """Numbers within 1e-9 relative, strings exactly; any structural difference is a failure."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f"{path}: keys differ: "
                                 f"{sorted(set(expected) ^ set(actual)) if isinstance(actual, dict) else type(actual)}")
        for key in expected:
            walk(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: list shape differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            walk(left, right, f"{path}[{index}]")
    elif isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise AssertionError(f"{path}: {actual!r} != {expected!r}")
    elif actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def block_of(text):
    begin, end = power.MARKERS
    first, second = text.find(begin), text.find(end)
    if first < 0 or second < 0:
        raise AssertionError("the document has no figure markers")
    return text[first:second+len(end)].strip("\n")


class PowerFigures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = power.analyse()
        cls.head = power.headline(cls.result)

    def test_the_committed_evidence_equals_a_fresh_recomputation(self):
        stored = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        walk(stored, self.result)

    def test_the_document_block_is_exactly_the_rendered_block(self):
        self.assertEqual(block_of(DOC.read_text(encoding="utf-8")),
                         power.render(self.result).strip("\n"))

    def test_every_headline_figure_quoted_in_the_ledger_matches(self):
        roadmap = ROADMAP.read_text(encoding="utf-8")
        state = STATE.read_text(encoding="utf-8")
        for name, value in self.head.items():
            self.assertIn(value, state, f"continuation state is stale for {name}")
        for name in ("budget_mde", "schedule_mde", "audit_mde", "band_seeds", "band_arm_hours",
                     "floor_design"):
            self.assertIn(self.head[name], roadmap, f"roadmap is stale for {name}")

    def test_the_band_brackets_the_point_and_moves_the_counts_apart(self):
        for lever in power.LEVERS:
            variance = self.result["levers"][lever]["variance"]
            band, cost = variance["band"], self.result["cost"][lever]
            self.assertLessEqual(band["variance_low"], band["variance"])
            self.assertLessEqual(band["variance"], band["variance_high"])
            self.assertLessEqual(cost["audit_low"]["seeds"], cost["audit_effect"]["seeds"])
            self.assertLessEqual(cost["audit_effect"]["seeds"], cost["audit_high"]["seeds"])
            self.assertGreaterEqual(cost["audit_high"]["seeds"], cost["floor"]["seeds"] // 2)

    def test_the_band_factors_are_the_two_degree_scale(self):
        band = power.chi_square_band(1.0)
        self.assertAlmostEqual(band["factor_low"], 2/(-2*math.log(0.05)), places=12)
        self.assertAlmostEqual(band["factor_high"], 2/(-2*math.log(0.95)), places=12)
        self.assertLess(band["factor_low"], 0.34)
        self.assertGreater(band["factor_high"], 19.0)

    def test_a_stale_evidence_number_is_caught(self):
        tampered = json.loads(json.dumps(self.result))
        tampered["levers"]["budget192"]["minimum_detectable_effect"] += 0.001
        with self.assertRaises(AssertionError):
            walk(tampered, self.result)

    def test_the_decomposition_parts_sum_to_the_primary_everywhere(self):
        # The stored parts are each rounded to six decimals, so three of them can miss the primary
        # by 3e-6; the identity is exact before rounding and is checked to that stored precision.
        for lever in power.LEVERS:
            for arm in power.ARMS:
                for seed, row in self.result["decomposition"][lever][arm].items():
                    self.assertAlmostEqual(sum(row["parts"].values()), row["cv_f1"], places=5)
            for seed, deltas in self.result["decomposition_deltas"][lever].items():
                self.assertAlmostEqual(sum(deltas["parts"].values()), deltas["cv_f1"], places=5)


if __name__ == "__main__":
    unittest.main()
