"""Every quoted power figure must equal what the analysis recomputes from the receipts.

This is the guard V-096 asks for: the method document's figure block and the headline figures in the
roadmap and continuation state are compared against a fresh `power_analysis.analyse()` run, so a hand
edit to a table, a stale evidence directory or a drifted number fails the suite instead of surviving as
prose.

The comparisons are plain functions so they can be pointed at tampered text as well as at the
committed files. The mutation tests at the end do exactly that: they perturb a figure in the generated
block, a row of the `20261103` decomposition table and a number in the committed evidence, and require
the guard to raise with a message naming the offending figure. A guard that accepted a mutation would
fail those tests, so the enforcement is demonstrated rather than assumed.
"""
import json
import math
import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]

import power_analysis as power

EVIDENCE = ROOT/"docs/evidence/m3-power-mde-r3/result.json"
DOC = ROOT/"docs/M3_MINIMUM_DETECTABLE_EFFECT.md"
REPLICATION = ROOT/"docs/M3_COMPUTE_REPLICATION.md"
ROADMAP = ROOT/"docs/ROADMAP.md"
STATE = ROOT/"docs/CONTINUATION_STATE.md"


def walk(actual, expected, path="result"):
    """Numbers within 1e-9 relative, strings exactly; the path of the first difference is named."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f"{path}: keys differ from the recomputation")
        for key in expected:
            walk(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: list shape differs from the recomputation")
        for index, (left, right) in enumerate(zip(actual, expected)):
            walk(left, right, f"{path}[{index}]")
    elif isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise AssertionError(f"{path}: {actual!r} != {expected!r}")
    elif actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def check_evidence(stored, result):
    """The committed evidence must equal the recomputation field by field, naming the first miss."""
    walk(stored, result)


def block_of(text):
    begin, end = power.MARKERS
    first, second = text.find(begin), text.find(end)
    if first < 0 or second < 0:
        raise AssertionError("the document has no power-figures block")
    return text[first:second+len(end)].strip("\n")


def check_document_block(text, result):
    expected = power.render(result).strip("\n")
    actual = block_of(text)
    if actual == expected:
        return
    for index, (left, right) in enumerate(zip(actual.splitlines(), expected.splitlines())):
        if left != right:
            raise AssertionError(f"figure block line {index+1} differs: found {left!r}, "
                                 f"expected {right!r}")
    raise AssertionError(
        f"figure block length differs: found {len(actual.splitlines())} lines, "
        f"expected {len(expected.splitlines())}")


def budget_row(result, seed):
    row = result["decomposition_deltas"]["budget192"][str(seed)]
    parts = row["parts"]
    return (f"| `{seed}` | {row['auc']:+.3f} | {parts['ranking_best_boundary']:+.3f} | "
            f"{parts['zero_boundary_cost']:+.3f} | {parts['transfer_cost']:+.3f} | "
            f"{row['cv_f1']:+.3f} |")


def schedule_triple(result, seed):
    parts = result["decomposition_deltas"]["cosine128"][str(seed)]["parts"]
    return (f"{parts['ranking_best_boundary']:+.3f}/{parts['zero_boundary_cost']:+.3f}/"
            f"{parts['transfer_cost']:+.3f}")


def check_replication_rows(text, result):
    for seed in power.SEEDS:
        expected = budget_row(result, seed)
        if expected not in text:
            raise AssertionError(f"decomposition row for budget192 seed {seed} is stale or missing; "
                                 f"expected {expected!r}")
        triple = schedule_triple(result, seed)
        if triple not in text:
            raise AssertionError(f"schedule decomposition triple for seed {seed} is stale or "
                                 f"missing; expected {triple!r}")


def band_designs(result):
    band = result["levers"]["budget192"]["variance"]["band"]
    seeds, hours = band["seeds"], band["arm_hours"]
    return (f"{seeds['low']} / {seeds['point']} / {seeds['high']} seeds and "
            f"{hours['low']:.2f} / {hours['point']:.2f} / {hours['high']:.2f} arm-hours")


def check_headlines(roadmap, state, result):
    head = power.headline(result)
    for name, value in head.items():
        if value not in state:
            raise AssertionError(f"continuation state is stale for {name}: expected {value!r}")
    for name in ("budget_mde", "schedule_mde", "audit_mde", "band_seeds", "band_arm_hours",
                 "floor_design"):
        if head[name] not in roadmap:
            raise AssertionError(f"roadmap is stale for {name}: expected {head[name]!r}")
    designs = band_designs(result)
    for label, text in (("roadmap", roadmap), ("continuation state", state)):
        if designs not in text:
            raise AssertionError(f"{label} does not carry the band's per-design counts: "
                                 f"expected {designs!r}")


class PowerFigures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = power.analyse()

    def test_the_committed_evidence_equals_a_fresh_recomputation(self):
        check_evidence(json.loads(EVIDENCE.read_text(encoding="utf-8")), self.result)

    def test_the_document_block_is_exactly_the_rendered_block(self):
        check_document_block(DOC.read_text(encoding="utf-8"), self.result)

    def test_the_replication_document_decomposition_rows_match_recomputation(self):
        check_replication_rows(REPLICATION.read_text(encoding="utf-8"), self.result)

    def test_every_headline_figure_quoted_in_the_ledger_matches(self):
        check_headlines(ROADMAP.read_text(encoding="utf-8"),
                        STATE.read_text(encoding="utf-8"), self.result)

    def test_the_band_brackets_the_point_and_moves_the_counts_apart(self):
        for lever in power.LEVERS:
            band = self.result["levers"][lever]["variance"]["band"]
            cost = self.result["cost"][lever]
            self.assertLessEqual(band["variance_low"], band["variance"])
            self.assertLessEqual(band["variance"], band["variance_high"])
            self.assertLessEqual(cost["audit_low"]["seeds"], cost["audit_effect"]["seeds"])
            self.assertLessEqual(cost["audit_effect"]["seeds"], cost["audit_high"]["seeds"])

    def test_the_band_factors_are_the_two_degree_scale(self):
        band = power.chi_square_band(1.0)
        self.assertAlmostEqual(band["factor_low"], 2/(-2*math.log(0.05)), places=12)
        self.assertAlmostEqual(band["factor_high"], 2/(-2*math.log(0.95)), places=12)
        self.assertLess(band["factor_low"], 0.34)
        self.assertGreater(band["factor_high"], 19.0)

    def test_the_decomposition_parts_sum_to_the_primary_everywhere(self):
        # The stored parts are each rounded to six decimals, so three of them can miss the primary
        # by 3e-6; the identity is exact before rounding and is checked to that stored precision.
        for lever in power.LEVERS:
            for arm in power.ARMS:
                for seed, row in self.result["decomposition"][lever][arm].items():
                    self.assertAlmostEqual(sum(row["parts"].values()), row["cv_f1"], places=5)
            for seed, deltas in self.result["decomposition_deltas"][lever].items():
                self.assertAlmostEqual(sum(deltas["parts"].values()), deltas["cv_f1"], places=5)

    def test_a_hand_edit_to_a_figure_in_the_block_is_rejected(self):
        text = DOC.read_text(encoding="utf-8")
        quoted = f"{self.result['levers']['budget192']['minimum_detectable_effect']:.6f}"
        tampered = text.replace(quoted, "0.083251", 1)
        self.assertNotEqual(tampered, text, "the mutation did not land in the document")
        check_document_block(text, self.result)
        with self.assertRaisesRegex(AssertionError, "figure block line"):
            check_document_block(tampered, self.result)

    def test_a_hand_edit_to_the_worked_example_is_rejected(self):
        text = REPLICATION.read_text(encoding="utf-8")
        row = self.result["decomposition_deltas"]["budget192"]["20261103"]["parts"]
        original = f"{row['ranking_best_boundary']:+.3f}"
        tampered = text.replace(f"| `20261103` | -0.030 | {original} |",
                                "| `20261103` | -0.030 | -0.159 |", 1)
        self.assertNotEqual(tampered, text, "the mutation did not land in the table")
        check_replication_rows(text, self.result)
        with self.assertRaisesRegex(AssertionError, "20261103"):
            check_replication_rows(tampered, self.result)

    def test_a_hand_edit_to_the_committed_evidence_is_rejected(self):
        stored = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        stored["levers"]["budget192"]["minimum_detectable_effect"] = round(
            stored["levers"]["budget192"]["minimum_detectable_effect"]+0.001, 6)
        with self.assertRaisesRegex(AssertionError, "minimum_detectable_effect"):
            walk(stored, self.result)

    def test_a_perturbed_evidence_copy_in_a_temp_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory)/"result.json"
            data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
            data["levers"]["cosine128"]["standard_error_bootstrap"] = 0.07
            copy.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "standard_error_bootstrap"):
                walk(json.loads(copy.read_text(encoding="utf-8")), self.result)


if __name__ == "__main__":
    unittest.main()
