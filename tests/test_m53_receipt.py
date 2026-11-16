"""Integrity controls for the M53 stack-anchor receipt.

`docs/M53_STACK_ANCHOR.md` fixed the instrument, its constants, the instrument check, the inherited falsifier, the three
branches, the predecessor bars and the no-regression rule before the run. These tests bind the receipt to that: the
target baseline still reproduces the committed artifact, M51's box and M52's fit are re-read from their own receipts,
the anchored sampling is the declared label-free instrument, the branch is the predeclared arithmetic on its own
numbers, the two named boards are reported against M50, and the prediction the run judged is still in the text.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from m53_stack_anchor_read import (  # noqa: E402
    AUC_GATE, BOX_MEAN_CELLS, DETECTION_GATE_CELLS, READS, branch_for,
)
from stack_anchor import CONSTANTS as ANCHOR_CONSTANTS  # noqa: E402

SCOPE = json.loads((ROOT/"docs/evidence/m53-stack-anchor-read/scope.json").read_text())
M50 = json.loads((ROOT/"docs/evidence/m50-board-box-read/scope.json").read_text())
M51 = json.loads((ROOT/"docs/evidence/m51-board-box-read/scope.json").read_text())
M52 = json.loads((ROOT/"docs/evidence/m52-lattice-fit-read/scope.json").read_text())
DECLARATION = (ROOT/"docs/M53_STACK_ANCHOR.md").read_text(encoding="utf-8")


class DeclarationOrderTest(unittest.TestCase):
    def test_the_declaration_precedes_the_receipt(self):
        self.assertEqual(SCOPE["predeclaration"], "docs/M53_STACK_ANCHOR.md")
        self.assertEqual(SCOPE["status"], "passed")
        self.assertIn("predeclared before any scoring run", DECLARATION)

    def test_the_protocol_is_the_campaign_protocol(self):
        self.assertEqual(SCOPE["population"]["cells"], 2986)
        self.assertEqual(SCOPE["population"]["boards"], 28)
        self.assertEqual(SCOPE["readout"]["features"], 30)
        self.assertTrue(SCOPE["readout"]["parameters_imported_unchanged"])
        self.assertEqual(tuple(SCOPE["measured"]), READS)

    def test_the_declared_constants_are_the_measured_ones(self):
        self.assertEqual(SCOPE["geometry"]["anchoring"]["constants"], ANCHOR_CONSTANTS)
        self.assertTrue(SCOPE["geometry"]["anchoring"]["label_free"])
        self.assertEqual(ANCHOR_CONSTANTS["floor_tolerance"], 1.0)
        self.assertEqual(ANCHOR_CONSTANTS["min_anchors"], 2)

    def test_the_prediction_the_run_was_judged_against_is_still_in_the_text(self):
        self.assertIn("Prediction, standing as written whatever the outcome", DECLARATION)
        self.assertIn("| F1 at `0.0` | `[0.80, 0.90]` | `0.919192` | **missed high**", DECLARATION)
        self.assertIn("| population mean displacement | `[0.42, 0.55]` cells | `0.470531` | hit |", DECLARATION)
        self.assertIn("| branch | `anchor-worse` | `anchor-worse` | hit |", DECLARATION)


class BaselineTest(unittest.TestCase):
    def test_the_target_read_reproduces_the_committed_artifact(self):
        self.assertAlmostEqual(SCOPE["measured"]["target_lattice"]["f1_at_zero"], 0.974763, places=6)
        self.assertAlmostEqual(SCOPE["measured"]["target_lattice"]["auc"], 0.999195, places=6)
        self.assertEqual(SCOPE["measured"]["target_lattice"]["counts_at_zero"]["true_positive"], 309)
        self.assertEqual(SCOPE["measured"]["target_lattice"]["counts_at_zero"]["false_negative"], 0)

    def test_m51s_box_is_re_measured_and_reproduces_m51(self):
        measured = SCOPE["measured"]["detected_box"]
        own = M51["measured"]["detected_box"]
        self.assertAlmostEqual(measured["f1_at_zero"], own["f1_at_zero"], places=9)
        self.assertAlmostEqual(measured["auc"], own["auc"], places=9)
        self.assertAlmostEqual(SCOPE["detector"]["box_mean_magnitude"], 0.383358, places=6)

    def test_m52s_fit_is_re_measured_and_reproduces_m52(self):
        measured = SCOPE["measured"]["refined_lattice"]
        own = M52["measured"]["refined_lattice"]
        self.assertAlmostEqual(measured["f1_at_zero"], own["f1_at_zero"], places=9)
        self.assertAlmostEqual(measured["auc"], own["auc"], places=9)
        self.assertAlmostEqual(SCOPE["detector"]["refined_mean_magnitude"], 0.380890, places=6)

    def test_the_anchored_sampling_is_the_declared_read(self):
        anchored = SCOPE["measured"]["stack_anchor"]
        self.assertEqual(SCOPE["geometry_shift"]["anchored_vs_target"]["mean_magnitude"],
                         SCOPE["detector"]["mean_magnitude"])
        self.assertAlmostEqual(anchored["f1_at_zero"], 0.919192, places=6)
        self.assertAlmostEqual(anchored["auc"], 0.910164, places=6)
        self.assertEqual(anchored["counts_at_zero"]["true_positive"], 273)
        self.assertEqual(anchored["counts_at_zero"]["false_positive"], 12)
        self.assertEqual(anchored["counts_at_zero"]["false_negative"], 36)


class BranchTest(unittest.TestCase):
    def test_the_branch_is_the_predeclared_arithmetic_on_its_own_numbers(self):
        detector = SCOPE["detector"]
        anchored = SCOPE["measured"]["stack_anchor"]
        expected = branch_for(detector["mean_magnitude"], detector["median_of_medians"],
                              anchored["f1_at_zero"], anchored["auc"])
        self.assertEqual(expected, "anchor-worse")
        self.assertEqual(SCOPE["branch"]["verdict"], expected)
        self.assertGreater(detector["mean_magnitude"], DETECTION_GATE_CELLS)
        self.assertGreater(detector["mean_magnitude"], BOX_MEAN_CELLS)
        self.assertGreaterEqual(anchored["auc"], AUC_GATE)

    def test_the_three_branches_are_reachable_and_disjoint(self):
        self.assertEqual(branch_for(0.10, 0.05, 0.95, 0.95), "anchor-accepted")
        self.assertEqual(branch_for(0.30, 0.08, 0.92, 0.95), "anchor-starved")
        self.assertEqual(branch_for(0.30, 0.40, 0.92, 0.85), "anchor-worse")
        self.assertEqual(branch_for(0.10, 0.05, 0.95, 0.85), "anchor-worse")


class GeometryTest(unittest.TestCase):
    def test_every_board_is_reported_against_both_predecessors(self):
        boards = SCOPE["geometry"]["detection"]["boards"]
        self.assertEqual(len(boards), 28)
        for row in boards:
            for key in ("displacement", "box_displacement", "refined_displacement"):
                self.assertIn("mean_magnitude", row[key])
            self.assertEqual(set(row["fit"]), {"blocks", "box", "pairs", "x", "y"})

    def test_the_population_statistics_reduce_over_the_boards(self):
        boards = SCOPE["geometry"]["detection"]["boards"]
        detector = SCOPE["detector"]
        means = [row["displacement"]["mean_magnitude"] for row in boards]
        self.assertAlmostEqual(detector["mean_magnitude"], sum(means)/len(means), places=5)
        self.assertEqual(detector["boards_over_gate"],
                         sum(1 for value in means if value > DETECTION_GATE_CELLS))
        self.assertEqual(detector["boards_kept_by_fit"],
                         sum(1 for row in boards if row["fit"]["x"]["kept"] and row["fit"]["y"]["kept"]))

    def test_the_two_named_boards_are_judged_against_m50_and_are_placed_exactly(self):
        named = SCOPE["no_regression"]["named"]
        for sample in ("training-20261031-v1", "training-20261031-v2"):
            self.assertAlmostEqual(named[sample]["m50_mean_cells"], 0.058827, places=6)
            self.assertAlmostEqual(named[sample]["m51_mean_cells"], 0.924144, places=6)
            self.assertAlmostEqual(named[sample]["m53_anchored_mean_cells"], 0.0, places=6)
        m50_named = M50["geometry"]["detection"]["boards"]
        for row in m50_named:
            if row["sample_id"] in named:
                self.assertAlmostEqual(named[row["sample_id"]]["m50_mean_cells"],
                                       row["displacement"]["mean_magnitude"], places=6)

    def test_the_no_regression_rule_is_reported_against_all_three_baselines(self):
        rule = SCOPE["no_regression"]
        for table in ("against_m50", "against_m51", "against_m52"):
            self.assertIn(table, rule)
            self.assertIn("breaches", rule[table])
        self.assertFalse(rule["cleared"])
        self.assertEqual(set(rule["against_m50"]["breaches"]),
                         {"training-20261027-v0", "training-20261027-v1", "training-20261027-v2",
                          "training-20261033-v0", "training-20261033-v1", "training-20261033-v2"})


class VerdictTest(unittest.TestCase):
    def test_the_verdict_does_not_reframe_the_miss(self):
        anchored = SCOPE["measured"]["stack_anchor"]
        self.assertEqual(SCOPE["branch"]["verdict"], "anchor-worse")
        self.assertIn("The anchors lose to M51's box", DECLARATION)
        self.assertIn("`cleared`: false", DECLARATION)
        # a win is reported as a win, and the win that exists is stated as one
        self.assertGreater(anchored["f1_at_zero"], SCOPE["measured"]["detected_box"]["f1_at_zero"])
        self.assertIn("highest F1 at the fixed", DECLARATION)

    def test_the_declared_predecessor_bars_are_stated_as_split(self):
        anchored = SCOPE["measured"]["stack_anchor"]
        m51 = SCOPE["measured"]["detected_box"]
        m52 = SCOPE["measured"]["refined_lattice"]
        self.assertGreater(anchored["f1_at_zero"], m51["f1_at_zero"])
        self.assertGreater(anchored["f1_at_zero"], m52["f1_at_zero"])
        self.assertLess(anchored["auc"], m51["auc"])
        self.assertGreater(anchored["auc"], m52["auc"])
        self.assertIn("split", DECLARATION)

    def test_the_successors_are_named_with_numbers(self):
        self.assertIn("M54 - floor consensus by block count", DECLARATION)
        self.assertIn("MIN_FLOOR_BLOCKS = 2", DECLARATION)
        self.assertIn("M55 - x phase from column occupancy", DECLARATION)
        self.assertIn("`0.470531` to `<= 0.25`", DECLARATION)


if __name__ == "__main__":
    unittest.main()
