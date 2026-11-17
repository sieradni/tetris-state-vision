"""Integrity controls for the M55 column-consensus receipt.

`docs/M55_COLUMN_CONSENSUS.md` fixed the instrument, its constants, the instrument check, the inherited falsifier, the
M51/M52/M53/M54 bars, the three branches, the no-regression rule and the prediction before the run. These tests bind the
receipt to that: the target baseline still reproduces the committed artifact, M51/M52 are re-read from their own
receipts, M54's per-board table is the baseline this arm is judged against, the branch is the predeclared arithmetic on
its own numbers, the two named boards are reported against M50, the `20261027` repair is the one M55 predicted, and the
prediction the run judged is still in the text.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from column_consensus import CONSTANTS as COLUMN_CONSTANTS  # noqa: E402
from m55_column_consensus_read import (  # noqa: E402
    AUC_GATE, BOX_MEAN_CELLS, DETECTION_GATE_CELLS, M54_F1_AT_ZERO, READS, branch_for,
)

SCOPE = json.loads((ROOT/"docs/evidence/m55-column-consensus-read/scope.json").read_text())
M50 = json.loads((ROOT/"docs/evidence/m50-board-box-read/scope.json").read_text())
M51 = json.loads((ROOT/"docs/evidence/m51-board-box-read/scope.json").read_text())
M52 = json.loads((ROOT/"docs/evidence/m52-lattice-fit-read/scope.json").read_text())
M53 = json.loads((ROOT/"docs/evidence/m53-stack-anchor-read/scope.json").read_text())
M54 = json.loads((ROOT/"docs/evidence/m54-floor-consensus-read/scope.json").read_text())
DECLARATION = (ROOT/"docs/M55_COLUMN_CONSENSUS.md").read_text(encoding="utf-8")
BOARDS = SCOPE["geometry"]["detection"]["boards"]
BY_ID = {row["sample_id"]: row for row in BOARDS}


class DeclarationOrderTest(unittest.TestCase):
    def test_the_declaration_precedes_the_receipt(self):
        self.assertEqual(SCOPE["predeclaration"], "docs/M55_COLUMN_CONSENSUS.md")
        self.assertEqual(SCOPE["status"], "passed")
        self.assertIn("predeclared before any scoring run", DECLARATION)

    def test_the_protocol_is_the_campaign_protocol(self):
        self.assertEqual(SCOPE["population"]["cells"], 2986)
        self.assertEqual(SCOPE["population"]["boards"], 28)
        self.assertEqual(SCOPE["readout"]["features"], 30)
        self.assertTrue(SCOPE["readout"]["parameters_imported_unchanged"])
        self.assertEqual(tuple(SCOPE["measured"]), READS)

    def test_the_declared_constants_are_the_measured_ones(self):
        constants = SCOPE["geometry"]["anchoring"]["constants"]
        self.assertEqual(constants, COLUMN_CONSTANTS)
        self.assertEqual(constants["min_column_blocks"], 2)
        self.assertEqual(constants["column_tolerance_cells"], 0.15)
        self.assertEqual(constants["column_inset_max_cells"], 0.20)
        self.assertEqual(constants["min_pitch_span_cells"], 3.0)
        self.assertEqual(constants["min_floor_blocks"], 2)
        self.assertTrue(SCOPE["geometry"]["anchoring"]["label_free"])

    def test_the_bars_the_run_was_judged_against_are_the_declared_ones(self):
        bars = SCOPE["bars"]
        self.assertAlmostEqual(bars["m54_f1_at_zero"], 0.944724, places=6)
        self.assertAlmostEqual(bars["m54_auc"], 0.922077, places=6)
        self.assertAlmostEqual(bars["m54_mean_cells"], 0.363060, places=6)
        self.assertAlmostEqual(bars["m53_f1_at_zero"], 0.919192, places=6)
        self.assertAlmostEqual(bars["m51_auc"], 0.933275, places=6)
        self.assertAlmostEqual(bars["m52_auc"], 0.908824, places=6)

    def test_the_prediction_the_run_was_judged_against_is_still_in_the_text(self):
        self.assertIn("Prediction, standing as written whatever the outcome", DECLARATION)
        self.assertIn("| F1 at `0.0` | `[0.93, 0.97]` | `0.946309` | hit |", DECLARATION)
        self.assertIn("| population mean displacement | `[0.30, 0.35]` cells | `0.329292` | hit |", DECLARATION)
        self.assertIn("| branch | `column-fallback` | `column-fallback` | hit |", DECLARATION)
        self.assertIn("| `20261027-v0/-v1` repaired | yes | `0.101015` / `0.054625` | hit |", DECLARATION)


class BaselineTest(unittest.TestCase):
    def test_the_target_read_reproduces_the_committed_artifact(self):
        measured = SCOPE["measured"]["target_lattice"]
        self.assertAlmostEqual(measured["f1_at_zero"], 0.974763, places=6)
        self.assertAlmostEqual(measured["auc"], 0.999195, places=6)
        self.assertEqual(measured["counts_at_zero"]["true_positive"], 309)
        self.assertEqual(measured["counts_at_zero"]["false_negative"], 0)

    def test_m51s_box_and_m52s_fit_are_re_measured(self):
        self.assertAlmostEqual(SCOPE["measured"]["detected_box"]["f1_at_zero"],
                               M51["measured"]["detected_box"]["f1_at_zero"], places=9)
        self.assertAlmostEqual(SCOPE["measured"]["refined_lattice"]["auc"],
                               M52["measured"]["refined_lattice"]["auc"], places=9)
        self.assertAlmostEqual(SCOPE["detector"]["box_mean_magnitude"], 0.383358, places=6)

    def test_m54s_table_is_the_baseline_and_matches_its_own_receipt(self):
        m54_boards = {row["sample_id"]: row for row in M54["geometry"]["detection"]["boards"]}
        named = SCOPE["no_regression"]["named"]
        for sample in ("training-20261031-v1", "training-20261031-v2"):
            self.assertAlmostEqual(named[sample]["m54_consensus_mean_cells"],
                                   m54_boards[sample]["displacement"]["mean_magnitude"], places=6)
            self.assertAlmostEqual(named[sample]["m55_columns_mean_cells"], 0.0, places=6)

    def test_the_column_consensus_read_is_the_declared_sampling(self):
        measured = SCOPE["measured"]["column_consensus"]
        self.assertAlmostEqual(measured["f1_at_zero"], 0.946309, places=6)
        self.assertAlmostEqual(measured["auc"], 0.922746, places=6)
        self.assertEqual(measured["counts_at_zero"]["true_positive"], 282)
        self.assertEqual(measured["counts_at_zero"]["false_positive"], 5)
        self.assertEqual(measured["counts_at_zero"]["false_negative"], 27)
        self.assertEqual(SCOPE["geometry_shift"]["anchored_vs_target"]["mean_magnitude"],
                         SCOPE["detector"]["mean_magnitude"])


class BranchTest(unittest.TestCase):
    def test_the_branch_is_the_predeclared_arithmetic_on_its_own_numbers(self):
        detector = SCOPE["detector"]
        measured = SCOPE["measured"]["column_consensus"]
        expected = branch_for(detector["mean_magnitude"], detector["median_of_medians"],
                              measured["f1_at_zero"], measured["auc"])
        self.assertEqual(expected, "column-fallback")
        self.assertEqual(SCOPE["branch"]["verdict"], expected)
        self.assertGreater(detector["mean_magnitude"], DETECTION_GATE_CELLS)
        self.assertLessEqual(detector["mean_magnitude"], BOX_MEAN_CELLS)
        self.assertGreaterEqual(measured["auc"], AUC_GATE)

    def test_the_three_branches_are_reachable_and_disjoint(self):
        self.assertEqual(branch_for(0.10, 0.05, 0.96, 0.95), "column-accepted")
        self.assertEqual(branch_for(0.30, 0.08, 0.95, 0.95), "column-fallback")
        self.assertEqual(branch_for(0.30, 0.40, 0.95, 0.85), "column-worse")
        self.assertEqual(branch_for(0.10, 0.05, 0.95, 0.85), "column-worse")

    def test_the_verdict_reports_the_win_and_the_miss_without_reframing(self):
        measured = SCOPE["measured"]["column_consensus"]
        self.assertGreater(measured["f1_at_zero"], M54_F1_AT_ZERO)
        self.assertGreater(measured["auc"], 0.922077)
        self.assertGreater(SCOPE["detector"]["mean_magnitude"], DETECTION_GATE_CELLS)
        self.assertIn("**Verdict: `column-fallback`**", DECLARATION)
        self.assertIn("The inherited falsifier is still missed: `0.329292 > 0.25`", DECLARATION)


class GeometryTest(unittest.TestCase):
    def test_every_board_is_reported_with_both_decisions(self):
        self.assertEqual(len(BOARDS), 28)
        for row in BOARDS:
            for key in ("displacement", "box_displacement", "refined_displacement"):
                self.assertIn("mean_magnitude", row[key])
            self.assertEqual(set(row["fit"]), {"blocks", "box", "pairs", "span_cells", "pitch_pinned", "x", "y"})
            self.assertIn("columns", row["fit"]["x"])
            self.assertEqual(row["fit"]["x"]["correction"]["columns"], 0)
            self.assertFalse(row["fit"]["x"]["correction"]["applied"])

    def test_the_population_statistics_reduce_over_the_boards(self):
        detector = SCOPE["detector"]
        means = [row["displacement"]["mean_magnitude"] for row in BOARDS]
        self.assertAlmostEqual(detector["mean_magnitude"], sum(means)/len(means), places=5)
        self.assertEqual(detector["boards_over_gate"],
                         sum(1 for value in means if value > DETECTION_GATE_CELLS))
        self.assertAlmostEqual(detector["worst_board_mean"], max(means), places=6)
        self.assertAlmostEqual(detector["median_of_medians"],
                               sorted(row["displacement"]["median_magnitude"] for row in BOARDS)[14], places=6)

    def test_the_columns_are_certified_or_declined_by_the_rule(self):
        certified = [row["sample_id"] for row in BOARDS if row["fit"]["x"]["columns"]["kept"]]
        self.assertEqual(len(certified), 19)
        for row in BOARDS:
            columns = row["fit"]["x"]["columns"]
            if columns["kept"]:
                self.assertLessEqual(columns["left_phase"], COLUMN_CONSTANTS["column_inset_max_cells"])
                self.assertGreaterEqual(columns["right_phase"], 1.0-COLUMN_CONSTANTS["column_inset_max_cells"])
            else:
                self.assertEqual(columns["shift"], 0)

    def test_the_pitch_span_rule_refuses_the_sparse_boards(self):
        unpinned = sorted(row["sample_id"] for row in BOARDS
                          if not (row["fit"]["pitch_pinned"]["x"] and row["fit"]["pitch_pinned"]["y"]))
        self.assertEqual(unpinned, ["training-20261027-v0", "training-20261027-v1", "training-20261027-v2",
                                    "training-20261032-v2"])
        for sample in unpinned:
            row = BY_ID[sample]
            self.assertFalse(row["fit"]["pitch_pinned"]["x"] and row["fit"]["pitch_pinned"]["y"])
            self.assertLess(min(row["fit"]["span_cells"]["x"], row["fit"]["span_cells"]["y"]),
                            COLUMN_CONSTANTS["min_pitch_span_cells"])

    def test_the_20261027_repair_is_the_one_the_declaration_predicted(self):
        for sample, box_value in (("training-20261027-v0", 0.101015), ("training-20261027-v1", 0.054625)):
            self.assertAlmostEqual(BY_ID[sample]["displacement"]["mean_magnitude"], box_value, places=6)
            self.assertAlmostEqual(BY_ID[sample]["displacement"]["mean_magnitude"],
                                   BY_ID[sample]["box_displacement"]["mean_magnitude"], places=6)
            m54_row = {row["sample_id"]: row for row in M54["geometry"]["detection"]["boards"]}[sample]
            self.assertAlmostEqual(m54_row["displacement"]["mean_magnitude"], 0.654811, places=6)

    def test_the_two_named_boards_are_judged_against_m50_and_held_at_zero(self):
        named = SCOPE["no_regression"]["named"]
        m50_boards = {row["sample_id"]: row for row in M50["geometry"]["detection"]["boards"]}
        for sample in ("training-20261031-v1", "training-20261031-v2"):
            self.assertAlmostEqual(named[sample]["m50_mean_cells"], 0.058827, places=6)
            self.assertAlmostEqual(named[sample]["m50_mean_cells"],
                                   m50_boards[sample]["displacement"]["mean_magnitude"], places=6)
            self.assertAlmostEqual(named[sample]["m55_columns_mean_cells"], 0.0, places=6)

    def test_the_no_regression_rule_is_reported_against_all_five_baselines(self):
        rule = SCOPE["no_regression"]
        for table in ("against_m50", "against_m51", "against_m52", "against_m53", "against_m54"):
            self.assertIn(table, rule)
            self.assertEqual(sorted(rule[table]["breaches"]), [])
            self.assertTrue(rule[table]["cleared"])
        self.assertTrue(rule["cleared"])


class SuccessorTest(unittest.TestCase):
    def test_the_successors_are_named_with_numbers(self):
        self.assertIn("M56 - the well frame, now for both axes", DECLARATION)
        self.assertIn("**reshapes** M54's named successor", DECLARATION)
        self.assertIn("0.098579", DECLARATION)
        self.assertIn("M57", DECLARATION)
        self.assertIn("M58", DECLARATION)


if __name__ == "__main__":
    unittest.main()
