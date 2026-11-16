"""Integrity controls for the M54 floor-consensus receipt.

`docs/M54_FLOOR_CONSENSUS.md` fixed the instrument, its constants, the instrument check, the inherited falsifier, the
M51/M52/M53 bars, the three branches, the no-regression rule and the prediction before the run. These tests bind the
receipt to that: the target baseline still reproduces the committed artifact, M51's box and M52's fit are re-read from
their own receipts, M53's own per-board table is the baseline M54 is judged against, the branch is the predeclared
arithmetic on its own numbers, the two named boards are reported against M50, and the prediction the run judged - and
the miss it recorded - are still in the text.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from floor_consensus import CONSTANTS as CONSENSUS_CONSTANTS  # noqa: E402
from m54_floor_consensus_read import (  # noqa: E402
    AUC_GATE, BOX_MEAN_CELLS, DETECTION_GATE_CELLS, M53_F1_AT_ZERO, READS, branch_for,
)

SCOPE = json.loads((ROOT/"docs/evidence/m54-floor-consensus-read/scope.json").read_text())
M50 = json.loads((ROOT/"docs/evidence/m50-board-box-read/scope.json").read_text())
M51 = json.loads((ROOT/"docs/evidence/m51-board-box-read/scope.json").read_text())
M52 = json.loads((ROOT/"docs/evidence/m52-lattice-fit-read/scope.json").read_text())
M53 = json.loads((ROOT/"docs/evidence/m53-stack-anchor-read/scope.json").read_text())
DECLARATION = (ROOT/"docs/M54_FLOOR_CONSENSUS.md").read_text(encoding="utf-8")
BOARDS = SCOPE["geometry"]["detection"]["boards"]
BY_ID = {row["sample_id"]: row for row in BOARDS}


class DeclarationOrderTest(unittest.TestCase):
    def test_the_declaration_precedes_the_receipt(self):
        self.assertEqual(SCOPE["predeclaration"], "docs/M54_FLOOR_CONSENSUS.md")
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
        self.assertEqual(constants, CONSENSUS_CONSTANTS)
        self.assertEqual(constants["min_floor_blocks"], 2)
        self.assertEqual(constants["floor_agreement_cells"], 0.15)
        self.assertEqual(constants["min_anchors"], 2)
        self.assertTrue(SCOPE["geometry"]["anchoring"]["label_free"])

    def test_the_bars_the_run_was_judged_against_are_the_declared_ones(self):
        bars = SCOPE["bars"]
        self.assertAlmostEqual(bars["m53_f1_at_zero"], 0.919192, places=6)
        self.assertAlmostEqual(bars["m53_auc"], 0.910164, places=6)
        self.assertAlmostEqual(bars["m53_mean_cells"], 0.470531, places=6)
        self.assertAlmostEqual(bars["m51_f1_at_zero"], 0.904376, places=6)
        self.assertAlmostEqual(bars["m51_auc"], 0.933275, places=6)
        self.assertAlmostEqual(bars["m52_f1_at_zero"], 0.893268, places=6)
        self.assertAlmostEqual(bars["m52_auc"], 0.908824, places=6)

    def test_the_prediction_the_run_was_judged_against_is_still_in_the_text(self):
        self.assertIn("Prediction, standing as written whatever the outcome", DECLARATION)
        self.assertIn("| F1 at `0.0` | `[0.90, 0.95]` | `0.944724` | hit |", DECLARATION)
        self.assertIn("| population mean displacement | `[0.34, 0.39]` cells | `0.363060` | hit |", DECLARATION)
        self.assertIn("| branch | `floor-fallback` | `floor-fallback` | hit |", DECLARATION)


class BaselineTest(unittest.TestCase):
    def test_the_target_read_reproduces_the_committed_artifact(self):
        measured = SCOPE["measured"]["target_lattice"]
        self.assertAlmostEqual(measured["f1_at_zero"], 0.974763, places=6)
        self.assertAlmostEqual(measured["auc"], 0.999195, places=6)
        self.assertEqual(measured["counts_at_zero"]["true_positive"], 309)
        self.assertEqual(measured["counts_at_zero"]["false_negative"], 0)

    def test_m51s_box_is_re_measured_and_reproduces_m51(self):
        self.assertAlmostEqual(SCOPE["measured"]["detected_box"]["f1_at_zero"],
                               M51["measured"]["detected_box"]["f1_at_zero"], places=9)
        self.assertAlmostEqual(SCOPE["measured"]["detected_box"]["auc"],
                               M51["measured"]["detected_box"]["auc"], places=9)
        self.assertAlmostEqual(SCOPE["detector"]["box_mean_magnitude"], 0.383358, places=6)

    def test_m52s_fit_is_re_measured_and_reproduces_m52(self):
        self.assertAlmostEqual(SCOPE["measured"]["refined_lattice"]["f1_at_zero"],
                               M52["measured"]["refined_lattice"]["f1_at_zero"], places=9)
        self.assertAlmostEqual(SCOPE["measured"]["refined_lattice"]["auc"],
                               M52["measured"]["refined_lattice"]["auc"], places=9)
        self.assertAlmostEqual(SCOPE["detector"]["refined_mean_magnitude"], 0.380890, places=6)

    def test_m53s_table_is_the_baseline_and_matches_its_own_receipt(self):
        m53_boards = {row["sample_id"]: row for row in M53["geometry"]["detection"]["boards"]}
        self.assertEqual(set(m53_boards), set(BY_ID))
        # the two named boards must be reported with M53's own value, not a transcription
        named = SCOPE["no_regression"]["named"]
        for sample in ("training-20261031-v1", "training-20261031-v2"):
            self.assertAlmostEqual(named[sample]["m53_anchored_mean_cells"],
                                   m53_boards[sample]["displacement"]["mean_magnitude"], places=6)

    def test_the_consensus_read_is_the_declared_sampling(self):
        measured = SCOPE["measured"]["floor_consensus"]
        self.assertAlmostEqual(measured["f1_at_zero"], 0.944724, places=6)
        self.assertAlmostEqual(measured["auc"], 0.922077, places=6)
        self.assertEqual(measured["counts_at_zero"]["true_positive"], 282)
        self.assertEqual(measured["counts_at_zero"]["false_positive"], 6)
        self.assertEqual(measured["counts_at_zero"]["false_negative"], 27)
        self.assertEqual(SCOPE["geometry_shift"]["anchored_vs_target"]["mean_magnitude"],
                         SCOPE["detector"]["mean_magnitude"])


class BranchTest(unittest.TestCase):
    def test_the_branch_is_the_predeclared_arithmetic_on_its_own_numbers(self):
        detector = SCOPE["detector"]
        measured = SCOPE["measured"]["floor_consensus"]
        expected = branch_for(detector["mean_magnitude"], detector["median_of_medians"],
                              measured["f1_at_zero"], measured["auc"])
        self.assertEqual(expected, "floor-fallback")
        self.assertEqual(SCOPE["branch"]["verdict"], expected)
        self.assertGreater(detector["mean_magnitude"], DETECTION_GATE_CELLS)
        self.assertLessEqual(detector["mean_magnitude"], BOX_MEAN_CELLS)
        self.assertGreaterEqual(measured["auc"], AUC_GATE)

    def test_the_three_branches_are_reachable_and_disjoint(self):
        self.assertEqual(branch_for(0.10, 0.05, 0.95, 0.95), "floor-accepted")
        self.assertEqual(branch_for(0.30, 0.08, 0.94, 0.95), "floor-fallback")
        self.assertEqual(branch_for(0.30, 0.40, 0.94, 0.85), "floor-worse")
        self.assertEqual(branch_for(0.10, 0.05, 0.95, 0.85), "floor-worse")

    def test_the_verdict_reports_the_win_and_the_miss_without_reframing(self):
        measured = SCOPE["measured"]["floor_consensus"]
        self.assertGreater(measured["f1_at_zero"], M53_F1_AT_ZERO)
        self.assertGreater(measured["f1_at_zero"], SCOPE["measured"]["detected_box"]["f1_at_zero"])
        self.assertLess(measured["auc"], SCOPE["measured"]["detected_box"]["auc"])
        self.assertGreater(SCOPE["detector"]["mean_magnitude"], DETECTION_GATE_CELLS)
        self.assertIn("**Verdict: `floor-fallback`**", DECLARATION)
        self.assertIn("The falsifier is missed: mean `0.363060 > 0.25`", DECLARATION)
        self.assertIn("highest F1 at `0.0`", DECLARATION)


class GeometryTest(unittest.TestCase):
    def test_every_board_is_reported_with_its_row_decision(self):
        self.assertEqual(len(BOARDS), 28)
        for row in BOARDS:
            for key in ("displacement", "box_displacement", "refined_displacement"):
                self.assertIn("mean_magnitude", row[key])
            self.assertEqual(set(row["fit"]), {"blocks", "box", "pairs", "x", "y"})
            self.assertIn("correction", row["fit"]["y"])
            self.assertIn("floor", row["fit"]["y"])

    def test_the_population_statistics_reduce_over_the_boards(self):
        detector = SCOPE["detector"]
        means = [row["displacement"]["mean_magnitude"] for row in BOARDS]
        self.assertAlmostEqual(detector["mean_magnitude"], sum(means)/len(means), places=5)
        self.assertEqual(detector["boards_over_gate"],
                         sum(1 for value in means if value > DETECTION_GATE_CELLS))
        self.assertAlmostEqual(detector["worst_board_mean"], max(means), places=6)
        self.assertAlmostEqual(detector["median_of_medians"],
                               sorted(row["displacement"]["median_magnitude"] for row in BOARDS)[14], places=6)

    def test_the_row_correction_fired_only_where_the_floor_was_pinned_below_the_box(self):
        corrected = [row["sample_id"] for row in BOARDS if row["fit"]["y"]["rows_corrected"]]
        self.assertEqual(corrected, ["training-20261031-v1", "training-20261031-v2"])
        for sample in corrected:
            floor = BY_ID[sample]["fit"]["y"]["floor"]
            self.assertTrue(floor["kept"])
            self.assertGreaterEqual(floor["members"], 2)
            self.assertEqual(BY_ID[sample]["fit"]["y"]["rows_corrected"], 1)
            self.assertAlmostEqual(BY_ID[sample]["displacement"]["mean_magnitude"], 0.0, places=6)

    def test_the_unpinned_floors_are_the_declared_fallback(self):
        unpinned = sorted(row["sample_id"] for row in BOARDS if not row["fit"]["y"]["floor"]["kept"])
        self.assertEqual(unpinned, ["training-20261027-v0", "training-20261027-v1", "training-20261027-v2",
                                    "training-20261035-v2"])
        for sample in unpinned:
            self.assertEqual(BY_ID[sample]["fit"]["y"]["rows_corrected"], 0)

    def test_the_two_named_boards_are_judged_against_m50_and_held_at_zero(self):
        named = SCOPE["no_regression"]["named"]
        m50_boards = {row["sample_id"]: row for row in M50["geometry"]["detection"]["boards"]}
        for sample in ("training-20261031-v1", "training-20261031-v2"):
            self.assertAlmostEqual(named[sample]["m50_mean_cells"], 0.058827, places=6)
            self.assertAlmostEqual(named[sample]["m50_mean_cells"],
                                   m50_boards[sample]["displacement"]["mean_magnitude"], places=6)
            self.assertAlmostEqual(named[sample]["m54_consensus_mean_cells"], 0.0, places=6)
            self.assertAlmostEqual(named[sample]["m53_anchored_mean_cells"], 0.0, places=6)

    def test_the_no_regression_rule_is_reported_against_all_four_baselines(self):
        rule = SCOPE["no_regression"]
        for table in ("against_m50", "against_m51", "against_m52", "against_m53"):
            self.assertIn(table, rule)
            self.assertIn("breaches", rule[table])
        self.assertFalse(rule["cleared"])
        self.assertTrue(rule["against_m53"]["cleared"])
        self.assertEqual(sorted(rule["against_m53"]["breaches"]), [])
        for table in ("against_m50", "against_m51", "against_m52"):
            self.assertEqual(sorted(rule[table]["breaches"]),
                             ["training-20261027-v0", "training-20261027-v1"])


class SuccessorTest(unittest.TestCase):
    def test_the_successors_are_named_with_numbers(self):
        self.assertIn("M55 - x phase from column occupancy", DECLARATION)
        self.assertIn("M56 - the well frame, not the cell grid", DECLARATION)
        self.assertIn("`0.363060` to `0.316000`", DECLARATION)
        self.assertIn("`<= 0.22`", DECLARATION)
        self.assertIn("M57", DECLARATION)


if __name__ == "__main__":
    unittest.main()
