"""Integrity controls for the M41 provenance-isolating receipt.

`docs/M41_SUPPORT_CONTRAST.md` fixes the bar (`0.372881` / `0.689775`) and the three discriminating outcomes
before the run, and declares the arm as M35's `60` inputs sampled at the support lattice. These tests bind the
receipt to those commitments: the recorded configuration is identical to M35's in every parameter count, the
verdict is the predeclared REPRODUCES branch rather than a post-hoc reading, the standing gate is reported,
and the split sums.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPE = json.loads((ROOT/"docs/evidence/m41-support-contrast-r1/scope.json").read_text())
DIAGNOSIS = json.loads((ROOT/"docs/evidence/m41-support-contrast-r1/diagnosis.json").read_text())
M35 = json.loads((ROOT/"docs/evidence/m35-contrast-lattice-r1/scope.json").read_text())


def manifest(run):
    return json.loads((ROOT/"runs"/run/"checkpoint/manifest.json").read_text())


class ConfigIdentityTest(unittest.TestCase):
    def test_the_arm_reads_the_declared_sixty_inputs(self):
        self.assertEqual(SCOPE["arm"]["inputs"], 60)
        self.assertEqual(SCOPE["arm"]["features"], 30)
        self.assertEqual(SCOPE["predeclaration"], "docs/M41_SUPPORT_CONTRAST.md")

    def test_provenance_is_the_only_varied_recorded_fact_against_m35(self):
        arm, partner = manifest("m41-support-contrast"), manifest("m35-contrast")
        self.assertEqual(SCOPE["arm"]["run"], "m41-support-contrast")
        self.assertEqual(arm["parameters"], partner["parameters"])
        self.assertEqual(arm["parameters"], 8025482)
        # the recorded settings that carried the confound in the previous pass are identical
        for key in ("dimension", "blocks", "occupancy_read_standardise", "occupancy_read_level",
                    "occupancy_stack_monotone", "occupancy_deform", "occupancy_context",
                    "occupancy_detail", "lattice_support", "occupancy_normalise", "occupancy_centre",
                    "occupancy_crossing_threshold", "component_capacity", "component_identity_readout",
                    "component_endpoint_term", "occupancy_factor", "occupancy_positive_weight"):
            self.assertEqual(arm["config"].get(key), partner["config"].get(key), key)
        # and the form id is the only field that differs - it carries the geometry and nothing else here
        differing = sorted(k for k in arm["config"] if arm["config"].get(k) != partner["config"].get(k))
        self.assertEqual(differing, ["occupancy_pixel_query"])
        self.assertEqual((arm["config"]["occupancy_pixel_query"],
                          partner["config"]["occupancy_pixel_query"]), (15, 8))
        self.assertNotEqual(arm["weights_sha256"], partner["weights_sha256"])


class DeclaredTest(unittest.TestCase):
    def test_the_bar_is_the_winners_rescored_numbers(self):
        self.assertEqual(SCOPE["bar"]["f1_at_zero"], 0.372881)
        self.assertEqual(SCOPE["bar"]["best"], 0.689775)
        self.assertFalse(SCOPE["bar"]["met"])
        self.assertFalse(SCOPE["cleared"])

    def test_the_standing_gate_is_reported(self):
        self.assertEqual(SCOPE["gates"], {"per_cell_f1": False, "ranking": False, "attribution": True})
        self.assertEqual(SCOPE["gates_anchored"], SCOPE["gates"])

    def test_the_verdict_is_the_predeclared_branch_not_a_post_hoc_reading(self):
        # REPRODUCES required AUC <= 0.910000 and best swept <= 0.480000, fixed before the run
        self.assertLessEqual(SCOPE["own"]["auc"], 0.910000)
        self.assertLessEqual(SCOPE["own_best"], 0.480000)


class MeasuredTest(unittest.TestCase):
    def test_the_receipt_is_the_one_the_document_quotes(self):
        self.assertAlmostEqual(SCOPE["own"]["f1_at_zero"], 0.272073, places=6)
        self.assertAlmostEqual(SCOPE["own"]["auc"], 0.819074, places=6)
        self.assertAlmostEqual(SCOPE["own_best"], 0.340000, places=6)
        self.assertAlmostEqual(SCOPE["own"]["best"]["threshold"], 1.20, places=6)

    def test_the_deltas_are_the_winners_own_numbers(self):
        self.assertAlmostEqual(SCOPE["gain_over_anchor_f1"], -0.100808, places=6)
        self.assertAlmostEqual(SCOPE["auc_delta_vs_anchor"], -0.135455, places=6)
        self.assertAlmostEqual(SCOPE["gain_over_anchor_best"], -0.349775, places=6)

    def test_the_split_sums_to_the_measured_span(self):
        self.assertTrue(SCOPE["fractions_sum_to_one"])
        self.assertAlmostEqual(SCOPE["span_closed_fraction"]+SCOPE["placement_fraction"]+
                               SCOPE["representation_fraction"], 1.0, places=6)
        self.assertAlmostEqual(SCOPE["span_points"]+SCOPE["placement_points"]+
                               SCOPE["representation_points"], SCOPE["anchors"]["span"], places=5)
        self.assertEqual(SCOPE["short_stage"], "closure")

    def test_the_matched_configuration_cost_is_the_one_the_document_states(self):
        geometry_auc = M35["own"]["auc"]-SCOPE["own"]["auc"]
        geometry_best = M35["own_best"]-SCOPE["own_best"]
        self.assertAlmostEqual(geometry_auc, 0.126522, places=6)
        self.assertAlmostEqual(geometry_best, 0.230048, places=6)
        # one recorded fact apart from M35, and the cost reproduces the strength axis rather than vanishing
        self.assertGreater(geometry_auc, 0.071268)
        self.assertGreater(geometry_best, 0.150000)

    def test_the_swamped_signature_is_the_one_the_document_records(self):
        errors = DIAGNOSIS["errors_at_zero"]
        self.assertEqual(errors["false_positive"]["count"], 1107)
        self.assertAlmostEqual(errors["false_positive"]["margin_median"], 2.549952, places=6)
        self.assertAlmostEqual(errors["true_positive"]["margin_low"], 0.282220, places=6)
        bands = DIAGNOSIS["false_positive_bands"]
        self.assertEqual(bands["rows_0_4"]["false_positives"]+bands["rows_5_9"]["false_positives"], 822)
        self.assertEqual(bands["rows_15_19"]["false_positives"], 238)


if __name__ == "__main__":
    unittest.main()
