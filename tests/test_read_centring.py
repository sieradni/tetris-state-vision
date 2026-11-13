"""Structural centring of the occupancy read: the additive mode is removed in the head, not charged by a term."""
import contextlib
import inspect
import io
import json
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from tetris_vision.losses import require_occupancy_centre
from tetris_vision.model import ModelConfig, centre_board

ROOT = Path(__file__).resolve().parents[1]
ROWS, COLUMNS, BATCH = 20, 10, 2


def board(values=None):
    """A synthetic read: 20 visible rows plus six spawn rows, as the model produces."""
    if values is not None:
        latent = torch.tensor(values, dtype=torch.float32).reshape(BATCH, -1)
    else:
        latent = torch.linspace(-3.0, 4.0, BATCH*(ROWS+6)*COLUMNS).reshape(BATCH, (ROWS+6)*COLUMNS)
    return latent.reshape(BATCH, ROWS+6, COLUMNS)


class CentreBoardTests(unittest.TestCase):
    def test_each_board_loses_its_own_visible_mean(self):
        latent = board()
        centred = centre_board(latent, ROWS)
        for index in range(BATCH):
            self.assertAlmostEqual(float(centred[index, :ROWS].mean()), 0.0, places=5)

    def test_an_additive_board_shift_changes_nothing(self):
        # Alignment invariance, exactly: the read is a function of relative cell structure only.
        latent = board()
        for shift in (7.5, -3.25, 0.0):
            self.assertTrue(torch.allclose(centre_board(latent, ROWS),
                                           centre_board(latent+shift, ROWS), atol=1e-6))
            self.assertTrue(torch.allclose(centre_board(latent, ROWS),
                                           centre_board(latent+shift, ROWS), rtol=0, atol=1e-6))

    def test_the_correction_is_per_board_not_global(self):
        latent = board().clone()
        latent[0] += 5.0
        latent[1] -= 7.0
        centred = centre_board(latent, ROWS)
        self.assertAlmostEqual(float(centred[0, :ROWS].mean()), 0.0, places=5)
        self.assertAlmostEqual(float(centred[1, :ROWS].mean()), 0.0, places=5)
        # A single global subtraction could not zero both boards, so the batch mean must not be the statistic.
        self.assertNotAlmostEqual(float(centred[0].mean()), 0.0, places=5)
        self.assertTrue(torch.allclose(centred[0], centre_board(latent[:1], ROWS)[0], atol=1e-6))
        self.assertTrue(torch.allclose(centred[1], centre_board(latent[1:], ROWS)[0], atol=1e-6))

    def test_relative_structure_is_untouched(self):
        # Subtracting a constant cannot change any difference between two cells, so the per-board
        # correction is itself constant across the board and every within-board pair difference survives.
        latent = board()
        correction = centre_board(latent, ROWS)-latent
        for index in range(BATCH):
            self.assertAlmostEqual(float(correction[index].std(unbiased=False)), 0.0, places=5)
            pairs = (latent[index, :ROWS].flatten()[:, None]-latent[index, :ROWS].flatten()[None, :])
            kept = (centre_board(latent, ROWS)[index, :ROWS].flatten()[:, None]
                    - centre_board(latent, ROWS)[index, :ROWS].flatten()[None, :])
            self.assertTrue(torch.allclose(kept, pairs, atol=1e-5))

    def test_the_spawn_rows_are_scored_but_never_enter_the_mean(self):
        latent = board()
        centred = centre_board(latent, ROWS)
        self.assertTrue(torch.allclose(centred[:, ROWS:],
                                       (latent-latent[:, :ROWS].mean(dim=(1, 2), keepdim=True))[:, ROWS:],
                                       atol=1e-6))
        # A hidden region above the board cannot move the statistic.
        hidden = latent.clone()
        hidden[:, ROWS:] += 100.0
        self.assertTrue(torch.allclose(centre_board(latent, ROWS)[:, :ROWS],
                                       centre_board(hidden, ROWS)[:, :ROWS], atol=1e-6))

    def test_a_constant_board_centres_to_zero(self):
        flat = centre_board(torch.full((BATCH, ROWS+6, COLUMNS), 4.25), ROWS)
        self.assertEqual(float(flat.abs().max()), 0.0)

    def test_a_single_visible_cell_centres_to_zero(self):
        one_row = torch.tensor([[[2.0]*COLUMNS, [5.0]*COLUMNS]], dtype=torch.float32)
        centred = centre_board(one_row, 1)
        self.assertEqual(float(centred[:, :1].abs().max()), 0.0)
        self.assertEqual(float(centred[0, 1].abs().max()), 3.0)


class DeclaredConfigTests(unittest.TestCase):
    def test_the_published_function_is_the_default(self):
        self.assertEqual(ModelConfig().occupancy_centre, 0)
        self.assertEqual(ModelConfig(occupancy_centre=1).occupancy_centre, 1)

    def test_an_undeclared_option_is_refused(self):
        for value in (2, -1, True, 1.0, "1"):
            with self.assertRaises(ValueError):
                ModelConfig(occupancy_centre=value)

    def test_centring_and_normalising_cannot_be_stacked(self):
        with self.assertRaises(ValueError):
            ModelConfig(occupancy_centre=1, occupancy_normalise=1)

    def test_the_admission_join_refuses_a_mismatch(self):
        summary = {"arguments": {"occupancy_centre": 1}, "recovery_contract": {"occupancy_centre": 1},
                   "config": {"occupancy_centre": 1}}
        self.assertIsNone(require_occupancy_centre(1, summary))
        for value, wrong in ((0, summary), (2, summary),
                             (1, {"arguments": {"occupancy_centre": 0},
                                  "recovery_contract": {"occupancy_centre": 1},
                                  "config": {"occupancy_centre": 1}}),
                             (1, {"arguments": {"occupancy_centre": 1},
                                  "recovery_contract": {"occupancy_centre": 1},
                                  "config": {"occupancy_centre": 0}}),
                             (1, {"arguments": {}, "recovery_contract": {}, "config": {}})):
            with self.assertRaises(ValueError):
                require_occupancy_centre(value, wrong)


class HeadPlacementTests(unittest.TestCase):
    def test_the_head_applies_the_centre_before_the_stages_are_emitted(self):
        source = Path(inspect.getsourcefile(ModelConfig)).read_text(encoding="utf-8")
        guard = source.index("if self.config.occupancy_centre:")
        applied = source.index("centre_board(occupancy_logits, self.config.visible_rows)")
        emitted = source.index("stages.append(", applied)
        self.assertGreater(applied, guard)
        self.assertGreater(emitted, applied)

    def test_the_centre_is_not_a_loss_term(self):
        # A structural option, not a sixth charge: no term function, no weight, no entry in the objective's weights.
        from tetris_vision import losses
        term_source = Path(inspect.getsourcefile(losses)).read_text(encoding="utf-8")
        self.assertNotIn("def centre_term", term_source)
        self.assertNotIn('"centre":', term_source)
        self.assertIn("def require_occupancy_centre", term_source)


class PlumbingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from train_m2_stream import arguments, validate_admission, main
        cls.source = Path(inspect.getsourcefile(validate_admission)).read_text(encoding="utf-8")
        cls.driver = inspect.getsource(main)
        cls.refusals = inspect.getsource(arguments)
        cls.signature = inspect.signature(validate_admission)
        cls.arguments = staticmethod(arguments)
        cls.base = ["--output", "runs/m14-centre-refusal-probe", "--model", "accuracy", "--steps", "2"]

    def test_the_driver_writes_the_lever_into_the_config(self):
        start = self.driver.index("replace(config,")
        block = self.driver[start:self.driver.index("config_json", start)]
        self.assertIn("occupancy_centre=args.occupancy_centre", block)

    def test_the_driver_records_the_lever_in_the_recovery_contract(self):
        self.assertIn('recovery_contract["occupancy_centre"] = args.occupancy_centre', self.driver)

    def test_the_admission_checks_the_lever_it_receives(self):
        self.assertIn("occupancy_centre", self.signature.parameters)
        self.assertEqual(self.signature.parameters["occupancy_centre"].default, 0)
        self.assertIn("require_occupancy_centre(occupancy_centre, summary)", self.source)

    def test_the_trainer_refuses_every_term_and_lever_beside_the_centred_read(self):
        self.assertIn("if args.occupancy_centre and (args.occupancy_normalise", self.refusals)
        for lever in (["--occupancy-midpoint", "1"], ["--occupancy-dense-hinge", "1"],
                      ["--occupancy-quantile-anchor", "1"], ["--occupancy-anchor", "1"],
                      ["--occupancy-margin-hinge", "1"], ["--occupancy-normalise", "1"],
                      ["--occupancy-deform", "2", "--occupancy-detail", "1"],
                      ["--occupancy-offset-rows", "1", "--occupancy-detail", "1"],
                      ["--occupancy-detail-stencil", "1", "--occupancy-detail", "1"],
                      ["--occupancy-resolution", "1"], ["--occupancy-context", "1"],
                      ["--occupancy-column", "1"], ["--occupancy-lattice-free", "1"]):
            stream = io.StringIO()
            with self.subTest(lever=lever), contextlib.redirect_stderr(stream):
                with self.assertRaises(SystemExit):
                    self.arguments(self.base+["--occupancy-centre", "1"]+lever)
            self.assertIn("the centred read is a structural option", stream.getvalue(), lever)

    def test_the_centred_read_alone_is_admitted(self):
        parsed = self.arguments(self.base+["--profile-square", "--occupancy-centre", "1"])
        self.assertEqual(parsed.occupancy_centre, 1)
        self.assertEqual(self.arguments(self.base+["--profile-square"]).occupancy_centre, 0)

    def test_the_run_family_is_owned(self):
        from record_stream_run import OWNED_RUN_PATTERN
        for name in ("m14-centre-control-r1", "m14-centre-centred-r1",
                     "m14-centre-centred-r1-sweep-r1-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        for name in ("m14-centre-", "m14-centre", "m14-centered-arm-r1"):
            self.assertFalse(OWNED_RUN_PATTERN.fullmatch(name), name)
        for name in ("m13-midpoint-control-r1", "m12-dense-hinge-control-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)


class CentringRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scope = ROOT/"docs/evidence/m14-read-centring-r1/scope.json"
        if not scope.exists():
            raise unittest.SkipTest("the centring record has not been curated yet")
        cls.scope = json.loads(scope.read_text(encoding="utf-8"))
        cls.prior = json.loads((ROOT/"docs/evidence/m10-supervised-anchor-r1/scope.json")
                               .read_text(encoding="utf-8"))
        cls.midpoint = json.loads((ROOT/"docs/evidence/m13-midpoint-r1/scope.json").read_text(encoding="utf-8"))
        cls.text = " ".join((ROOT/"docs/M14_READ_CENTRING.md").read_text(encoding="utf-8").split())

    def test_the_arms_and_the_single_declared_difference(self):
        self.assertEqual(sorted(self.scope["arms"]), ["centred", "control"])
        self.assertEqual({name: arm["occupancy_centre"] for name, arm in self.scope["arms"].items()},
                         {"control": 0, "centred": 1})
        self.assertEqual(self.scope["aliasing_guard"]["arms_differ_in"], ["occupancy_centre"])
        self.assertTrue(self.scope["aliasing_guard"]["weights_differ"])
        self.assertTrue(self.scope["aliasing_guard"]["not_the_previous_block"])

    def test_control_reproduces_the_recorded_control(self):
        self.assertEqual(self.scope["arms"]["control"]["auc"], self.prior["arms"]["control"]["auc"])
        self.assertEqual(self.scope["arms"]["control"]["f1_at_zero"],
                         self.prior["arms"]["control"]["f1_at_zero"])

    def test_the_structural_option_adds_no_term(self):
        mechanism = self.scope["mechanism"]
        self.assertEqual(mechanism["loss_terms_added"], 0)
        self.assertIsNone(mechanism["weight"])
        self.assertIsNone(mechanism["margin"])
        self.assertEqual(mechanism["parameters_added"], 0)
        self.assertTrue(mechanism["declared_before_receipts"])

    def test_the_readings_follow_the_printed_numbers(self):
        arm, control = self.scope["arms"]["centred"], self.scope["arms"]["control"]
        delta = round(arm["f1_at_zero"]-control["f1_at_zero"], 6)
        expected = ("centre-clears-the-boundary" if delta >= .05 else
                    "centre-moves-without-clearing" if delta > 0 else "centre-fails")
        self.assertEqual(self.scope["reading"], expected)
        self.assertEqual(self.scope["secondary_reading"],
                         "ranking-preserved" if round(arm["auc"]-control["auc"], 6) >= -.02 else "ranking-degraded")
        self.assertEqual(self.scope["overshot_below_zero"], arm["best"]["threshold"] < 0.0)

    def test_the_document_quotes_the_published_scope(self):
        for name, arm in self.scope["arms"].items():
            self.assertIn(f"{round(arm['auc'], 6):,.6f}", self.text, name)
            self.assertIn(f"{round(arm['f1_at_zero'], 6):,.6f}", self.text, name)
        self.assertIn(self.scope["reading"], self.text)
        self.assertIn(self.scope["secondary_reading"], self.text)

    def test_every_reference_row_is_quoted(self):
        rows = {name: arm for name, arm in self.scope["reference"].items() if isinstance(arm, dict)}
        self.assertEqual(len(rows), 5)
        for name, arm in rows.items():
            self.assertIn(f"{round(arm['auc'], 6):,.6f}", self.text, name)

    def test_the_level_drift_and_fit_gate_are_reported(self):
        for name, arm in self.scope["arms"].items():
            self.assertAlmostEqual(arm["positive_share_at_zero"],
                                   (arm["operating_counts"]["true_positive"]
                                    + arm["operating_counts"]["false_positive"])/5600, places=6)
        gate = self.scope["fit_gate"]
        self.assertEqual(gate["boards"], 28)
        self.assertEqual(gate["exact_nonempty_boards"],
                         self.scope["arms"]["centred"]["exact_boards_at_zero"]["exact"])


if __name__ == "__main__":
    unittest.main()
