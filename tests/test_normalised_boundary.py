"""The declared output parameterisation: the occupancy read is trained where 0.0 is the boundary."""
import json
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from tetris_vision.losses import require_occupancy_normalise
from tetris_vision.model import ModelConfig, normalise_board

ROOT = Path(__file__).resolve().parents[1]
ROWS, COLUMNS, BATCH = 20, 10, 2


def board(values=None):
    """A synthetic latent map: 20 visible rows plus six spawn rows, as the model produces."""
    if values is not None:
        latent = torch.tensor(values, dtype=torch.float32).reshape(BATCH, -1)
    else:
        latent = torch.linspace(-3.0, 4.0, BATCH*(ROWS+6)*COLUMNS).reshape(BATCH, (ROWS+6)*COLUMNS)
    return latent.reshape(BATCH, ROWS+6, COLUMNS)


class NormalisedBoardTests(unittest.TestCase):
    """The claim is about the transform, so the transform is what is tested."""

    def test_the_visible_board_is_centred_with_unit_spread(self):
        latent = board()
        normalised = normalise_board(latent, ROWS)
        visible = normalised[:, :ROWS]
        for index in range(BATCH):
            self.assertAlmostEqual(float(visible[index].mean()), 0.0, places=5)
            self.assertAlmostEqual(float(visible[index].std(unbiased=False)), 1.0, places=5)

    def test_an_additive_board_shift_cancels_exactly(self):
        # The registration error the row offset exposed is an additive shift of the board's read;
        # the parameterisation must be blind to it or the head can spend capacity re-centring.
        latent = board()
        for shift in (7.5, -3.25, 0.0):
            self.assertTrue(torch.allclose(normalise_board(latent, ROWS),
                                           normalise_board(latent+shift, ROWS), atol=1e-6))

    def test_a_positive_rescaling_cancels(self):
        latent = board()
        for scale in (0.25, 3.0, 12.0):
            self.assertTrue(torch.allclose(normalise_board(latent, ROWS),
                                           normalise_board(latent*scale, ROWS), atol=1e-5))

    def test_a_negative_rescaling_inverts_the_decision(self):
        # A declared limit rather than a hidden one: the transform is scale-invariant only in sign,
        # so a mirrored read would flip every decision and the test says so.
        latent = board()
        flipped = normalise_board(-latent, ROWS)
        self.assertTrue(torch.allclose(flipped, -normalise_board(latent, ROWS), atol=1e-6))

    def test_a_constant_board_decides_nothing(self):
        flat = normalise_board(torch.zeros(BATCH, ROWS+6, COLUMNS), ROWS)
        self.assertEqual(float(flat.abs().max()), 0.0)

    def test_the_spawn_rows_are_scored_by_the_visible_statistics(self):
        latent = board()
        normalised = normalise_board(latent, ROWS)
        self.assertTrue(torch.equal(normalised[:, ROWS:],
                                    ((latent-latent[:, :ROWS].mean(dim=(1, 2), keepdim=True))
                                     / latent[:, :ROWS].std(dim=(1, 2), unbiased=False,
                                                            keepdim=True).clamp_min(1e-6))[:, ROWS:]))


class DeclaredConfigTests(unittest.TestCase):
    def test_the_published_function_is_the_default(self):
        self.assertEqual(ModelConfig().occupancy_normalise, 0)
        self.assertEqual(ModelConfig(occupancy_normalise=1).occupancy_normalise, 1)

    def test_an_undeclared_parameterisation_is_refused(self):
        for value in (2, -1, True, 1.0, "1"):
            with self.assertRaises(ValueError):
                ModelConfig(occupancy_normalise=value)

    def test_the_admission_join_refuses_a_mismatch(self):
        summary = {"arguments": {"occupancy_normalise": 1},
                   "recovery_contract": {"occupancy_normalise": 1}}
        self.assertIsNone(require_occupancy_normalise(1, summary))
        for value, wrong in ((0, summary), (1, {"arguments": {"occupancy_normalise": 0},
                                                "recovery_contract": {"occupancy_normalise": 1}}),
                             (1, {"arguments": {}, "recovery_contract": {}}),
                             (2, summary)):
            with self.assertRaises(ValueError):
                require_occupancy_normalise(value, wrong)


class PlumbingTests(unittest.TestCase):
    """The lever must be carried by the config, the summary and the admission - not by a default."""

    @classmethod
    def setUpClass(cls):
        from train_m2_stream import validate_admission, main
        cls.source = Path(__import__("inspect").getsourcefile(validate_admission)).read_text(
            encoding="utf-8")
        cls.driver = __import__("inspect").getsource(main)
        cls.signature = __import__("inspect").signature(validate_admission)

    def test_the_driver_writes_the_lever_into_the_config(self):
        start = self.driver.index("replace(config,")
        block = self.driver[start:self.driver.index("config_json", start)]
        self.assertIn("occupancy_normalise=args.occupancy_normalise", block)

    def test_the_driver_records_the_lever_in_the_recovery_contract(self):
        self.assertIn('recovery_contract["occupancy_normalise"] = args.occupancy_normalise',
                      self.driver)

    def test_the_admission_checks_the_lever_it_receives(self):
        self.assertIn("occupancy_normalise", self.signature.parameters)
        self.assertEqual(self.signature.parameters["occupancy_normalise"].default, 0)
        self.assertIn("require_occupancy_normalise(occupancy_normalise, summary)", self.source)

    def test_the_run_family_is_owned(self):
        from record_stream_run import OWNED_RUN_PATTERN
        for name in ("m8-normbound-control-r1", "m8-normbound-norm-r1-sweep-r1",
                     "m8-normbound-norm-r1-sweep-r1-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        for name in ("m8-normbound-", "m8-norm-bound-r1", "m8-normbound"):
            self.assertFalse(OWNED_RUN_PATTERN.fullmatch(name), name)


class NormalisedBoundaryRecordTests(unittest.TestCase):
    """The published record, once it exists; skipped until then."""

    @classmethod
    def setUpClass(cls):
        scope = ROOT/"docs/evidence/m8-normalised-boundary-r1/scope.json"
        if not scope.exists():
            raise unittest.SkipTest("the normalised-boundary record has not been curated yet")
        cls.scope = json.loads(scope.read_text(encoding="utf-8"))
        cls.prior = json.loads((ROOT/"docs/evidence/m7-calibration-r1/scope.json")
                               .read_text(encoding="utf-8"))
        cls.text = " ".join((ROOT/"docs/M8_NORMALISED_BOUNDARY.md").read_text(encoding="utf-8").split())

    def test_the_arms_and_their_parameterisation(self):
        self.assertEqual(sorted(self.scope["arms"]), ["control", "norm"])
        self.assertEqual({name: arm["occupancy_normalise"] for name, arm in self.scope["arms"].items()},
                         {"control": 0, "norm": 1})

    def test_control_reproduces_the_recorded_control(self):
        self.assertEqual(self.scope["arms"]["control"]["auc"], self.prior["arms"]["control"]["auc"])
        self.assertEqual(self.scope["arms"]["control"]["f1_at_zero"],
                         self.prior["arms"]["control"]["f1_at_zero"])

    def test_the_readings_follow_the_printed_numbers(self):
        arm = self.scope["arms"]["norm"]
        delta = round(arm["f1_at_zero"]-self.scope["arms"]["control"]["f1_at_zero"], 6)
        expected = ("normalisation-clears-the-boundary" if delta >= .05 else
                    "normalisation-moves-without-clearing" if delta > 0 else
                    "normalisation-fails")
        self.assertEqual(self.scope["reading"], expected)
        auc_delta = round(arm["auc"]-self.scope["arms"]["control"]["auc"], 6)
        self.assertEqual(self.scope["secondary_reading"],
                         "ranking-preserved" if auc_delta >= -.02 else "ranking-degraded")

    def test_the_document_quotes_the_published_scope(self):
        for name, arm in self.scope["arms"].items():
            self.assertIn(f"{round(arm['auc'], 6):,.6f}", self.text, name)
            self.assertIn(f"{round(arm['f1_at_zero'], 6):,.6f}", self.text, name)
        self.assertIn(self.scope["reading"], self.text)
        self.assertIn(self.scope["secondary_reading"], self.text)

    def test_the_fit_gate_is_reported_on_exact_boards(self):
        self.assertEqual(self.scope["fit_gate"]["boards"], 28)
        self.assertEqual(self.scope["fit_gate"]["cleared"],
                         self.scope["fit_gate"]["exact_nonempty_boards"] == 28)


if __name__ == "__main__":
    unittest.main()
