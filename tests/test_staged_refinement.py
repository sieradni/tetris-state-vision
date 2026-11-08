"""Contracts for the staged frozen-backbone type-head refinement mode."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from train_m2_stream import arguments, staged_mode


def profile(argv):
    return ["--output", "runs/staged-profile", "--model", "pilot", "--steps", "2",
            "--profile-square", *argv]


class StagedRefinementContractTests(unittest.TestCase):
    def test_frozen_stage_requires_initial_weights_and_ordered_fit(self):
        for argv in (["--freeze-backbone"],
                     ["--freeze-backbone", "--init-weights-from", "runs/source",
                      "--curriculum", "all-class-interleaved-v1"],
                     ["--freeze-backbone", "--init-weights-from", "runs/source",
                      "--layer-evidence"]):
            with self.assertRaises(SystemExit):
                arguments(profile(argv))
        parsed = arguments(profile(["--freeze-backbone", "--init-weights-from", "runs/source"]))
        self.assertTrue(parsed.freeze_backbone)
        self.assertEqual(str(parsed.init_weights_from), "runs/source")
        self.assertEqual(parsed.curriculum, "all-class-fit-v1")

    def test_initial_weights_and_resume_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            arguments(["--output", "runs/x", "--model", "pilot", "--steps", "4",
                       "--admission", "runs/profile", "--init-weights-from", "runs/source",
                       "--resume-from", "runs/other"])

    def test_admission_pins_the_staged_mode_instead_of_inheriting_it(self):
        plain = {"arguments": {"freeze_backbone": False, "init_weights_from": None}}
        self.assertTrue(staged_mode(plain, freeze_backbone=False, init_weights_from=None))
        self.assertFalse(staged_mode(plain, freeze_backbone=True, init_weights_from=Path("runs/source")))
        staged = {"arguments": {"freeze_backbone": True, "init_weights_from": "C:\\runs\\source"}}
        self.assertTrue(staged_mode(staged, freeze_backbone=True, init_weights_from=Path("C:/runs/source")))
        self.assertFalse(staged_mode(staged, freeze_backbone=True, init_weights_from=None))
        self.assertFalse(staged_mode(staged, freeze_backbone=False, init_weights_from=Path("C:/runs/source")))
        # A different source artifact is a different start, not the same admission.
        self.assertFalse(staged_mode(staged, freeze_backbone=True, init_weights_from=Path("C:/runs/other")))
        # Non-boolean declarations are refused rather than coerced.
        self.assertFalse(staged_mode({"arguments": {"freeze_backbone": "yes"}},
                                     freeze_backbone=True, init_weights_from=Path("C:/runs/source")))


if __name__ == "__main__":
    unittest.main()
