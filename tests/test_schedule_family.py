"""The schedule-lever arms must be an owned run family (docs/M3_OPTIMISER_SCHEDULE.md)."""
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]/"scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from record_stream_run import require_owned_run  # noqa: E402

ARMS = ("m3-occsched-constant-r1", "m3-occsched-cosine-r1",
        "m3-occsched-constant-r1-sweep-r1", "m3-occsched-cosine-r1-sweep-r1-profile-r1")


class ScheduleFamilyTests(unittest.TestCase):
    def test_declared_arm_and_sweep_names_are_owned(self):
        for name in ARMS:
            self.assertEqual(require_owned_run(name), name)

    def test_family_keeps_its_bounds(self):
        for name in ("m3-occsched", "occsched-constant-r1", "m3-occsched-"+51*"a",
                     "m3-occsched-x/y", "m3-occsched-Constant-r1"):
            with self.assertRaises(ValueError):
                require_owned_run(name)

    def test_family_does_not_admit_other_prefixes(self):
        for name in ("m3-occschedule-constant-r1", "m3-schedule-constant-r1"):
            with self.assertRaises(ValueError):
                require_owned_run(name)


if __name__ == "__main__":
    unittest.main()
