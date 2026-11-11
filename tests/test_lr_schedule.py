"""Tests for the declared learning-rate schedule (docs/M3_OPTIMISER_SCHEDULE.md)."""
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]/"scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_m2_stream import cosine_rate  # noqa: E402

START, FINAL, TOTAL = 1e-3, 1e-4, 256


class CosineRateTests(unittest.TestCase):
    def test_endpoints_are_exact_not_evaluated(self):
        self.assertEqual(cosine_rate(START, FINAL, 1, TOTAL), START)
        self.assertEqual(cosine_rate(START, FINAL, TOTAL, TOTAL), FINAL)

    def test_schedule_is_non_increasing(self):
        rates = [cosine_rate(START, FINAL, step, TOTAL) for step in range(1, TOTAL+1)]
        self.assertTrue(all(later <= earlier for earlier, later in zip(rates, rates[1:])))

    def test_declared_arm_is_strictly_decaying(self):
        rates = [cosine_rate(START, FINAL, step, TOTAL) for step in range(1, TOTAL+1)]
        self.assertEqual(len(rates), TOTAL)
        self.assertEqual(len(set(rates)), TOTAL)

    def test_middle_steps_bracket_the_cosine_midpoint(self):
        mean = (START+FINAL)*.5
        self.assertAlmostEqual(cosine_rate(START, FINAL, 129, TOTAL), mean, places=5)
        self.assertLess(cosine_rate(START, FINAL, 129, TOTAL), cosine_rate(START, FINAL, 128, TOTAL))

    def test_single_step_segment_keeps_the_start_rate(self):
        self.assertEqual(cosine_rate(START, FINAL, 1, 1), START)
        self.assertEqual(cosine_rate(START, START, 1, 1), START)

    def test_rejects_an_end_rate_above_the_start(self):
        with self.assertRaises(ValueError):
            cosine_rate(FINAL, START, 1, TOTAL)

    def test_rejects_non_positive_end_rate(self):
        with self.assertRaises(ValueError):
            cosine_rate(START, 0.0, 1, TOTAL)

    def test_rejects_steps_outside_the_segment(self):
        for step, total in ((0, TOTAL), (TOTAL+1, TOTAL), (1, 0)):
            with self.assertRaises(ValueError):
                cosine_rate(START, FINAL, step, total)

    def test_rejects_non_integer_step(self):
        with self.assertRaises(ValueError):
            cosine_rate(START, FINAL, 1.0, TOTAL)


if __name__ == "__main__":
    unittest.main()
