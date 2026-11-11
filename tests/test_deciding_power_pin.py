"""Pin the deciding-power figure so it cannot drift away from the receipt or the prose.

The saved power record must be exactly what the power instrument recomputes from the receipt it names, and the
sentence quoted in the width-lever record and the continuation state must be exactly the sentence the record
produces. Either the code, the receipt, the record or the prose diverging fails this test.
"""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"scripts")]

from replay_equivalence import first_difference
from sweep_deciding_power import digest, figure_appears, power


def latest_record():
    candidates = sorted((ROOT/"docs/evidence").glob("m3-sweep-deciding-power-*/power.json"))
    if not candidates:
        raise AssertionError("deciding-power record missing")
    return candidates[-1]


class DecidingPowerPin(unittest.TestCase):
    def setUp(self):
        self.path = latest_record()
        self.record = json.loads(self.path.read_text(encoding="utf-8"))

    def test_the_record_is_recomputable_from_the_receipt_it_names(self):
        result_path = ROOT/self.record["result_path"]
        self.assertTrue(result_path.exists(), result_path)
        fresh = power(json.loads(result_path.read_text(encoding="utf-8")))
        for key in ("status", "run", "deciding_split", "population", "auc", "operating_point", "grid", "verdict"):
            difference = first_difference(fresh[key], self.record[key], key)
            self.assertIsNone(difference, difference)

    def test_the_record_names_the_receipt_digest(self):
        # Only the evidence digest is pinned: a source digest would fail on a docstring or a renamed local,
        # which are irrelevant edits, while recomputation already catches any behavioural change.
        self.assertEqual(self.record["sources"]["result"], digest(ROOT/self.record["result_path"]))

    def test_the_documented_figure_is_the_records_sentence(self):
        path = str(self.path.relative_to(ROOT)).replace("\\", "/")
        for document in ("docs/M3_TRUNK_WIDTH_LEVER.md", "docs/CONTINUATION_STATE.md"):
            text = (ROOT/document).read_text(encoding="utf-8")
            self.assertTrue(figure_appears(text, self.record, path), document)

    def test_the_figure_states_the_verdict_is_a_real_finding(self):
        self.assertTrue(self.record["verdict"]["real_finding"])
        self.assertLess(self.record["auc"]["hanley_mcneil"]["high"], self.record["auc"]["auc_bar"])
        self.assertTrue(self.record["grid"]["best_is_interior"])


if __name__ == "__main__":
    unittest.main()
