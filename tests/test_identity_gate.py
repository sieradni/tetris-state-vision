"""The identity discrimination rule opens only when both heads emit enough classes."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

import identity_gate


def head(distinct, constant=False):
    return {"distinct_predicted_classes": distinct, "constant_predictor": constant}


class IdentityGateTests(unittest.TestCase):
    def test_both_heads_clear_three_classes(self):
        self.assertTrue(identity_gate.identity_gate_opens(head(3), head(3)))
        self.assertEqual(identity_gate.identity_verdict(head(4), head(5)), "opens")

    def test_one_collapsed_head_stands_the_rule(self):
        self.assertFalse(identity_gate.identity_gate_opens(head(3), head(2)))
        self.assertFalse(identity_gate.identity_gate_opens(head(2), head(3)))
        self.assertEqual(identity_gate.identity_verdict(head(3), head(1, True)), "stands")

    def test_a_constant_head_never_opens(self):
        self.assertFalse(identity_gate.identity_gate_opens(head(1, True), head(3)))

    def test_the_boundary_is_inclusive_and_the_bar_is_explicit(self):
        self.assertTrue(identity_gate.identity_gate_opens(head(2), head(2), threshold=2))
        self.assertFalse(identity_gate.identity_gate_opens(head(2), head(2), threshold=3))

    def test_exact_counts_cannot_open_the_rule(self):
        # A head's exact count equals its class frequency when it is a constant, so only the
        # emitted-class census decides.
        values = dict(head(1, True), piece_tokens={"known": 54, "correct": 54, "errors": 0})
        self.assertFalse(identity_gate.identity_gate_opens(values, head(3)))

    def test_malformed_heads_are_refused(self):
        for call in (lambda: identity_gate.identity_gate_opens({}, head(3)),
                     lambda: identity_gate.identity_gate_opens(head(3), head(True)),
                     lambda: identity_gate.identity_gate_opens(head(9), head(3)),
                     lambda: identity_gate.identity_gate_opens(head(3), {"distinct_predicted_classes": 3}),
                     lambda: identity_gate.identity_gate_opens(head(3), head(3), threshold=9),
                     lambda: identity_gate.identity_gate_opens(head(3), head(1, True) | {"constant_predictor": True,
                                                                                        "distinct_predicted_classes": 3})):
            with self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
