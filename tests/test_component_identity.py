"""Declared component-identity loss factor: scale-only, never a silent change."""
import importlib.util
import io
from contextlib import redirect_stderr
from pathlib import Path
import sys
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch.nn import functional as F
    from tetris_vision.components import component_loss, component_tensors
    from tetris_vision.state import FallingPiece, Icon, Queue, VisionState
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
    from train_m2_stream import arguments


def state(queue, hold):
    return VisionState(None, FallingPiece("unknown"), FallingPiece("unknown"), queue, hold, "needs_review")


def perfect_logits(target):
    def categorical(values, classes):
        return F.one_hot(values.clamp_min(0), classes).float()*20-10
    return {"queue_tokens": categorical(target["queue_tokens"], 8),
            "queue_slot_status": categorical(target["queue_slot_status"], 3),
            "queue_presence": target["queue_presence"].float()*20-10,
            "queue_status": categorical(target["queue_status"], 4),
            "queue_anchors": torch.stack((target["queue_start_observed"], target["queue_end_observed"]), -1).float()*20-10,
            "hold_status": categorical(target["hold_status"], 5), "hold_piece": categorical(target["hold_piece"], 7)}


def fixture(queue=(Icon("observed", "T"), Icon("observed", "I")), hold=Icon("observed", "O")):
    target = component_tensors([state(Queue(queue), hold)], capacity=4)
    return perfect_logits(target), target


@unittest.skipUnless(HAS_TORCH, "install the optional train dependency for M3 gates")
class ComponentIdentityFactorTests(unittest.TestCase):
    def test_factor_one_is_the_published_component_objective(self):
        output, target = fixture()
        baseline, baseline_terms = component_loss(output, target)
        explicit, explicit_terms = component_loss(output, target, identity_factor=1)
        self.assertTrue(torch.equal(baseline, explicit))
        self.assertEqual(set(baseline_terms), set(explicit_terms))
        self.assertTrue(all(torch.equal(baseline_terms[key], explicit_terms[key]) for key in baseline_terms))

    def test_factor_four_scales_only_the_two_identity_terms(self):
        output, target = fixture()
        baseline, terms = component_loss(output, target)
        scaled, scaled_terms = component_loss(output, target, identity_factor=4)
        self.assertTrue(all(torch.equal(terms[key], scaled_terms[key]) for key in terms))
        identity = terms["queue_tokens"]+terms["hold_piece"]
        self.assertTrue(torch.allclose(scaled, baseline+3*identity, atol=1e-6))

    def test_the_factor_never_reweights_masked_or_unknown_identity(self):
        output, target = fixture(queue=(Icon("observed", "T"), Icon("unknown"), Icon("observed", "I")))
        baseline, terms = component_loss(output, target)
        scaled, _ = component_loss(output, target, identity_factor=4)
        identity = terms["queue_tokens"]+terms["hold_piece"]
        self.assertTrue(torch.allclose(scaled, baseline+3*identity, atol=1e-6))
        self.assertEqual(int(target["queue_token_known"].sum()), 3)

    def test_unpublished_factors_are_refused(self):
        output, target = fixture()
        for factor in (0, 2, 3, 8, True, 4.0):
            with self.assertRaisesRegex(ValueError, "component identity factor"):
                component_loss(output, target, identity_factor=factor)

    def test_the_driver_requires_the_joint_heads_for_a_factor(self):
        base = ["--output", "unused", "--model", "accuracy", "--steps", "2", "--profile-square"]
        self.assertEqual(arguments(base).component_identity_factor, 1)
        self.assertEqual(arguments([*base, "--component-capacity", "64", "--component-identity-factor", "4"]
                                   ).component_identity_factor, 4)
        for extra in (["--component-identity-factor", "4"],
                      ["--component-capacity", "64", "--component-identity-factor", "2"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments([*base, *extra])


if __name__ == "__main__":
    unittest.main()
