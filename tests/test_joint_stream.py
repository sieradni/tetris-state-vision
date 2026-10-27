"""Raw component metrics and nested parity; no training/accuracy claim."""
import importlib.util
from pathlib import Path
import sys
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch.nn import functional as F
    from tetris_vision.components import component_metrics, component_tensors
    from tetris_vision.state import FallingPiece, Icon, Queue, VisionState
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
    from train_m2_stream import equal_tree


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


@unittest.skipUnless(HAS_TORCH, "optional train dependency required")
class JointStreamTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_exact_readable_components_require_every_supported_head(self):
        target = component_tensors([state(Queue((Icon("observed", "T"), Icon("observed", "I"))), Icon("observed", "O"))], capacity=4)
        output = perfect_logits(target)
        result = component_metrics(output, target)[0]
        self.assertTrue(result["component_state_readable"])
        self.assertTrue(result["exact_readable_components"])
        output["queue_tokens"][0, 0].fill_(-10)
        output["queue_tokens"][0, 0, 6] = 10
        result = component_metrics(output, target)[0]
        self.assertFalse(result["queue_supported_exact"])
        self.assertFalse(result["exact_readable_components"])
        self.assertTrue(result["hold_supported_exact"])

    def test_masked_middle_never_becomes_complete_success(self):
        target = component_tensors([state(Queue((Icon("observed", "T"), Icon("unknown"), Icon("observed", "I"))), Icon("empty"))], capacity=4)
        output = perfect_logits(target)
        output["queue_tokens"][0, 1].fill_(-10)
        output["queue_tokens"][0, 1, 7] = 10  # Unknown identity cannot establish a valid decoded queue.
        result = component_metrics(output, target)[0]
        self.assertTrue(result["supported_components_exact"])
        self.assertFalse(result["component_state_readable"])
        self.assertIsNone(result["exact_readable_components"])

    def test_cropped_endpoint_and_unknown_hold_are_unreadable(self):
        cases = [state(Queue((Icon("observed", "L"),), "cropped", False, True), Icon("empty")),
                 state(Queue((), "visible", True, True), Icon("unknown"))]
        target = component_tensors(cases, capacity=4)
        results = component_metrics(perfect_logits(target), target)
        self.assertTrue(all(row["supported_components_exact"] for row in results))
        self.assertTrue(all(row["exact_readable_components"] is None for row in results))

    def test_absent_and_visible_empty_are_distinct_readable_states(self):
        cases = [state(Queue((), "absent", False, False), Icon("absent")),
                 state(Queue((), "visible", True, True), Icon("empty"))]
        target = component_tensors(cases, capacity=4)
        output = perfect_logits(target)
        self.assertTrue(all(row["exact_readable_components"] for row in component_metrics(output, target)))
        output["queue_status"][0] = output["queue_status"][1]
        result = component_metrics(output, target)
        self.assertFalse(result[0]["exact_readable_components"])
        self.assertTrue(result[1]["exact_readable_components"])

    def test_batch_readability_never_leaks_between_images(self):
        cases = [state(Queue((Icon("observed", "I"),)), Icon("observed", "T")),
                 state(Queue((), "unknown", False, False), Icon("unknown"))]
        target = component_tensors(cases, capacity=4)
        results = component_metrics(perfect_logits(target), target)
        self.assertTrue(results[0]["exact_readable_components"])
        self.assertIsNone(results[1]["exact_readable_components"])
        self.assertEqual(sum(row["component_state_readable"] for row in results), 1)

    def test_invalid_class_and_dtype_refuse_before_readability_score(self):
        target = component_tensors([state(Queue((), "unknown", False, False), Icon("unknown"))], capacity=4)
        output = perfect_logits(target)
        target["hold_status"][0] = -1
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            component_metrics(output, target)
        target["hold_status"][0] = 4
        output["queue_status"] = output["queue_status"].long()
        with self.assertRaisesRegex(ValueError, "output shapes"):
            component_metrics(output, target)

    def test_nested_component_tensor_shape_dtype_and_values_are_exact(self):
        expected = {"occupancy": torch.ones(1, 20, 10), "components": {"queue_tokens": torch.arange(40.).reshape(1, 5, 8)}}
        cloned = {"occupancy": expected["occupancy"].clone(), "components": {"queue_tokens": expected["components"]["queue_tokens"].clone()}}
        self.assertTrue(equal_tree(expected, cloned))
        cloned["components"]["queue_tokens"][0, 0, 0] = 99
        self.assertFalse(equal_tree(expected, cloned))
        self.assertFalse(equal_tree(torch.ones(1), torch.ones(1, dtype=torch.float64)))
        self.assertFalse(equal_tree((torch.ones(1),), [torch.ones(1)]))
        self.assertFalse(equal_tree({"old": torch.ones(1)}, {"new": torch.ones(1)}))


if __name__ == "__main__":
    unittest.main()
