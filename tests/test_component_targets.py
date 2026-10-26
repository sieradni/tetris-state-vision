from dataclasses import replace
import unittest

from PIL import ImageChops

from tetris_vision.component_targets import EOS, HOLD_STATUSES, PIECES, QUEUE_STATUSES, component_labels
from tetris_vision.render import Appearance, render, sample_scene
from tetris_vision.state import FallingPiece, Icon, Queue, VisionState


def state(queue, hold=Icon("unknown")):
    return VisionState(None, FallingPiece("unknown"), FallingPiece("unknown"), queue, hold, "needs_review")


class ComponentTargetTests(unittest.TestCase):
    def test_identified_model_output_is_not_admitted_as_ground_truth(self):
        original = state(Queue((), "unknown", False, False))
        with self.assertRaisesRegex(ValueError, "not a model output"):
            component_labels(replace(original, producer="model", model_id="weights-sha"))
        with self.assertRaises(TypeError):
            component_labels(original.to_dict())

    def test_unknown_middle_retains_index_and_presence(self):
        value = component_labels(state(Queue((Icon("observed", "T"), Icon("unknown"), Icon("observed", "I")))))
        self.assertEqual(value["queue_tokens"][:4], (PIECES.index("T"), -1, PIECES.index("I"), EOS))
        self.assertEqual(value["queue_token_known"][:4], (True, False, True, True))
        self.assertEqual(value["queue_presence"][:4], (True, True, True, False))
        self.assertEqual(value["queue_presence_known"][:5], (True, True, True, True, False))
        self.assertEqual(value["queue_count"], 3)

    def test_unseen_tail_never_trains_eos_or_hidden_length(self):
        value = component_labels(state(Queue((Icon("observed", "L"), Icon("cropped")), "cropped", False, True)))
        self.assertEqual(value["queue_tokens"][0], PIECES.index("L"))
        self.assertNotIn(EOS, value["queue_tokens"])
        self.assertIsNone(value["queue_count"])
        self.assertFalse(any(value["queue_presence_known"][2:]))

    def test_unseen_start_masks_absolute_slots_even_when_end_visible(self):
        first = component_labels(state(Queue((Icon("observed", "I"),), "visible", True, False)), capacity=4)
        second = component_labels(state(Queue(tuple(Icon("observed", "T") for _ in range(64)), "visible", True, False)), capacity=4)
        self.assertEqual(first, second)
        self.assertFalse(any(first["queue_token_known"]))
        self.assertFalse(any(first["queue_slot_status_known"]))
        self.assertIsNone(first["queue_count"])
        self.assertTrue(first["queue_end_observed"])

    def test_absent_is_not_visible_empty(self):
        absent = component_labels(state(Queue((), "absent", False, False)))
        empty = component_labels(state(Queue((), "visible", True, True)))
        self.assertEqual(absent["queue_status"], QUEUE_STATUSES.index("absent"))
        self.assertFalse(any(absent["queue_token_known"]))
        self.assertEqual(empty["queue_tokens"][0], EOS)
        self.assertEqual(empty["queue_count"], 0)
        self.assertTrue(empty["queue_presence_known"][0])
        self.assertFalse(empty["queue_presence"][0])

    def test_hold_status_and_identity_are_independent(self):
        for status in HOLD_STATUSES:
            icon = Icon(status, "Z" if status == "observed" else None)
            value = component_labels(state(Queue((), "unknown", False, False), icon))
            self.assertEqual(value["hold_status"], HOLD_STATUSES.index(status))
            self.assertEqual(value["hold_piece_known"], status == "observed")
            self.assertEqual(value["hold_piece"], PIECES.index("Z") if status == "observed" else None)

    def test_capacity_boundary_has_explicit_end_position(self):
        queue = Queue(tuple(Icon("observed", "T") for _ in range(64)))
        value = component_labels(state(queue))
        self.assertEqual(value["queue_tokens"], (PIECES.index("T"),)*64+(EOS,))
        self.assertEqual(value["queue_count"], 64)
        with self.assertRaisesRegex(ValueError, "no silent truncation"):
            component_labels(state(queue), capacity=4)
        for capacity in (0, 65, True):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                component_labels(state(queue), capacity=capacity)

    def test_identical_raster_hidden_queue_and_hold_have_equal_masked_targets(self):
        scene = sample_scene(44)
        style = Appearance(crop=(60, 0, 282, 540))
        first = render(replace(scene, queue=("I",), hold="L"), style)
        second = render(replace(scene, queue=("T",)*9, hold="Z"), style)
        self.assertIsNone(ImageChops.difference(first.image, second.image).getbbox())
        a, b = component_labels(first.state), component_labels(second.state)
        self.assertEqual(a, b)
        self.assertFalse(any(a["queue_token_known"]))
        self.assertFalse(a["hold_piece_known"])
        self.assertIsNone(a["queue_count"])


if __name__ == "__main__":
    unittest.main()
