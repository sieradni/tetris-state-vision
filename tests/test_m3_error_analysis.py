"""Adversarial masked component reporting and frozen metric compatibility."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from m3_error_analysis import analyze_components, pack_components, replay_keywords, summarize_components

HAS_TORCH = importlib.util.find_spec("torch") is not None


def fixture(capacity=4):
    target = {"capacity": capacity, "queue_tokens": [2, 0, 7]+[-1]*(capacity-2),
              "queue_token_known": [True]*3+[False]*(capacity-2),
              "queue_slot_status": [0, 0]+[-1]*(capacity-2), "queue_slot_status_known": [True]*2+[False]*(capacity-2),
              "queue_presence": [True, True]+[False]*(capacity-2), "queue_presence_known": [True]*3+[False]*(capacity-3),
              "queue_status": 0, "queue_start_observed": True, "queue_end_observed": True,
              "hold_status": 0, "hold_piece": 1, "hold_piece_known": True}
    prediction = {"queue_tokens": [2, 0, 7]+[6]*(capacity-2), "queue_slot_status": [0]*capacity,
                  "queue_presence": [True, True]+[False]*(capacity-2), "queue_anchors": [True, True],
                  "queue_status": 0, "hold_status": 0, "hold_piece": 1}
    return prediction, target


class ComponentErrorAnalysisTests(unittest.TestCase):
    def test_exact_case_separates_piece_and_endpoint_counts(self):
        result = analyze_components(*fixture())
        self.assertTrue(result["exact_readable_components"])
        self.assertEqual(result["queue"]["piece_tokens"], {"known": 2, "correct": 2, "error_indices": []})
        self.assertEqual(result["queue"]["eos_tokens"]["known"], 1)
        self.assertEqual(result["known_queue_tokens"], 3)
        self.assertEqual(result["hold"]["target_piece"], "O")

    def test_swapped_sequence_is_wrong_at_original_indices(self):
        prediction, target = fixture()
        prediction["queue_tokens"][:2] = [0, 2]
        result = analyze_components(prediction, target)
        self.assertEqual(result["queue"]["piece_tokens"]["error_indices"], [0, 1])
        self.assertFalse(result["queue_supported_exact"])
        self.assertTrue(result["hold_supported_exact"])

    def test_unknown_middle_is_not_compacted_or_counted_as_readable(self):
        prediction, target = fixture()
        target.update(queue_tokens=[2, -1, 0, 7, -1], queue_token_known=[True, False, True, True, False],
                      queue_slot_status=[0, 1, 0, -1], queue_slot_status_known=[True, True, True, False],
                      queue_presence=[True, True, True, False], queue_presence_known=[True]*4)
        prediction.update(queue_tokens=[2, 7, 0, 7, 6], queue_slot_status=[0, 1, 0, 2], queue_presence=[True, True, True, False])
        first = analyze_components(prediction, target)
        prediction["queue_tokens"][1] = 4
        self.assertEqual(first, analyze_components(prediction, target))
        self.assertTrue(first["supported_components_exact"])
        self.assertIsNone(first["exact_readable_components"])
        prediction["queue_tokens"][2] = 2
        self.assertEqual(analyze_components(prediction, target)["queue"]["piece_tokens"]["error_indices"], [2])

    def test_unseen_start_has_no_absolute_identity_or_count(self):
        prediction, target = fixture()
        target.update(queue_start_observed=False, queue_end_observed=False, queue_status=2, queue_tokens=[-1]*5, queue_token_known=[False]*5,
                      queue_slot_status=[-1]*4, queue_slot_status_known=[False]*4, queue_presence_known=[False]*4)
        prediction.update(queue_status=2, queue_anchors=[False, False])
        result = analyze_components(prediction, target)
        self.assertEqual(result["known_queue_tokens"], 0)
        self.assertEqual(result["known_presence_positions"], 0)
        self.assertIsNone(result["exact_readable_components"])
        target["queue_tokens"][0], target["queue_token_known"][0] = 0, True
        with self.assertRaisesRegex(ValueError, "unseen start"):
            analyze_components(prediction, target)

    def test_unseen_end_never_supplies_eos(self):
        prediction, target = fixture()
        target.update(queue_end_observed=False, queue_status=2, queue_tokens=[2, 0, -1, -1, -1],
                      queue_token_known=[True, True, False, False, False], queue_presence_known=[True, True, False, False])
        prediction.update(queue_status=2, queue_anchors=[True, False])
        result = analyze_components(prediction, target)
        self.assertTrue(result["supported_components_exact"])
        self.assertIsNone(result["exact_readable_components"])
        self.assertEqual(result["queue"]["eos_tokens"]["known"], 0)
        target["queue_tokens"][2], target["queue_token_known"][2] = 7, True
        with self.assertRaisesRegex(ValueError, "endpoint"):
            analyze_components(prediction, target)

    def test_unknown_hold_masks_identity_and_readable_denominator(self):
        prediction, target = fixture()
        target.update(hold_status=4, hold_piece_known=False)
        prediction["hold_status"] = 4
        first = analyze_components(prediction, target)
        target["hold_piece"], prediction["hold_piece"] = 6, 2
        self.assertEqual(first, analyze_components(prediction, target))
        self.assertTrue(first["hold_supported_exact"])
        self.assertIsNone(first["hold"]["piece_correct"])
        self.assertIsNone(first["exact_readable_components"])

    def test_visible_empty_and_absent_are_distinct(self):
        prediction, target = fixture()
        target.update(queue_tokens=[7, -1, -1, -1, -1], queue_token_known=[True, False, False, False, False],
                      queue_slot_status=[-1]*4, queue_slot_status_known=[False]*4,
                      queue_presence=[False]*4, queue_presence_known=[True, False, False, False], hold_status=1, hold_piece_known=False)
        prediction.update(queue_tokens=[7]*5, queue_presence=[False]*4, hold_status=1)
        self.assertTrue(analyze_components(prediction, target)["exact_readable_components"])
        target.update(queue_status=1, queue_start_observed=False, queue_end_observed=False,
                      queue_tokens=[-1]*5, queue_token_known=[False]*5, queue_presence_known=[False]*4)
        prediction.update(queue_status=1, queue_anchors=[False, False])
        self.assertTrue(analyze_components(prediction, target)["exact_readable_components"])
        prediction["queue_status"] = 0
        self.assertFalse(analyze_components(prediction, target)["exact_readable_components"])

    def test_capacity64_keeps_the_extra_endpoint_position(self):
        prediction, target = fixture(64)
        target.update(queue_tokens=[0]*64+[7], queue_token_known=[True]*65,
                      queue_slot_status=[0]*64, queue_slot_status_known=[True]*64,
                      queue_presence=[True]*64, queue_presence_known=[True]*64)
        prediction.update(queue_tokens=[0]*64+[0], queue_slot_status=[0]*64, queue_presence=[True]*64)
        result = analyze_components(prediction, target)
        self.assertEqual(result["queue"]["piece_tokens"]["correct"], 64)
        self.assertEqual(result["queue"]["eos_tokens"]["error_indices"], [64])
        self.assertFalse(result["exact_readable_components"])

    def test_no_known_tokens_or_hold_identity_has_null_accuracy(self):
        prediction, target = fixture()
        target.update(queue_status=3, queue_start_observed=False, queue_end_observed=False,
                      queue_tokens=[-1]*5, queue_token_known=[False]*5, queue_slot_status=[-1]*4,
                      queue_slot_status_known=[False]*4, queue_presence_known=[False]*4, hold_status=4, hold_piece_known=False)
        prediction.update(queue_status=3, queue_anchors=[False, False], hold_status=4)
        case = {"split": "development", "sample_id": "a", "group_id": "one", "diagnostics": {"components": analyze_components(prediction, target)}}
        summary = summarize_components([case])["development"]
        self.assertEqual(summary["supported_components_exact"], 1)
        self.assertEqual(summary["readable_cases"], 0)
        self.assertEqual(summary["exact_readable_components"], 0)
        self.assertIsNone(summary["queue"]["piece_tokens"]["accuracy"])
        self.assertIsNone(summary["hold"]["piece_accuracy"])

    def test_summary_keeps_dependent_ancestry_and_denominators(self):
        result = analyze_components(*fixture())
        cases = [{"split": "development", "sample_id": str(i), "group_id": "one", "diagnostics": {"components": copy.deepcopy(result)}} for i in range(3)]
        cases[0]["diagnostics"]["components"]["component_state_readable"] = False
        cases[0]["diagnostics"]["components"]["exact_readable_components"] = None
        summary = summarize_components(cases)["development"]
        self.assertEqual((summary["images"], summary["ancestry_groups"], summary["readable_cases"], summary["exact_readable_components"]), (3, 1, 2, 2))
        with self.assertRaises(ValueError):
            summarize_components(cases+[cases[0]])

    def test_malformed_contracts_refuse(self):
        for mutation in (lambda p, t: t.update(capacity=True), lambda p, t: p["queue_presence"].__setitem__(0, 1),
                         lambda p, t: t["queue_token_known"].__setitem__(0, False),
                         lambda p, t: p.update(queue_tokens=[0]), lambda p, t: t.update(hold_piece_known=False)):
            prediction, target = fixture()
            mutation(prediction, target)
            with self.assertRaises(ValueError):
                analyze_components(prediction, target)

    def test_legacy_sampler_receives_no_new_keyword(self):
        sampler, target = replay_keywords({}, 0)
        def legacy(group, view, split, seed):
            return (group, view, split, seed)
        self.assertEqual(legacy(0, 1, "development", 2, **sampler), (0, 1, "development", 2))
        self.assertEqual(target, {})
        self.assertEqual(replay_keywords({"final_capture": True}, 64), ({"final_capture": True}, {"component_capacity": 64}))

    def test_changed_capture_policy_refuses(self):
        for recipe, capacity in (({}, 64), ({"final_capture": True}, 0), ({"final_capture": 1}, 64)):
            with self.assertRaisesRegex(ValueError, "capture policy"):
                replay_keywords(recipe, capacity)


@unittest.skipUnless(HAS_TORCH, "optional train dependency required")
class ComponentTensorDiagnosticTests(unittest.TestCase):
    def test_decomposition_matches_independent_frozen_style_metrics(self):
        import torch
        from torch.nn import functional as F
        from tetris_vision.components import component_metrics, component_tensors
        from tetris_vision.state import FallingPiece, Icon, Queue, VisionState
        from probe_stream_errors import require_frozen_callables
        root = Path(__file__).resolve().parents[1]
        require_frozen_callables(root/"src", component_metrics)
        with self.assertRaisesRegex(ValueError, "not frozen"):
            require_frozen_callables(root/"scripts", component_metrics)
        states = [VisionState(None, FallingPiece("unknown"), FallingPiece("unknown"), queue, hold, "needs_review")
                  for queue, hold in ((Queue((Icon("observed", "T"), Icon("unknown"), Icon("observed", "I"))), Icon("empty")),
                                      (Queue((), "absent", end_observed=False, start_observed=False), Icon("unknown")),
                                      (Queue((Icon("observed", "L"),), "cropped", end_observed=False, start_observed=True), Icon("observed", "O")),
                                      (Queue(()), Icon("empty")))]
        for state in states:
            target = component_tensors([state], capacity=64)
            output = {key: F.one_hot(target[key].clamp_min(0), classes).float()*8-4 for key, classes in
                      (("queue_tokens", 8), ("queue_slot_status", 3), ("queue_status", 4), ("hold_status", 5), ("hold_piece", 7))}
            output["queue_presence"] = target["queue_presence"].float()*8-4
            output["queue_anchors"] = torch.stack((target["queue_start_observed"], target["queue_end_observed"]), -1).float()*8-4
            for corrupt in (False, True):
                if corrupt:
                    output["queue_status"] = output["queue_status"].roll(1, -1)
                    output["hold_status"] = output["hold_status"].roll(1, -1)
                actual = analyze_components(*pack_components(output, target))
                for key, value in component_metrics(output, target)[0].items():
                    self.assertEqual(actual[key], value, key)

    def test_zero_presence_and_anchor_logits_are_false(self):
        import torch
        prediction, target = fixture()
        output = {key: torch.nn.functional.one_hot(torch.tensor([prediction[key]]), classes).float() for key, classes in
                  (("queue_tokens", 8), ("queue_slot_status", 3), ("queue_status", 4), ("hold_status", 5), ("hold_piece", 7))}
        output.update(queue_presence=torch.zeros(1, 4), queue_anchors=torch.zeros(1, 2))
        tensors = {key: torch.tensor([value]) for key, value in target.items() if key != "capacity"}
        tensors["capacity"] = 4
        packed, truth = pack_components(output, tensors)
        self.assertEqual(packed["queue_presence"], [False]*4)
        self.assertEqual(packed["queue_anchors"], [False]*2)
        result = analyze_components(packed, truth)
        self.assertEqual(result["queue"]["presence"]["error_indices"], [0, 1])
        self.assertFalse(result["queue"]["start_correct"])


if __name__ == "__main__":
    unittest.main()
