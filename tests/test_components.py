"""M3 masking, shared-query and compatibility gates; no learning/accuracy claim."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import ImageChops

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from tetris_vision.checkpoint import load_checkpoint, save_checkpoint
    from tetris_vision.components import ComponentDecoder, component_loss, component_tensors
    from tetris_vision.layout import Layout
    from tetris_vision.losses import joint_loss, targets_for
    from tetris_vision.model import ModelConfig, TetrisStateNet
    from tetris_vision.preprocess import prepare_image
    from tetris_vision.render import Appearance, render, sample_scene
    from tetris_vision.runtime import decode_partial
    from tetris_vision.state import FallingPiece, Icon, Queue, VisionState


def observation(queue, hold=None):
    return VisionState(None, FallingPiece("unknown"), FallingPiece("unknown"), queue, hold or Icon("unknown"), "needs_review")


@unittest.skipUnless(HAS_TORCH, "install the optional train dependency for M3 gates")
class ComponentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(73)
        self.head = ComponentDecoder(32, 4)
        self.memory = torch.randn(1, 13, 32, requires_grad=True)
        self.valid = torch.ones(1, 13, dtype=torch.bool)
        self.board = torch.randn(1, 1, 32, requires_grad=True)

    def output(self, capacity):
        return self.head(self.memory, self.valid, self.board, capacity=capacity)

    def test_shared_parameters_and_prefix_predictions_extend_to64(self):
        count = sum(value.numel() for value in self.head.parameters())
        short, long = self.output(4), self.output(64)
        self.assertEqual(long["queue_tokens"].shape, (1, 65, 8))
        self.assertEqual(long["queue_presence"].shape, (1, 64))
        self.assertEqual(count, sum(value.numel() for value in self.head.parameters()))
        self.assertTrue(torch.allclose(short["queue_tokens"], long["queue_tokens"][:, :5], atol=1e-6, rtol=1e-6))

    def test_masked_context_values_cannot_change_predictions(self):
        self.valid[:, -3:] = False
        original = self.output(4)
        changed = self.memory.detach().clone()
        changed[:, -3:] = 999.
        alternate = self.head(changed, self.valid, self.board, capacity=4)
        self.assertTrue(all(torch.equal(value, alternate[key]) for key, value in original.items()))

    def test_unknown_semantics_have_zero_gradient_but_visibility_learns(self):
        target = component_tensors([observation(Queue((), "unknown", False, False))], capacity=4)
        loss, terms = component_loss(self.output(4), target)
        loss.backward()
        for key in ("queue_tokens", "queue_slot_status", "queue_presence", "hold_piece"):
            self.assertEqual(float(terms[key].detach()), 0)
            self.assertEqual(float(getattr(self.head, key).weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.head.hold_status.weight.grad.abs().sum()), 0)
        self.assertTrue(all(value.grad is None or bool(torch.isfinite(value.grad).all()) for value in self.head.parameters()))

    def test_unknown_middle_does_not_shift_or_train_its_identity(self):
        queue = Queue((Icon("observed", "T"), Icon("unknown"), Icon("observed", "I")))
        target = component_tensors([observation(queue)], capacity=4)
        output = self.output(4)
        output["queue_tokens"].retain_grad()
        component_loss(output, target)[0].backward()
        self.assertEqual(target["queue_tokens"][0, :4].tolist(), [2, -1, 0, 7])
        self.assertEqual(float(output["queue_tokens"].grad[0, 1].abs().sum()), 0)
        self.assertGreater(float(output["queue_tokens"].grad[0, 2].abs().sum()), 0)

    def test_cropped_tail_has_no_eos_or_future_presence_gradient(self):
        queue = Queue((Icon("observed", "L"), Icon("cropped")), "cropped", False, True)
        target = component_tensors([observation(queue)], capacity=4)
        output = self.output(4)
        for key in ("queue_tokens", "queue_presence"):
            output[key].retain_grad()
        component_loss(output, target)[0].backward()
        self.assertNotIn(7, target["queue_tokens"].flatten().tolist())
        self.assertEqual(float(output["queue_tokens"].grad[:, 1:].abs().sum()), 0)
        self.assertEqual(float(output["queue_presence"].grad[:, 2:].abs().sum()), 0)

    def test_unseen_start_does_not_admit_hidden_absolute_length(self):
        first = observation(Queue((Icon("observed", "I"),), "visible", True, False))
        second = observation(Queue((Icon("observed", "T"),)*64, "visible", True, False))
        left, right = (component_tensors([state], capacity=4) for state in (first, second))
        self.assertTrue(all(torch.equal(value, right[key]) if torch.is_tensor(value) else value == right[key] for key, value in left.items()))
        self.assertFalse(bool(left["queue_token_known"].any()))
        self.assertFalse(bool(left["queue_count_known"].any()))

    def test_eos_after_item64_is_supervised_without_extra_item(self):
        target = component_tensors([observation(Queue((Icon("observed", "T"),)*64))], capacity=64)
        output = self.output(64)
        output["queue_tokens"].retain_grad()
        component_loss(output, target)[0].backward()
        self.assertEqual(int(target["queue_tokens"][0, 64]), 7)
        self.assertEqual(target["queue_presence"].shape, (1, 64))
        self.assertGreater(float(output["queue_tokens"].grad[0, 64].abs().sum()), 0)

    def test_identical_hidden_rasters_have_equal_tensor_targets(self):
        scene, style = sample_scene(44), Appearance(crop=(60, 0, 282, 540))
        first = render(replace(scene, queue=("I",), hold="L"), style)
        second = render(replace(scene, queue=("T",)*9, hold="Z"), style)
        self.assertIsNone(ImageChops.difference(first.image, second.image).getbbox())
        left, right = (component_tensors([frame.state], capacity=64) for frame in (first, second))
        self.assertTrue(all(torch.equal(value, right[key]) if torch.is_tensor(value) else value == right[key] for key, value in left.items()))
        self.assertFalse(bool(left["hold_piece_known"].any()))
        self.assertFalse(bool(left["queue_token_known"].any()))

    def test_target_masks_cannot_smuggle_hidden_start_end_or_hold_identity(self):
        unknown = observation(Queue((), "unknown", False, False))
        target = component_tensors([unknown], capacity=4)
        target["queue_tokens"][0, 0], target["queue_token_known"][0, 0] = 0, True
        with self.assertRaisesRegex(ValueError, "unseen queue start"):
            component_loss(self.output(4), target)
        target = component_tensors([observation(Queue((), "cropped", False, True))], capacity=4)
        target["queue_tokens"][0, 0], target["queue_token_known"][0, 0] = 7, True
        with self.assertRaisesRegex(ValueError, "unseen queue end"):
            component_loss(self.output(4), target)
        target = component_tensors([unknown], capacity=4)
        target["hold_piece_known"][0] = True
        with self.assertRaisesRegex(ValueError, "unobserved hold"):
            component_loss(self.output(4), target)

    def test_budgets_empty_context_and_prediction_targets_refuse(self):
        for capacity in (0, 65, True):
            with self.assertRaises(ValueError):
                self.output(capacity)
        with self.assertRaises(ValueError):
            self.head(self.memory, torch.zeros_like(self.valid), self.board, capacity=4)
        for capacity in (-1, 65, True):
            with self.assertRaises(ValueError):
                replace(ModelConfig.pilot(), component_capacity=capacity)
        predicted = replace(observation(Queue((), "unknown", False, False)), model_id="guessed-weights")
        with self.assertRaisesRegex(ValueError, "model output"):
            component_tensors([predicted], capacity=4)

    def test_m3_component_targets_refuse_resize_and_invalid_alpha(self):
        frame = render(sample_scene(4))
        resized = prepare_image(frame.image, max_side=128)
        with self.assertRaisesRegex(ValueError, "unchanged fully valid source"):
            targets_for(frame.state, resized, component_capacity=64)
        invalid = frame.image.convert("RGBA")
        invalid.putpixel((0, 0), (0, 0, 0, 0))
        prepared = prepare_image(invalid, max_side=960)
        with self.assertRaisesRegex(ValueError, "unchanged fully valid source"):
            targets_for(frame.state, prepared, component_capacity=64)

    def test_m2_outputs_unchanged_with_shared_board_weights(self):
        base = TetrisStateNet(ModelConfig.pilot()).eval()
        joint = TetrisStateNet(replace(ModelConfig.pilot(), component_capacity=4)).eval()
        incompatibility = joint.load_state_dict(base.state_dict(), strict=False)
        self.assertFalse(incompatibility.unexpected_keys)
        self.assertTrue(all(name.startswith("components.") for name in incompatibility.missing_keys))
        self.assertFalse(any(name.startswith("components.") for name in base.state_dict()))
        pixels, valid = torch.rand(1, 3, 96, 112), torch.ones(1, 1, 96, 112, dtype=torch.bool)
        with torch.no_grad():
            old, new = base(pixels, valid), joint(pixels, valid)
        self.assertTrue(all(torch.equal(value, new["final"][key]) for key, value in old["final"].items()))
        self.assertEqual(len(new["stages"]), 2)
        self.assertTrue(all("components" in stage for stage in new["stages"]))

    def test_joint_loss_reaches_encoder_and_component_queries(self):
        layout = Layout(canvas=(208, 256), board_left=8, board_bottom=248, cell_size=10,
                        queue_origin=(122, 8), hold_origin=(122, 184), icon_size=6)
        scene = replace(sample_scene(4), queue=("T", "I", "O"), hold="L")
        frame = render(scene, layout=layout)
        prepared = prepare_image(frame.image, max_side=256)
        model = TetrisStateNet(replace(ModelConfig.pilot(), component_capacity=4))
        target = targets_for(frame.state, prepared, component_capacity=4)
        output = model(prepared.pixels[None], prepared.valid[None])
        loss, terms = joint_loss(output, target)
        self.assertIn("components", terms)
        loss.backward()
        self.assertGreater(float(model.components.queue_tokens.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.components.index_query.grad.abs().sum()), 0)
        self.assertGreater(float(model.encoder.stages[0][0].weight.grad.abs().sum()), 0)
        self.assertTrue(all(value.grad is None or bool(torch.isfinite(value.grad).all()) for value in model.parameters()))
        component_only = TetrisStateNet(replace(ModelConfig.pilot(), component_capacity=4))
        output = component_only(prepared.pixels[None], prepared.valid[None])
        component_loss(output["final"]["components"], target["components"])[0].backward()
        self.assertGreater(float(component_only.encoder.stages[0][0].weight.grad.abs().sum()), 0)

    def test_m3_checkpoint_round_trip_and_legacy_config(self):
        pixels, valid = torch.rand(1, 3, 96, 112), torch.ones(1, 1, 96, 112, dtype=torch.bool)
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            joint = TetrisStateNet(replace(ModelConfig.pilot(), component_capacity=4)).eval()
            save_checkpoint(root/"joint", joint, {"test": True})
            restored, manifest = load_checkpoint(root/"joint")
            self.assertEqual(manifest["architecture"], "TetrisStateNet-M3")
            with torch.no_grad():
                before, after = joint(pixels, valid)["final"], restored.eval()(pixels, valid)["final"]
            self.assertTrue(all(torch.equal(value, after["components"][key]) for key, value in before["components"].items()))
            manifest["architecture"] = "TetrisStateNet-M2"
            (root/"joint/manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "architecture identity"):
                load_checkpoint(root/"joint")
            old = TetrisStateNet(ModelConfig.pilot()).eval()
            manifest = save_checkpoint(root/"old", old, {"test": True})
            del manifest["config"]["component_capacity"]
            (root/"old/manifest.json").write_text(json.dumps(manifest))
            restored, _ = load_checkpoint(root/"old")
            self.assertEqual(restored.config.component_capacity, 0)
            self.assertTrue(all(torch.equal(value, restored.state_dict()[key]) for key, value in old.state_dict().items()))

    def test_uncalibrated_api_keeps_components_unknown(self):
        frame = render(sample_scene(4))
        prepared = prepare_image(frame.image, max_side=96)
        model = TetrisStateNet(replace(ModelConfig.pilot(), component_capacity=4)).eval()
        with torch.no_grad():
            output = model(prepared.pixels[None], prepared.valid[None])["final"]
            state = decode_partial(output, prepared, model_id="untrained-joint")
        self.assertEqual(state.decision, "needs_review")
        self.assertEqual(state.queue.visibility, "unknown")
        self.assertEqual(state.hold.status, "unknown")
        self.assertEqual(state.producer, "TetrisStateNet-M3-uncalibrated")


if __name__ == "__main__":
    unittest.main()
