from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest

from PIL import Image

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from tetris_vision.checkpoint import load_checkpoint, save_checkpoint
    from tetris_vision.losses import diagnostic_metrics, joint_loss, targets_for
    from tetris_vision.model import ModelConfig, TetrisStateNet, decode_lattice, sample_features
    from tetris_vision.preprocess import prepare_image
    from tetris_vision.render import Appearance, render, sample_scene


@unittest.skipUnless(HAS_TORCH, "install the optional train dependency for M2 gates")
class ModelContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(44)
        self.model = TetrisStateNet(ModelConfig.pilot())
        self.pixels = torch.rand(1, 3, 96, 112)
        self.valid = torch.ones(1, 1, 96, 112, dtype=torch.bool)

    def test_aspect_ratio_rounding_and_padding_own_geometry(self):
        result = render(sample_scene(4))
        prepared = prepare_image(result.image, max_side=383)
        self.assertEqual(prepared.resized_size, (340, 383))
        self.assertEqual(prepared.padded_size, (352, 384))
        self.assertEqual(int(prepared.valid.sum()), 340*383)
        target = targets_for(result.state, prepared)
        self.assertAlmostEqual(float(target["lattice"][0, 0, 0]), 64*(340/480)/352, places=6)
        self.assertAlmostEqual(float(target["lattice"][0, 0, 1]), 490*(383/540)/384, places=6)

    def test_transparent_rgb_is_not_hidden_evidence(self):
        left = Image.new("RGBA", (80, 80), (255, 0, 0, 0))
        right = Image.new("RGBA", (80, 80), (0, 255, 0, 0))
        for image in (left, right):
            image.putpixel((7, 7), (5, 6, 7, 255))
        a, b = prepare_image(left), prepare_image(right)
        self.assertTrue(torch.equal(a.pixels, b.pixels))
        self.assertTrue(torch.equal(a.valid, b.valid))
        self.assertEqual(int(a.valid.sum()), 1)

    def test_forward_has_finite_complete_heads_and_bounded_samples(self):
        outputs = self.model(self.pixels, self.valid)
        self.assertEqual(len(outputs["stages"]), 2)
        final = outputs["final"]
        self.assertEqual(final["occupancy"].shape, (1, 26, 10))
        self.assertEqual(final["semantic"].shape, (1, 26, 10, 8))
        self.assertEqual(final["layer_type"].shape, (1, 2, 8))
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in final.values()))
        self.assertLessEqual(float(final["sample_offsets"].detach().abs().max()), .450001)
        geometry = final["lattice"]
        self.assertTrue(bool((geometry[:, 1, 0] > 0).all() and (geometry[:, 2, 1] < 0).all()))

    def test_lattice_is_nonfolding_even_for_extreme_logits(self):
        lattice = decode_lattice(torch.tensor([[80., -80., -80., 80.], [-80., 80., 80., -80.]]))
        determinant = lattice[:, 1, 0]*lattice[:, 2, 1]-lattice[:, 1, 1]*lattice[:, 2, 0]
        self.assertTrue(bool((determinant < 0).all()))
        self.assertTrue(bool(torch.isfinite(lattice).all()))

    def test_recognition_loss_reaches_learned_geometry_without_geometry_loss(self):
        output = self.model(self.pixels, self.valid)["final"]
        output["occupancy"].mean().backward()
        self.assertGreater(float(self.model.initial_box[1].weight.grad.abs().sum()), 0)

    def test_values_in_padding_cannot_change_predictions(self):
        self.valid[:, :, 80:, :] = False
        modified = self.pixels.clone()
        modified[:, :, 80:, :] = 100
        with torch.no_grad():
            left = self.model(self.pixels, self.valid)["final"]["occupancy"]
            right = self.model(modified, self.valid)["final"]["occupancy"]
        self.assertTrue(torch.equal(left, right))

    def test_sample_coordinates_and_gradient_reach_lattice(self):
        # Feature bins have source boundary centers (.5,1.5,...). A linear x
        # field supplies an independent analytic oracle for align_corners=False.
        feature = (torch.arange(16).float()+.5).reshape(1, 1, 1, 16).expand(1, 1, 16, 16)
        lattice = torch.tensor([[[.1, .8], [.05, 0.], [0., -.05]]], requires_grad=True)
        coordinates = torch.tensor([[2.5, 2.5]])
        samples, points = sample_features(feature, lattice, coordinates, torch.zeros(1, 1, 1, 2))
        self.assertAlmostEqual(float(samples.detach()[0, 0, 0, 0]), float(points.detach()[0, 0, 0, 0])*16, places=5)
        samples.sum().backward()
        self.assertGreater(float(lattice.grad.abs().sum()), 1)

    def test_all_unknown_targets_have_finite_loss_and_gradients(self):
        result = render(sample_scene(4), Appearance(locked_alpha=0))
        prepared = prepare_image(result.image, max_side=96)
        target = targets_for(result.state, prepared)
        target["registered"].fill_(False)
        target["known"].fill_(False)
        target["typed"].fill_(False)
        target["layer_known"].fill_(False)
        target["pose_known"].fill_(False)
        target["rotations"].fill_(False)
        outputs = self.model(prepared.pixels[None], prepared.valid[None])
        loss, terms = joint_loss(outputs, target)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in self.model.parameters()))
        self.assertEqual(float(terms["occupancy"].detach()), 0)
        self.assertIsNone(diagnostic_metrics(outputs, target)["cell_accuracy"])

    def test_unknown_types_do_not_gate_occupancy_or_train_type_head(self):
        result = render(sample_scene(4), Appearance(monochrome=True))
        prepared = prepare_image(result.image, max_side=384)
        target = targets_for(result.state, prepared)
        self.assertTrue(bool(target["known"].any()))
        self.assertFalse(bool(target["typed"].any()))
        outputs = self.model(prepared.pixels[None], prepared.valid[None])
        loss, terms = joint_loss(outputs, target)
        loss.backward()
        self.assertEqual(float(terms["semantic"].detach()), 0)
        self.assertGreater(float(self.model.occupancy.weight.grad.abs().sum()), 0)
        self.assertEqual(float(self.model.semantic.weight.grad.abs().sum()), 0)

    def test_unregistered_crop_supplies_no_canonical_training_targets(self):
        result = render(sample_scene(4), Appearance(crop=(90, 0, 480, 540)))
        prepared = prepare_image(result.image, max_side=384)
        target = targets_for(result.state, prepared)
        self.assertFalse(bool(target["registered"].any()))
        self.assertFalse(bool(target["known"].any()))
        self.assertFalse(bool(target["pose_known"].any()))

    def test_one_valid_pixel_never_creates_all_masked_attention_nan(self):
        valid = torch.zeros_like(self.valid)
        valid[:, :, 7, 7] = True
        output = self.model(self.pixels, valid)["final"]
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in output.values()))

    def test_work_budget_and_empty_mask_fail_before_encoder(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            self.model(self.pixels.expand(5, -1, -1, -1), self.valid.expand(5, -1, -1, -1))
        with self.assertRaisesRegex(ValueError, "empty"):
            self.model(self.pixels, torch.zeros_like(self.valid))
        with self.assertRaises(ValueError):
            ModelConfig(heads=0)

    def test_symmetric_rotation_targets_are_sets_not_arbitrary_labels(self):
        from tetris_vision.render import Pose
        scene = replace(sample_scene(4, "empty"), active=Pose("O", 3, 14))
        result = render(scene)
        prepared = prepare_image(result.image, max_side=384)
        target = targets_for(result.state, prepared)
        self.assertTrue(bool(target["rotations"][0, 0].all()))

    def test_checkpoint_round_trip_and_corruption_refusal(self):
        self.model.eval()
        with torch.no_grad():
            expected = self.model(self.pixels, self.valid)["final"]["occupancy"]
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)/"checkpoint"
            record = save_checkpoint(root, self.model, {"test": True})
            loaded, restored = load_checkpoint(root)
            self.assertEqual(record["weights_sha256"], restored["weights_sha256"])
            with torch.no_grad():
                actual = loaded.eval()(self.pixels, self.valid)["final"]["occupancy"]
            self.assertTrue(torch.equal(expected, actual))
            with (root/"weights.pt").open("ab") as output:
                output.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_checkpoint(root)
