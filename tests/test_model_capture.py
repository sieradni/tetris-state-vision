"""Final-capture visibility contracts, including paired identical raster cases."""
from dataclasses import replace
import importlib.util
import unittest

from PIL import ImageChops

from tetris_vision.component_targets import component_labels
from tetris_vision.layout import Layout
from tetris_vision.model_capture import model_capture_style, render_for_model
from tetris_vision.render import Appearance, render, sample_scene


class ModelCaptureTests(unittest.TestCase):
    def test_weak_final_icons_do_not_reuse_known_source_identities(self):
        scene = sample_scene(4)
        layout = Layout(canvas=(1024, 1024), icon_size=6)
        source = render(scene, layout=layout)
        final = render_for_model(scene, layout=layout, max_side=512)
        self.assertTrue(all(icon.status == "observed" for icon in source.state.queue.items))
        self.assertEqual(source.state.hold.status, "observed")
        self.assertEqual(final.image.size, (512, 512))
        self.assertTrue(all(icon.status == "unknown" for icon in final.state.queue.items))
        self.assertEqual(final.state.hold.status, "unknown")
        self.assertEqual(final.state.board.lattice.u, (10., 0.))
        self.assertTrue(all(cell.occupied is not None for row in final.state.board.rows for cell in row))
        labels = component_labels(final.state)
        self.assertFalse(any(labels["queue_token_known"][:len(scene.queue)]))
        self.assertFalse(labels["hold_piece_known"])

    def test_noninteger_capture_uses_final_rounded_axes_for_geometry(self):
        scene = sample_scene(4)
        style = Appearance(crop=(16, 12, 468, 520), output_size=(377, 461))
        effective = model_capture_style(style, Layout(), max_side=320)
        final = render_for_model(scene, style, max_side=320)
        self.assertEqual(effective.output_size, (262, 320))
        self.assertEqual(final.image.size, (262, 320))
        lattice = final.state.board.lattice
        self.assertAlmostEqual(lattice.origin[0], 48*262/452)
        self.assertAlmostEqual(lattice.origin[1], 478*320/508)
        self.assertAlmostEqual(lattice.u[0], 20*262/452)
        self.assertAlmostEqual(lattice.v[1], -20*320/508)

    def test_hidden_queue_length_and_hold_stay_identical_after_resize(self):
        scene, style = sample_scene(44), Appearance(crop=(60, 0, 282, 540))
        first = render_for_model(replace(scene, queue=("I",), hold="L"), style, max_side=512)
        second = render_for_model(replace(scene, queue=("T",)*9, hold="Z"), style, max_side=512)
        self.assertIsNone(ImageChops.difference(first.image, second.image).getbbox())
        self.assertEqual(first.state.to_dict(), second.state.to_dict())
        labels = component_labels(first.state)
        self.assertFalse(any(labels["queue_token_known"]))
        self.assertFalse(labels["hold_piece_known"])
        self.assertFalse(labels["queue_count_known"])

    def test_occluded_middle_retains_slot_without_latent_identity(self):
        scene, style = sample_scene(44), Appearance(occluded_queue=(1,))
        first = render_for_model(replace(scene, queue=("T", "O", "I")), style, max_side=384)
        second = render_for_model(replace(scene, queue=("T", "L", "I")), style, max_side=384)
        self.assertIsNone(ImageChops.difference(first.image, second.image).getbbox())
        left, right = (component_labels(frame.state, capacity=4) for frame in (first, second))
        self.assertEqual(left, right)
        self.assertEqual(left["queue_tokens"][:4], (2, -1, 0, 7))
        self.assertTrue(left["queue_presence"][1])
        self.assertTrue(left["queue_presence_known"][1])

    def test_unseen_prefix_count_does_not_survive_matching_visible_span(self):
        scene, style = sample_scene(44), Appearance(crop=(0, 132, 480, 540))
        first = render_for_model(replace(scene, queue=("L", "T", "I")), style, max_side=384)
        shifted = replace(Layout(), queue_origin=(308, 40))
        second = render_for_model(replace(scene, queue=("S", "Z", "T", "I")), style, shifted, max_side=384)
        self.assertIsNone(ImageChops.difference(first.image, second.image).getbbox())
        self.assertFalse(first.state.queue.start_observed)
        self.assertEqual(tuple(icon.piece for icon in first.state.queue.items), ("T", "I"))
        left, right = (component_labels(frame.state, capacity=4) for frame in (first, second))
        self.assertEqual(left, right)
        self.assertFalse(any(left["queue_token_known"]))
        self.assertFalse(left["queue_count_known"])

    def test_unscaled_capture_preserves_existing_pixels_and_labels(self):
        scene = sample_scene(17)
        original, final = render(scene), render_for_model(scene, max_side=960)
        self.assertEqual(original.image.tobytes(), final.image.tobytes())
        self.assertEqual(original.state.to_dict(), final.state.to_dict())

    def test_bounds_do_not_silently_distort_or_upscale(self):
        for side in (0, 63, 1537, True):
            with self.assertRaises(ValueError):
                model_capture_style(Appearance(), Layout(), max_side=side)
        panorama = Layout(canvas=(1536, 192), board_bottom=184, cell_size=8)
        with self.assertRaisesRegex(ValueError, "below renderer capacity"):
            model_capture_style(Appearance(), panorama, max_side=128)
        with self.assertRaisesRegex(ValueError, "outside source"):
            model_capture_style(Appearance(crop=(0, 0, 1000, 1000)), Layout())
        effective = model_capture_style(Appearance(output_size=(96, 128)), Layout(), max_side=512)
        self.assertEqual(effective.output_size, (96, 128))


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional train dependency required")
class PreparedCaptureTests(unittest.TestCase):
    def test_final_capture_targets_need_no_second_resize(self):
        import torch
        from tetris_vision.losses import targets_for
        from tetris_vision.preprocess import prepare_image
        torch.set_num_threads(1)
        style = Appearance(crop=(16, 12, 468, 520), output_size=(377, 461))
        final = render_for_model(sample_scene(4), style, max_side=320)
        prepared = prepare_image(final.image, max_side=320)
        self.assertEqual(prepared.source_size, prepared.resized_size)
        self.assertEqual(prepared.source_size, (262, 320))
        target = targets_for(final.state, prepared, component_capacity=64)
        self.assertEqual(int(target["known"].sum()), 200)
        self.assertEqual(target["components"]["queue_tokens"].shape, (1, 65))
        self.assertTrue(bool(target["components"]["queue_token_known"].any()))
        # A second unobserved resize still refuses; no synthetic shortcut bypass.
        smaller = prepare_image(final.image, max_side=256)
        with self.assertRaisesRegex(ValueError, "unchanged fully valid source"):
            targets_for(final.state, smaller, component_capacity=64)

    def test_prepared_identical_rasters_have_identical_masked_targets(self):
        import torch
        from tetris_vision.losses import targets_for
        from tetris_vision.preprocess import prepare_image
        torch.set_num_threads(1)
        scene, style = sample_scene(44), Appearance(crop=(60, 0, 282, 540))
        frames = [render_for_model(replace(scene, queue=queue, hold=hold), style, max_side=512)
                  for queue, hold in ((("I",), "L"), (("T",)*9, "Z"))]
        prepared = [prepare_image(frame.image, max_side=512) for frame in frames]
        self.assertTrue(torch.equal(prepared[0].pixels, prepared[1].pixels))
        targets = [targets_for(frame.state, image, component_capacity=64)["components"]
                   for frame, image in zip(frames, prepared)]
        self.assertTrue(all(torch.equal(value, targets[1][key]) if torch.is_tensor(value) else value == targets[1][key]
                            for key, value in targets[0].items()))


if __name__ == "__main__":
    unittest.main()
