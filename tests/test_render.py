from dataclasses import replace
import unittest

from tetris_vision.render import Appearance, Pose, Scene, families, render, sample_scene
from tetris_vision.state import VisionState


class RenderIntegrityTests(unittest.TestCase):
    def test_independent_pixel_oracle_all_200_cells(self):
        # Hand-coded pixel coordinates/color values, independent of lattice and
        # shape helpers. Asymmetry witnesses both row direction and column order.
        rows = [[None]*10 for _ in range(20)]
        for c, r in ((0, 0), (9, 1), (2, 7), (6, 19)):
            rows[r][c] = "T"
        result = render(Scene(tuple(map(tuple, rows)), queue=(), hold=None))
        for row in range(20):
            for column in range(10):
                pixel = result.image.getpixel((74+20*column, 480-20*row))
                occupied = pixel == (176, 90, 230)
                self.assertEqual(occupied, rows[row][column] is not None, (column, row))
                self.assertEqual(occupied, result.state.board.rows[row][column].occupied)

    def test_seeded_images_and_labels_are_deterministic(self):
        scene, style = sample_scene(17), Appearance(seed=20, background="blocks")
        left, right = render(scene, style), render(scene, style)
        self.assertEqual(left.image.tobytes(), right.image.tobytes())
        self.assertEqual(left.state, right.state)

    def test_grid_and_palette_counterfactuals_preserve_locked_occupancy(self):
        scene = sample_scene(21)
        results = [render(scene, style) for _, style in families(21).items() if style.crop is None]
        occupancies = [tuple(tuple(c.occupied for c in row) for row in result.state.board.rows) for result in results]
        self.assertTrue(all(value == occupancies[0] for value in occupancies))
        self.assertNotEqual(results[0].image.tobytes(), results[1].image.tobytes())

    def test_monochrome_preserves_occupancy_but_not_unknown_types(self):
        state = render(sample_scene(11), Appearance(monochrome=True)).state
        self.assertTrue(any(cell.occupied for row in state.board.rows for cell in row))
        self.assertTrue(all(cell.semantic is None for row in state.board.rows for cell in row))

    def test_partial_crop_is_unknown_not_latent_truth(self):
        state = render(sample_scene(11), Appearance(crop=(90, 0, 480, 477))).state
        self.assertEqual(state.decision, "needs_review")
        self.assertIsNone(state.board.rows[0][0].occupied)
        self.assertEqual(state.board.rows[0][0].visibility, "cropped")
        self.assertIsNone(state.board.lattice)

    def test_anchor_preserving_crop_and_noninteger_resize_transform_centers(self):
        scene = sample_scene(19)
        before = render(scene).state.board.lattice
        after = render(scene, Appearance(crop=(20, 20, 450, 520), output_size=(516, 650))).state.board.lattice
        for c, r in ((0, 0), (9, 19), (4, 12)):
            x, y = before.center(c, r)
            xx, yy = after.center(c, r)
            self.assertAlmostEqual(xx, (x-20)*1.2)
            self.assertAlmostEqual(yy, (y-20)*1.3)

    def test_missing_panels_never_leak_latent_icons(self):
        scene = replace(sample_scene(16), queue_present=False, hold_present=False)
        state = render(scene).state
        self.assertEqual(state.queue.visibility, "absent")
        self.assertEqual(state.queue.items, ())
        self.assertEqual(state.hold.status, "absent")
        self.assertIsNone(state.hold.piece)

    def test_repeated_and_zero_queue_slots(self):
        state = render(replace(sample_scene(8), queue=("T", "T", "I"))).state
        self.assertEqual(tuple(i.piece for i in state.queue.items), ("T", "T", "I"))
        self.assertEqual(render(replace(sample_scene(8), queue=())).state.queue.items, ())

    def test_cropped_queue_preserves_unknown_positions_and_endpoint(self):
        state = render(sample_scene(8), Appearance(crop=(0, 0, 350, 540))).state
        self.assertEqual(len(state.queue.items), 5)
        self.assertTrue(all(i.status == "cropped" and i.piece is None for i in state.queue.items))
        self.assertFalse(state.queue.end_observed)

    def test_fully_invisible_layer_does_not_supply_known_occupancy(self):
        state = render(sample_scene(8), Appearance(locked_alpha=0)).state
        self.assertTrue(all(cell.occupied is None for row in state.board.rows for cell in row))

    def test_empty_board_with_no_evidence_has_no_invented_lattice(self):
        scene = replace(sample_scene(8, "empty"), active=None, ghost=None, queue_present=False, hold_present=False)
        state = render(scene, Appearance(grid=False, border=False, board_alpha=0)).state
        self.assertIsNone(state.board.lattice)
        self.assertEqual(state.decision, "needs_review")

    def test_sparse_and_empty_are_explicit(self):
        for density, lower, upper in (("empty", 0, 0), ("sparse", 1, 4), ("dense", 40, 75)):
            for seed in range(3):
                state = render(sample_scene(seed, density)).state
                count = sum(cell.occupied is True for row in state.board.rows for cell in row)
                self.assertTrue(lower <= count <= upper)

    def test_active_head_never_deletes_locked_tetromino(self):
        scene = sample_scene(9, "empty")
        rows = [list(row) for row in scene.locked]
        for c, r in Pose("T", 6, 9).cells:
            rows[r][c] = "T"
        result = render(replace(scene, locked=tuple(map(tuple, rows))))
        self.assertEqual(sum(c.occupied is True for row in result.state.board.rows for c in row), 4)
        self.assertEqual(result.state.active.cells, scene.active.cells)

    def test_semantic_hard_negative_changes_exactly_one_locked_cell(self):
        scene = sample_scene(2, "empty")
        rows = [list(row) for row in scene.locked]
        rows[0][9] = "L"
        before, after = render(scene), render(replace(scene, locked=tuple(map(tuple, rows))))
        self.assertNotEqual(before.image.tobytes(), after.image.tobytes())
        self.assertEqual(sum(a != b for ra, rb in zip(before.state.board.rows, after.state.board.rows)
                             for a, b in zip(ra, rb)), 1)

    def test_spawn_clipping_does_not_leak_off_image_pose(self):
        state = render(replace(sample_scene(2, "empty"), active=Pose("T", 3, 24))).state
        self.assertEqual(state.active.status, "cropped")
        self.assertEqual(state.active.cells, ())

    def test_all_families_roundtrip(self):
        for style in families(23).values():
            result = render(sample_scene(23), style)
            self.assertEqual(VisionState.from_json(result.state.to_json()), result.state)
            self.assertTrue(all(mask.size == result.image.size for mask in result.layer_masks.values()))

    def test_indistinguishable_active_and_locked_states_are_both_ambiguous(self):
        scene = replace(sample_scene(1, "empty"), ghost=None, active=Pose("T", 3, 12))
        rows = [list(row) for row in scene.locked]
        for c, r in scene.active.cells:
            rows[r][c] = "T"
        other = replace(scene, locked=tuple(map(tuple, rows)), active=None)
        style = Appearance(layer_cues=False)
        before, after = render(scene, style), render(other, style)
        self.assertEqual(before.image.tobytes(), after.image.tobytes())
        self.assertEqual(before.state, after.state)
        self.assertEqual(before.state.active.status, "unknown")
        self.assertEqual(before.state.decision, "needs_review")

    def test_unseen_palette_types_are_not_inferred_from_renderer_memory(self):
        scene = replace(sample_scene(1, "empty"), queue=("T",), hold=None)
        rows = [list(row) for row in scene.locked]
        rows[0][0], rows[0][1] = "Z", "T"
        state = render(replace(scene, locked=tuple(map(tuple, rows))), Appearance(palette_shift=2)).state
        self.assertTrue(state.board.rows[0][0].occupied)
        self.assertIsNone(state.board.rows[0][0].semantic)
        self.assertEqual(state.board.rows[0][1].semantic, "T")


if __name__ == "__main__":
    unittest.main()
