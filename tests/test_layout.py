from dataclasses import replace
import unittest

from tetris_vision.layout import Layout, sample_layout
from tetris_vision.render import Appearance, Pose, Scene, render, sample_scene
from tetris_vision.state import VisionState


class ObservationEquivalenceTests(unittest.TestCase):
    """Standing visibility-contract instrument; compare actual final rasters."""
    def assert_equivalent(self, left, right, style, layout=Layout()):
        a, b = render(left, style, layout), render(right, style, layout)
        self.assertEqual(a.image.tobytes(), b.image.tobytes(), "counterexample must really be indistinguishable")
        self.assertEqual(a.state, b.state, "latent state leaked into image-supported labels")
        return a.state

    def test_fully_cropped_queue_does_not_leak_length_or_identity(self):
        scene = sample_scene(8)
        state = self.assert_equivalent(replace(scene, queue=("T",)),
                                      replace(scene, queue=("I",)*8), Appearance(crop=(0, 0, 300, 540)))
        self.assertEqual(state.queue.items, ())
        self.assertEqual(state.queue.visibility, "unknown")
        self.assertFalse(state.queue.start_observed)

    def test_cropped_hold_does_not_leak_identity(self):
        scene = sample_scene(8)
        state = self.assert_equivalent(replace(scene, hold="T"), replace(scene, hold="I"),
                                      Appearance(crop=(60, 0, 480, 540)))
        self.assertEqual(state.hold.status, "unknown")

    def test_unseen_spawn_piece_and_absent_piece_have_same_uncertainty(self):
        scene = replace(sample_scene(8, "empty"), ghost=None)
        state = self.assert_equivalent(replace(scene, active=Pose("I", 3, 25)), replace(scene, active=None), Appearance())
        self.assertEqual(state.active.status, "unknown")

    def test_occluded_locked_cell_is_not_a_known_removal(self):
        scene = sample_scene(8, "empty")
        rows = [list(row) for row in scene.locked]
        rows[3][2] = "Z"
        state = self.assert_equivalent(scene, replace(scene, locked=tuple(map(tuple, rows))),
                                      Appearance(occluded_cells=((2, 3),)))
        self.assertIsNone(state.board.rows[3][2].occupied)

    def test_occluded_middle_preview_keeps_unknown_index(self):
        scene = replace(sample_scene(8), queue=("T", "T", "I"))
        state = self.assert_equivalent(scene, replace(scene, queue=("T", "Z", "I")), Appearance(occluded_queue=(1,)))
        self.assertEqual(len(state.queue.items), 3)
        self.assertEqual(state.queue.items[1].status, "unknown")
        self.assertEqual(state.queue.items[2].piece, "I")

    def test_unseen_queue_prefix_does_not_recover_latent_items(self):
        scene = sample_scene(8)
        state = self.assert_equivalent(scene, replace(scene, queue=("Z", *scene.queue[1:])),
                                      Appearance(crop=(0, 135, 480, 540)))
        self.assertEqual(len(state.queue.items), 4)
        self.assertFalse(state.queue.start_observed)

    def test_unseen_queue_suffix_does_not_recover_latent_count(self):
        scene = sample_scene(8)
        state = self.assert_equivalent(replace(scene, queue=("T", "T")),
                                      replace(scene, queue=("T", "T", "I", "L")),
                                      Appearance(crop=(0, 0, 480, 150)))
        self.assertEqual(len(state.queue.items), 2)
        self.assertFalse(state.queue.end_observed)

    def test_unanchored_empty_board_does_not_reveal_latent_origin(self):
        scene = replace(sample_scene(8, "empty"), active=None, ghost=None)
        style = Appearance(grid=False, border=False, board_alpha=0)
        left = render(scene, style)
        right = render(scene, style, replace(Layout(), board_left=100))
        self.assertEqual(left.image.tobytes(), right.image.tobytes())
        self.assertEqual(left.state, right.state)

    def test_header_only_crop_cannot_distinguish_empty_and_hidden_queue(self):
        scene = sample_scene(8)
        state = self.assert_equivalent(replace(scene, queue=()), replace(scene, queue=("T", "I")),
                                      Appearance(crop=(0, 0, 480, 100)))
        self.assertEqual(state.queue.items, ())
        self.assertFalse(state.queue.start_observed)

    def test_occluded_hold_cannot_reveal_empty_or_piece(self):
        scene = sample_scene(8)
        state = self.assert_equivalent(replace(scene, hold=None), replace(scene, hold="T"), Appearance(hold_occluded=True))
        self.assertEqual(state.hold.status, "unknown")

    def test_invisible_active_layer_does_not_reveal_presence_or_mask_position(self):
        scene = replace(sample_scene(8, "empty"), ghost=None)
        self.assert_equivalent(scene, replace(scene, active=None), Appearance(active_alpha=0))

    def test_invisible_ghost_layer_does_not_reveal_presence_or_mask_position(self):
        scene = sample_scene(8, "empty")
        self.assert_equivalent(scene, replace(scene, ghost=None), Appearance(ghost_alpha=0))


class LayoutIntegrityTests(unittest.TestCase):
    def test_independent_oracle_across_three_source_scales(self):
        rows = [[None]*10 for _ in range(20)]
        for c, r in ((0, 0), (9, 1), (2, 7), (6, 19)):
            rows[r][c] = "T"
        scene = Scene(tuple(map(tuple, rows)), queue=(), hold=None)
        for scale in (12, 20, 27):
            layout = Layout(canvas=(800, 900), board_left=100, board_bottom=800,
                            cell_size=scale, queue_origin=(500, 74))
            result = render(scene, Appearance(), layout)
            for r in range(20):
                for c in range(10):
                    # Coordinates independently specified from the chosen source
                    # origin/spacing, not read from the renderer's geometry.
                    x, y = 100+(c+.5)*scale, 800-(r+.5)*scale
                    occupied = result.image.getpixel((int(x), int(y))) == (176, 90, 230)
                    self.assertEqual(occupied, rows[r][c] is not None)
                    self.assertEqual(occupied, result.state.board.rows[r][c].occupied)
                    self.assertEqual(result.state.board.lattice.center(c, r), (x, y))

    def test_all_queue_flows_preserve_repeats_and_order(self):
        scene = replace(sample_scene(5), queue=("T", "T", "I", "L", "S", "Z"))
        for flow in ("vertical", "horizontal", "wrapped"):
            layout = sample_layout(25, len(scene.queue), flow=flow)
            state = render(scene, Appearance(), layout).state
            self.assertEqual(tuple(item.piece for item in state.queue.items), scene.queue)
            self.assertTrue(state.queue.start_observed)
            self.assertTrue(state.queue.end_observed)
            points = [layout.slot_origin(i) for i in range(6)]
            self.assertEqual(len(set(points)), 6)
            if flow == "wrapped":
                self.assertLess(points[3][0], points[2][0])
                self.assertGreater(points[3][1], points[2][1])

    def test_randomized_layouts_are_bounded_and_nonoverlapping(self):
        origins, scales, flows = set(), set(), set()
        for seed in range(30):
            scene = sample_scene(seed, varied=True)
            layout = sample_layout(seed, len(scene.queue))
            layout.validate_panels(scene)
            origins.add((layout.board_left, layout.board_bottom))
            scales.add(layout.cell_size)
            flows.add(layout.queue_flow)
        self.assertGreater(len(origins), 20)
        self.assertGreater(len(scales), 8)
        self.assertEqual(flows, {"vertical", "horizontal", "wrapped"})

    def test_independent_preview_skin_cannot_supply_board_palette_mapping(self):
        scene = replace(sample_scene(3, "empty"), queue=("Z",), hold=None, active=None, ghost=None)
        rows = [list(row) for row in scene.locked]
        rows[0][0] = "Z"
        scene = replace(scene, locked=tuple(map(tuple, rows)))
        matched = render(scene, Appearance(palette_shift=2, preview_palette_shift=2)).state
        separate = render(scene, Appearance(palette_shift=2, preview_palette_shift=5)).state
        self.assertEqual(matched.board.rows[0][0].semantic, "Z")
        self.assertIsNone(separate.board.rows[0][0].semantic)
        self.assertTrue(separate.board.rows[0][0].occupied)

    def test_all_rotations_and_piece_types_have_valid_active_targets(self):
        for piece in "IOTSZJL":
            for rotation in range(4):
                scene = replace(sample_scene(3, "empty"), active=Pose(piece, 3, 14, rotation), ghost=None)
                state = render(scene).state
                self.assertEqual(state.active.piece, piece)
                self.assertEqual(state.active.cells, scene.active.cells)

    def test_varied_sampler_covers_types_lengths_and_absence(self):
        types, rotations, lengths, absence = set(), set(), set(), set()
        for seed in range(200):
            scene = sample_scene(seed, varied=True)
            if scene.active:
                types.add(scene.active.piece)
                rotations.add(scene.active.rotation)
            lengths.add(len(scene.queue))
            absence.add((scene.queue_present, scene.hold_present))
        self.assertEqual(types, set("IOTSZJL"))
        self.assertEqual(rotations, set(range(4)))
        self.assertEqual(lengths, set(range(11)))
        self.assertEqual(absence, {(True, True), (True, False), (False, True), (False, False)})

    def test_historical_schema_is_read_conservatively(self):
        state = render(sample_scene(6)).state.to_dict()
        state["schema_version"] = "1.0"
        del state["queue"]["start_observed"]
        migrated = VisionState.from_dict(state)
        self.assertEqual(migrated.schema_version, "1.1")
        self.assertTrue(migrated.queue.start_observed)

    def test_source_panel_overlap_is_explicitly_refused(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            render(sample_scene(5), layout=replace(Layout(), queue_origin=(70, 100)))
