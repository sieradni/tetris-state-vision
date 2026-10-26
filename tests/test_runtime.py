import importlib.util
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from tetris_vision.adapter import IncompleteState, to_solver_input
    from tetris_vision.checkpoint import save_checkpoint
    from tetris_vision.model import ModelConfig, TetrisStateNet
    from tetris_vision.preprocess import prepare_image
    from tetris_vision.runtime import ImageParser, decode_partial
    from tetris_vision.state import VisionState


@unittest.skipUnless(HAS_TORCH, "optional train dependency required")
class RuntimeContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.prepared = prepare_image(Image.new("RGB", (160, 160), (10, 20, 30)))
        self.output = {
            "lattice": torch.tensor([[[.1, .9], [.08, 0.], [0., -.04]]]),
            "registration": torch.tensor([20.]), "occupancy": torch.full((1, 26, 10), 20.),
            "observability": torch.full((1, 26, 10), 20.), "semantic": torch.zeros((1, 26, 10, 8)),
            "layer_known": torch.full((1, 2), 20.), "layer_type": torch.zeros((1, 2, 8)),
            "pose_rotation": torch.zeros((1, 2, 4)), "pose_column": torch.zeros((1, 2, 10)),
            "pose_row": torch.zeros((1, 2, 26))}
        self.output["layer_type"][:, :, 1] = 30  # I
        self.output["pose_rotation"][:, :, 0] = 30
        self.output["pose_column"][:, :, 4] = 30
        self.output["pose_row"][:, :, 5] = 30

    def test_piece_guess_never_deletes_locked_cells_or_forces_semantic_type(self):
        state = decode_partial(self.output, self.prepared, model_id="fixture")
        self.assertEqual(sum(cell.occupied is True for row in state.board.rows for cell in row), 200)
        self.assertTrue(all(cell.semantic is None for row in state.board.rows for cell in row))
        self.assertEqual(state.active.status, "observed")
        self.assertEqual(state.decision, "needs_review")
        self.assertEqual(state.queue.visibility, "unknown")
        self.assertFalse(state.queue.start_observed)
        self.assertEqual(state.hold.status, "unknown")
        with self.assertRaises(IncompleteState):
            to_solver_input(state, hold_available=True)

    def test_unknown_registration_masks_every_canonical_output(self):
        self.output["registration"].fill_(-20)
        state = decode_partial(self.output, self.prepared, model_id="fixture")
        self.assertIsNone(state.board.lattice)
        self.assertTrue(all(cell.occupied is None for row in state.board.rows for cell in row))
        self.assertEqual(state.active.status, "unknown")
        self.assertEqual(state.ghost.status, "unknown")

    def test_bottom_first_asymmetric_cells_survive_json_serialization(self):
        self.output["occupancy"].fill_(-20)
        self.output["occupancy"][0, 0, 1] = 20
        self.output["occupancy"][0, 19, 8] = 20
        state = decode_partial(self.output, self.prepared, model_id="fixture")
        restored = VisionState.from_json(state.to_json())
        self.assertTrue(restored.board.rows[0][1].occupied)
        self.assertTrue(restored.board.rows[19][8].occupied)
        self.assertFalse(restored.board.rows[19][1].occupied)
        self.assertFalse(restored.board.rows[0][8].occupied)

    def test_source_alpha_hole_cannot_be_a_known_cell(self):
        image = Image.new("RGBA", (160, 160), (10, 20, 30, 255))
        ImageDraw.Draw(image).rectangle((65, 100, 85, 115), fill=(255, 0, 0, 0))
        state = decode_partial(self.output, prepare_image(image), model_id="fixture")
        self.assertIsNotNone(state.board.lattice)
        self.assertIsNone(state.board.rows[5][4].occupied)

    def test_joint_pose_selection_respects_shape_capacity_without_clamping_cells(self):
        self.output["pose_column"].zero_()
        self.output["pose_column"][:, :, 9] = 30  # Invalid for a horizontal I.
        state = decode_partial(self.output, self.prepared, model_id="fixture")
        self.assertEqual(state.active.status, "observed")
        self.assertEqual(len(set(state.active.cells)), 4)
        self.assertTrue(all(0 <= c < 10 and 0 <= r < 26 for c, r in state.active.cells))

    def test_absent_active_requires_observed_spawn_margin(self):
        self.output["layer_type"].zero_()
        self.output["layer_type"][:, :, 0] = 30
        state = decode_partial(self.output, self.prepared, model_id="fixture")
        self.assertEqual(state.active.status, "unknown")
        self.assertEqual(state.ghost.status, "absent")

    def test_saved_checkpoint_drives_public_one_image_api_and_never_accepts(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch)/"artifact"
            record = save_checkpoint(path, TetrisStateNet(ModelConfig.pilot()),
                                     {"arguments": {"max_side": 96}})
            parser = ImageParser.from_checkpoint(path)
            state = parser.parse_image(Image.new("RGB", (160, 192), (10, 20, 30)))
            self.assertEqual(state.model_id, record["weights_sha256"])
            self.assertEqual(state.decision, "needs_review")
            self.assertEqual(VisionState.from_json(state.to_json()), state)
            invisible = parser.parse_image(Image.new("RGBA", (160, 192), (255, 0, 0, 0)))
            self.assertEqual(invisible.decision, "unsupported")
            self.assertIsNone(invisible.board)
