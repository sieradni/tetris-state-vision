from dataclasses import replace
import unittest

from tetris_vision.adapter import IncompleteState, to_solver_input
from tetris_vision.render import sample_scene, render
from tetris_vision.state import Cell, FallingPiece, Icon, Lattice, Queue, VisionState


class StateContractTests(unittest.TestCase):
    def setUp(self):
        self.state = render(sample_scene(9)).state

    def test_json_round_trip(self):
        self.assertEqual(VisionState.from_json(self.state.to_json()), self.state)

    def test_unknown_never_means_empty(self):
        self.assertIsNone(Cell(None).occupied)
        self.assertNotEqual(Cell(None), Cell(False))
        for args in ((False, "T"), (None, "T"), (1, None)):
            with self.assertRaises(ValueError):
                Cell(*args)
        with self.assertRaises(ValueError):
            Cell(True, visibility="hidden")

    def test_unknown_queue_slot_keeps_index(self):
        queue = Queue((Icon("observed", "T"), Icon("unknown"), Icon("observed", "I")), "cropped", False)
        state = replace(self.state, queue=queue, decision="needs_review")
        restored = VisionState.from_json(state.to_json())
        self.assertEqual(restored.queue.items[2].piece, "I")
        self.assertIsNone(restored.queue.items[1].piece)
        with self.assertRaises(IncompleteState):
            to_solver_input(restored, hold_available=True)

    def test_absent_queue_is_not_empty(self):
        self.assertNotEqual(Queue((), "absent", False), Queue(()))
        with self.assertRaises(ValueError):
            Queue((), "absent", True)

    def test_temporal_hold_availability_is_explicit(self):
        with self.assertRaises(TypeError):
            to_solver_input(self.state)
        with self.assertRaises(IncompleteState):
            to_solver_input(self.state, hold_available=None)

    def test_active_is_consumed_once_and_rows_are_bottom_first(self):
        request = to_solver_input(self.state, hold_available=False)
        expected = self.state.active.piece + "".join(i.piece for i in self.state.queue.items)
        self.assertEqual(request.pieces, expected)
        self.assertEqual(len(request.pieces), len(self.state.queue.items)+1)
        for r, row in enumerate(self.state.board.rows):
            self.assertEqual(request.rows_bottom_first[r], "".join("1" if c.occupied else "0" for c in row))
        self.assertFalse(request.hold_available)

    def test_absent_hold_refuses_solver_conversion(self):
        with self.assertRaises(IncompleteState):
            to_solver_input(replace(self.state, hold=Icon("absent")), hold_available=True)

    def test_unknown_active_does_not_steal_first_preview(self):
        state = replace(self.state, active=FallingPiece("unknown"), decision="needs_review")
        with self.assertRaises(IncompleteState):
            to_solver_input(state, hold_available=True)

    def test_accepted_cannot_contain_unknown(self):
        with self.assertRaises(ValueError):
            replace(self.state, active=FallingPiece("unknown"))
        with self.assertRaises(ValueError):
            replace(self.state, board=replace(self.state.board, lattice=None))

    def test_footprint_must_match_piece(self):
        with self.assertRaises(ValueError):
            FallingPiece("observed", "I", ((0, 0), (1, 0), (0, 1), (1, 1)))
        # Rotation/translation allowed, reflection must exchange J and L.
        FallingPiece("observed", "I", ((3, 2), (3, 3), (3, 4), (3, 5)))
        with self.assertRaises(ValueError):
            FallingPiece("observed", "J", ((0, 0), (1, 0), (2, 0), (2, 1)))

    def test_schema_and_unknown_fields_fail_loudly(self):
        value = self.state.to_dict()
        value["schema_version"] = "99"
        with self.assertRaises(ValueError):
            VisionState.from_dict(value)
        value = self.state.to_dict()
        value["board"]["extra"] = True
        with self.assertRaises(ValueError):
            VisionState.from_dict(value)

    def test_folded_or_nonfinite_geometry_is_rejected(self):
        for lattice in (((0, 0), (20, 0), (0, 20)), ((float("nan"), 0), (20, 0), (0, -20))):
            with self.assertRaises(ValueError):
                Lattice(*lattice)


if __name__ == "__main__":
    unittest.main()
