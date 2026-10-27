import hashlib
import json
from pathlib import Path
import sys
import unittest

from PIL import ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from m2_stream_data import sample_case


class StreamDataTests(unittest.TestCase):
    def test_budget_and_split_refusal(self):
        for group, view, split, seed in ((-1, 0, "training", 1), (4096, 0, "training", 1),
                                         (0, 3, "training", 1), (0, 0, "test", 1),
                                         (0, 0, "training", -1), (True, 0, "training", 1)):
            with self.subTest(group=group, view=view, split=split), self.assertRaises(ValueError):
                sample_case(group, view, split, seed)

    def test_pixels_labels_and_metadata_are_deterministic(self):
        first, second = (sample_case(2, 1, "training", 20) for _ in range(2))
        self.assertEqual(first.frame.image.tobytes(), second.frame.image.tobytes())
        self.assertEqual(first.frame.state, second.frame.state)
        self.assertEqual(first.record, second.record)
        self.assertEqual(first.record["state_sha256"], hashlib.sha256(first.frame.state.to_json().encode()).hexdigest())

    def test_appearances_share_ancestry_and_supported_state(self):
        clean, no_grid = (sample_case(2, view, "training", 20) for view in (0, 1))
        self.assertEqual(clean.record["group_id"], no_grid.record["group_id"])
        self.assertEqual(clean.frame.state, no_grid.frame.state)
        self.assertIsNotNone(ImageChops.difference(clean.frame.image, no_grid.frame.image).getbbox())

    def test_holdout_seed_spaces_are_disjoint(self):
        train, development = (sample_case(0, 0, split, 20) for split in ("training", "development"))
        self.assertNotEqual(train.record["group_id"], development.record["group_id"])
        self.assertNotEqual(train.record["seed"], development.record["seed"])
        self.assertNotEqual(train.record["image_pixels_sha256"], development.record["image_pixels_sha256"])

    def test_intervention_preserves_other_fields_and_only_changes_its_patch(self):
        for group in (0, 1, 2, 5):
            with self.subTest(group=group):
                original = sample_case(group, 0 if group % 2 == 0 else 1, "training", 20)
                changed = sample_case(group, 2, "training", 20)
                c, r = changed.record["intervention"]["cell"]
                self.assertEqual(original.record["group_id"], changed.record["group_id"])
                self.assertEqual(original.frame.state.board.lattice, changed.frame.state.board.lattice)
                delta = [(x, y) for y in range(20) for x in range(10) if original.frame.state.board.rows[y][x] != changed.frame.state.board.rows[y][x]]
                self.assertEqual(delta, [(c, r)])
                for field in ("active", "ghost", "queue", "hold", "decision"):
                    self.assertEqual(getattr(original.frame.state, field), getattr(changed.frame.state, field))
                self.assertTrue(all((c, r) not in layer.cells for layer in (changed.frame.state.active, changed.frame.state.ghost)))
                box = ImageChops.difference(original.frame.image, changed.frame.image).getbbox()
                lattice = original.frame.state.board.lattice
                left, top = lattice.origin[0]+c*lattice.u[0], lattice.origin[1]+(r+1)*lattice.v[1]
                self.assertTrue(left <= box[0] < box[2] <= left+lattice.u[0])
                self.assertTrue(top <= box[1] < box[3] <= top-lattice.v[1])

    def test_varied_source_layouts_remain_visible_at_512_input(self):
        layouts, densities = set(), set()
        for group in range(12):
            case = sample_case(group, 0, "training", 20)
            board = case.frame.state.board
            self.assertEqual(sum(cell.occupied is not None for row in board.rows for cell in row), 200)
            scale = 512/max(case.frame.image.size)
            self.assertGreaterEqual(min(abs(board.lattice.u[0]), abs(board.lattice.v[1]))*scale, 10)
            layouts.add(tuple(board.lattice.origin))
            densities.add(case.record["density"])
        self.assertGreater(len(layouts), 6)
        self.assertEqual(densities, {"empty", "sparse", "dense"})

    def test_square_admission_frame_covers_the_maximum_input_shape(self):
        case = sample_case(0, 0, "training", 20, profile_square=True)
        self.assertEqual(case.frame.image.size, (512, 512))
        self.assertTrue(case.record["profile_square"])
        self.assertEqual(sum(cell.occupied is not None for row in case.frame.state.board.rows for cell in row), 200)
        with self.assertRaises(ValueError):
            sample_case(1, 0, "training", 20, profile_square=True)

    def test_default_m2_sample_preserves_named_historical_record(self):
        path = Path(__file__).resolve().parents[1]/"docs/evidence/m2-stream-learn768-r3/training.jsonl"
        with path.open(encoding="utf-8") as source:
            historical = json.loads(source.readline())
        historical.pop("step")
        actual = sample_case(historical["group_index"], historical["view"], historical["split"], historical["base_seed"])
        self.assertEqual(json.loads(json.dumps(actual.record)), historical)
        self.assertNotIn("final_capture", actual.record)

    def test_m3_final_captures_are_bounded_deterministic_and_distinct_policy(self):
        for group in (0, 1, 2, 5, 11):
            with self.subTest(group=group):
                first, second = (sample_case(group, 0, "training", 20, final_capture=True) for _ in range(2))
                self.assertLessEqual(max(first.frame.image.size), 512)
                self.assertEqual(first.frame.image.tobytes(), second.frame.image.tobytes())
                self.assertEqual(first.record, second.record)
                self.assertTrue(first.record["final_capture"])
                self.assertEqual(first.record["capture_max_side"], 512)
                self.assertEqual(tuple(first.record["appearance"]["output_size"]), first.frame.image.size)
                self.assertTrue(first.record["policy"].startswith("M3-stream-final-capture-v1"))
                self.assertEqual(sum(cell.occupied is not None for row in first.frame.state.board.rows for cell in row), 200)
        with self.assertRaises(ValueError):
            sample_case(0, 0, "training", 20, final_capture=1)

    def test_joint_square_profile_and_counterfactual_keep_fixed_budget(self):
        case = sample_case(0, 0, "training", 20, profile_square=True, final_capture=True)
        self.assertEqual(case.frame.image.size, (512, 512))
        original, changed = (sample_case(2, view, "training", 20, final_capture=True) for view in (0, 2))
        delta = [(c, r) for r in range(20) for c in range(10)
                 if original.frame.state.board.rows[r][c] != changed.frame.state.board.rows[r][c]]
        self.assertEqual(delta, [tuple(changed.record["intervention"]["cell"])])
        for field in ("active", "ghost", "queue", "hold"):
            self.assertEqual(getattr(original.frame.state, field), getattr(changed.frame.state, field))


if __name__ == "__main__":
    unittest.main()
