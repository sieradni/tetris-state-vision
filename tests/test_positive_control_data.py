from collections import Counter
import hashlib
import json
import struct
import sys
import tempfile
from pathlib import Path
import unittest

from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from m2_positive_control_data import LAYOUT, PIECES, generate, sample_control


class PositiveControlDataTests(unittest.TestCase):
    def test_public_labels_balance_all_types_and_absence_with_known_locked_cells(self):
        counts = {role: Counter() for role in ("active", "ghost")}
        boards = Counter()
        for index in range(16):
            for view in range(2):
                case = sample_control(index, view)
                self.assertEqual(case.frame.image.size, (512, 512))
                cells = [cell for row in case.frame.state.board.rows for cell in row]
                self.assertEqual(sum(cell.occupied is not None for cell in cells), 200)
                boards[any(cell.occupied for cell in cells)] += 1
                for role in counts:
                    public = getattr(case.frame.state, role)
                    self.assertIn(public.status, ("observed", "absent"))
                    counts[role][public.piece if public.status == "observed" else "absent"] += 1
        for value in counts.values():
            self.assertEqual(value, {**{piece: 4 for piece in PIECES}, "absent": 4})
        self.assertEqual(boards, {False: 16, True: 16})

    def test_each_piece_has_two_supported_positions_and_preserves_locked_stack(self):
        for index in range(0, 14, 2):
            cases = [sample_control(index + offset, 0) for offset in (0, 1)]
            for role in ("active", "ghost"):
                footprints = [set(getattr(case.frame.state, role).cells) for case in cases]
                self.assertNotEqual(*footprints)
                self.assertTrue(all(0 <= c < 10 and 0 <= r < 20 for footprint in footprints for c, r in footprint))
            occupied = {(c, r) for r, row in enumerate(cases[0].frame.state.board.rows) for c, cell in enumerate(row) if cell.occupied}
            self.assertEqual(occupied, {(c, 0) for c in range(8)})
            self.assertFalse(occupied & set(cases[0].frame.state.active.cells))
            self.assertFalse(occupied & set(cases[0].frame.state.ghost.cells))

    def test_two_styles_have_different_pixels_same_supported_state_and_exact_replay(self):
        for index in (0, 3, 8, 13, 14, 15):
            clean, no_grid, replay = sample_control(index, 0), sample_control(index, 1), sample_control(index, 0)
            self.assertEqual(clean.frame.state, no_grid.frame.state)
            self.assertIsNotNone(ImageChops.difference(clean.frame.image, no_grid.frame.image).getbbox())
            self.assertEqual(clean.frame.image.tobytes(), replay.frame.image.tobytes())
            self.assertEqual(clean.record, replay.record)
            self.assertEqual(hashlib.sha256(clean.frame.state.to_json().encode()).hexdigest(), clean.record["state_sha256"])

    def test_known_absence_covers_full_spawn_domain_and_invalid_indices_refuse(self):
        self.assertTrue(LAYOUT.contains(LAYOUT.spawn_box))
        for index in (14, 15):
            for view in (0, 1):
                state = sample_control(index, view).frame.state
                self.assertEqual((state.active.status, state.ghost.status), ("absent", "absent"))
        for index, view in ((-1, 0), (16, 0), (True, 0), (0, 2), (0, False), (1.0, 0)):
            with self.assertRaises(ValueError):
                sample_control(index, view)

    def test_written_bytes_match_manifest_and_state_identities_and_refuse_reuse(self):
        root = Path(__file__).resolve().parents[1] / "runs"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="control-fixture-test-", dir=root) as temporary:
            self.assertTrue(Path(temporary).resolve().is_relative_to(root.resolve()))
            output = Path(temporary) / "fixtures"
            summary = generate(output)
            manifest = (output / "manifest.jsonl").read_bytes()
            self.assertEqual(hashlib.sha256(manifest).hexdigest(), summary["manifest_sha256"])
            records = [json.loads(line) for line in manifest.splitlines()]
            self.assertEqual(len(records), 32)
            for record in records:
                self.assertEqual(hashlib.sha256((output / record["state"]).read_bytes()).hexdigest(), record["state_sha256"])
                self.assertEqual(hashlib.sha256((output / record["image"]).read_bytes()).hexdigest(), record["image_sha256"])
                with Image.open(output / record["image"]) as image:
                    data = struct.pack("<II", *image.size) + image.mode.encode() + image.tobytes()
                    self.assertEqual(hashlib.sha256(data).hexdigest(), record["image_pixels_sha256"])
            with self.assertRaises(FileExistsError):
                generate(output)
            self.assertEqual((output / "manifest.jsonl").read_bytes(), manifest)


if __name__ == "__main__":
    unittest.main()
