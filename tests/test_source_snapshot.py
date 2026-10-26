import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from source_snapshot import SCRIPTS, capture_source, verify_source


class SourceSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)/"repo"
        (self.root/"src/tetris_vision/nested").mkdir(parents=True)
        (self.root/"scripts").mkdir()
        for relative in ("src/tetris_vision/__init__.py", "src/tetris_vision/nested/contract.py",
                         "pyproject.toml", "uv.lock", *("scripts/"+name for name in SCRIPTS)):
            (self.root/relative).write_text("VALUE = 17\n", encoding="utf-8")
        self.output = Path(self.temporary.name)/"archive"

    def test_parent_edits_do_not_change_archived_imports_or_nested_identity(self):
        record = capture_source(self.root, self.output)
        (self.root/"src/tetris_vision/nested/contract.py").write_text("VALUE = 99\n")
        self.assertEqual(verify_source(self.output, record["catalogue_sha256"]), record)
        spec = importlib.util.spec_from_file_location("snapshot_fixture", self.output/"tetris_vision/nested/contract.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.VALUE, 17)
        with self.assertRaises(FileExistsError):
            capture_source(self.root, self.output)

    def test_mutated_and_extra_importable_source_are_rejected(self):
        record = capture_source(self.root, self.output)
        extra = self.output/"tetris_vision/new.py"
        extra.write_text("VALUE = 1\n")
        with self.assertRaisesRegex(ValueError, "unlisted"):
            verify_source(self.output, record["catalogue_sha256"])
        extra.unlink()  # Test-owned temporary fixture only.
        (self.output/"tetris_vision/nested/contract.py").write_text("VALUE = 99\n")
        with self.assertRaisesRegex(ValueError, "identity changed"):
            verify_source(self.output, record["catalogue_sha256"])

    def test_byte_budget_fails_before_archive_creation(self):
        (self.root/"src/tetris_vision/oversized.py").write_bytes(b" "*(1024**2+1))
        with self.assertRaisesRegex(ValueError, "budget"):
            capture_source(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_archived_tree_can_be_captured_again_for_replay(self):
        first = capture_source(self.root, self.output)
        replay = Path(self.temporary.name)/"replay"
        self.assertEqual(capture_source(self.output, replay), first)
        self.assertEqual(verify_source(replay, first["catalogue_sha256"]), first)


if __name__ == "__main__":
    unittest.main()
