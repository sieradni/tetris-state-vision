"""Regression: normalized current manifests versus raw evidence copies (V-023)."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from record_controller_recovery import verify_saved_copy


class EvidenceCopyContract(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.source = Path(self.folder.name)/"source.json"
        self.saved = Path(self.folder.name)/"saved.json"
        self.source.write_bytes(b'{"step":3,"tensors_sha256":"abc"}\n')

    def test_raw_copy_requires_identical_bytes(self):
        self.saved.write_bytes(self.source.read_bytes())
        verify_saved_copy(self.saved, self.source)

    def test_raw_parent_manifest_reformat_is_rejected(self):
        self.saved.write_bytes(b'{\n  "step": 3,\n  "tensors_sha256": "abc"\n}\n')
        with self.assertRaisesRegex(ValueError, "curated content differs"):
            verify_saved_copy(self.saved, self.source)

    def test_normalized_current_manifest_keeps_every_value(self):
        self.saved.write_bytes(b'{\r\n  "tensors_sha256": "abc",\r\n  "step": 3\r\n}\r\n')
        verify_saved_copy(self.saved, self.source, normalized=True)

    def test_normalization_does_not_accept_changed_state(self):
        self.saved.write_bytes(b'{"step":4,"tensors_sha256":"abc"}\n')
        with self.assertRaisesRegex(ValueError, "curated content differs"):
            verify_saved_copy(self.saved, self.source, normalized=True)


if __name__ == "__main__":
    unittest.main()
