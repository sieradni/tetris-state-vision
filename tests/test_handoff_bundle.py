"""Transport integrity, latest-snapshot selection and non-destructive restore."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "scripts"))
import handoff_bundle as bundle


class HandoffBundleTests(unittest.TestCase):
    def setUp(self):
        self.base = WORKSPACE / "runs/handoff-tests"
        self.base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="test-", dir=self.base)
        self.folder = Path(self.temporary.name).resolve()
        self.assertTrue(self.folder.is_relative_to(self.base.resolve()))
        self.repo = self.folder / "source"
        self.repo.mkdir()
        self.command("git", "init", "-b", "main", str(self.repo))
        (self.repo / "README.md").write_bytes(b"tracked source\n")
        (self.repo / ".gitignore").write_bytes(b"runs/\n.venv/\ndata/\n.handoff/\n")
        self.command("git", "-C", str(self.repo), "add", "README.md", ".gitignore")
        self.command("git", "-C", str(self.repo), "-c", "user.name=HandoffTest", "-c", "user.email=test@example.invalid",
                     "commit", "--no-gpg-sign", "-m", "fixture")
        run = self.repo / "runs/model"
        (run / "checkpoint").mkdir(parents=True)
        (run / "checkpoint/weights.pt").write_bytes(b"synthetic tensor fixture")
        generation = "step-000001-"+"a"*32
        latest = run / "training-snapshots" / generation
        latest.mkdir(parents=True)
        (latest / "manifest.json").write_bytes(b'{"step":1}\n')
        (latest / "tensors.pt").write_bytes(b"latest optimizer fixture")
        old = run / "training-snapshots/step-000000-old"
        old.mkdir()
        (old / "tensors.pt").write_bytes(b"old excluded fixture")
        (run / "latest-training.json").write_text(json.dumps({"generation": generation,
            "manifest_sha256": hashlib.sha256((latest / "manifest.json").read_bytes()).hexdigest()}))
        (run / "z-local.bin").write_bytes(b"curated fixture")
        (self.repo / "data").mkdir()
        (self.repo / "data/private.png").write_bytes(b"private excluded fixture")
        self.archive = self.folder / "handoff.zip"
        with patch.object(bundle, "ROOT", self.repo), patch.object(bundle, "RUNS", ["runs/model"]):
            self.created = bundle.build(self.archive)

    def tearDown(self):
        self.assertTrue(self.folder.resolve().is_relative_to(self.base.resolve()))
        self.temporary.cleanup()

    def command(self, *args):
        return subprocess.check_output(args, stderr=subprocess.STDOUT).decode().strip()

    def clone(self):
        target = self.folder / "clone"
        self.command("git", "clone", "--no-hardlinks", str(self.repo), str(target))
        return target

    def test_source_current_artifacts_and_only_latest_optimizer_are_included(self):
        info = bundle.verify(self.archive)
        names = {item["path"] for item in info["files"]}
        self.assertIn("README.md", names)
        self.assertIn("runs/model/checkpoint/weights.pt", names)
        self.assertFalse(any("step-000000-old" in name or name.startswith(("data/", ".git/", ".venv/")) for name in names))
        self.assertEqual(info["commit"], self.created["commit"])
        self.assertTrue(self.archive.with_suffix(".zip.sha256").is_file())

    def test_restore_writes_only_ignored_artifacts_and_is_idempotent(self):
        target = self.clone()
        before = (target / "README.md").read_bytes()
        with patch.object(bundle, "ROOT", target):
            first = bundle.restore(self.archive)
            second = bundle.restore(self.archive)
        self.assertGreater(first["new_artifact_files"], 0)
        self.assertEqual(second["new_artifact_files"], 0)
        self.assertEqual((target / "README.md").read_bytes(), before)
        self.assertEqual(self.command("git", "-C", str(target), "status", "--porcelain"), "")

    def test_different_existing_run_data_refuses_before_any_other_write(self):
        target = self.clone()
        conflict = target / "runs/model/z-local.bin"
        conflict.parent.mkdir(parents=True)
        conflict.write_bytes(b"other owner data")
        with patch.object(bundle, "ROOT", target), self.assertRaises(ValueError):
            bundle.restore(self.archive)
        self.assertEqual(conflict.read_bytes(), b"other owner data")
        self.assertFalse((target / "runs/model/checkpoint/weights.pt").exists())

    def test_dirty_source_is_never_overwritten(self):
        target = self.clone()
        (target / "README.md").write_bytes(b"contributor edits")
        with patch.object(bundle, "ROOT", target), self.assertRaises(ValueError):
            bundle.restore(self.archive)
        self.assertEqual((target / "README.md").read_bytes(), b"contributor edits")
        self.assertFalse((target / "runs/model").exists())

    def test_tampered_member_fails_integrity(self):
        bad = self.folder / "tampered.zip"
        with zipfile.ZipFile(self.archive) as original, zipfile.ZipFile(bad, "w") as output:
            for item in original.infolist():
                if not item.is_dir():
                    value = original.read(item)
                    output.writestr(item.filename, b"changed" if item.filename == "README.md" else value)
        with self.assertRaises(ValueError):
            bundle.verify(bad)

    def test_traversal_and_duplicate_members_refuse(self):
        with self.assertRaises(ValueError):
            bundle.member_path("../outside")
        with self.assertRaises(ValueError):
            bundle.member_path("runs/../../outside")
        bad = self.folder / "duplicate.zip"
        shutil.copyfile(self.archive, bad)
        with zipfile.ZipFile(bad, "a") as output:
            output.writestr("README.md", b"duplicate")
        with self.assertRaises(ValueError):
            bundle.verify(bad)


if __name__ == "__main__":
    unittest.main()
