from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import atomic_write


class AtomicWriteTests(unittest.TestCase):
    def paths(self, directory):
        source, target = Path(directory)/"new.tmp", Path(directory)/"latest.json"
        source.write_bytes(b"complete new state")
        target.write_bytes(b"complete old state")
        return source, target

    def sharing_error(self):
        error = PermissionError("injected transient Windows sharing denial")
        error.winerror = 5
        return error

    def test_transient_sharing_denial_retries_then_publishes_complete_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = self.paths(directory)
            original = atomic_write.os.replace
            calls = []
            def transient(a, b):
                calls.append((a, b))
                if len(calls) < 3:
                    self.assertEqual(target.read_bytes(), b"complete old state")
                    raise self.sharing_error()
                original(a, b)
            with mock.patch.object(atomic_write.os, "replace", transient), mock.patch.object(atomic_write.time, "sleep") as sleep:
                atomic_write.replace_retry(source, target)
            self.assertEqual(len(calls), 3)
            self.assertEqual(sleep.call_args_list, [mock.call(.01), mock.call(.02)])
            self.assertEqual(target.read_bytes(), b"complete new state")
            self.assertFalse(source.exists())

    def test_persistent_denial_is_finite_and_preserves_old_pointer_and_new_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = self.paths(directory)
            with mock.patch.object(atomic_write.os, "replace", side_effect=self.sharing_error()) as replace, mock.patch.object(atomic_write.time, "sleep") as sleep:
                with self.assertRaises(PermissionError):
                    atomic_write.replace_retry(source, target)
            self.assertEqual(replace.call_count, 6)
            self.assertEqual(sleep.call_count, 5)
            self.assertAlmostEqual(sum(call.args[0] for call in sleep.call_args_list), .15)
            self.assertEqual(target.read_bytes(), b"complete old state")
            self.assertEqual(source.read_bytes(), b"complete new state")

    def test_unrelated_io_errors_are_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = self.paths(directory)
            with mock.patch.object(atomic_write.os, "replace", side_effect=FileNotFoundError("missing source")) as replace, mock.patch.object(atomic_write.time, "sleep") as sleep:
                with self.assertRaises(FileNotFoundError):
                    atomic_write.replace_retry(source, target)
            self.assertEqual(replace.call_count, 1)
            sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
