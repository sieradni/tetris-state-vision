import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from tetris_vision.cli import main
from tetris_vision.dataset import generate, split_group


class DatasetTests(unittest.TestCase):
    def test_counterfactual_groups_share_split_and_have_separate_latent_records(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch) / "demo"
            result = generate(root, count=1, seed=7)
            records = [json.loads(line) for line in (root/"manifest.jsonl").read_text().splitlines()]
            self.assertEqual(result["images"], 5)
            self.assertEqual(len({r["group_id"] for r in records}), 1)
            self.assertEqual(len({r["split"] for r in records}), 1)
            self.assertEqual(len({r["image_sha256"] for r in records}), 5)
            for record in records:
                state = json.loads((root/record["state"]).read_text())
                self.assertNotIn("latent", state)
                self.assertNotIn("seed", state)
                self.assertTrue((root/record["latent"]).exists())
            with self.assertRaises(FileExistsError):
                generate(root, count=1)

    def test_generation_budget_is_enforced_before_writing(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)/"never"
            for count in (0, -1, 65):
                with self.assertRaises(ValueError):
                    generate(root, count=count)
                self.assertFalse(root.exists())

    def test_split_is_stable(self):
        self.assertEqual(split_group("same-state"), split_group("same-state"))
        self.assertTrue({split_group(f"group-{i}") for i in range(1000)} == {"train", "development", "test"})

    def test_cli_validate_and_evaluate_missing_predictions(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)/"demo"
            generate(root, count=1, seed=7)
            records = [json.loads(line) for line in (root/"manifest.jsonl").read_text().splitlines()]
            predictions = root/"predictions.jsonl"
            predictions.write_text("")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["validate", str(root/records[0]["state"])]), 0)
            self.assertTrue(json.loads(output.getvalue())["valid"])
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["evaluate", "--manifest", str(root/"manifest.jsonl"),
                                       "--predictions", str(predictions)]), 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["readable"], 4)
            self.assertEqual(report["abstained_readable"], 4)
            self.assertEqual(report["exact_rate"], 0)

    def test_cli_refuses_modified_ground_truth(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)/"demo"
            generate(root, count=1, seed=7)
            record = json.loads((root/"manifest.jsonl").read_text().splitlines()[0])
            state_path = root/record["state"]
            state_path.write_text(state_path.read_text()+"\n")
            predictions = root/"predictions.jsonl"
            predictions.write_text("")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                main(["evaluate", "--manifest", str(root/"manifest.jsonl"), "--predictions", str(predictions)])


if __name__ == "__main__":
    unittest.main()
