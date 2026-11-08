"""Capture-side representation plumbing; no training or accuracy claim."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from m2_curriculum import replay_case
from m2_stream_data import sample_case


class CaptureSideTests(unittest.TestCase):
    def test_capture_side_is_recorded_and_changes_only_the_model_frame(self):
        coarse = sample_case(0, 0, "training", 20261026, final_capture=True)
        finer = sample_case(0, 0, "training", 20261026, final_capture=True, max_side=640)
        self.assertEqual(coarse.record["capture_max_side"], 512)
        self.assertEqual(finer.record["capture_max_side"], 640)
        self.assertEqual(max(coarse.frame.image.size), 512)
        self.assertEqual(max(finer.frame.image.size), 640)
        self.assertNotEqual(coarse.record["image_pixels_sha256"], finer.record["image_pixels_sha256"])
        self.assertEqual(coarse.record["sample_id"], finer.record["sample_id"])
        self.assertEqual(coarse.record["group_id"], finer.record["group_id"])
        self.assertEqual(coarse.record["layout"], finer.record["layout"])

    def test_recorded_capture_side_replays_byte_identically_and_refuses_another_side(self):
        for side in (512, 640):
            case = sample_case(3, 1, "training", 20261026, final_capture=True, max_side=side)
            record = json.loads(json.dumps(case.record))
            replayed = replay_case(record, split="training", component_capacity=64, max_side=side)
            self.assertEqual(json.loads(json.dumps(replayed.record)), record)
        with self.assertRaises(ValueError):
            replay_case(record, split="training", component_capacity=64, max_side=512)

    def test_capture_side_is_bounded(self):
        for side in (0, 32, 63, 1537, 2048):
            with self.assertRaises(ValueError):
                sample_case(0, 0, "training", 20261026, final_capture=True, max_side=side)


if __name__ == "__main__":
    unittest.main()
