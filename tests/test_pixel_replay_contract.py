import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pixel_replay_contract import verify_non_type_outputs


def output(value=0.0):
    stage = {"layer_type": torch.tensor([value]), "occupancy": torch.tensor([0.0]),
             "nested": {"layer_type": torch.tensor([0.0])}}
    return {"stages": [stage], "final": stage}


class PixelReplayContractTest(unittest.TestCase):
    def test_only_type_logits_may_change(self):
        verify_non_type_outputs(output(), output(1.0), torch)

    def test_non_type_bytes_and_nested_names_are_checked(self):
        for key in ("occupancy", "nested"):
            a, b = output(), output()
            if key == "occupancy":
                b["final"][key][0] = -0.0
            else:
                b["final"][key]["layer_type"][0] = 1.0
            with self.assertRaises(ValueError):
                verify_non_type_outputs(a, b, torch)

    def test_missing_stage_and_shape_change_rejected(self):
        a, b = output(), output()
        b["stages"] = []
        with self.assertRaises(ValueError):
            verify_non_type_outputs(a, b, torch)
        b = output()
        b["final"]["layer_type"] = torch.zeros(2)
        with self.assertRaises(ValueError):
            verify_non_type_outputs(a, b, torch)


if __name__ == "__main__":
    unittest.main()
