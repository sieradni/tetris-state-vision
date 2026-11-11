"""Declared trunk identity and the dimension knob for the width lever (M3_TRUNK_WIDTH_LEVER)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]

from tetris_vision.checkpoint import load_checkpoint, save_checkpoint
from tetris_vision.model import ModelConfig, TetrisStateNet, architecture_name


class ArchitectureIdentity(unittest.TestCase):
    def test_published_and_pilot_shapes_keep_their_historical_names(self):
        self.assertEqual(architecture_name(ModelConfig()), "TetrisStateNet-M2")
        self.assertEqual(architecture_name(ModelConfig(component_capacity=64)), "TetrisStateNet-M3")
        self.assertEqual(architecture_name(ModelConfig.pilot()), "TetrisStateNet-M2")

    def test_the_declared_wider_shape_is_its_own_identity(self):
        self.assertEqual(architecture_name(ModelConfig(component_capacity=64, dimension=240)),
                         "TetrisStateNet-M3-W48x96x192x256-D240")
        self.assertEqual(architecture_name(ModelConfig(component_capacity=64, widths=(64, 128, 224, 256))),
                         "TetrisStateNet-M3-W64x128x224x256-D192")

    def test_a_configuration_that_cannot_be_named_is_refused(self):
        with self.assertRaises(ValueError):
            ModelConfig(dimension=190)  # not a multiple of 4
        with self.assertRaises(ValueError):
            ModelConfig(dimension=200)  # 200 % heads(6) != 0
        with self.assertRaises(ValueError):
            ModelConfig(widths=(48, 96, 192, 264))  # above the 256-channel cap


class DeclaredKnob(unittest.TestCase):
    def test_trainer_defaults_to_the_published_dimension(self):
        from train_m2_stream import arguments
        parsed = arguments(["--output", "runs/x", "--model", "accuracy", "--steps", "2",
                            "--profile-square"])
        self.assertEqual(parsed.trunk_dimension, 192)

    def test_trainer_accepts_the_declared_wider_dimension(self):
        from train_m2_stream import arguments
        parsed = arguments(["--output", "runs/x", "--model", "accuracy", "--steps", "2",
                            "--profile-square", "--trunk-dimension", "240"])
        self.assertEqual(parsed.trunk_dimension, 240)

    def test_the_width_family_is_owned_and_nothing_else_is(self):
        from record_stream_run import require_owned_run
        self.assertEqual(require_owned_run("m3-trunkwidth-d240-r1"), "m3-trunkwidth-d240-r1")
        for name in ("trunkwidth-d240-r1", "m3-trunkwidth-", "m3-trunkwidth-D240"):
            with self.assertRaises(ValueError):
                require_owned_run(name)


class CheckpointIdentity(unittest.TestCase):
    def test_a_widened_checkpoint_binds_and_keeps_its_own_name(self):
        config = ModelConfig(widths=(16, 32, 64, 96), dimension=80, heads=4, blocks=2,
                             component_capacity=8)
        model = TetrisStateNet(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/"checkpoint"
            record = save_checkpoint(checkpoint, model, {})
            self.assertEqual(record["architecture"], "TetrisStateNet-M3-W16x32x64x96-D80")
            restored, manifest = load_checkpoint(checkpoint)
            self.assertEqual(manifest["architecture"], record["architecture"])
            self.assertEqual(restored.config.dimension, 80)
            manifest_path = checkpoint/"manifest.json"
            tampered = json.loads(manifest_path.read_text())
            tampered["architecture"] = "TetrisStateNet-M3"
            manifest_path.write_text(json.dumps(tampered))
            with self.assertRaises(ValueError):
                load_checkpoint(checkpoint)
