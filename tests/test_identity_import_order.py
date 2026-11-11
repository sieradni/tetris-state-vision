"""Architecture identity must not depend on what `sys.path` holds (V-090).

Replay instruments put a run's frozen archive ahead of the working tree, and an archive-era `model.py` has no
derived-name helper. The shadow scenario therefore runs in a fresh interpreter with the archive-era package
first on the path: that reproduces the real race without purging modules out from under the rest of the suite.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]

from record_stream_run import checkpoint_architecture

CONFIG = {"widths": [48, 96, 192, 256], "dimension": 192, "heads": 6, "blocks": 6, "refinements": 2,
          "visible_rows": 20, "spawn_rows": 6, "columns": 10, "max_batch": 4, "max_pixels": 1048576,
          "component_capacity": 64}

PROBE = """
import importlib, json, pathlib, sys
root = pathlib.Path(__ROOT__)
sys.path[:0] = [__SHADOW__, str(root/"scripts"), str(root/"src")]
try:
    from tetris_vision.model import architecture_name
    shadow_import_fails = False
except ImportError:
    shadow_import_fails = True
from record_stream_run import checkpoint_architecture
config = __CONFIG__
summary = {"config": dict(config), "component_capacity": 64}
identity = checkpoint_architecture(summary, {"architecture": "TetrisStateNet-M3", "config": dict(config)})
try:
    checkpoint_architecture(summary, {"architecture": "TetrisStateNet-M2", "config": dict(config)})
    mismatch_rejected = False
except ValueError:
    mismatch_rejected = True
widened = dict(config, dimension=240)
widened_name = checkpoint_architecture(
    {"config": widened, "component_capacity": 64},
    {"architecture": "TetrisStateNet-M3-W48x96x192x256-D240", "config": widened})
rule_source = importlib.import_module("tetris_vision_identity.model").__file__
print(json.dumps({"shadow_import_fails": shadow_import_fails, "identity": identity,
                  "mismatch_rejected": mismatch_rejected, "widened": widened_name,
                  "rule_source": rule_source}))
"""


class IdentityImportOrder(unittest.TestCase):
    def test_identity_survives_an_archive_era_model_ahead_on_the_path(self):
        with tempfile.TemporaryDirectory() as folder:
            shadow = Path(folder)/"tetris_vision"
            shadow.mkdir()
            (shadow/"__init__.py").write_text("", encoding="utf-8")
            (shadow/"model.py").write_text(
                '"""Archive-era model: no derived-name helper exists here."""\n'
                "class ModelConfig:\n"
                "    pass\n", encoding="utf-8")
            probe = (PROBE.replace("__ROOT__", repr(str(ROOT)))
                     .replace("__SHADOW__", repr(folder))
                     .replace("__CONFIG__", repr(CONFIG)))
            done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            payload = json.loads(done.stdout.strip().splitlines()[-1])
        self.assertTrue(payload["shadow_import_fails"],
                        "the shadow must be archive-era: its model module has no architecture_name")
        self.assertEqual(payload["identity"], "TetrisStateNet-M3")
        self.assertTrue(payload["mismatch_rejected"])
        self.assertEqual(payload["widened"], "TetrisStateNet-M3-W48x96x192x256-D240")
        self.assertTrue(Path(payload["rule_source"]).resolve().is_relative_to(ROOT/"src"))

    def test_a_widened_configuration_is_named_by_the_working_tree_rule(self):
        config = dict(CONFIG, dimension=240)
        summary = {"config": config, "component_capacity": 64}
        manifest = {"architecture": "TetrisStateNet-M3-W48x96x192x256-D240", "config": config}
        self.assertEqual(checkpoint_architecture(summary, manifest),
                         "TetrisStateNet-M3-W48x96x192x256-D240")
        with self.assertRaises(ValueError):
            checkpoint_architecture(summary, {"architecture": "TetrisStateNet-M3", "config": config})

    def test_the_identity_rule_is_the_working_tree_implementation(self):
        summary = {"config": dict(CONFIG), "component_capacity": 64}
        manifest = {"architecture": "TetrisStateNet-M3", "config": dict(CONFIG)}
        self.assertEqual(checkpoint_architecture(summary, manifest), "TetrisStateNet-M3")
        module = sys.modules.get("tetris_vision_identity.model")
        self.assertIsNotNone(module)
        self.assertTrue(Path(module.__file__).resolve().is_relative_to(ROOT/"src"))


if __name__ == "__main__":
    unittest.main()
