"""No-Torch tests of durable journal and resume preflight behavior."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import accumulation_checkpoint as checkpoint
from accumulation_contract import group_specs
from accumulation_journal import group_entry


def entry(policy, update):
    curriculum = "all-class-fit-v1" if policy == "ordered-eight-v1" else "all-class-interleaved-v1"
    recipes = [{"sample_id": f"all-class-{i:02d}-v{v}", "group_id": f"all-class-{i:02d}",
                "index": i, "view": v, "split": "fit", "curriculum": curriculum, "profile_square": False,
                "image_pixels_sha256": "a"*64, "state_sha256": "b"*64}
               for _, i, v in group_specs(policy, update)]
    return group_entry(policy, update, recipes, mean_loss=1., gradient_norm=2.)


class FakeModel:
    def __init__(self):
        self.items = [object(), object()]

    def parameters(self):
        return iter(self.items)


class FakeOptimizer:
    def __init__(self, model):
        self.param_groups = [{"params": model.items, "lr": .001, "weight_decay": .01,
                              "betas": (.9, .999), "eps": 1e-8, "amsgrad": False, "maximize": False}]
        self.state = {parameter: {"step": 1.} for parameter in model.items}


class CheckpointContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "source"
        self.run.mkdir()
        self.dest = self.root / "dest"
        self.dest.mkdir()
        self.policy = "balanced-eight-v1"
        self.contract = checkpoint.make_contract(self.policy, source_catalogue_sha256="a"*64,
                                                 config={"dimension": 64, "component_capacity": 0})
        self.model = FakeModel()
        self.optimizer = FakeOptimizer(self.model)
        checkpoint.create_journal(self.run)

    def publish_fixture(self, *, step=1, digest_override=None):
        prefix = (self.run / checkpoint.LEDGER).read_bytes()
        folder = self.run / "training-snapshots" / (f"step-{step:06d}-" + "b"*32)
        folder.mkdir(parents=True)
        record = {"step": step, "contract": self.contract, "ledgers": {checkpoint.LEDGER:
                  {"bytes": len(prefix), "sha256": hashlib.sha256(prefix).hexdigest()}}}
        data = json.dumps(record).encode()
        (folder / "manifest.json").write_bytes(data)
        pointer = {"format": "tetris-vision-training-snapshot", "version": 1, "generation": folder.name,
                   "manifest_sha256": digest_override or hashlib.sha256(data).hexdigest()}
        (self.run / "latest-training.json").write_text(json.dumps(pointer))
        return prefix, pointer

    def test_journal_append_is_exclusive_and_boundary_checked(self):
        with self.assertRaises(FileExistsError):
            checkpoint.create_journal(self.run)
        state = checkpoint.append_group(self.run, self.policy, entry(self.policy, 1))
        self.assertEqual((state["images_seen"], state["optimizer_updates"]), (8, 1))
        before = (self.run / checkpoint.LEDGER).read_bytes()
        with self.assertRaises(ValueError):
            checkpoint.append_group(self.run, self.policy, entry(self.policy, 3))
        self.assertEqual((self.run / checkpoint.LEDGER).read_bytes(), before)

    def test_recovery_ignores_uncommitted_valid_suffix(self):
        checkpoint.append_group(self.run, self.policy, entry(self.policy, 1))
        prefix, pointer = self.publish_fixture()
        checkpoint.append_group(self.run, self.policy, entry(self.policy, 2))
        with patch.object(checkpoint, "load_training_snapshot", return_value={"step": 1,
                          "ledger_prefixes": {checkpoint.LEDGER: prefix}, "pointer": pointer}) as load:
            state = checkpoint.restore_boundary(self.run, self.dest, self.model, self.optimizer, self.contract)
        self.assertEqual(state["images_seen"], 8)
        self.assertEqual((self.dest / checkpoint.LEDGER).read_bytes(), prefix)
        self.assertNotEqual((self.run / checkpoint.LEDGER).read_bytes(), prefix)
        load.assert_called_once()

    def test_malformed_or_mismatched_snapshot_refuses_before_model_load(self):
        checkpoint.append_group(self.run, self.policy, entry(self.policy, 1))
        self.publish_fixture(step=2)
        with patch.object(checkpoint, "load_training_snapshot") as load:
            with self.assertRaises(ValueError):
                checkpoint.restore_boundary(self.run, self.dest, self.model, self.optimizer, self.contract)
            load.assert_not_called()
        self.assertFalse((self.dest / checkpoint.LEDGER).exists())

    def test_corrupt_prefix_and_changed_contract_refuse(self):
        checkpoint.append_group(self.run, self.policy, entry(self.policy, 1))
        self.publish_fixture()
        changed = dict(self.contract, seed=42)
        with self.assertRaises(ValueError):
            checkpoint.restore_boundary(self.run, self.dest, self.model, self.optimizer, changed)
        (self.run / checkpoint.LEDGER).write_bytes(b"corrupt\n")
        with patch.object(checkpoint, "load_training_snapshot") as load:
            with self.assertRaises(ValueError):
                checkpoint.restore_boundary(self.run, self.dest, self.model, self.optimizer, self.contract)
            load.assert_not_called()

    def test_snapshot_publication_requires_committed_group_and_optimizer_ownership(self):
        with patch.object(checkpoint, "save_training_snapshot") as save:
            with self.assertRaises(ValueError):
                checkpoint.save_boundary(self.run, self.model, self.optimizer, self.contract)
            checkpoint.append_group(self.run, self.policy, entry(self.policy, 1))
            self.optimizer.param_groups[0]["params"] = self.model.items[:1]
            with self.assertRaises(ValueError):
                checkpoint.save_boundary(self.run, self.model, self.optimizer, self.contract)
            save.assert_not_called()

    def test_snapshot_rejects_wrong_actual_optimizer_step_and_hyperparameters(self):
        checkpoint.append_group(self.run, self.policy, entry(self.policy, 1))
        with patch.object(checkpoint, "save_training_snapshot") as save:
            self.optimizer.state[self.model.items[0]]["step"] = 2
            with self.assertRaisesRegex(ValueError, "step"):
                checkpoint.save_boundary(self.run, self.model, self.optimizer, self.contract)
            self.optimizer.state[self.model.items[0]]["step"] = 1
            self.optimizer.param_groups[0]["lr"] = .004
            with self.assertRaisesRegex(ValueError, "hyperparameters"):
                checkpoint.save_boundary(self.run, self.model, self.optimizer, self.contract)
            save.assert_not_called()

    def test_restore_rejects_wrong_optimizer_before_loading_state(self):
        checkpoint.append_group(self.run, self.policy, entry(self.policy, 1))
        self.publish_fixture()
        self.optimizer.param_groups[0]["weight_decay"] = 0.
        with patch.object(checkpoint, "load_training_snapshot") as load:
            with self.assertRaisesRegex(ValueError, "hyperparameters"):
                checkpoint.restore_boundary(self.run, self.dest, self.model, self.optimizer, self.contract)
            load.assert_not_called()
        self.assertFalse((self.dest / checkpoint.LEDGER).exists())

    def test_snapshot_rejects_pending_group_gradients(self):
        class Parameter:
            def __init__(self):
                self.grad = object()
        parameter = Parameter()
        self.model.items[:] = [parameter]
        self.optimizer.param_groups[0]["params"] = self.model.items
        self.optimizer.state = {parameter: {"step": 1.}}
        checkpoint.append_group(self.run, self.policy, entry(self.policy, 1))
        with patch.object(checkpoint, "save_training_snapshot") as save:
            with self.assertRaisesRegex(ValueError, "clear gradients"):
                checkpoint.save_boundary(self.run, self.model, self.optimizer, self.contract)
            save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
