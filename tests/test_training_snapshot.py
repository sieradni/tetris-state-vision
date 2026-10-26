import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

if any(os.environ.get(name) != "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
    raise RuntimeError("run recovery fixtures through the bounded pre-import CPU policy")
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import training_snapshot as snapshots


class TrainingSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.assertEqual(torch.get_num_threads(), 1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.contract = {"source_sha256": "fixture-source", "seed": 19, "target_steps": 6,
                         "data": "deterministic-index-v1", "rng": "torch_cpu_only"}

    def create(self, seed=19):
        torch.manual_seed(seed)
        model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Dropout(.3), torch.nn.Linear(8, 2))
        return model, torch.optim.AdamW(model.parameters(), lr=.003)

    def train(self, model, optimizer, start, end):
        for step in range(start, end):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            x = torch.tensor([[step/10, .25, -.5]])
            (model(x)-torch.tensor([[.2, -.3]])).square().mean().backward()
            optimizer.step()

    def save(self, model, optimizer, step=3, ledgers=()):
        return snapshots.save_training_snapshot(self.run, model, optimizer, step=step,
                                                contract=self.contract, ledgers=ledgers)

    def restore(self, model, optimizer, contract=None):
        return snapshots.load_training_snapshot(self.run, model, optimizer,
                                                contract=self.contract if contract is None else contract)

    def equal_tree(self, left, right):
        if torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.equal_tree(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.equal_tree(a, b)
        else:
            self.assertEqual(left, right)

    def test_resumed_adam_dropout_matches_continuous_model_optimizer_and_rng(self):
        continuous, optimizer = self.create()
        self.train(continuous, optimizer, 0, 6)
        expected = copy.deepcopy((continuous.state_dict(), optimizer.state_dict(), torch.get_rng_state()))
        interrupted, partial = self.create()
        self.train(interrupted, partial, 0, 3)
        before_save = torch.get_rng_state().clone()
        self.save(interrupted, partial)
        self.assertTrue(torch.equal(before_save, torch.get_rng_state()))
        resumed, recovery = self.create(900)
        state = self.restore(resumed, recovery)
        self.assertEqual(state["step"], 3)
        self.train(resumed, recovery, state["step"], 6)
        self.equal_tree(expected, (resumed.state_dict(), recovery.state_dict(), torch.get_rng_state()))

    def test_ledger_appends_are_ignored_but_committed_prefix_changes_fail(self):
        model, optimizer = self.create()
        self.train(model, optimizer, 0, 3)
        ledger = self.run/"training.jsonl"
        prefix = b'{"step":1}\n{"step":2}\n{"step":3}\n'
        ledger.write_bytes(prefix)
        self.save(model, optimizer, ledgers=(ledger.name,))
        ledger.write_bytes(prefix+b'{"step":4}\n')
        self.assertEqual(self.restore(model, optimizer)["ledger_prefixes"], {ledger.name: prefix})
        for damaged in (prefix[:-1], prefix.replace(b'1', b'9')):
            ledger.write_bytes(damaged)
            with self.assertRaisesRegex(ValueError, "prefix changed/truncated"):
                self.restore(model, optimizer)

    def test_interrupted_publication_preserves_previous_complete_generation(self):
        model, optimizer = self.create()
        self.train(model, optimizer, 0, 3)
        self.save(model, optimizer)
        prior = (self.run/"latest-training.json").read_bytes()
        self.train(model, optimizer, 3, 6)
        write = snapshots._write_json
        def interrupted(path, value):
            if path.name == "latest-training.json":
                raise OSError("injected interruption before pointer publication")
            write(path, value)
        with mock.patch.object(snapshots, "_write_json", interrupted), self.assertRaises(OSError):
            self.save(model, optimizer, step=6)
        self.assertEqual((self.run/"latest-training.json").read_bytes(), prior)
        self.assertEqual(self.restore(model, optimizer)["step"], 3)
        self.assertEqual(len(list((self.run/"training-snapshots").iterdir())), 2)

    def test_changed_recipe_or_parameter_order_is_rejected(self):
        model, optimizer = self.create()
        self.train(model, optimizer, 0, 3)
        self.save(model, optimizer)
        for key, value in (("source_sha256", "other-source"), ("target_steps", 9), ("seed", 20)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identity differs"):
                self.restore(model, optimizer, {**self.contract, key: value})
        wrong = torch.optim.AdamW(reversed(list(model.parameters())), lr=.003)
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.restore(model, wrong)

    def test_tensor_corruption_fails_before_model_mutation(self):
        model, optimizer = self.create()
        self.train(model, optimizer, 0, 3)
        pointer = self.save(model, optimizer)
        artifact = self.run/"training-snapshots"/pointer["generation"]/"tensors.pt"
        data = bytearray(artifact.read_bytes())
        data[-1] ^= 1
        artifact.write_bytes(data)
        recipient, empty = self.create(90)
        before = copy.deepcopy(recipient.state_dict())
        with self.assertRaisesRegex(ValueError, "tensor identity changed"):
            self.restore(recipient, empty)
        self.equal_tree(before, recipient.state_dict())

    def test_manifest_identity_and_pointer_path_cannot_be_substituted(self):
        model, optimizer = self.create()
        self.train(model, optimizer, 0, 3)
        pointer = self.save(model, optimizer)
        manifest = self.run/"training-snapshots"/pointer["generation"]/"manifest.json"
        manifest.write_bytes(manifest.read_bytes()+b" ")
        with self.assertRaisesRegex(ValueError, "manifest identity changed"):
            self.restore(model, optimizer)
        pointer["generation"] = "../other-run"
        (self.run/"latest-training.json").write_text(json.dumps(pointer))
        with self.assertRaisesRegex(ValueError, "pointer/generation"):
            self.restore(model, optimizer)

    def test_snapshot_bounds_and_optimizer_ownership_are_enforced(self):
        model, optimizer = self.create()
        for step in (0, 4097, True):
            with self.subTest(step=step), self.assertRaises(ValueError):
                self.save(model, optimizer, step=step)
        with self.assertRaises(ValueError):
            self.save(model, optimizer, ledgers=("../training.jsonl",))
        with self.assertRaisesRegex(ValueError, "AdamW"):
            self.save(model, torch.optim.SGD(model.parameters(), lr=.1))
        with self.assertRaisesRegex(ValueError, "each named model parameter"):
            self.save(model, torch.optim.AdamW(list(model.parameters())[:1], lr=.003))
        with mock.patch.object(snapshots, "MAX_TENSORS", 1), self.assertRaisesRegex(ValueError, "byte budget"):
            self.save(model, optimizer)
        self.assertFalse((self.run/"latest-training.json").exists())


if __name__ == "__main__":
    unittest.main()
