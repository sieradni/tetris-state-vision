import contextlib
import copy
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

if any(os.environ.get(name) != "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
    raise RuntimeError("bounded pre-import CPU policy required")
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_m2_stream import arguments, make_optimizer
from training_snapshot import save_training_snapshot, load_training_snapshot


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.body = torch.nn.Linear(3, 3)
        self.layer_type = torch.nn.Linear(3, 2)

    def forward(self, x):
        return self.layer_type(self.body(x))


class TypeHeadOptimizerTest(unittest.TestCase):
    def test_default_matches_original_optimizer_update(self):
        torch.manual_seed(9)
        a = TinyModel()
        b = copy.deepcopy(a)
        one = make_optimizer(a, .001, 1)
        reference = torch.optim.AdamW(b.parameters(), lr=.001, weight_decay=.01)
        for model, optimizer in ((a, one), (b, reference)):
            model(torch.ones(1, 3)).square().sum().backward()
            optimizer.step()
        self.assertEqual(len(one.param_groups), 1)
        for x, y in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(x, y))

    def test_only_type_head_gets_candidate_rate(self):
        model = TinyModel()
        optimizer = make_optimizer(model, .001, 4)
        names = {id(p): name for name, p in model.named_parameters()}
        self.assertEqual([{names[id(p)] for p in group["params"]} for group in optimizer.param_groups],
                         [{"body.weight", "body.bias"}, {"layer_type.weight", "layer_type.bias"}])
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [.001, .004])
        self.assertEqual([group["weight_decay"] for group in optimizer.param_groups], [.01, .01])
        with self.assertRaises(ValueError):
            make_optimizer(model, .001, 2)

    def test_snapshot_restores_groups_and_rejects_changed_factor(self):
        model = TinyModel()
        optimizer = make_optimizer(model, .001, 4)
        model(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            contract = {"type_head_rate_factor": 4}
            save_training_snapshot(directory, model, optimizer, step=1, contract=contract)
            restored = TinyModel()
            other = make_optimizer(restored, .001, 4)
            load_training_snapshot(directory, restored, other, contract=contract)
            self.assertEqual([group["lr"] for group in other.param_groups], [.001, .004])
            for x, y in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(x, y))
            with self.assertRaises(ValueError):
                load_training_snapshot(directory, restored, other, contract={"type_head_rate_factor": 1})

    def test_candidate_scope_is_explicit(self):
        base = ["--output", "unused", "--model", "pilot", "--steps", "2", "--profile-square"]
        self.assertEqual(arguments(base).type_head_rate_factor, 1)
        self.assertEqual(arguments(base + ["--curriculum", "all-class-fit-v1", "--type-head-rate-factor", "4"]).type_head_rate_factor, 4)
        for extra in (["--type-head-rate-factor", "4"],
                      ["--curriculum", "all-class-fit-v1", "--type-head-rate-factor", "2"],
                      ["--curriculum", "all-class-fit-v1", "--type-head-rate-factor", "4", "--learning-rate", ".002"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(base + extra)
