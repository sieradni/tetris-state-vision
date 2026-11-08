import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

if any(os.environ.get(name) != "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
    raise RuntimeError("bounded pre-import CPU policy required")
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from accumulation_contract import GradientGroup, apply_group, committed_boundary, group_specs, update_count
from training_snapshot import load_training_snapshot, save_training_snapshot


class AccumulationContractTest(unittest.TestCase):
    def test_finite_grouping_and_equal_image_exposure(self):
        online = [group_specs("online-interleaved-v1", i)[0] for i in range(1, 1025)]
        balanced = [row for i in range(1, 129) for row in group_specs("balanced-eight-v1", i)]
        ordered = [row for i in range(1, 129) for row in group_specs("ordered-eight-v1", i)]
        self.assertEqual(online, balanced)
        self.assertEqual([row[0] for row in online], list(range(1, 1025)))
        for offset in range(0, 1024, 32):
            expected = {(index, view) for index in range(16) for view in range(2)}
            self.assertEqual({row[1:] for row in balanced[offset:offset+32]}, expected)
            self.assertEqual({row[1:] for row in ordered[offset:offset+32]}, expected)
        for update in range(1, 129):
            self.assertEqual({index//2 for _, index, _ in group_specs("balanced-eight-v1", update)}, set(range(8)))
            self.assertEqual(len({index//2 for _, index, _ in group_specs("ordered-eight-v1", update)}), 2)

    def test_class_mapping_matches_curated_public_observations(self):
        evidence = ROOT / "docs/evidence/type-forgetting-control-r1"
        scope = json.loads((evidence / "scope.json").read_text())
        path = evidence / "result.json"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), scope["artifact_sha256"]["result.json"])
        point = json.loads(path.read_text())["observations"][0]
        public = {case["sample_id"]: case["roles"] for case in point["cases"]}
        for update in range(1, 5):
            for _, index, view in group_specs("balanced-eight-v1", update):
                roles = public[f"all-class-{index:02d}-v{view}"]
                for role in ("active", "ghost"):
                    self.assertTrue(roles[role]["target_known"])
                    expected = 1 + index//2 if index < 14 else 0
                    self.assertEqual(roles[role]["target_type"], expected)

    def test_boundaries_and_invalid_policies_refuse(self):
        self.assertEqual(update_count("balanced-eight-v1"), 128)
        self.assertEqual(committed_boundary("balanced-eight-v1", 64), 8)
        for value in (1, 7, 1025, -8, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                committed_boundary("balanced-eight-v1", value)
        for policy, update in (("wrong", 1), ("ordered-eight-v1", 0), ("balanced-eight-v1", 129), ("online-interleaved-v1", True)):
            with self.assertRaises(ValueError):
                group_specs(policy, update)

    def test_streamed_gradients_equal_full_batch_mean_before_single_clip(self):
        torch.manual_seed(7)
        micro = torch.nn.Linear(3, 2, dtype=torch.float64)
        batch = copy.deepcopy(micro)
        x = torch.arange(24, dtype=torch.float64).reshape(8, 3) / 17
        y = torch.arange(16, dtype=torch.float64).reshape(8, 2) / 11
        group = GradientGroup(8)
        for i in range(8):
            group.backward((micro(x[i:i+1])-y[i:i+1]).square().mean())
        group.finish()
        (batch(x)-y).square().mean().backward()
        for actual, reference in zip(micro.parameters(), batch.parameters()):
            self.assertTrue(torch.allclose(actual.grad, reference.grad, atol=1e-12, rtol=1e-12))
        one = torch.nn.utils.clip_grad_norm_(micro.parameters(), 1.0)
        other = torch.nn.utils.clip_grad_norm_(batch.parameters(), 1.0)
        self.assertTrue(torch.allclose(one, other, atol=1e-12, rtol=1e-12))
        for actual, reference in zip(micro.parameters(), batch.parameters()):
            self.assertTrue(torch.allclose(actual.grad, reference.grad, atol=1e-12, rtol=1e-12))
        with self.assertRaises(ValueError):
            group.backward(micro(x[:1]).sum())
        with self.assertRaises(ValueError):
            GradientGroup(8).finish()

    def test_interrupted_partial_group_is_replayed_from_committed_boundary(self):
        torch.manual_seed(11)
        original = torch.nn.Linear(3, 2, dtype=torch.float64)
        resumed = copy.deepcopy(original)
        direct_optimizer = torch.optim.AdamW(original.parameters(), lr=.001)
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=.001)
        x = torch.arange(48, dtype=torch.float64).reshape(16, 3) / 19
        y = torch.arange(32, dtype=torch.float64).reshape(16, 2) / 23

        def microsteps(model, optimizer, first, count):
            optimizer.zero_grad(set_to_none=True)
            group = GradientGroup(8)
            for i in range(first, first+count):
                group.backward((model(x[i:i+1])-y[i:i+1]).square().mean())
            return group

        for model, optimizer in ((original, direct_optimizer), (resumed, resumed_optimizer)):
            group = microsteps(model, optimizer, 0, 8)
            group.finish()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            contract = {"policy": "balanced-eight-v1", "committed_images": 8}
            save_training_snapshot(directory, resumed, resumed_optimizer, step=1, contract=contract)
            partial = microsteps(resumed, resumed_optimizer, 8, 3)
            with self.assertRaises(ValueError):
                partial.finish()
            load_training_snapshot(directory, resumed, resumed_optimizer, contract=contract)
            group = microsteps(resumed, resumed_optimizer, 8, 8)
            group.finish()
            torch.nn.utils.clip_grad_norm_(resumed.parameters(), 1.0)
            resumed_optimizer.step()
        group = microsteps(original, direct_optimizer, 8, 8)
        group.finish()
        torch.nn.utils.clip_grad_norm_(original.parameters(), 1.0)
        direct_optimizer.step()
        for expected, actual in zip(original.parameters(), resumed.parameters()):
            self.assertTrue(torch.equal(expected, actual))
        for expected, actual in zip(direct_optimizer.state.values(), resumed_optimizer.state.values()):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.assertTrue(torch.equal(expected[key], actual[key]))

    def test_failed_microstep_never_updates_model_or_optimizer(self):
        torch.manual_seed(21)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        before = copy.deepcopy(model.state_dict())

        def interrupted(index):
            if index == 3:
                raise RuntimeError("interrupted")
            return model(torch.ones(1, 3)).square().mean()

        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            apply_group(model, optimizer, tuple(range(8)), interrupted)
        self.assertEqual(len(optimizer.state), 0)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

        with self.assertRaisesRegex(ValueError, "finite scalar"):
            apply_group(model, optimizer, tuple(range(8)), lambda _: torch.tensor(float("nan")))
        self.assertEqual(len(optimizer.state), 0)

    def test_complete_group_steps_once_after_mean_and_clip(self):
        torch.manual_seed(23)
        model = torch.nn.Linear(3, 2, dtype=torch.float64)
        reference = copy.deepcopy(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        baseline = torch.optim.AdamW(reference.parameters(), lr=.001)
        x = torch.arange(24, dtype=torch.float64).reshape(8, 3) / 13
        y = torch.arange(16, dtype=torch.float64).reshape(8, 2) / 17
        result = apply_group(model, optimizer, tuple(range(8)),
                             lambda i: (model(x[i:i+1])-y[i:i+1]).square().mean())
        (reference(x)-y).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
        baseline.step()
        self.assertEqual(result["images"], 8)
        for a, b in zip(model.parameters(), reference.parameters()):
            self.assertTrue(torch.allclose(a, b, atol=1e-12, rtol=1e-12))
            self.assertEqual(int(optimizer.state[a]["step"]), 1)


if __name__ == "__main__":
    unittest.main()
