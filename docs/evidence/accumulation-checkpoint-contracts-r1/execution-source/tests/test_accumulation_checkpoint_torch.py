"""Toy neural integration of durable accumulation boundaries; not training admission."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest

if any(os.environ.get(name) != "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
    raise RuntimeError("bounded pre-import CPU policy required")
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from accumulation_checkpoint import append_group, create_journal, make_contract, restore_boundary, save_boundary
from accumulation_contract import group_specs
from accumulation_journal import execute_group
from train_m2_stream import equal_tree


def recipes(policy, update):
    return [{"sample_id": f"all-class-{index:02d}-v{view}", "group_id": f"all-class-{index:02d}",
             "index": index, "view": view, "split": "fit", "curriculum": "all-class-interleaved-v1",
             "profile_square": False, "image_pixels_sha256": "a"*64, "state_sha256": "b"*64}
            for _, index, view in group_specs(policy, update)]


class AccumulationCheckpointTorchTest(unittest.TestCase):
    def test_actual_adam_dropout_recovery_discards_uncommitted_group_and_partial_gradients(self):
        torch.manual_seed(61)
        model = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.Dropout(.25), torch.nn.Linear(6, 2))
        restored = copy.deepcopy(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001, weight_decay=.01)
        policy = "balanced-eight-v1"
        contract = make_contract(policy, source_catalogue_sha256="c"*64,
                                 config={"dimension": 3, "component_capacity": 0})

        def loss_for(network):
            def loss(recipe):
                x = torch.tensor([[recipe["index"]/16, recipe["view"]/2, .25]])
                return (network(x)-torch.tensor([[.2, -.1]])).square().mean()
            return loss

        with tempfile.TemporaryDirectory() as directory:
            source, dest = Path(directory)/"source", Path(directory)/"resumed"
            source.mkdir()
            dest.mkdir()
            create_journal(source)
            append_group(source, policy, execute_group(model, optimizer, policy, 1, recipes(policy, 1), loss_for(model)))
            optimizer.zero_grad(set_to_none=True)
            save_boundary(source, model, optimizer, contract)
            for update in (2, 3):
                append_group(source, policy, execute_group(model, optimizer, policy, update, recipes(policy, update), loss_for(model)))
            final_rng = torch.get_rng_state().clone()
            final_optimizer = copy.deepcopy(optimizer.state_dict())
            final_model = copy.deepcopy(model.state_dict())

            count = 0
            def interrupted(recipe):
                nonlocal count
                count += 1
                if count == 4:
                    raise RuntimeError("forced inside group")
                return loss_for(restored)(recipe)
            with self.assertRaisesRegex(RuntimeError, "forced inside group"):
                execute_group(restored, restored_optimizer, policy, 2, recipes(policy, 2), interrupted)
            self.assertFalse((dest / "groups.jsonl").exists())
            state = restore_boundary(source, dest, restored, restored_optimizer, contract)
            self.assertEqual((state["images_seen"], state["optimizer_updates"]), (8, 1))
            for update in (2, 3):
                append_group(dest, policy, execute_group(restored, restored_optimizer, policy, update, recipes(policy, update), loss_for(restored)))
            self.assertTrue(equal_tree(final_model, restored.state_dict()))
            self.assertTrue(equal_tree(final_optimizer, restored_optimizer.state_dict()))
            self.assertTrue(torch.equal(final_rng, torch.get_rng_state()))
            self.assertEqual((source / "groups.jsonl").read_bytes(), (dest / "groups.jsonl").read_bytes())


if __name__ == "__main__":
    unittest.main()
