"""Integrated whole-group accumulation loop: cadence, accounting and recovery.

This exercises the trainer's own loop against the durable journal/snapshot
bridge with a toy network. It is not training admission or real-image evidence.
"""
import contextlib
import copy
import hashlib
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
import unittest

if any(os.environ.get(name) != "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
    raise RuntimeError("bounded pre-import CPU policy required")
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from accumulation_checkpoint import create_journal, make_contract, restore_boundary
from train_m2_stream import (NATIVE_CACHE_LIMITS, accumulation_profile, arguments, equal_tree,
                             run_accumulation_loop, validate_admission)

CONFIG = {"dimension": 3, "component_capacity": 0}


def recipe_for(policy):
    curriculum = "all-class-fit-v1" if policy == "ordered-eight-v1" else "all-class-interleaved-v1"

    def recipe(image, index, view):
        return {"sample_id": f"all-class-{index:02d}-v{view}", "group_id": f"all-class-{index:02d}",
                "index": index, "view": view, "split": "fit", "curriculum": curriculum,
                "profile_square": False,
                "image_pixels_sha256": hashlib.sha256(f"pixels:{index}:{view}".encode("ascii")).hexdigest(),
                "state_sha256": hashlib.sha256(f"state:{index}:{view}".encode("ascii")).hexdigest()}
    return recipe


def loss_for(network):
    def loss(recipe):
        sample = torch.tensor([[recipe["index"]/16, recipe["view"]/2, .25]])
        return (network(sample)-torch.tensor([[.2, -.1]])).square().mean()
    return loss


def network_and_optimizer(seed):
    torch.manual_seed(seed)
    network = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.Dropout(.25), torch.nn.Linear(6, 2))
    return network, torch.optim.AdamW(network.parameters(), lr=.001, weight_decay=.01)


class AccumulationDriverTest(unittest.TestCase):
    def test_whole_groups_record_images_and_updates_separately(self):
        policy = "balanced-eight-v1"
        model, optimizer = network_and_optimizer(83)
        contract = make_contract(policy, source_catalogue_sha256="c"*64, config=CONFIG)
        groups, evaluations, progress = [], [], []
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)/"run"
            run.mkdir()
            create_journal(run)
            result = run_accumulation_loop(
                run=run, policy=policy, contract=contract, model=model, optimizer=optimizer,
                updates_total=3, recipe_for=recipe_for(policy), loss_for=loss_for(model),
                checkpoint_every=16, evaluate_every=8, wall_seconds=600, started=time.monotonic(),
                on_group=lambda update, state, entry: groups.append(
                    (update, state["images_seen"], state["optimizer_updates"])),
                on_evaluation=lambda update, state: evaluations.append((update, state["images_seen"])),
                on_progress=lambda update, state, mean_loss: progress.append(
                    (update, state["images_seen"], mean_loss)))
            self.assertEqual(groups, [(1, 8, 1), (2, 16, 2), (3, 24, 3)])
            self.assertEqual(evaluations, [(1, 8), (2, 16), (3, 24)])
            self.assertEqual([row[0] for row in progress], [1, 2, 3])
            self.assertTrue(all(0 < row[2] < 1 for row in progress))
            self.assertEqual(len((run/"groups.jsonl").read_text().splitlines()), 3)
            self.assertEqual(result["state"]["status"], "partial")
            self.assertEqual(result["snapshotted_images"], 24)
            pointer = json.loads((run/"latest-training.json").read_text())
            manifest = json.loads((run/"training-snapshots"/pointer["generation"]/"manifest.json").read_text())
            self.assertEqual(manifest["step"], 3)
            self.assertEqual(set(manifest["ledgers"]), {"groups.jsonl"})
            # Snapshots only at the declared image cadence plus the final group.
            self.assertEqual(len(list((run/"training-snapshots").iterdir())), 2)

    def test_profile_characterization_measures_warm_state_and_actual_reload(self):
        policy = "balanced-eight-v1"
        model, optimizer = network_and_optimizer(101)
        contract = make_contract(policy, source_catalogue_sha256="c"*64, config=CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)/"run"
            run.mkdir()
            create_journal(run)
            run_accumulation_loop(
                run=run, policy=policy, contract=contract, model=model, optimizer=optimizer,
                updates_total=2, recipe_for=recipe_for(policy), loss_for=loss_for(model),
                checkpoint_every=8, evaluate_every=8, wall_seconds=600, started=time.monotonic(),
                on_group=lambda update, state, entry: None)
            profile = accumulation_profile(run=run, model=model, optimizer=optimizer, contract=contract,
                                           policy=policy, completed=2, maximum_pixels=512**2)
            self.assertEqual((profile["step"], profile["group_size"], profile["images"]), (2, 8, 16))
            self.assertEqual(profile["microsteps"], 16)
            self.assertEqual(profile["input_pixels"], 512**2)
            self.assertEqual(profile["policy"], policy)
            self.assertGreater(profile["state_bytes"], 0)
            self.assertGreater(profile["parameter_tensors"], 0)
            self.assertTrue(profile["all_parameters_initialized"])
            self.assertTrue(profile["backward_completed"])
            self.assertTrue(profile["recovery_exact"])
            self.assertGreater(profile["warm_peak_commit_bytes"], 0)
            self.assertGreaterEqual(profile["reload_peak_commit_bytes"], profile["warm_peak_commit_bytes"])
            self.assertGreaterEqual(profile["reload_commit_delta_bytes"], 0)
            self.assertGreaterEqual(profile["reload_seconds"], 0)
            self.assertGreater(profile["reload_seconds"], 0)
            self.assertEqual((profile["device"], profile["device_memory_bytes"]), ("cpu", 0))

    def test_accumulation_admission_requires_the_matching_group_size_profile(self):
        policy = "balanced-eight-v1"
        recovery_contract = {"curriculum": "all-class-interleaved-v1", "type_head_rate_factor": 1}
        warm = {"step": 2, "input_pixels": 512**2, "all_parameters_initialized": True,
                "backward_completed": True, "state_bytes": 1, "policy": policy,
                "group_size": 8, "microsteps": 16, "recovery_exact": True,
                "optimizer_updates_per_second": .1}
        summary = {"status": "complete", "steps": 2, "profile_square": True, "resource_profile_version": 3,
                   "training_recovery_profile_exact": True, "checkpoint_round_trip_exact": True,
                   "max_input_pixels_seen": 512**2, "native_cache_limits": NATIVE_CACHE_LIMITS,
                   "max_side": 512, "threads": 1, "workers": 0, "prefetch": 0, "tensor_cache_bytes": 0,
                   "catalogue_sha256": "a"*64, "model_kind": "pilot", "config": {},
                   "job_ceiling_bytes": 640*1024**2, "development_groups": 6, "cpu_only": True,
                   "system_pressure_guard": "GetPerformanceInfo/0.2s",
                   "supervisor_script_sha256": "b"*64, "curriculum": "all-class-interleaved-v1",
                   "type_head_rate_factor": 1, "accumulation_policy": policy, "images_per_update": 8,
                   "optimizer_updates_per_second": .1,
                   "arguments": {"curriculum": "all-class-interleaved-v1", "type_head_rate_factor": 1,
                                 "accumulation_policy": policy},
                   "recovery_contract": recovery_contract, "warm_optimizer_backward": warm}
        job = {"status": "passed", "exit_code": 0, "work_items": 16, "memory_bytes": 640*1024**2,
               "system_pressure_guard": "GetPerformanceInfo/0.2s",
               "supervisor_script_sha256": "b"*64}
        expected = dict(catalogue="a"*64, model="pilot", config={}, ceiling=640*1024**2,
                        development_groups=6, supervisor_sha="b"*64,
                        curriculum="all-class-interleaved-v1", accumulation_policy=policy,
                        profile_work_items=16)
        validate_admission(summary, job, **expected)
        for mutate in ({"group_size": 1}, {"microsteps": 2}, {"recovery_exact": False},
                       {"policy": "online-interleaved-v1"}):
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate_admission({**summary, "warm_optimizer_backward": {**warm, **mutate}},
                                   job, **expected)
        for mutate, broken in (("work_items", 2), ("exit_code", 1)):
            with self.subTest(job_field=mutate), self.assertRaises(ValueError):
                validate_admission(summary, {**job, mutate: broken}, **expected)
        for mutate in ({"accumulation_policy": "online-interleaved-v1"},
                       {"images_per_update": 1},
                       {"optimizer_updates_per_second": 0}):
            with self.subTest(summary_field=tuple(mutate)), self.assertRaises(ValueError):
                validate_admission({**summary, **mutate}, job, **expected)

    def test_profile_cli_accepts_only_the_matching_accumulation_arm(self):
        base = ["--output", "unused", "--model", "pilot", "--steps", "2", "--profile-square"]
        args = arguments(base+["--curriculum", "all-class-interleaved-v1",
                               "--accumulation-policy", "balanced-eight-v1"])
        self.assertEqual(args.accumulation_policy, "balanced-eight-v1")
        self.assertIsNone(arguments(base+["--curriculum", "stream-v1"]).accumulation_policy)
        for wrong in (base+["--curriculum", "all-class-interleaved-v1",
                             "--accumulation-policy", "ordered-eight-v1"],
                      ["--output", "unused", "--model", "pilot", "--steps", "128", "--profile-square",
                       "--curriculum", "all-class-interleaved-v1",
                       "--accumulation-policy", "balanced-eight-v1"]):
            with self.subTest(arguments=wrong), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    arguments(wrong)

    def test_interrupted_group_is_discarded_and_replayed_to_identical_bytes(self):
        policy = "balanced-eight-v1"
        contract = make_contract(policy, source_catalogue_sha256="c"*64, config=CONFIG)
        model, optimizer = network_and_optimizer(97)
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory)/"reference"
            reference.mkdir()
            create_journal(reference)
            run_accumulation_loop(
                run=reference, policy=policy, contract=contract, model=model, optimizer=optimizer,
                updates_total=3, recipe_for=recipe_for(policy), loss_for=loss_for(model),
                checkpoint_every=8, evaluate_every=8, wall_seconds=600, started=time.monotonic(),
                on_group=lambda update, state, entry: None)
            expected = (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()),
                        torch.get_rng_state().clone())
            expected_journal = (reference/"groups.jsonl").read_bytes()

            source = Path(directory)/"source"
            source.mkdir()
            create_journal(source)
            resumed_model, resumed_optimizer = network_and_optimizer(97)
            run_accumulation_loop(
                run=source, policy=policy, contract=contract, model=resumed_model,
                optimizer=resumed_optimizer, updates_total=1, recipe_for=recipe_for(policy),
                loss_for=loss_for(resumed_model), checkpoint_every=8, evaluate_every=8,
                wall_seconds=600, started=time.monotonic(),
                on_group=lambda update, state, entry: None)

            calls = 0
            pointer_before = (source/"latest-training.json").read_text()

            def interrupted(recipe):
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise RuntimeError("forced inside group")
                return loss_for(resumed_model)(recipe)
            with self.assertRaisesRegex(RuntimeError, "forced inside group"):
                run_accumulation_loop(
                    run=source, policy=policy, contract=contract, model=resumed_model,
                    optimizer=resumed_optimizer, updates_total=3, start_update=2,
                    recipe_for=recipe_for(policy), loss_for=interrupted, checkpoint_every=8,
                    evaluate_every=8, wall_seconds=600, started=time.monotonic(),
                    on_group=lambda update, state, entry: None)
            self.assertEqual(len((source/"groups.jsonl").read_text().splitlines()), 1)
            self.assertEqual(pointer_before, (source/"latest-training.json").read_text())

            resumed = Path(directory)/"resumed"
            resumed.mkdir()
            state = restore_boundary(source, resumed, resumed_model, resumed_optimizer, contract)
            self.assertEqual((state["images_seen"], state["optimizer_updates"]), (8, 1))
            result = run_accumulation_loop(
                run=resumed, policy=policy, contract=contract, model=resumed_model,
                optimizer=resumed_optimizer, updates_total=3, start_update=2,
                recipe_for=recipe_for(policy), loss_for=loss_for(resumed_model),
                checkpoint_every=8, evaluate_every=8, wall_seconds=600, started=time.monotonic(),
                on_group=lambda update, state, entry: None)
            self.assertEqual(result["state"]["status"], "partial")
            self.assertTrue(equal_tree(expected[0], resumed_model.state_dict()))
            self.assertTrue(equal_tree(expected[1], resumed_optimizer.state_dict()))
            self.assertTrue(torch.equal(expected[2], torch.get_rng_state()))
            self.assertEqual(expected_journal, (resumed/"groups.jsonl").read_bytes())


if __name__ == "__main__":
    unittest.main()
