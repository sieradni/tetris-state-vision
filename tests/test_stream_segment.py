"""Finite controller contracts; source verification is isolated, no model runs."""
from contextlib import contextmanager, redirect_stderr
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import run_stream_segment as segment


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@contextmanager
def fixture(joint=False):
    name = "m3-stream-pilot-profile-r1" if joint else "m2-stream-accuracy-probe"
    evidence = ROOT/"docs/evidence"/name
    read = lambda name: json.loads((evidence/name).read_text())
    profile, job, manifest, archive = (read("summary.json"), read(name+"-job.json"),
                                      read("checkpoint-manifest.json"), read("source-manifest.json"))
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        (root/"scripts").mkdir()
        shutil.copyfile(ROOT/"scripts/run_bounded.py", root/"scripts/run_bounded.py")
        admission = root/"runs"/name
        write(admission/"summary.json", profile)
        write(admission.parent/(name+"-job.json"), job)
        write(admission/"checkpoint/manifest.json", manifest)
        trainer = admission/"code/scripts/train_m2_stream.py"
        trainer.parent.mkdir(parents=True)
        trainer.write_text("# Dummy fixture only; never executed.\n")
        args = segment.arguments(["--name", "m3-stream-study" if joint else "m2-stream-study", "--target-steps", "4096",
                                  "--updates", "512", "--admission", str(admission)])

        def verify(path, expected=None):
            if expected != archive["catalogue_sha256"]:
                raise ValueError("source catalogue differs")
            return copy.deepcopy(archive)

        # verify_source has its own byte-level gates. These fixtures isolate
        # planning/coupling; dummy source is never trusted for native execution.
        with patch.object(segment, "ROOT", root), patch.object(segment, "verify_source", side_effect=verify), patch.object(segment.subprocess, "Popen") as spawn:
            yield SimpleNamespace(root=root, args=args, profile=profile, job=job, manifest=manifest,
                                  archive=archive, admission=admission, spawn=spawn, joint=joint)


def resume(case, *, step=1152, target=4096):
    parent = case.root/"runs/parent"
    trainer = parent/"code/scripts/train_m2_stream.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("# Archived resume fixture, never executed.\n")
    contract = {"catalogue_sha256": case.archive["catalogue_sha256"], "config": copy.deepcopy(case.profile["config"]),
                "steps": target, "seed": 17, "learning_rate": .0003, "evaluate_every": 16, "checkpoint_every": 8,
                "development_groups": 3, "profile_square": False, "weight_decay": .01, "max_side": 512,
                "threads": 1, "rng": "torch_cpu_only", "native_cache_limits": segment.NATIVE_CACHE_LIMITS}
    if case.joint:
        contract["input_policy"] = "M3-final-capture-v1"
    manifest = {"step": step, "contract": contract}
    generation = f"step-{step:06d}-"+"a"*32
    path = parent/"training-snapshots"/generation/"manifest.json"
    write(path, manifest)
    publish(case, parent, path)
    case.args.resume_from = parent
    case.args.target_steps = target
    return manifest, path


def publish(case, parent, path):
    write(parent/"latest-training.json", {"generation": path.parent.name,
          "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})


def outcome(case, plan, updates=2, complete=False):
    summary = {"status": "complete" if complete else "step_partial", "steps": plan["start"]+updates,
               "requested_steps": plan["target_steps"], "updates_this_run": updates, "checkpoint_round_trip_exact": True,
               "config": copy.deepcopy(plan["config"]), "catalogue_sha256": plan["catalogue_sha256"], "model_kind": plan["model_kind"],
               "final_training_probe": {"images": 9, "cases": [{"sample_id": "t"+str(i)} for i in range(9)]},
               "development": {"images": 3*plan["development_groups"],
                               "cases": [{"sample_id": "d"+str(i)} for i in range(3*plan["development_groups"])]}}
    if case.joint:
        summary["component_capacity"] = 64
    job = {"status": "passed" if complete else "failed", "exit_code": 0 if complete else 2,
           "work_items": plan["budget"], "memory_bytes": plan["memory_mib"]*1024**2}
    return summary, job, copy.deepcopy(case.manifest)


class StreamSegmentTests(unittest.TestCase):
    def test_fit_policy_routes_and_resumes_only_its_matching_recipe_and32_probes(self):
        with fixture() as case:
            case.profile.update(curriculum="all-class-fit-v1")
            case.profile["arguments"]["curriculum"] = "all-class-fit-v1"
            case.profile["recovery_contract"]["curriculum"] = "all-class-fit-v1"
            case.manifest["metadata"]["recovery_contract"]["curriculum"] = "all-class-fit-v1"
            write(case.admission/"summary.json", case.profile)
            write(case.admission/"checkpoint/manifest.json", case.manifest)
            plan = segment.segment_plan(case.args)
            self.assertEqual(plan["curriculum"], "all-class-fit-v1")
            self.assertIn("--curriculum", plan["worker"])
            summary, job, manifest = outcome(case, plan)
            summary.update(curriculum="all-class-fit-v1", recovery_contract={"curriculum": "all-class-fit-v1"})
            summary["final_training_probe"] = {"images": 32, "cases": [{"sample_id": str(i)} for i in range(32)]}
            self.assertEqual(segment.validate_segment(summary, job, manifest, plan), 50)
            summary["curriculum"] = "stream-v1"
            with self.assertRaises(ValueError):
                segment.validate_segment(summary, job, manifest, plan)
            saved, path = resume(case, step=32, target=1024)
            saved["contract"]["curriculum"] = "all-class-fit-v1"
            write(path, saved)
            publish(case, case.args.resume_from, path)
            self.assertEqual(segment.segment_plan(case.args)["start"], 32)
            saved["contract"]["curriculum"] = "stream-v1"
            write(path, saved)
            publish(case, case.args.resume_from, path)
            with self.assertRaises(ValueError):
                segment.segment_plan(case.args)

    def test_unknown_or_joint_fit_policy_refuses_before_spawn(self):
        for joint, policy in ((False, "unknown"), (True, "all-class-fit-v1")):
            with fixture(joint) as case:
                case.profile.update(curriculum=policy)
                case.profile["recovery_contract"]["curriculum"] = policy
                case.manifest["metadata"]["recovery_contract"]["curriculum"] = policy
                write(case.admission/"summary.json", case.profile)
                write(case.admission/"checkpoint/manifest.json", case.manifest)
                with self.assertRaises(ValueError):
                    segment.segment_plan(case.args)
                case.spawn.assert_not_called()

    def test_layer_evidence_routes_only_from_verified_configuration_and_resume(self):
        for joint in (False, True):
            with fixture(joint) as case:
                case.profile["config"]["layer_evidence"] = True
                case.manifest["config"]["layer_evidence"] = True
                write(case.admission/"summary.json", case.profile)
                write(case.admission/"checkpoint/manifest.json", case.manifest)
                plan = segment.segment_plan(case.args)
                self.assertTrue(plan["layer_evidence"])
                self.assertIn("--layer-evidence", plan["worker"])
                record, path = resume(case)
                self.assertIn("--layer-evidence", segment.segment_plan(case.args)["worker"])
                record["contract"]["config"]["layer_evidence"] = False
                write(path, record)
                publish(case, case.args.resume_from, path)
                with self.assertRaises(ValueError):
                    segment.segment_plan(case.args)

    def test_legacy_archive_omits_new_layer_keyword_and_nonboolean_refuses(self):
        with fixture() as case:
            self.assertNotIn("--layer-evidence", segment.segment_plan(case.args)["worker"])
            case.profile["config"]["layer_evidence"] = 1
            case.manifest["config"]["layer_evidence"] = 1
            write(case.admission/"summary.json", case.profile)
            write(case.admission/"checkpoint/manifest.json", case.manifest)
            with self.assertRaises(ValueError):
                segment.segment_plan(case.args)

    def test_arguments_keep_finite_names_targets_updates_and_capacity(self):
        base = ["--name", "m3-stream-study", "--target-steps", "4096", "--updates", "512", "--admission", "unused"]
        self.assertIsNone(segment.arguments(base).component_capacity)
        self.assertEqual(segment.arguments([*base, "--component-capacity", "64"]).component_capacity, 64)
        for extra in (["--component-capacity", "9"], ["--updates", "513"], ["--target-steps", "4097"], ["--name", "../bad"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                segment.arguments([*base, *extra])

    def test_fresh_legacy_uses_profile_archive_without_new_keyword(self):
        with fixture() as case:
            plan = segment.segment_plan(case.args)
            self.assertEqual(plan["worker"][1], str(case.admission/"code/scripts/train_m2_stream.py"))
            self.assertNotIn("--component-capacity", plan["worker"])
            self.assertNotIn("--curriculum", plan["worker"])
            self.assertEqual((plan["memory_mib"], plan["model_kind"], plan["budget"]), (1024, "accuracy", 512))
            self.assertFalse(plan["pipeline"].exists())
            case.spawn.assert_not_called()

    def test_joint_uses_actual_pilot_configuration_and_measured_ceiling(self):
        with fixture(True) as case:
            plan = segment.segment_plan(case.args)
            worker = plan["worker"]
            self.assertEqual(worker[worker.index("--component-capacity")+1], "64")
            self.assertEqual((plan["architecture"], plan["model_kind"], plan["memory_mib"]), ("TetrisStateNet-M3", "pilot", 640))
            self.assertEqual(plan["config"], case.profile["config"])

    def test_legacy1152_resume_preserves_all_exposed_recipe_settings(self):
        with fixture() as case:
            resume(case)
            plan = segment.segment_plan(case.args)
            worker = plan["worker"]
            self.assertEqual((plan["start"], plan["budget"]), (1152, 512))
            self.assertEqual(worker[1], str(case.args.resume_from/"code/scripts/train_m2_stream.py"))
            for flag, value in (("--seed", "17"), ("--learning-rate", "0.0003"), ("--evaluate-every", "16"),
                                ("--checkpoint-every", "8"), ("--development-groups", "3")):
                self.assertEqual(worker[worker.index(flag)+1], value)
            self.assertNotIn("--component-capacity", worker)

    def test_joint_resume_clips_updates_to_original_target(self):
        with fixture(True) as case:
            resume(case, step=1, target=3)
            plan = segment.segment_plan(case.args)
            self.assertEqual((plan["start"], plan["budget"], plan["component_capacity"]), (1, 2, 64))

    def test_changed_target_source_configuration_and_capture_policy_refuse(self):
        for mutation in (lambda c: c.update(steps=4095), lambda c: c.update(catalogue_sha256="0"*64),
                         lambda c: c.update(config={"component_capacity": 10}), lambda c: c.update(input_policy="wrong")):
            with fixture(True) as case:
                manifest, path = resume(case)
                mutation(manifest["contract"])
                write(path, manifest)
                publish(case, case.args.resume_from, path)
                with self.assertRaises(ValueError):
                    segment.segment_plan(case.args)
                case.spawn.assert_not_called()
                self.assertFalse((case.root/"runs"/(case.args.name+"-pipeline")).exists())

    def test_pointer_digest_mismatch_refuses_before_work(self):
        with fixture() as case:
            manifest, path = resume(case)
            manifest["step"] = 1153
            write(path, manifest)
            with self.assertRaisesRegex(ValueError, "manifest identity"):
                segment.segment_plan(case.args)
            case.spawn.assert_not_called()

    def test_cold_guard_mismatch_and_unsupported_ceiling_refuse(self):
        for mutation in (lambda p: p["warm_optimizer_backward"].update(backward_completed=False),
                         lambda p: p.update(supervisor_script_sha256="0"*64),
                         lambda p: p.update(job_ceiling_bytes=2048*1024**2)):
            with fixture() as case:
                mutation(case.profile)
                write(case.admission/"summary.json", case.profile)
                with self.assertRaises(ValueError):
                    segment.segment_plan(case.args)
                case.spawn.assert_not_called()

    def test_name_and_explicit_capacity_must_match_actual_artifact(self):
        for name, capacity in (("m2-stream-study", None), ("m3-stream-study", 0)):
            with fixture(True) as case:
                case.args.name, case.args.component_capacity = name, capacity
                with self.assertRaisesRegex(ValueError, "actual component capacity"):
                    segment.segment_plan(case.args)

    def test_completed_diagnostic_cannot_be_extended_or_restarted(self):
        for target in (2, 3):
            with fixture() as case:
                resume(case, step=2, target=2)
                case.args.target_steps = target
                with self.assertRaises(ValueError):
                    segment.segment_plan(case.args)

    def test_existing_outputs_are_preserved(self):
        with fixture() as case:
            output = case.root/"runs"/case.args.name
            output.mkdir()
            (output/"keep").write_text("owner-data")
            with self.assertRaises(FileExistsError):
                segment.segment_plan(case.args)
            self.assertEqual((output/"keep").read_text(), "owner-data")

    def test_result_counts_partial_and_complete_cursor_and_real_probe_population(self):
        with fixture(True) as case:
            resume(case, step=1, target=3)
            plan = segment.segment_plan(case.args)
            self.assertEqual(segment.validate_segment(*outcome(case, plan, 1), plan), 18)
            self.assertEqual(segment.validate_segment(*outcome(case, plan, 2, True), plan), 18)

    def test_result_wrong_cursor_resources_source_and_probe_duplicates_refuse(self):
        for mutation in (lambda s, j: s.update(steps=2), lambda s, j: j.update(memory_bytes=1024*1024**2),
                         lambda s, j: s.update(catalogue_sha256="0"*64), lambda s, j: s.update(checkpoint_round_trip_exact=False),
                         lambda s, j: s["development"]["cases"].__setitem__(0, s["development"]["cases"][1])):
            with fixture(True) as case:
                resume(case, step=1, target=3)
                plan = segment.segment_plan(case.args)
                summary, job, manifest = outcome(case, plan, 2, True)
                mutation(summary, job)
                with self.assertRaises(ValueError):
                    segment.validate_segment(summary, job, manifest, plan)

    def test_controller_source_change_cannot_receive_verified_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)/"controller.py"
            source.write_text("original")
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            with patch.object(segment, "__file__", str(source)):
                segment.require_controller_source(before)
                source.write_text("edited")
                with self.assertRaisesRegex(ValueError, "source changed"):
                    segment.require_controller_source(before)

    def test_plan_only_verifies_without_output_creation_or_child_launch(self):
        with fixture(True) as case:
            case.args.plan_only = True
            output = io.StringIO()
            with patch.object(segment, "require_bounded_job"), patch.object(segment, "arguments", return_value=case.args), patch("sys.stdout", output):
                self.assertEqual(segment.main(), 0)
            result = json.loads(output.getvalue())
            self.assertEqual((result["status"], result["component_capacity"], result["budget"]), ("plan_verified", 64, 512))
            self.assertFalse((case.root/"runs"/case.args.name).exists())
            self.assertFalse((case.root/"runs"/(case.args.name+"-pipeline")).exists())
            case.spawn.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "native Windows admission")
    def test_no_named_job_refuses_before_profile_or_output_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = dict(os.environ)
            environment.pop("TETRIS_VISION_JOB_NAME", None)
            result = subprocess.run([sys.executable, str(ROOT/"scripts/run_stream_segment.py"), "--name", "m3-stream-nojob-test",
                                     "--target-steps", "3", "--updates", "1", "--admission", str(Path(temporary)/"missing")],
                                    capture_output=True, text=True, env=environment, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("named supervisor Job identity", result.stderr)


if __name__ == "__main__":
    unittest.main()
