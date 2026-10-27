"""Mask-first accounting and synthetic protocol fixtures; no Torch import."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from target_exposure import pack_layers, summarize
from probe_stream_exposure import ROOT, LEDGERS, recipe_digest
from record_stream_exposure import validate_report


def raw():
    return {"layer_known": [True, False], "pose_known": [True, False], "layer_type": [1, 0],
            "column": [2, 0], "row": [16, 0], "rotations": [[True, False, True, False], [False]*4]}


def record(step=1, group="group0"):
    return {"population": "training_updates", "step": step, "sample_id": "sample"+str(step),
            "group_id": group, "layers": pack_layers(raw())}


class TargetExposureTests(unittest.TestCase):
    def test_unknown_padded_type_and_pose_values_do_not_change_exposure(self):
        first, second = raw(), raw()
        second["layer_type"][1] = 7
        second["column"][1], second["row"][1], second["rotations"][1] = 9, 25, [True]*4
        self.assertEqual(pack_layers(first), pack_layers(second))
        self.assertEqual(pack_layers(first)["ghost"], {"status": "unknown", "type": None, "pose": None})

    def test_known_absent_is_supervised_separately_from_unknown(self):
        item = raw()
        item["layer_known"][1] = True
        row = record()
        row["layers"] = pack_layers(item)
        count = summarize([row])["training_updates"]["layers"]["ghost"]
        self.assertEqual((count["absent"], count["unknown"], count["type_counts"][0], count["pose_targets"]), (1, 0, 1, 0))

    def test_roles_and_equivalent_rotations_stay_independent(self):
        count = summarize([record()])["training_updates"]["layers"]
        self.assertEqual(count["active"]["allowed_rotation_counts"], [1, 0, 1, 0])
        self.assertEqual(count["active"]["column_counts"][2], 1)
        self.assertEqual(count["ghost"]["pose_targets"], 0)
        self.assertEqual(sum(count["ghost"]["type_counts"]), 0)

    def test_repeated_views_count_updates_but_group_exposure_once(self):
        count = summarize([record(i) for i in (1, 2, 3)])["training_updates"]
        self.assertEqual((count["targets"], count["ancestry_groups"]), (3, 1))
        self.assertEqual(count["layers"]["active"]["type_counts"][1], 3)
        self.assertEqual(count["layers"]["active"]["unique_group_type_exposures"][1], 1)

    def test_unknown_or_absent_pose_and_invalid_known_coordinates_refuse(self):
        for mutate in (lambda r: r["pose_known"].__setitem__(1, True),
                       lambda r: r["layer_type"].__setitem__(0, 0),
                       lambda r: r["column"].__setitem__(0, 10),
                       lambda r: r["rotations"].__setitem__(0, [False]*4)):
            item = raw()
            mutate(item)
            with self.assertRaises(ValueError):
                pack_layers(item)

    def test_masked_pose_fields_do_not_become_pose_counts(self):
        item = raw()
        item["pose_known"][0] = False
        item["column"][0] = 9
        item["row"][0] = 25
        row = record()
        row["layers"] = pack_layers(item)
        count = summarize([row])["training_updates"]["layers"]["active"]
        self.assertEqual((count["observed"], count["pose_targets"]), (1, 0))
        self.assertEqual(sum(count["column_counts"]), 0)

    def test_duplicate_cursor_invalid_population_and_unknown_padded_export_refuse(self):
        with self.assertRaises(ValueError):
            summarize([record(), record()])
        for key, value in (("population", "hidden_truth"), ("step", True)):
            row = record()
            row[key] = value
            with self.assertRaises(ValueError):
                summarize([row])
        row = record()
        row["layers"]["ghost"]["type"] = 0
        with self.assertRaises(ValueError):
            summarize([row])

    def test_bookkeeping_is_pure_and_does_not_alias_rotation_labels(self):
        item = raw()
        packed = pack_layers(item)
        item["rotations"][0][0] = False
        self.assertTrue(packed["active"]["pose"]["rotations"][0])
        rows = [record()]
        before = copy.deepcopy(rows)
        summarize(rows)
        self.assertEqual(rows, before)


NAME = "m2-stream-exposure-fixture"


def fixture():
    # This is a logical protocol fixture, not captured neural target evidence.
    summary = {"status": "step_partial", "steps": 3, "requested_steps": 4096, "updates_this_run": 3,
               "checkpoint_round_trip_exact": True, "config": {"component_capacity": 0}, "profile_square": False,
               "checkpoint_sha256": "a"*64, "catalogue_sha256": "b"*64}
    source_job = {"status": "failed", "exit_code": 2, "work_items": 3}
    manifest = {"architecture": "TetrisStateNet-M2", "config": summary["config"],
                "weights_sha256": summary["checkpoint_sha256"], "torch_version": "fixture"}
    recipes, records = {}, []
    for population, count in zip(LEDGERS, (3, 9, 18)):
        recipes[population] = []
        for i in range(count):
            split = "development" if population == "development_probes" else "training"
            recipe = {"sample_id": f"{split}-{i}", "group_id": f"{split}-group{i//3}", "view": i % 3,
                      "density": "sparse", "skin": "clean"}
            if population == "training_updates":
                recipe["step"] = i+1
            recipes[population].append(recipe)
            records.append({**recipe, "population": population, "step": recipe.get("step"),
                            "recipe_sha256": recipe_digest(recipe), "layers": pack_layers(raw())})
    result = {"status": "passed", "format": "masked-target-exposure-v1", "run": "m2-stream-source",
              "run_status": summary["status"], "global_steps": 3, "target_steps": 4096,
              "core_architecture": "TetrisStateNet-M2", "config": summary["config"], "torch_version": "fixture",
              "weights_sha256": "a"*64, "source_catalogue_sha256": "b"*64, "type_labels": ["absent", *"IOTSZJL"],
              "target_records": len(records), "model_forwards": 0, "new_updates": 0, "weights_unchanged": True,
              "job_ceiling_bytes": 640*1024**2, "counts": summarize(records)}
    job = {"status": "passed", "exit_code": 0, "work_items": len(records), "memory_bytes": 640*1024**2,
           "command": ["python", str(ROOT/"scripts/probe_stream_exposure.py"), "--run", result["run"], "--output", str(ROOT/"runs"/NAME)]}
    return result, job, summary, source_job, manifest, records, recipes


class ExposureEvidenceTests(unittest.TestCase):
    def test_logical_partial_protocol_is_accepted_without_mutation(self):
        values = fixture()
        before = copy.deepcopy(values)
        self.assertEqual(validate_report(*values, name=NAME), "TetrisStateNet-M2")
        self.assertEqual(values, before)

    def test_original_status_weight_architecture_runtime_and_dialect_are_bound(self):
        for key, value in (("run_status", "complete"), ("global_steps", 4), ("weights_sha256", "c"*64),
                           ("core_architecture", "TetrisStateNet-M3"), ("torch_version", "other"), ("type_labels", [])):
            values = fixture()
            values[0][key] = value
            with self.assertRaises(ValueError):
                validate_report(*values, name=NAME)

    def test_native_command_work_and_no_forward_contract_are_bound(self):
        for mutate in (lambda r, j: j["command"].__setitem__(5, str(ROOT/"runs/other")),
                       lambda r, j: j.update(work_items=1), lambda r, j: j.update(status="running"),
                       lambda r, j: r.update(model_forwards=1), lambda r, j: r.update(new_updates=1),
                       lambda r, j: r.update(weights_unchanged=False)):
            values = fixture()
            mutate(*values[:2])
            with self.assertRaises(ValueError):
                validate_report(*values, name=NAME)

    def test_swapped_or_altered_recipe_identity_refuses(self):
        for mutate in (lambda rows: rows.reverse(), lambda rows: rows[0].update(group_id="other"),
                       lambda rows: rows[0].update(recipe_sha256="c"*64)):
            values = fixture()
            mutate(values[5])
            with self.assertRaises(ValueError):
                validate_report(*values, name=NAME)

    def test_forged_counts_or_unmasked_unknown_target_refuses(self):
        values = fixture()
        values[0]["counts"]["training_updates"]["layers"]["active"]["type_counts"][1] += 1
        with self.assertRaisesRegex(ValueError, "aggregate"):
            validate_report(*values, name=NAME)
        values = fixture()
        values[5][0]["layers"]["ghost"]["type"] = 0
        with self.assertRaises(ValueError):
            validate_report(*values, name=NAME)

    def test_failed_source_cannot_be_relabelled_as_valid_saved_replay(self):
        values = fixture()
        values[3].update(status="memory_pressure", exit_code=125)
        with self.assertRaises(ValueError):
            validate_report(*values, name=NAME)


if __name__ == "__main__":
    unittest.main()
