"""Metadata-only fixtures adapted from preserved native pilot geometry data."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from geometry_contract import DENSE_METRICS, input_contract, summarize
from record_stream_geometry import validate_report

NAME = "m2-stream-geometry-fixture"


def fixture():
    saved = ROOT/"docs/evidence/m2-stream-pilot-geometry"
    original = ROOT/"docs/evidence/m2-stream-learn768-r3"
    read = lambda path: json.loads(path.read_text())
    report, job = read(saved/"result.json"), read(saved/"m2-stream-pilot-geometry-job.json")
    summary, source_job = read(original/"summary.json"), read(original/"m2-stream-learn768-r3-job.json")
    manifest = read(saved/"input-checkpoint-manifest.json")
    recipes = {split: [json.loads(line) for line in (original/(split+"-probe.jsonl")).read_text().splitlines()]
               for split in ("training", "development")}
    report.update(diagnostic_format_version=2, run="m2-stream-learn768-r3", run_status=summary["status"],
                  global_steps=summary["steps"], target_steps=summary["requested_steps"],
                  core_architecture=manifest["architecture"], config=manifest["config"], component_capacity=0)
    job["command"] = ["python", str(ROOT/"scripts/probe_stream_geometry.py"), "--run", report["run"], "--output", str(ROOT/"runs"/NAME)]
    for row in report["cases"]:
        row["view"] = next(recipe["view"] for recipe in recipes[row["split"]] if recipe["sample_id"] == row["sample_id"])
    return report, job, summary, source_job, manifest, recipes


class GeometryEvidenceTests(unittest.TestCase):
    def test_dense_dialect_preserves_explicit_observed_and_absent_denominators(self):
        values = fixture()
        report, _, summary, _, manifest, _ = values
        for population in (summary, manifest):
            population["config"]["layer_evidence"] = True
        for row in report["cases"]:
            for mode in ("automatic", "oracle_geometry"):
                row[mode].update(dict(zip(DENSE_METRICS, (2, 0, 1, 0))))
            field = "final_training_probe" if row["split"] == "training" else "development"
            saved = next(case for case in summary[field]["cases"] if case["sample_id"] == row["sample_id"])
            saved.update(dict(zip(DENSE_METRICS, (2, 0, 1, 0))))
        report["totals"] = summarize(report["cases"], layer_evidence=True)
        self.assertEqual(validate_report(*values, name=NAME), "TetrisStateNet-M2")
        self.assertEqual(report["totals"]["development"]["automatic"]["dense_observed_visible_grids_known"], 18)
        with self.assertRaises(ValueError):
            summarize(report["cases"])

    def test_dense_counts_cannot_hide_missing_observed_or_overcount_exact_grids(self):
        records = fixture()[0]["cases"]
        for row in records:
            for mode in ("automatic", "oracle_geometry"):
                row[mode].update(dict(zip(DENSE_METRICS, (2, 0, 1, 0))))
        records[0]["automatic"]["dense_layer_grids_exact"] = 2
        with self.assertRaisesRegex(ValueError, "dense"):
            summarize(records, layer_evidence=True)

    def test_native_completed_metadata_and_negative_oracle_effect_are_valid(self):
        values = fixture()
        before = copy.deepcopy(values)
        self.assertEqual(validate_report(*values, name=NAME), "TetrisStateNet-M2")
        self.assertEqual(values, before)
        totals = values[0]["totals"]["development"]
        self.assertLess(totals["oracle_geometry"]["cell_accuracy"], totals["automatic"]["cell_accuracy"])

    def test_controlled_partial_requires_explicit_flag_and_preserves_status(self):
        report, job, summary, source_job, manifest, recipes = fixture()
        summary.update(status="step_partial", requested_steps=4096)
        source_job.update(status="failed", exit_code=2)
        report.update(run_status="step_partial", target_steps=4096)
        with self.assertRaises(ValueError):
            validate_report(report, job, summary, source_job, manifest, recipes, name=NAME)
        job["command"].append("--allow-partial")
        self.assertEqual(validate_report(report, job, summary, source_job, manifest, recipes, name=NAME), "TetrisStateNet-M2")
        self.assertEqual((report["run_status"], report["global_steps"]), ("step_partial", 768))

    def test_failed_or_incomplete_source_and_wrong_architecture_refuse(self):
        values = fixture()
        values[3].update(status="memory_pressure", exit_code=125)
        with self.assertRaises(ValueError):
            input_contract(values[2], values[3], values[4], allow_partial=True)
        values = fixture()
        values[2]["config"]["component_capacity"] = 64
        values[2]["component_capacity"] = 64
        values[4]["config"]["component_capacity"] = 64
        values[4]["architecture"] = "TetrisStateNet-M3"
        with self.assertRaisesRegex(ValueError, "M2-only"):
            input_contract(values[2], values[3], values[4])

    def test_forged_aggregate_is_recomputed(self):
        values = fixture()
        values[0]["totals"]["development"]["oracle_geometry"]["exact_locked_boards"] = 18
        with self.assertRaisesRegex(ValueError, "aggregate"):
            validate_report(*values, name=NAME)

    def test_command_output_work_memory_and_oracle_policy_are_bound(self):
        for mutation in (lambda r, j: j["command"].__setitem__(3, "m2-stream-other"),
                         lambda r, j: j["command"].__setitem__(5, str(ROOT/"runs/other")),
                         lambda r, j: j["command"].append("--unexpected"),
                         lambda r, j: j.update(work_items=27), lambda r, j: j.update(memory_bytes=2**31),
                         lambda r, j: r.update(automatic_accuracy_claim_allowed=True),
                         lambda r, j: r.update(oracle_geometry_provided=False)):
            values = fixture()
            mutation(*values[:2])
            with self.assertRaises(ValueError):
                validate_report(*values, name=NAME)

    def test_checkpoint_configuration_weight_and_partial_tag_are_bound(self):
        for key, value in (("weights_sha256", "0"*64), ("core_architecture", "TetrisStateNet-M3"),
                           ("config", {}), ("run_status", "step_partial"), ("global_steps", 769),
                           ("target_steps", 4096), ("component_capacity", 64)):
            values = fixture()
            values[0][key] = value
            with self.assertRaises(ValueError):
                validate_report(*values, name=NAME)

    def test_missing_duplicate_case_and_changed_ancestry_refuse(self):
        for mutation in (lambda r: r["cases"].pop(),
                         lambda r: r["cases"].__setitem__(1, copy.deepcopy(r["cases"][0])),
                         lambda r: r["cases"][0].update(group_id="another-session"),
                         lambda r: r["cases"][0].update(view=99)):
            values = fixture()
            mutation(values[0])
            with self.assertRaises(ValueError):
                validate_report(*values, name=NAME)

    def test_changed_automatic_metric_and_oracle_target_count_refuse(self):
        for mutation in (lambda row: row["automatic"].update(exact_known_layers=0),
                         lambda row: row["oracle_geometry"].update(known_layers=1, exact_known_layers=1)):
            values = fixture()
            mutation(values[0]["cases"][0])
            with self.assertRaises(ValueError):
                validate_report(*values, name=NAME)

    def test_nonfinite_unsupported_metrics_and_tautological_geometry_refuse(self):
        for key, value in (("cell_accuracy", float("nan")), ("known_cells", 199),
                           ("exact_locked_board", 1), ("mean_geometry_error_cells", 0)):
            records = fixture()[0]["cases"]
            records[0]["oracle_geometry"][key] = value
            with self.assertRaisesRegex(ValueError, "finite supported"):
                summarize(records)

    def test_legacy_reports_do_not_silently_acquire_new_scope(self):
        values = fixture()
        del values[0]["diagnostic_format_version"]
        with self.assertRaises(ValueError):
            validate_report(*values, name=NAME)


if __name__ == "__main__":
    unittest.main()
