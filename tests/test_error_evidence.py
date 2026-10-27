"""Adversarial metadata gates built from a preserved actual partial M2 report."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from record_stream_errors import validate_report


NAME = "m2-stream-accuracy704-errors-r2"


def fixture():
    report = ROOT/"docs/evidence"/NAME
    source = ROOT/"docs/evidence/m2-stream-accuracy4096-s2"
    read = lambda path: json.loads(path.read_text())
    result, job = read(report/"result.json"), read(report/"native-job.json")
    summary, manifest = read(source/"summary.json"), read(source/"checkpoint-manifest.json")
    recipes = {split: [json.loads(line) for line in (source/(split+"-probe.jsonl")).read_text().splitlines()]
               for split in ("training", "development")}
    # Historical report predates format2. Construct metadata-only version2
    # fixtures from its preserved ordinary metrics; this performs no inference.
    result.update(diagnostic_format_version=2, core_architecture=manifest["architecture"], config=manifest["config"], component_capacity=0)
    for row in result["cases"]:
        field = "final_training_probe" if row["split"] == "training" else "development"
        ordinary = next(case for case in summary[field]["cases"] if case["sample_id"] == row["sample_id"])
        row["ordinary_metrics"] = {key: value for key, value in ordinary.items() if key != "sample_id"}
    return result, job, summary, manifest, recipes


class ErrorEvidenceTests(unittest.TestCase):
    def test_actual_partial_metadata_is_accepted_without_rewriting_it(self):
        values = fixture()
        before = copy.deepcopy(values)
        self.assertEqual(validate_report(*values, name=NAME), "TetrisStateNet-M2")
        self.assertEqual(values, before)
        self.assertEqual(values[0]["run_status"], "step_partial")
        self.assertEqual((values[0]["global_steps"], values[0]["target_steps"]), (704, 4096))

    def test_weight_configuration_architecture_and_partial_tag_mismatch_refuse(self):
        for mutation in (lambda r, j, s, m: r.update(weights_sha256="0"*64),
                         lambda r, j, s, m: r.update(global_steps=705),
                         lambda r, j, s, m: r.update(run_status="complete"),
                         lambda r, j, s, m: r.update(component_capacity=64),
                         lambda r, j, s, m: m.update(architecture="TetrisStateNet-M3")):
            values = fixture()
            mutation(*values[:4])
            with self.assertRaises(ValueError):
                validate_report(*values, name=NAME)

    def test_native_command_work_and_ordinary_only_contract_refuse(self):
        for mutation in (lambda r, j: j["command"].__setitem__(3, "m3-stream-other"),
                         lambda r, j: j["command"].__setitem__(5, str(ROOT/"runs/other")),
                         lambda r, j: j["command"].extend(["--oracle", "true"]),
                         lambda r, j: j.update(work_items=54), lambda r, j: j.update(status="running"),
                         lambda r, j: r.update(label_inputs_to_model=True), lambda r, j: r.update(weights_unchanged=False)):
            values = fixture()
            mutation(*values[:2])
            with self.assertRaises(ValueError):
                validate_report(*values, name=NAME)

    def test_missing_duplicate_and_changed_ancestry_cases_refuse(self):
        for mutation in (lambda r: r["cases"].pop(), lambda r: r["cases"].append(copy.deepcopy(r["cases"][0])),
                         lambda r: r["cases"][0].update(group_id="different-session"),
                         lambda r: r["cases"][0].update(split="development")):
            values = fixture()
            mutation(values[0])
            values[0]["forwards"] = values[1]["work_items"] = len(values[0]["cases"])
            with self.assertRaises(ValueError):
                validate_report(*values, name=NAME)

    def test_changed_ordinary_metric_refuses(self):
        values = fixture()
        values[0]["cases"][0]["ordinary_metrics"]["loss"] += .001
        with self.assertRaisesRegex(ValueError, "ordinary metrics"):
            validate_report(*values, name=NAME)

    def test_decomposition_cannot_disagree_with_saved_metrics(self):
        for mutation in (lambda d: d["locked"].update(exact_board=not d["locked"]["exact_board"]),
                         lambda d: d["geometry"].update(mean_cells=d["geometry"]["mean_cells"]+1)):
            values = fixture()
            mutation(values[0]["cases"][0]["diagnostics"])
            with self.assertRaisesRegex(ValueError, "decomposition"):
                validate_report(*values, name=NAME)

    def test_aggregate_is_recomputed_instead_of_trusted(self):
        values = fixture()
        values[0]["totals"]["development"]["locked_populations"]["nonempty"]["exact_boards"] = 14
        with self.assertRaisesRegex(ValueError, "aggregate"):
            validate_report(*values, name=NAME)

    def test_legacy_format_and_foreign_component_dialect_refuse(self):
        values = fixture()
        del values[0]["diagnostic_format_version"]
        with self.assertRaisesRegex(ValueError, "ordinary diagnostic"):
            validate_report(*values, name=NAME)
        values = fixture()
        values[0]["cases"][0]["diagnostics"]["components"] = {}
        with self.assertRaisesRegex(ValueError, "component diagnostics"):
            validate_report(*values, name=NAME)


if __name__ == "__main__":
    unittest.main()
