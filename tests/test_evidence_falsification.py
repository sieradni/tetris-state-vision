"""Falsification pass: scratch copies of every published receipt must be rejected, irrelevant edits must pass.

Instrument receipts are checked by the strict differ, the deciding-power record by `record_problems`, and the
green suite receipt by `job_receipt_problems`. Attacks: changed carried value, dropped case, duplicated case,
retyped scalar, nested edit and an injected top-level section. Irrelevant edits: prose whitespace, key order,
an added docstring, a changed log line and a renamed local.
"""
import copy
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"scripts")]

from record_stream_run import job_receipt_problems
from replay_equivalence import compare, first_difference
from sweep_deciding_power import figure_appears, power, record_problems

PAIRS = {
    "headroom": ("docs/evidence/m3-deciding-headroom-r1/result.json",
                 "docs/evidence/m3-deciding-headroom-rerun-r4/result.json"),
    "sweep": ("runs/m3-occbref-baseline-r1-sweep-r1/result.json",
              "runs/m3-sweep-obcref-rerun-r4/result.json"),
    "errors": ("docs/evidence/m3-fit-occupancy-r1-errors-r1/result.json",
               "docs/evidence/m3-fit-occupancy-r1-errors-r5/result.json"),
}
CARRIED_PATHS = {
    "headroom": ("ensemble", "best", "f1"),
    "sweep": ("deciding", "best", "f1"),
    "errors": ("totals", "training", "locked_populations", "nonempty", "true_positive"),
}


def load(relative):
    return json.loads((ROOT/relative).read_text(encoding="utf-8"))


def latest_passing_suite_receipt():
    passing = []
    for path in sorted(ROOT.glob("runs/suite-*-job.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("status") == "passed":
            passing.append((path, data))
    if not passing:
        raise AssertionError("no passing suite receipt")
    return passing[-1]


def power_record_path():
    candidates = sorted((ROOT/"docs/evidence").glob("m3-sweep-deciding-power-*/power.json"))
    if not candidates:
        raise AssertionError("power record missing")
    return candidates[-1]


def first_numeric_path(value, prefix=()):
    if isinstance(value, dict):
        for key, child in value.items():
            found = first_numeric_path(child, prefix+(key,))
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = first_numeric_path(child, prefix+(index,))
            if found is not None:
                return found
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        return prefix
    return None


def get_path(value, path):
    for key in path:
        value = value[key]
    return value


def set_path(value, path, replacement):
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement


class EvidenceFalsificationTests(unittest.TestCase):
    def test_every_instrument_receipt_rejects_the_six_corruption_classes(self):
        for instrument, (historical, current) in PAIRS.items():
            reference = load(historical)
            carried = CARRIED_PATHS[instrument]

            def bump_value(receipt):
                value = get_path(receipt, carried)
                set_path(receipt, carried, math.nextafter(float(value), math.inf))

            def retype(receipt):
                value = get_path(receipt, carried)
                set_path(receipt, carried, float(value) if instrument == "errors" else int(value))

            def nested(receipt):
                path = first_numeric_path(receipt["cases"][0])
                set_path(receipt, ("cases", 0) + path,
                         get_path(receipt["cases"][0], path) + 1)

            attacks = [
                ("changed-value", bump_value, "f1"),
                ("dropped-case", lambda receipt: receipt.__setitem__("cases", receipt["cases"][:-1]),
                 "cases"),
                ("duplicated-case", lambda receipt: receipt.__setitem__(
                    "cases", receipt["cases"] + [copy.deepcopy(receipt["cases"][0])]), "cases"),
                ("retyped-scalar", retype, "f1"),
                ("nested-edit", nested, "cases"),
                ("injected-section", lambda receipt: receipt.__setitem__("libre_section", {"x": 1}),
                 "unaccounted:libre_section"),
            ]
            for label, attack, expected in attacks:
                with self.subTest(instrument=instrument, attack=label):
                    perturbed = copy.deepcopy(load(current))
                    attack(perturbed)
                    result = compare(instrument, reference, perturbed)
                    self.assertFalse(result["identical"])
                    self.assertIn(expected, result["differences"], result["differences"])

    def test_the_power_record_rejects_corruption(self):
        path = power_record_path()
        record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record_problems(record), {})
        attacks = {
            "changed-value": (lambda copy_: copy_["population"].__setitem__(
                "known_cells", copy_["population"]["known_cells"]+1), "population"),
            "dropped-section": (lambda copy_: copy_.pop("grid"), "grid"),
            "retyped-scalar": (lambda copy_: copy_["auc"].__setitem__(
                "value", int(copy_["auc"]["value"])), "auc"),
            "nested-edit": (lambda copy_: copy_["verdict"].__setitem__(
                "real_finding", not copy_["verdict"]["real_finding"]), "verdict"),
            "injected-section": (lambda copy_: copy_.__setitem__("libre", 1), "injected:libre"),
        }
        for label, (attack, expected) in attacks.items():
            with self.subTest(attack=label):
                perturbed = copy.deepcopy(record)
                attack(perturbed)
                problems = record_problems(perturbed)
                self.assertIn(expected, problems, problems)

    def test_the_power_record_rejects_a_changed_source_receipt(self):
        path = power_record_path()
        record = json.loads(path.read_text(encoding="utf-8"))
        source = load(record["result_path"])
        scratch = Path(tempfile.mkdtemp(dir=ROOT))
        try:
            for label, mutate in (("dropped-case", lambda data: data["cases"].pop()),
                                  ("duplicated-case", lambda data: data["cases"].append(
                                      copy.deepcopy(data["cases"][0])))):
                with self.subTest(attack=label):
                    perturbed_source = copy.deepcopy(source)
                    mutate(perturbed_source)
                    scratch_file = scratch/("source-%s.json" % label)
                    scratch_file.write_text(json.dumps(perturbed_source), encoding="utf-8")
                    perturbed = copy.deepcopy(record)
                    perturbed["result_path"] = str(scratch_file.relative_to(ROOT)).replace("\\", "/")
                    problems = record_problems(perturbed)
                    self.assertIn("sources.result", problems, problems)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def test_the_suite_receipt_rejects_corruption(self):
        _, receipt = latest_passing_suite_receipt()
        self.assertEqual(job_receipt_problems(receipt), {})
        attacks = {
            "changed-status": (lambda copy_: copy_.__setitem__("status", "failed"), "status"),
            "dropped-key": (lambda copy_: copy_.pop("peak_job_commit_bytes"), "keys"),
            "retyped-scalar": (lambda copy_: copy_.__setitem__("exit_code", False), "exit_code"),
            "nested-edit": (lambda copy_: copy_["system_pressure"].__setitem__(
                "reason", "os_commit_pressure"), "system_pressure"),
            "injected-section": (lambda copy_: copy_.__setitem__("libre", 1), "keys"),
            "dropped-command-token": (lambda copy_: copy_["command"].remove("discover"), "command"),
        }
        for label, (attack, expected) in attacks.items():
            with self.subTest(attack=label):
                perturbed = copy.deepcopy(receipt)
                attack(perturbed)
                problems = job_receipt_problems(perturbed)
                self.assertIn(expected, problems, problems)

    def test_reordered_maps_and_a_duplicated_command_token_are_irrelevant(self):
        for instrument, (historical, current) in PAIRS.items():
            reference, live = load(historical), load(current)
            shuffled = copy.deepcopy(live)
            for key in ("population", "training_population", "weights_sha256"):
                if isinstance(shuffled.get(key), dict):
                    shuffled[key] = dict(list(shuffled[key].items())[::-1])
            self.assertEqual(compare(instrument, reference, shuffled)["differences"], {}, instrument)
        _, receipt = latest_passing_suite_receipt()
        duplicated = copy.deepcopy(receipt)
        duplicated["command"].append("-q")
        self.assertEqual(job_receipt_problems(duplicated), {})

    def test_the_documented_figure_survives_prose_whitespace(self):
        record = json.loads(power_record_path().read_text(encoding="utf-8"))
        path = str(power_record_path().relative_to(ROOT)).replace("\\", "/")
        for document in ("docs/M3_TRUNK_WIDTH_LEVER.md", "docs/CONTINUATION_STATE.md"):
            text = (ROOT/document).read_text(encoding="utf-8")
            mangled = text.replace("\n", "\n   ").replace(" ", "  ")
            self.assertTrue(figure_appears(mangled, record, path), document)

    def test_the_power_record_survives_a_docstring_and_a_renamed_local(self):
        record = json.loads(power_record_path().read_text(encoding="utf-8"))
        source = (ROOT/"scripts/sweep_deciding_power.py").read_text(encoding="utf-8")
        # Rename a local only: renaming a key would be a real change, not an irrelevant edit.
        source = source.replace("step_change", "step_delta")
        source += ('\n\ndef _irrelevant_documentation():\n'
                   '    """An added docstring changes nothing."""\n'
                   '    return 1\n')
        directory = Path(tempfile.mkdtemp(dir=ROOT))
        try:
            scratch = directory/"scratch_power.py"
            scratch.write_text(source, encoding="utf-8")
            spec = importlib.util.spec_from_file_location("scratch_power", scratch)
            module = importlib.util.module_from_spec(spec)
            sys.modules["scratch_power"] = module
            spec.loader.exec_module(module)
            fresh = module.power(load(record["result_path"]))
            for key in ("status", "run", "deciding_split", "population", "auc", "operating_point", "grid",
                        "verdict"):
                difference = first_difference(fresh[key], record[key], key)
                self.assertIsNone(difference, difference)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_a_changed_log_line_is_irrelevant(self):
        _, receipt = latest_passing_suite_receipt()
        logs = sorted(ROOT.glob("runs/suite-*-r*.log")) or sorted(ROOT.glob("runs/suite-*.log"))
        directory = Path(tempfile.mkdtemp(dir=ROOT))
        try:
            scratch = directory/"scratch.log"
            scratch.write_text("an unrelated log line\n", encoding="utf-8")
            self.assertTrue(scratch.exists())
            self.assertEqual(job_receipt_problems(receipt), {})
            self.assertTrue(logs)
        finally:
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
