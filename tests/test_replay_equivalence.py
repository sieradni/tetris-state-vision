"""Adversarial equivalence tests: the differ must reject near misses, not only large edits.

Every carried field is attacked with ULP-sized float changes, 1e-12 nudges, type substitutions (int/float,
bool/int, string for number), NaN and infinity, one-sided keys, dropped or duplicated cases, and nested changes.
Each attack must still name its protected field. Reordering is order-sensitive for the case list and
order-insensitive for keyed maps, and both expectations are pinned. Unmodified pairs must compare clean, an
undeclared top-level section must fail the comparison, and the declared-absent fields must fail if the
instrument starts recording them.
"""
import copy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"scripts")]

from replay_equivalence import ABSENT, PROTECTED_FIELDS, compare, first_difference

PAIRS = {
    "headroom": ("docs/evidence/m3-deciding-headroom-r1/result.json",
                 "docs/evidence/m3-deciding-headroom-rerun-r4/result.json"),
    "sweep": ("runs/m3-occbref-baseline-r1-sweep-r1/result.json",
              "runs/m3-sweep-obcref-rerun-r4/result.json"),
    "errors": ("docs/evidence/m3-fit-occupancy-r1-errors-r1/result.json",
               "docs/evidence/m3-fit-occupancy-r1-errors-r5/result.json"),
}


def load(relative):
    return json.loads((ROOT/relative).read_text(encoding="utf-8"))


def resolve(obj, path):
    for key in path:
        obj = obj[next(iter(obj)) if key is None else key]
    return obj


def assign(receipt, path, value):
    target = receipt
    for key in path[:-1]:
        target = target[next(iter(target)) if key is None else key]
    last = path[-1]
    target[next(iter(target)) if last is None else last] = value


TARGETS = {
    "headroom": {
        "verdict": [(("verdict", "label"), "str")],
        "attribution": [(("agreement_at_zero", "both_right"), "int")],
        "f1": [(("verdict", "ensemble_best_f1"), "float"), (("ensemble", "best", "f1"), "float")],
        "threshold": [(("ensemble", "best", "threshold"), "float")],
        "population": [(("population", "training_groups"), "int")],
        "weights": [(("weights_sha256", None), "str")],
        "cases": [],
    },
    "sweep": {
        "verdict": [(("verdict",), "str")],
        "f1": [(("deciding", "best", "f1"), "float")],
        "threshold": [(("deciding", "best", "threshold"), "float")],
        "population": [(("training_population", "evaluated_groups"), "int")],
        "weights": [(("weights_sha256",), "str")],
        "cases": [],
    },
    "errors": {
        "verdict": [(("run_status",), "str"), (("automatic_metrics_reproduced",), "bool")],
        "f1": [(("totals", "training", "locked_populations", "nonempty", "true_positive"), "int")],
        "population": [(("training_population", "recorded_groups"), "int")],
        "weights": [(("weights_sha256",), "str")],
        "cases": [],
    },
}


def scalar_attacks(value, kind):
    attacks = {}
    if kind == "float":
        attacks["ulp"] = math.nextafter(value, math.inf)
        attacks["tiny"] = value + 1e-12
        attacks["nan"] = float("nan")
        attacks["inf"] = math.inf
        attacks["string"] = "%.6f" % value
        attacks["int-type"] = int(value)
    elif kind == "int":
        attacks["ulp"] = value + 1
        attacks["float-type"] = float(value)
        if value in (0, 1):
            attacks["bool-type"] = bool(value)
    elif kind == "bool":
        attacks["int-type"] = int(value)
    elif kind == "str":
        attacks["nested"] = value + "x"
    return attacks


def case_attacks(receipt):
    sample = receipt["cases"][0]

    def numeric_path(obj, prefix):
        if isinstance(obj, dict):
            for key, child in obj.items():
                found = numeric_path(child, prefix + [key])
                if found:
                    return found
        elif isinstance(obj, list):
            for index, child in enumerate(obj):
                found = numeric_path(child, prefix + [index])
                if found:
                    return found
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            return prefix
        return None

    leaf = numeric_path(sample, [])

    def nested(receipt):
        target = receipt["cases"][0]
        for key in leaf[:-1]:
            target = target[key]
        current = target[leaf[-1]]
        target[leaf[-1]] = math.nextafter(current, math.inf) if isinstance(current, float) else current+1

    return {
        "dropped-case": lambda r: r.__setitem__("cases", r["cases"][:-1]),
        "duplicated-case": lambda r: r.__setitem__("cases", r["cases"] + [copy.deepcopy(r["cases"][0])]),
        "extra-case-key": lambda r: r["cases"][0].__setitem__("libre", 1),
        "nested-change": nested,
    }


class ReplayEquivalenceTests(unittest.TestCase):
    def test_unmodified_pairs_compare_clean(self):
        for instrument, (historical, current) in PAIRS.items():
            result = compare(instrument, load(historical), load(current))
            self.assertTrue(result["identical"], (instrument, result["differences"]))
            self.assertEqual(result["unaccounted"], [], instrument)
            self.assertTrue(all(result["identity"].values()), (instrument, result["identity"]))
            self.assertEqual(result["fields_compared"],
                             [field for field in PROTECTED_FIELDS if field in result["fields_compared"]],
                             instrument)
            self.assertEqual(set(result["fields_compared"]) | set(result["unchecked"]),
                             set(PROTECTED_FIELDS), instrument)

    def test_near_miss_attacks_are_rejected_and_name_the_field(self):
        for instrument, (historical, current) in PAIRS.items():
            with self.subTest(instrument=instrument):
                reference, live = load(historical), load(current)
                for field, targets in TARGETS[instrument].items():
                    for path, kind in targets:
                        base = resolve(live, path)
                        for label, value in scalar_attacks(base, kind).items():
                            with self.subTest(instrument=instrument, field=field, attack=label):
                                perturbed = copy.deepcopy(live)
                                assign(perturbed, path, value)
                                result = compare(instrument, reference, perturbed)
                                self.assertFalse(result["identical"])
                                self.assertIn(field, result["differences"], result["differences"])
                    if field == "cases":
                        for label, mutate in case_attacks(live).items():
                            with self.subTest(instrument=instrument, field=field, attack=label):
                                perturbed = copy.deepcopy(live)
                                mutate(perturbed)
                                result = compare(instrument, reference, perturbed)
                                self.assertFalse(result["identical"])
                                self.assertIn(field, result["differences"], result["differences"])

    def test_case_order_is_sensitive_and_map_order_is_not(self):
        for instrument, (historical, current) in PAIRS.items():
            with self.subTest(instrument=instrument):
                reference, live = load(historical), load(current)
                reordered = copy.deepcopy(live)
                reordered["cases"] = list(reversed(reordered["cases"]))
                self.assertIn("cases", compare(instrument, reference, reordered)["differences"])
                shuffled = copy.deepcopy(live)
                for key in ("population", "training_population"):
                    if key in shuffled and isinstance(shuffled[key], dict):
                        items = list(shuffled[key].items())[::-1]
                        shuffled[key] = dict(items)
                self.assertEqual(compare(instrument, reference, shuffled)["differences"], {})

    def test_keyed_map_reordering_is_not_a_difference(self):
        reference, live = load(PAIRS["headroom"][0]), load(PAIRS["headroom"][1])
        shuffled = copy.deepcopy(live)
        shuffled["weights_sha256"] = dict(list(shuffled["weights_sha256"].items())[::-1])
        self.assertEqual(compare("headroom", reference, shuffled)["differences"], {})

    def test_a_missing_carrier_is_a_difference_not_a_skip(self):
        reference, live = load(PAIRS["headroom"][0]), load(PAIRS["headroom"][1])
        del live["ensemble"]["best"]
        result = compare("headroom", reference, live)
        self.assertIn("threshold", result["differences"])
        self.assertIn("f1", result["differences"])

    def test_an_undeclared_top_level_section_fails(self):
        for instrument, (historical, current) in PAIRS.items():
            with self.subTest(instrument=instrument):
                reference, live = load(historical), load(current)
                live = copy.deepcopy(live)
                live["libre_section"] = {"note": "new evidence"}
                result = compare(instrument, reference, live)
                self.assertFalse(result["identical"])
                self.assertIn("unaccounted:libre_section", result["differences"])
                self.assertEqual(result["unaccounted"], ["libre_section"])

    def test_absent_fields_fail_if_their_instrument_starts_recording_them(self):
        sweep_reference, sweep_live = load(PAIRS["sweep"][0]), load(PAIRS["sweep"][1])
        injected = copy.deepcopy(sweep_live)
        injected["deciding"] = dict(injected["deciding"], attribution={"both_right": 1})
        self.assertIn("attribution", compare("sweep", sweep_reference, injected)["differences"])
        errors_reference, errors_live = load(PAIRS["errors"][0]), load(PAIRS["errors"][1])
        injected = copy.deepcopy(errors_live)
        injected["threshold"] = 0.0
        self.assertIn("threshold", compare("errors", errors_reference, injected)["differences"])
        nested = copy.deepcopy(errors_live)
        nested["cases"][0]["threshold"] = 0.0
        self.assertIn("threshold", compare("errors", errors_reference, nested)["differences"])

    def test_documented_defaults_are_strict_about_value_and_type(self):
        reference, live = load(PAIRS["errors"][0]), load(PAIRS["errors"][1])
        for value, label in ((2, "value"), (True, "bool-for-int"), (1.0, "float-for-int")):
            with self.subTest(attack=label):
                perturbed = copy.deepcopy(live)
                perturbed["occupancy_positive_weight"] = value
                result = compare("errors", reference, perturbed)
                self.assertIn("occupancy_positive_weight", result["differences"], label)
        perturbed = copy.deepcopy(live)
        perturbed["training_population"] = dict(perturbed["training_population"], recorded_groups=4)
        self.assertIn("population", compare("errors", reference, perturbed)["differences"])

    def test_every_protected_field_is_carried_and_absences_are_declared(self):
        covered = set()
        for instrument, (historical, current) in PAIRS.items():
            result = compare(instrument, load(historical), load(current))
            covered.update(result["fields_compared"])
            absent = set(PROTECTED_FIELDS) - set(result["fields_compared"])
            self.assertEqual(absent, set(ABSENT[instrument]), instrument)
            for field in absent:
                self.assertTrue(result["unchecked"][field].strip(), (instrument, field))
        self.assertEqual(covered, set(PROTECTED_FIELDS))

    def test_first_difference_is_type_strict(self):
        self.assertIsNone(first_difference(1, 1))
        self.assertIsNotNone(first_difference(1, 1.0))
        self.assertIsNotNone(first_difference(0, False))
        self.assertIsNotNone(first_difference("0.5", 0.5))
        self.assertIsNone(first_difference(float("nan"), float("nan")))
        self.assertIsNotNone(first_difference(float("nan"), 0.0))
        self.assertIsNotNone(first_difference(math.inf, -math.inf))
        self.assertIsNone(first_difference({"a": 1, "b": 2}, {"b": 2, "a": 1}))
        self.assertIsNotNone(first_difference([1, 2], [2, 1]))
        self.assertIsNotNone(first_difference({"a": 1}, {"a": 1, "b": 2}))


if __name__ == "__main__":
    unittest.main()
