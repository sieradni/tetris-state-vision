"""Strict, type-aware field-by-field equivalence between a historical replay receipt and a new one.

The seven protected replay fields are read through declared carriers and compared with `first_difference`,
which is strict about type (bool is not int, int is not float), length, key sets and order, and treats NaN as
equal only to NaN. A carrier missing on either side is a difference. Any top-level section that is neither
compared nor listed as metadata is reported as `unaccounted:<key>`, so a newly added section cannot ship
unguarded. Fields an instrument does not record are declared absent with the keys that would prove it started
recording them; if those keys ever appear, the comparison fails instead of passing quietly.

`tests/test_replay_equivalence.py` attacks every carried field with near-miss perturbations and requires the
comparison to name that field.
"""
import math

PROTECTED_FIELDS = ("verdict", "attribution", "f1", "threshold", "population", "weights", "cases")
MISSING = object()


def first_difference(before, after, path=""):
    """Describe the first strict difference between two values, or return None.

    Strict: types must match exactly, dict key sets must match exactly, sequences must have the same length
    and order, NaN equals only NaN, and infinities must match sign.
    """
    label = path or "value"
    if type(before) is not type(after):
        return "%s: type %s != %s" % (label, type(before).__name__, type(after).__name__)
    if isinstance(before, float):
        before_nan, after_nan = math.isnan(before), math.isnan(after)
        if before_nan or after_nan:
            return None if (before_nan and after_nan) else "%s: nan mismatch (%r != %r)" % (label, before, after)
        return None if before == after else "%s: %r != %r" % (label, before, after)
    if isinstance(before, dict):
        missing = sorted(set(before) - set(after))
        if missing:
            return "%s: missing keys %s" % (label, missing)
        extra = sorted(set(after) - set(before))
        if extra:
            return "%s: unexpected keys %s" % (label, extra)
        for key in before:
            child = "%s.%s" % (path, key) if path else str(key)
            difference = first_difference(before[key], after[key], child)
            if difference:
                return difference
        return None
    if isinstance(before, (list, tuple)):
        if len(before) != len(after):
            return "%s: length %d != %d" % (label, len(before), len(after))
        for index, (left, right) in enumerate(zip(before, after)):
            difference = first_difference(left, right, "%s[%d]" % (path, index))
            if difference:
                return difference
        return None
    return None if before == after else "%s: %r != %r" % (label, before, after)


def _cases(receipt):
    cases = receipt["cases"]
    if not isinstance(cases, list) or not cases:
        raise KeyError("cases")
    return cases


def _errors_f1(receipt):
    totals = receipt["totals"]
    if not isinstance(totals, dict):
        raise KeyError("totals")
    pooled, evidence = {}, {}
    for split, block in totals.items():
        populations = block.get("locked_populations") if isinstance(block, dict) else None
        if not isinstance(populations, dict):
            raise KeyError("totals." + split + ".locked_populations")
        for name, counts in populations.items():
            key = split + "/" + name
            numerator = 2*counts["true_positive"]
            denominator = numerator+counts["false_positive"]+counts["false_negative"]
            pooled[key] = numerator/denominator if denominator else None
            # The counts are part of the F1 evidence: a type drift here must be visible in this field too.
            evidence[key] = [counts["true_positive"], counts["false_positive"], counts["false_negative"],
                             counts["true_negative"]]
    return {"pooled": pooled, "counts": evidence}


CARRIERS = {
    "headroom": {
        "verdict": ("verdict.label", lambda receipt: receipt["verdict"]["label"]),
        "attribution": ("agreement_at_zero", lambda receipt: receipt["agreement_at_zero"]),
        "f1": ("verdict.ensemble_best_f1/ensemble.best.f1/ensemble.f1_at_zero",
               lambda receipt: (receipt["verdict"]["ensemble_best_f1"], receipt["ensemble"]["best"]["f1"],
                                receipt["ensemble"]["f1_at_zero"])),
        "threshold": ("ensemble.best.threshold", lambda receipt: receipt["ensemble"]["best"]["threshold"]),
        "population": ("population", lambda receipt: receipt["population"]),
        "weights": ("weights_sha256", lambda receipt: receipt["weights_sha256"]),
        "cases": ("cases", _cases),
    },
    "sweep": {
        "verdict": ("verdict", lambda receipt: receipt["verdict"]),
        "attribution": None,
        "f1": ("deciding.best.f1/development.best.f1",
               lambda receipt: (receipt["deciding"]["best"]["f1"], receipt["development"]["best"]["f1"])),
        "threshold": ("deciding.best.threshold/development.best.threshold",
                      lambda receipt: (receipt["deciding"]["best"]["threshold"],
                                       receipt["development"]["best"]["threshold"])),
        "population": ("training_population", lambda receipt: receipt["training_population"]),
        "weights": ("weights_sha256", lambda receipt: receipt["weights_sha256"]),
        "cases": ("cases", _cases),
    },
    "errors": {
        "verdict": ("run_status/automatic_metrics_reproduced/weights_unchanged",
                    lambda receipt: (receipt["run_status"], receipt["automatic_metrics_reproduced"],
                                     receipt["weights_unchanged"])),
        "attribution": None,
        "f1": ("totals locked-population F1", _errors_f1),
        "threshold": None,
        "population": ("training_population", lambda receipt: receipt["training_population"]),
        "weights": ("weights_sha256", lambda receipt: receipt["weights_sha256"]),
        "cases": ("cases", _cases),
    },
}

CARRIER_ROOTS = {
    "headroom": {"verdict", "agreement_at_zero", "ensemble", "population", "weights_sha256", "cases"},
    "sweep": {"verdict", "deciding", "development", "training_population", "weights_sha256", "cases"},
    "errors": {"run_status", "automatic_metrics_reproduced", "weights_unchanged", "totals",
               "training_population", "weights_sha256", "cases"},
}

SUPPORTING = {
    "headroom": ("status", "arms", "members", "deciding", "grid", "ensemble"),
    "sweep": ("status", "run", "run_status", "deciding", "development", "grid", "component_capacity",
              "curriculum"),
    "errors": ("status", "run", "totals", "development_extension", "component_capacity",
               "component_identity_factor",
               "component_endpoint_term", "curriculum", "global_steps", "target_steps", "occupancy_factor",
               "diagnostic_format_version", "label_inputs_to_model", "source_catalogue_sha256"),
}

METADATA = ("sources", "elapsed_seconds", "admission", "scope", "forwards", "job_ceiling_bytes")
IDENTITY_KEYS = ("config", "core_architecture")

ABSENT = {
    "headroom": {},
    "sweep": {
        "attribution": {
            "reason": "the sweep has no attribution table: its rule scores pooled AUC and swept F1 only, and "
                      "attribution is measured by the headroom instrument and the width lever",
            "forbidden": ("attribution",),
        },
    },
    "errors": {
        "attribution": {
            "reason": "the error probe replays per-case diagnostics; it has no lever-attribution table",
            "forbidden": ("attribution",),
        },
        "threshold": {
            "reason": "the error probe reports no swept decision threshold or threshold grid",
            "forbidden": ("threshold", "thresholds"),
        },
    },
}

ERRORS_POPULATION = {"recorded_groups": 3, "evaluated_groups": 3, "evaluated_cases": 9,
                     "extension_cases": 0, "base_seed": 20261026}

DOCUMENTED_LATER = {"errors": {"occupancy_positive_weight": 1}}
DOCUMENTED_CARRIERS = {"errors": {"population": ERRORS_POPULATION}}


def _contains_key(value, keys):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in keys or _contains_key(child, keys):
                return True
    elif isinstance(value, list):
        return any(_contains_key(child, keys) for child in value)
    return False


def _short(value):
    text = repr(value)
    return text if len(text) <= 140 else text[:137] + "..."


def compare(instrument, historical, current):
    """Return every protected-field difference; absence is a difference, never a pass."""
    if instrument not in CARRIERS:
        raise ValueError("unknown replay instrument: " + str(instrument))
    differences, compared = {}, []
    for field in PROTECTED_FIELDS:
        carrier = CARRIERS[instrument][field]
        if carrier is None:
            forbidden = ABSENT[instrument][field]["forbidden"]
            hits = [name for name in forbidden
                    if _contains_key(historical, (name,)) or _contains_key(current, (name,))]
            if hits:
                differences[field] = ("%s: instrument now records %s; declare a carrier instead of an absent "
                                      "note" % (field, ", ".join(hits)))
            continue
        path, extract = carrier
        compared.append(field)
        try:
            before = extract(historical)
        except (KeyError, TypeError, IndexError) as error:
            expected = DOCUMENTED_LATER.get(instrument, {}).get(field, MISSING)
            if expected is MISSING:
                expected = DOCUMENTED_CARRIERS.get(instrument, {}).get(field, MISSING)
            if expected is MISSING:
                differences[field] = path + ": absent from the historical receipt (" + str(error) + ")"
                continue
            try:
                after = extract(current)
            except (KeyError, TypeError, IndexError):
                after = MISSING
            difference = first_difference(after, expected, path)
            if difference:
                differences[field] = difference + " (documented default)"
            continue
        try:
            after = extract(current)
        except (KeyError, TypeError, IndexError) as error:
            differences[field] = path + ": absent from the current receipt (" + str(error) + ")"
            continue
        difference = first_difference(before, after, path)
        if difference:
            differences[field] = difference
    for key in SUPPORTING[instrument]:
        present_before, present_after = key in historical, key in current
        if present_before != present_after:
            differences["supporting:" + key] = "section present on one side only"
            continue
        if not present_before:
            continue
        difference = first_difference(historical[key], current[key], key)
        if difference:
            differences["supporting:" + key] = difference
    documented = {}
    for field, expected in DOCUMENTED_LATER.get(instrument, {}).items():
        actual = current.get(field, MISSING)
        if field in historical:
            difference = first_difference(actual, historical[field], field)
        else:
            difference = first_difference(actual, expected, field)
        if difference:
            differences[field] = difference + " (documented default)"
        documented[field] = {"expected": expected,
                             "recorded": None if actual is MISSING else actual}
    identity = {}
    for key in IDENTITY_KEYS:
        if key in historical or key in current:
            difference = first_difference(historical.get(key, MISSING), current.get(key, MISSING), key)
            identity[key] = difference is None
            if difference:
                differences["identity:" + key] = difference
    accounted = (set(CARRIER_ROOTS[instrument]) | set(SUPPORTING[instrument]) | set(METADATA) |
                 set(IDENTITY_KEYS) | set(DOCUMENTED_LATER.get(instrument, {})))
    unaccounted = sorted((set(historical) | set(current)) - accounted)
    for key in unaccounted:
        differences["unaccounted:" + key] = "top-level section neither compared nor declared"
    unchecked = {field: spec["reason"] for field, spec in ABSENT[instrument].items()
                 if field not in differences}
    return {"instrument": instrument, "identical": not differences, "differences": differences,
            "fields_compared": compared, "unchecked": unchecked, "unaccounted": unaccounted,
            "documented": documented, "identity": identity}
