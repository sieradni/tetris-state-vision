"""Pure acceptance/denominator contract for the declared32-image fit population."""
from collections import Counter
import math

CLASSES = tuple("IOTSZJL") + ("absent",)


def summarize_fit(records):
    if type(records) is not list or len(records) != 32:
        raise ValueError("complete32-image fit population required")
    if (any(type(row) is not dict or type(row.get("sample_id")) is not str or type(row.get("group_id")) is not str for row in records) or
            len({row["sample_id"] for row in records}) != 32 or Counter(Counter(row["group_id"] for row in records).values()) != {2: 16}):
        raise ValueError("unique32 views in16 paired ancestries required")
    roles = {role: {label: {"known": 0, "exact": 0} for label in CLASSES} for role in ("active", "ghost")}
    known_semantic = correct_semantic = exact_boards = exact_layers = observed_exact = absent_exact = 0
    geometry = []
    for row in records:
        if (type(row.get("exact_locked_board")) is not bool or type(row.get("known_cells")) is not int or row["known_cells"] != 200 or
                type(row.get("mean_geometry_error_cells")) not in (int, float) or
                not math.isfinite(row["mean_geometry_error_cells"]) or row["mean_geometry_error_cells"] < 0):
            raise ValueError("supported finite board/geometry metrics required")
        fit = row["fit_case"]
        known, correct = fit["semantic_known"], fit["semantic_correct"]
        if type(known) is not int or type(correct) is not int or not 0 <= correct <= known <= 200:
            raise ValueError("supported semantic counts required")
        known_semantic += known
        correct_semantic += correct
        exact_boards += int(row["exact_locked_board"])
        geometry.append(row["mean_geometry_error_cells"])
        if type(fit["roles"]) is not dict or set(fit["roles"]) != {"active", "ghost"}:
            raise ValueError("independent role metrics required")
        right = 0
        for role, item in fit["roles"].items():
            label = item["class"]
            if (label not in CLASSES or item["known"] is not True or type(item["observed"]) is not bool or
                    item["observed"] != (label != "absent") or type(item["exact"]) is not bool):
                raise ValueError("supported class/absence metrics required")
            roles[role][label]["known"] += 1
            roles[role][label]["exact"] += int(item["exact"])
            right += int(item["exact"])
            if item["observed"]:
                observed_exact += int(item["exact"])
            else:
                absent_exact += int(item["exact"])
        if (type(row["known_layers"]) is not int or row["known_layers"] != 2 or
                type(row["exact_known_layers"]) is not int or row["exact_known_layers"] != right):
            raise ValueError("role aggregates differ")
        exact_layers += right
    if known_semantic != 128 or any(value["known"] != 4 for labels in roles.values() for value in labels.values()):
        raise ValueError("declared all-class/semantic population differs")
    mean_geometry = sum(geometry) / len(geometry)
    passed = (exact_boards == 32 and exact_layers == 64 and correct_semantic == 128 and mean_geometry < .25)
    return {"passed": passed, "images": 32, "ancestry_groups": 16, "exact_locked_boards": exact_boards,
            "known_layers": 64, "exact_known_layers": exact_layers,
            "observed_layers": 56, "exact_observed_layers": observed_exact,
            "absent_layers": 8, "exact_absent_layers": absent_exact,
            "known_semantic_cells": known_semantic, "correct_semantic_cells": correct_semantic,
            "mean_geometry_error_cells": mean_geometry, "roles_by_target_class": roles,
            "criterion": "32 exact boards;64 exact supported roles including56 observed;128 correct supported semantic cells;mean geometry<.25 cells.",
            "scope": "Memorization diagnostic only; not independent generalization or real complete-state acceptance."}
