"""Pure bounded M2 diagnostics. No model inputs, label changes or parser corrections."""
import math


def _matrix(value, rows, columns, check):
    if (type(value) is not list or len(value) != rows or
            any(type(row) is not list or len(row) != columns for row in value) or
            any(not check(cell) for row in value for cell in row)):
        raise ValueError("diagnostic matrix shape/value differs")
    return value


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1_000_000


def _index(value, stop):
    return type(value) is int and 0 <= value < stop


def _boolean(value):
    return type(value) is bool


def _probability(logit):
    return 1/(1+math.exp(-logit)) if logit >= 0 else math.exp(logit)/(1+math.exp(logit))


def _divide(numerator, denominator):
    return numerator/denominator if denominator else None


def _geometry(prediction, target, registered):
    # Match the frozen loss's axis-normalized 200 centers + four corners.
    if not registered:
        return {"supported": False}
    spacing = (math.hypot(*target[1]), math.hypot(*target[2]))
    if min(spacing) < 1e-6:
        raise ValueError("registered diagnostic lattice is degenerate")
    delta = [[a-b for a, b in zip(p, t)] for p, t in zip(prediction, target)]
    def error(column, row):
        return math.hypot(*[(delta[0][axis]+column*delta[1][axis]+row*delta[2][axis])/spacing[axis]
                            for axis in range(2)])
    corners = [error(c, r) for c, r in ((0, 0), (10, 0), (0, 20), (10, 20))]
    errors = [error(c+.5, r+.5) for r in range(20) for c in range(10)]+corners
    vector_error = lambda vector: math.hypot(vector[0]/spacing[0], vector[1]/spacing[1])
    return {"supported": True, "mean_cells": sum(errors)/len(errors),
            "rms_cells": math.sqrt(sum(value*value for value in errors)/len(errors)),
            "max_cells": max(errors), "corner_errors_cells": corners,
            "origin_error_cells": vector_error(delta[0]),
            "per_column_basis_error_cells": vector_error(delta[1]),
            "per_row_basis_error_cells": vector_error(delta[2])}


def analyze_case(prediction, target):
    """One 10x20 bottom-first board and two independent active/ghost layers.

    Logits use the frozen diagnostic's strict >0 rule. Unknown labels contribute
    no occupancy/type/pose truth. Confidence values are uncalibrated diagnostics.
    """
    logits = _matrix(prediction["occupancy_logits"], 20, 10, _number)
    visible = _matrix(prediction["observability_logits"], 20, 10, _number)
    semantic = _matrix(prediction["semantic"], 20, 10, lambda value: _index(value, 8))
    occupied = _matrix(target["occupied"], 20, 10, _boolean)
    known = _matrix(target["known"], 20, 10, _boolean)
    typed = _matrix(target["typed"], 20, 10, _boolean)
    types = _matrix(target["semantic"], 20, 10, lambda value: _index(value, 8))
    lattice = _matrix(prediction["lattice"], 3, 2, _number)
    truth_lattice = _matrix(target["lattice"], 3, 2, _number)
    if not _number(prediction["registration_logit"]) or not _boolean(target["registered"]):
        raise ValueError("finite registration logit and explicit label required")
    if any(typed[r][c] and not (known[r][c] and occupied[r][c]) for r in range(20) for c in range(10)):
        raise ValueError("typed cells must be supported and occupied")
    if not target["registered"] and any(any(row) for row in known):
        raise ValueError("canonical cell labels require registration")
    confusion = {name: 0 for name in ("true_positive", "true_negative", "false_positive", "false_negative")}
    visibility = {"false_known": 0, "false_unknown": 0}
    row_errors, row_known, errors, confidence = [0]*20, [0]*20, [], []
    typed_count = semantic_correct = joint_type_correct = 0
    predicted_rows = []
    for r in range(20):
        predicted_rows.append("".join("1" if value > 0 else "0" for value in logits[r]))
        for c in range(10):
            predicted = logits[r][c] > 0
            visibility["false_known"] += int(visible[r][c] > 0 and not known[r][c])
            visibility["false_unknown"] += int(visible[r][c] <= 0 and known[r][c])
            if not known[r][c]:
                continue
            row_known[r] += 1
            actual = occupied[r][c]
            name = ("true_positive" if predicted else "false_negative") if actual else ("false_positive" if predicted else "true_negative")
            confusion[name] += 1
            probability = _probability(logits[r][c] if actual else -logits[r][c])
            confidence.append(probability)
            if predicted != actual:
                row_errors[r] += 1
                errors.append({"column": c, "row": r, "target_occupied": actual, "predicted_occupied": predicted,
                               "correct_class_probability": probability})
            if typed[r][c]:
                typed_count += 1
                correct = semantic[r][c] == types[r][c]
                semantic_correct += int(correct)
                joint_type_correct += int(correct and predicted)
    count = sum(row_known)
    positive = confusion["true_positive"]+confusion["false_negative"]
    errors.sort(key=lambda value: (value["correct_class_probability"], value["row"], value["column"]))
    locked = {**confusion, "known_cells": count, "positive_cells": positive, "negative_cells": count-positive,
              "cell_accuracy": _divide(count-len(errors), count),
              "precision": _divide(confusion["true_positive"], confusion["true_positive"]+confusion["false_positive"]),
              "recall": _divide(confusion["true_positive"], positive),
              "exact_supported_cells": not errors if count else None,
              "exact_board": not errors if count == 200 else None,
              "population": ("nonempty" if positive else "empty") if count == 200 else "partial",
              "row_errors_bottom_first": row_errors, "row_known_bottom_first": row_known,
              "predicted_rows_bottom_first": predicted_rows, "worst_errors": errors[:16],
              "minimum_correct_class_probability": min(confidence) if confidence else None,
              "typed_cells": typed_count, "correct_type_head_cells": semantic_correct,
              "correct_joint_occupied_type_cells": joint_type_correct}
    if (type(prediction["layers"]) is not list or len(prediction["layers"]) != 2 or
            type(target["layers"]) is not list or len(target["layers"]) != 2):
        raise ValueError("independent active and ghost slots required")
    layers = {}
    for name, pred, truth in zip(("active", "ghost"), prediction["layers"], target["layers"]):
        if (not _number(pred["known_logit"]) or not _index(pred["type"], 8) or not _index(pred["rotation"], 4) or
                not _index(pred["column"], 10) or not _index(pred["row"], 26) or
                not _boolean(truth["known"]) or not _boolean(truth["pose_known"]) or not _index(truth["type"], 8) or
                not _index(truth["column"], 10) or not _index(truth["row"], 26) or
                type(truth["rotations"]) is not list or len(truth["rotations"]) != 4 or
                not all(_boolean(value) for value in truth["rotations"])):
            raise ValueError("layer diagnostic shape/value differs")
        if truth["pose_known"] and (not truth["known"] or truth["type"] == 0 or not any(truth["rotations"])):
            raise ValueError("known pose requires observed layer and equivalent rotations")
        item = {"target_known": truth["known"], "target_pose_known": truth["pose_known"],
                "target_type": truth["type"] if truth["known"] else None,
                "predicted_type": pred["type"], "predicted_known": pred["known_logit"] > 0,
                "exact": None, "type_correct": None, "column_correct": None, "row_correct": None, "rotation_correct": None}
        if truth["known"]:
            item["type_correct"] = pred["type"] == truth["type"]
            item["exact"] = item["predicted_known"] and item["type_correct"]
            if truth["pose_known"]:
                item.update(column_correct=pred["column"] == truth["column"], row_correct=pred["row"] == truth["row"],
                            rotation_correct=truth["rotations"][pred["rotation"]])
                item["exact"] = item["exact"] and item["column_correct"] and item["row_correct"] and item["rotation_correct"]
        layers[name] = item
    return {"locked": locked, "visibility": visibility, "layers": layers,
            "registration_correct": (prediction["registration_logit"] > 0) == target["registered"],
            "geometry": _geometry(lattice, truth_lattice, target["registered"])}


def recipe_strata(recipe):
    if recipe.get("curriculum") is None:
        return recipe["density"], recipe["skin"]
    if (recipe["curriculum"] not in ("all-class-fit-v1", "all-class-interleaved-v1") or recipe.get("split") != "fit" or
            type(recipe.get("index")) is not int or not 0 <= recipe["index"] < 16 or
            type(recipe.get("view")) is not int or recipe["view"] not in (0, 1)):
        raise ValueError("supported fit recipe strata required")
    return ("sparse" if recipe["index"] % 2 == 0 else "empty"), ("clean" if recipe["view"] == 0 else "no_grid")


def fit_metrics_from_diagnostics(diagnostic):
    """Rebuild fit scores from independently decomposed raw heads and masks."""
    locked = diagnostic["locked"]
    known, correct = locked["typed_cells"], locked["correct_type_head_cells"]
    if type(known) is not int or type(correct) is not int or not 0 <= correct <= known <= 200:
        raise ValueError("supported semantic counts required")
    roles = {}
    for role in ("active", "ghost"):
        item = diagnostic["layers"][role]
        supported = item["target_known"]
        kind = item["target_type"]
        if (type(supported) is not bool or type(item["target_pose_known"]) is not bool or
                supported and (not _index(kind, 8) or type(item["exact"]) is not bool) or
                not supported and (kind is not None or item["exact"] is not None)):
            raise ValueError("supported role decomposition required")
        roles[role] = {"known": supported, "observed": item["target_pose_known"], "exact": item["exact"],
                       "class": (("IOTSZJL"[kind-1] if kind else "absent") if supported else "unknown")}
    return {"semantic_known": known, "semantic_correct": correct, "roles": roles}


def summarize(records, *, max_cases=45):
    """Bounded dependent probes; explicit68-case fit extension, no interval."""
    if type(max_cases) is not int or max_cases not in (45, 68) or type(records) is not list or not 1 <= len(records) <= max_cases:
        raise ValueError("finite explicit45/68 diagnostic case bound required")
    if (any(row["split"] not in {"training", "development"} or type(row["group_id"]) is not str or
            not row["group_id"] or type(row["sample_id"]) is not str or not row["sample_id"] for row in records) or
            len({row["sample_id"] for row in records}) != len(records)):
        raise ValueError("unique sample and explicit split/ancestry identities required")
    results = {}
    for split in ("training", "development"):
        cases = [row for row in records if row["split"] == split]
        if not cases:
            continue
        result = {"images": len(cases), "ancestry_groups": len({row["group_id"] for row in cases})}
        result["locked_populations"] = {}
        for population in ("empty", "nonempty", "partial"):
            locked = [row["diagnostics"]["locked"] for row in cases if row["diagnostics"]["locked"]["population"] == population]
            result["locked_populations"][population] = {"images": len(locked),
                "exact_boards": sum(row["exact_board"] is True for row in locked),
                **{key: sum(row[key] for row in locked) for key in ("known_cells", "true_positive", "true_negative", "false_positive", "false_negative")}}
        result["layers"] = {}
        for layer in ("active", "ghost"):
            known = [row["diagnostics"]["layers"][layer] for row in cases if row["diagnostics"]["layers"][layer]["target_known"]]
            result["layers"][layer] = {"known": len(known), "exact": sum(row["exact"] for row in known),
                "unknown_predictions": sum(not row["predicted_known"] for row in known),
                **{key.removesuffix("_correct")+"_errors": sum(row[key] is False for row in known) for key in ("type_correct", "column_correct", "row_correct", "rotation_correct")}}
        geometry = [row["diagnostics"]["geometry"] for row in cases if row["diagnostics"]["geometry"]["supported"]]
        result["geometry"] = {"supported_images": len(geometry),
            **{key: sum(row[key] for row in geometry)/len(geometry) if geometry else None
               for key in ("mean_cells", "origin_error_cells", "per_column_basis_error_cells", "per_row_basis_error_cells")},
            "worst_point_cells": max(row["max_cells"] for row in geometry) if geometry else None}
        results[split] = result
    return results
