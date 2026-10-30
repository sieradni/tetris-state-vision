"""Pure masked queue/hold diagnostics; no decoded state or inferred hidden length."""

PIECES = tuple("IOTSZJL")
QUEUE_STATUSES = ("visible", "absent", "cropped", "unknown")
HOLD_STATUSES = ("observed", "empty", "absent", "cropped", "unknown")


def _index(value, size):
    return type(value) is int and 0 <= value < size


def _vector(value, size, valid):
    if type(value) is not list or len(value) != size or not all(valid(item) for item in value):
        raise ValueError("component diagnostic shape/value differs")
    return value


def replay_keywords(recipe, capacity):
    """Do not send new sampler/target keywords to frozen legacy M2 sources."""
    if type(capacity) is not int or not 0 <= capacity <= 64:
        raise ValueError("bounded component capacity required")
    capture = recipe.get("final_capture", False)
    if type(capture) is not bool or capture != bool(capacity):
        raise ValueError("probe capture policy differs from component configuration")
    return ({"final_capture": True}, {"component_capacity": capacity}) if capacity else ({}, {})


def pack_components(output, target):
    """One-image tensor outputs/targets to bounded plain diagnostic values."""
    prediction = {key: output[key][0].argmax(-1).tolist() for key in
                  ("queue_tokens", "queue_slot_status", "queue_status", "hold_status", "hold_piece")}
    prediction["queue_presence"] = (output["queue_presence"][0] > 0).tolist()
    prediction["queue_anchors"] = (output["queue_anchors"][0] > 0).tolist()
    truth = {key: target[key][0].tolist() for key in
             ("queue_tokens", "queue_token_known", "queue_slot_status", "queue_slot_status_known",
              "queue_presence", "queue_presence_known", "queue_status", "queue_start_observed",
              "queue_end_observed", "hold_status", "hold_piece", "hold_piece_known")}
    truth["capacity"] = target["capacity"]
    return prediction, truth


def _counts(predicted, truth, mask):
    errors = [index for index, known in enumerate(mask) if known and predicted[index] != truth[index]]
    return {"known": sum(mask), "correct": sum(mask)-len(errors), "error_indices": errors}


def analyze_components(prediction, target):
    capacity = target["capacity"]
    if type(capacity) is not int or not 1 <= capacity <= 64:
        raise ValueError("bounded component diagnostic capacity required")
    boolean = lambda value: type(value) is bool
    for key, mask_key, size, classes in (("queue_tokens", "queue_token_known", capacity+1, 8),
                                         ("queue_slot_status", "queue_slot_status_known", capacity, 3)):
        _vector(prediction[key], size, lambda value: _index(value, classes))
        _vector(target[key], size, lambda value: type(value) is int and -1 <= value < classes)
        mask = _vector(target[mask_key], size, boolean)
        if mask != [value >= 0 for value in target[key]]:
            raise ValueError("component uncertainty mask differs")
    _vector(prediction["queue_presence"], capacity, boolean)
    _vector(prediction["queue_anchors"], 2, boolean)
    _vector(target["queue_presence"], capacity, boolean)
    _vector(target["queue_presence_known"], capacity, boolean)
    for key, classes in (("queue_status", 4), ("hold_status", 5), ("hold_piece", 7)):
        if not _index(prediction[key], classes) or not _index(target[key], classes):
            raise ValueError("component diagnostic class outside vocabulary")
    if not all(boolean(target[key]) for key in ("queue_start_observed", "queue_end_observed", "hold_piece_known")):
        raise ValueError("explicit component observation flags required")
    start, end = target["queue_start_observed"], target["queue_end_observed"]
    if not start and any(any(target[key]) for key in ("queue_token_known", "queue_slot_status_known", "queue_presence_known")):
        raise ValueError("unseen start cannot label absolute queue indices")
    eos = [known and value == 7 for known, value in zip(target["queue_token_known"], target["queue_tokens"])]
    if sum(eos) > 1 or any(eos) and not end:
        raise ValueError("EOS requires one observed endpoint")
    if target["hold_piece_known"] != (target["hold_status"] == 0):
        raise ValueError("unobserved hold cannot label piece identity")

    piece_mask = [known and value != 7 for known, value in zip(target["queue_token_known"], target["queue_tokens"])]
    fields = {"piece_tokens": _counts(prediction["queue_tokens"], target["queue_tokens"], piece_mask),
              "eos_tokens": _counts(prediction["queue_tokens"], target["queue_tokens"], eos),
              "slot_status": _counts(prediction["queue_slot_status"], target["queue_slot_status"], target["queue_slot_status_known"]),
              "presence": _counts(prediction["queue_presence"], target["queue_presence"], target["queue_presence_known"])}
    status_right = prediction["queue_status"] == target["queue_status"]
    anchors_right = [pred == truth for pred, truth in zip(prediction["queue_anchors"], (start, end))]
    queue_right = status_right and all(anchors_right) and not any(row["error_indices"] for row in fields.values())
    hold_status_right = prediction["hold_status"] == target["hold_status"]
    hold_piece_right = prediction["hold_piece"] == target["hold_piece"] if target["hold_piece_known"] else None
    hold_right = hold_status_right and hold_piece_right is not False
    identities_known = all(not (known and present) or (slot == 0 and token_known)
                           for known, present, slot, token_known in zip(target["queue_presence_known"],
                               target["queue_presence"], target["queue_slot_status"], target["queue_token_known"]))
    queue_readable = target["queue_status"] == 1 or (target["queue_status"] == 0 and start and end and identities_known and any(eos))
    readable = queue_readable and target["hold_status"] <= 2
    return {"capacity": capacity, "queue_supported_exact": queue_right, "hold_supported_exact": hold_right,
            "supported_components_exact": queue_right and hold_right, "component_state_readable": readable,
            "exact_readable_components": queue_right and hold_right if readable else None,
            "known_queue_tokens": sum(target["queue_token_known"]),
            "known_presence_positions": sum(target["queue_presence_known"]), "hold_identity_known": target["hold_piece_known"],
            "queue": {"target_status": QUEUE_STATUSES[target["queue_status"]],
                      "predicted_status": QUEUE_STATUSES[prediction["queue_status"]],
                      "status_correct": status_right, "start_correct": anchors_right[0], "end_correct": anchors_right[1], **fields},
            "hold": {"target_status": HOLD_STATUSES[target["hold_status"]],
                     "predicted_status": HOLD_STATUSES[prediction["hold_status"]], "status_correct": hold_status_right,
                     "target_piece": PIECES[target["hold_piece"]] if target["hold_piece_known"] else None,
                     "predicted_piece": PIECES[prediction["hold_piece"]] if target["hold_piece_known"] else None,
                     "piece_correct": hold_piece_right}}


def summarize_components(records):
    if (type(records) is not list or not 1 <= len(records) <= 45 or
            any(row["split"] not in {"training", "development"} or not isinstance(row["sample_id"], str) or
                not row["sample_id"] or not isinstance(row["group_id"], str) or not row["group_id"] for row in records) or
            len({row["sample_id"] for row in records}) != len(records)):
        raise ValueError("finite unique component cases with split/ancestry required")
    result = {}
    for split in ("training", "development"):
        rows = [row["diagnostics"]["components"] for row in records if row["split"] == split]
        if not rows:
            continue
        values = {"images": len(rows), "ancestry_groups": len({row["group_id"] for row in records if row["split"] == split}),
                  "readable_cases": sum(row["component_state_readable"] for row in rows),
                  "exact_readable_components": sum(row["exact_readable_components"] is True for row in rows),
                  "supported_components_exact": sum(row["supported_components_exact"] for row in rows),
                  "queue_supported_exact": sum(row["queue_supported_exact"] for row in rows),
                  "hold_supported_exact": sum(row["hold_supported_exact"] for row in rows)}
        values["queue"] = {key+"_errors": sum(not row["queue"][key+"_correct"] for row in rows) for key in ("status", "start", "end")}
        for field in ("piece_tokens", "eos_tokens", "slot_status", "presence"):
            known = sum(row["queue"][field]["known"] for row in rows)
            correct = sum(row["queue"][field]["correct"] for row in rows)
            values["queue"][field] = {"known": known, "correct": correct, "errors": known-correct,
                                       "accuracy": correct/known if known else None}
        hold_known = sum(row["hold_identity_known"] for row in rows)
        hold_correct = sum(row["hold"]["piece_correct"] is True for row in rows)
        values["hold"] = {"status_errors": sum(not row["hold"]["status_correct"] for row in rows),
                          "known_piece_cases": hold_known, "correct_pieces": hold_correct,
                          "piece_accuracy": hold_correct/hold_known if hold_known else None}
        result[split] = values
    return result
