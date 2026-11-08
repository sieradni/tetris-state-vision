"""Pure masked queue/hold diagnostics; no decoded state or inferred hidden length."""

PIECES = tuple("IOTSZJL")
# A supervised piece position may be predicted as the endpoint token; that is an
# emitted class, not an unknown, so it must be tallied rather than rejected.
QUEUE_CLASSES = PIECES+("EOS",)
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


def _position_spread(tokens, mask):
    """Largest distance of one supervised position's logits from their mean.

    Zero means every supervised position received identical logits, so the position
    axis carried no information into the decision. A positive value proves the
    position axis reaches the logits even when every argmax is still one class.
    """
    rows = [row for row, known in zip(tokens, mask) if known]
    if len(rows) < 2 or not rows[0]:
        return 0.0
    mean = [sum(row[index] for row in rows)/len(rows) for index in range(len(rows[0]))]
    return max(sum((row[index]-mean[index])**2 for index in range(len(mean)))**.5 for row in rows)


def pack_components(output, target):
    """One-image tensor outputs/targets to bounded plain diagnostic values."""
    prediction = {key: output[key][0].argmax(-1).tolist() for key in
                  ("queue_tokens", "queue_slot_status", "queue_status", "hold_status", "hold_piece")}
    prediction["queue_presence"] = (output["queue_presence"][0] > 0).tolist()
    prediction["queue_anchors"] = (output["queue_anchors"][0] > 0).tolist()
    prediction["queue_position_spread"] = _position_spread(output["queue_tokens"][0].tolist(),
                                                           target["queue_token_known"][0].tolist())
    truth = {key: target[key][0].tolist() for key in
             ("queue_tokens", "queue_token_known", "queue_slot_status", "queue_slot_status_known",
              "queue_presence", "queue_presence_known", "queue_status", "queue_start_observed",
              "queue_end_observed", "hold_status", "hold_piece", "hold_piece_known")}
    truth["capacity"] = target["capacity"]
    return prediction, truth


def _counts(predicted, truth, mask):
    errors = [index for index, known in enumerate(mask) if known and predicted[index] != truth[index]]
    return {"known": sum(mask), "correct": sum(mask)-len(errors), "error_indices": errors}


def _predicted_classes(predicted, mask, names):
    """Supervised positions only: which classes the head actually emits.

    A single-class distribution means the head is a constant predictor, so any
    exact/supervised ratio it earns is that class's frequency in the targets and
    not class discrimination. Unknown positions never enter the tally.
    """
    counts = {}
    for index, known in enumerate(mask):
        if known:
            if not _index(predicted[index], len(names)):
                raise ValueError("predicted identity class outside the declared vocabulary")
            name = names[predicted[index]]
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items())), len(counts), sum(counts.values())


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
    spread = prediction.get("queue_position_spread")
    if spread is not None and (type(spread) not in (int, float) or spread != spread or not 0 <= spread < 1e12):
        raise ValueError("position-axis spread must be a finite nonnegative number")
    queue_identity, queue_distinct, queue_supervised = _predicted_classes(prediction["queue_tokens"], piece_mask, QUEUE_CLASSES)
    hold_identity = PIECES[prediction["hold_piece"]] if target["hold_piece_known"] else None
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
    identity = {"queue_predicted_classes": queue_identity, "queue_supervised_positions": queue_supervised,
                "queue_distinct_predicted_classes": queue_distinct,
                "queue_constant_predictor": bool(queue_supervised > 1 and queue_distinct == 1),
                "queue_position_logit_spread": spread,
                "hold_predicted_piece": hold_identity}
    return {"capacity": capacity, "queue_supported_exact": queue_right, "hold_supported_exact": hold_right,
            "supported_components_exact": queue_right and hold_right, "component_state_readable": readable,
            "exact_readable_components": queue_right and hold_right if readable else None,
            "identity": identity,
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
        # Identity discrimination: what the heads emit, not only how often they are right.
        queue_predicted, hold_predicted = {}, {}
        for row in rows:
            identity = row.get("identity")
            if identity is None:
                continue
            for name, count in identity["queue_predicted_classes"].items():
                queue_predicted[name] = queue_predicted.get(name, 0) + count
            if identity["hold_predicted_piece"]:
                hold_predicted[identity["hold_predicted_piece"]] = \
                    hold_predicted.get(identity["hold_predicted_piece"], 0) + 1
        values["queue"]["predicted_classes"] = dict(sorted(queue_predicted.items()))
        values["queue"]["distinct_predicted_classes"] = len(queue_predicted)
        values["queue"]["constant_predictor"] = bool(sum(queue_predicted.values()) > 1 and len(queue_predicted) == 1)
        values["hold"]["predicted_classes"] = dict(sorted(hold_predicted.items()))
        values["hold"]["distinct_predicted_classes"] = len(hold_predicted)
        values["hold"]["constant_predictor"] = bool(sum(hold_predicted.values()) > 1 and len(hold_predicted) == 1)
        spreads = [row["identity"]["queue_position_logit_spread"] for row in rows
                   if (row.get("identity") or {}).get("queue_position_logit_spread") is not None]
        values["queue"]["position_logit_spread_max"] = max(spreads) if spreads else None
        values["queue"]["position_logit_spread_mean"] = sum(spreads)/len(spreads) if spreads else None
        values["queue"]["position_axis_reaches_logits"] = bool(spreads and max(spreads) > 0)
        result[split] = values
    return result
