"""Pure mask-first accounting of the targets actually supplied to a trainer."""


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def pack_layers(raw):
    require(all(type(raw[key]) is list and len(raw[key]) == 2 for key in
                ("layer_known", "pose_known", "layer_type", "column", "row", "rotations")),
            "two independent target slots required")
    result = {}
    for index, role in enumerate(("active", "ghost")):
        known, posed = raw["layer_known"][index], raw["pose_known"][index]
        require(type(known) is bool and type(posed) is bool, "explicit layer masks required")
        require(not posed or known, "unknown layer cannot have pose supervision")
        label, pose = None, None
        if known:
            label = raw["layer_type"][index]
            require(type(label) is int and 0 <= label < 8, "known layer type outside eight-class dialect")
            if posed:
                column, row, rotations = raw["column"][index], raw["row"][index], raw["rotations"][index]
                require(label > 0 and type(column) is int and 0 <= column < 10 and
                        type(row) is int and 0 <= row < 26 and type(rotations) is list and
                        len(rotations) == 4 and all(type(value) is bool for value in rotations) and any(rotations),
                        "pose requires supported piece, coordinates and equivalent rotations")
                pose = {"column": column, "row": row, "rotations": list(rotations)}
        result[role] = {"status": "unknown" if not known else "absent" if label == 0 else "observed",
                        "type": label, "pose": pose}
    return result


def summarize(rows):
    require(type(rows) is list and 1 <= len(rows) <= 4141, "finite1..4141 target records required")
    populations = {}
    seen = set()
    for record in rows:
        population = record["population"]
        require(population in {"training_updates", "training_probes", "development_probes"}, "explicit exposure population required")
        sample, group = record["sample_id"], record["group_id"]
        require(type(sample) is str and sample and type(group) is str and group, "sample/ancestry identity required")
        key = (population, record["step"] if population == "training_updates" else sample)
        require(type(record["step"]) is int and 1 <= record["step"] <= 4096 if population == "training_updates"
                else record["step"] is None, "explicit finite update cursor required")
        require(key not in seen, "duplicate target record")
        seen.add(key)
        item = populations.setdefault(population, {"targets": 0, "groups": set(), "layers": {}})
        item["targets"] += 1
        item["groups"].add(group)
        require(set(record["layers"]) == {"active", "ghost"}, "independent active and ghost targets required")
        for role in ("active", "ghost"):
            layer = record["layers"][role]
            status, label, pose = layer["status"], layer["type"], layer["pose"]
            require(set(layer) == {"status", "type", "pose"}, "packed layer fields differ")
            if status == "unknown":
                require(label is None and pose is None, "unknown target cannot retain padded labels")
            else:
                require(status in {"absent", "observed"} and type(label) is int and 0 <= label < 8 and
                        (status == "absent") == (label == 0), "supported status/type differs")
            count = item["layers"].setdefault(role, {"unknown": 0, "absent": 0, "observed": 0,
                "type_counts": [0]*8, "groups_by_type": [set() for _ in range(8)], "pose_targets": 0,
                "column_counts": [0]*10, "row_counts": [0]*26, "allowed_rotation_counts": [0]*4})
            count[status] += 1
            if label is not None:
                count["type_counts"][label] += 1
                count["groups_by_type"][label].add(group)
            if pose is not None:
                require(status == "observed" and set(pose) == {"column", "row", "rotations"} and
                        type(pose["column"]) is int and 0 <= pose["column"] < 10 and
                        type(pose["row"]) is int and 0 <= pose["row"] < 26 and
                        type(pose["rotations"]) is list and len(pose["rotations"]) == 4 and
                        all(type(value) is bool for value in pose["rotations"]) and any(pose["rotations"]),
                        "supported pose dialect differs")
                count["pose_targets"] += 1
                count["column_counts"][pose["column"]] += 1
                count["row_counts"][pose["row"]] += 1
                for index, allowed in enumerate(pose["rotations"]):
                    count["allowed_rotation_counts"][index] += int(allowed)
    for item in populations.values():
        item["ancestry_groups"] = len(item.pop("groups"))
        for count in item["layers"].values():
            count["unique_group_type_exposures"] = [len(groups) for groups in count.pop("groups_by_type")]
    return populations
