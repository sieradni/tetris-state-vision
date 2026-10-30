"""Descriptive readout diagnostics; labels enter analysis, never model inputs."""
from collections import Counter
import math


def analyze(rows):
    if not 1 <= len(rows) <= 136:
        raise ValueError("finite feature population required")
    dimensions, identities = set(), set()
    for row in rows:
        identity = (row["split"], row["sample_id"], row["role"])
        if identity in identities or row["role"] not in ("active", "ghost"):
            raise ValueError("unique role/sample required")
        identities.add(identity)
        if (row["split"] not in ("training", "development") or not row["group_id"] or
                type(row["known"]) is not bool or type(row["target_type"]) is not int or
                not 0 <= row["target_type"] < 8 or len(row["logits"]) != 8 or
                not 1 <= len(row["features"]) <= 256 or
                any(type(x) not in (int, float) or not math.isfinite(x)
                    for x in row["features"] + row["logits"])):
            raise ValueError("invalid finite feature/label contract")
        dimensions.add(len(row["features"]))
    if len(dimensions) != 1:
        raise ValueError("feature dimensions differ")
    output = {}
    for role in ("active", "ghost"):
        role_rows = [row for row in rows if row["role"] == role]
        training = [row for row in role_rows if row["split"] == "training" and row["known"]]
        role_result = {}
        for split in ("training", "development"):
            selected = [row for row in role_rows if row["split"] == split and row["known"]]
            cases = []
            for row in selected:
                # Remove EVERY view of the query ancestry, even across split labels.
                references = [other for other in training if other["group_id"] != row["group_id"]]
                classes = sorted({other["target_type"] for other in references})
                predicted = None
                distances = {}
                if row["target_type"] in classes:
                    for label in classes:
                        examples = [other["features"] for other in references if other["target_type"] == label]
                        center = [math.fsum(values)/len(examples) for values in zip(*examples)]
                        distances[str(label)] = math.fsum((x-y)**2 for x, y in zip(row["features"], center))
                    minimum = min(distances.values())
                    winners = [int(label) for label, distance in distances.items() if distance == minimum]
                    # Ties abstain; a stable index is not class information.
                    predicted = winners[0] if len(winners) == 1 else None
                native = max(range(8), key=lambda i: row["logits"][i])
                cases.append({"sample_id": row["sample_id"], "group_id": row["group_id"],
                              "target_type": row["target_type"], "native_type": native,
                              "prototype_type": predicted, "reference_classes": classes,
                              "eligible": row["target_type"] in classes,
                              "squared_distances": distances})
            eligible = [case for case in cases if case["eligible"]]
            role_result[split] = {"known": len(selected), "unknown_excluded": sum(
                not row["known"] for row in role_rows if row["split"] == split),
                "ancestries": len({row["group_id"] for row in selected}),
                "target_counts": dict(sorted(Counter(row["target_type"] for row in selected).items())),
                "native_correct": sum(case["native_type"] == case["target_type"] for case in cases),
                "prototype_eligible": len(eligible),
                "prototype_correct": sum(case["prototype_type"] == case["target_type"] for case in eligible),
                "prototype_ties": sum(case["prototype_type"] is None for case in eligible), "cases": cases}
        output[role] = role_result
    return {"dimension": dimensions.pop(), "roles": output,
            "scope": "Descriptive Euclidean class-centroid readout of frozen features; all views of query ancestry excluded. Training images were used to train the network. No independent generalization, causal bottleneck, calibrated confidence or replacement decoder claim. Missing reference classes are ineligible; exact ties abstain. Unknown targets are excluded."}
