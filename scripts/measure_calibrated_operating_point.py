"""Score a frozen sweep at a per-board threshold derived from the read's own logits.

Read-only re-analysis: for every deciding board the occupancy logits are the read's own output, so a
threshold chosen from that board's distribution uses no label, no swept F1 and no held-out boundary. The
rule is Otsu's criterion - the split that maximises between-class variance - with candidates at the midpoints
of consecutive distinct logits, ties to the lower candidate, and a constant board choosing its own value so
nothing is decided. The same rule is applied to every arm, which is what makes the arms comparable.

No training, no weight change, no decoded state, no accuracy claim. Ranking is untouched by construction: a
scoring rule cannot change AUC, and this instrument reports it unchanged so the record cannot mistake one
for the other.
"""
import argparse
import json
from pathlib import Path
import sys

from measure_row_error import boundaries
from probe_model import require_bounded_job

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 640*1024**2
ARMS = ("control", "wide", "down")
BOARDS = 28
OCCUPIED_CELLS = 667
KNOWN_CELLS = BOARDS*200
RECOVERY_FRACTION = .85


def read(path):
    return json.loads(Path(path).read_text())


def otsu(values):
    """The threshold that maximises between-class variance of `values`, with no labels."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no logits to calibrate")
    distinct = sorted(set(ordered))
    if len(distinct) == 1:
        return distinct[0]
    total = len(ordered)
    total_sum = sum(ordered)
    best = None
    below_count = 0
    below_sum = 0.0
    index = 0
    while index < len(ordered):
        value = ordered[index]
        stop = index
        while stop+1 < len(ordered) and ordered[stop+1] == value:
            stop += 1
        below_count += stop-index+1
        below_sum += value*(stop-index+1)
        index = stop+1
        if below_count == total:
            break
        above_count = total-below_count
        above_sum = total_sum-below_sum
        mean_below = below_sum/below_count
        mean_above = above_sum/above_count
        variance = below_count*above_count*(mean_below-mean_above)**2
        candidate = (value+ordered[index])/2
        if best is None or variance > best[0]:
            best = (variance, candidate)
    if best is None:
        return distinct[0]
    return best[1]


def counts(logits, labels, threshold):
    result = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
    for logit, label in zip(logits, labels):
        predicted = logit > threshold
        if label and predicted:
            result["true_positive"] += 1
        elif label:
            result["false_negative"] += 1
        elif predicted:
            result["false_positive"] += 1
        else:
            result["true_negative"] += 1
    return result


def scores(block):
    precision = block["true_positive"]/max(1, block["true_positive"]+block["false_positive"])
    recall = block["true_positive"]/max(1, block["true_positive"]+block["false_negative"])
    f1 = (2*block["true_positive"]
          /max(1, 2*block["true_positive"]+block["false_positive"]+block["false_negative"]))
    return {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}


def arm_block(result, name):
    config = result["config"]
    rows, columns = config["visible_rows"], config["columns"]
    cases = [case for case in result["cases"]
             if case["split"] == "training" and any(case["occupied"])]
    if len(cases) != BOARDS:
        raise ValueError(f"{name} sweep does not carry the deciding boards")
    zero = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
    calibrated = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
    thresholds, boards = [], []
    row_error = {"zero": [], "calibrated": []}
    labels_total = 0
    for case in cases:
        logits, labels = case["logits"], case["occupied"]
        labels_total += sum(1 for label in labels if label)
        threshold = otsu(logits)
        thresholds.append(threshold)
        for key, value in counts(logits, labels, 0.0).items():
            zero[key] += value
        for key, value in counts(logits, labels, threshold).items():
            calibrated[key] += value
        for key, chosen in (("zero", 0.0), ("calibrated", threshold)):
            decided, occupied = boundaries(logits, labels, rows, columns, chosen)
            board_errors = [got-want for got, want in zip(decided, occupied)
                            if got is not None and want is not None]
            row_error[key].extend(board_errors)
            if key == "zero":
                first = board_errors
        boards.append({"sample_id": case["sample_id"], "group_id": case["group_id"],
                       "threshold": round(threshold, 6),
                       "distance_to_label_best": round(abs(threshold-result["deciding"]["best"]["threshold"]), 6),
                       "f1": scores(counts(logits, labels, threshold))["f1"],
                       "zero_f1": scores(counts(logits, labels, 0.0))["f1"],
                       "board_row_error_zero": (round(sum(first)/len(first), 6) if first else None)})
    if labels_total != OCCUPIED_CELLS:
        raise ValueError(f"{name} sweep decided another population")
    mean = sum(thresholds)/len(thresholds)
    spread = (sum((value-mean)**2 for value in thresholds)/len(thresholds))**.5
    zero_f1 = scores(zero)["f1"]
    calibrated_f1 = scores(calibrated)["f1"]
    return {
        "arm": {"occupancy_deform": config.get("occupancy_deform", 0),
                "occupancy_offset_rows": config.get("occupancy_offset_rows", 0),
                "weights_sha256": result["weights_sha256"], "sweep": result["run"]},
        "zero": {"threshold": 0.0, **scores(zero), **zero},
        "label_best": {"threshold": result["deciding"]["best"]["threshold"],
                       "f1": result["deciding"]["best"]["f1"],
                       "precision": result["deciding"]["best"]["precision"]},
        "calibrated": {"rule": "otsu-logits", **scores(calibrated), **calibrated},
        "f1_gain_of_calibration_over_zero": round(calibrated_f1-zero_f1, 6),
        "recovery_of_the_label_chosen_gain": (
            round((calibrated_f1-zero_f1)/max(1e-9, result["deciding"]["best"]["f1"]-zero_f1), 6)),
        "thresholds": {"mean": round(mean, 6), "sd": round(spread, 6),
                       "min": round(min(thresholds), 6), "max": round(max(thresholds), 6),
                       "boards": len(thresholds)},
        "distance_to_label_best": {
            "mean": round(sum(entry["distance_to_label_best"] for entry in boards)/len(boards), 6),
            "within_half": round(sum(1 for entry in boards
                                     if entry["distance_to_label_best"] <= .5)/len(boards), 6)},
        "row_error": {
            "zero": {"mean": round(sum(row_error["zero"])/len(row_error["zero"]), 6),
                     "columns": len(row_error["zero"])},
            "calibrated": {"mean": round(sum(row_error["calibrated"])/len(row_error["calibrated"]), 6),
                           "columns": len(row_error["calibrated"])}},
        "boards": boards,
    }


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    job = read(ROOT/"runs"/(args.output.name+"-job.json"))
    if job["memory_bytes"] != CEILING_BYTES:
        raise ValueError("calibration ceiling differs from the predeclared 640 MiB")
    if args.output.exists():
        raise ValueError("an evidence directory already exists")
    arms = {}
    for name in ARMS:
        sweep = f"m7-calibration-{name}-r1-sweep-r1"
        result = read(ROOT/"runs"/sweep/"result.json")
        if result.get("status") != "passed" or result.get("run") != f"m7-calibration-{name}-r1":
            raise ValueError(f"{name} sweep is not a passed measurement of its own run")
        if result.get("forwards") != 54 or result["deciding"]["occupied_cells"] != OCCUPIED_CELLS:
            raise ValueError(f"{name} sweep is not the deciding measurement")
        if len(result["deciding"]["thresholds"]) != 81 or 0.0 not in result["deciding"]["thresholds"]:
            raise ValueError(f"{name} sweep did not use the standing grid")
        arms[name] = arm_block(result, name)
    out = {
        "status": "passed",
        "instrument": "m7-calibration-r1",
        "scope": ("Read-only scoring of three frozen sweeps at a per-board threshold derived from the "
                  "read's own logits (Otsu's criterion, no labels, no swept F1, no held-out boundary). "
                  "AUC is unchanged by construction: a scoring rule does not touch ranking. No "
                  "training, no weight change, no decoded state, no accuracy claim."),
        "job_ceiling_bytes": job["memory_bytes"],
        "rule": {"name": "otsu-logits", "label_free": True,
                 "candidates": "midpoints of consecutive distinct logits, ties to the lower candidate",
                 "constant_board": "chooses its own value, so nothing is decided"},
        "arms": arms,
        "population": {"split": "training", "boards": BOARDS, "known_cells": KNOWN_CELLS,
                       "occupied_cells": OCCUPIED_CELLS},
        "sources": {name: f"runs/m7-calibration-{name}-r1-sweep-r1/result.json" for name in ARMS},
    }
    args.output.mkdir(parents=True)
    with (args.output/"result.json").open("x", encoding="utf-8") as stream:
        json.dump(out, stream, indent=2, allow_nan=False)
    print(json.dumps({name: {"zero_f1": arms[name]["zero"]["f1"],
                             "calibrated_f1": arms[name]["calibrated"]["f1"],
                             "thresholds": arms[name]["thresholds"],
                             "row_error": arms[name]["row_error"]} for name in ARMS}, indent=2))


if __name__ == "__main__":
    main()
