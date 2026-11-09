"""Bounded read-only probe: can the captured pixels recover the committed labels?

`docs/M3_OCCUPANCY_PIXEL_CEILING.md` declares the instrument. This probe replays the 28
nonempty deciding boards of a committed occupancy sweep from that run's frozen archive,
extracts per-cell pixel features (no model, no forward pass), fits a group-held-out logistic
read-out on the committed labels, and repeats the identical pipeline on within-group
shuffled labels as the control. It moves no weight, threshold, gate or label policy.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from pixel_ceiling import (cell_features, ceiling_verdict, evaluate, raster,  # noqa: E402
                           shuffled_labels)
from probe_model import require_bounded_job  # noqa: E402
from record_stream_run import require_owned_run  # noqa: E402
from source_snapshot import verify_source  # noqa: E402

CEILING_BYTES = 640*1024**2
DECIDING_CASES = 36
NONEMPTY_CASES = 28
OCCUPIED_CELLS = 667
EMPTY_CELLS = 4933
ANCESTRY_GROUPS = 12
PROFILE_CASES = 2
SOURCES = tuple("scripts/"+name for name in
                ("probe_pixel_ceiling.py", "pixel_ceiling.py", "probe_model.py",
                 "record_stream_run.py", "source_snapshot.py"))


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-cases", type=int, default=DECIDING_CASES)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--profile", action="store_true")
    mode.add_argument("--admission", type=Path)
    args = parser.parse_args()
    require_owned_run(args.output.name)
    if args.maximum_cases != DECIDING_CASES:
        raise ValueError("the declared ceiling population is thirty-six boards")
    sweep = read(ROOT/"runs"/args.sweep/"result.json")
    sweep_job = read(ROOT/"runs"/(args.sweep+"-job.json"))
    if sweep.get("status") != "passed" or sweep_job.get("status") != "passed" or sweep_job.get("exit_code") != 0:
        raise ValueError("the committed sweep this pass joins did not pass")
    run = ROOT/"runs"/sweep["run"]
    summary = read(run/"summary.json")
    if sweep["weights_sha256"] != summary["checkpoint_sha256"]:
        raise ValueError("the sweep and its run disagree about the measured weights")
    ceiling = read(ROOT/"runs"/(args.output.name+"-job.json"))["memory_bytes"]
    if ceiling != CEILING_BYTES:
        raise ValueError("the probe ceiling differs from the predeclared 640 MiB")
    cases = [row for row in sweep["cases"] if row["split"] == "training"]
    if len(cases) != DECIDING_CASES:
        raise ValueError("thirty-six deciding training boards are required")
    if any(row["known_cells"] != 200 or len(row["logits"]) != 200 for row in cases):
        raise ValueError("every deciding board must carry all two hundred known cells")
    decided = sweep["deciding"]
    if (decided["nonempty_images"], decided["occupied_cells"],
            decided["empty_cells_on_nonempty_boards"]) != (NONEMPTY_CASES, OCCUPIED_CELLS, EMPTY_CELLS):
        raise ValueError("the joined sweep is not the declared deciding population")
    zero = [row for row in decided["rows"] if row["threshold"] == 0.0][0]
    if (zero["true_positive"], zero["false_negative"]) != (OCCUPIED_CELLS-6, 6):
        raise ValueError("the joined sweep's zero row does not reproduce its labels")

    verify_source(run/"code", summary["catalogue_sha256"])
    sys.path[:0] = [str(run/"code"/"scripts"), str(run/"code")]
    sys.dont_write_bytecode = True
    from m2_curriculum import replay_case, sample_for
    from probe_population import RECORDED_TRAINING_GROUPS, training_extension_specs
    from tetris_vision.preprocess import prepare_image

    base_seed = summary["arguments"]["seed"]
    capacity, max_side = summary["config"]["component_capacity"], summary["max_side"]
    curriculum = summary["curriculum"]
    recipes = {row["sample_id"]: row for row in
               (json.loads(line) for line in (run/"training-probe.jsonl").read_text().splitlines())}
    for spec in training_extension_specs(base_seed, RECORDED_TRAINING_GROUPS, ANCESTRY_GROUPS):
        case = sample_for(spec["group_index"], spec["view"], "training", spec["base_seed"],
                          curriculum=curriculum, component_capacity=capacity, max_side=max_side)
        recipes[case.record["sample_id"]] = json.loads(json.dumps(case.record))
    if set(recipes) != {row["sample_id"] for row in cases}:
        raise ValueError("the replayed recipes and the swept boards are not one population")
    saved = {row["sample_id"] for row in
             (json.loads(line) for line in (run/"training-probe.jsonl").read_text().splitlines())}

    def case_features(row):
        recipe = recipes[row["sample_id"]]
        case = replay_case(recipe, split="training", curriculum=curriculum,
                           component_capacity=capacity, max_side=max_side)
        if row["sample_id"] in saved:
            from PIL import Image
            with Image.open(run/"training-probe"/(row["sample_id"]+".png")) as image:
                if image.tobytes() != case.frame.image.tobytes() or image.size != case.frame.image.size:
                    raise ValueError("saved deciding raster differs from its replay")
        prepared = prepare_image(case.frame.image, max_side=max_side)
        board = case.frame.state.board.lattice.transformed(prepared.scale_x, prepared.scale_y)
        matrix = cell_features(raster(case.frame.image), board)
        return {"sample_id": row["sample_id"], "group_id": row["group_id"],
                "known_cells": int(matrix.shape[0]),
                "occupied": sum(row["occupied"]),
                "features_sha256": hashlib.sha256(matrix.astype("<f8").tobytes()).hexdigest()}, matrix

    started = time.monotonic()
    if args.profile:
        args.output.mkdir(parents=True, exist_ok=False)
        profile = [case_features(row) for row in cases[:PROFILE_CASES]]
        (args.output/"summary.json").write_text(json.dumps(
            {"status": "complete", "sweep": args.sweep, "run": sweep["run"],
             "weights_sha256": sweep["weights_sha256"], "catalogue_sha256": summary["catalogue_sha256"],
             "admission_ceiling_bytes": ceiling, "cases": DECIDING_CASES,
             "profile_cases": [row for row, _ in profile],
             "scope": "Resource profile only: two feature extractions, no fit and no claim."},
            indent=2)+"\n", encoding="utf-8")
        print(json.dumps({"status": "complete", "profile": True, "cases": DECIDING_CASES}), flush=True)
        return
    profile_summary = read(args.admission/"summary.json")
    profile_job = read(ROOT/"runs"/(args.admission.name+"-job.json"))
    if (profile_summary.get("status") != "complete" or profile_job.get("status") != "passed" or
            profile_job.get("exit_code") != 0 or
            profile_summary.get("weights_sha256") != sweep["weights_sha256"] or
            profile_summary.get("catalogue_sha256") != summary["catalogue_sha256"] or
            profile_summary.get("cases") != DECIDING_CASES):
        raise ValueError("the profile does not admit this sweep's population")

    args.output.mkdir(parents=True, exist_ok=False)
    measured, matrices, labels, groups, ids, empties = [], [], [], [], [], 0
    for row in cases:
        record, matrix = case_features(row)
        if not any(row["occupied"]):
            empties += 1
            continue
        measured.append(record)
        matrices.append(matrix)
        labels.extend(row["occupied"])
        groups.extend([row["group_id"]]*len(row["occupied"]))
        ids.extend([row["sample_id"]]*len(row["occupied"]))
        (args.output/"progress.json").write_text(json.dumps({"cases": len(measured)})+"\n")
    if (len(measured), empties, sum(labels), len(labels)-sum(labels),
            len({row["group_id"] for row in measured})) != (
            NONEMPTY_CASES, DECIDING_CASES-NONEMPTY_CASES, OCCUPIED_CELLS, EMPTY_CELLS,
            len({row["group_id"] for row in measured})):
        raise ValueError("the deciding population differs from the predeclaration")
    import numpy as np
    features = np.vstack(matrices)
    labels = np.array(labels)
    groups = np.array(groups)
    control_labels = shuffled_labels(labels, groups)
    if int(control_labels.sum()) != OCCUPIED_CELLS:
        raise ValueError("the control permutation changed the class budget")
    real = evaluate(features, labels, groups)
    control = evaluate(features, control_labels, groups)
    verdict = ceiling_verdict(real, control)
    case_records = []
    cursor = 0
    for record in measured:
        cells = record["known_cells"]
        case_records.append({**record,
                             "pixel_logits": [round(float(value), 6)
                                              for value in real["logits"][cursor:cursor+cells]],
                             "control_logits": [round(float(value), 6)
                                                for value in control["logits"][cursor:cursor+cells]]})
        cursor += cells
    result = {
        "status": "passed", "sweep": args.sweep, "run": sweep["run"],
        "weights_sha256": sweep["weights_sha256"], "catalogue_sha256": summary["catalogue_sha256"],
        "job_ceiling_bytes": ceiling, "cases": DECIDING_CASES,
        "population": {"nonempty_boards": NONEMPTY_CASES, "empty_boards": empties,
                       "occupied_cells": int(labels.sum()), "empty_cells": int(len(labels)-labels.sum()),
                       "ancestry_groups": len(set(groups.tolist()))},
        "features": {"per_cell": int(features.shape[1]), "interior": 3, "statistic": "logistic, ridge 1."},
        "measured": {key: real[key] for key in ("pooled_auc", "fold_mean_auc", "operating", "best", "per_fold")},
        "control": {key: control[key] for key in ("pooled_auc", "fold_mean_auc", "operating", "best", "per_fold")},
        "verdict": verdict,
        "model_reference": {"source": args.sweep, "operating": {
            key: zero[key] for key in ("threshold", "true_positive", "false_positive", "false_negative",
                                       "precision", "recall", "f1")},
            "best": {key: decided["best"][key] for key in
                     ("threshold", "true_positive", "false_positive", "false_negative", "precision",
                      "recall", "f1")},
            "auc": decided["auc"], "empty_population": decided["empty_population"]},
        "cases": case_records,
        "sources": {name: digest(ROOT/name) for name in SOURCES},
        "elapsed_seconds": time.monotonic()-started,
        "scope": ("Read-only pixel-only read-out of the committed occupancy labels on the committed "
                  "deciding boards; no model forward, no training, no weight, threshold, gate or label "
                  "change, and no accuracy, real-image or complete-state claim."),
    }
    with (args.output/"result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in ("verdict", "population", "model_reference")}, indent=2))


if __name__ == "__main__":
    main()
