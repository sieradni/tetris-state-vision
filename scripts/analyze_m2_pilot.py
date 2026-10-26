"""Bounded checkpoint diagnostics and controlled-prefix verification.

Only synthetic pilot manifests are supported. No geometry enters model.forward.
"""
import argparse
import json
from pathlib import Path
import sys

from probe_model import require_bounded_job

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def curve(path):
    if path.stat().st_size > 4*1024**2:
        raise ValueError("curve byte budget exceeded")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not 1 <= len(rows) <= 512 or [row["step"] for row in rows] != list(range(1, len(rows)+1)):
        raise ValueError("invalid finite optimizer ledger")
    return [{key: value for key, value in row.items() if key != "elapsed_seconds"} for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_bounded_job()
    import torch
    from PIL import Image, ImageDraw
    from tetris_vision.checkpoint import file_digest, load_checkpoint
    from tetris_vision.losses import diagnostic_metrics, geometry_error, targets_for
    from tetris_vision.preprocess import prepare_image
    from tetris_vision.state import VisionState
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    summary, baseline = (read_json(directory/"summary.json") for directory in (args.run, args.baseline))
    if summary["status"] != "complete" or baseline["status"] != "complete" or summary["threads"] != 1:
        raise ValueError("complete one-thread pilot required")
    manifest = args.run/"manifest.jsonl"
    if manifest.stat().st_size > 1024**2:
        raise ValueError("manifest byte budget exceeded")
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    if len(records) != 12 or not 128 <= summary["arguments"]["max_side"] <= 640:
        raise ValueError("diagnostic sample/input budget exceeded")
    before, after = curve(args.baseline/"curve.jsonl"), curve(args.run/"curve.jsonl")
    controlled = {"identical_manifest": file_digest(manifest) == baseline["manifest_sha256"] == summary["manifest_sha256"],
                  "identical_driver": baseline["script_sha256"] == summary["script_sha256"],
                  "identical_initial_metrics": baseline["initial_training"] == summary["initial_training"],
                  "prefix_steps": len(before), "prefix_exact": after[:len(before)] == before,
                  "baseline_runtime_source_sha256": baseline["runtime_source_sha256"],
                  "run_runtime_source_sha256": summary["runtime_source_sha256"],
                  "note": "Aggregate hash delta from unused runtime.py requires the separately recorded source diff; exact prefix is checked, not assumed."}
    for key in ("seed", "states", "max_side", "threads", "learning_rate"):
        controlled[f"same_{key}"] = baseline["arguments"][key] == summary["arguments"][key]
    if not all(value for key, value in controlled.items() if type(value) is bool):
        raise ValueError("controlled baseline/optimizer prefix did not reproduce")
    model, checkpoint = load_checkpoint(args.run/"checkpoint")
    if checkpoint["weights_sha256"] != summary["checkpoint_sha256"]:
        raise ValueError("summary/checkpoint weight identity differs")
    model.eval()
    cases = []
    for record in records:
        paths = []
        for field, maximum in (("image", 64*1024**2), ("state", 1024**2)):
            path = (args.run/record[field]).resolve()
            if not path.is_relative_to(args.run.resolve()) or path.stat().st_size > maximum or file_digest(path) != record[field+"_sha256"]:
                raise ValueError("pilot asset path/byte identity invalid")
            paths.append(path)
        with Image.open(paths[0]) as image:
            if image.width*image.height > 8_000_000:
                raise ValueError("image pixel budget exceeded")
            prepared = prepare_image(image, max_side=summary["arguments"]["max_side"])
        target = targets_for(VisionState.from_json(paths[1].read_text()), prepared)
        with torch.no_grad():
            output = model(prepared.pixels[None], prepared.valid[None])
        final = output["final"]
        known, truth = target["known"][0], target["occupied"][0].bool()
        predicted = final["occupancy"][0, :20] > 0
        counts = {name: int(mask.sum()) for name, mask in {
            "tp": known & truth & predicted, "fp": known & ~truth & predicted,
            "fn": known & truth & ~predicted, "tn": known & ~truth & ~predicted}.items()}
        typed = target["typed"][0]
        semantic_right = final["semantic"][0, :20].argmax(-1) == target["semantic"][0]
        layers = []
        for index, name in enumerate(("active", "ghost")):
            target_known = bool(target["layer_known"][0, index])
            guessed_type = int(final["layer_type"][0, index].argmax())
            guessed_column = int(final["pose_column"][0, index].argmax())
            guessed_row = int(final["pose_row"][0, index].argmax())
            guessed_rotation = int(final["pose_rotation"][0, index].argmax())
            correct_type = guessed_type == int(target["layer_type"][0, index])
            correct_pose = None
            if bool(target["pose_known"][0, index]):
                correct_pose = guessed_column == int(target["column"][0, index]) and guessed_row == int(target["row"][0, index]) and bool(target["rotations"][0, index, guessed_rotation])
            layers.append({"layer": name, "target_known": target_known,
                           "type_correct": correct_type if target_known else None, "pose_correct": correct_pose,
                           "exact": bool(correct_type and final["layer_known"][0, index] > 0 and correct_pose is not False) if target_known else None,
                           "predicted_type_index": guessed_type, "target_type_index": int(target["layer_type"][0, index]) if target_known else None,
                           "predicted_pose": [guessed_column, guessed_row, guessed_rotation],
                           "target_position": [int(target["column"][0, index]), int(target["row"][0, index])] if correct_pose is not None else None})
        cases.append({"sample_id": record["sample_id"], "split": record["split"], "density": record["density"], "skin": record["skin"],
                      "group_id": record["group_id"], "counts": counts, **diagnostic_metrics(output, target),
                      "known_typed_cells": int(typed.sum()), "semantic_head_correct": int((typed & semantic_right).sum()),
                      "occupied_and_semantic_correct": int((typed & semantic_right & predicted).sum()),
                      "stage_geometry_error_cells": [float(geometry_error(stage["lattice"], target["lattice"]).norm(dim=-1).mean()) for stage in output["stages"]],
                      "layers": layers, "target_occupancy": truth.int().tolist(), "predicted_occupancy": predicted.int().tolist()})

    aggregates = {}
    for split in ("training", "development"):
        subset = [case for case in cases if case["split"] == split]
        if not subset:
            raise ValueError("both pilot splits required")
        totals = {key: sum(case["counts"][key] for case in subset) for key in ("tp", "fp", "fn", "tn")}
        totals.update(images=len(subset), ancestry_groups=len({case["group_id"] for case in subset}),
                      exact_locked_boards=sum(case["exact_locked_board"] is True for case in subset),
                      mean_geometry_error_cells=sum(case["mean_geometry_error_cells"] for case in subset)/len(subset),
                      exact_known_layers=sum(layer["exact"] is True for case in subset for layer in case["layers"]))
        totals["cell_accuracy"] = (totals["tp"]+totals["tn"])/sum(totals[key] for key in ("tp", "fp", "fn", "tn"))
        totals["occupied_precision"] = totals["tp"]/(totals["tp"]+totals["fp"]) if totals["tp"]+totals["fp"] else None
        totals["occupied_recall"] = totals["tp"]/(totals["tp"]+totals["fn"]) if totals["tp"]+totals["fn"] else None
        reference = summary["final_training" if split == "training" else "development"]
        if any(totals[key] != reference[key] for key in ("images", "exact_locked_boards", "exact_known_layers")) or abs(totals["cell_accuracy"]-reference["cell_accuracy"]) > 1e-6:
            raise ValueError("independent checkpoint diagnostics disagree with run summary")
        aggregates[split] = totals
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "passed", "controlled_repeat": controlled, "checkpoint_sha256": checkpoint["weights_sha256"],
              "analysis_script_sha256": file_digest(Path(__file__)), "aggregates": aggregates, "cases": cases,
              "claim": "Tiny matched synthetic diagnostics. Six images/split are three ancestry groups; no complete-state or real-image acceptance."}
    (args.output/"analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    tile_w, tile_h, cell = 180, 300, 11
    canvas = Image.new("RGB", (6*tile_w, 2*tile_h+48), (23, 27, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), "LOCKED CELL ERRORS - bottom row at bottom; raw logits, not accepted parser output", fill="white")
    draw.text((10, 25), "Blue = true occupied | red = false occupied | orange = missed occupied | gray = empty", fill=(180, 190, 200))
    for index, case in enumerate(cases):
        x, y = (index % 6)*tile_w+24, (index//6)*tile_h+48
        draw.text((x-16, y+4), case["split"]+" "+case["density"], fill="white")
        draw.text((x-16, y+20), case["skin"], fill=(180, 190, 200))
        for r in range(20):
            for c in range(10):
                truth, predicted = case["target_occupancy"][r][c], case["predicted_occupancy"][r][c]
                color = (60, 175, 220) if truth and predicted else (235, 90, 100) if predicted else (245, 175, 65) if truth else (48, 55, 63)
                left, top = x+c*cell, y+40+(19-r)*cell
                draw.rectangle((left, top, left+cell-2, top+cell-2), fill=color)
        draw.text((x-16, y+266), f"FP {case['counts']['fp']} / FN {case['counts']['fn']}", fill="white")
    canvas.save(args.output/"locked-errors.png")
    print(json.dumps({"status": "passed", "controlled_repeat": controlled, "aggregates": aggregates}, indent=2))


if __name__ == "__main__":
    main()
