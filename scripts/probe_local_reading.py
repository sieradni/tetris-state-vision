"""Fixed-layout single-cell interventions on a retained synthetic checkpoint.

This measures sensitivity and error, not real-image or complete-state accuracy.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sys

from probe_model import require_bounded_job

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_bounded_job()
    import torch
    from PIL import Image, ImageChops
    from tetris_vision.checkpoint import file_digest, load_checkpoint
    from tetris_vision.dataset import source_digest
    from tetris_vision.layout import Layout
    from tetris_vision.losses import geometry_error, targets_for
    from tetris_vision.preprocess import prepare_image
    from tetris_vision.render import Appearance, render, sample_scene
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = source_digest()
    summary = json.loads((args.run/"summary.json").read_text())
    manifest = args.run/"manifest.jsonl"
    if manifest.stat().st_size > 1024**2 or summary["runtime_source_sha256"] != source or summary["arguments"]["max_side"] != 384:
        raise ValueError("expected the unchanged 384-side synthetic pilot source")
    if file_digest(manifest) != summary["manifest_sha256"]:
        raise ValueError("manifest identity changed")
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    if len(records) != 12:
        raise ValueError("only the twelve-image pilot manifest is admitted")
    records = [row for row in records if row["split"] == "training"]
    if len(records) != 6:
        raise ValueError("six training appearances required")
    model, checkpoint = load_checkpoint(args.run/"checkpoint")
    if checkpoint["weights_sha256"] != summary["checkpoint_sha256"]:
        raise ValueError("checkpoint identity differs")
    model.eval()
    forwards, cases, trials = 0, [], []

    def evaluate(frame):
        nonlocal forwards
        forwards += 1
        if forwards > 54:
            raise ValueError("54-forward work budget exceeded")
        prepared = prepare_image(frame.image, max_side=384)
        target = targets_for(frame.state, prepared)
        if int(target["known"].sum()) != 200:
            raise ValueError("intervention must retain all 200 visible known cells")
        with torch.no_grad():
            output = model(prepared.pixels[None], prepared.valid[None])["final"]
        return output["occupancy"][0, :20], output["lattice"], target

    for record in records:
        scene = sample_scene(record["seed"], record["density"], varied=True)
        layout, appearance = Layout(**record["layout"]), Appearance(**record["appearance"])
        original = render(scene, appearance, layout)
        path = (args.run/record["image"]).resolve()
        if not path.is_relative_to(args.run.resolve()) or file_digest(path) != record["image_sha256"]:
            raise ValueError("baseline image identity changed")
        with Image.open(path) as saved:
            if saved.mode != original.image.mode or saved.size != original.image.size or saved.tobytes() != original.image.tobytes():
                raise ValueError("source/seed/layout does not reproduce saved raster exactly")
        original_state = original.state.to_dict()
        state_path = (args.run/record["state"]).resolve()
        if not state_path.is_relative_to(args.run.resolve()) or file_digest(state_path) != record["state_sha256"] or json.loads(state_path.read_text()) != original_state:
            raise ValueError("supported baseline labels do not reproduce")
        base_logits, base_lattice, base_target = evaluate(original)
        base_prediction = base_logits > 0
        base_exact = bool((base_prediction == base_target["occupied"][0].bool()).all())
        cases.append({"sample_id": record["sample_id"], "exact_baseline_board": base_exact,
                      "baseline_geometry_error_cells": float(geometry_error(base_lattice, base_target["lattice"]).norm(dim=-1).mean())})
        blocked = {cell for layer in (scene.active, scene.ghost) if layer for cell in layer.cells}
        picks, rng = [], random.Random(record["seed"]+0xCE11)
        for occupied in (False, True):
            choices = [(c, r) for r in range(20) for c in range(10) if (c, r) not in blocked and
                       (scene.locked[r][c] is not None) == occupied]
            rng.shuffle(choices)
            picks.extend(choices[:4])
        for c, r in picks:
            locked = [list(row) for row in scene.locked]
            before = locked[r][c] is not None
            locked[r][c] = None if before else "IOTSZJLG"[(record["seed"]+c+3*r) % 8]
            changed = render(replace(scene, locked=tuple(tuple(row) for row in locked)), appearance, layout)
            changed_state = changed.state.to_dict()
            modified_cells = [(x, y) for y in range(20) for x in range(10) if original_state["board"]["rows"][y][x] != changed_state["board"]["rows"][y][x]]
            if modified_cells != [(c, r)] or any(original_state[key] != changed_state[key] for key in original_state if key != "board") or original.state.board.lattice != changed.state.board.lattice:
                raise ValueError("intervention changed unrelated supported state")
            box = ImageChops.difference(original.image, changed.image).getbbox()
            bounds = layout.cell_box(c, r)
            if box is None or not (bounds[0] <= box[0] < box[2] <= bounds[2] and bounds[1] <= box[1] < box[3] <= bounds[3]):
                raise ValueError("raster delta escaped the chosen visible cell")
            logits, lattice, target = evaluate(changed)
            prediction = logits > 0
            unrelated = (prediction != base_prediction).clone()
            unrelated[r, c] = False
            probability_before, probability_after = float(base_logits[r, c].sigmoid()), float(logits[r, c].sigmoid())
            trials.append({"sample_id": record["sample_id"], "seed": record["seed"], "skin": record["skin"], "cell": [c, r],
                           "before": before, "after": not before, "raster_changed_bbox": box,
                           "raster_pixels_sha256": hashlib.sha256(changed.image.tobytes()).hexdigest(),
                           "targeted_correct": bool(prediction[r, c]) == (not before),
                           "exact_counterfactual_board": bool((prediction == target["occupied"][0].bool()).all()),
                           "unrelated_prediction_changes": int(unrelated.sum()),
                           "probability_before": probability_before, "probability_after": probability_after,
                           "signed_probability_delta": (probability_after-probability_before)*(-1 if before else 1),
                           "lattice_drift_cells": float((geometry_error(lattice, base_target["lattice"])-geometry_error(base_lattice, base_target["lattice"])).norm(dim=-1).mean())})
    if source_digest() != source or not trials:
        raise ValueError("source changed during probe or no interventions")
    aggregate = {"baseline_exact_boards": sum(case["exact_baseline_board"] for case in cases), "baseline_images": len(cases),
                 "single_cell_trials": len(trials), "distinct_intervention_groups": len({(row["seed"], *row["cell"], row["after"]) for row in trials}),
                 "targeted_correct": sum(row["targeted_correct"] for row in trials),
                 "exact_counterfactual_boards": sum(row["exact_counterfactual_board"] for row in trials),
                 "unrelated_changed_cells": sum(row["unrelated_prediction_changes"] for row in trials),
                 "correct_direction_probability_changes": sum(row["signed_probability_delta"] > 0 for row in trials),
                 "forwards": forwards, "max_forwards": 54}
    report = {"status": "passed", "aggregate": aggregate, "cases": cases, "trials": trials,
              "checkpoint_sha256": checkpoint["weights_sha256"], "runtime_source_sha256": source,
              "script_sha256": file_digest(Path(__file__)), "source_run_status": summary["status"], "completed_training_steps": summary["steps"],
              "scope": "Same-layout visible single-cell interventions; no oracle input. All other supported fields/raster regions checked unchanged. Matched appearances are dependent, not independent test images."}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"status": "passed", "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
