"""M48 auxiliary measurement: how far the committed model's lattice sits from the target lattice.

`docs/M48_GEOMETRY_READ.md` reports that the committed read-out collapses at the predicted lattice without saying
why. This read-only diagnostic measures the geometry difference directly on the same twenty-eight boards: the
per-cell displacement of the predicted lattice from the target lattice in cell units, per axis, and the drift of the
thirty statistics' own distribution between the two samplings. No model change, no training, no label, no threshold.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from pixel_ceiling import cell_features, raster  # noqa: E402
from probe_model import require_bounded_job  # noqa: E402
from record_stream_run import require_owned_run  # noqa: E402
from source_snapshot import verify_source  # noqa: E402

CEILING_BYTES = 1024*1024**2
DECIDING_CASES = 36
NONEMPTY_CASES = 28
ANCESTRY_GROUPS = 12
PROFILE_CASES = 2
SOURCES = ("scripts/m48_lattice_displacement.py", "scripts/pixel_ceiling.py", "scripts/probe_model.py",
           "scripts/record_stream_run.py", "scripts/source_snapshot.py")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Lattice:
    def __init__(self, origin, u, v):
        self.origin, self.u, self.v = origin, u, v


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default="m3-occstack-lever-r2-sweep-r1")
    parser.add_argument("--model", default="m47-control-winner-config-r6")
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--profile", action="store_true")
    mode.add_argument("--admission", type=Path)
    args = parser.parse_args()
    require_owned_run(args.output.name)
    ceiling = read(ROOT/"runs"/(args.output.name+"-job.json"))["memory_bytes"]
    if ceiling != CEILING_BYTES:
        raise ValueError("the diagnostic ceiling differs from the declared 1024 MiB")
    sweep = read(ROOT/"runs"/args.sweep/"result.json")
    run = ROOT/"runs"/sweep["run"]
    summary = read(run/"summary.json")
    cases = [row for row in sweep["cases"] if row["split"] == "training"]
    if len(cases) != DECIDING_CASES:
        raise ValueError("thirty-six deciding boards are required")
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
    started = time.monotonic()
    prepared_rows = []
    for row in cases:
        if not any(row["occupied"]):
            continue
        case = replay_case(recipes[row["sample_id"]], split="training", curriculum=curriculum,
                           component_capacity=capacity, max_side=max_side)
        prepared = prepare_image(case.frame.image, max_side=max_side)
        board = case.frame.state.board.lattice.transformed(prepared.scale_x, prepared.scale_y)
        prepared_rows.append({"sample_id": row["sample_id"], "case": case, "board": board,
                              "pixels": prepared.pixels[None].clone(), "valid": prepared.valid[None].clone(),
                              "width": prepared.pixels.shape[-1], "height": prepared.pixels.shape[-2]})
    if args.profile:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/"summary.json").write_text(json.dumps(
            {"status": "complete", "sweep": args.sweep, "run": sweep["run"], "cases": DECIDING_CASES,
             "profiled": [row["sample_id"] for row in prepared_rows[:PROFILE_CASES]],
             "scope": "Resource profile only: two replays, no forward and no claim."}, indent=2)+chr(10),
            encoding="utf-8")
        print(json.dumps({"status": "complete", "profile": True}), flush=True)
        return
    profile_summary = read(args.admission/"summary.json")
    if profile_summary.get("status") != "complete" or profile_summary.get("cases") != DECIDING_CASES:
        raise ValueError("the profile does not admit this population")

    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for name in [module for module in list(sys.modules) if module.split(".")[0] == "tetris_vision"]:
        del sys.modules[name]
    model_run = ROOT/"runs"/args.model
    model_manifest = read(model_run/"checkpoint/manifest.json")
    model_code = model_run/"code"
    sys.path[:0] = [str(model_code/"scripts"), str(model_code)]
    import importlib
    model_module = importlib.import_module("tetris_vision.model")
    if not Path(model_module.__file__).resolve().is_relative_to(model_code):
        raise ValueError("the geometry source model implementation is not the committed one")
    from tetris_vision.checkpoint import load_checkpoint
    model, _ = load_checkpoint(model_run/"checkpoint")
    model.eval()

    displacements, target_rows, predicted_rows = [], [], []
    args.output.mkdir(parents=True, exist_ok=False)
    for entry in prepared_rows:
        board, target_pixels = entry["board"], cell_features(raster(entry["case"].frame.image), entry["board"])
        with torch.no_grad():
            output = model(entry["pixels"], entry["valid"])
        lattice = output["final"]["lattice"][0].detach().cpu().tolist()
        width, height = entry["width"], entry["height"]
        predicted = Lattice([lattice[0][0]*width, lattice[0][1]*height],
                            [lattice[1][0]*width, lattice[1][1]*height],
                            [lattice[2][0]*width, lattice[2][1]*height])
        predicted_pixels = cell_features(raster(entry["case"].frame.image), predicted)
        # cell units: the target lattice's own column and row vectors are one cell
        u = np.asarray(board.u, dtype=float)
        v = np.asarray(board.v, dtype=float)
        for row in range(20):
            for column in range(10):
                index = row*10+column
                target_centre = np.asarray(board.origin, dtype=float)+(column+.5)*u+(row+.5)*v
                predicted_centre = (np.asarray(predicted.origin, dtype=float)
                                    +(column+.5)*np.asarray(predicted.u, dtype=float)
                                    +(row+.5)*np.asarray(predicted.v, dtype=float))
                delta = predicted_centre-target_centre
                displacements.append((float(delta @ u/(u @ u)), float(delta @ v/(v @ v))))
        target_rows.append(target_pixels)
        predicted_rows.append(predicted_pixels)
        (args.output/"progress.json").write_text(json.dumps({"boards": len(target_rows)})+chr(10))
    target = np.vstack(target_rows)
    predicted = np.vstack(predicted_rows)
    columns = np.array([d[0] for d in displacements])
    rows = np.array([d[1] for d in displacements])
    magnitude = np.sqrt(columns**2+rows**2)
    drift = (predicted.mean(0)-target.mean(0))/(target.std(0)+1e-12)
    scope = {
        "status": "passed", "instrument": args.output.name,
        "scope": ("Read-only lattice diagnostic for docs/M48_GEOMETRY_READ.md: displacement of the committed "
                  "model's predicted lattice from the target lattice, and the drift of the thirty statistics "
                  "between the two samplings. No training, no label, no threshold."),
        "geometry": {"source": args.model, "weights_sha256": model_manifest["weights_sha256"],
                     "boards": len(target_rows)},
        "population": {"boards": len(target_rows), "cells": int(magnitude.size)},
        "displacement_cells": {
            "mean_magnitude": round(float(magnitude.mean()), 6),
            "median_magnitude": round(float(np.median(magnitude)), 6),
            "p90_magnitude": round(float(np.quantile(magnitude, .9)), 6),
            "mean_columns": round(float(columns.mean()), 6),
            "mean_rows": round(float(rows.mean()), 6),
            "std_columns": round(float(columns.std()), 6),
            "std_rows": round(float(rows.std()), 6),
        },
        "feature_drift": {
            "mean_shift_in_target_std": round(float(np.abs(drift).mean()), 6),
            "max_shift_in_target_std": round(float(np.abs(drift).max()), 6),
            "features_over_three_std": int((np.abs(drift) > 3).sum()),
        },
        "sources": {name: digest(ROOT/name) for name in SOURCES},
        "elapsed_seconds": time.monotonic()-started,
    }
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(scope, stream, indent=2, allow_nan=False)
    print(json.dumps(scope["displacement_cells"] | {"feature_drift": scope["feature_drift"]}, indent=2))


if __name__ == "__main__":
    main()
