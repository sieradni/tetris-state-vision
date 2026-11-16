"""M49 auxiliary: why the support registration misses, measured on the same boards.

`docs/M49_REGISTRATION_READ.md` reports that the declared label-free support lattice is displaced a mean
`8.256949` cells from the target lattice and that the committed read-out therefore scores `0.0` F1 at `0.0` there.
This read-only diagnostic measures the cause: the extent of the valid raster (what the recipe maps the board's
rows and columns onto), the extent of the board's own box in the same pixels, the ratio between them, and the
cell-size error the recipe inherits. No training, no forward, no label, no threshold, and no change to the arm.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from pixel_ceiling import raster  # noqa: E402
from probe_model import require_bounded_job  # noqa: E402
from record_stream_run import require_owned_run  # noqa: E402
from source_snapshot import verify_source  # noqa: E402

CEILING_BYTES = 640*1024**2
DECIDING_CASES = 36
NONEMPTY_CASES = 28
ANCESTRY_GROUPS = 12
ROWS, COLUMNS = 20, 10
SOURCES = ("scripts/m49_frame_extent.py", "scripts/pixel_ceiling.py", "scripts/probe_model.py",
           "scripts/record_stream_run.py", "scripts/source_snapshot.py")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default="m3-occstack-lever-r2-sweep-r1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_owned_run(args.output.name)
    ceiling = read(ROOT/"runs"/(args.output.name+"-job.json"))["memory_bytes"]
    if ceiling != CEILING_BYTES:
        raise ValueError("the diagnostic ceiling differs from the declared 640 MiB")
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
    args.output.mkdir(parents=True, exist_ok=False)
    boards = []
    for row in cases:
        if not any(row["occupied"]):
            continue
        case = replay_case(recipes[row["sample_id"]], split="training", curriculum=curriculum,
                           component_capacity=capacity, max_side=max_side)
        prepared = prepare_image(case.frame.image, max_side=max_side)
        board = case.frame.state.board.lattice.transformed(prepared.scale_x, prepared.scale_y)
        mask = prepared.valid[0]
        height, width = int(mask.shape[0]), int(mask.shape[1])
        rows_hit, columns_hit = mask.any(dim=1), mask.any(dim=0)
        top = int(rows_hit.to("cpu").numpy().argmax())
        bottom = height-int(rows_hit.flip(0).to("cpu").numpy().argmax())
        left = int(columns_hit.to("cpu").numpy().argmax())
        right = width-int(columns_hit.flip(0).to("cpu").numpy().argmax())
        origin, column, step = board.origin, board.u, board.v
        board_left, board_right = origin[0], origin[0]+COLUMNS*column[0]
        board_bottom, board_top = origin[1], origin[1]+ROWS*step[1]
        boards.append({
            "sample_id": row["sample_id"], "raster": [width, height],
            "valid_box": [left, top, right, bottom],
            "board_box": [round(min(board_left, board_right), 3), round(min(board_top, board_bottom), 3),
                          round(max(board_left, board_right), 3), round(max(board_top, board_bottom), 3)],
            "valid_width": right-left, "valid_height": bottom-top,
            "board_width": abs(COLUMNS*column[0]), "board_height": abs(ROWS*step[1]),
            "cell_width": abs(column[0]), "cell_height": abs(step[1]),
            "registered_cell_width": (right-left)/COLUMNS, "registered_cell_height": (bottom-top)/ROWS,
            "board_inside_valid": bool(board_left >= left and board_right <= right and
                                       board_bottom <= bottom and board_top >= top),
        })
        (args.output/"progress.json").write_text(json.dumps({"boards": len(boards)})+chr(10))
    if len(boards) != NONEMPTY_CASES:
        raise ValueError("the twenty-eight nonempty deciding boards are required")
    import numpy as np
    ratio_w = np.array([b["valid_width"]/b["board_width"] for b in boards])
    ratio_h = np.array([b["valid_height"]/b["board_height"] for b in boards])
    cell_w = np.array([b["registered_cell_width"]/b["cell_width"] for b in boards])
    cell_h = np.array([b["registered_cell_height"]/b["cell_height"] for b in boards])
    coverage = np.array([(b["board_width"]*b["board_height"])/(b["valid_width"]*b["valid_height"])
                         for b in boards])
    scope = {
        "status": "passed", "instrument": args.output.name,
        "scope": ("Read-only: the valid raster extent against the board's own box on the same twenty-eight boards, "
                  "to say why the declared support registration is displaced. No forward, no label, no threshold."),
        "population": {"boards": len(boards)},
        "extents": {
            "valid_to_board_width_ratio": {"mean": round(float(ratio_w.mean()), 6),
                                           "median": round(float(np.median(ratio_w)), 6),
                                           "min": round(float(ratio_w.min()), 6),
                                           "max": round(float(ratio_w.max()), 6)},
            "valid_to_board_height_ratio": {"mean": round(float(ratio_h.mean()), 6),
                                            "median": round(float(np.median(ratio_h)), 6),
                                            "min": round(float(ratio_h.min()), 6),
                                            "max": round(float(ratio_h.max()), 6)},
            "registered_cell_width_ratio": {"mean": round(float(cell_w.mean()), 6),
                                            "max": round(float(cell_w.max()), 6)},
            "registered_cell_height_ratio": {"mean": round(float(cell_h.mean()), 6),
                                             "max": round(float(cell_h.max()), 6)},
            "board_area_share_of_valid": {"mean": round(float(coverage.mean()), 6),
                                          "median": round(float(np.median(coverage)), 6)},
            "board_inside_valid_boards": int(sum(b["board_inside_valid"] for b in boards)),
        },
        "boards": boards,
        "sources": {name: digest(ROOT/name) for name in SOURCES},
        "elapsed_seconds": time.monotonic()-started,
    }
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(scope, stream, indent=2, allow_nan=False)
    print(json.dumps(scope["extents"], indent=2))


if __name__ == "__main__":
    main()
