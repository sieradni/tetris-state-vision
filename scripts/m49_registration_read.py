"""M49: does a label-free raster registration buy the committed read back?

`docs/M49_REGISTRATION_READ.md` declares this measurement. It applies the campaign's committed per-cell read-out
(logistic, ridge 1., thirty statistics) to a third sampling of the *same* function: the lattice registered from the
board's own raster support, computed from the valid mask alone. Nothing is trained, no parameter is fitted on this
arm's rows, and the read-out is accepted only if it reproduces the committed artifact's logits on the target lattice
first. The target, predicted and registered reads are scored on one population in one pass so they are comparable.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from oracle_pairing_read import score  # noqa: E402
from pixel_ceiling import cell_features, raster  # noqa: E402
from pixel_query_deciding_read import population  # noqa: E402
from probe_model import require_bounded_job  # noqa: E402
from record_stream_run import require_owned_run  # noqa: E402
from source_snapshot import verify_source  # noqa: E402

CEILING_BYTES = 1024*1024**2
DECIDING_CASES = 36
NONEMPTY_CASES = 28
OCCUPIED_CELLS = 667
EMPTY_CELLS = 4933
ANCESTRY_GROUPS = 12
PROFILE_CASES = 2
RIDGE = 1.0
ITERATIONS = 40
EXPECTED_CELLS = 200
ROWS, COLUMNS = 20, 10
SOURCES = tuple("scripts/"+name for name in
                ("m49_registration_read.py", "pixel_ceiling.py", "oracle_pairing_read.py",
                 "pixel_query_deciding_read.py", "probe_model.py", "source_snapshot.py",
                 "record_stream_run.py"))
READS = ("target_lattice", "predicted_lattice", "registered_lattice")

REGISTRATION_GATE_F1 = 0.70
REGISTRATION_GATE_AUC = 0.90
CONTROL_FLOOR = 0.382881
PARTIAL_AUC = 0.86


def branch_for(f1_at_zero, auc):
    """The predeclared branch selection: registration recovers the read, partly, or not at all."""
    if f1_at_zero >= REGISTRATION_GATE_F1 and auc >= REGISTRATION_GATE_AUC:
        return "registration-recovers"
    if f1_at_zero >= CONTROL_FLOOR or auc >= PARTIAL_AUC:
        return "registration-partial"
    return "registration-fails"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Lattice:
    """The three vectors `pixel_ceiling.cell_features` reads, in image pixels."""

    def __init__(self, origin, u, v):
        self.origin = origin
        self.u = u
        self.v = v


def displacement_cells(source, target):
    """Per-cell centres of `source` in the target lattice's own cell units."""
    import numpy as np
    u = np.asarray(target.u, dtype=float)
    v = np.asarray(target.v, dtype=float)
    columns, rows = [], []
    for row in range(ROWS):
        for column in range(COLUMNS):
            index = np.array([column+.5, row+.5])
            centre = np.asarray(target.origin, dtype=float)+index[0]*u+index[1]*v
            moved = (np.asarray(source.origin, dtype=float)+index[0]*np.asarray(source.u, dtype=float)
                     +index[1]*np.asarray(source.v, dtype=float))
            delta = moved-centre
            columns.append(float(delta @ u/(u @ u)))
            rows.append(float(delta @ v/(v @ v)))
    columns, rows = np.array(columns), np.array(rows)
    magnitude = np.sqrt(columns**2+rows**2)
    return {"mean_magnitude": round(float(magnitude.mean()), 6),
            "median_magnitude": round(float(np.median(magnitude)), 6),
            "p90_magnitude": round(float(np.quantile(magnitude, .9)), 6),
            "mean_columns": round(float(columns.mean()), 6),
            "mean_rows": round(float(rows.mean()), 6)}


def fit_folds(features, labels, groups):
    """The committed read-out, kept per fold so it can be applied to another sampling unchanged."""
    import numpy as np
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    groups = np.asarray(groups)
    folds, logits = {}, np.zeros(len(features))
    for name in sorted(set(groups.tolist())):
        train, test = groups != name, groups == name
        if not labels[train].any() or not (~labels[train].astype(bool)).any():
            raise ValueError("every training fold needs both classes")
        mean = features[train].mean(0)
        scale = features[train].std(0)
        scale[scale < 1e-9] = 1.
        train_matrix = np.c_[np.clip((features[train]-mean)/scale, -8, 8), np.ones(int(train.sum()))]
        weights = np.zeros(train_matrix.shape[1])
        penalty = np.eye(train_matrix.shape[1])*RIDGE
        penalty[-1, -1] = 0.
        target = labels[train]
        for _ in range(ITERATIONS):
            probability = 1/(1+np.exp(-np.clip(train_matrix @ weights, -30, 30)))
            weight = np.clip(probability*(1-probability), 1e-6, None)
            adjusted = train_matrix @ weights+(target-probability)/weight
            weights = np.linalg.solve(train_matrix.T @ (weight[:, None]*train_matrix)+penalty,
                                      train_matrix.T @ (weight*adjusted))
        folds[name] = (mean, scale, weights)
        test_matrix = np.c_[np.clip((features[test]-mean)/scale, -8, 8), np.ones(int(test.sum()))]
        logits[test] = test_matrix @ weights
    return folds, logits


def apply_folds(folds, features, groups):
    """Apply each held-out fold's fitted transform to *new* features of the same cells."""
    import numpy as np
    features = np.asarray(features, dtype=np.float64)
    groups = np.asarray(groups)
    logits = np.zeros(len(features))
    for name, (mean, scale, weights) in folds.items():
        mask = groups == name
        matrix = np.c_[np.clip((features[mask]-mean)/scale, -8, 8), np.ones(int(mask.sum()))]
        logits[mask] = matrix @ weights
    return logits


def cell_diagnosis(fixed, rows):
    errors = {"true_positive": [], "false_positive": [], "false_negative": [], "true_negative": []}
    cells = []
    for fixed_row, (value, _) in zip(fixed, rows):
        sample, _, target, _, anchored = fixed_row
        cells.append((sample, target, anchored, value))
        key = ("true_" if (value > 0) == bool(anchored) else "false_")+("positive" if value > 0 else "negative")
        errors[key].append((target//10, target % 10, value))

    def summary(points):
        if not points:
            return {"count": 0}
        values = sorted(value for _, _, value in points)
        pick = lambda fraction: round(values[min(len(values)-1, int(fraction*len(values)))], 6)
        return {"count": len(points), "margin_low": pick(0.0), "margin_median": pick(0.5)}

    counted = {key: len(value) for key, value in errors.items()}
    occupied_total = counted["true_positive"]+counted["false_negative"]
    return {"cells": len(cells), "errors_at_zero": {key: summary(value) for key, value in errors.items()},
            "precision_at_zero": round(counted["true_positive"]/max(
                counted["true_positive"]+counted["false_positive"], 1), 6),
            "recall_at_zero": round(counted["true_positive"]/max(occupied_total, 1), 6),
            "occupied_cells": occupied_total,
            "false_positives_by_row": {str(row): sum(1 for r, _, _ in errors["false_positive"] if r == row)
                                       for row in range(ROWS)},
            "false_positives_by_column": {str(column): sum(1 for _, c, _ in errors["false_positive"]
                                                          if c == column) for column in range(COLUMNS)},
            "bands": {f"rows_{low}_{high}": sum(1 for r, _, _ in errors["false_positive"] if low <= r <= high)
                      for low, high in ((0, 4), (5, 9), (10, 14), (15, 19))}}


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default="m3-occstack-lever-r2-sweep-r1")
    parser.add_argument("--pixel-ceiling", default="m3-occpix-ceiling-r1-r2")
    parser.add_argument("--model", default="m47-control-winner-config-r6")
    parser.add_argument("--reference", default="m3-occscale-registration-r2")
    parser.add_argument("--predeclaration", default="docs/M49_REGISTRATION_READ.md")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-cases", type=int, default=DECIDING_CASES)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--profile", action="store_true")
    mode.add_argument("--admission", type=Path)
    args = parser.parse_args()
    require_owned_run(args.output.name)
    if args.maximum_cases != DECIDING_CASES:
        raise ValueError("the declared population is thirty-six deciding boards")
    ceiling = read(ROOT/"runs"/(args.output.name+"-job.json"))["memory_bytes"]
    if ceiling != CEILING_BYTES:
        raise ValueError("the measurement ceiling differs from the declared 1024 MiB")
    sweep = read(ROOT/"runs"/args.sweep/"result.json")
    run = ROOT/"runs"/sweep["run"]
    summary = read(run/"summary.json")
    committed = read(ROOT/"runs"/args.pixel_ceiling/"result.json")
    if sweep["weights_sha256"] != summary["checkpoint_sha256"] or committed["weights_sha256"] != sweep["weights_sha256"]:
        raise ValueError("the committed read-out, sweep and run disagree about the measured weights")
    cases = [row for row in sweep["cases"] if row["split"] == "training"]
    if len(cases) != DECIDING_CASES or any(row["known_cells"] != EXPECTED_CELLS for row in cases):
        raise ValueError("thirty-six deciding boards with two hundred cells each are required")
    decided = sweep["deciding"]
    if (decided["nonempty_images"], decided["occupied_cells"], decided["empty_cells_on_nonempty_boards"]) != (
            NONEMPTY_CASES, OCCUPIED_CELLS, EMPTY_CELLS):
        raise ValueError("the joined sweep is not the declared population")

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

    def target_features(row):
        case = replay_case(recipes[row["sample_id"]], split="training", curriculum=curriculum,
                           component_capacity=capacity, max_side=max_side)
        prepared = prepare_image(case.frame.image, max_side=max_side)
        board = case.frame.state.board.lattice.transformed(prepared.scale_x, prepared.scale_y)
        matrix = cell_features(raster(case.frame.image), board)
        record = {"sample_id": row["sample_id"], "group_id": row["group_id"],
                  "known_cells": int(matrix.shape[0]), "occupied": sum(row["occupied"]),
                  "features_sha256": hashlib.sha256(matrix.astype("<f8").tobytes()).hexdigest()}
        return record, matrix, case, prepared, board

    started = time.monotonic()
    if args.profile:
        args.output.mkdir(parents=True, exist_ok=False)
        nonempty = [row for row in cases if any(row["occupied"])]
        compiled = [target_features(row) for row in nonempty[:PROFILE_CASES]]
        keyed = {row["sample_id"]: row["features_sha256"] for row in committed["cases"]}
        if len(keyed) != NONEMPTY_CASES:
            raise ValueError("the committed read-out did not ship the twenty-eight nonempty boards")
        matches = [record["features_sha256"] == keyed[record["sample_id"]] for record, *_ in compiled]
        if len(matches) != PROFILE_CASES:
            raise ValueError("the profile did not pin one digest per profiled board")
        (args.output/"summary.json").write_text(json.dumps(
            {"status": "complete", "sweep": args.sweep, "run": sweep["run"],
             "weights_sha256": sweep["weights_sha256"], "catalogue_sha256": summary["catalogue_sha256"],
             "admission_ceiling_bytes": ceiling, "cases": DECIDING_CASES,
             "profile_cases": [record for record, *_ in compiled], "target_features_match": matches,
             "scope": "Resource profile only: two feature extractions, no fit and no claim."},
            indent=2)+"\n", encoding="utf-8")
        print(json.dumps({"status": "complete", "profile": True, "target_features_match": matches}), flush=True)
        return
    profile_summary = read(args.admission/"summary.json")
    profile_job = read(ROOT/"runs"/(args.admission.name+"-job.json"))
    if (profile_summary.get("status") != "complete" or profile_job.get("status") != "passed" or
            profile_summary.get("weights_sha256") != sweep["weights_sha256"] or
            profile_summary.get("cases") != DECIDING_CASES or not all(profile_summary["target_features_match"])):
        raise ValueError("the profile does not admit this population")

    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    args.output.mkdir(parents=True, exist_ok=False)
    known = {row["sample_id"]: row for row in committed["cases"]}
    if len(known) != NONEMPTY_CASES:
        raise ValueError("the committed read-out did not ship the twenty-eight nonempty boards")
    collected, matrices, labels, groups, boards = [], [], [], [], {}
    for row in cases:
        record, matrix, case, prepared, board = target_features(row)
        if not any(row["occupied"]):
            continue
        if record["features_sha256"] != known[row["sample_id"]]["features_sha256"]:
            raise ValueError(f"{row['sample_id']}: target-lattice features differ from the committed artifact")
        collected.append({"record": record, "case": case, "pixels": prepared.pixels[None].clone(),
                          "valid": prepared.valid[None].clone(), "width": prepared.pixels.shape[-1],
                          "height": prepared.pixels.shape[-2]})
        boards[row["sample_id"]] = board
        matrices.append(matrix)
        labels.extend(row["occupied"])
        groups.extend([row["group_id"]]*len(row["occupied"]))
        (args.output/"progress.json").write_text(json.dumps({"cases": len(collected)})+chr(10))
    # the replay and its features come from the committed sweep's frozen code; the geometry comes from the
    # committed model's own frozen code, so the module is purged between the two passes (V-180).
    for name in [module for module in list(sys.modules) if module.split(".")[0] == "tetris_vision"]:
        del sys.modules[name]
    model_run = ROOT/"runs"/args.model
    model_summary = read(model_run/"summary.json")
    model_manifest = read(model_run/"checkpoint/manifest.json")
    if model_manifest["weights_sha256"][:12] != "eddec1fd4af2":
        raise ValueError("the geometry source is not the committed winner checkpoint")
    model_code = model_run/"code"
    sys.path[:0] = [str(model_code/"scripts"), str(model_code)]
    import importlib
    model_module = importlib.import_module("tetris_vision.model")
    if not Path(model_module.__file__).resolve().is_relative_to(model_code):
        raise ValueError("the geometry source model implementation is not the committed one")
    if not hasattr(model_module, "pixel_support_lattice"):
        raise ValueError("the geometry source carries no declared support registration")
    from tetris_vision.checkpoint import load_checkpoint
    model, loaded = load_checkpoint(model_run/"checkpoint")
    if loaded["weights_sha256"] != model_summary["checkpoint_sha256"]:
        raise ValueError("the loaded geometry source differs from its run")
    model.eval()
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    predicted_rows, registered_rows = [], []
    registered_lattices = {}
    for entry in collected:
        with torch.no_grad():
            output = model(entry["pixels"], entry["valid"])
        lattice = output["final"]["lattice"][0].detach().cpu().tolist()
        width, height = entry["width"], entry["height"]
        predicted = Lattice([lattice[0][0]*width, lattice[0][1]*height],
                            [lattice[1][0]*width, lattice[1][1]*height],
                            [lattice[2][0]*width, lattice[2][1]*height])
        with torch.no_grad():
            registered = model_module.pixel_support_lattice(entry["valid"], ROWS, COLUMNS, "cpu", torch.float64)[0]
        support = registered.detach().cpu().tolist()
        support_lattice = Lattice([support[0][0]*width, support[0][1]*height],
                                  [support[1][0]*width, support[1][1]*height],
                                  [support[2][0]*width, support[2][1]*height])
        predicted_rows.append(cell_features(raster(entry["case"].frame.image), predicted))
        registered_rows.append(cell_features(raster(entry["case"].frame.image), support_lattice))
        registered_lattices[entry["record"]["sample_id"]] = (predicted, support_lattice)
    records = [entry["record"] for entry in collected]
    if (len(records), sum(labels), len(labels)-sum(labels)) != (NONEMPTY_CASES, OCCUPIED_CELLS, EMPTY_CELLS):
        raise ValueError("the measured population differs from the predeclaration")
    features = np.vstack(matrices)
    predicted_features = np.vstack(predicted_rows)
    registered_features = np.vstack(registered_rows)
    labels = np.array(labels)
    groups = np.array(groups)

    folds, target_logits = fit_folds(features, labels, groups)
    committed_logits = np.concatenate([np.array(known[record["sample_id"]]["pixel_logits"]) for record in records])
    reproduction = float(np.abs(target_logits-committed_logits).max())
    if reproduction > 1e-6:
        raise ValueError(f"the read-out reconstruction does not reproduce the committed logits ({reproduction})")
    predicted_logits = apply_folds(folds, predicted_features, groups)
    registered_logits = apply_folds(folds, registered_features, groups)
    if not all(torch.equal(value, model.state_dict()[name]) for name, value in weights.items()):
        raise ValueError("the geometry read changed model tensors")

    reference = read(ROOT/"runs"/args.reference/"result.json")
    fixed = population(reference)
    if len(fixed) != 2986:
        raise ValueError("the fixed population is not the declared 2,986 cells")
    order = [record["sample_id"] for record in records]
    cursor = {sample: index*EXPECTED_CELLS for index, sample in enumerate(order)}
    fixed_rows = [row for row in fixed if row[0] in cursor]
    if len(fixed_rows) != 2986:
        raise ValueError("the scored rows are not the fixed population")
    reads = {"target_lattice": target_logits, "predicted_lattice": predicted_logits,
             "registered_lattice": registered_logits}
    scored = {name: [(float(values[cursor[sample]+target]), anchored)
                     for sample, _, target, _, anchored in fixed_rows]
              for name, values in reads.items()}
    measured = {name: score(rows) for name, rows in scored.items()}
    diagnosis = {name: cell_diagnosis(fixed_rows, rows) for name, rows in scored.items()}

    predicates = {sample: displacement_cells(predicted, boards[sample])
                  for sample, (predicted, _) in registered_lattices.items()}
    shift_readings = {sample: displacement_cells(registered, boards[sample])
                      for sample, (_, registered) in registered_lattices.items()}
    geometry_shift = {
        "predicted_vs_target": {key: round(float(np.mean([value[key] for value in predicates.values()])), 6)
                                for key in ("mean_magnitude", "median_magnitude", "p90_magnitude",
                                            "mean_columns", "mean_rows")},
        "registered_vs_target": {key: round(float(np.mean([value[key] for value in shift_readings.values()])), 6)
                                 for key in ("mean_magnitude", "median_magnitude", "p90_magnitude",
                                             "mean_columns", "mean_rows")},
    }
    anchor = read(ROOT/"docs/evidence/m47-pixel-sole-r1/scope.json")["anchors"]
    verdict = branch_for(measured["registered_lattice"]["f1_at_zero"], measured["registered_lattice"]["auc"])
    reading = {
        "registration-recovers": (
            "registration-recovers: a label-free raster registration places the statistics well enough that the "
            "committed read-out keeps ceiling-level quality, so the defect is the placement the model predicts"),
        "registration-partial": (
            "registration-partial: registration recovers the ordering but not the fixed crossing, so the residual "
            "is a scale or offset the read cannot absorb"),
        "registration-fails": (
            "registration-fails: the support bounding box does not place the statistics, so label-free raster "
            "registration is ruled out and placement must be addressed inside the model"),
    }[verdict]
    scope = {
        "status": "passed", "instrument": args.output.name, "predeclaration": args.predeclaration,
        "scope": ("Read-only: the committed per-cell read-out applied to the same thirty statistics at three "
                  "samplings (target, predicted, registered-from-support), no training, no refit on this arm's "
                  "inputs and no weight change."),
        "readout": {"provenance": args.pixel_ceiling, "statistic": "logistic, ridge 1.", "features": 30,
                    "parameters_imported_unchanged": True,
                    "reconstruction": {"max_abs_logit_difference": reproduction,
                                       "committed_logits_reproduced": reproduction <= 1e-6},
                    "refit_note": ("the committed artifact ships out-of-fold logits, not coefficients, so the "
                                   "read-out is reconstructed by running its own committed estimator on the "
                                   "committed features, labels and folds and is accepted only when it reproduces "
                                   "those logits to 1e-6; the same fitted folds then transform the registered and "
                                   "predicted features, so no parameter is fitted on this arm's rows")},
        "geometry": {"source": args.model, "weights_sha256": model_manifest["weights_sha256"],
                     "stage": "final", "boards": len(records),
                     "registration": {"recipe": "pixel_support_lattice(valid, 20, 10)",
                                      "source_file": str(Path(model_module.__file__).relative_to(ROOT)),
                                      "label_free": True}},
        "population": {"cells": 2986, "boards": len(records), "target_features_match": True},
        "measured": measured,
        "geometry_shift": geometry_shift,
        "anchors": {"pixel_best": anchor["pixel_best"], "pixel_auc": 0.999195, "pixel_f1_at_zero": 0.974763,
                    "control_auc": anchor["anchor_auc"], "control_f1_at_zero": anchor["anchor_f1_at_zero"],
                    "control_precision_at_zero": 0.230430,
                    "m48_predicted_f1_at_zero": 0.037152, "m48_predicted_auc": 0.442341,
                    "m48_predicted_shift_cells": 2.479843},
        "branch": {"verdict": verdict, "f1_gate_at_zero": REGISTRATION_GATE_F1, "auc_gate": REGISTRATION_GATE_AUC,
                   "control_floor_f1_at_zero": CONTROL_FLOOR, "partial_auc": PARTIAL_AUC, "reading": reading},
        "diagnosis": diagnosis,
        "elapsed_seconds": time.monotonic()-started,
        "sources": {name: digest(ROOT/name) for name in SOURCES},
    }
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(scope, stream, indent=2, allow_nan=False)
    (args.output/"diagnosis.json").write_text(json.dumps(diagnosis["registered_lattice"], indent=2)+"\n",
                                              encoding="utf-8")
    print(json.dumps({"verdict": verdict, "registered": measured["registered_lattice"],
                      "target": measured["target_lattice"], "predicted": measured["predicted_lattice"],
                      "geometry_shift": geometry_shift}, indent=2))


if __name__ == "__main__":
    main()
