"""M48: does the committed ceiling read-out's quality survive the model's predicted lattice?

`docs/M48_GEOMETRY_READ.md` declares this measurement. It applies the campaign's committed per-cell read-out
(logistic, ridge 1., thirty statistics) to two samplings of the *same* function: the target lattice the ceiling
read-out was measured at, and the lattice the committed model predicts. Nothing is trained, no weight or read-out
constant is fitted on this arm's rows, and the read-out is accepted only if it reproduces the committed artifact's
own logits on the target lattice first.
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
SOURCES = tuple("scripts/"+name for name in
                ("m48_geometry_read.py", "pixel_ceiling.py", "oracle_pairing_read.py",
                 "pixel_query_deciding_read.py", "probe_model.py", "source_snapshot.py",
                 "record_stream_run.py"))


CELL_GATE = 0.70
CONTROL_FLOOR = 0.382881


def branch_for(f1_at_zero):
    """The predeclared branch selection: geometry binds, costs, or the read-out binds."""
    if f1_at_zero >= CELL_GATE:
        return "readout-binds"
    if f1_at_zero >= CONTROL_FLOOR:
        return "geometry-costs"
    return "geometry-binds"


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


def fit_folds(features, labels, groups):
    """The committed read-out, kept per fold so it can be applied to another sampling unchanged.

    This is `pixel_ceiling.logistic_fit`'s exact loop with the fitted transform retained; the caller proves the
    identity by requiring these logits to equal that function's, and the committed artifact's, on the target
    lattice before any predicted-lattice number is produced.
    """
    import numpy as np
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    groups = np.asarray(groups)
    folds, logits = {}, np.zeros(len(features))
    for name in sorted(set(groups.tolist())):
        train, test = groups != name, groups == name
        # the committed estimator's own guard: every training fold needs both classes
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


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default="m3-occstack-lever-r2-sweep-r1")
    parser.add_argument("--pixel-ceiling", default="m3-occpix-ceiling-r1-r2")
    parser.add_argument("--model", default="m47-control-winner-config-r6")
    parser.add_argument("--reference", default="m3-occscale-registration-r2")
    parser.add_argument("--predeclaration", default="docs/M48_GEOMETRY_READ.md")
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
        return record, matrix, case, prepared

    started = time.monotonic()
    if args.profile:
        args.output.mkdir(parents=True, exist_ok=False)
        nonempty = [row for row in cases if any(row["occupied"])]
        compiled = [target_features(row) for row in nonempty[:PROFILE_CASES]]
        keyed = {row["sample_id"]: row["features_sha256"] for row in committed["cases"]}
        # the committed artifact ships the twenty-eight nonempty deciding boards; empty boards were never fitted
        if len(keyed) != NONEMPTY_CASES:
            raise ValueError("the committed read-out did not ship the twenty-eight nonempty boards")
        matches = [record["features_sha256"] == keyed[record["sample_id"]] for record, *_ in compiled]
        if len(matches) != PROFILE_CASES:
            raise ValueError("the profile did not pin one digest per profiled board")
        (args.output/"summary.json").write_text(json.dumps(
            {"status": "complete", "sweep": args.sweep, "run": sweep["run"],
             "weights_sha256": sweep["weights_sha256"], "catalogue_sha256": summary["catalogue_sha256"],
             "admission_ceiling_bytes": ceiling, "cases": DECIDING_CASES,
             "profile_cases": [record for record, *_ in compiled],
             "target_features_match": matches,
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
    from tetris_vision.checkpoint import load_checkpoint
    model_run = ROOT/"runs"/args.model
    model_summary = read(model_run/"summary.json")
    model_manifest = read(model_run/"checkpoint/manifest.json")
    if model_manifest["weights_sha256"][:12] != "eddec1fd4af2":
        raise ValueError("the geometry source is not the committed winner checkpoint")
    model_code = model_run/"code"
    sys.path[:0] = [str(model_code/"scripts"), str(model_code)]
    model, loaded = load_checkpoint(model_run/"checkpoint")
    if loaded["weights_sha256"] != model_summary["checkpoint_sha256"]:
        raise ValueError("the loaded geometry source differs from its run")
    model.eval()
    weights = {name: value.clone() for name, value in model.state_dict().items()}

    args.output.mkdir(parents=True, exist_ok=False)
    known = {row["sample_id"]: row for row in committed["cases"]}
    if len(known) != NONEMPTY_CASES:
        raise ValueError("the committed read-out did not ship the twenty-eight nonempty boards")
    records, matrices, predicted_rows, labels, groups = [], [], [], [], []
    for row in cases:
        record, matrix, case, prepared = target_features(row)
        if not any(row["occupied"]):
            continue
        if record["features_sha256"] != known[row["sample_id"]]["features_sha256"]:
            raise ValueError(f"{row['sample_id']}: target-lattice features differ from the committed artifact")
        with torch.no_grad():
            output = model(prepared.pixels[None], prepared.valid[None])
        stage = output["final"]
        lattice = stage["lattice"][0].detach().cpu().tolist()
        height, width = raster(case.frame.image).shape[:2]
        predicted = Lattice([lattice[0][0]*width, lattice[0][1]*height],
                            [lattice[1][0]*width, lattice[1][1]*height],
                            [lattice[2][0]*width, lattice[2][1]*height])
        matrix_predicted = cell_features(raster(case.frame.image), predicted)
        records.append(record)
        matrices.append(matrix)
        predicted_rows.append(matrix_predicted)
        labels.extend(row["occupied"])
        groups.extend([row["group_id"]]*len(row["occupied"]))
        (args.output/"progress.json").write_text(json.dumps({"cases": len(records)})+"\n")
    if (len(records), sum(labels), len(labels)-sum(labels)) != (NONEMPTY_CASES, OCCUPIED_CELLS, EMPTY_CELLS):
        raise ValueError("the measured population differs from the predeclaration")
    features = np.vstack(matrices)
    predicted_features = np.vstack(predicted_rows)
    labels = np.array(labels)
    groups = np.array(groups)

    folds, target_logits = fit_folds(features, labels, groups)
    committed_logits = np.concatenate([np.array(known[record["sample_id"]]["pixel_logits"]) for record in records])
    reproduction = float(np.abs(target_logits-committed_logits).max())
    if reproduction > 1e-6:
        raise ValueError(f"the read-out reconstruction does not reproduce the committed logits ({reproduction})")
    predicted_logits = apply_folds(folds, predicted_features, groups)
    if not all(torch.equal(value, model.state_dict()[name]) for name, value in weights.items()):
        raise ValueError("the geometry read changed model tensors")

    reference = read(ROOT/"runs"/args.reference/"result.json")
    fixed = population(reference)
    if len(fixed) != 2986:
        raise ValueError("the fixed population is not the declared 2,986 cells")
    order = [record["sample_id"] for record in records]
    cursor = {sample: index*EXPECTED_CELLS for index, sample in enumerate(order)}
    target_rows = [(float(target_logits[cursor[sample]+target]), anchored)
                   for sample, _, target, _, anchored in fixed if sample in cursor]
    predicted_read_rows = [(float(predicted_logits[cursor[sample]+target]), anchored)
                           for sample, _, target, _, anchored in fixed if sample in cursor]
    if len(target_rows) != 2986 or len(predicted_read_rows) != 2986:
        raise ValueError("the scored rows are not the fixed population")
    measured = {"target_lattice": score(target_rows), "predicted_lattice": score(predicted_read_rows)}

    def cell_diagnosis(rows):
        cells = [(*fixed_row, value) for fixed_row, (value, _) in
                 zip([row for row in fixed if row[0] in cursor], rows)]
        errors = {"true_positive": [], "false_positive": [], "false_negative": [], "true_negative": []}
        for sample, _, target, _, anchored, margin in cells:
            key = ("true_" if (margin > 0) == bool(anchored) else "false_")+(
                "positive" if margin > 0 else "negative")
            errors[key].append((target//10, target % 10, margin))

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
                                           for row in range(20)},
                "false_positives_by_column": {str(column): sum(1 for _, c, _ in errors["false_positive"]
                                                              if c == column) for column in range(10)},
                "bands": {f"rows_{low}_{high}": sum(1 for r, _, _ in errors["false_positive"]
                                                    if low <= r <= high)
                          for low, high in ((0, 4), (5, 9), (10, 14), (15, 19))}}

    diagnosis = {name: cell_diagnosis(rows) for name, rows in
                 (("target_lattice", target_rows), ("predicted_lattice", predicted_read_rows))}
    anchor = read(ROOT/"docs/evidence/m47-pixel-sole-r1/scope.json")["anchors"]
    verdict = branch_for(measured["predicted_lattice"]["f1_at_zero"])
    scope = {
        "status": "passed", "instrument": args.output.name, "predeclaration": args.predeclaration,
        "scope": ("Read-only: the committed per-cell read-out applied to the same thirty statistics at two "
                  "samplings, with no training, no refit on this arm's inputs and no weight change."),
        "readout": {"provenance": args.pixel_ceiling, "statistic": "logistic, ridge 1.", "features": 30,
                    "parameters_imported_unchanged": True,
                    "reconstruction": {"max_abs_logit_difference": reproduction,
                                       "committed_logits_reproduced": reproduction <= 1e-6},
                    "refit_note": ("the committed artifact ships out-of-fold logits, not coefficients, so the "
                                   "read-out is reconstructed by running its own committed estimator on the "
                                   "committed features, labels and folds and is accepted only when it reproduces "
                                   "those logits to 1e-6; the same fitted folds then transform the predicted-"
                                   "lattice features, so no parameter is fitted on this arm's rows")},
        "geometry": {"source": args.model, "weights_sha256": model_manifest["weights_sha256"],
                     "stage": "final", "boards": len(records)},
        "population": {"cells": 2986, "boards": len(records), "target_features_match": True},
        "measured": measured,
        "anchors": {"pixel_best": anchor["pixel_best"], "pixel_auc": 0.999195, "pixel_f1_at_zero": 0.974763,
                    "control_auc": anchor["anchor_auc"], "control_f1_at_zero": anchor["anchor_f1_at_zero"],
                    "control_precision_at_zero": 0.230430},
        "branch": {"verdict": verdict, "cell_gate_f1_at_zero": CELL_GATE,
                   "control_floor_f1_at_zero": CONTROL_FLOOR,
                   "reading": ("readout-binds: the statistics order the cells even at the predicted lattice, so "
                               "the head's training is the binding constraint"
                               if verdict == "readout-binds" else
                               "geometry-costs: placement takes a measurable share but does not remove the "
                               "evidence" if verdict == "geometry-costs" else
                               "geometry-binds: at the predicted lattice the evidence no longer separates, so "
                               "sampling geometry is the binding constraint")},
        "diagnosis": diagnosis,
        "elapsed_seconds": time.monotonic()-started,
        "sources": {name: digest(ROOT/name) for name in SOURCES},
    }
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(scope, stream, indent=2, allow_nan=False)
    (args.output/"diagnosis.json").write_text(json.dumps(diagnosis["predicted_lattice"], indent=2)+"\n",
                                              encoding="utf-8")
    print(json.dumps({"verdict": verdict, "predicted": measured["predicted_lattice"],
                      "target": measured["target_lattice"]}, indent=2))


if __name__ == "__main__":
    main()
