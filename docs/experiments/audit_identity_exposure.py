"""Measure identity supervision exposure on a completed M3 run's recorded probes.

Model-free: replays the frozen recorded recipes, rebuilds the queue/hold targets
from source labels and joins the published per-case diagnostics. No inference, no
training, no new lever, no target or mask change. Counts only: masked identity
stays excluded and unknown never becomes a label.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_model import require_bounded_job  # noqa: E402
from record_stream_run import require_owned_run, validate_completion  # noqa: E402
from source_snapshot import verify_source  # noqa: E402

PIECES = tuple("IOTSZJL")
EOS = 7
PUBLISHED = ROOT / "docs/evidence/m3-identity-factor-r1-errors-r3/result.json"


def read(path, limit=1024**2):
    if path.stat().st_size > limit:
        raise ValueError("audit JSON byte budget exceeded")
    return json.loads(path.read_text())


def require_frozen_callables(code, *functions):
    if any(not Path(inspect.unwrap(function).__code__.co_filename).resolve().is_relative_to(code)
           for function in functions):
        raise ValueError("exposure audit implementation is not frozen")


def tally_labels(packed):
    """Masked queue/hold identity by class and index; unknown stays unknown."""
    classes, positions, labels = {}, {}, []
    eos = 0
    for index, (token, known) in enumerate(zip(packed["queue_tokens"], packed["queue_token_known"])):
        if not known:
            continue
        if token == EOS:
            eos += 1
            continue
        classes[PIECES[token]] = classes.get(PIECES[token], 0) + 1
        positions[str(index)] = positions.get(str(index), 0) + 1
        labels.append([index, token])
    hold_known = bool(packed["hold_piece_known"])
    return {"classes": classes, "positions": positions, "labels": labels, "eos": eos,
            "hold_known": hold_known, "hold_piece": int(packed["hold_piece"]) if hold_known else None,
            "queue_status": int(packed["queue_status"]), "hold_status": int(packed["hold_status"])}


def summarize(rows):
    flat = [row for _split, row in rows]
    distinct = lambda key: len({row[key] for row in flat})
    queue_classes, queue_positions, hold_classes = {}, {}, {}
    errors_by_class, errors_by_position = {}, {}
    hold_errors_by_class = {}
    correct = hold_correct = 0
    statuses, hold_statuses = {}, {}
    for _split, row in rows:
        tally, errors, hold_wrong = row["tally"], set(row["errors"]), row["hold_wrong"]
        for key, value in tally["classes"].items():
            queue_classes[key] = queue_classes.get(key, 0) + value
        for key, value in tally["positions"].items():
            queue_positions[key] = queue_positions.get(key, 0) + value
        statuses[tally["queue_status"]] = statuses.get(tally["queue_status"], 0) + 1
        hold_statuses[tally["hold_status"]] = hold_statuses.get(tally["hold_status"], 0) + 1
        for index, token in tally["labels"]:
            if index in errors:
                errors_by_position[str(index)] = errors_by_position.get(str(index), 0) + 1
                errors_by_class[PIECES[token]] = errors_by_class.get(PIECES[token], 0) + 1
            else:
                correct += 1
        if tally["hold_known"]:
            hold_classes[PIECES[tally["hold_piece"]]] = hold_classes.get(PIECES[tally["hold_piece"]], 0) + 1
            if hold_wrong:
                hold_errors_by_class[PIECES[tally["hold_piece"]]] = \
                    hold_errors_by_class.get(PIECES[tally["hold_piece"]], 0) + 1
            else:
                hold_correct += 1
    known = sum(queue_classes.values())
    hold_known = sum(hold_classes.values())
    return {"images": len(rows), "ancestry_groups": distinct("group_id"),
            "queue": {"supervised_piece_tokens": known, "eos_labels": sum(row["tally"]["eos"] for _s, row in rows),
                      "distinct_classes": len(queue_classes), "classes": dict(sorted(queue_classes.items())),
                      "positions": dict(sorted(queue_positions.items(), key=lambda item: int(item[0]))),
                      "errors_by_class": dict(sorted(errors_by_class.items())),
                      "errors_by_position": dict(sorted(errors_by_position.items(), key=lambda item: int(item[0]))),
                      "correct_tokens": correct,
                      "correct_by_class": {key: queue_classes[key] - errors_by_class.get(key, 0) for key in sorted(queue_classes)},
                      "target_statuses": dict(sorted(statuses.items()))},
            "hold": {"supervised_pieces": hold_known, "distinct_classes": len(hold_classes),
                     "classes": dict(sorted(hold_classes.items())), "correct_pieces": hold_correct,
                     "errors_by_class": dict(sorted(hold_errors_by_class.items())),
                     "target_statuses": dict(sorted(hold_statuses.items()))}}


def main():
    ceiling = require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--published", type=Path, default=PUBLISHED)
    args = parser.parse_args()
    require_owned_run(args.run)
    run = ROOT / "runs" / args.run
    summary = read(run / "summary.json")
    job = read(ROOT / "runs" / (args.run + "-job.json"))
    validate_completion(summary, job, allow_partial=True)
    code = run / "code"
    verify_source(code, summary["catalogue_sha256"])
    capacity = summary["config"].get("component_capacity", 0)
    if capacity != 64:
        raise ValueError("audit requires the completed capacity64 joint configuration")
    published = read(args.published)
    if published["run"] != args.run or published["component_capacity"] != capacity:
        raise ValueError("published diagnostics belong to another run/configuration")
    saved = {row["sample_id"]: row for row in published["cases"]}
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    sys.path[:0] = [str(code / "scripts"), str(code)]
    sys.dont_write_bytecode = True
    os.environ.update({"ONEDNN_PRIMITIVE_CACHE_CAPACITY": "16", "DNNL_PRIMITIVE_CACHE_CAPACITY": "16"})
    import torch
    from PIL import Image
    from m2_curriculum import replay_case
    from m2_stream_data import sample_case
    from tetris_vision.checkpoint import file_digest
    from tetris_vision.losses import targets_for
    from tetris_vision.preprocess import prepare_image
    require_frozen_callables(code, replay_case, sample_case, targets_for)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    curriculum = summary.get("curriculum", "stream-v1")
    rows, recipes_seen = [], 0
    for split in ("training", "development"):
        ledger = run / (split + "-probe.jsonl")
        if ledger.stat().st_size > 2 * 1024**2:
            raise ValueError("probe ledger byte budget exceeded")
        recipes = [json.loads(line) for line in ledger.read_text().splitlines()]
        expected = 9 if split == "training" else 18
        if len(recipes) != expected:
            raise ValueError("recorded probe population differs from the published set")
        for recipe in recipes:
            case = (replay_case(recipe, split=split, curriculum=curriculum, component_capacity=capacity)
                    if "curriculum" in summary else
                    sample_case(recipe["group_index"], recipe["view"], split, recipe["base_seed"], final_capture=True))
            if json.loads(json.dumps(case.record)) != recipe:
                raise ValueError("audit recipe/pixel/label identity differs")
            stem = run / (split + "-probe") / recipe["sample_id"]
            with Image.open(stem.with_suffix(".png")) as image:
                if (image.mode != case.frame.image.mode or image.size != case.frame.image.size or
                        image.tobytes() != case.frame.image.tobytes()):
                    raise ValueError("saved probe raster differs")
            if file_digest(stem.with_suffix(".json")) != recipe["state_sha256"]:
                raise ValueError("saved probe annotation differs")
            prepared = prepare_image(case.frame.image, max_side=512)
            target = targets_for(case.frame.state, prepared, component_capacity=capacity)
            if not bool(target["registered"].all()) or int(target["known"].sum()) != 200:
                raise ValueError("audited probe requires200 supported registered cells")
            components = target["components"]
            packed = {key: components[key][0].tolist() for key in
                      ("queue_tokens", "queue_token_known", "hold_piece", "hold_piece_known")}
            packed.update({"queue_status": int(components["queue_status"][0]),
                           "hold_status": int(components["hold_status"][0])})
            tally = tally_labels(packed)
            row = saved.get(recipe["sample_id"])
            if row is None:
                raise ValueError("published case missing for audited probe")
            errors = row["diagnostics"]["components"]["queue"]["piece_tokens"]["error_indices"]
            hold_wrong = row["diagnostics"]["components"]["hold"]["piece_correct"] is False
            if row["diagnostics"]["components"]["known_queue_tokens"] != sum(tally["classes"].values()) + tally["eos"]:
                raise ValueError("published identity exposure disagrees with rebuilt targets")
            rows.append((split, {"sample_id": recipe["sample_id"], "group_id": recipe["group_id"],
                                 "split": split, "tally": tally, "errors": errors, "hold_wrong": hold_wrong}))
            recipes_seen += 1
            (args.output / "progress.json").write_text(json.dumps({"cases": recipes_seen, "sample_id": recipe["sample_id"]}) + "\n")
    splits = {}
    for split in ("training", "development"):
        subsets = [row for label, row in rows if label == split]
        if not subsets:
            raise ValueError("split present without audited probes")
        splits[split] = summarize([(split, row) for row in subsets])
    checks = {
        "training_piece_tokens": (splits["training"]["queue"]["supervised_piece_tokens"], 54),
        "development_piece_tokens": (splits["development"]["queue"]["supervised_piece_tokens"], 90),
        "training_hold_known": (splits["training"]["hold"]["supervised_pieces"], 9),
        "development_hold_known": (splits["development"]["hold"]["supervised_pieces"], 12),
        "training_correct_tokens": (splits["training"]["queue"]["correct_tokens"], 3),
        "development_correct_tokens": (splits["development"]["queue"]["correct_tokens"], 9),
        "training_correct_holds": (splits["training"]["hold"]["correct_pieces"], 0),
        "development_correct_holds": (splits["development"]["hold"]["correct_pieces"], 3),
    }
    if any(actual != expected for actual, expected in checks.values()):
        raise ValueError("audit does not reproduce the published exposure/correctness counts: " + json.dumps(checks))
    result = {
        "status": "passed", "audit_format_version": 1, "run": args.run, "run_status": summary["status"],
        "curriculum": curriculum, "component_capacity": capacity, "images": len(rows),
        "job_ceiling_bytes": ceiling, "source_catalogue_sha256": summary["catalogue_sha256"],
        "weights_sha256": summary["checkpoint_sha256"],
        "published_diagnostics": args.published.relative_to(ROOT).as_posix(),
        "published_result_sha256": hashlib.sha256(args.published.read_bytes()).hexdigest(),
        "splits": splits,
        "published_counts_reproduced": {key: {"actual": actual, "published": expected}
                                        for key, (actual, expected) in checks.items()},
        "cases": [{"split": split, "sample_id": row["sample_id"], "group_id": row["group_id"],
                   "labels": row["tally"]["labels"], "hold_piece": row["tally"]["hold_piece"],
                   "hold_known": row["tally"]["hold_known"], "hold_wrong": row["hold_wrong"],
                   "error_indices": row["errors"]} for split, row in rows],
        "elapsed_seconds": time.monotonic() - started,
        "scope": ("Identity supervision exposure on the completed identity-factor run's recorded probes only. "
                  "Counts of masked labels, classes and positions; no inference, capacity, mask, target, order "
                  "or loss change, no training, and no complete-state, real-image or release claim."),
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with (args.output / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
