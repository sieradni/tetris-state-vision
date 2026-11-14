"""Three-factor decomposition of the locked-occupancy deficit (read-only).

`docs/M27_THREE_FACTOR.md` declares this instrument. It reads two committed receipts and
replays nothing: the registration-attribution receipt (per-case model logits, labels, the
known mask and the model-cell -> target-cell mapping) and the pixel-ceiling receipt (per-case
pixel read-out logits on the same boards and cells).

Three reads are scored on the same co-scoreable cells - the model at its own lattice, the
same weights with registration granted, and a sufficient pixel read-out - so the deficit
splits into registration, representation and boundary terms instead of being asserted to be
one of them. The pixel read-out is a function of the raster and the labels only, so it is
arm-independent; the model side is the arm whose mapping is committed.

No model, no forward pass, no weight, threshold, gate or label change.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from gate_power import f1_from_counts  # noqa: E402
from occupancy_sweep import pooled_auc, threshold_grid  # noqa: E402
from probe_model import require_bounded_job  # noqa: E402

DECIDING_BOARDS = 36
NONEMPTY_BOARDS = 28
OCCUPIED_CELLS = 667
EMPTY_CELLS = 4933
MATCHED_CELLS = 2986
ANCESTRY_GROUPS = 12
CELLS = 200
REGISTRATION_FLOOR = 0.10
REPRESENTATION_FLOOR = 0.10
JOIN_TOLERANCE = 1e-6
CEILING_BYTES = 128*1024**2
SOURCES = ("scripts/oracle_pairing_read.py",)


def digest(path, maximum=4*1024**2):
    value, consumed = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            consumed += len(block)
            if consumed > maximum:
                raise ValueError("decomposition source byte budget exceeded")
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def nonempty(cases):
    return [case for case in cases if any(case["occupied"])]


def pairs(cases, *, pairing, pixel=None):
    """(logit, label) rows for one pairing, on known target cells only.

    ``pairing='as_is'`` reads the label at the logit's own cell. ``pairing='oracle'`` reads
    the label at the target cell the model cell covers, and drops model cells that map
    outside the target board. ``pixel`` supplies a second cell-keyed logit list per case, so
    a pixel read can be scored on exactly the cells an oracle pairing selects.
    """
    rows = []
    for case in nonempty(cases):
        logits = case[("pixel_logits" if pixel else "logits")]
        if len(logits) != CELLS:
            raise ValueError("every deciding board must carry two hundred cells")
        if pixel and case.get("known_cells", CELLS) != CELLS:
            raise ValueError("a pixel read-out board must carry every known cell")
        for index in range(CELLS):
            if pairing == "as_is":
                target = index
            else:
                target = case["mapping"][index]
                if target is None:
                    continue
            if not pixel and not case["known"][target]:
                continue
            value = case["pixel_logits"][target] if pixel else logits[index]
            rows.append((float(value), bool(case["occupied"][target])))
    return rows


def score(rows):
    """Pooled AUC, F1 at 0.0 and the best point of the standing grid."""
    positive = [value for value, label in rows if label]
    negative = [value for value, label in rows if not label]
    if not positive or not negative:
        raise ValueError("the scored cells need both classes")
    at_zero = counts_at(rows, 0.0)
    best = None
    for threshold in threshold_grid():
        counts = counts_at(rows, threshold)
        position = f1_from_counts(counts["true_positive"], counts["false_positive"],
                                  counts["false_negative"])
        predicted = counts["true_positive"]+counts["false_positive"]
        precision = counts["true_positive"]/predicted if predicted else 0.0
        if (best is None or position > best["f1"]+1e-12 or
                (abs(position-best["f1"]) <= 1e-12 and precision > best["precision"]+1e-12)):
            best = {"threshold": threshold, "f1": position, "precision": precision,
                    "counts": counts}
    return {"cells": len(rows), "occupied": len(positive), "empty": len(negative),
            "auc": pooled_auc(positive, negative),
            "f1_at_zero": at_zero["f1"],
            "counts_at_zero": {key: at_zero[key] for key in
                               ("true_positive", "false_positive", "false_negative")},
            "precision_at_zero": at_zero["precision"], "recall_at_zero": at_zero["recall"],
            "best": best}


def counts_at(rows, threshold):
    true_positive = false_positive = false_negative = 0
    for value, label in rows:
        occupied = value > threshold
        if label and occupied:
            true_positive += 1
        elif label:
            false_negative += 1
        elif occupied:
            false_positive += 1
    counts = {"true_positive": true_positive, "false_positive": false_positive,
              "false_negative": false_negative}
    predicted = true_positive+false_positive
    observed = true_positive+false_negative
    counts["f1"] = f1_from_counts(true_positive, false_positive, false_negative)
    counts["precision"] = true_positive/predicted if predicted else 0.0
    counts["recall"] = true_positive/observed if observed else 0.0
    return counts


def require_close(name, measured, published, tolerance=JOIN_TOLERANCE):
    if not math.isfinite(measured) or abs(measured-published) > tolerance:
        raise ValueError(f"join refused: {name} recomputes {measured!r}, receipt publishes {published!r}")
    return measured


def matched_rows(cases, pixel_cases, *, pairing):
    """Oracle/as-is rows on model cells, plus the pixel read on the same target cells."""
    pixels = {case["sample_id"]: case for case in pixel_cases}
    model, pixel = [], []
    for case in nonempty(cases):
        other = pixels.get(case["sample_id"])
        if other is None or len(other["pixel_logits"]) != CELLS:
            raise ValueError("the pixel receipt does not carry every deciding board")
        for index in range(CELLS):
            target = index if pairing == "as_is" else case["mapping"][index]
            if target is None or not case["known"][target]:
                continue
            label = bool(case["occupied"][target])
            model.append((float(case["logits"][index]), label))
            pixel.append((float(other["pixel_logits"][target]), label))
    return model, pixel


def decomposition(reads):
    registration = reads["model_oracle"]["best"]["f1"]-reads["model_matched_as_is"]["best"]["f1"]
    representation = reads["pixel_matched"]["best"]["f1"]-reads["model_oracle"]["best"]["f1"]
    boundary = reads["model_oracle"]["best"]["f1"]-reads["model_oracle"]["f1_at_zero"]
    if registration >= REGISTRATION_FLOOR and representation >= REPRESENTATION_FLOOR:
        verdict = "two-factor-first-order"
    elif registration >= REGISTRATION_FLOOR:
        verdict = "registration-first-order"
    elif representation >= REPRESENTATION_FLOOR:
        verdict = "representation-first-order"
    else:
        verdict = "single-deficit"
    return {"registration_gain": round(registration, 6),
            "representation_gap": round(representation, 6),
            "boundary_gap": round(boundary, 6),
            "floors": {"registration": REGISTRATION_FLOOR,
                       "representation": REPRESENTATION_FLOOR},
            "verdict": verdict}


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--pixel-ceiling", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registration = read(ROOT/"runs"/args.registration/"result.json")
    pixel = read(ROOT/"runs"/args.pixel_ceiling/"result.json")
    for name, receipt in (("registration", registration), ("pixel ceiling", pixel)):
        if receipt.get("status") != "passed":
            raise ValueError(f"the committed {name} receipt did not pass")
    if registration.get("forwards") != DECIDING_BOARDS:
        raise ValueError("the registration receipt is not the 36-board population")
    cases, pixel_cases = registration["cases"], pixel["cases"]
    deciding = registration.get("population", {})
    if (deciding.get("nonempty_boards"), deciding.get("occupied_cells")) != (
            NONEMPTY_BOARDS, OCCUPIED_CELLS):
        raise ValueError("the registration receipt is not the declared deciding population")
    if len(nonempty(cases)) != NONEMPTY_BOARDS or len(pixel_cases) != NONEMPTY_BOARDS:
        raise ValueError("both receipts must carry the 28 nonempty boards")
    if sum(sum(case["occupied"]) for case in cases) != OCCUPIED_CELLS:
        raise ValueError("the occupied cell count differs from the declared 667")
    if sum(sum(case["occupied"]) for case in pixel_cases) != OCCUPIED_CELLS:
        raise ValueError("the pixel receipt's occupied count differs from the declared 667")

    full_as_is = score(pairs(cases, pairing="as_is"))
    matched_as_is, pixel_as_is_subset = matched_rows(cases, pixel_cases, pairing="as_is")
    matched_oracle, pixel_same = matched_rows(cases, pixel_cases, pairing="oracle")
    if len(matched_as_is) != MATCHED_CELLS or len(matched_oracle) != MATCHED_CELLS:
        raise ValueError(f"the matched population is not the declared {MATCHED_CELLS} cells")
    published = registration["summary"]
    require_close("model as-is matched F1", score(matched_as_is)["f1_at_zero"],
                  published["as_is"]["f1"])
    require_close("model as-is matched AUC", score(matched_as_is)["auc"],
                  published["as_is"]["auc"])
    require_close("model oracle matched F1", score(matched_oracle)["f1_at_zero"],
                  published["oracle"]["f1"])
    require_close("model oracle matched AUC", score(matched_oracle)["auc"],
                  published["oracle"]["auc"])
    require_close("model as-is full F1", full_as_is["f1_at_zero"],
                  published["as_is_full"]["f1"])
    require_close("model as-is full AUC", full_as_is["auc"], published["as_is_full"]["auc"])
    pixel_full = score(pairs(pixel_cases, pairing="as_is", pixel=True))
    require_close("pixel full F1", pixel_full["f1_at_zero"],
                  pixel["measured"]["operating"]["f1"])
    require_close("pixel full AUC", pixel_full["auc"], pixel["measured"]["pooled_auc"])
    for name, rows in (("pixel matched as-is", pixel_as_is_subset), ("pixel matched", pixel_same)):
        if score(rows)["cells"] != MATCHED_CELLS:
            raise ValueError(f"{name} does not cover the matched population")
    if len({case["group_id"] for case in cases}) != ANCESTRY_GROUPS:
        raise ValueError("the deciding population is not the twelve ancestry groups")

    reads = {"model_full_as_is": full_as_is,
             "model_matched_as_is": score(matched_as_is),
             "model_oracle": score(matched_oracle),
             "pixel_matched": score(pixel_same),
             "pixel_full": pixel_full}
    reading = decomposition(reads)
    result = {
        "status": "passed", "instrument": "m27-three-factor-r1",
        "predeclaration": "docs/M27_THREE_FACTOR.md",
        "registration_receipt": args.registration,
        "pixel_ceiling_receipt": args.pixel_ceiling,
        "model_weights_sha256": registration["weights_sha256"],
        "pixel_arm_independent": True,
        "population": {"deciding_boards": DECIDING_BOARDS, "nonempty_boards": NONEMPTY_BOARDS,
                       "anchored_cells": OCCUPIED_CELLS+EMPTY_CELLS,
                       "occupied_cells": OCCUPIED_CELLS, "empty_cells": EMPTY_CELLS,
                       "matched_cells": MATCHED_CELLS,
                       "matched_share": round(MATCHED_CELLS/(OCCUPIED_CELLS+EMPTY_CELLS), 6),
                       "ancestry_groups": ANCESTRY_GROUPS},
        "joins": {"model_matched_as_is": {"f1": published["as_is"]["f1"],
                                          "auc": published["as_is"]["auc"]},
                  "model_oracle": {"f1": published["oracle"]["f1"],
                                   "auc": published["oracle"]["auc"]},
                  "model_full_as_is": {"f1": published["as_is_full"]["f1"],
                                       "auc": published["as_is_full"]["auc"]},
                  "pixel_full": {"f1": pixel["measured"]["operating"]["f1"],
                                 "auc": pixel["measured"]["pooled_auc"]}},
        "reads": reads, "reading": reading,
        "sources": {name: digest(ROOT/name) for name in SOURCES},
        "scope": ("Read-only decomposition of the locked-occupancy deficit on the committed "
                  "deciding boards: two receipts joined, no forward pass, no training, no "
                  "weight, threshold, gate or label change, and no accuracy, real-image or "
                  "complete-state claim."),
    }
    job = read(ROOT/"runs"/(args.output.name+"-job.json"))
    if job.get("status") != "passed" or job.get("exit_code") != 0:
        raise ValueError("the bounded job did not pass")
    if job.get("memory_bytes") != CEILING_BYTES:
        raise ValueError("the job ceiling differs from the declared 128 MiB")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"scope.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                                          encoding="utf-8")
    print(json.dumps({"reading": reading,
                      "reads": {name: {"auc": round(value["auc"], 6),
                                       "f1_at_zero": round(value["f1_at_zero"], 6),
                                       "best": round(value["best"]["f1"], 6),
                                       "threshold": value["best"]["threshold"],
                                       "cells": value["cells"]}
                                 for name, value in reads.items()}}, indent=2))


if __name__ == "__main__":
    main()
