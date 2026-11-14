"""Compare the standing global gate with the declared per-board rule on one frozen arm (read-only).

`docs/M31_DECISION_RULE.md` declares this instrument. Three reads are scored on the identical 2,986
fixed cells and both fixed label vectors:

(a) the unchanged global `0.0` boundary, exactly as the gate scores it;
(b) the declared label-free per-board rule - the committed `otsu_board` crossing of each board's own
    visible logits, taking no label at all;
(c) an oracle per-board boundary fitted on the scored labels, reported only as an upper bound and
    excluded from every verdict and gate.

No training, no forward pass, no weight, threshold-grid, gate or label change.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(ROOT/"src"))

from oracle_pairing_read import MATCHED_CELLS, require_close, score  # noqa: E402
from pixel_query_deciding_read import population, rows_for  # noqa: E402
from probe_model import require_bounded_job  # noqa: E402
from tetris_vision.model import ModelConfig, otsu_board  # noqa: E402

REFERENCE_OWN = 0.543974
REFERENCE_PIXEL = 0.987179
CONTROL_BEST = 0.219626
CELL_GATE = 0.70
AUC_GATE = 0.90
ATTRIBUTION_FLOOR = 0.05
CEILING_BYTES = 640*1024**2
GRID = [round(-8+0.01*index, 2) for index in range(1601)]


def read(path):
    return json.loads(Path(path).read_text())


def boards(arm, fixed, pairing):
    """Rows grouped by board, with the labels of the chosen fixed vector."""
    by_board = {case["sample_id"]: case for case in arm["cases"]}
    grouped = {}
    for sample_id, index, _, identity, anchored in fixed:
        grouped.setdefault(sample_id, []).append(
            (float(by_board[sample_id]["logits"][index]),
             identity if pairing == "identity" else anchored))
    return grouped


def label_free_offsets(arm, visible_rows):
    """The committed per-board crossing of each board's own visible logits."""
    offsets = {}
    for case in arm["cases"]:
        logits = torch.tensor(case["logits"], dtype=torch.float64)
        if logits.numel() != len(case["logits"]) or logits.numel() % 10:
            raise ValueError("unexpected logit shape")
        grid = logits.reshape(-1, 10)[None]
        shifted = otsu_board(grid, visible_rows)
        offsets[case["sample_id"]] = float((grid-shifted).reshape(-1)[0])
    return offsets


def oracle_offsets(grouped):
    """Per-board best F1 boundary from the standing grid - an upper bound, never a verdict."""
    offsets = {}
    for sample_id, rows in grouped.items():
        best, best_threshold = -1.0, 0.0
        for threshold in GRID:
            tp = sum(1 for value, label in rows if value > threshold and label)
            fp = sum(1 for value, label in rows if value > threshold and not label)
            fn = sum(1 for value, label in rows if value <= threshold and label)
            position = 2*tp/(2*tp+fp+fn) if tp else 0.0
            if position > best+1e-12:
                best, best_threshold = position, threshold
        offsets[sample_id] = best_threshold
    return offsets


def flatten(grouped, offsets=None):
    out = []
    for sample_id, rows in grouped.items():
        shift = 0.0 if offsets is None else offsets[sample_id]
        out.extend((value-shift, label) for value, label in rows)
    return out


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="m30-level-calibrate-r5")
    parser.add_argument("--registration", default="-registration")
    parser.add_argument("--control", default="m28-pixelread-base-r1")
    parser.add_argument("--reference", default="m3-occscale-registration-r2")
    parser.add_argument("--pixel-ceiling", default="m3-occpix-ceiling-r1-r2")
    parser.add_argument("--m30", type=Path,
                        default=ROOT/"docs/evidence/m30-operating-point-r1/scope.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    m30 = read(args.m30)
    reference = read(ROOT/"runs"/args.reference/"result.json")
    pixel = read(ROOT/"runs"/args.pixel_ceiling/"result.json")
    fixed = population(reference)
    if len(fixed) != MATCHED_CELLS:
        raise ValueError("the fixed population is not the declared 2,986 cells")
    if (sum(1 for _, _, _, identity, _ in fixed if identity),
            sum(1 for _, _, _, _, anchored in fixed if anchored)) != (214, 309):
        raise ValueError("the label vectors are not the declared 214/309 occupied cells")
    pixel_read = score([(float(next(c for c in pixel["cases"] if c["sample_id"] == sample)
                               ["pixel_logits"][target]), anchored)
                        for sample, _, target, _, anchored in fixed])
    require_close("pixel read", pixel_read["best"]["f1"], REFERENCE_PIXEL)
    reference_own = score(rows_for(reference["cases"], fixed, "identity"))
    require_close("reference own", reference_own["best"]["f1"], REFERENCE_OWN)

    arm = read(ROOT/"runs"/(args.arm+args.registration)/"result.json")
    job = read(ROOT/"runs"/(args.arm+"-job.json"))
    if arm.get("status") != "passed" or job.get("status") != "passed":
        raise ValueError("the frozen arm's receipts did not pass")
    control = read(ROOT/"runs"/(args.control+"-registration-r1")/"result.json")
    control_own = score(rows_for(control["cases"], fixed, "identity"))
    require_close("control own", control_own["best"]["f1"], CONTROL_BEST)

    visible_rows = ModelConfig.pilot().visible_rows
    offsets = label_free_offsets(arm, visible_rows)
    reads = {}
    for pairing in ("identity", "anchored"):
        grouped = boards(arm, fixed, pairing)
        global_read = score(flatten(grouped))
        free = score(flatten(grouped, offsets))
        oracle = score(flatten(grouped, oracle_offsets(grouped)))
        stored = m30["variants"]["calibrate"]["own"]
        expected = stored["f1_at_zero"] if pairing == "identity" else None
        if expected is not None:
            require_close("global read joins the M30 receipt", global_read["f1_at_zero"], expected)
        per_board = [len(rows) for rows in grouped.values()]
        if sum(per_board) != 2986:
            raise ValueError("the board grouping does not cover the fixed population")
        reads[pairing] = {
            "global": global_read, "label_free": free, "oracle": oracle,
            "boards": len(grouped),
            "label_free_f1_at_zero": round(free["f1_at_zero"], 6),
            "oracle_f1_at_zero": round(oracle["f1_at_zero"], 6),
            "label_free_best": round(free["best"]["f1"], 6),
            "oracle_best": round(oracle["best"]["f1"], 6),
            "global_f1_at_zero": round(global_read["f1_at_zero"], 6),
            "global_best": round(global_read["best"]["f1"], 6),
        }

    identity = reads["identity"]
    gate = {"per_cell_f1": identity["global_f1_at_zero"] >= CELL_GATE,
            "ranking": (identity["global"]["auc"] >= AUC_GATE and
                        identity["global_best"] >= CELL_GATE),
            "attribution": (identity["global_best"]-control_own["best"]["f1"]) >= ATTRIBUTION_FLOOR}
    branch = "decision-rule" if identity["label_free_f1_at_zero"] >= CELL_GATE else "representation"
    result = {
        "status": "passed", "instrument": "m31-decision-rule-r1",
        "predeclaration": "docs/M31_DECISION_RULE.md",
        "population": {"fixed_cells": len(fixed), "occupied_identity": 214,
                       "occupied_anchored": 309, "boards": identity["boards"]},
        "arm": {"run": args.arm, "weights_sha256": arm["weights_sha256"],
                "registration": args.registration, "visible_rows": visible_rows,
                "job_name": job.get("job_name")},
        "anchors": {"control_best": round(control_own["best"]["f1"], 6),
                    "reference_own_best": REFERENCE_OWN, "pixel_best": REFERENCE_PIXEL,
                    "global_gate_best": m30["anchors"]["anchor_best"]},
        "reads": reads,
        "gate": gate, "cleared": all(gate.values()),
        "decision": {
            "branch": branch, "cell_gate": CELL_GATE,
            "reading": ("A passing label-free per-board read is a decision-rule change, not evidence "
                        "that the read improved: the unchanged gate keeps failing on the global read."
                        if branch == "decision-rule" else
                        "The per-board boundary does not reach the per-cell gate, so the deficit is the "
                        "read's own per-cell scores and the declared structural follow-up is the "
                        "context-carrying hidden projection."),
            "oracle_excluded": True,
            "oracle_role": "diagnostic upper bound only; excluded from the gate, the branch and every verdict",
        },
        "scope": ("Read-only scoring of one frozen arm's committed logits under three boundaries on "
                  "one fixed population; no training, no forward pass, no weight, threshold-grid, gate "
                  "or label change."),
    }
    receipt = [json.loads(line) for line in
               (ROOT/"runs"/(args.output.name+"-job.json")).read_text().splitlines() if line.strip()]
    if not any(record.get("memory_bytes") == CEILING_BYTES for record in receipt):
        raise ValueError("the reader ceiling differs from the declared 640 MiB")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"scope.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                                          encoding="utf-8")
    print(json.dumps({"branch": branch, "gate": gate,
                      "identity": {key: identity[key] for key in
                                   ("global_f1_at_zero", "global_best", "label_free_f1_at_zero",
                                    "label_free_best", "oracle_f1_at_zero", "oracle_best")},
                      "anchored": {key: reads["anchored"][key] for key in
                                   ("global_f1_at_zero", "global_best", "label_free_f1_at_zero",
                                    "label_free_best", "oracle_f1_at_zero", "oracle_best")}},
                     indent=2))


if __name__ == "__main__":
    main()
