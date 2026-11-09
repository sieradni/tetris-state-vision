"""Registration attribution: would correct cell placement recover the occupancy decisions?

`docs/M3_OCCUPANCY_SCALED.md` declares this instrument as the fallback for the named alternative
suspect. It is a read-only pairing diagnostic: each model cell's centre is located in the *target*
cell coordinates using the target lattice, so a logit can be compared with the label of the cell it
physically covers (oracle registration) and with the label at its own index (the standing gate
pairing). Both pairings are scored on the *same* cells, so the delta measures alignment only.

Nothing here reads a pixel, a hidden state or a new label: the pairing is a pure function of the two
lattices, which is why different hidden states yielding identical geometry pair identically.
"""
import math

from gate_power import f1_from_counts
from occupancy_sweep import pooled_auc

COLUMNS = 10
ROWS = 20
F1_FLOOR = 0.05
BOOTSTRAP_DRAWS = 400
BOOTSTRAP_SEED = 20261026


def lattice_of(value, name="lattice"):
    """Pull the (origin, u, v) basis out of its three two-vectors."""
    if type(value) is not dict:
        raise ValueError(f"{name} must be a mapping with origin/u/v")
    basis = []
    for key in ("origin", "u", "v"):
        vector = value.get(key)
        if not isinstance(vector, (list, tuple)) or len(vector) != 2:
            raise ValueError(f"{name} {key} must be two numbers")
        for entry in vector:
            if isinstance(entry, bool) or not isinstance(entry, (int, float)) or not math.isfinite(entry):
                raise ValueError(f"{name} {key} must be finite numbers")
        basis.append((float(vector[0]), float(vector[1])))
    origin, u, v = basis
    determinant = u[0]*v[1]-u[1]*v[0]
    if abs(determinant) < 1e-12:
        raise ValueError(f"{name} basis is degenerate")
    return origin, u, v, determinant


def centres(predicted):
    """Model cell centres in the shared normalized pixel frame, bottom-first rows."""
    origin, u, v, _ = lattice_of(predicted, "predicted")
    return [((origin[0]+(c+.5)*u[0]+(r+.5)*v[0], origin[1]+(c+.5)*u[1]+(r+.5)*v[1]), c, r)
            for r in range(ROWS) for c in range(COLUMNS)]


def map_cells(predicted, target):
    """Each model cell's target index (row-major) and its offset in target cells, or ``None``."""
    origin, u, v, determinant = lattice_of(target, "target")
    mapping, offsets = [], []
    for point, c, r in centres(predicted):
        dx, dy = point[0]-origin[0], point[1]-origin[1]
        column = (dx*v[1]-dy*v[0])/determinant
        row = (u[0]*dy-u[1]*dx)/determinant
        index = math.floor(column)
        line = math.floor(row)
        inside = 0 <= index < COLUMNS and 0 <= line < ROWS
        mapping.append(line*COLUMNS+index if inside else None)
        offsets.append((column-c-.5, row-r-.5))
    return mapping, offsets


def counts_of(pairs):
    true_positive = false_positive = false_negative = true_negative = 0
    for logit, label in pairs:
        if label and logit > 0.0:
            true_positive += 1
        elif label:
            false_negative += 1
        elif logit > 0.0:
            false_positive += 1
        else:
            true_negative += 1
    return {"true_positive": true_positive, "false_positive": false_positive,
            "false_negative": false_negative, "true_negative": true_negative}


def score(counts):
    f1 = f1_from_counts(counts["true_positive"], counts["false_positive"], counts["false_negative"])
    predicted = counts["true_positive"]+counts["false_positive"]
    observed = counts["true_positive"]+counts["false_negative"]
    return {"counts": counts, "f1": f1,
            "precision": counts["true_positive"]/predicted if predicted else 0.0,
            "recall": counts["true_positive"]/observed if observed else 0.0}


def pairs_for(case, pairing):
    """(logit, label) pairs under ``as_is`` (own index) or ``oracle`` (covered target cell)."""
    logits, occupied, known = case["logits"], case["occupied"], case["known"]
    mapping = case.get("mapping")
    if not (len(logits) == len(occupied) == len(known) == COLUMNS*ROWS):
        raise ValueError("case must carry one logit, label and known flag per board cell")
    if pairing == "as_is":
        return [(value, occupied[index]) for index, value in enumerate(logits) if known[index]]
    if pairing != "oracle":
        raise ValueError("pairing must be as_is or oracle")
    if not isinstance(mapping, list) or len(mapping) != COLUMNS*ROWS:
        raise ValueError("oracle pairing requires the stored mapping")
    pairs = []
    for index, value in enumerate(logits):
        target = mapping[index]
        if target is None or not known[target]:
            continue
        pairs.append((value, occupied[target]))
    return pairs


def matched_indices(case):
    """Model cells that both pairings can score: a mapped target cell that is known."""
    mapping, known = case.get("mapping") or [], case["known"]
    return [index for index, target in enumerate(mapping)
            if known[index] and target is not None and 0 <= target < len(known) and known[target]]


def matched_pairs(case, pairing):
    """Both pairings restricted to the cells they can score in common."""
    index = matched_indices(case)
    if pairing == "as_is":
        return [(case["logits"][i], case["occupied"][i]) for i in index]
    if pairing == "oracle":
        return [(case["logits"][i], case["occupied"][case["mapping"][i]]) for i in index]
    raise ValueError("pairing must be as_is or oracle")


def auc_of(pairs):
    positive = [value for value, label in pairs if label]
    negative = [value for value, label in pairs if not label]
    if not positive or not negative:
        return None
    return pooled_auc(positive, negative)


def summarize(cases):
    """Both pairings on matched cells, the full-set as-is reading, and the cluster interval."""
    if not cases:
        raise ValueError("at least one case is required")
    nonempty = [case for case in cases if any(case["occupied"])]
    if not nonempty:
        raise ValueError("the deciding population requires at least one nonempty board")
    matched = {"as_is": [], "oracle": []}
    full = []
    offsets, mapped, shifted = [], 0, 0
    for case in nonempty:
        for pairing in ("as_is", "oracle"):
            matched[pairing].extend(matched_pairs(case, pairing))
        full.extend(pairs_for(case, "as_is"))
        offsets.extend(abs(value) for pair in case["offsets"] for value in pair)
        mapped += sum(1 for target in case["mapping"] if target is not None)
        shifted += sum(1 for i, target in enumerate(case["mapping"])
                       if target is not None and target != i)
    total = COLUMNS*ROWS*len(nonempty)
    measured = {name: score(counts_of(pairs)) for name, pairs in matched.items()}
    measured["as_is_full"] = score(counts_of(full))
    measured["as_is"]["auc"] = auc_of(matched["as_is"])
    measured["oracle"]["auc"] = auc_of(matched["oracle"])
    measured["as_is_full"]["auc"] = auc_of(full)
    delta = measured["oracle"]["f1"]-measured["as_is"]["f1"]
    measured.update({
        "images": len(nonempty), "empty_images": len(cases)-len(nonempty),
        "cells": total, "matched_cells": len(matched["as_is"]),
        "mapped_share": mapped/total, "shifted_share": shifted/total,
        "mean_abs_offset_cells": sum(offsets)/len(offsets) if offsets else 0.0,
        "f1_delta": delta,
        "attribution": {"delta": delta, "bootstrap": paired_bootstrap(cases)}})
    return measured


def ancestry_of(sample_id):
    return "-".join(sample_id.split("-")[:2])


def paired_bootstrap(cases):
    """Whole-ancestry-cluster resampling of both pairings on the same sampled clusters."""
    from cluster_interval import cluster_bootstrap
    grouped = {}
    for case in (case for case in cases if any(case["occupied"])):
        grouped.setdefault(ancestry_of(case["sample_id"]), []).append(case)
    if len(grouped) < 2:
        raise ValueError("at least two ancestry clusters required")
    ordered = tuple(sorted(grouped.items()))

    def statistic(sample):
        rows = [case for cluster in sample for case in cluster]
        values = {}
        for pairing in ("as_is", "oracle"):
            pairs = [pair for case in rows for pair in matched_pairs(case, pairing)]
            auc = auc_of(pairs)
            if auc is None:
                return None
            values["f1_"+pairing] = score(counts_of(pairs))["f1"]
            values["auc_"+pairing] = auc
        values["f1_delta"] = values["f1_oracle"]-values["f1_as_is"]
        return values

    return cluster_bootstrap(statistic, ordered, ("f1_as_is", "f1_oracle", "auc_as_is",
                                                  "auc_oracle", "f1_delta"),
                             draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)


def attribution_verdict(summary, *, floor=F1_FLOOR):
    """``registration_implicated`` iff the point delta clears the floor and its interval excludes 0."""
    delta = summary["f1_delta"]
    interval = summary["attribution"]["bootstrap"]["f1_delta"]
    implicated = delta >= floor and interval["low"] > 0
    return {"verdict": "registration_implicated" if implicated else "registration_exonerated",
            "registration_implicated": implicated, "registered": implicated,
            "f1_delta": delta, "floor": floor, "f1_delta_interval": interval,
            "statement": ("integer-scale misregistration alone accounts for at least the declared "
                          "five F1 points" if implicated else
                          "correcting cell placement does not recover the declared five F1 points")}
