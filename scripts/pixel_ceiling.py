"""Pixel-only separability ceiling for the committed locked-occupancy labels.

`docs/M3_OCCUPANCY_PIXEL_CEILING.md` declares this instrument once. It is read-only: no
model, no training and no label policy. It asks one question the occupancy campaign has
never tested - whether the committed determinate labels are recoverable from the captured
raster at all - by fitting a small logistic read-out on per-cell pixel features with one
ancestry group held out, and reporting the same cell-F1 / AUC quantities the model gates
use, on the same cells.

The declared statistics matter (V-066): a leave-one-group-out score can be near-constant
within a fold while its pooled AUC across folds looks high, and an unregularized
least-squares score extrapolates past any fixed threshold grid. So the governing quantity
is the pooled count-based F1 at the fit's own 0.5 boundary, the label-shuffled control runs
the identical pipeline, and per-fold mean AUC is reported beside the descriptive pooled
AUC. No verdict reads the pooled AUC.
"""
import numpy as np

F1_THRESHOLD = 0.70
AUC_THRESHOLD = 0.90
SHUFFLE_SEED = 20261026
RIDGE = 1.0
ITERATIONS = 40
FEATURE_CELLS = 3
FEATURE_WIDTH = FEATURE_CELLS*FEATURE_CELLS*3+3


def raster(image):
    """RGB float array of a captured frame; the only raster entry point."""
    return np.asarray(image.convert("RGB"), dtype=np.float64)


def cell_features(pixels, lattice):
    """Per-cell pixel features, in the lattice's own image-space units.

    3x3 interior sRGB means (27) plus per-channel interior standard deviation (3). Only
    the raster and the target lattice are read; no state label enters this function.
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("a decoded RGB raster is required")
    height, width = pixels.shape[:2]
    origin = np.asarray(lattice.origin, dtype=np.float64)
    u = np.asarray(lattice.u, dtype=np.float64)
    v = np.asarray(lattice.v, dtype=np.float64)
    if min(abs(u).max(), abs(v).max()) < 2:
        raise ValueError("cell lattice is below pixel scale")
    extent = .3*(np.abs(u)+np.abs(v))
    rows = []
    for r in range(20):
        for c in range(10):
            center = origin+(c+.5)*u+(r+.5)*v
            x0, y0 = np.floor(center-extent).astype(int)
            x1, y1 = np.ceil(center+extent).astype(int)
            patch = pixels[max(0, y0):min(height, y1), max(0, x0):min(width, x1)]
            if patch.size == 0:
                raise ValueError("cell patch lies outside the captured raster")
            values = []
            for row in range(FEATURE_CELLS):
                for column in range(FEATURE_CELLS):
                    block = patch[int(row*len(patch)/FEATURE_CELLS):int((row+1)*len(patch)/FEATURE_CELLS),
                                  int(column*patch.shape[1]/FEATURE_CELLS):int((column+1)*patch.shape[1]/FEATURE_CELLS)]
                    values.extend(block.reshape(-1, 3).mean(0))
            values.extend(patch.reshape(-1, 3).std(0))
            rows.append(values)
    features = np.array(rows)
    if features.shape != (200, FEATURE_WIDTH) or not np.isfinite(features).all():
        raise ValueError("invalid cell feature matrix")
    return features


def tie_aware_auc(scores, labels):
    """Rank AUC with averaged ties; constant scores are exactly chance."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(bool)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores)+1)
    values = scores[order]
    start = 0
    for index in range(1, len(scores)+1):
        if index == len(scores) or values[index] != values[start]:
            if index-start > 1:
                ranks[order[start:index]] = (start+1+index)/2
            start = index
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        raise ValueError("AUC requires both label classes")
    return float((ranks[labels].sum()-positive*(positive+1)/2)/(positive*negative))


def counts(logits, labels, threshold=0.):
    predicted = np.asarray(logits) > threshold
    labels = np.asarray(labels).astype(bool)
    return {"true_positive": int((predicted & labels).sum()),
            "false_positive": int((predicted & ~labels).sum()),
            "false_negative": int((~predicted & labels).sum())}


def f1(position):
    true_positive, false_positive, false_negative = (
        position["true_positive"], position["false_positive"], position["false_negative"])
    return 2*true_positive/max(1, 2*true_positive+false_positive+false_negative)


def precision(position):
    return position["true_positive"]/max(1, position["true_positive"]+position["false_positive"])


def recall(position):
    return position["true_positive"]/max(1, position["true_positive"]+position["false_negative"])


def threshold_grid(low=-6., high=6., step=.15):
    return tuple(round(low+index*step, 2) for index in range(int(round((high-low)/step))+1))


def sweep(logits, labels):
    positions = [counts(logits, labels, threshold) for threshold in threshold_grid()]
    scores = [f1(position) for position in positions]
    best = int(np.argmax(scores))
    return {"f1": scores[best], "threshold": threshold_grid()[best], "position": positions[best]}


def logistic_fit(features, labels, groups):
    """Leave-one-group-out logistic regression on standardized cell features."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    groups = np.asarray(groups)
    names = sorted(set(groups.tolist()))
    if len(names) < 2:
        raise ValueError("at least two ancestry groups are required")
    logits = np.zeros(len(features))
    per_fold = {}
    for name in names:
        train, test = groups != name, groups == name
        if not labels[train].any() or not (~labels[train].astype(bool)).any():
            raise ValueError("every training fold needs both classes")
        mean = features[train].mean(0)
        scale = features[train].std(0)
        scale[scale < 1e-9] = 1.
        train_matrix = np.c_[np.clip((features[train]-mean)/scale, -8, 8), np.ones(int(train.sum()))]
        test_matrix = np.c_[np.clip((features[test]-mean)/scale, -8, 8), np.ones(int(test.sum()))]
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
        logits[test] = test_matrix @ weights
        per_fold[name] = {"cells": int(test.sum()), "occupied": int(labels[test].sum()),
                          "auc": tie_aware_auc(logits[test], labels[test])}
    return logits, per_fold


def shuffled_labels(labels, groups, seed=SHUFFLE_SEED):
    """Within-group label permutation: same counts per image, no spatial mapping."""
    labels = np.asarray(labels).copy()
    groups = np.asarray(groups)
    generator = np.random.default_rng(seed)
    for name in sorted(set(groups.tolist())):
        mask = groups == name
        labels[mask] = generator.permutation(labels[mask])
    return labels


def evaluate(features, labels, groups):
    logits, per_fold = logistic_fit(features, labels, groups)
    operating = counts(logits, labels, 0.)
    return {"logits": logits, "per_fold": per_fold,
            "pooled_auc": tie_aware_auc(logits, labels),
            "fold_mean_auc": float(np.mean([row["auc"] for row in per_fold.values()])),
            "operating": {**operating, "f1": f1(operating), "precision": precision(operating),
                          "recall": recall(operating)},
            "best": sweep(logits, labels)}


def ceiling_verdict(measured, control):
    """The declared rule. Separable needs the F1 bar and the AUC bar on held-out groups.

    The control must collapse under the identical pipeline; if it does not, the instrument
    is invalid and the pass is recorded as such rather than as a measurement.
    """
    if control["operating"]["f1"] >= F1_THRESHOLD or control["fold_mean_auc"] >= AUC_THRESHOLD:
        return {"status": "instrument-invalid", "reason": "label-shuffled control did not collapse"}
    if measured["operating"]["f1"] >= F1_THRESHOLD and measured["fold_mean_auc"] >= AUC_THRESHOLD:
        return {"status": "separable", "f1_gap": round(F1_THRESHOLD-measured["operating"]["f1"], 6),
                "auc_gap": round(AUC_THRESHOLD-measured["fold_mean_auc"], 6)}
    return {"status": "not-separable",
            "f1_gap": round(F1_THRESHOLD-measured["operating"]["f1"], 6),
            "auc_gap": round(AUC_THRESHOLD-measured["fold_mean_auc"], 6)}
