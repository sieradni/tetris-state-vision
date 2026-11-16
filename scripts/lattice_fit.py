"""M52's lattice refinement: fit the board's own cell lines instead of trusting a windowed search.

M51 measured the residual precisely: its box is right to about `0.7 %` of a cell in width on every board, but on four
boards the *pitch* the objective prefers is not the capture's pitch, and a pitch error of a few percent accumulates
across ten columns into a fraction of a cell at the far edge (and cost `0.075` F1 at the fixed crossing).

This module takes a detected box and refits it from the interior evidence, identically on every board:

- for each expected column and row boundary, the edge-energy profile inside the box is searched within a declared
  window around that boundary, and the boundary's sub-pixel centre is the profile's own centroid in that window;
- a boundary whose profile peak is far below the median boundary's is *missing* (a board drawn without a grid line
  there) and is not used;
- the pitch is the Theil-Sen slope over the used boundaries - a median of pairwise slopes, which one displaced line
  cannot move - and the origin is the box origin plus the mean residual of the used boundaries against the box's own
  expectation, so the box's pixel convention is preserved exactly;
- boundaries whose residual exceeds a declared allowance are dropped once and the fit is repeated;
- if too few boundaries localise on an axis, that axis keeps the detected box's pitch and origin, unchanged.

Nothing here reads a label, a prevalence, a scored cell, the model's lattice or the target lattice: the input is the
raster and the detected box, both label-free, and the output is a lattice.
"""
import numpy as np

COLUMNS, ROWS = 10, 20
SEARCH_SHARE = 0.35
MISSING_SHARE = 0.35
OUTLIER_CELLS = 0.25
MIN_LINES = {"x": 6, "y": 12}
CONSTANTS = {"search_share": SEARCH_SHARE, "missing_share": MISSING_SHARE, "outlier_cells": OUTLIER_CELLS,
             "min_lines_x": MIN_LINES["x"], "min_lines_y": MIN_LINES["y"]}


def profile(grey, axis):
    """Edge energy per line index along one axis: a cell boundary shows as a peak."""
    if axis == "x":
        return np.abs(np.diff(grey, axis=1)).sum(0)
    return np.abs(np.diff(grey, axis=0)).sum(1)


def localise(profile, expected, pitch):
    """The sub-pixel centre of the strongest boundary within `SEARCH_SHARE` of a cell of `expected`."""
    radius = max(1, int(round(SEARCH_SHARE*pitch)))
    lower = max(0, int(round(expected))-radius)
    upper = min(profile.size-1, int(round(expected))+radius)
    if upper-lower < 2:
        return None
    window = profile[lower:upper+1]
    peak = int(np.argmax(window))
    if peak == 0 or peak == window.size-1:
        return None
    weight = window[max(0, peak-1):peak+2]
    offsets = np.array([-1.0, 0.0, 1.0])[:weight.size]
    if weight.sum() <= 0:
        return None
    return float(lower+peak+float((offsets*weight).sum()/weight.sum()))


def theil_sen(indices, centres):
    """Median of pairwise slopes, and the intercept that centres the residuals."""
    slopes = [(centres[j]-centres[i])/(indices[j]-indices[i])
              for i in range(len(indices)) for j in range(i+1, len(indices))]
    slope = float(np.median(slopes))
    intercept = float(np.median(np.asarray(centres)-slope*np.asarray(indices)))
    return slope, intercept


def fit_axis(energy, origin, pitch, count):
    """Refine one axis: localised boundaries, one outlier pass, and the declared fallback."""
    indices, centres, peaks = [], [], []
    for index in range(count+1):
        expected = origin+index*pitch
        if expected < 1 or expected > energy.size-2:
            continue
        centre = localise(energy, expected, pitch)
        if centre is None:
            continue
        indices.append(index)
        centres.append(centre)
        peaks.append(float(energy[int(round(centre))]))
    if len(indices) < 3:
        return {"used": len(indices), "kept": False, "reason": "too few boundaries localised"}
    floor = MISSING_SHARE*float(np.median(peaks))
    used = [(index, centre) for index, centre, peak in zip(indices, centres, peaks) if peak >= floor]
    dropped_missing = [index for index, peak in zip(indices, peaks) if peak < floor]
    if len(used) < MIN_LINES["x" if count == COLUMNS else "y"]:
        return {"used": len(used), "kept": False, "reason": "too few boarded boundaries",
                "missing": dropped_missing}
    pairs = used
    slope, intercept = theil_sen([pair[0] for pair in pairs], [pair[1] for pair in pairs])
    residuals = [centre-(slope*index+intercept) for index, centre in pairs]
    allowance = OUTLIER_CELLS*abs(slope)
    kept = [(index, centre) for index, centre, residual in zip([p[0] for p in pairs], [p[1] for p in pairs],
                                                               residuals)
            if abs(residual-np.median(residuals)) <= allowance]
    dropped_outliers = [index for (index, _), residual in zip(pairs, residuals)
                        if abs(residual-np.median(residuals)) > allowance]
    if len(kept) >= MIN_LINES["x" if count == COLUMNS else "y"]:
        pairs = kept
        slope, intercept = theil_sen([pair[0] for pair in pairs], [pair[1] for pair in pairs])
    shift = float(np.mean([centre-(origin+index*pitch) for index, centre in pairs]))
    fitted = origin+shift
    residuals = [centre-(slope*index+intercept) for index, centre in pairs]
    return {"used": len(indices), "kept": True, "missing": dropped_missing, "outliers": dropped_outliers,
            "boundaries": len(pairs), "pitch": round(float(slope), 6), "origin": round(float(fitted), 6),
            "shift_pixels": round(float(shift), 6), "max_residual_pixels": round(float(np.abs(residuals).max()), 6)}


def refine(pixels, box):
    """The refined lattice: fitted pitch and origin per axis, with the box kept wherever the fit falls back."""
    pixels = np.asarray(pixels, dtype=np.float64)
    grey = pixels.mean(2)
    frame = box["frame"]
    pitch_x = float(box["u"][0])
    pitch_y = float(-box["v"][1])
    fit_x = fit_axis(profile(grey, "x"), float(frame["left"]), pitch_x, COLUMNS)
    fit_y = fit_axis(profile(grey, "y"), float(frame["top"]), pitch_y, ROWS)
    left = fit_x["origin"] if fit_x.get("kept") else float(frame["left"])
    top = fit_y["origin"] if fit_y.get("kept") else float(frame["top"])
    step_x = fit_x["pitch"] if fit_x.get("kept") else pitch_x
    step_y = fit_y["pitch"] if fit_y.get("kept") else pitch_y
    return {"origin": [left, top+ROWS*step_y], "u": [step_x, 0.0], "v": [0.0, -step_y],
            "frame": {"left": left, "right": left+COLUMNS*step_x, "top": top, "bottom": top+ROWS*step_y},
            "fit": {"x": fit_x, "y": fit_y}, "source": "refined"}
