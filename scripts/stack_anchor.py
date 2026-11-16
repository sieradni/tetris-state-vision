"""M53's stack-anchored lattice: the visible blocks' own edges place the board, not the box's phase.

M52 measured the residual precisely: refining a lattice *around* a detected box cannot reach a gross offset, because
its windows are `0.35` of a cell wide, and two captures sit `1.93` and `5.18` cells out. Those boards' boxes are in the
wrong place while their *blocks* are visible at the right place, sitting on the board's floor.

This module reads the blocks:

- the saturated mask is segmented into components, and each component's axis-aligned edges are candidate cell
  boundaries - a block occupies whole cells, so its edges lie on the lattice's lines;
- the pitch is a median over pairwise edge differences divided by their nearest whole number of cells, written per
  axis as `median({d / round(d / pitch_seed)})` restricted to pairs the seed explains to within `RATIO_TOLERANCE` -
  a median, so one displaced anchor cannot move it;
- the **vertical origin is the floor**: the stack rests on the board's bottom, so the lowest visible block edge is the
  board's `origin` row, which is the one absolute anchor the box cannot supply when its phase is wrong;
- the **horizontal origin is the box's left snapped to the anchors' own phase**: the circular median of the anchor
  positions modulo the fitted pitch, moved to the nearest multiple of the pitch from the box's left;
- per axis, too few anchors means the box's own value for that axis is kept, unchanged.

No label, prevalence, scored cell, model lattice or target lattice is read: the input is the raster and M51's label-free
box, and the output is a lattice.
"""
import numpy as np

COLUMNS, ROWS = 10, 20
SATURATION = 60.0
MIN_BLOCK_CELLS = 0.25
MAX_BLOCK_CELLS = 5.0
RATIO_TOLERANCE = 0.35
MIN_PAIRS = 3
MIN_ANCHORS = 2
FLOOR_TOLERANCE = 1.0
CONSTANTS = {"saturation": SATURATION, "min_block_cells": MIN_BLOCK_CELLS, "max_block_cells": MAX_BLOCK_CELLS,
             "floor_tolerance": FLOOR_TOLERANCE,
             "ratio_tolerance": RATIO_TOLERANCE, "min_pairs": MIN_PAIRS, "min_anchors": MIN_ANCHORS}


def components(mask):
    """Axis-aligned bounding boxes of the mask's connected components, by row runs and union-find."""
    parent = {}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    runs, previous = [], []
    boxes = {}
    for y in range(mask.shape[0]):
        row = mask[y]
        edges = np.flatnonzero(np.diff(np.r_[0, row.view(np.int8), 0]))
        current = []
        for start, stop in zip(edges[0::2], edges[1::2]):
            identifier = len(parent)
            parent[identifier] = identifier
            boxes[identifier] = [int(start), int(stop)-1, y, y]
            runs.append(identifier)
            current.append((identifier, int(start), int(stop)-1))
        for identifier, start, stop in current:
            for other, other_start, other_stop in previous:
                if start <= other_stop and other_start <= stop:
                    union(identifier, other)
        previous = current
    merged = {}
    for identifier, (x0, x1, y0, y1) in boxes.items():
        root = find(identifier)
        if root not in merged:
            merged[root] = [x0, x1, y0, y1]
        else:
            box = merged[root]
            merged[root] = [min(box[0], x0), max(box[1], x1), min(box[2], y0), max(box[3], y1)]
    return sorted(merged.values())


def blocks(pixels, pitch_x, pitch_y, frame):
    """The visible blocks: saturated components whose centre falls inside the box (grown by a cell).

    A run of same-coloured blocks merges into one component, so no upper size bound is imposed - the merged stack's
    own bounding box still has cell boundaries as its edges, and that is exactly the anchor the floor needs.
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    mask = (pixels.max(2)-pixels.min(2)) > SATURATION
    grown = {"left": float(frame["left"])-pitch_x, "right": float(frame["right"])+pitch_x,
             "top": float(frame["top"])-pitch_y, "bottom": float(frame["bottom"])+pitch_y}
    found = []
    for x0, x1, y0, y1 in components(mask):
        width, height = x1-x0+1, y1-y0+1
        if width < MIN_BLOCK_CELLS*pitch_x or height < MIN_BLOCK_CELLS*pitch_y:
            continue
        centre_x, centre_y = (x0+x1)/2.0, (y0+y1)/2.0
        if not (grown["left"] <= centre_x <= grown["right"] and grown["top"] <= centre_y <= grown["bottom"]):
            continue
        found.append({"left": float(x0), "right": float(x1+1), "top": float(y0), "bottom": float(y1+1)})
    return found


def anchors(found):
    """The cell boundaries the blocks pin, per axis."""
    x = sorted({edge for box in found for edge in (box["left"], box["right"])})
    y = sorted({edge for box in found for edge in (box["top"], box["bottom"])})
    return x, y


def pitch_from(positions, seed):
    """Median over pairwise spacings divided by their nearest whole number of cells."""
    candidates = []
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            difference = abs(positions[j]-positions[i])
            if difference < 0.5*seed:
                continue
            multiple = max(1, int(round(difference/seed)))
            if multiple > ROWS:
                continue
            if abs(difference/multiple-seed) <= RATIO_TOLERANCE*seed:
                candidates.append(difference/multiple)
    if len(candidates) < MIN_PAIRS:
        return None, len(candidates)
    return float(np.median(candidates)), len(candidates)


def circular_median(values, period):
    """The median of angles, folded onto a period, as an offset in `[0, period)`."""
    angles = np.asarray(values, dtype=float)*2.0*np.pi/period
    total = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    return float((total % (2.0*np.pi))*period/(2.0*np.pi))


def anchor_lattice(pixels, box):
    """The stack-anchored lattice, with M51's box kept on any axis the anchors cannot place."""
    frame = box["frame"]
    pitch_x = float(box["u"][0])
    pitch_y = float(-box["v"][1])
    found = blocks(pixels, pitch_x, pitch_y, frame)
    left_edges, top_edges = anchors(found)
    report = {"blocks": len(found), "box": {"left": float(frame["left"]), "top": float(frame["top"]),
                                            "pitch_x": pitch_x, "pitch_y": pitch_y}}

    fitted_x, pairs_x = pitch_from(left_edges, pitch_x)
    fitted_y, pairs_y = pitch_from(top_edges, pitch_y)
    report["pairs"] = {"x": pairs_x, "y": pairs_y}

    if fitted_x is not None and len(left_edges) >= MIN_ANCHORS:
        phase = circular_median(left_edges, fitted_x)
        offset = phase-fmod(frame["left"], fitted_x)
        offset -= fitted_x*round(offset/fitted_x)
        left = float(frame["left"]+offset)
        report["x"] = {"kept": True, "pitch": round(fitted_x, 6), "origin": round(left, 6),
                       "phase_pixels": round(phase, 6), "snap_pixels": round(offset, 6),
                       "anchors": len(left_edges)}
    else:
        left, fitted_x = float(frame["left"]), pitch_x
        report["x"] = {"kept": False, "reason": "too few horizontal anchors", "anchors": len(left_edges)}

    if fitted_y is not None and len(top_edges) >= MIN_ANCHORS:
        # the stack's lowest edge is the floor to within the drawing's inset, so it corrects the box by whole cells
        # only: the box keeps the pixel convention, the anchors decide which row pair it sits on
        floor_edge = max(box["bottom"] for box in found)
        rows = round((floor_edge-float(frame["bottom"]))/fitted_y)
        # the lowest visible block is the floor only when the stack actually rests on it: a falling piece can sit
        # anywhere, so an anchor that disagrees with the box's own floor by more than a cell is not trusted
        consistent = abs(floor_edge-float(frame["bottom"])) <= FLOOR_TOLERANCE*fitted_y
        rows = int(rows) if consistent else 0
        bottom = float(frame["bottom"])+rows*fitted_y
        report["y"] = {"kept": True, "pitch": round(fitted_y, 6), "origin": round(bottom, 6),
                       "anchors": len(top_edges), "floor_edge": round(float(floor_edge), 6),
                       "rows_corrected": int(rows), "floor_consistent": bool(consistent)}
        top = bottom-ROWS*fitted_y
        step_y = fitted_y
    else:
        top, step_y = float(frame["top"]), pitch_y
        report["y"] = {"kept": False, "reason": "too few vertical anchors", "anchors": len(top_edges)}

    return {"origin": [left, top+ROWS*step_y], "u": [fitted_x, 0.0], "v": [0.0, -step_y],
            "frame": {"left": left, "right": left+COLUMNS*fitted_x, "top": top, "bottom": top+ROWS*step_y},
            "fit": report, "source": "stack"}


def fmod(value, period):
    return float(value)-period*np.floor(float(value)/period)
