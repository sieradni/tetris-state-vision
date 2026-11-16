"""M51's board-box detector: one declared objective over label-free wall and grid candidates.

`docs/M51_DETECTOR_REPAIR.md` declares this instrument. M50's detector needed a full-height wall pair and failed on two
named captures; this module replaces it with a candidate search evaluated identically on every board:

- **walls**: the pair of columns and rows whose edge runs are longest, subject to the board's ten-by-twenty aspect,
  read off the inner edges of the wall clusters - M50's detector, kept as one generator among several;
- **grid**: the cell spacing and phase whose lines carry the most edge energy, then the longest run of lines at that
  spacing, which needs no full-height wall at all.

One objective picks between them: `wall support + grid alignment + (1 - cell uniformity) + stack weight * containment`,
each term computed from the raster alone. Nothing keys off which capture it is; no label, prevalence, scored cell or
fitted value enters. The target lattice is never read here - only by the scorer, afterwards.
"""
import numpy as np

COLUMNS, ROWS = 10, 20
EDGE = 12.0
WALL_SHARE = 0.5
CLUSTER = 6
MIN_WIDTH = 60
MAX_WIDTH = 400
ASPECT_SLACK = 0.06
SHARE_FLOOR = 0.25
SHARE_STEP = 0.05
CELL_MIN = 10.0
CELL_MAX = 22.0
CELL_STEP = 0.1
CELL_PHASE_STEP = 0.5
SATURATION = 60.0
STACK_WEIGHT = 0.5
BOARD_SPAN = 0.3
FLOOR_BAND = 0.25

CONSTANTS = {"edge": EDGE, "wall_share": WALL_SHARE, "cluster": CLUSTER, "min_width": MIN_WIDTH,
             "max_width": MAX_WIDTH, "aspect_slack": ASPECT_SLACK, "share_floor": SHARE_FLOOR,
             "share_step": SHARE_STEP, "cell_min": CELL_MIN, "cell_max": CELL_MAX, "cell_step": CELL_STEP,
             "cell_phase_step": CELL_PHASE_STEP,
             "saturation": SATURATION, "stack_weight": STACK_WEIGHT, "board_span": BOARD_SPAN, "floor_band": FLOOR_BAND}


def longest_runs(mask):
    """Longest run of True along each line of a boolean edge mask."""
    best = np.zeros(mask.shape[1], dtype=np.int64)
    for index in range(mask.shape[1]):
        line = mask[:, index].astype(np.int8)
        if not line.any():
            continue
        change = np.flatnonzero(np.diff(np.r_[np.int8(0), line, np.int8(0)]))
        if change.size >= 2:
            best[index] = int((change[1::2]-change[0::2]).max())
    return best


def axis_window(energy, spacing, count):
    """The window of `count` lines at `spacing` that best reads as a boarded cell run.

    Two label-free constraints follow from the recipe the campaign assumes (a boarded ten-by-twenty well drawn with
    cell lines): the window's two end lines are the board's border, so they carry at least BORDER_SHARE of the
    strongest line energy on that axis; and at the true pitch the cell interiors between the lines are flat, while a
    pitch a fraction of the cell size can put every line on a block edge. The declared score is the line sum over the
    summed energy at the cell midpoints, maximised subject to the border constraint, falling back to the unconstrained
    maximum when no window is bordered.
    """
    best = None
    for phase in np.arange(0.0, spacing, CELL_PHASE_STEP):
        positions = np.arange(phase, energy.size, spacing).round().astype(int)
        positions = positions[(positions >= 0) & (positions < energy.size)]
        if positions.size < count:
            continue
        values = energy[positions]
        totals = np.convolve(values, np.ones(count), mode="valid")
        start = int(np.argmax(totals))
        chosen = positions[start:start+count]
        middles = ((chosen[:-1]+chosen[1:])/2.0).astype(int)
        between = float(energy[middles].sum()) if middles.size else 0.0
        score = float(totals[start])/(1.0+between)
        if best is None or score > best[0]:
            best = (score, between, float(chosen[0]), float(chosen[-1]))
    return best


def grid_candidate(energy_column, energy_row):
    """The cell pitch, phase and window that carry the most bordered line contrast on both axes.

    The pitch is chosen with the same declared score, summed over the two axes, so a single square cell size serves
    the whole box and the candidate is one box rather than two independent ones.
    """
    best = None
    for spacing in np.arange(CELL_MIN, CELL_MAX+1e-9, CELL_STEP):
        columns = axis_window(energy_column, spacing, COLUMNS+1)
        rows = axis_window(energy_row, spacing, ROWS+1)
        if columns is None or rows is None:
            continue
        score = columns[0]+rows[0]
        if best is None or score > best[0]:
            best = (score, float(spacing), columns, rows)
    if best is None:
        return None
    _, spacing, columns, rows = best
    x0, x1 = columns[2], columns[3]
    y_top, y_bottom = rows[2], rows[3]
    return {"origin": [x0, y_bottom], "u": [(x1-x0)/COLUMNS, 0.0], "v": [0.0, -(y_bottom-y_top)/ROWS],
            "frame": {"left": x0, "right": x1, "top": y_top, "bottom": y_bottom, "spacing": spacing,
                      "column_lines": COLUMNS+1, "row_lines": ROWS+1},
            "source": "grid"}


def wall_candidate(column_runs, row_runs):
    """M50's detector: the tallest wall pair that brackets a ten-by-twenty board."""
    def candidates(runs):
        # a board border runs across most of the capture: an absolute span floor finds a faint border that a share of
        # the brightest line would miss, and the strength ranking below still prefers the strongest such rectangle
        span = np.flatnonzero(runs >= BOARD_SPAN*runs.size)
        if span.size >= 2:
            return span
        share = WALL_SHARE
        while share >= SHARE_FLOOR:
            found = np.flatnonzero(runs >= share*runs.max())
            if found.size >= 2:
                return found
            share -= SHARE_STEP
        raise ValueError("no wall pair on this axis")

    column_candidates = candidates(column_runs)
    row_candidates = candidates(row_runs)
    best = None
    for left in column_candidates:
        for right in column_candidates:
            box_width = int(right-left)
            if box_width < MIN_WIDTH or box_width > MAX_WIDTH:
                continue
            for top in row_candidates:
                target = top+2*box_width
                for bottom in row_candidates[np.abs(row_candidates-target) <= ASPECT_SLACK*box_width]:
                    key = (int(min(column_runs[left], column_runs[right],
                                   row_runs[top], row_runs[bottom])),
                           box_width*int(bottom-top), -abs(int(bottom-top)-2*box_width))
                    if best is None or key > best[0]:
                        best = (key, int(left), int(right), int(top), int(bottom), True)
    if best is None:
        for left in column_candidates:
            for right in column_candidates:
                box_width = int(right-left)
                if box_width < MIN_WIDTH or box_width > MAX_WIDTH:
                    continue
                for top in row_candidates:
                    for bottom in row_candidates:
                        if bottom <= top:
                            continue
                        key = (box_width*int(bottom-top),
                               int(min(column_runs[left], column_runs[right],
                                       row_runs[top], row_runs[bottom])))
                        if best is None or key > best[0]:
                            best = (key, int(left), int(right), int(top), int(bottom), False)
    if best is None:
        raise ValueError("no wall rectangle")
    _, left, right, top, bottom, strict = best

    def inner_edge(positions, anchor, direction):
        cluster = (positions[(positions >= anchor) & (positions <= anchor+CLUSTER)] if direction > 0
                   else positions[(positions >= anchor-CLUSTER) & (positions <= anchor)])
        return float(cluster.max()+1) if direction > 0 else float(cluster.min())

    x0 = inner_edge(column_candidates, left, +1)
    x1 = inner_edge(column_candidates, right, -1)
    y_top = inner_edge(row_candidates, top, +1)
    y_bottom = inner_edge(row_candidates, bottom, -1)
    return {"origin": [x0, y_bottom], "u": [(x1-x0)/COLUMNS, 0.0], "v": [0.0, -(y_bottom-y_top)/ROWS],
            "frame": {"left": x0, "right": x1, "top": y_top, "bottom": y_bottom,
                      "walls": [left, right, top, bottom], "strict_aspect": strict},
            "source": "walls"}


def score_box(pixels, grey, box):
    """The declared objective, every term from the raster only."""
    height, width = grey.shape
    x0, y_top = box["frame"]["left"], box["frame"]["top"]
    x1, y_bottom = box["frame"]["right"], box["frame"]["bottom"]
    cell = box["u"][0]
    edge_x = np.abs(np.diff(grey, axis=1)).sum(0)
    edge_y = np.abs(np.diff(grey, axis=0)).sum(1)
    column_runs = longest_runs(np.abs(np.diff(grey, axis=1)) > EDGE)
    row_runs = longest_runs(np.abs(np.diff(grey, axis=0)).T > EDGE)
    wall = min(int(column_runs[min(max(int(round(x0)), 0), column_runs.size-1)]),
               int(column_runs[min(max(int(round(x1)), 0), column_runs.size-1)]),
               int(row_runs[min(max(int(round(y_top)), 0), row_runs.size-1)]),
               int(row_runs[min(max(int(round(y_bottom)), 0), row_runs.size-1)]))
    wall = wall/max(1, int(max(column_runs.max(), row_runs.max())))
    columns = np.clip((x0+np.arange(1, COLUMNS)*cell).round().astype(int), 0, edge_x.size-1)
    rows = np.clip((y_top+np.arange(1, ROWS)*cell).round().astype(int), 0, edge_y.size-1)
    grid = float((edge_x[columns].mean()/max(float(edge_x.max()), 1e-9)
                  + edge_y[rows].mean()/max(float(edge_y.max()), 1e-9))/2.0)
    interior = []
    for row in range(ROWS):
        for column in range(COLUMNS):
            xa = int(round(x0+(column+.15)*cell))
            xb = int(round(x0+(column+.85)*cell))
            ya = int(round(y_top+(row+.15)*cell))
            yb = int(round(y_top+(row+.85)*cell))
            if xb <= xa or yb <= ya or xa < 0 or ya < 0 or xb > width or yb > height:
                continue
            interior.append(float(grey[ya:yb, xa:xb].std()))
    uniformity = float(np.mean(interior)) if interior else 1.0
    uniformity = min(1.0, uniformity/max(float(grey.std()), 1e-9))
    saturated = (pixels.max(2)-pixels.min(2)) > SATURATION
    total = int(saturated.sum())
    inside = saturated[max(0, int(round(y_top))):max(0, int(round(y_bottom))),
                       max(0, int(round(x0))):max(0, int(round(x1)))]
    stack = float(inside.sum())/max(total, 1)
    rows_with = np.flatnonzero(inside.any(axis=1)) if inside.size else np.array([], dtype=np.int64)
    floor = 1.0 if rows_with.size == 0 else float(inside.shape[0]-1-rows_with.max())/max(1, inside.shape[0])
    floor_support = 1.0-min(1.0, floor/FLOOR_BAND)
    area = max(0.0, (x1-x0))*max(0.0, (y_bottom-y_top))/float(width*height)
    score = wall+grid+(1.0-uniformity)+STACK_WEIGHT*stack
    return {"score": round(float(score), 6), "wall_support": round(float(wall), 6),
            "grid_alignment": round(float(grid), 6), "cell_uniformity": round(float(1.0-uniformity), 6),
            "stack_containment": round(float(stack), 6), "floor_support": round(float(floor_support), 6),
            "area_share": round(float(area), 6)}


def detect_box(pixels):
    """The board's own box, from the raster alone: the best-scoring ten-by-twenty candidate."""
    pixels = np.asarray(pixels, dtype=np.float64)
    height, width = pixels.shape[:2]
    grey = pixels.mean(2)
    edge_x = np.abs(np.diff(grey, axis=1)).sum(0)
    edge_y = np.abs(np.diff(grey, axis=0)).sum(1)
    column_runs = longest_runs(np.abs(np.diff(grey, axis=1)) > EDGE)
    row_runs = longest_runs(np.abs(np.diff(grey, axis=0)).T > EDGE)
    if not edge_x.any() or not edge_y.any() or column_runs.max() < 2*ROWS and row_runs.max() < 2*ROWS:
        raise ValueError("the capture carries no structure to register against")

    candidates, notes = [], {}
    try:
        candidates.append(wall_candidate(column_runs, row_runs))
        notes["walls"] = "candidate"
    except ValueError as error:
        notes["walls"] = str(error)
    grid = grid_candidate(edge_x, edge_y)
    if grid is not None:
        candidates.append(grid)
        notes["grid"] = "candidate"
        notes["spacing"] = round(grid["frame"]["spacing"], 3)
    else:
        notes["grid"] = "no periodic lines"
    if not candidates:
        raise ValueError("no ten-by-twenty candidate on this capture")

    scored = []
    for candidate in candidates:
        reading = score_box(pixels, grey, candidate)
        scored.append((reading["score"], candidate, reading))
    scored.sort(key=lambda entry: -entry[0])
    best = dict(scored[0][1])
    best["background"] = [0.0, 0.0, 0.0]
    best["interior"] = [best["frame"]["left"], best["frame"]["top"], best["frame"]["right"],
                        best["frame"]["bottom"]]
    best["raster"] = [width, height]
    best["objective"] = scored[0][2]
    best["candidates"] = [{"source": candidate["source"], "score": value, "support": support}
                          for value, candidate, support in scored]
    best["notes"] = notes
    if best["frame"]["right"]-best["frame"]["left"] < 3*COLUMNS:
        raise ValueError("the chosen board box is too small to hold the board")
    return best
