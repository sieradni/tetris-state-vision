"""Conservative image-supported targets, separate from latent scene state.

New hidden-component policies must have paired identical-raster equivalence
tests. See V-001, V-002 and V-005: knowing renderer memory is not observability.
"""
from .layout import intersects
from .state import Board, Cell, FallingPiece, Icon, Queue, VisionState


def observe(scene, style, layout):
    crop = style.crop or (0, 0, *layout.canvas)
    left, top, right, bottom = crop
    output = style.output_size or (right-left, bottom-top)
    sx, sy = output[0]/(right-left), output[1]/(bottom-top)
    full = lambda box: left <= box[0] and top <= box[1] and box[2] <= right and box[3] <= bottom
    seen = lambda box: intersects(box, crop)
    full_frame = crop == (0, 0, *layout.canvas)
    strong = min(layout.cell_size*sx, layout.cell_size*sy) >= 10 and style.blur_radius <= .5
    icons_strong = min(layout.icon_size*sx, layout.icon_size*sy) >= 5 and style.blur_radius <= .5
    layer_visible = {"locked": style.locked_alpha >= 128 and strong,
                     "active": style.active_alpha >= 128 and strong,
                     "ghost": style.ghost_alpha >= 64 and strong}

    def layer_state(pose, name):
        if not layer_visible[name]:
            return FallingPiece("unknown")
        if name == "active" and not style.layer_cues and (pose is not None or
                any(piece for row in scene.locked for piece in row)):
            return FallingPiece("unknown")
        if name == "ghost" and pose is not None and not style.layer_cues:
            return FallingPiece("unknown")
        if pose is None:
            bounds = layout.spawn_box if name == "active" else layout.board_box
            return FallingPiece("absent" if full(bounds) else "unknown")
        boxes = [layout.cell_box(c, r) for c, r in pose.cells]
        if not any(seen(box) for box in boxes):
            return FallingPiece("unknown")
        if not all(full(box) for box in boxes):
            return FallingPiece("cropped")
        if any(cell in style.occluded_cells for cell in pose.cells):
            return FallingPiece("unknown")
        if name == "ghost" and scene.active and set(pose.cells) & set(scene.active.cells):
            return FallingPiece("unknown")
        return FallingPiece("observed", pose.piece, pose.cells)

    active, ghost = layer_state(scene.active, "active"), layer_state(scene.ghost, "ghost")
    queue_box = layout.queue_box(len(scene.queue))
    header = (*layout.queue_origin, layout.queue_origin[0]+41, layout.queue_origin[1]+26)
    if not scene.queue_present:
        known_absent = full_frame or full(layout.queue_box(10))
        queue = Queue((), "absent" if known_absent else "unknown", False, False)
    elif not seen(queue_box):
        queue = Queue((), "unknown", False, False)
    else:
        # Keep only a visible span, including unknown interior slots. Never emit
        # the count of entirely unseen leading/trailing latent previews.
        indices = [i for i in range(len(scene.queue)) if seen(layout.slot_box(i))]
        span = range(min(indices), max(indices)+1) if indices else ()
        items = tuple(Icon("observed", scene.queue[i]) if full(layout.slot_box(i)) and icons_strong
                      and i not in style.occluded_queue else
                      Icon("unknown" if full(layout.slot_box(i)) else "cropped") for i in span)
        visible = full(queue_box)
        start = full(header) and ((not scene.queue and visible) or
                                 bool(indices) and min(indices) == 0 and full(layout.slot_box(0)))
        queue = Queue(items, "visible" if visible else "cropped", visible, start)
    if not seen(layout.hold_box):
        hold = Icon("unknown")
    elif not scene.hold_present:
        hold = Icon("absent" if full_frame or full(layout.hold_box) else "unknown")
    elif not full(layout.hold_box):
        hold = Icon("cropped")
    elif style.hold_occluded or not icons_strong:
        hold = Icon("unknown")
    else:
        hold = Icon("observed", scene.hold) if scene.hold else Icon("empty")

    same_palette = (style.preview_palette_shift is None or
                    (style.preview_palette_shift-style.palette_shift) % 8 == 0)
    same_mono = style.preview_monochrome is None or style.preview_monochrome == style.monochrome
    support = {item.piece for item in queue.items if item.status == "observed"} if same_palette and same_mono else set()
    if same_palette and same_mono and hold.status == "observed":
        support.add(hold.piece)
    if active.status == "observed":
        support.add(active.piece)
    uncertain = set(style.occluded_cells)
    if not style.layer_cues:
        uncertain.update((c, r) for r, row in enumerate(scene.locked) for c, piece in enumerate(row) if piece)
    for pose, label, alpha in ((scene.active, active, style.active_alpha),
                               (scene.ghost, ghost, style.ghost_alpha)):
        if pose and label.status == "unknown" and alpha > 0 and any(seen(layout.cell_box(c, r)) for c, r in pose.cells):
            uncertain.update(pose.cells)
    if scene.ghost and style.ghost_alpha > 0:
        uncertain.update((c, r) for c, r in scene.ghost.cells if 0 <= r < 20 and scene.locked[r][c] is not None)
    has_anchor = style.grid or style.border or style.board_alpha >= 128
    registered = has_anchor and full(layout.board_box)
    if not registered:
        # Piece type may still be visually suggested, but this schema's observed
        # layer promises a canonical footprint. Never expose latent indexing.
        active, ghost = FallingPiece("unknown"), FallingPiece("unknown")
    rows = []
    for r, row in enumerate(scene.locked):
        labels = []
        for c, piece in enumerate(row):
            if not registered:
                labels.append(Cell(None, visibility="cropped" if has_anchor else "hidden"))
            elif not full(layout.cell_box(c, r)):
                labels.append(Cell(None, visibility="cropped"))
            elif not layer_visible["locked"] or (c, r) in uncertain:
                labels.append(Cell(None, visibility="hidden"))
            else:
                typed = not style.monochrome and (style.palette_shift % 8 == 0 or piece in support)
                labels.append(Cell(piece is not None, piece if typed else None))
        rows.append(tuple(labels))
    lattice = layout.lattice.transformed(sx, sy, -left*sx, -top*sy) if registered else None
    review = (lattice is None or any(cell.occupied is None for row in rows for cell in row)
              or active.status not in {"observed", "absent"} or ghost.status not in {"observed", "absent"}
              or queue.visibility not in {"visible", "absent"} or not queue.start_observed and queue.visibility == "visible"
              or any(item.status != "observed" for item in queue.items) or hold.status in {"cropped", "unknown"})
    return VisionState(Board(tuple(rows), lattice), active, ghost, queue, hold,
                       "needs_review" if review else "accepted", producer="synthetic-v2")
