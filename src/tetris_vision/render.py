"""Deterministic layered synthetic scenes with separate latent and supported labels.

Scope: digital axis-aligned 10x20 boards. Visibility is deliberately conservative:
partial cells/icons are unknown. This is a data renderer, not image recognition.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

from PIL import Image, ImageDraw, ImageFilter

from .shapes import SHAPES
from .state import Board, Cell, FallingPiece, Icon, Lattice, Queue, VisionState, check

PALETTE = {"I": (38, 211, 224), "O": (242, 203, 56), "T": (176, 90, 230),
           "S": (89, 202, 92), "Z": (231, 77, 99), "J": (78, 111, 233),
           "L": (240, 151, 57), "G": (159, 169, 184)}


@dataclass(frozen=True)
class Pose:
    piece: str
    column: int
    row: int

    def __post_init__(self):
        check(self.piece in SHAPES, "invalid pose piece")
        check(type(self.column) is int and type(self.row) is int, "integer pose required")

    @property
    def cells(self):
        return tuple((c + self.column, r + self.row) for c, r in SHAPES[self.piece])


@dataclass(frozen=True)
class Scene:
    locked: tuple[tuple[str | None, ...], ...]
    active: Pose | None = None
    ghost: Pose | None = None
    queue: tuple[str, ...] = ("T", "T", "I", "L", "S")
    hold: str | None = "O"
    queue_present: bool = True
    hold_present: bool = True

    def __post_init__(self):
        check(len(self.locked) == 20 and all(len(row) == 10 for row in self.locked), "10x20 scene required")
        check(all(value is None or value in PALETTE for row in self.locked for value in row),
              "invalid locked semantic label")
        check(len(self.queue) <= 10 and all(piece in SHAPES for piece in self.queue), "scene queue capacity 10")
        check(self.hold is None or self.hold in SHAPES, "invalid hold")
        for pose in (self.active, self.ghost):
            if pose is not None:
                for c, r in pose.cells:
                    check(0 <= c < 10 and -4 <= r < 26, "pose exceeds supported spawn margin")
        # Reject physically conflicting locked/active cells; ghost remains an
        # independent visual layer and may coincide with a resting active piece.
        if self.active:
            check(all(not (0 <= r < 20 and self.locked[r][c] is not None)
                      for c, r in self.active.cells), "active intersects locked cells")


@dataclass(frozen=True)
class Appearance:
    seed: int = 0
    background: str = "solid"
    skin: str = "flat"
    grid: bool = True
    border: bool = True
    board_alpha: int = 235
    locked_alpha: int = 255
    active_alpha: int = 255
    ghost_alpha: int = 110
    palette_shift: int = 0
    monochrome: bool = False
    layer_cues: bool = True
    crop: tuple[int, int, int, int] | None = None
    output_size: tuple[int, int] | None = None
    blur_radius: float = 0

    def __post_init__(self):
        check(self.background in {"solid", "gradient", "blocks"}, "unknown background")
        check(self.skin in {"flat", "bevel", "outline"}, "unknown skin")
        check(all(type(value) is int and 0 <= value <= 255 for value in
                  (self.board_alpha, self.locked_alpha, self.active_alpha, self.ghost_alpha)), "invalid alpha")
        check(0 <= self.blur_radius <= 2, "blur radius outside initial pilot range")
        if self.crop:
            l, t, r, b = self.crop
            check(0 <= l < r <= 480 and 0 <= t < b <= 540, "crop outside source canvas")
        if self.output_size:
            check(len(self.output_size) == 2 and all(type(v) is int and 64 <= v <= 1536
                  for v in self.output_size), "output dimensions must be 64..1536")


@dataclass
class Rendered:
    image: Image.Image
    state: VisionState
    # Latent Scene is retained separately by dataset metadata; never merge into state.
    layer_masks: dict[str, Image.Image]


def sample_scene(seed: int, density: str = "dense") -> Scene:
    rng = random.Random(seed)
    check(density in {"empty", "sparse", "dense"}, "invalid density")
    rows = [[None] * 10 for _ in range(20)]
    count = {"empty": 0, "sparse": rng.randint(1, 4), "dense": rng.randint(40, 75)}[density]
    # Authored boards may float; do not impose divisibility or support rules.
    for index in rng.sample(range(90), count):
        rows[index // 10][index % 10] = rng.choice(tuple(PALETTE))
    return Scene(tuple(tuple(row) for row in rows), Pose("T", 3, 16), Pose("T", 3, 10))


def families(seed: int):
    """Counterfactual appearances of ONE latent scene; seed/group are shared."""
    return {
        "clean": Appearance(seed=seed),
        "no_grid": Appearance(seed=seed + 1, grid=False, border=False, skin="bevel", palette_shift=3),
        "transparent": Appearance(seed=seed + 2, background="blocks", board_alpha=0,
                                  locked_alpha=160, grid=False, skin="outline"),
        "monochrome": Appearance(seed=seed + 3, background="gradient", monochrome=True,
                                 skin="bevel", ghost_alpha=240, active_alpha=180),
        "cropped": Appearance(seed=seed + 4, crop=(90, 30, 385, 477), grid=False,
                              skin="outline", output_size=(354, 536)),
    }


def render(scene: Scene, style: Appearance = Appearance()) -> Rendered:
    rng = random.Random(style.seed)
    image = Image.new("RGBA", (480, 540), (24, 29, 42, 255))
    draw = ImageDraw.Draw(image)
    if style.background == "gradient":
        for y in range(540):
            draw.line((0, y, 479, y), fill=(22 + y // 9, 26 + y // 12, 48 + y // 7, 255))
    elif style.background == "blocks":
        for _ in range(180):
            x, y = rng.randrange(480), rng.randrange(540)
            draw.rectangle((x, y, x + rng.randrange(8, 60), y + rng.randrange(8, 60)),
                           fill=tuple(rng.randrange(30, 120) for _ in range(3)) + (255,))
    lattice = Lattice((64., 490.), (20., 0.), (0., -20.))
    masks = {name: Image.new("L", image.size, 0) for name in ("locked", "ghost", "active", "icons")}

    def composite(paint):
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        paint(ImageDraw.Draw(layer))
        image.alpha_composite(layer)

    def backdrop(d):
        d.rectangle((64, 90, 263, 489), fill=(9, 14, 23, style.board_alpha))
        if style.grid:
            for x in range(64, 265, 20):
                d.line((x, 90, x, 489), fill=(112, 129, 150, 95))
            for y in range(90, 491, 20):
                d.line((64, y, 263, y), fill=(112, 129, 150, 95))
        if style.border:
            d.rectangle((62, 88, 265, 491), outline=(165, 182, 207, 255), width=2)
    composite(backdrop)

    keys = tuple(PALETTE)

    def color(piece):
        return (182, 186, 196) if style.monochrome else PALETTE[keys[(keys.index(piece) + style.palette_shift) % len(keys)]]

    def mino(d, box, piece, alpha, mask, skin=None):
        rgb = color(piece)
        x0, y0, x1, y1 = box
        box = (x0 + 1, y0 + 1, x1 - 2, y1 - 2)
        actual_skin = skin or style.skin
        if actual_skin == "outline":
            d.rectangle(box, outline=rgb + (alpha,), width=2)
            ImageDraw.Draw(mask).rectangle(box, outline=alpha, width=2)
        else:
            d.rectangle(box, fill=rgb + (alpha,))
            ImageDraw.Draw(mask).rectangle(box, fill=alpha)
            if actual_skin == "bevel":
                light = tuple(min(255, v + 50) for v in rgb)
                d.line((box[0], box[3], box[0], box[1], box[2], box[1]), fill=light + (alpha,), width=2)

    def cell_box(c, r):
        return (64 + c * 20, 490 - (r + 1) * 20, 64 + (c + 1) * 20, 490 - r * 20)

    composite(lambda d: [mino(d, cell_box(c, r), piece, style.locked_alpha, masks["locked"])
                         for r, row in enumerate(scene.locked) for c, piece in enumerate(row) if piece])
    # Identifiable pilot families use distinct flat/bevel/outline conventions.
    # Matching appearances are also supported, but MUST receive ambiguous labels.
    other_skins = [skin for skin in ("flat", "bevel", "outline") if skin != style.skin]
    active_skin, ghost_skin = other_skins
    for name, pose, alpha in (("ghost", scene.ghost, style.ghost_alpha),
                               ("active", scene.active, style.active_alpha)):
        if pose:
            composite(lambda d, p=pose, a=alpha, n=name: [
                mino(d, cell_box(c, r), p.piece, a, masks[n],
                     (ghost_skin if n == "ghost" else active_skin) if style.layer_cues else style.skin)
                for c, r in p.cells])

    hold_box = (8, 80, 55, 145)
    queue_box = (308, 74, 397, 114 + max(1, len(scene.queue)) * 34)
    slot_boxes = [(320, 100 + i * 34, 368, 124 + i * 34) for i in range(len(scene.queue))]

    def icons(d):
        def icon(piece, x, y):
            for c, r in SHAPES[piece]:
                mino(d, (x + c * 10, y + (1-r) * 10, x + (c+1) * 10, y + (2-r) * 10),
                     piece, 255, masks["icons"], "flat")
        if scene.hold_present:
            d.rectangle(hold_box, fill=(19, 25, 38, 240), outline=(128, 143, 168, 255))
            d.text((12, 85), "H", fill=(215, 224, 240, 255))
            if scene.hold:
                icon(scene.hold, 12, 110)
        if scene.queue_present:
            d.rectangle(queue_box, fill=(19, 25, 38, 240), outline=(128, 143, 168, 255))
            d.text((320, 80), "NEXT", fill=(215, 224, 240, 255))
            for i, piece in enumerate(scene.queue):
                icon(piece, 320, 102 + i * 34)
    composite(icons)

    crop = style.crop or (0, 0, 480, 540)
    l, t, right, bottom = crop
    output_size = style.output_size or (right-l, bottom-t)
    sx, sy = output_size[0] / (right-l), output_size[1] / (bottom-t)
    lattice = lattice.transformed(sx, sy, -l*sx, -t*sy)

    def full(box):
        return l <= box[0] and t <= box[1] and box[2] <= right and box[3] <= bottom

    # A type is not identifiable from monochrome locked fragments. Occupancy is
    # independent. Extremely weak alpha/scale/blur are reviewed, not latent truth.
    strong = min(20*sx, 20*sy) >= 10 and style.blur_radius <= .5
    layer_visible = {"locked": style.locked_alpha >= 128 and strong,
                     "active": style.active_alpha >= 128 and strong,
                     "ghost": style.ghost_alpha >= 64 and strong}
    def layer_state(pose, name):
        if name == "active" and not style.layer_cues and (pose is not None or
                any(piece for row in scene.locked for piece in row)):
            return FallingPiece("unknown")
        if name == "ghost" and pose is not None and not style.layer_cues:
            return FallingPiece("unknown")
        if pose is None:
            # A crop could hide an entire active/ghost piece.
            return FallingPiece("absent" if full((64, 0, 264, 490)) else "unknown")
        if not all(full(cell_box(c, r)) for c, r in pose.cells):
            return FallingPiece("cropped")
        if not layer_visible[name]:
            return FallingPiece("unknown")
        if name == "ghost" and scene.active and set(pose.cells) & set(scene.active.cells):
            return FallingPiece("unknown")
        return FallingPiece("observed", pose.piece, pose.cells)

    visible_active = layer_state(scene.active, "active")
    visible_ghost = layer_state(scene.ghost, "ghost")
    # If visual layers cannot be disentangled, abstain on corresponding locked
    # cells rather than supervising an invisible underlying empty/occupied cell.
    uncertain_cells = set()
    if not style.layer_cues:
        uncertain_cells.update((c, r) for r, row in enumerate(scene.locked)
                               for c, piece in enumerate(row) if piece)
    for pose, label in ((scene.active, visible_active), (scene.ghost, visible_ghost)):
        if pose and label.status == "unknown":
            uncertain_cells.update(pose.cells)
    if scene.ghost:
        uncertain_cells.update((c, r) for c, r in scene.ghost.cells
                               if 0 <= r < 20 and scene.locked[r][c] is not None)
    rows = []
    for r, row in enumerate(scene.locked):
        labels = []
        for c, piece in enumerate(row):
            if not full(cell_box(c, r)):
                labels.append(Cell(None, visibility="cropped"))
            elif not layer_visible["locked"] or (c, r) in uncertain_cells:
                labels.append(Cell(None, visibility="hidden"))
            else:
                support = {item for item in scene.queue} if scene.queue_present and full(queue_box) else set()
                if scene.hold_present and full(hold_box) and scene.hold:
                    support.add(scene.hold)
                if visible_active.status == "observed":
                    support.add(visible_active.piece)
                typed = (not style.monochrome and
                         (style.palette_shift % len(PALETTE) == 0 or piece in support))
                labels.append(Cell(piece is not None, piece if typed else None))
        rows.append(tuple(labels))
    queue_visibility = "absent" if not scene.queue_present else "visible" if full(queue_box) else "cropped"
    queue_items = () if not scene.queue_present else tuple(
        Icon("observed", piece) if full(box) and strong else Icon("cropped" if not full(box) else "unknown")
        for piece, box in zip(scene.queue, slot_boxes))
    queue = Queue(queue_items, queue_visibility, queue_visibility == "visible")
    hold = Icon("absent") if not scene.hold_present else Icon("cropped") if not full(hold_box) else (
        Icon("unknown") if not strong else Icon("observed", scene.hold) if scene.hold else Icon("empty"))

    has_anchor = style.grid or style.border or style.board_alpha >= 128 or scene.active is not None or scene.ghost is not None
    # Unanchored transparent scenes do not justify known empty cells or geometry.
    if not has_anchor or not full((64, 90, 264, 490)):
        lattice = None
    if not has_anchor:
        rows = [tuple(Cell(None, visibility="hidden") for _ in row) for row in rows]
    review = (lattice is None or any(cell.occupied is None for row in rows for cell in row)
              or visible_active.status not in {"observed", "absent"}
              or visible_ghost.status not in {"observed", "absent"}
              or queue_visibility == "cropped" or any(item.status != "observed" for item in queue_items)
              or hold.status in {"cropped", "unknown"})
    state = VisionState(Board(tuple(rows), lattice), visible_active, visible_ghost, queue, hold,
                        "needs_review" if review else "accepted", producer="synthetic-v1")
    image = image.crop(crop).resize(output_size, Image.Resampling.LANCZOS).convert("RGB")
    if style.blur_radius:
        image = image.filter(ImageFilter.GaussianBlur(style.blur_radius))
    masks = {name: mask.crop(crop).resize(output_size, Image.Resampling.NEAREST) for name, mask in masks.items()}
    return Rendered(image, state, masks)
