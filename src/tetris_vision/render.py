"""Deterministic layered synthetic scenes with separate latent and supported labels.

Scope: digital axis-aligned 10x20 boards. Visibility is deliberately conservative:
partial cells/icons are unknown. This is a data renderer, not image recognition.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

from PIL import Image, ImageDraw, ImageFilter

from .layout import Layout
from .observe import observe
from .shapes import SHAPES, rotated_shape
from .state import VisionState, check

PALETTE = {"I": (38, 211, 224), "O": (242, 203, 56), "T": (176, 90, 230),
           "S": (89, 202, 92), "Z": (231, 77, 99), "J": (78, 111, 233),
           "L": (240, 151, 57), "G": (159, 169, 184)}


@dataclass(frozen=True)
class Pose:
    piece: str
    column: int
    row: int
    rotation: int = 0

    def __post_init__(self):
        check(self.piece in SHAPES, "invalid pose piece")
        check(type(self.column) is int and type(self.row) is int, "integer pose required")
        check(type(self.rotation) is int and 0 <= self.rotation < 4, "rotation must be 0..3")

    @property
    def cells(self):
        return tuple((c + self.column, r + self.row) for c, r in rotated_shape(self.piece, self.rotation))


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
    preview_palette_shift: int | None = None
    preview_monochrome: bool | None = None
    preview_skin: str = "flat"
    occluded_queue: tuple[int, ...] = ()
    hold_occluded: bool = False
    occluded_cells: tuple[tuple[int, int], ...] = ()
    crop: tuple[int, int, int, int] | None = None
    output_size: tuple[int, int] | None = None
    blur_radius: float = 0

    def __post_init__(self):
        check(self.background in {"solid", "gradient", "blocks"}, "unknown background")
        check(self.skin in {"flat", "bevel", "outline"}, "unknown skin")
        check(self.preview_skin in {"flat", "bevel", "outline"}, "unknown preview skin")
        check(type(self.palette_shift) is int and (self.preview_palette_shift is None or
              type(self.preview_palette_shift) is int), "integer palette shift required")
        check(self.preview_monochrome is None or type(self.preview_monochrome) is bool, "boolean preview monochrome")
        check(all(type(index) is int and 0 <= index < 10 for index in self.occluded_queue), "invalid queue occlusion")
        check(all(len(cell) == 2 and all(type(v) is int for v in cell) and
                  0 <= cell[0] < 10 and 0 <= cell[1] < 20 for cell in self.occluded_cells), "invalid cell occlusion")
        check(all(type(value) is int and 0 <= value <= 255 for value in
                  (self.board_alpha, self.locked_alpha, self.active_alpha, self.ghost_alpha)), "invalid alpha")
        check(0 <= self.blur_radius <= 2, "blur radius outside initial pilot range")
        if self.crop:
            l, t, r, b = self.crop
            check(all(type(value) is int for value in self.crop) and
                  0 <= l < r <= 1536 and 0 <= t < b <= 1536, "invalid bounded crop")
        if self.output_size:
            check(len(self.output_size) == 2 and all(type(v) is int and 64 <= v <= 1536
                  for v in self.output_size), "output dimensions must be 64..1536")


@dataclass
class Rendered:
    image: Image.Image
    state: VisionState
    # Latent Scene is retained separately by dataset metadata; never merge into state.
    layer_masks: dict[str, Image.Image]


def sample_scene(seed: int, density: str = "dense", *, varied=False) -> Scene:
    rng = random.Random(seed)
    check(density in {"empty", "sparse", "dense"}, "invalid density")
    rows = [[None] * 10 for _ in range(20)]
    count = {"empty": 0, "sparse": rng.randint(1, 4), "dense": rng.randint(40, 75)}[density]
    # Authored boards may float; do not impose divisibility or support rules.
    for index in rng.sample(range(90), count):
        rows[index // 10][index % 10] = rng.choice(tuple(PALETTE))
    if not varied:
        return Scene(tuple(tuple(row) for row in rows), Pose("T", 3, 16), Pose("T", 3, 10))
    piece, rotation = rng.choice(tuple(SHAPES)), rng.randrange(4)
    shape = rotated_shape(piece, rotation)
    column = rng.randrange(10-max(c for c, _ in shape))
    active = Pose(piece, column, rng.randint(14, 16), rotation) if rng.random() > .15 else None
    ghost = Pose(piece, column, 10, rotation) if active and rng.random() > .2 else None
    length = rng.randint(0, 10)
    queue = tuple(rng.choice(tuple(SHAPES)) for _ in range(length))
    return Scene(tuple(tuple(row) for row in rows), active, ghost, queue,
                 rng.choice((None, *SHAPES)), rng.random() > .15, rng.random() > .15)


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


def diverse_families(seed, scene, layout):
    """Controlled appearances crossed with a separately sampled source layout."""
    width, height = layout.canvas
    if seed % 2:
        crop = (layout.board_left+layout.cell_size//2, 0, width, height)
    else:
        crop = (0, 0, layout.board_box[2]+5, height)
    output = (max(64, round((crop[2]-crop[0])*.9)), max(64, round(height*.9)))
    return {
        "clean": Appearance(seed=seed),
        "no_grid": Appearance(seed=seed+1, grid=False, border=False, skin="bevel", background="gradient"),
        "independent_previews": Appearance(seed=seed+2, palette_shift=2,
                                           preview_palette_shift=5, preview_skin="outline",
                                           ghost_alpha=240, active_alpha=180),
        "occluded": Appearance(seed=seed+3, background="blocks", board_alpha=0,
                               locked_alpha=200, grid=False, skin="outline", hold_occluded=True,
                               occluded_queue=(len(scene.queue)//2,) if scene.queue else (),
                               occluded_cells=((seed % 10, seed // 10 % 20),)),
        "cropped": Appearance(seed=seed+4, crop=crop, output_size=output),
    }


def render(scene: Scene, style: Appearance = Appearance(), layout: Layout = Layout()) -> Rendered:
    layout.validate_panels(scene)
    if style.crop:
        check(style.crop[2] <= layout.canvas[0] and style.crop[3] <= layout.canvas[1], "crop outside source canvas")
    check(all(index < len(scene.queue) for index in style.occluded_queue), "occluded queue index out of range")
    rng = random.Random(style.seed)
    width, height = layout.canvas
    image = Image.new("RGBA", layout.canvas, (24, 29, 42, 255))
    draw = ImageDraw.Draw(image)
    if style.background == "gradient":
        for y in range(height):
            draw.line((0, y, width-1, y), fill=(min(255, 22+y//9), min(255, 26+y//12), min(255, 48+y//7), 255))
    elif style.background == "blocks":
        for _ in range(180):
            x, y = rng.randrange(width), rng.randrange(height)
            draw.rectangle((x, y, x + rng.randrange(8, 60), y + rng.randrange(8, 60)),
                           fill=tuple(rng.randrange(30, 120) for _ in range(3)) + (255,))
    masks = {name: Image.new("L", image.size, 0) for name in ("locked", "ghost", "active", "icons")}

    def composite(paint):
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        paint(ImageDraw.Draw(layer))
        image.alpha_composite(layer)

    def backdrop(d):
        x0, y0, x1, y1 = layout.board_box
        d.rectangle((x0, y0, x1-1, y1-1), fill=(9, 14, 23, style.board_alpha))
        if style.grid:
            for x in range(x0, x1+1, layout.cell_size):
                d.line((x, y0, x, y1-1), fill=(112, 129, 150, 95))
            for y in range(y0, y1+1, layout.cell_size):
                d.line((x0, y, x1-1, y), fill=(112, 129, 150, 95))
        if style.border:
            d.rectangle((x0-2, y0-2, x1+1, y1+1), outline=(165, 182, 207, 255), width=2)
    composite(backdrop)

    keys = tuple(PALETTE)

    def color(piece, preview=False):
        mono = style.preview_monochrome if preview and style.preview_monochrome is not None else style.monochrome
        shift = style.preview_palette_shift if preview and style.preview_palette_shift is not None else style.palette_shift
        return (182, 186, 196) if mono else PALETTE[keys[(keys.index(piece)+shift) % len(keys)]]

    def mino(d, box, piece, alpha, mask, skin=None, preview=False):
        rgb = color(piece, preview)
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

    cell_box = layout.cell_box

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

    hold_box = layout.hold_box
    queue_box = layout.queue_box(len(scene.queue))

    def icons(d):
        def icon(piece, x, y):
            k = layout.icon_size
            for c, r in SHAPES[piece]:
                mino(d, (x+c*k, y+(1-r)*k, x+(c+1)*k, y+(2-r)*k),
                     piece, 255, masks["icons"], style.preview_skin, True)
        if scene.hold_present:
            d.rectangle(hold_box, fill=(19, 25, 38, 240), outline=(128, 143, 168, 255))
            hx, hy = layout.hold_origin
            d.text((hx+4, hy+5), "H", fill=(215, 224, 240, 255))
            if scene.hold:
                icon(scene.hold, hx+4, hy+30)
        if scene.queue_present:
            d.rectangle(queue_box, fill=(19, 25, 38, 240), outline=(128, 143, 168, 255))
            qx, qy = layout.queue_origin
            d.text((qx+12, qy+6), "NEXT", fill=(215, 224, 240, 255))
            for i, piece in enumerate(scene.queue):
                icon(piece, *layout.slot_origin(i))
    composite(icons)

    def occlusion(d):
        for index in style.occluded_queue:
            x0, y0, x1, y1 = layout.slot_box(index)
            d.rectangle((x0, y0, x1-1, y1-1), fill=(14, 18, 26, 255))
        if style.hold_occluded and scene.hold_present:
            x0, y0, x1, y1 = hold_box
            d.rectangle((x0+2, y0+25, x1-2, y1-2), fill=(14, 18, 26, 255))
        for c, r in style.occluded_cells:
            x0, y0, x1, y1 = cell_box(c, r)
            d.rectangle((x0, y0, x1-1, y1-1), fill=(14, 18, 26, 255))
    composite(occlusion)
    crop = style.crop or (0, 0, *layout.canvas)
    l, t, right, bottom = crop
    output_size = style.output_size or (right-l, bottom-t)
    state = observe(scene, style, layout)
    image = image.crop(crop).resize(output_size, Image.Resampling.LANCZOS).convert("RGB")
    if style.blur_radius:
        image = image.filter(ImageFilter.GaussianBlur(style.blur_radius))
    masks = {name: mask.crop(crop).resize(output_size, Image.Resampling.NEAREST) for name, mask in masks.items()}
    return Rendered(image, state, masks)
