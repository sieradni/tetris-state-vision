"""Bounded source geometry. Recognition never receives this latent layout."""
from dataclasses import dataclass
import random

from .state import Lattice, check


def intersects(left, right):
    return max(left[0], right[0]) < min(left[2], right[2]) and max(left[1], right[1]) < min(left[3], right[3])


@dataclass(frozen=True)
class Layout:
    canvas: tuple[int, int] = (480, 540)
    board_left: int = 64
    board_bottom: int = 490
    cell_size: int = 20
    queue_origin: tuple[int, int] = (308, 74)
    hold_origin: tuple[int, int] = (8, 80)
    icon_size: int = 10
    queue_flow: str = "vertical"
    wrap_columns: int = 3

    def __post_init__(self):
        check(len(self.canvas) == 2 and all(type(v) is int and 128 <= v <= 1536 for v in self.canvas), "canvas bounds 128..1536")
        check(type(self.cell_size) is int and 8 <= self.cell_size <= 40, "cell size bounds 8..40")
        check(type(self.icon_size) is int and 6 <= self.icon_size <= 16, "icon size bounds 6..16")
        check(self.queue_flow in {"vertical", "horizontal", "wrapped"}, "invalid queue flow")
        check(type(self.wrap_columns) is int and 1 <= self.wrap_columns <= 5, "invalid wrap width")
        check(type(self.board_left) is int and type(self.board_bottom) is int, "integer board origin required")
        check(all(len(pair) == 2 and all(type(v) is int for v in pair) for pair in (self.queue_origin, self.hold_origin)),
              "integer panel origins required")
        check(self.contains(self.board_box), "source board outside canvas")

    def contains(self, box):
        return 0 <= box[0] < box[2] <= self.canvas[0] and 0 <= box[1] < box[3] <= self.canvas[1]

    @property
    def board_box(self):
        x, bottom, k = self.board_left, self.board_bottom, self.cell_size
        return (x, bottom - 20*k, x + 10*k, bottom)

    @property
    def spawn_box(self):
        x, top, right, bottom = self.board_box
        return (x, top - 6*self.cell_size, right, bottom)

    @property
    def lattice(self):
        return Lattice((float(self.board_left), float(self.board_bottom)),
                       (float(self.cell_size), 0.), (0., -float(self.cell_size)))

    def cell_box(self, column, row):
        x, bottom, k = self.board_left, self.board_bottom, self.cell_size
        return (x+column*k, bottom-(row+1)*k, x+(column+1)*k, bottom-row*k)

    @property
    def hold_box(self):
        x, y = self.hold_origin
        return (x, y, x+4*self.icon_size+7, y+2*self.icon_size+45)

    def slot_origin(self, index):
        x, y = self.queue_origin
        column, row = ((0, index) if self.queue_flow == "vertical" else
                       (index, 0) if self.queue_flow == "horizontal" else
                       (index % self.wrap_columns, index // self.wrap_columns))
        return (x+12+column*(5*self.icon_size+8), y+28+row*(3*self.icon_size+4))

    def slot_box(self, index):
        x, y = self.slot_origin(index)
        return (x, y-2, x+4*self.icon_size+8, y+2*self.icon_size+2)

    def queue_box(self, length):
        length = max(1, length)
        columns, rows = ((1, length) if self.queue_flow == "vertical" else
                         (length, 1) if self.queue_flow == "horizontal" else
                         (min(length, self.wrap_columns), (length+self.wrap_columns-1)//self.wrap_columns))
        x, y = self.queue_origin
        return (x, y, x+41+columns*(4*self.icon_size+8)+(columns-1)*(self.icon_size+2),
                y+40+rows*(3*self.icon_size+4))

    def validate_panels(self, scene):
        boxes = [self.board_box]
        if scene.queue_present:
            boxes.append(self.queue_box(len(scene.queue)))
        if scene.hold_present:
            boxes.append(self.hold_box)
        check(all(self.contains(box) for box in boxes), "source panel outside canvas")
        check(not any(intersects(a, b) for i, a in enumerate(boxes) for b in boxes[i+1:]),
              "panels overlap board/another panel; use explicit occlusion instead")


def sample_layout(seed, queue_length, *, flow=None):
    rng = random.Random(seed)
    flow = flow or rng.choice(("vertical", "horizontal", "wrapped"))
    k, icon = rng.randint(12, 24), rng.randint(6, 12)
    left, top = rng.randint(75, 145), rng.randint(6*k+12, 6*k+60)
    bottom = top+20*k
    columns = max(1, queue_length) if flow == "horizontal" else 3 if flow == "wrapped" else 1
    queue_width = 41+columns*(4*icon+8)+(columns-1)*(icon+2)
    width = max(480, left+10*k+queue_width+100)
    height = max(480, bottom+rng.randint(24, 70), 70+40+max(1, queue_length)*(3*icon+4))
    return Layout((width, height), left, bottom, k,
                  (left+10*k+rng.randint(24, 50), rng.randint(20, 70)),
                  (rng.randint(5, 14), rng.randint(30, 100)), icon, flow)
