"""Versioned, image-supported state. Unknown occupancy is never coerced to empty."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math

PIECES = frozenset("IOTSZJL")
CELL_TYPES = PIECES | {"G"}
SCHEMA_VERSION = "1.1"


def check(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Cell:
    occupied: bool | None
    semantic: str | None = None
    visibility: str = "visible"

    def __post_init__(self):
        check(self.occupied is None or type(self.occupied) is bool, "occupancy must be bool/null")
        check(self.semantic is None or self.semantic in CELL_TYPES, "invalid semantic cell type")
        check(self.semantic is None or self.occupied is True, "type requires known occupancy")
        check(self.visibility in {"visible", "cropped", "hidden"}, "invalid cell visibility")
        check(self.visibility == "visible" or self.occupied is None, "unseen cell must be unknown")


@dataclass(frozen=True)
class Lattice:
    """Image pixels: origin at bottom-left boundary; v points toward upper rows."""
    origin: tuple[float, float]
    u: tuple[float, float]
    v: tuple[float, float]

    def __post_init__(self):
        check(all(len(pair) == 2 for pair in (self.origin, self.u, self.v)), "2-D geometry required")
        check(all(type(x) in (int, float) and math.isfinite(x)
                  for pair in (self.origin, self.u, self.v) for x in pair), "finite geometry required")
        check(self.u[0] * self.v[1] - self.u[1] * self.v[0] < -1e-8,
              "nonfolding bottom-first lattice required")

    def center(self, column: int, row: int):
        return tuple(self.origin[i] + (column + .5) * self.u[i] + (row + .5) * self.v[i]
                     for i in range(2))

    def transformed(self, scale_x: float, scale_y: float, offset_x=0., offset_y=0.):
        check(scale_x > 0 and scale_y > 0, "positive resize scales required")
        return Lattice((self.origin[0] * scale_x + offset_x, self.origin[1] * scale_y + offset_y),
                       (self.u[0] * scale_x, self.u[1] * scale_y),
                       (self.v[0] * scale_x, self.v[1] * scale_y))


@dataclass(frozen=True)
class Board:
    rows: tuple[tuple[Cell, ...], ...]
    lattice: Lattice | None

    def __post_init__(self):
        check(1 <= len(self.rows) <= 40, "board supports 1..40 rows")
        check(1 <= len(self.rows[0]) <= 16, "board supports 1..16 columns")
        check(all(len(row) == len(self.rows[0]) for row in self.rows), "ragged board")
        check(all(isinstance(cell, Cell) for row in self.rows for cell in row), "Cell objects required")

    @property
    def width(self):
        return len(self.rows[0])

    @property
    def height(self):
        return len(self.rows)


@dataclass(frozen=True)
class FallingPiece:
    status: str
    piece: str | None = None
    cells: tuple[tuple[int, int], ...] = ()

    def __post_init__(self):
        check(self.status in {"observed", "absent", "unknown", "cropped"}, "invalid piece status")
        if self.status == "observed":
            check(self.piece in PIECES, "observed piece needs identity")
            check(len(self.cells) == 4 and len(set(self.cells)) == 4, "piece needs four distinct cells")
            check(all(len(cell) == 2 and all(type(x) is int for x in cell) for cell in self.cells),
                  "canonical integer cells required")
            # Rotation-independent tetromino shape, without trusting caller rotation labels.
            from .shapes import is_piece_shape
            check(is_piece_shape(self.piece, self.cells), "footprint disagrees with piece type")
        else:
            check(self.piece is None and not self.cells, "unobserved piece cannot leak latent state")


@dataclass(frozen=True)
class Icon:
    status: str
    piece: str | None = None

    def __post_init__(self):
        check(self.status in {"observed", "empty", "absent", "unknown", "cropped"}, "invalid icon status")
        check((self.status == "observed" and self.piece in PIECES) or
              (self.status != "observed" and self.piece is None), "status/identity mismatch")


@dataclass(frozen=True)
class Queue:
    items: tuple[Icon, ...]
    visibility: str = "visible"
    end_observed: bool = True
    start_observed: bool | None = None

    def __post_init__(self):
        if self.start_observed is None:
            object.__setattr__(self, "start_observed", self.visibility == "visible")
        check(self.visibility in {"visible", "absent", "cropped", "unknown"}, "invalid queue visibility")
        check(type(self.end_observed) is bool, "boolean end flag required")
        check(type(self.start_observed) is bool, "boolean start flag required")
        check(len(self.items) <= 64, "queue capacity exceeded (64)")
        check(all(item.status in {"observed", "unknown", "cropped"} for item in self.items),
              "queue slots cannot be empty/absent; retain an unknown slot")
        check(self.visibility != "absent" or (not self.items and not self.end_observed),
              "absent queue is not known empty")
        check(not self.end_observed or self.visibility == "visible", "unseen endpoint cannot be known")
        check(self.visibility != "absent" or not self.start_observed, "absent queue has no known start")


@dataclass(frozen=True)
class VisionState:
    board: Board | None
    active: FallingPiece
    ghost: FallingPiece
    queue: Queue
    hold: Icon
    decision: str
    schema_version: str = SCHEMA_VERSION
    producer: str = "annotation"
    model_id: str | None = None
    player: int = 0

    def __post_init__(self):
        check(self.schema_version == SCHEMA_VERSION, "unsupported state schema")
        check(self.decision in {"accepted", "needs_review", "no_board", "unsupported"}, "invalid decision")
        check(type(self.player) is int and self.player >= 0, "invalid player index")
        check(self.decision != "accepted" or self.board is not None, "accepted state needs board")
        check(self.decision != "no_board" or self.board is None, "no_board contradicts board")
        check(not any(layer.status == "observed" for layer in (self.active, self.ghost)) or
              self.board is not None and self.board.lattice is not None,
              "canonical piece pose requires supported board geometry")
        if self.decision == "accepted":
            check(self.board.lattice is not None, "accepted state needs image-supported geometry")
            check(all(cell.occupied is not None for row in self.board.rows for cell in row),
                  "accepted state cannot hide unknown occupancy")
            check(self.active.status in {"observed", "absent"} and
                  self.ghost.status in {"observed", "absent"}, "unknown layers need review")
            check(self.hold.status in {"observed", "empty", "absent"} and
                  self.queue.visibility in {"visible", "absent"} and
                  all(icon.status == "observed" for icon in self.queue.items),
                  "unknown components need review")
            check(self.queue.visibility != "visible" or self.queue.start_observed,
                  "unknown queue start needs review")

    def to_dict(self):
        # JSON normalization ensures callers see the same arrays as file readers.
        return json.loads(json.dumps(asdict(self)))

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"

    @classmethod
    def from_dict(cls, document):
        try:
            fields = dict(document)
            legacy = fields.get("schema_version") == "1.0"
            # Read historical annotations without silently claiming a cropped
            # queue's first visible item is the next item. Writes always use 1.1.
            if legacy:
                fields["schema_version"] = SCHEMA_VERSION
            board = fields.pop("board")
            if board is not None:
                board = dict(board)
                raw_lattice = board.pop("lattice")
                lattice = None if raw_lattice is None else Lattice(**{
                    key: tuple(value) for key, value in raw_lattice.items()})
                rows = tuple(tuple(Cell(**cell) for cell in row) for row in board.pop("rows"))
                check(not board, "unknown board fields")
                board = Board(rows, lattice)
            active = dict(fields.pop("active"))
            ghost = dict(fields.pop("ghost"))
            for layer in (active, ghost):
                layer["cells"] = tuple(tuple(cell) for cell in layer.get("cells", ()))
            if legacy and board is not None and board.lattice is None:
                # v1 annotations predate the shared canonical-frame invariant.
                # Preserve the file; read it with masked targets, not latent pose.
                active, ghost = {"status": "unknown"}, {"status": "unknown"}
                visibility = "cropped" if any(cell.visibility == "cropped" for row in board.rows for cell in row) else "hidden"
                board = Board(tuple(tuple(Cell(None, visibility=visibility) for _ in row) for row in board.rows), None)
                if fields.get("decision") == "accepted":
                    fields["decision"] = "needs_review"
            queue = dict(fields.pop("queue"))
            queue["items"] = tuple(Icon(**item) for item in queue["items"])
            return cls(board=board, active=FallingPiece(**active), ghost=FallingPiece(**ghost),
                       queue=Queue(**queue), hold=Icon(**fields.pop("hold")), **fields)
        except (TypeError, KeyError, AttributeError) as error:
            raise ValueError(f"invalid VisionState document: {error}") from error

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))
