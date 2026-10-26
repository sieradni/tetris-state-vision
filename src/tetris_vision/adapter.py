"""Pure solver request contract. No runtime dependency on tetrguide."""
from dataclasses import dataclass

from .state import VisionState


class IncompleteState(ValueError):
    pass


@dataclass(frozen=True)
class SolverInput:
    rows_bottom_first: tuple[str, ...]
    pieces: str
    hold: str | None
    hold_available: bool


def to_solver_input(state: VisionState, *, hold_available: bool) -> SolverInput:
    """Temporal hold availability is caller-supplied; it cannot be read from pixels."""
    if type(hold_available) is not bool:
        raise IncompleteState("hold availability must be supplied explicitly as a boolean")
    if state.decision != "accepted" or state.board is None:
        raise IncompleteState("state needs review or has no board")
    if any(cell.occupied is None for row in state.board.rows for cell in row):
        raise IncompleteState("unknown locked occupancy")
    if state.active.status != "observed":
        raise IncompleteState("current piece is unresolved; the first preview is not current")
    if state.queue.visibility != "visible" or any(item.status != "observed" for item in state.queue.items):
        raise IncompleteState("queue is absent, cropped or contains unknown slots")
    if state.hold.status not in {"empty", "observed"}:
        raise IncompleteState("hold panel is absent or unreadable")
    return SolverInput(tuple("".join("1" if cell.occupied else "0" for cell in row)
                             for row in state.board.rows),
                       state.active.piece + "".join(item.piece for item in state.queue.items),
                       state.hold.piece, hold_available)
