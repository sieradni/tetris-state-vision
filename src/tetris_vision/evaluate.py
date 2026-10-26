"""Exact complete-state scoring. Abstention is a failure on readable examples."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

from .state import VisionState, check


def semantic_key(state: VisionState, *, colored=False, ghost=False):
    if state.board is None:
        return None
    rows = tuple(tuple((cell.occupied, cell.semantic) if colored else cell.occupied
                       for cell in row) for row in state.board.rows)
    piece_key = lambda layer: (layer.status, layer.piece, tuple(sorted(layer.cells)))
    queue = (state.queue.visibility, state.queue.start_observed, state.queue.end_observed,
             tuple((item.status, item.piece) for item in state.queue.items))
    key = (state.player, rows, piece_key(state.active), queue, (state.hold.status, state.hold.piece))
    return key + (piece_key(state.ghost),) if ghost else key


def exact_state(target: VisionState, predicted: VisionState | None, *, colored=False, ghost=False):
    return bool(target.decision == "accepted" and predicted is not None and
                predicted.decision == "accepted" and
                semantic_key(target, colored=colored, ghost=ghost) ==
                semantic_key(predicted, colored=colored, ghost=ghost))


def wilson(successes, total):
    if total == 0:
        return None
    z = 1.959963984540054
    p, d = successes / total, 1 + z*z/total
    center = (p + z*z/(2*total)) / d
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / d
    return [max(0, center-radius), min(1, center+radius)]


@dataclass(frozen=True)
class Example:
    sample_id: str
    group_id: str
    family: str
    target: VisionState


def evaluate(examples: list[Example], predictions: dict[str, VisionState]):
    ids = [example.sample_id for example in examples]
    check(len(ids) == len(set(ids)), "duplicate ground-truth id")
    check(not set(predictions) - set(ids), "predictions contain unknown sample ids")
    slices = defaultdict(lambda: {"readable": 0, "exact": 0, "abstained": 0})
    readable = correct = colored_correct = colored_total = all_layers = abstained = confident_wrong = 0
    ambiguous = ambiguous_accepted = 0
    groups = set()
    for example in examples:
        target, prediction = example.target, predictions.get(example.sample_id)
        groups.add(example.group_id)
        if target.decision != "accepted":
            ambiguous += 1
            ambiguous_accepted += int(prediction is not None and prediction.decision == "accepted")
            continue
        readable += 1
        success = exact_state(target, prediction)
        abstention = prediction is None or prediction.decision != "accepted"
        correct += success
        abstained += abstention
        confident_wrong += int(not success and not abstention)
        all_layers += exact_state(target, prediction, ghost=True)
        type_identifiable = all(cell.occupied is not True or cell.semantic is not None
                                for row in target.board.rows for cell in row)
        colored_total += type_identifiable
        colored_correct += int(type_identifiable and exact_state(target, prediction, colored=True))
        item = slices[example.family]
        item["readable"] += 1
        item["exact"] += success
        item["abstained"] += abstention
    return {"images": len(examples), "independent_groups": len(groups), "readable": readable,
            "exact_solver_state": correct, "exact_all_visual_layers": all_layers,
            "exact_rate": correct/readable if readable else None,
            "image_wilson_95": wilson(correct, readable),
            "abstained_readable": abstained, "accepted_wrong_readable": confident_wrong,
            "colored_identifiable": colored_total, "exact_colored_state": colored_correct,
            "ambiguous_or_unsupported": ambiguous, "accepted_ambiguous": ambiguous_accepted,
            "by_family": dict(slices),
            "limitation": "Image-level interval assumes independence; no cluster-aware release claim. "
                          "Renderer self-comparisons are label checks, not model accuracy."}
