"""Pure image-supported queue/hold labels for the upcoming joint decoder.

Only observation fields enter this projection; corpus/training admission still
verifies provenance and permission. Identified model outputs are refused.
Unknown is masked, never converted into an empty panel or end token.
"""
from .state import VisionState

PIECES = tuple("IOTSZJL")
QUEUE_STATUSES = ("visible", "absent", "cropped", "unknown")
HOLD_STATUSES = ("observed", "empty", "absent", "cropped", "unknown")
SLOT_STATUSES = ("observed", "unknown", "cropped")
EOS = len(PIECES)


def component_labels(state: VisionState, *, capacity=64):
    if not isinstance(state, VisionState):
        raise TypeError("validated VisionState required")
    if state.model_id is not None:
        raise ValueError("observation labels required, not a model output")
    if type(capacity) is not int or not 1 <= capacity <= 64:
        raise ValueError("queue capacity must be 1..64")
    queue = state.queue
    if queue.start_observed and len(queue.items) > capacity:
        raise ValueError("observed queue prefix exceeds capacity; no silent truncation")
    tokens = [-1]*(capacity+1)  # Extra position represents EOS after item64.
    status = [-1]*capacity
    present, presence_known = [False]*capacity, [False]*capacity
    count = None
    if queue.start_observed:
        for index, icon in enumerate(queue.items):
            present[index], presence_known[index] = True, True
            status[index] = SLOT_STATUSES.index(icon.status)
            if icon.status == "observed":
                tokens[index] = PIECES.index(icon.piece)
        if queue.end_observed:
            count = len(queue.items)
            tokens[count] = EOS
            if count < capacity:
                presence_known[count] = True  # Known endpoint, not unseen future.
    hold_known = state.hold.status == "observed"
    return {"capacity": capacity,
            "queue_status": QUEUE_STATUSES.index(queue.visibility),
            "queue_start_observed": queue.start_observed, "queue_end_observed": queue.end_observed,
            "queue_count": count, "queue_count_known": count is not None,
            "queue_tokens": tuple(tokens), "queue_token_known": tuple(value >= 0 for value in tokens),
            "queue_slot_status": tuple(status), "queue_slot_status_known": tuple(value >= 0 for value in status),
            "queue_presence": tuple(present), "queue_presence_known": tuple(presence_known),
            "hold_status": HOLD_STATUSES.index(state.hold.status),
            "hold_piece": PIECES.index(state.hold.piece) if hold_known else None, "hold_piece_known": hold_known}
