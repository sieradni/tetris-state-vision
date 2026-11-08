"""Shared bounded queue/hold queries and image-supported M3 loss targets.

Raw experimental heads only. Neither this module nor its loss admits model
predictions as labels, reconstructs hidden queue length or certifies accuracy.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .component_targets import component_labels
from .state import check


class ComponentDecoder(nn.Module):
    """Board-conditioned role/index queries; same parameters at every index.

    At most65 positions (64 items plus EOS), processed in chunks of16. No
    per-position classifier/embedding or prefix-length-dependent position code.
    The two global roles are queue visibility/anchors and hold status/identity.
    """
    def __init__(self, dimension, heads):
        super().__init__()
        check(type(dimension) is int and 16 <= dimension <= 256 and dimension % 2 == 0 and
              type(heads) is int and 1 <= heads <= 8 and dimension % heads == 0,
              "bounded compatible component dimension/head count required")
        self.dimension = dimension
        self.roles = nn.Parameter(torch.randn(1, 2, dimension)*.02)
        self.index_query = nn.Parameter(torch.randn(1, 1, dimension)*.02)
        self.attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.norm = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(nn.Linear(dimension, 2*dimension), nn.GELU(), nn.Linear(2*dimension, dimension))
        positions = torch.arange(65).float()[:, None]
        frequency = torch.exp(-math.log(10000)*torch.arange(0, dimension, 2).float()/dimension)
        encoded = torch.stack(((positions*frequency).sin(), (positions*frequency).cos()), -1).flatten(1)
        self.register_buffer("index_encoding", encoded)
        self.queue_tokens = nn.Linear(dimension, 8)  # Seven pieces + observed EOS.
        self.shared_type = None  # Declared shared read-out; None keeps the separate heads.
        self.queue_eos = None
        # Declared prototype read-out: per-class image queries, not one fixed vector map.
        self.piece_queries = None
        self.hold_queries = None
        # Declared positional read-out: every queue position reads the frame through its
        # own query, so a position's own content can change its class.
        self.position_queries = None
        self.queue_presence = nn.Linear(dimension, 1)
        self.queue_slot_status = nn.Linear(dimension, 3)
        self.queue_status = nn.Linear(dimension, 4)
        self.queue_anchors = nn.Linear(dimension, 2)  # start/end observed, independently.
        self.hold_status = nn.Linear(dimension, 5)
        self.hold_piece = nn.Linear(dimension, 7)

    def forward(self, memory, memory_valid, board, *, capacity):
        check(type(capacity) is int and 1 <= capacity <= 64, "component capacity must be1..64")
        check(memory.ndim == 3 and 1 <= memory.shape[0] <= 4 and 1 <= memory.shape[1] <= 516 and
              memory.shape[2] == self.dimension and memory.is_floating_point(), "bounded component memory required")
        check(memory_valid.shape == memory.shape[:2] and memory_valid.dtype == torch.bool and
              bool(memory_valid.any(1).all()), "observed component context required")
        check(board.shape == (memory.shape[0], 1, self.dimension) and board.dtype == memory.dtype and
              board.device == memory.device and memory_valid.device == memory.device and bool(torch.isfinite(memory).all()) and
              bool(torch.isfinite(board).all()), "finite board-conditioned component query required")
        def read(query):
            value = query+self.attention(self.norm(query), memory, memory, key_padding_mask=~memory_valid, need_weights=False)[0]
            return value+self.feed_forward(self.norm(value))
        roles = read(self.roles+board)
        parts = []
        for start in range(0, capacity+1, 16):
            index = self.index_encoding[start:min(start+16, capacity+1)].to(dtype=memory.dtype)
            query = self.index_query+index[None]+board+roles[:, :1]
            parts.append(read(query))
        values = torch.cat(parts, dim=1)
        if self.shared_type is not None:
            # Declared lever: piece identity is the cell-type read-out plus one EOS logit.
            tokens = torch.cat((self.shared_type(values)[..., :7], self.queue_eos(values)), -1)
            hold = self.shared_type(roles[:, 1])[..., :7]
        elif self.piece_queries is not None:
            # Declared lever: every class is an image query, so the classifier's class
            # vectors depend on the observed frame rather than being one fixed map.
            scale = self.dimension**.5
            prototypes = read(self.piece_queries.expand(memory.shape[0], -1, -1)+board)
            if self.position_queries is not None:
                # Declared lever: the position side is image-conditioned too, so two
                # positions can read different panel content instead of sharing one code.
                positioned = read(self.position_queries[:, :capacity+1].expand(memory.shape[0], -1, -1)
                                  +board+roles[:, :1])
                tokens = positioned@prototypes.transpose(1, 2)/scale
            else:
                tokens = values@prototypes.transpose(1, 2)/scale
            hold_set = read(self.hold_queries.expand(memory.shape[0], -1, -1)+board+roles[:, 1:2])
            hold = (hold_set.mean(1)[:, None, :]@prototypes[:, :7].transpose(1, 2)).squeeze(1)/scale
        else:
            tokens, hold = self.queue_tokens(values), self.hold_piece(roles[:, 1])
        return {"queue_tokens": tokens,
                "queue_presence": self.queue_presence(values[:, :capacity]).squeeze(-1),
                "queue_slot_status": self.queue_slot_status(values[:, :capacity]),
                "queue_status": self.queue_status(roles[:, 0]), "queue_anchors": self.queue_anchors(roles[:, 0]),
                "hold_status": self.hold_status(roles[:, 1]), "hold_piece": hold}

    def share_type_readout(self, classifier):
        """Declared lever: identity reads through the cell-type classifier."""
        check(type(classifier) is nn.Linear and classifier.weight.shape == (8, self.dimension) and
              classifier.bias is not None and classifier.bias.shape == (8,),
              "shared identity read-out needs the8-class cell-type head")
        self.shared_type = classifier
        self.queue_eos = nn.Linear(self.dimension, 1)
        # The separate heads would be dead parameters with no resident Adam state, which
        # the resource profile refuses; remove them instead of carrying them unused.
        self.queue_tokens = None
        self.hold_piece = None
        return self

    def use_positional_queries(self):
        """Declared lever: per-position image queries on top of the validated banks.

        The class bank and the four-query hold bank are unchanged; only the queue
        position side changes. Each of the `capacity+1` positions carries its own
        learned query that attends the observed component memory and the board, so a
        position's own content can move its class logits. That is the analogue, one
        level down, of the per-class image queries: it makes position discrimination
        possible by construction rather than through one shared index code.
        """
        self.use_prototype_readout()
        self.position_queries = nn.Parameter(torch.randn(1, 65, self.dimension)*.02)
        return self

    def use_prototype_readout(self):
        """Declared lever: class identity is a per-piece query over the observed frame.

        Eight learned class queries attend the component memory and the board, so the
        read-out for every class is image-conditioned and distinct by construction; a
        position is classified against those image-derived class vectors. The hold
        piece reads its own four-query bank over the same memory instead of one shared
        role vector, so it is not forced through a single collapsed embedding.
        """
        check(self.shared_type is None and self.queue_tokens is not None and self.hold_piece is not None,
              "prototype identity read-out is exclusive with the other read-outs")
        self.piece_queries = nn.Parameter(torch.randn(1, 8, self.dimension)*.02)
        self.hold_queries = nn.Parameter(torch.randn(1, 4, self.dimension)*.02)
        self.queue_tokens = None
        self.hold_piece = None
        return self


def component_tensors(states, *, capacity):
    """Tensor projection of the verified observation contract, batch1..4."""
    check(type(states) in (list, tuple) and 1 <= len(states) <= 4, "bounded component observation batch required")
    labels = [component_labels(state, capacity=capacity) for state in states]
    result = {"capacity": capacity}
    for key in ("queue_status", "hold_status", "queue_tokens", "queue_slot_status"):
        result[key] = torch.tensor([label[key] for label in labels], dtype=torch.long)
    for key in ("queue_start_observed", "queue_end_observed", "queue_token_known", "queue_slot_status_known",
                "queue_presence", "queue_presence_known", "hold_piece_known", "queue_count_known"):
        result[key] = torch.tensor([label[key] for label in labels], dtype=torch.bool)
    result["hold_piece"] = torch.tensor([label["hold_piece"] if label["hold_piece_known"] else 0 for label in labels], dtype=torch.long)
    result["queue_count"] = torch.tensor([label["queue_count"] if label["queue_count_known"] else 0 for label in labels], dtype=torch.long)
    return result


def _validate_contract(output, target):
    capacity = target["capacity"]
    check(type(capacity) is int and 1 <= capacity <= 64, "bounded component loss capacity required")
    check(torch.is_tensor(output["queue_tokens"]) and output["queue_tokens"].ndim == 3, "rank3 component token logits required")
    batch = output["queue_tokens"].shape[0]
    check(1 <= batch <= 4, "bounded component loss batch required")
    shapes = {"queue_tokens": (batch, capacity+1, 8), "queue_presence": (batch, capacity),
              "queue_slot_status": (batch, capacity, 3), "queue_status": (batch, 4), "queue_anchors": (batch, 2),
              "hold_status": (batch, 5), "hold_piece": (batch, 7)}
    check(all(output[key].shape == shape and output[key].is_floating_point() and bool(torch.isfinite(output[key]).all()) for key, shape in shapes.items()),
          "finite component output shapes required")
    target_shapes = {"queue_tokens": (batch, capacity+1), "queue_token_known": (batch, capacity+1),
                     "queue_slot_status": (batch, capacity), "queue_slot_status_known": (batch, capacity),
                     "queue_presence": (batch, capacity), "queue_presence_known": (batch, capacity),
                     "queue_status": (batch,), "hold_status": (batch,), "hold_piece": (batch,), "hold_piece_known": (batch,),
                     "queue_start_observed": (batch,), "queue_end_observed": (batch,)}
    bool_keys = {"queue_token_known", "queue_slot_status_known", "queue_presence", "queue_presence_known", "hold_piece_known",
                 "queue_start_observed", "queue_end_observed"}
    check(all(target[key].shape == shape and target[key].dtype == (torch.bool if key in bool_keys else torch.long)
              for key, shape in target_shapes.items()), "component target shapes/dtypes differ")
    check(torch.equal(target["queue_token_known"], target["queue_tokens"] >= 0) and
          torch.equal(target["queue_slot_status_known"], target["queue_slot_status"] >= 0), "component uncertainty mask differs")
    for key, low, high in (("queue_tokens", -1, 7), ("queue_slot_status", -1, 2),
                           ("queue_status", 0, 3), ("hold_status", 0, 4), ("hold_piece", 0, 6)):
        check(bool(((target[key] >= low) & (target[key] <= high)).all()), "component target class outside vocabulary")
    start = target["queue_start_observed"]
    indexed = target["queue_token_known"].any(1) | target["queue_slot_status_known"].any(1) | target["queue_presence_known"].any(1)
    check(not bool((indexed & ~start).any()), "unseen queue start cannot have absolute indexed targets")
    eos = ((target["queue_tokens"] == 7) & target["queue_token_known"]).any(1)
    check(not bool((eos & ~target["queue_end_observed"]).any()), "unseen queue end cannot train EOS")
    check(torch.equal(target["hold_piece_known"], target["hold_status"] == 0), "unobserved hold cannot train piece identity")


def component_loss(output, target, identity_factor=1):
    """Masked semantic/observation losses; all coefficients are pilot hypotheses.

    `identity_factor` scales only the queue-token and hold-piece terms for the
    predeclared component-identity lever;1 is the published objective.
    """
    if type(identity_factor) is not int or identity_factor not in (1, 4):
        raise ValueError("component identity factor must be1 or4")
    _validate_contract(output, target)
    def masked(value, mask):
        return (value*mask).sum()/mask.sum().clamp_min(1)
    def semantic(key, mask):
        # Unknown placeholders never become a class0 training target.
        safe = torch.where(mask, target[key], torch.zeros_like(target[key]))
        logits = output[key].movedim(-1, 1)
        return masked(F.cross_entropy(logits, safe, reduction="none"), mask)
    terms = {"queue_tokens": semantic("queue_tokens", target["queue_token_known"]),
             "queue_slot_status": semantic("queue_slot_status", target["queue_slot_status_known"]),
             "queue_presence": masked(F.binary_cross_entropy_with_logits(output["queue_presence"], target["queue_presence"].float(), reduction="none"),
                                      target["queue_presence_known"]),
             "queue_status": F.cross_entropy(output["queue_status"], target["queue_status"]),
             "queue_anchors": F.binary_cross_entropy_with_logits(output["queue_anchors"], torch.stack((target["queue_start_observed"], target["queue_end_observed"]), -1).float()),
             "hold_status": F.cross_entropy(output["hold_status"], target["hold_status"]),
             "hold_piece": semantic("hold_piece", target["hold_piece_known"])}
    weights = {"queue_tokens": 1.*identity_factor, "queue_slot_status": .25, "queue_presence": .25,
               "queue_status": .25, "queue_anchors": .25, "hold_status": .5, "hold_piece": 1.*identity_factor}
    return sum(weights[key]*value for key, value in terms.items()), terms


@torch.no_grad()
def component_metrics(output, target):
    """Per-image raw-head diagnostics, separating readable and uncertain targets.

    Matching the known parts of a cropped/unknown target is not a complete
    component success. These values are not decoded/API or calibrated accuracy.
    """
    _validate_contract(output, target)
    batch = output["queue_tokens"].shape[0]
    def masked_equal(predicted, truth, mask):
        return ((predicted == truth) | ~mask).reshape(batch, -1).all(1)
    token_right = masked_equal(output["queue_tokens"].argmax(-1), target["queue_tokens"], target["queue_token_known"])
    slot_right = masked_equal(output["queue_slot_status"].argmax(-1), target["queue_slot_status"], target["queue_slot_status_known"])
    presence_right = masked_equal(output["queue_presence"] > 0, target["queue_presence"], target["queue_presence_known"])
    anchors = torch.stack((target["queue_start_observed"], target["queue_end_observed"]), -1)
    queue_right = ((output["queue_status"].argmax(-1) == target["queue_status"]) &
                   ((output["queue_anchors"] > 0) == anchors).all(1) & token_right & slot_right & presence_right)
    hold_right = ((output["hold_status"].argmax(-1) == target["hold_status"]) &
                  masked_equal(output["hold_piece"].argmax(-1), target["hold_piece"], target["hold_piece_known"]))
    present = target["queue_presence"] & target["queue_presence_known"]
    identities_known = (((target["queue_slot_status"] == 0) & target["queue_token_known"][:, :target["capacity"]]) | ~present).all(1)
    eos_known = ((target["queue_tokens"] == 7) & target["queue_token_known"]).any(1)
    queue_readable = ((target["queue_status"] == 1) | ((target["queue_status"] == 0) &
                      target["queue_start_observed"] & target["queue_end_observed"] & identities_known & eos_known))
    hold_readable = target["hold_status"] <= 2  # observed, empty or absent.
    result = []
    for index in range(batch):
        readable = bool(queue_readable[index] & hold_readable[index])
        right = bool(queue_right[index] & hold_right[index])
        result.append({"queue_supported_exact": bool(queue_right[index]), "hold_supported_exact": bool(hold_right[index]),
                       "supported_components_exact": right, "component_state_readable": readable,
                       "exact_readable_components": right if readable else None,
                       "known_queue_tokens": int(target["queue_token_known"][index].sum()),
                       "known_presence_positions": int(target["queue_presence_known"][index].sum()),
                       "hold_identity_known": bool(target["hold_piece_known"][index])})
    return result
