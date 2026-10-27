"""Optional independent visible-cell evidence; never subtract from locked state."""
import torch
from torch import nn

from .state import check


class LayerEvidence(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.cells = nn.Linear(dimension, 2)

    def forward(self, cells):
        # Canonical tokens are bottom-first: first200 are the visible20 rows.
        visible = cells[:, :200]
        logits = self.cells(visible).transpose(1, 2)
        pooled = logits.softmax(-1) @ visible
        return logits.reshape(cells.shape[0], 2, 20, 10), pooled


def dense_targets(state, base, canonical_support):
    """Project only supported public layer cells through prepared image masks."""
    check(state.model_id is None, "observation labels required, not a model output")
    check(canonical_support.shape == (1, 20, 10) and canonical_support.dtype == torch.bool,
          "one prepared canonical image-coverage mask required")
    known = (canonical_support[:, None].expand(-1, 2, -1, -1) &
             base["layer_known"][:, :, None, None] & base["registered"][:, None, None, None])
    occupied = torch.zeros_like(known, dtype=base["occupied"].dtype)
    for role, layer in enumerate((state.active, state.ghost)):
        if not bool(base["layer_known"][0, role]):
            continue
        check(layer.status in {"observed", "absent"}, "known role must have a supported public status")
        if layer.status == "observed":
            for column, row in layer.cells:
                if 0 <= column < 10 and 0 <= row < 20 and bool(known[0, role, row, column]):
                    occupied[0, role, row, column] = 1
    return {"occupied": occupied, "known": known}


def grid_metrics(logits, target, layer_type):
    """Visible grids only; absent and observed denominators remain separate."""
    known = target["known"]
    right = ((logits > 0) == target["occupied"].bool()) | ~known
    full = known.flatten(2).all(-1)
    exact = right.flatten(2).all(-1) & full
    observed = full & (layer_type > 0)
    return {"dense_layer_grids_known": int(full.sum()), "dense_layer_grids_exact": int(exact.sum()),
            "dense_observed_visible_grids_known": int(observed.sum()),
            "dense_observed_visible_grids_exact": int((exact & observed).sum())}
