"""Finite, explicit image/update mapping for the proposed fixed-fit study.

This module does not train a model and is not an admission to training.
"""
from m2_interleaved_order import interleaved_spec

POLICIES = {"online-interleaved-v1": (1, True),
            "balanced-eight-v1": (8, True),
            "ordered-eight-v1": (8, False)}
IMAGE_BUDGET = 1024


def update_count(policy):
    if type(policy) is not str or policy not in POLICIES:
        raise ValueError("explicit accumulation policy required")
    return IMAGE_BUDGET // POLICIES[policy][0]


def group_specs(policy, update):
    """Return global one-based image cursor and fixed (fixture, view) pairs."""
    total = update_count(policy)
    if type(update) is not int or not 1 <= update <= total:
        raise ValueError("update outside the fixed finite study")
    size, interleaved = POLICIES[policy]
    start = (update-1)*size+1
    rows = []
    for image in range(start, start+size):
        index, view = interleaved_spec(image) if interleaved else (((image-1)//2) % 16, (image-1) % 2)
        rows.append((image, index, view))
    return tuple(rows)


def committed_boundary(policy, images):
    """Map a fully committed image prefix to the optimizer cursor; refuse partial groups."""
    update_count(policy)
    size = POLICIES[policy][0]
    if type(images) is not int or not 0 <= images <= IMAGE_BUDGET or images % size:
        raise ValueError("image cursor must be a complete finite update boundary")
    return images // size


class GradientGroup:
    """Stream one loss at a time; caller must zero, then clip/step only after finish."""
    def __init__(self, size):
        if type(size) is not int or size not in (1, 8):
            raise ValueError("one or eight microsteps required")
        self.size, self.completed = size, 0

    def backward(self, loss):
        if self.completed >= self.size:
            raise ValueError("gradient group already full")
        (loss / self.size).backward()
        self.completed += 1

    def finish(self):
        if self.completed != self.size:
            raise ValueError("partial gradient groups cannot be stepped")


def apply_group(model, optimizer, records, loss_for):
    """Apply one complete streamed group; never step after a failed microstep.

    This primitive does not publish checkpoints or catch optimizer-step errors;
    the driver must terminate without committing if the step itself fails.
    """
    import math
    import torch
    if not isinstance(records, (tuple, list)) or len(records) not in (1, 8) or not callable(loss_for):
        raise ValueError("finite one- or eight-image group and loss callable required")
    optimizer.zero_grad(set_to_none=True)
    group = GradientGroup(len(records))
    values = []
    try:
        for record in records:
            loss = loss_for(record)
            if not torch.is_tensor(loss) or loss.numel() != 1 or not bool(torch.isfinite(loss).all()):
                raise ValueError("finite scalar loss required")
            values.append(float(loss.detach()))
            group.backward(loss)
        group.finish()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    optimizer.step()
    return {"mean_loss": math.fsum(values)/len(values), "gradient_norm": float(norm), "images": len(values)}
