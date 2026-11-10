"""One matched-start contract for every lever, parameter-adding or not.

V-071: the box-origin curator inherited the geometry lever's guard, which compared the two saved
`initial-weights.pt` digests. That is correct only when the lever adds no parameter. This module states
the condition once, in the form that holds either way, so no later lever can inherit the wrong one:
every published tensor must be bitwise identical, and the lever's own added tensors must sit exactly at
their declared initialization.
"""
import torch


def matched_start(lever_state, baseline_state, *, added=()):
    """Prove both arms start from the same function, not merely the same source file.

    `added` names the tensors the lever adds. As a mapping it declares their exact initial values; as a
    bare set each added tensor must be exactly zero, which is the declared initialization for every
    lever in this campaign and the reason step zero is the published function.
    """
    if not isinstance(lever_state, dict) or not isinstance(baseline_state, dict):
        raise ValueError("initial tensor files must both be state dictionaries")
    declared = dict(added) if isinstance(added, dict) else {name: None for name in added}
    if set(lever_state)-set(baseline_state) != set(declared):
        raise ValueError("the arms differ by tensors outside the declared lever parameters")
    if set(baseline_state)-set(lever_state):
        raise ValueError("the lever arm is missing published tensors")
    for name, tensor in baseline_state.items():
        if not torch.equal(tensor, lever_state[name]):
            raise ValueError(f"shared initial tensor {name} differs between the arms")
    for name, expected in declared.items():
        value = lever_state[name]
        if expected is None:
            if int(torch.count_nonzero(value)):
                raise ValueError(f"declared lever tensor {name} must start at exactly zero")
        elif not torch.equal(value, expected):
            raise ValueError(f"declared lever tensor {name} does not match its declared init")
    return True


def step_zero_identical(lever_summary, baseline_summary,
                        *, probes=("initial_training_probe", "initial_development")):
    """Both arms must report the same measured loss on the same cases at step zero."""
    for name in probes:
        if lever_summary.get(name) != baseline_summary.get(name):
            raise ValueError(f"step zero is not the same measured function in both arms ({name})")
    return True
