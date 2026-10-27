"""Explicit data-policy routing for source-frozen streaming and fixed-fit studies.

The fixed drawings have no stochastic data seed. The trainer's seed still binds
model initialization/recovery. No model inputs or targets are constructed here.
"""
import json

from m2_interleaved_order import POLICY as INTERLEAVED, distinct_groups as interleaved_groups, interleaved_spec
from m2_positive_control_data import sample_control
from m2_stream_data import StreamCase, sample_case

STREAM = "stream-v1"
ALL_CLASS = "all-class-fit-v1"
FIT_POLICIES = (ALL_CLASS, INTERLEAVED)
POLICIES = (STREAM, *FIT_POLICIES)


def validate_curriculum(curriculum, component_capacity=0):
    if type(curriculum) is not str or curriculum not in POLICIES:
        raise ValueError("explicit supported curriculum required")
    if type(component_capacity) is not int or component_capacity != 0 and not 10 <= component_capacity <= 64:
        raise ValueError("component capacity must be0 or10..64")
    if curriculum in FIT_POLICIES and component_capacity:
        raise ValueError("all-class fit is a capacity0 diagnostic")


def training_spec(step, curriculum=STREAM, *, profile_square=False):
    validate_curriculum(curriculum)
    if type(step) is not int or not 1 <= step <= 4096 or type(profile_square) is not bool:
        raise ValueError("finite global step1..4096 and boolean profile flag required")
    if profile_square:
        if step > 2:
            raise ValueError("square profile has only two updates")
        return 0, 0
    if curriculum == INTERLEAVED:
        return interleaved_spec(step)
    return (((step - 1) // 2) % 16, (step - 1) % 2) if curriculum == ALL_CLASS else ((step - 1) // 3, (step - 1) % 3)


def probe_specs(split, groups, curriculum=STREAM):
    validate_curriculum(curriculum)
    if split not in ("training", "development") or type(groups) is not int or not 1 <= groups <= 12:
        raise ValueError("explicit probe split and groups1..12 required")
    return tuple((group, view) for group in range(16 if split == "training" and curriculum in FIT_POLICIES else groups)
                 for view in range(2 if split == "training" and curriculum in FIT_POLICIES else 3))


def sample_for(group, view, split, seed, *, curriculum=STREAM, component_capacity=0, profile_square=False):
    validate_curriculum(curriculum, component_capacity)
    if split not in ("training", "development") or type(seed) is not int or not 0 <= seed <= 2**31 - 1 or type(profile_square) is not bool:
        raise ValueError("explicit split, finite seed and boolean profile flag required")
    if curriculum in FIT_POLICIES and split == "training":
        if profile_square and (group, view) != (0, 0):
            raise ValueError("fit resource profile must use its first full-square drawing")
        case = sample_control(group, view)
        return StreamCase(case.frame, {**case.record, "curriculum": curriculum, "profile_square": profile_square})
    return sample_case(group, view, split, seed, profile_square=profile_square, final_capture=bool(component_capacity))


def training_case(step, seed, *, curriculum=STREAM, component_capacity=0, profile_square=False):
    group, view = training_spec(step, curriculum, profile_square=profile_square)
    return sample_for(group, view, "training", seed, curriculum=curriculum,
                      component_capacity=component_capacity, profile_square=profile_square)


def distinct_training_groups(completed, curriculum=STREAM):
    validate_curriculum(curriculum)
    if type(completed) is not int or not 0 <= completed <= 4096:
        raise ValueError("completed cursor0..4096 required")
    if curriculum == INTERLEAVED:
        return interleaved_groups(completed)
    return min(16, (completed + 1) // 2) if curriculum == ALL_CLASS else (completed + 2) // 3


def replay_case(record, *, split, curriculum=STREAM, component_capacity=0):
    """Verify a complete serialized recipe; callers separately bind global order."""
    validate_curriculum(curriculum, component_capacity)
    if type(record) is not dict:
        raise ValueError("serialized recipe object required")
    try:
        if curriculum in FIT_POLICIES and split == "training":
            if record.get("curriculum") != curriculum or record.get("split") != "fit":
                raise ValueError("fit recipe identity differs")
            case = sample_for(record["index"], record["view"], split, 0, curriculum=curriculum,
                              component_capacity=component_capacity, profile_square=record["profile_square"])
        else:
            if record.get("split") != split or "curriculum" in record:
                raise ValueError("stream recipe identity differs")
            case = sample_for(record["group_index"], record["view"], split, record["base_seed"], curriculum=curriculum,
                              component_capacity=component_capacity, profile_square=record["profile_square"])
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("invalid serialized curriculum recipe") from error
    if json.loads(json.dumps(case.record)) != record:
        raise ValueError("recipe/pixel/state identity differs from replay")
    return case
