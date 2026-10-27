"""Prospective order-only fit policy; not integrated/admitted for training yet.

Each32-step cycle contains the original32 (index,view) pairs once. Every eight
successive positions within a cycle contain all seven piece classes and absence.
Hash ranks depend only on this versioned policy's fixed seed and global cycle;
they consume no process RNG and can be recovered at any finite global cursor.
"""
import hashlib

ORDER_SEED = 20261027
POLICY = "all-class-interleaved-v1"


def cycle_order(cycle):
    if type(cycle) is not int or not 0 <= cycle < 128:
        raise ValueError("finite interleaved cycle0..127 required")

    def rank(item):
        return hashlib.sha256(f"{POLICY}:{ORDER_SEED}:{cycle}:{item}".encode("ascii")).digest()

    poses = {label: sorted(((variant, view) for variant in (0, 1) for view in (0, 1)),
                           key=lambda pair: rank(f"pose:{label}:{pair[0]}:{pair[1]}"))
             for label in range(8)}
    result = []
    for round_index in range(4):
        labels = sorted(range(8), key=lambda label: rank(f"class:{round_index}:{label}"))
        for label in labels:
            variant, view = poses[label][round_index]
            result.append((2 * label + variant, view))
    return tuple(result)


def interleaved_spec(step):
    if type(step) is not int or not 1 <= step <= 4096:
        raise ValueError("finite global step1..4096 required")
    cycle, offset = divmod(step - 1, 32)
    return cycle_order(cycle)[offset]


def distinct_groups(completed):
    if type(completed) is not int or not 0 <= completed <= 4096:
        raise ValueError("finite completed cursor0..4096 required")
    return len({index for index, _ in cycle_order(0)[:min(completed, 32)]})
