"""The declared deciding population for stream probe replays (instrument change).

`docs/M3_WIDENED_DECIDING_POPULATION.md` states the change once. The deciding split of the
fit gate, the per-cell gate, the occupancy ranking sweep and the identity discrimination rule
is the **training** probe population, and every published M3 run recorded three groups (nine
images) of it while `--development-groups` already admitted twelve. This module widens the
deciding population to the canonical twelve groups (thirty-six images) with every gate
threshold unchanged, and it is the single implementation the error replay and the occupancy
sweep both call, so the two populations cannot drift apart (the V-052 declaration pattern).

Group indices 3..11 under a run's own base seed are that run's own training drawings:
`m2_curriculum.training_spec` maps step -> (group, view), so these are the stream's steps
10..36. The widening therefore adds no new data, no new seed and no new appearance family;
like the recorded nine, these are drawings the model was trained on, which keeps the gates
training-fit instruments and not generalization estimates.
"""

RECORDED_TRAINING_GROUPS = 3
CANONICAL_TRAINING_GROUPS = 12
TRAINING_VIEWS = 3


def require_training_population(recorded, evaluated):
    """Only the recorded three groups or the canonical twelve are admissible."""
    if type(recorded) is not int or type(evaluated) is not int:
        raise ValueError("integer training group counts required")
    if recorded != RECORDED_TRAINING_GROUPS:
        raise ValueError("recorded training probe population is three groups")
    if evaluated not in (RECORDED_TRAINING_GROUPS, CANONICAL_TRAINING_GROUPS):
        raise ValueError("training population must stay recorded or reach the canonical twelve groups")
    return evaluated


def training_extension_specs(base_seed, recorded=RECORDED_TRAINING_GROUPS,
                             evaluated=CANONICAL_TRAINING_GROUPS):
    """The drawings that widen the deciding population, in training-stream order.

    ``base_seed`` is the run's own recorded probe seed, so each spec is exactly the
    ``sample_for(group, view, "training", base_seed)`` drawing the stream consumed.
    """
    if type(base_seed) is not int or not 0 <= base_seed <= 2**31-1:
        raise ValueError("recorded base seed required")
    require_training_population(recorded, evaluated)
    return tuple({"group_index": group, "view": view, "base_seed": base_seed}
                 for group in range(recorded, evaluated) for view in range(TRAINING_VIEWS))
