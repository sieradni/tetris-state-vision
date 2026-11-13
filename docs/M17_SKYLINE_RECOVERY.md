# M17 — deciding the board as one skyline

Declared before any receipt. Status at declaration: engine, admission path and tests written and green; no block run.

## Why this arm, after V-139

V-139 corrected the exact-board metric: the `1 of 28` in the M12 and M13 records was produced by dropping columns whose
decision or label was empty, and under the strict definitions **every arm in the program is 0 of 28** (79 sweeps
audited). The closest any arm ever came is 2 wrong cells on its best board.

So the gate is real, not a measurement artefact, and it is far from moving. Sixteen arms decided every column on its own.
Under the monotone read a column is a step, so a board's state is a *skyline* — one height per column — and nothing in
the per-column rule asks those ten decisions to form one coherent state. Boards now sit within about a row of the truth,
which is exactly where independent decisions jitter: a column can flip one row because its own evidence is marginal while
its neighbours' is not. This arm decides the board jointly.

## The mechanism, exactly

`--occupancy-skyline-recovery 1`, applied **after** the envelope, centring and the crossing rule, recovers the board's
skyline with a dynamic programme over heights `0..visible_rows`:

    emission_c(h) = sum(logits below h) - sum(logits at or above h)      # signed mass on each side of the split
    objective     = sum_c emission_c(h_c) - 1.0 * sum_c |h_c - h_(c-1)|  # one skyline, one penalty
    decided       = every cell put on the side of its recovered height, keeping the read's own magnitude

The emission is maximised exactly at the boundary of a monotone column and is *sharp* in the split row — unlike a
difference of means, which is nearly flat across the rows just above a step, so a penalty would swamp it. The penalty is
one logit, the same unit the repository's declared margin uses; beside a column's mass (tens of logits) it only decides
near-ties, which is precisely the ±1 jitter this arm targets.

**Declared constants, all a priori:** 21 states (heights `0..20`, the board's own geometry), one penalty (`1.0` logit),
one numerical floor (`1e-4` for the magnitude, so a decided cell is never exactly at the boundary). No label, no target,
no prevalence, no threshold grid, no swept constant, no fitted weight, and no parameter to train. Ties resolve to the
lower height, so the programme is deterministic. The function's signature is `(logits, visible_rows, penalty)` and its
executable body names no label, target, occupied mask, prevalence or threshold grid — asserted by a test, not described.

## Declared falsification (before the run)

Two conditions, both recorded whatever happens:

1. **Recovery falsified** if the primary at the declared boundary does not exceed **the per-column rule's own reading**
   (M16's ruled arm, `0.405439`) on the same read. Then the columns' independence was not what the remaining errors were
   made of, and the honest record is that a coherence prior buys nothing here — with no penalty sweep, no state-grid
   change and no re-tuned arm offered in its place.
2. **Gate falsified** if the strict exact-board count stays at **0 of 28**. This is expected to be the likely outcome; it
   is declared here so that it cannot be reported later as a surprise or softened into the boundary number.

If the gate moves, the record must say by how much **and** whether the newly exact boards are genuinely exact — every
visible cell right — or whether they are the same class of artefact V-139 found (a board scored on a subset of its
columns). The strict cell definition and the strict boundary definition are both reported, per arm.

## Arms

| arm | flags | role |
|---|---|---|
| `m17-skyline-control-r1` | all structural options `0` | must reproduce AUC `0.8226433610683003` and plain F1@`0` `0.381640` exactly, else the comparison is void |
| `m17-skyline-recovered-r1` | envelope + centring + crossing rule + skyline recovery | the joint decision on the best published read |

Both: 64 steps, seed `20261026`, `--init-weights-from runs/m3-percell-balance-r1`, `--development-groups 6`,
`--evaluate-every 64`, `--checkpoint-every 32`/`64`, then the standing 54-forward sweep. Evidence
`docs/evidence/m17-skyline-recovery-r1/`, instrument `m17-skyline-recovery-r1`, family `m17-skyline-`.

Field-off identity: the control is the published read with every structural option at `0`, so it must reproduce the
published `0.8226433610683003` / `0.381640` exactly — that is the bit-for-bit proof that the field off is the published
function.

## Refusals, through the real parser with stderr asserted

`0` and `1` admitted, `0` the published per-column decision; `2`, `-1` and non-integers refused by the parser. Beside the
option the trainer refuses the normalised output, the five supervision terms and every read-moving lever (`context`,
`detail_stencil`, `resolution`, `deform`, `placed`, `column`, `offset_rows`, `lattice_free`), each asserting the declared
wording. The other three structural read options compose with it by declaration.

## Aliasing guard

At curator level, by assertion: each sweep's `weights_sha256` equals *its own* study's `checkpoint_sha256`; the two arms'
weights differ; the arms' configs differ in exactly `occupancy_centre`, `occupancy_crossing_threshold`,
`occupancy_skyline_recovery` and `occupancy_stack_monotone`; and the neighbour checkpoint (M15's combined arm) is
excluded **by name**.

## Reported per arm, on the corrected strict metric

Plain-binary F1 at the declared boundary (`0.0`), AUC, true positives and false negatives, **strict exact boards** (every
visible cell right) and **strict boundary-exact boards** (every column's decided top right, an empty column compared as a
value), row error, occupancy loss, and the swept optimum for reference.

## Realisation and Measured

**Realisation.** `skyline_shift` in `src/tetris_vision/model.py` (a dynamic programme over `0..visible_rows` per column,
emission = signed mass on each side of the split, penalty `1.0` logit, deterministic tie-break to the lower height),
`occupancy_skyline_recovery` as a validated config field refused beside `occupancy_normalise` and admitted beside the
envelope, centring and crossing rule, applied **last** in the head; `--occupancy-skyline-recovery 0|1` with the declared
refusals; `require_occupancy_skyline_recovery` in the admission join over config, arguments and recovery contract; family
`m17-skyline-` registered before launch. `tests/test_skyline_recovery.py` (22 tests) covers a clean board recovered
exactly, the marginal-column repair the joint decision exists for, determinism, one-skyline-not-ten-crossings, spawn-row
and per-board isolation, the penalty being the only constant, the declared composition order, the leakage guard, config
validation, the admission join, every refusal through the real parser, and the record bindings.

Receipts: `docs/evidence/m17-skyline-recovery-r1/scope.json` (curation at `1,900 MiB`), `runs/m17-skyline-<arm>-r1/`
studies (64 steps, `1,792 MiB`), sweeps (54 forwards, `640 MiB`). One new catalogue, `5c02208e`. Every stage `passed 0`.

**Field-off identity.** Control reproduced AUC `0.8226433610683003` and plain F1 at `0.0` `0.381640` exactly, which is
the proof that the option off is the published function.

| arm | AUC | plain-binary F1 at `0.0` | best swept F1 | TP / FN | cells positive at `0.0` | strict exact | boundary-exact | row error | occupancy loss |
|---|---|---|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 661 / 6 | 2,797 (49.95%) | 0 of 28 | 0 of 28 | +7.819 | 0.7684 |
| envelope + centring + rule + skyline | 0.620664 | **0.308333** | 0.330257 @ `-0.15` | 592 / 75 | 3,173 (56.66%) | **0 of 28** | **0 of 28** | +3.993 | 0.5988 |

**Aliasing, by assertion.** The arms' configs differ in exactly `occupancy_centre`, `occupancy_crossing_threshold`,
`occupancy_skyline_recovery` and `occupancy_stack_monotone`; each sweep's weights equal its own study's checkpoint; the
two differ from each other; and the neighbour block's checkpoint (M15's combined arm) is excluded by name. Nine
reference rows (M16 rule, M15 combined, M15 envelope, M14, M13–M9) are carried beside the block.

| reference | AUC | plain F1 at `0.0` |
|---|---|---|
| *M16 crossing rule* | *0.835241* | *0.405439* |
| *M15 combined read* | *0.888912* | *0.390805* |
| *M15 envelope only* | *0.845948* | *0.409161* |
| *M14 centred read* | *0.815395* | *0.374142* |
| *M13 midpoint anchor* | *0.830500* | *0.310966* |
| *M12 dense hinge* | *0.862838* | *0.354770* |
| *M11 quantile anchor* | *0.815340* | *0.264308* |
| *M10 extremal anchor* | *0.885462* | *0.310966* |
| *M9 hinge* | *0.864667* | *0.251701* |

### Reading: `skyline-fails`, secondary `ranking-degraded`

Primary delta against control **`-0.073307`** and against the per-column rule **`-0.097106`**; AUC **`-0.201979`**.
**Both declared falsifications fired.**

1. **Recovery falsified**: the joint decision does not beat the per-column rule it was meant to improve (`0.308333`
   against `0.405439`), and its AUC is the worst any arm in the program has measured (`0.620664`).
2. **Gate falsified**: strict exact boards stay at **`0 of 28`**, and so does boundary-exact. No board newly fits - there
   is nothing to audit for the V-139 class of artefact, because nothing moved.

**What went wrong, stated as the mechanism rather than as a tuning note.** The emission is a *volume* criterion - the
signed mass on each side of a split - so it prefers heights that accumulate positive read, not the level at which the
column actually steps: the arm decides `56.66%` of cells positive where control decides `49.95%`, recall collapses from
`0.9910` to `0.8876` (`6` -> `75` false negatives), and the one quantity that did improve in the earlier arms, the row
error, is *worse* than the read it was applied to (`3.993` rows against the M16 rule's `1.189`). The penalty of one logit
is small beside a column's mass, so the programme behaved close to `argmax` of a volume score, which is not the crossing
the read states. Replacing each cell's magnitude with its side of the recovered height then destroyed the cross-board
ordering the AUC measures, which is why the ranking fell to `0.62` rather than merely dipping.

**No re-tune is offered or performed.** A softer penalty, a different emission, or a magnitude-preserving emission are all
plausible variants of *this* arm, and every one of them is a change to the statistic being maximised - exactly the
per-arm tuning the declaration forbids. The honest record is that the joint-coherence hypothesis, as declared, is
falsified on this population.

### What the failure tells us about the gate

Seventeen arms now separate cleanly by *where* they act. Decision-side arms (`M16`'s per-column rule, `M17`'s joint
skyline) cannot get below the read's own row error, and the skyline prior actively hurts. Read-side arms carry the best
placement ever measured (`M15` envelope, `1.189` rows) and the best ranking (`M15` combined, `0.888912` AUC), and the
closest any board ever got to exact is **2 wrong cells** (`m3-occlatfree-lever`, AUC `0.840147`, three of its boards
within ten cells). A 200-cell board at AUC `0.89` implies roughly ten to twenty expected cell errors; the gate asks for
zero on all 28. So the remaining distance is a **per-cell read limit**, not a decision or structure limit: every structural
arm now improves placement or ranking and none improves exactness, and the only board that ever came within two cells was
produced by a cell-read lever.

### Limits

Twenty-eight boards, twelve images, one catalogue, one block, one seed, one launch; synthetic label-integrity
populations, not photographed boards, not complete states, not image quality.

## Limits

Twenty-eight boards, twelve images, one catalogue, one block, one seed, one launch. Synthetic label-integrity
populations: not photographed boards, not complete states, not image quality.
