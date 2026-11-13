# M14 — removing the read's additive mode

**Status: predeclared.** The mechanism, the arms, the readings and the declared failure mode below were fixed before
the first training receipt of this block existed and are not revised by its result. The `Measured` section is appended
afterwards.

## Why a structural change, not a sixth term

Five supervision terms have been measured on the published read-out and every one of them was falsified by the same
mechanism: a global shift of the read can partially satisfy any per-board statement, so the optimiser buys the term
with a level change and the boundary the consumer applies (`0.0`) is left unused.

| pass | added statement | cells/quantity charged | F1 at `0.0` | swept optimum | positive at `0.0` |
|---|---|---|---|---|---|
| M9 | one-sided hinge | 2 cells | 0.251701 | `2.70` | — |
| M10 | two-sided extremal anchor | 2 cells | 0.310966 | `2.25` | 64.5% |
| M11 | class-quantile anchor | 10% of each class | 0.264308 | `2.25` | 78.0% |
| M12 | dense class-balanced hinge | every cell | 0.354770 | `3.15` | 54.7% |
| M13 | midpoint anchor | one location | 0.310966 | `3.00` | 64.5% |

The only arm that improved on control at all (M12: best swept `+0.054`, out-of-fold `+0.065`) did so on **shape**. Not
one arm placed the boundary. So this arm stops charging the shift and removes it.

## The mechanism

`--occupancy-centre 1` applies, inside the head, before the objective or any scorer sees the tensor:

```
L <- L - mean( L[:, :visible_rows] )          per board, over that board's own visible cells
```

No label, no threshold, no fitted parameter, no other board, nothing read about prevalence. The statistic is the
board's own mean over its *visible* window; the spawn rows above the board are scored by that statistic and never enter
it. Unlike M8's `normalise`, the spread is untouched: dividing by a board statistic rescales the relative structure the
read has learned, while subtracting a constant cannot change any difference between two cells, so

- alignment invariance is **exact**: for any constant `c`, `centre(L + c) == centre(L)` bit-for-bit up to float error;
- the correction is **per board**, not global: a batch's boards each lose their own mean;
- a constant board centres to zeros, and a single visible cell centres to zero;
- `0` is exactly the published function, unchanged in every digit.

## Declared constants — fixed, never revised for this arm

| constant | value | note |
|---|---|---|
| `occupancy_centre` | `0` or `1` | structural option; `1` centres each board's read |
| `ANCHOR_WEIGHT` | — | **not used**: this arm adds no loss term, so there is nothing to weight |
| `margin` | — | **not used**: no band, no tolerance, no target other than `0.0` |

There is nothing to tune here, and that is the point of the arm: no margin, no weight, no quantile, no balance, no
target. The only declaration is where the subtraction happens — in the head, so the tensor the loss differentiates and
the tensor a consumer thresholds are identical.

## What is refused

`--occupancy-centre 1` refuses every term that charges the shift it removes — `--occupancy-midpoint`,
`--occupancy-dense-hinge`, `--occupancy-quantile-anchor`, `--occupancy-anchor`, `--occupancy-margin-hinge` — because
they are moot on a read with no additive mode and stacking them would confound the measurement; it refuses
`--occupancy-normalise` (the read cannot be both de-meaned and z-scored, and the model config refuses the pair
independently); and it refuses every lever that changes what is read, where it is sampled or where zero sits:
`--occupancy-offset-rows`, `--occupancy-deform`, `--occupancy-detail-stencil`, `--occupancy-resolution`,
`--occupancy-context`, `--occupancy-column`, `--occupancy-placed`, `--occupancy-lattice-free`. `0` remains the
published objective.

## Expected reading, declared before the numbers

**Expected: `centre-clears-the-boundary`.** The falsifiable form: the dense arm's *shape* was worth `0.465582` at
`3.15`; if the only missing ingredient is that the read sat high, then removing the level should make plain F1 at `0.0`
approach that `0.465582`, well past the declared bar of control's `0.381640 + 0.05`.

**The declared failure mode.** The mean of *all* cells is not the F1-optimal level on a population that is roughly
`12%` occupied: with far more empty than occupied cells, the board mean is dragged down by empty cells, so centring can
push the read too far the other way. If the swept optimum lands **below** zero and the primary does not clear, the
verdict is **`centre-fails`** and the record says so plainly — an overshoot is not an operating point either. The
standing bars apply unchanged: primary `+0.05` over control, AUC tolerance `-0.02`.

## Arms — one catalogue, one block

Family **`m14-centre-`** (distinct from `m9-margin-`, `m10-anchor-`, `m11-quantile-anchor-`, `m12-dense-hinge-`,
`m13-midpoint-`), registered in the run gate before launch.

| run | occupancy read | config |
|---|---|---|
| `m14-centre-control-r1` | published | `occupancy_centre = 0` |
| `m14-centre-centred-r1` | centred in the head | `occupancy_centre = 1` |

Same envelope and rules as M7–M13: same seed handling (`20261026`), same discriminator and scoring path, same 64-step
study, same 54-forward sweep over the standing 81-point grid, same 12 leave-one-group-out groups, one new catalogue.
The config differs in exactly one field, and the curator asserts that.

## Readings

Primary: plain-binary F1 at the standing `0.0` boundary, decided straight from the read the head emits, no threshold
selection and no post-processing — the same reading names and threshold style as M8.

| delta against control | reading |
|---|---|
| `>= 0.05` | `centre-clears-the-boundary` |
| `> 0` and `< 0.05` | `centre-moves-without-clearing` |
| `<= 0` | `centre-fails` |

Secondary: deciding AUC with the standing `-0.02` tolerance. Control must reproduce `0.8226433610683003` and
`0.381640` **exactly** or the comparison does not count. Reference rows printed beside the table, each in its own
catalogue and none re-verdicted: M13 midpoint (`0.830500` / `0.310966`), M12 dense hinge (`0.862838` / `0.354770`),
M11 quantile anchor (`0.815340` / `0.264308`), M10 extremal anchor (`0.885462` / `0.310966`), M9 hinge
(`0.864667` / `0.251701`).

## The aliasing trap, and how this block is proven to be its own

A structural option changes the read globally, which makes a mis-wired arm able to reproduce a neighbour's aggregate
almost exactly. M13 already produced an arm whose `0.0` counts, F1 and row diagnostic were identical to M10's while its
AUC was not, and that had to be explained. This arm therefore carries **curator-level assertions** rather than an
eyeball:

1. each sweep's `weights_sha256` equals its own study's `checkpoint_sha256` (already enforced by every curator here);
2. the two arms' sweep weights differ, so neither can be a copy of the other;
3. the treated arm's config differs from control's in exactly one field, `occupancy_centre`, and in no other;
4. the treated arm's `occupancy_centre` is `1` in the config, in the arguments and in the recovery contract;
5. if any aggregate coincides with a reference row to six decimals, the scope records it explicitly as a coincidence
   with the assertion that produced it, never as silent agreement.

## Fit gate

The count of boards the `0.0` decision reproduces exactly on the treated arm's own sweep logits; `28/28` is **TRUE**.
It has been `1` of `28` since M12, and moving it is the second thing this arm could demonstrate.

## If this also fails

Declared as such, without tuning anything (there is nothing to tune), without a sweep selection and without re-running
for different numbers. What it would rule out: if removing the additive mode does not place the boundary, then no
placement of `0.0` on this read can be bought by parameterisation either, and the remaining explanation is that the
read's *cell-level* errors — not its level and not its ranking — bound the operating point, which would send the next
arm to the read's inputs (the sampling geometry it reads through) rather than to its scale or its supervision.

## Realisation and Measured

**Realisation.** `centre_board` in `src/tetris_vision/model.py` (with `occupancy_centre` as a validated config field,
refused beside `occupancy_normalise`), applied in the head immediately before the stage dict is emitted and before any
loss or scorer; `--occupancy-centre 0|1` in `scripts/train_m2_stream.py` with the declared refusals;
`require_occupancy_centre` in the admission join (config, arguments and recovery contract); the value in the config, the
recovery contract and the summary; family `m14-centre-` registered in the run gate before launch;
`tests/test_read_centring.py` covering alignment invariance, the per-board statistic, spawn-row exclusion, the
zero-variance and single-cell cases, both-direction refusals through the real parser, and the record bindings.
`0` is the published function. The block needed one repair before it could run (V-135) and then passed every stage.

Receipts: `docs/evidence/m14-read-centring-r1/scope.json` (curation at its declared `1,900 MiB`),
`runs/m14-centre-<arm>-r1/` studies (64 steps, `1,792 MiB`, `329.0 s` and `327.2 s`),
`runs/m14-centre-<arm>-r1-sweep-r1/` sweeps (54 forwards, `640 MiB`). One new catalogue, `c9544334`.

**Reproduction first.** Control came back bit-identical: AUC `0.8226433610683003` and plain F1 at `0.0` `0.381640`,
exact string equality on both, so the comparison counts.

| arm | AUC | plain-binary F1 at `0.0` | best swept F1 | precision / recall at `0.0` | cells positive at `0.0` | exact boards at `0.0` | out-of-fold F1 | row error | occupancy loss |
|---|---|---|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 0.2363 / 0.9910 | 2,797 (49.95%) | 0 of 28 | 0.400399 | +7.819 | 0.7684 |
| centred in the head | 0.815395 | **0.374142** | 0.393230 @ `1.95` | 0.2312 / 0.9805 | 2,829 (50.52%) | 0 of 28 | 0.378651 | +8.007 | 0.6166 |
| *M13 midpoint anchor (reference)* | *0.830500* | *0.310966* | *0.401408* @ *`3.00`* | — | *3,610 (64.5%)* | *1 of 28* | *0.348472* | *+9.779* | — |
| *M12 dense hinge (reference)* | *0.862838* | *0.354770* | *0.465582* @ *`3.15`* | — | *3,065 (54.7%)* | *1 of 28* | *0.465582* | *+8.544* | — |
| *M11 quantile anchor (reference)* | *0.815340* | *0.264308* | *0.404026* | — | *4,365 (78.0%)* | — | *0.379323* | *+10.020* | — |
| *M10 extremal anchor (reference)* | *0.885462* | *0.310966* | *0.540811* | — | *3,610 (64.5%)* | — | *0.540811* | *+9.779* | — |
| *M9 hinge (reference)* | *0.864667* | *0.251701* | *0.474210* | — | — | — | *0.429457* | *+10.691* | — |

**The block is proven to be its own, by assertion rather than by eyeball.** The two arms' configs differ in exactly one
field (`occupancy_centre`); each sweep's weights equal its own study's checkpoint (`bf1ca813...`, `005b3dcf...`); the two
weights differ; and the previous block's checkpoint is excluded by name. Unlike M13 nothing here coincides suspiciously
with a reference row: control's numbers are the published ones by construction and every treated aggregate differs.

### Reading: `centre-fails`, secondary `ranking-preserved`

Primary delta **`-0.007498`** (bar `+0.05`) - the declared reading is **`centre-fails`**. Secondary AUC delta
**`-0.007248`**, inside the tolerance - `ranking-preserved`. Best swept F1 **`-0.018804`** at `1.95`
(`+1.05` against control's `0.90`), out-of-fold **`-0.021748`**, cells positive **`+0.57` points**, row error
**`+0.188`**, occupancy loss **`-0.152`** (the centred read fits its labels better). Fit gate **FALSE** at `0` of `28`.
The overshoot flag is **false**: the treated optimum did not move below zero.

**Verdict: removing the additive mode does not place the boundary either - and it is the closest any arm has come.**
The primary delta is `0.0075`, seven times smaller than the next-best supervised arm (`M12`'s `-0.027`) and an order
below M11's `-0.117`. The level drift collapsed: the positive share went from control's `49.95%` to `50.52%`, `+0.57`
points where the previous best arm moved it `+14.5`. That is what a *pinned* level looks like - a zero-mean read puts
half its cells above zero - and it is the structural proof that the transform does exactly what it says.

**But the pinned level is the wrong level, and for a quantitative reason the declaration named in advance.** A board's
mean over *all* its visible cells is not the F1-optimal boundary on a `12%`-occupied population: with the read centred
at its own mean, the swept optimum lands `1.95` logits above `0.0`, so `0.0` still cuts through the middle of the
population instead of above the empty mode. The declared mechanism was right and its direction was not - this was not
an overshoot below zero, it was a pin at the mean.

### What the sixth negation rules out

Together with M9-M13 this now rules out the two halves of the hypothesis at once. Adding supervision cannot place the
boundary (five terms, five negations, and the best of them gained on shape, not placement); *removing the level* cannot
place it either, because the only label-free level available to the read is its own mean, and the mean is not the
operating point when the population is `12%` occupied. What remains is not a scale, not a level and not a supervision
shape: it is where the read's own *structure* puts the crossing.

### The next structural arm, declared now

**Per-column stack-contiguity of the read** (`--occupancy-stack-monotone 1`): parameterise the occupancy read as a
per-column cumulative maximum from the bottom of the board, so each column can cross `0.0` at most once and the crossing
*is* the column's height. Nothing about prevalence is read: monotonicity is the domain's own constraint (a locked stack
rests on the floor), so the crossing count is implied by the geometry rather than taught by labels, and no threshold,
margin, weight or parameter is introduced. Declared failure mode with it: real boards contain holes and overhangs, so a
strictly monotone read cannot express them and may trade per-cell accuracy for placement - its primary bar (`+0.05`) and
the `-0.02` AUC tolerance are the standing numbers, and if it cannot clear them while moving the exact-board count the
honest conclusion is that placement and cell accuracy are in genuine tension on this read. Its own document must restate
the mechanism, the constants (there are none to tune) and the failure mode before its own receipt.

### Limits

Twenty-eight boards, twelve images, one catalogue, one block, one seed, one launch. Two arms, one control, one
catalogue; the connection to real screenshots, complete states or quality remains outside this record.
