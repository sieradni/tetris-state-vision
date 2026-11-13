# M12 — charging every cell, class-balanced

**Status: predeclared.** Every constant, the mechanism, the arms, the readings and the fallback below were fixed before
the first training receipt of this block existed and are not revised by its result. The `Measured` section is appended
afterwards, quotes the receipts, and never changes the declarations.

## What three passes established

| pass | supervision | cells charged per board | F1 at `0.0` | AUC | cells positive at `0.0` | swept optimum |
|---|---|---|---|---|---|---|
| M9 | one-sided hinge | 2 | 0.251701 | 0.864667 | — | `2.70` |
| M10 | two-sided extremal anchor | 2 | 0.310966 | 0.885462 | 64.5% | `2.25` |
| M11 | class-quantile anchor | 10% of each class | 0.264308 | 0.815340 | 78.0% | `2.25` |
| control | none | 0 | **0.381640** | 0.822643 | 49.9% | `+0.90` |

Every supervised arm raised the read instead of pinning it, and the swept optimum moved with it. The diagnosis is not
"wrong summary statistic" — it is that a per-board *summary* charges one number per board, so raising the occupied
summary is paid once while raising the whole read is spread across thousands of cells the summary never counts. The
read's **free additive level** absorbs the pressure, and the more supervision it is under, the higher it drifts. M11
declared this and named the fix: charge density instead of a statistic.

## The mechanism

For every qualifying board — carrying **both** occupied and empty *known* visible cells; `unknown` is never counted as
empty and never charged — with `L` the unnormalised occupancy logits of that board's known visible cells:

```
per board:  0.5 * mean_{occupied} relu(ANCHOR_MARGIN - L)   +   0.5 * mean_{empty} relu(ANCHOR_MARGIN + L)
term      :  mean over qualifying boards of the above, pooled as `dense` at ANCHOR_WEIGHT beside the occupancy loss
```

Every known cell is charged, both classes at once, so a global level shift is credited on the occupied side and paid on
the empty side in the same step. Only cells inside the declared band carry gradient, which is exactly the band that
decides a `0.0` boundary — unlike BCE, whose gradient decays exponentially on confident cells and is dominated by the
thousands of cells far from the boundary.

**The class balance is equal class weight, not a cell count.** `0.5/0.5` is fixed here so the `4,933` empty cells cannot
outvote the `667` occupied ones; it reads nothing about how many cells carry each label, so no prevalence is taught and
nothing is weighted by frequency. A board's means are divided by that board's own class counts — arithmetic on the
board's own cells, not a population statistic.

## Declared constants — fixed, never revised for this arm

| constant | value | note |
|---|---|---|
| `ANCHOR_MARGIN` | `1.0` | unchanged from M9/M10/M11 |
| `ANCHOR_WEIGHT` | `0.5` | unchanged from M9/M10/M11 |
| `DENSE_OCCUPIED_BALANCE` | `0.5` | equal class weight on the occupied mean |
| `DENSE_EMPTY_BALANCE` | `0.5` | equal class weight on the empty mean |

The only change from M11 is density and balance: the same margin, the same pooling weight, over every cell instead of a
tenth of each class.

## What is refused

`--occupancy-dense-hinge 1` refuses to stack with `--occupancy-quantile-anchor`, `--occupancy-anchor` or
`--occupancy-margin-hinge` (four statements about one boundary is not this arm), and refuses every lever that changes
what is read, where it is sampled or where zero sits: `--occupancy-normalise`, `--occupancy-offset-rows`,
`--occupancy-deform`, `--occupancy-detail-stencil`, `--occupancy-resolution`, `--occupancy-context`,
`--occupancy-column`, `--occupancy-placed`, `--occupancy-lattice-free`. No output parameterisation, no post-hoc
threshold rule, no calibration, no sweep selection, no envelope special-case, no prevalence weighting. `0` is
bit-for-bit the published objective.

## The fit gate, and why satisfaction implies a `0.0` boundary

**Statement.** If the term is satisfied — per-board charge `0.0` — then every known occupied cell has `logit >= 1` and
every known empty cell has `logit <= -1`. Both are a full margin away from `0.0` on their correct side, so deciding at
`0.0` reproduces the label on **every** board. Satisfaction therefore implies an exact `0.0` boundary on the published
population, and the fit gate is the count of boards the `0.0` decision reproduces exactly: `28/28` is **TRUE**, anything
less is **FALSE**. This is the arm's fit gate and is reported as such, computed from the treated arm's own sweep logits
at the standing `0.0` boundary — not from the swept optimum.

## Expected reading, declared before the numbers

**Expected: `dense-clears-the-boundary`**, and the reason is mechanical rather than hopeful: the charge falls on the
band, in both directions, at every cell, so the level cannot drift upward without immediately paying a larger empty-side
charge in the same step — the exact failure mode that made M9/M10/M11 raise the read.

**Declared risk, equally explicit:** this is the strongest constraint tried so far and the M8 precedent (a per-board
normalisation that bought `0.0` and cost the ranking) says a strong constraint can buy the operating point at the price
of separation. If AUC falls below control by more than `0.02` the honest reading is `ranking-degraded` and the trade is
reported as a trade. And if the paid charge stays high — the term never near zero on its own batches — then the read is
genuinely ambiguous at the band and no supervision of the read can place the boundary; the next arm is then the
fallback declared below.

## Arms — one catalogue, one block

Family **`m12-dense-hinge-`** (distinct from `m9-margin-`, `m10-anchor-` and `m11-quantile-anchor-`), registered in the
run gate before launch.

| run | occupancy loss | supervision |
|---|---|---|
| `m12-dense-hinge-control-r1` | published | none |
| `m12-dense-hinge-banded-r1` | published | `--occupancy-dense-hinge 1` |

Same envelope and rules as M7–M11: same published read-out, same seed handling (`20261026`), same discriminator and
scoring path, same 64-step study, same 54-forward sweep over the standing 81-point grid, same 12 leave-one-group-out
groups, one new catalogue.

## Readings

Primary: plain-binary F1 at the standing `0.0` boundary, decided straight from the unnormalised read, no threshold
selection, no calibration, no post-processing — the same reading names and threshold style as M8.

| delta against control | reading |
|---|---|
| `>= 0.05` | `dense-clears-the-boundary` |
| `> 0` and `< 0.05` | `dense-moves-without-clearing` |
| `<= 0` | `dense-fails` |

Secondary: deciding AUC with M8/M9/M10/M11's `-0.02` tolerance — `ranking-preserved` or `ranking-degraded`.

Control must reproduce the recorded control **exactly** — AUC `0.8226433610683003`, plain F1 at `0.0` `0.381640` — in
the new catalogue, or the comparison does not count and no arm is re-run to make it agree. The M10 extremal anchor
(`0.885462` / `0.310966`) and the M9 one-sided hinge (`0.864667` / `0.251701`) are printed beside the table as
reference rows in their own catalogues, neither re-measured nor re-verdicted. The M11 quantile arm
(`0.815340` / `0.264308`) is carried as the immediate predecessor.

## If this also fails: attack the level, not the shape

A negative or partial reading is declared as such: no tuning of margin, weight or class balance until it flips, no
widening to a sweep, no re-run for different numbers.

The declared next structural idea removes the read's additive degree of freedom **directly** instead of adding another
supervision shape. For every qualifying board, with `m_occ` and `m_empty` the means of its known occupied and empty
logits:

```
term = mean over qualifying boards [ |(m_occupied + m_empty) / 2| ]        # midpoint anchored at exactly 0.0
```

pooled at `ANCHOR_WEIGHT = 0.5`, no margin, primary metric and readings unchanged. Why this and not more of the above: a
global shift `+delta` moves *both* means by `delta`, so this term grows linearly and unconditionally on any drift, and it
cannot be satisfied by raising the occupied class — the reason every previous term could be. It is a location instrument,
not a separation instrument: the published BCE already supplies separation, and collapse is the case it punishes. Equal
class weight, known cells only, no parameters, no threshold. Its margin (none), weight (`0.5`) and metric must be
declared in its own document before its own receipt.

## Realisation and Measured

**Realisation.** `dense_hinge_term` in `src/tetris_vision/losses.py` with `ANCHOR_MARGIN = 1.0`, `ANCHOR_WEIGHT = 0.5`,
`DENSE_OCCUPIED_BALANCE = 0.5`, `DENSE_EMPTY_BALANCE = 0.5` as module constants; pooled as the `dense` term through
`stage_loss`/`joint_loss`; `--occupancy-dense-hinge 0|1` with the declared refusals; `require_occupancy_dense_hinge` in
the admission join; the value in the recovery contract and at every objective call site; family `m12-dense-hinge-`
registered in the run gate before launch; `tests/test_dense_hinge.py` covering the arithmetic against an independent
reference, the class balance, the contrast with the quantile anchor, the unknown guard, the no-qualifying-board case,
the gradient reach, the admission join and every refusal. `0` is the published objective. The block launched once and
all eight stages passed first time.

Receipts: `docs/evidence/m12-dense-hinge-r1/scope.json` (curation at its declared `1,900 MiB`),
`runs/m12-dense-hinge-<arm>-r1/` studies (64 steps, `1,792 MiB`, `151.0 s` and `153.1 s`),
`runs/m12-dense-hinge-<arm>-r1-sweep-r1/` sweeps (54 forwards, `640 MiB`). One new catalogue, `24c76620...`.

**Reproduction first.** Control came back bit-identical in the new catalogue: AUC `0.8226433610683003` and plain F1
at `0.0` `0.381640`, exact string equality on both, so the comparison counts.

| arm | AUC | plain-binary F1 at `0.0` | best swept F1 | precision / recall at `0.0` | cells positive at `0.0` | exact boards at `0.0` | out-of-fold F1 | row error |
|---|---|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 0.2363 / 0.9910 | 2,797 (49.9%) | 0 of 28 | 0.400399 | +7.819 |
| dense class-balanced hinge | 0.862838 | **0.354770** | **0.465582** @ `3.15` | 0.2160 / 0.9925 | 3,065 (54.7%) | **1 of 28** | **0.465582** | **+8.544** |
| *M10 extremal anchor (reference)* | *0.885462* | *0.310966* | *0.540811* | — | *3,610 (64.5%)* | — | *0.540811* | *+9.779* |
| *M11 quantile anchor (reference)* | *0.815340* | *0.264308* | *0.404026* | — | *4,365 (78.0%)* | — | *0.379323* | *+10.020* |
| *M9 one-sided hinge (reference)* | *0.864667* | *0.251701* | *0.474210* | — | — | — | *0.429457* | *+10.691* |

Operating counts at `0.0`: control `661 TP / 2,136 FP / 6 FN`, dense arm `662 / 2,403 / 5`, of `667` occupied and
`4,933` empty cells. The charge the arm actually paid: mean **`0.670`** over 64 steps (against the extremal anchor's
`3.004` and the quantile arm's `2.417` - `3.6x` lower), satisfied on 16 of 64 steps, `0.0` on the final step.

### Reading: `dense-fails`, secondary `ranking-preserved`

Primary delta **`-0.026870`** (bar `+0.05`) - the declared reading is **`dense-fails`**. Secondary AUC delta
**`+0.040195`** - `ranking-preserved`, and in fact a **gain**. Best swept F1 **`+0.053548`**, out-of-fold
**`+0.065183`**, cells positive **`+4.79` points**, row error **`+0.725`** (better), fit gate **FALSE** at `1` of `28`
exact boards - the first non-zero exact count any arm has recorded at `0.0`. Verdict `not-cleared` (per-cell F1
`0.354770` against `0.70`; AUC `0.862838` against `0.90`; attribution not met because the primary is negative).

**Verdict: charging every cell does not place the `0.0` boundary, and it is the first supervision that fixed
everything except the placement.** It beat control on the swept best F1 (`0.465582` against `0.412034`) and on
out-of-fold (`0.465582` against `0.400399`) - the first arm to do either - improved the row diagnostic, cut the paid
charge `3.6x` and the level drift from `+28.0` to `+4.8` points of the population, and produced the first exactly-decided
board at `0.0`. What it did not do is put `0.0` where the decisions are: the swept optimum moved *up* again, to `3.15`,
and `54.7%` of cells still sit above zero. The declared expectation was `dense-clears-the-boundary`; the observed
reading is `dense-fails`, and the difference between them is exactly the additive level, because everything the dense
band controls improved.

### The level, now isolated

Across the four arms the ranking behaves as density predicts and the placement does not: the more of the read is
charged, the better the separation and the smaller the drift, while `0.0` stays unused. The two quantities now diverge
cleanly - `0.862838` AUC with a `3.15` optimum says the read separates well and sits about `3` logits high, and
nothing in a per-cell term can fix a translation it is indifferent to at the optimum (the band is wide and the model
can slide inside it). That is the free additive degree of freedom, and the next arm removes it rather than charging
around it.

### The next structural arm, declared now

**Midpoint anchoring.** For every qualifying board, with `m_occ` and `m_empty` the means of its known occupied and
empty logits:

```
term = mean over qualifying boards [ |(m_occupied + m_empty) / 2| ]
```

pooled at `ANCHOR_WEIGHT = 0.5`, **no margin** (the target location is exactly `0.0`), equal class weight, known cells
only, no parameters, no threshold, nothing read about prevalence. Why this instead of another shape: a global shift
`+delta` moves both means by `delta`, so this term grows linearly and unconditionally on drift, and - unlike every
previous term - it cannot be satisfied by raising the occupied class. It is a location instrument, not a separation
instrument: the published BCE already supplies separation and punishes collapse, which is the failure mode to watch,
so its declared reading keeps the same `-0.02` AUC tolerance and its own primary bar of `+0.05`. Its constants, metric
and readings must be declared in its own document before its own receipt.

### Limits

Twenty-eight boards, twelve images, one catalogue, one block, one seed, one launch. This measures whether dense
class-balanced supervision of the declared margin moves the plain-binary operating point on the published read-out; it
is not a real-image, complete-state or quality claim.

## Correction (V-139, 2026-11-12)

The exact-board figure quoted above was produced by a curator rule that dropped any column whose decision or label was
empty, so a board that filled genuinely empty columns could be scored exact. Recomputed strictly from the same retained
sweep (`runs/m12-dense-hinge-banded-r1-sweep-r1/result.json`), this arm has **0 of 28** exact non-empty boards, not the number printed above.
The scope file is left as published: it is immutable evidence of what the instrument said at the time.
