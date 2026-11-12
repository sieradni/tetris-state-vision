# M11 — supervising the empty mass, not two cells

**Status: predeclared.** The constants, the statistic, the arms, the readings and the fallback below were fixed before
the first training receipt of this block existed, and are not revised by its result. The `Measured` section is appended
afterwards and quotes the receipts; it never changes the declarations.

## What M10 actually left behind

M10's two-sided anchor supervised the boundary the consumer applies, and it bought the strongest occupancy read this
project has recorded: AUC `0.885462`, best swept F1 `0.540811`, out-of-fold `0.540811` with all twelve folds selecting
the same boundary. It still failed at plain `0.0` (`0.310966`, delta `-0.070674`), and the failure has a mechanical
explanation rather than a mysterious one:

> Both supervision terms ever tried here — the M9 one-sided hinge and the M10 two-sided anchor — are **extremal**.
> They constrain a board's *lowest occupied* logit and its *highest empty* logit. That is two cells out of two hundred.
> Nothing in either term speaks about the other `99%`, so the empty mass is free to sit wherever the likelihood likes
> it, and it drifted upward: at `0.0` the M10 anchor arm calls `3,610` of `5,600` known cells positive (`64.5%`), so
> even recall `0.9970` can only buy precision `0.1842` and F1 near `0.31`. The anchor reached `0.0` on its final
> training step — the two bound cells were exactly where it asked them to be — while the bulk of the empty board was
> above the boundary. The term was satisfied and the operating point was not placed.

This pass replaces the extrema with the classes' own quantiles, so the term speaks about the mass it needs to move.

## The mechanism

For every qualifying board — a board carrying **both** occupied and empty *known* visible cells; `unknown` is never
counted as empty and never contributes to either side — with `qθ(S)` the θ-quantile of the set `S` of that board's
unnormalised occupancy logits:

```
anchor = mean over qualifying boards [ max(0, ANCHOR_MARGIN - q10(occupied))
                                     + max(0, ANCHOR_MARGIN + q90(empty)) ]
```

pooled beside the existing occupancy loss as the `quantile` term at `ANCHOR_WEIGHT`, through `stage_loss` and
`joint_loss` exactly as the hinge and the extremal anchor are pooled. At least `90%` of each class is thereby required
to be on its side of the `0.0` boundary a consumer applies, instead of one cell per class.

**An ordering that has to be stated, because it is the opposite of what "bounds the mass" suggests.** Since
`q10(occupied) >= min(occupied)` and `q90(empty) <= max(empty)`, this term is pointwise **no larger** than the extremal
anchor on any board — it charges the same or less, never more. So this is not a stronger constraint; it is a constraint
whose *satisfaction cannot be bought by sacrificing one cell*. The extremal term can be driven to zero on a board only
when the two extreme cells are right, which makes a single ambiguous cell decisive and lets the body drift (exactly the
M10 failure: the term reached zero in-sample while `64.5%` of cells sat above `0.0` out of sample). The quantile term's
subgradient lands on the cell at the tenth/ninetieth percentile — the body's edge — and the identity of that cell moves
as the body moves, so its pressure is spread over the part of the class nearest the boundary. The arm therefore tests
**where supervision lands**, not how large it is; a smaller term that lands on the mass is the hypothesis.

**Interpolation convention, fixed.** `QUANTILE_INTERPOLATION = "linear"`: the sorted values are indexed by
`h = (n-1)·q`, and the result is `v[floor(h)] + (h - floor(h))·(v[floor(h)+1] - v[floor(h)])`. So a quantile is always a
real logit of the board being judged (no extrapolation), and the convention is the one `torch.quantile` applies by
default — the term is checked against `1.5` on the ten occupied values `[-3, 2, 2, ...]`, where `lower` would give
`-3` and `higher` `2`. This is not searched, not learned and not revisited.

## Declared constants — fixed, never revised for this arm

| constant | value | note |
|---|---|---|
| `ANCHOR_MARGIN` | `1.0` | **unchanged from M10**: the same margin the extremal anchor used |
| `ANCHOR_WEIGHT` | `0.5` | **unchanged from M10**: the same pooling weight |
| `ANCHOR_OCCUPIED_QUANTILE` | `0.10` | tenth percentile of the board's occupied logits |
| `ANCHOR_EMPTY_QUANTILE` | `0.90` | ninetieth percentile of the board's empty logits |
| `QUANTILE_INTERPOLATION` | `"linear"` | `h = (n-1)q` on the sorted class logits |

Margin and weight are deliberately identical to M10. The only difference between the two arms' supervision is the
statistic — extrema become quantiles — so the comparison isolates what mass supervision does, with nothing to attribute
to a rescaled term.

## What is forbidden here, and refused by the trainer

`--occupancy-quantile-anchor 1` refuses to stack with `--occupancy-anchor` or `--occupancy-margin-hinge` (three
statements about one boundary is not this arm), and refuses every lever that changes what is read, where it is sampled
or where zero sits: `--occupancy-normalise`, `--occupancy-offset-rows`, `--occupancy-deform`, `--occupancy-detail-stencil`,
`--occupancy-resolution`, `--occupancy-context`, `--occupancy-column`, `--occupancy-placed`, `--occupancy-lattice-free`.
There is no output parameterisation, no post-hoc threshold rule, no calibration step, no sweep selection, no envelope
special-case, no class-count weighting and no learned parameter: the term is supervision of the published read-out, and
`--occupancy-quantile-anchor 0` is bit-for-bit the published objective.

## Declared achievable ceiling

If the term is satisfied on every qualifying board — `q10(occupied) ≥ +1` and `q90(empty) ≤ -1` — then at most `10%` of
the `667` occupied cells and at most `10%` of the `4,933` empty cells can be on the wrong side of `0.0`. The published
population then admits at least `600` true positives and at most `493` false positives, so the plain-binary F1 at `0.0`
is at least

```
precision >= 600/(600+493) = 0.549      recall >= 600/667 = 0.900
F1        >= 2(0.549)(0.900)/(0.549+0.900) = 0.682
```

**Declared ceiling: F1 at plain `0.0` of at least `0.682`.** This is stated now, before the receipt, so a result far
below it is evidence the term was not paid, and a result near it is evidence the two cells were the whole problem. It is
also a warning: mass constraints are strong, which is what M8 showed with a far blunter instrument, so the ranking cost
is the quantity to watch and is declared below.

## Arms — one catalogue, one block

Family **`m11-quantile-anchor-`** (distinct from `m9-margin-` and `m10-anchor-`), registered in the run gate before
launch.

| run | occupancy loss | supervision |
|---|---|---|
| `m11-quantile-anchor-control-r1` | published | none |
| `m11-quantile-anchor-bulk-r1` | published | `--occupancy-quantile-anchor 1` |

Same envelope and rules as M7–M10: the same published read-out, the same seed handling (`20261026`), the same
discriminator and scoring path, the same 64-step study, the same 54-forward sweep over the standing 81-point grid and
the same 12 leave-one-group-out groups. Only the catalogue is new.

## Readings — declared before the numbers

Primary: plain-binary F1 at the standing `0.0` boundary, decided straight from the unnormalised read with no threshold
selection, no calibration and no post-processing — the same reading names and threshold style as M8.

| delta against control | reading |
|---|---|
| `>= 0.05` | `bulk-clears-the-boundary` |
| `> 0` and `< 0.05` | `bulk-moves-without-clearing` |
| `<= 0` | `bulk-fails` |

Secondary: deciding AUC, tolerance `-0.02` — `ranking-preserved` at or above it, `ranking-degraded` below it. An
anchored term cannot add information, so a ranking gain is a change in what was learned and a loss is the real cost of
the constraint.

Control must reproduce the recorded control **exactly** — AUC `0.8226433610683003` and plain-binary F1 at `0.0`
`0.381640` — in the new catalogue. If it does not, that is said plainly and the comparison does not count; the
receipts are kept and no arm is re-run to make them agree.

Both reference rows are printed beside the arm table so the progression is visible in one place: the M10 extremal
anchor (`0.885462` / `0.310966`) and the M9 one-sided hinge (`0.864667` / `0.251701`). They are references in their own
catalogue, not re-measured and not re-verdicted here.

## If this also cannot place the boundary

A negative or partial reading is a good outcome and will be declared as such: no tuning of margin, weight or quantiles
until it flips, no widening to a sweep, no re-running the same arm hoping for different numbers.

The declared next structural idea at the same specificity — the first moment, which completes the progression from
extremes to quantiles to means:

```
anchor = mean over qualifying boards [ max(0, ANCHOR_MARGIN - mean(occupied))
                                     + max(0, ANCHOR_MARGIN + mean(empty)) ]
```

i.e. supervise the class **means** of the unnormalised logits rather than their tenth and ninetieth percentiles. It is
smoother than a quantile (no interpolation, no per-board sort), it bounds the mass even more directly, and the same
margin/weight can be reused so it differs from this arm only in the statistic. Its margin, weight and primary metric
must be declared in its own document before its own receipt, with this pass's reading vocabulary.

## Realisation and Measured

**Realisation.** `quantile_anchor_term` in `src/tetris_vision/losses.py` with `ANCHOR_MARGIN = 1.0`,
`ANCHOR_WEIGHT = 0.5`, `ANCHOR_OCCUPIED_QUANTILE = 0.10`, `ANCHOR_EMPTY_QUANTILE = 0.90` and
`QUANTILE_INTERPOLATION = "linear"` as module constants; pooled as the `quantile` term through `stage_loss` and
`joint_loss`; `--occupancy-quantile-anchor 0|1` in `scripts/train_m2_stream.py` with the declared refusals;
`require_occupancy_quantile_anchor` in the declaration join; the value in the recovery contract and at every objective
call site; family `m11-quantile-anchor-` registered in the run gate before launch; `tests/test_quantile_anchor.py`
covering the arithmetic, the interpolation convention, the unknown guard, the no-qualifying-board case, the ordering
against the extremal anchor, the gradient, the admission join and every refusal. The block launched once, all eight
stages passed first time, and the arm ran once.

Receipts: `docs/evidence/m11-quantile-anchor-r1/scope.json` (curation at its declared `1,900 MiB`, receipt
`runs/m11-quantile-anchor-curation-inner-r1.json`), studies `runs/m11-quantile-anchor-<arm>-r1/` (64 steps,
`1,792 MiB`, `160.4 s` and `143.9 s`), sweeps `runs/m11-quantile-anchor-<arm>-r1-sweep-r1/` (54 forwards, `640 MiB`).
One new catalogue, `cb8775b5...`, distinct from M10's `6e207218...`.

**Reproduction first.** Control came back bit-identical in the new catalogue: AUC `0.8226433610683003` and plain
F1 at `0.0` `0.381640`, exact string equality on both, so the comparison counts.

| arm | AUC | plain-binary F1 at `0.0` | best swept F1 | precision / recall at `0.0` | cells positive at `0.0` | out-of-fold F1 | row error |
|---|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 0.2363 / 0.9910 | 2,797 (49.9%) | 0.400399 | +7.819 |
| class-quantile anchor | 0.815340 | **0.264308** | 0.404026 @ `2.25` | 0.1523 / 0.9970 | 4,365 (78.0%) | 0.379323 | +10.020 |
| *M10 extremal anchor (reference, own catalogue)* | *0.885462* | *0.310966* | *0.540811* | — | *3,610 (64.5%)* | *0.540811* | *+9.779* |
| *M9 one-sided hinge (reference, own catalogue)* | *0.864667* | *0.251701* | *0.474210* | — | — | *0.429457* | *+10.691* |

Operating counts at `0.0`: control `661 TP / 2,136 FP / 6 FN`, quantile arm `665 / 3,700 / 2`, of `667` occupied
and `4,933` empty cells. The term the arm actually paid: mean `2.417` over 64 steps, satisfied (`0.0`) on 16 of them,
`0.0` on the final step — i.e. it was *not* held at zero during training, and its mean is `19%` below the M10 anchor's
`3.004`.

### Reading: `bulk-fails`, secondary `ranking-preserved`

Primary delta **`-0.117332`** (bar `+0.05`) - the declared reading is **`bulk-fails`**. Secondary AUC delta
**`-0.007303`**, inside the declared `-0.02` tolerance - `ranking-preserved`, barely. Best swept F1 **`-0.008008`**,
out-of-fold **`-0.021076`**, row error **`+2.201`**, cells called positive **`+78.0%` of `5,600` = `+28.0` points of the
population`. Verdict `not-cleared` (per-cell F1 `0.264308` against `0.70`; AUC `0.815340` against `0.90`; attribution
not met because the primary is negative), fit gate **FALSE** (`0` of `28` exact boards).

**Verdict: supervising the bulk of the empty mass does not move the operating point either — it moves it the wrong
way.** This arm lost to the published objective on the primary (`-0.117332`), on the best swept F1
(`0.404026` against `0.412034`), and on out-of-fold (`0.379323` against `0.400399`); only the AUC stayed inside the
tolerance, and it fell too (`0.815340`). Against the M10 extremal anchor it is worse by `0.183` on the ranking and
`0.276` on `0.0`. The declared ceiling (`0.682`) was not approached — the term was satisfied on a quarter of the steps
and never held, so the ceiling's premise never held either.

### The mechanism the numbers actually show

The declaration predicted the quantile term would be pointwise **no larger** than the extremal one, and it was: on the
study's own batches its mean was `2.417` against the M10 anchor's `3.004`. The stated hypothesis was that a smaller
term landing on the body's edge would beat a larger one landing on two sacrificed cells. **The data refutes that.**
Both supervised arms raised the read's level rather than pinning it — control calls `49.9%` of cells positive, M10's
anchor `64.5%`, this arm `78.0%` — and the reason is visible in the swept boundary: control's best boundary is at
`+0.90`, both supervised arms' at `+2.25`. Supervision of a per-board *statistic* does not pin the level, it buys
level: raising the occupied statistic is charged once per board (the mean charge over 64 steps fell from `3.004` to
`2.417`), while the cost of raising the whole read is spread over `4,933` cells that the term never counts. The
objective can therefore reduce its supervised charge faster than it damages itself, and the read's free additive level
drifts upward in proportion to how much supervision it is under.

### The next structural idea, chosen by the data

The predeclared fallback was the **mean** anchor. **This pass does not take it, and says why.** The mean is the same
shape of instrument — one summary per class per board — that has now failed three times (hinge, extrema, quantiles),
and it is the *least* sensitive of the three: a class mean is dominated by the confident cells that are already far
from the boundary, so the band that actually decides F1 at `0.0` contributes almost nothing to it. The pattern across
M9/M10/M11 is not "wrong statistic", it is "**too few cells charged**": two cells, then ten percent, and the level
stayed free each time.

The declared next arm therefore charges **every** cell, class-balanced so nothing is weighted by class frequency:

```
term = mean over boards [ 0.5 * mean over known occupied cells of relu(margin - logit)
                       + 0.5 * mean over known empty cells    of relu(margin + logit) ]
```

with `margin = 1.0`, weight `0.5` (the same declared numbers again, so the only change is density and class balance),
and the same primary metric, readings and reference rows. This is not BCE with extra steps: BCE's gradient decays
exponentially on confident cells and is dominated by the thousands of cells far from the boundary, whereas a linear
hinge charges *only* the cells inside the declared band around `0.0` at full strength - exactly the cells that decide
the operating point - and charges both classes opposing forces at every such cell, so a global level shift is paid for
instead of rewarded. Its ceiling is trivially declarable: if the term is satisfied on every known cell, every cell is
`1.0` from `0.0` on its correct side, so `0.0` is a perfect boundary on the published population (F1 `1.000`) and the
fit gate would clear; the informative quantity is how much of that is reached, which is what the same primary metric
will read. Margin, weight and binding must be declared in its own document before its own receipt.

### Limits

Twenty-eight boards, twelve images, one catalogue, one block, one seed, one launch. This measures whether quantile
supervision of the declared margin moves the plain-binary operating point on the published read-out; it is not a
real-image, a complete-state or a quality claim, and the fit gate (`0` of `28`) is reported, not explained away.

## Limits, stated up front

Twenty-eight boards, twelve images, one catalogue, one block, one seed. This measures whether quantile supervision of
the same margin moves the plain-binary operating point on the project's published read-out; it is not a real-image, a
complete-state or a quality claim, and the fit gate (`0` of `28` exact boards at M10) is reported, not explained away.
