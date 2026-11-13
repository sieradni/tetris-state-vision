# M13 — putting the boundary where the read lives

**Status: predeclared.** The mechanism, the constants, the arms, the readings and the failure rule below were fixed
before the first training receipt of this block existed and are not revised by its result. The `Measured` section is
appended afterwards.

## Why a location term, and nothing else

Five arms have now been measured on the published read-out, and the swept boundary moved every time:

| pass | supervision | swept optimum | cells positive at `0.0` | AUC |
|---|---|---|---|---|
| control | none | `+0.90` | 49.9% | 0.822643 |
| M9 | one-sided hinge (2 cells) | `2.70` | — | 0.864667 |
| M10 | two-sided extremal anchor (2 cells) | `2.25` | 64.5% | 0.885462 |
| M11 | class-quantile anchor (10% of each class) | `2.25` | 78.0% | 0.815340 |
| M12 | dense class-balanced hinge (every cell) | `3.15` | 54.7% | 0.862838 |

M12 settled which lever moves the *shape*: charging every cell beat control on best swept F1 (`0.465582` against
`0.412034`) and out-of-fold (`0.465582` against `0.400399`), improved the row diagnostic and produced the first exact
board at `0.0`. It did not settle *placement*: the read separates well and sits about three logits high, so `0.0` —
the boundary every consumer applies — is not where the decisions are. Each earlier term could be paid off by a global
shift (a summary is charged once per board; a wide band is indifferent to sliding inside itself). This term is the
first whose only job is the location, and it cannot be paid off that way.

## The mechanism

For every qualifying board — carrying both occupied and empty *known* visible cells; `unknown` is never counted as
empty — with `m_occupied` and `m_empty` the means of that board's known visible occupancy logits:

```
term = mean over qualifying boards [ |0.5*m_occupied + 0.5*m_empty - MIDPOINT_TARGET| ]
```

pooled beside the occupancy loss as the `midpoint` term at `ANCHOR_WEIGHT`. A shift `+delta` of the whole read moves
both means by `delta`, so the charge grows linearly and unconditionally on drift: raising the occupied class can no
longer reduce it, which is what every previous term allowed. Equal class weight, known cells only, no parameters, no
threshold, nothing read about prevalence.

**No margin.** `MIDPOINT_TARGET` is the boundary itself (`0.0`). A nonzero target would be a *separation* claim, and
separation is the published objective's job; this term claims only that the boundary between the classes sits where
the consumer puts it.

## Declared constants — fixed, never revised for this arm

| constant | value | note |
|---|---|---|
| `ANCHOR_WEIGHT` | `0.5` | unchanged from M9–M12 |
| `MIDPOINT_TARGET` | `0.0` | the boundary a consumer applies; **no margin** |
| `MIDPOINT_OCCUPIED_BALANCE` | `0.5` | equal class weight on the occupied mean |
| `MIDPOINT_EMPTY_BALANCE` | `0.5` | equal class weight on the empty mean |

Balance is equal class weight, never a cell count: `2` occupied against `18` empty charges exactly what `10` against
`10` charges.

## What is refused

`--occupancy-midpoint 1` refuses to stack with `--occupancy-dense-hinge`, `--occupancy-quantile-anchor`,
`--occupancy-anchor` or `--occupancy-margin-hinge` (five statements about one boundary is not this arm), and refuses
every lever that changes what is read, where it is sampled or where zero sits: `--occupancy-normalise`,
`--occupancy-offset-rows`, `--occupancy-deform`, `--occupancy-detail-stencil`, `--occupancy-resolution`,
`--occupancy-context`, `--occupancy-column`, `--occupancy-placed`, `--occupancy-lattice-free`. No output
parameterisation, no post-hoc threshold rule, no calibration, no sweep selection, no envelope special-case, no
prevalence weighting, no parameter. `0` is bit-for-bit the published objective.

## Expected reading, declared before the numbers

**Expected: `midpoint-clears-the-boundary`.** The falsifiable form: the dense arm reached `0.465582` at `3.15` while
its read sat three logits high, so if placement is the only remaining obstacle, moving the midpoint to `0.0` should make
the plain-binary F1 at `0.0` approach that `0.465582` — well past the declared bar of control's `0.381640 + 0.05`.

**The failure mode to watch, declared with it.** This term carries no separation pressure of its own; it can be
satisfied by pulling the occupied mean *down* to meet the empty mean, i.e. by collapsing the two classes onto each
other. The published objective punishes that, and the AUC tolerance is the declared instrument for catching it:

> **Rule.** If the deciding AUC falls more than `0.02` below control's, the reading is **`midpoint-fails`** whatever
> the primary says, and the record says so plainly — a boundary placed by destroying the separation is not an
> operating point.

## Arms — one catalogue, one block

Family **`m13-midpoint-`** (distinct from `m9-margin-`, `m10-anchor-`, `m11-quantile-anchor-`, `m12-dense-hinge-`),
registered in the run gate before launch.

| run | occupancy loss | supervision |
|---|---|---|
| `m13-midpoint-control-r1` | published | none |
| `m13-midpoint-level-r1` | published | `--occupancy-midpoint 1` |

Same envelope and rules as M7–M12: same published read-out, same seed handling (`20261026`), same discriminator and
scoring path, same 64-step study, same 54-forward sweep over the standing 81-point grid, same 12 leave-one-group-out
groups, one new catalogue.

## Readings

Primary: plain-binary F1 at the standing `0.0` boundary, decided straight from the unnormalised read, no threshold
selection and no post-processing — the same reading names and threshold style as M8.

| delta against control | reading |
|---|---|
| `>= 0.05` | `midpoint-clears-the-boundary` |
| `> 0` and `< 0.05` | `midpoint-moves-without-clearing` |
| `<= 0` | `midpoint-fails` |

Secondary: deciding AUC with the standing `-0.02` tolerance (`ranking-preserved` / `ranking-degraded`), with the
override rule above.

Control must reproduce the recorded control **exactly** — AUC `0.8226433610683003`, plain F1 at `0.0` `0.381640` — or
the comparison does not count and nothing is re-run to make it agree. Reference rows printed beside the table, each in
its own catalogue and none re-verdicted: M12 dense hinge (`0.862838` / `0.354770`), M11 quantile anchor
(`0.815340` / `0.264308`), M10 extremal anchor (`0.885462` / `0.310966`), M9 one-sided hinge (`0.864667` / `0.251701`).

## Fit gate

The count of boards the `0.0` decision reproduces exactly on the treated arm's own sweep logits — `28/28` is **TRUE**.
M12 recorded `1` of `28`, which is the number this arm is expected to move if placement was the obstacle.

## If this also fails

Declared as such, without tuning the target, the weight or the balance, without widening to a sweep and without
re-running for different numbers.

The next structural idea removes the additive degree of freedom **by construction** instead of charging it: centre the
read inside the head, `logit - mean over the board's visible cell window`, applied to the occupancy logits before any
scorer sees them (`--occupancy-centre 1`). It needs no labels, no parameters and no threshold, it differs from M8's
`normalise` in that it removes only the additive mode and leaves the spread untouched, and it cannot be paid off by
anything the objective does, because the shift is gone by the time the loss is computed. Its failure mode must be
declared with it: the mean of *all* cells is not the F1-optimal level when the errors are asymmetric, so a hard centring
may overshoot in the other direction — which is exactly what its own predeclaration must state, with its primary bar and
AUC tolerance fixed before its own receipt.

## Realisation and Measured

**Realisation.** `midpoint_term` in `src/tetris_vision/losses.py` with `MIDPOINT_TARGET = 0.0`,
`MIDPOINT_OCCUPIED_BALANCE = 0.5`, `MIDPOINT_EMPTY_BALANCE = 0.5` (and the declared `ANCHOR_WEIGHT = 0.5`); pooled
as the `midpoint` term through `stage_loss`/`joint_loss`; `--occupancy-midpoint 0|1` refusing all four existing terms
and every read-moving lever; `require_occupancy_midpoint` in the admission join; the value in the recovery contract and
at every objective call site; family `m13-midpoint-` registered in the run gate before launch;
`tests/test_midpoint_anchor.py` covering the arithmetic against an independent reference, the equal-weight property,
the two-way contrast with the band, the unknown guard, the no-qualifying-board case, the joint-sign gradient, the
admission join and every refusal. `0` is the published objective. The block launched once; all eight stages passed
first time.

Receipts: `docs/evidence/m13-midpoint-r1/scope.json` (curation at its declared `1,900 MiB`),
`runs/m13-midpoint-<arm>-r1/` studies (64 steps, `1,792 MiB`, `344.4 s` and `339.8 s` - this box was busier than the
earlier passes), `runs/m13-midpoint-<arm>-r1-sweep-r1/` sweeps (54 forwards, `640 MiB`). One new catalogue, `19077be4`.

**Reproduction first.** Control came back bit-identical in the new catalogue: AUC `0.8226433610683003` and plain F1
at `0.0` `0.381640`, exact string equality on both, so the comparison counts.

| arm | AUC | plain-binary F1 at `0.0` | best swept F1 | precision / recall at `0.0` | cells positive at `0.0` | exact boards at `0.0` | out-of-fold F1 | row error |
|---|---|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 0.2363 / 0.9910 | 2,797 (49.9%) | 0 of 28 | 0.400399 | +7.819 |
| midpoint anchor | 0.830500 | **0.310966** | 0.401408 @ `3.00` | 0.1842 / 0.9970 | 3,610 (64.5%) | 1 of 28 | 0.348472 | +9.779 |
| *M12 dense hinge (reference)* | *0.862838* | *0.354770* | *0.465582* @ *`3.15`* | — | *3,065 (54.7%)* | *1 of 28* | *0.465582* | *+8.544* |
| *M10 extremal anchor (reference)* | *0.885462* | *0.310966* | *0.540811* | — | *3,610 (64.5%)* | — | *0.540811* | *+9.779* |
| *M11 quantile anchor (reference)* | *0.815340* | *0.264308* | *0.404026* | — | *4,365 (78.0%)* | — | *0.379323* | *+10.020* |
| *M9 one-sided hinge (reference)* | *0.864667* | *0.251701* | *0.474210* | — | — | — | *0.429457* | *+10.691* |

Operating counts at `0.0`: control `661 TP / 2,136 FP / 6 FN`, midpoint arm `665 / 2,945 / 2`, of `667` occupied and
`4,933` empty cells. The charge the arm actually paid: mean **`0.923`** over 64 steps (`25%` of steps at `0.0`, `0.0`
on the final step) - between the dense arm's `0.670` and the extremal anchor's `3.004`, and never held.

**A coincidence worth recording, because it looks like a copy error and is not.** This arm's operating counts at `0.0`
(`665 / 2,945 / 2`), its F1 at `0.0` (`0.310966`) and its row diagnostic (`9.779`) are *identical* to the M10 extremal
anchor's. They are different arms: different catalogue (`19077be4` against `6e207218`), different checkpoints
(`970a5609...` against `149e930f...`), different AUC (`0.830500` against `0.885462`) and a different swept optimum. The
`0.0` decision pattern coincides because both reads sit at a `64.5%` positive share with the same two errors on the
same boards; the rankings inside the classes differ. The curator refuses a sweep whose weights differ from its own
run's checkpoint, so this is two measurements, not one measurement twice.

### Reading: `midpoint-fails`, secondary `ranking-preserved`

Primary delta **`-0.070674`** (bar `+0.05`) - the declared reading is **`midpoint-fails`**. Secondary AUC delta
**`+0.007857`**, inside the tolerance, so the override rule did not fire and `ranking-preserved` stands. Best swept F1
**`-0.010626`**, out-of-fold **`-0.051927`**, cells positive **`+14.52` points**, row error **`+1.959731`** (worse), fit
gate **FALSE** at `1` of `28`.

**Verdict: supervising the location does not place the boundary - the level moved the wrong way again.** The declared
expectation was `midpoint-clears-the-boundary`, with the falsifiable hint that F1 at `0.0` should approach the dense
arm's swept `0.465582` if placement were the only obstacle. Instead the positive share rose a further `14.5` points to
`64.5%`, the swept optimum stayed high at `3.00`, and the primary fell by `0.071`. The AUC held (`+0.0079`), so this is
not the collapse failure mode the declaration named in advance - the classes stayed separated and the read simply sat
higher. The term was not satisfied either: its mean charge was `0.923`, not zero, on `48` of `64` steps.

### What this closes off

A per-board *location* is no easier to buy than a per-board *shape*: the midpoint can be driven toward zero from either
side, so the published likelihood's own pull (which raises the occupied class) wins the trade, and the equilibrium the
optimiser reaches still has the midpoint above the boundary. Together with M9-M12 this rules out the whole approach of
**adding supervision that a shift of the read can partially satisfy** - five terms, five negations, and the two arms that
came closest (`M12`: best swept `+0.054`, out-of-fold `+0.065`) did so on shape, not placement. What no arm did is
remove the shift, which is the thing the read must not have.

### The next structural arm, declared now

**Structural centring of the read** (`--occupancy-centre 1`): inside the head, before the loss or any scorer sees it,
subtract the board's own mean over its visible cell window from the occupancy logits - `L - mean(L)` over that board's
`rows x columns` window, no labels, no thresholds, no parameters, nothing read about prevalence. It differs from M8's
`normalise` in exactly one way and that is the point: it removes only the additive mode and leaves the spread untouched,
so it cannot be paid off by a level shift because the level is gone by the time the objective is computed.

Declared with its failure mode, because it has a real one: the mean of *all* cells is not the F1-optimal level when the
population is unbalanced (`12%` occupied), so a hard centring can overshoot the other way - and if that happens the
honest reading is `centre-fails` reported as such, with the swept optimum moving *below* zero as the evidence. Its
primary bar (`+0.05` over control's `0.381640`) and AUC tolerance (`-0.02`) are those standing numbers; its own
document must restate them with the mechanism before its own receipt.

### Limits

Twenty-eight boards, twelve images, one catalogue, one block, one seed, one launch. This measures whether supervising
the location of the published read-out's boundary moves the plain-binary operating point; it is not a real-image,
complete-state or quality claim.

## Correction (V-139, 2026-11-12)

The exact-board figure quoted above was produced by a curator rule that dropped any column whose decision or label was
empty, so a board that filled genuinely empty columns could be scored exact. Recomputed strictly from the same retained
sweep (`runs/m13-midpoint-level-r1-sweep-r1/result.json`), this arm has **0 of 28** exact non-empty boards, not the number printed above.
The scope file is left as published: it is immutable evidence of what the instrument said at the time.
