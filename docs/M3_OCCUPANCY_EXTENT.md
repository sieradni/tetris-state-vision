# Occupancy extent lever — supervise the size of the lattice box, not where it sits

Status: predeclared (2026-11-09). The only receipts that predate this commit are the two 2-step
admission profiles (`runs/m3-occextent-{lever,baseline}-r1-profile-r1`, 40.4 s / 39.5 s, peaks
1,033,248,768 B / 1,038,548,992 B) and the test launches `runs/m3-occextent-tests-r1..r3`. No study,
sweep, probe or evidence directory existed and **no measurement number had been seen** when these
gates, thresholds, falsifiers and the next-suspect rule were fixed; the measured section is appended
only after the canonical receipts exist.

## Where this comes from

Two levers have been spent and refused, and both left the box extent untouched:

| pass | verdict | occupancy attribution | placement cost | extent u / v |
|---|---|---|---|---|
| geometry stage ([M3_OCCUPANCY_GEOMETRY.md](M3_OCCUPANCY_GEOMETRY.md)) | `not-cleared` | +0.0781 (-0.0152..+0.1243) | -19.9% (1.9812 -> 2.3757) | 1.2231 / 1.1386 |
| box origin ([M3_OCCUPANCY_TRANSLATION.md](M3_OCCUPANCY_TRANSLATION.md)) | `not-cleared` | +0.0102 (-0.1700..+0.0872) | -11.0% (1.9812 -> 2.1988) | 1.3242 / 1.1709 |

The origin lever's declared decomposition then took its first branch - the smallest signed reduction was
`-0.1947 < 0.25`, so the origin barely moved - and named **box extent** next, on the same receipts'
measurement: L1/L2 **1.3242 / 1.1709**, both past the 0.10 log-ratio floor, with the row axis already
the smallest remaining candidate before it. The published box head predicts `width` and `height` from one
global token through `decode_lattice`, and nothing in the objective treats the *scale* of the basis
separately from its position: the geometry term is a smooth-L1 on all 204 projected points, so an
extent bias of ~20% is diluted into a mean that the origin errors dominate.

## The single declared change

**A relative log-scale term on the final lattice.**

| arm | `box_extent_weight` | meaning |
|---|---|---|
| lever | `1.0` | the published objective plus `1.0 * mean_registered(|log(‖u_pred‖/‖u_target‖)|, |log(‖v_pred‖/‖v_target‖)|)`, evaluated on the **final** decoder lattice - the basis the occupancy read-out samples |
| baseline | `0.0` | the published objective, byte-identical to today |

The term is exactly the quantity the placement probe reports as L1/L2, axis-resolved and scale-free, so
the lever supervises the measured defect rather than a proxy. Weight `1.0` is declared as *equal to the
published geometry term's own weight*, not tuned: no value between 0 and 1 is searched, and `0.0`/`1.0`
is the whole declared set. No new parameter, no new input, no read-out change, no threshold change, and
the origin and occupancy read-outs are untouched, which is what distinguishes this from the two refused
levers: the origin lever added an input channel, the geometry lever reallocated where an existing term
was applied, and this one adds an axis-resolved scale term to the objective.

Everything else is the campaign's matched 256-step configuration: same published checkpoint
`runs/m3-percell-balance-r1` (`9ae1fef9f649...`), 256 continuation steps, published center-sample
read-out (`occupancy_detail 1`, stencil `0`), `occupancy_context 0`, published geometry stage policy
(`geometry_final_only 0`), published box head (`lattice_support 0`), loss factor `4`, positive weight
`4`, `seed 20261026`, learning rate `.001`, curriculum `stream-v1`, capture `512`, capacity `64`,
positional identity read-out, length endpoint, 1,152 MiB, 1,800 s. Both arms run under this pass's own
captured catalogue; earlier 256-step runs are unmatched cross-pass references and cannot enter this
verdict.

## Matched start: proven, not assumed

This lever adds no parameter, so the two arms must save **bitwise identical** initial tensors. The shared
contract `scripts/matched_start.py` (V-071) is used, not a copy: its no-addition branch requires every
tensor in both saved states to be equal, and `step_zero_identical` additionally requires the two arms'
step-0 training and development probe metrics to be equal, so step zero is the same measured function.
The parameter-adding branch of the same contract is what the origin lever now uses; one test file binds
both branches so neither can drift. If a launch is refused by it, the launch is preserved under
`runs/attempts/` with a note and re-recorded under a fresh number; none is evidence.

## Envelope and declared work arithmetic

Per arm: **1,152 MiB** aggregate commit, **256 steps**, **1,800 s** runner wall, one heavy job at a time.
An arm that cannot finish inside that envelope is **NOT ADMISSIBLE AS DECLARED**; nothing is relaxed or
re-run.

| launch | work items | what is counted |
|---|---|---|
| each profile | 2 | 2 optimizer updates |
| each study | **256** | 256 optimizer updates |
| each sweep profile | 2 | 2 forwards |
| each measured sweep | **54** | 36 deciding + 18 development forwards |
| each placement probe | **36** | 36 deciding forwards, 640 MiB / 300 s |
| curation | **8** | eight receipts read |

## Scoring population: the full deciding population

Every known cell of every nonempty deciding board - 28 boards, 200 cells each, 5,600 known cells, 667
occupied and 4,933 empty, twelve ancestry groups - at threshold `0.0`, plus the standing 81-point grid on
`[-6, 6]` step 0.15 for the best swept F1. The oracle pairing's co-scoreable subset is **not** a judgement
population: it is exactly where alignment is granted.

## The four declared conditions

**The lever CLEARS iff all four hold:**

1. **Per-cell gate** - occupied-cell F1 at `0.0` on the full population `>= 0.70`.
2. **Ranking gate** - `sweep_verdict` is `informative` (AUC `>= 0.90` and some swept point `>= 0.70`).
3. **Attribution floor** - best swept F1 exceeds the matched baseline's by `>= 0.05`, with the
   whole-ancestry-cluster paired bootstrap (400 draws, seed `20261026`) reported beside it; the point
   delta carries the floor, as in every preceding pass.
4. **Extent condition** - **both** axis log-ratios fall to at most **half** their matched baseline
   magnitude: `1 - |log L_axis| / |log L_axis, baseline| >= 0.5` for the column axis and the row axis. A
   ratio that crosses 1.0 counts by its log magnitude, since `|log|` is the measured quantity; an extent
   that merely changes sign without shrinking is not a reduction.

Otherwise the pass is **`not-cleared`** with the failing conditions named mechanically.

## Falsifiers, stated before the measurement

These are interpretation rules, not extra gates, and they bind the write-up:

- **Extent not moved** - if either axis reduces by `< 0.5`, the lever failed its own condition and the
  write-up may not claim extent supervision works, whatever occupancy does.
- **Extent moved, placement not improved** - if both axes reduce by `>= 0.5` but mean placement error is
  not below the baseline's 1.9812 cells, then the extent parameter moved without the placement error
  following, which falsifies "extent is the binding placement constraint" and forbids crediting the
  lever with the placement residual.
- **Extent moved, occupancy not attributed** - if both axes reduce by `>= 0.5` and attribution is
  `< 0.05`, then extent is not what limits the occupancy score, which falsifies the ranking that extent
  precedes read-out inheritance.

## Declared next-suspect rule (non-circular)

`r` is the smallest per-axis extent reduction. First match wins; the rule can never return `extent`:

| rule | condition | next suspect |
|---|---|---|
| geometry_capacity | `r < 0.25` (the box extent barely moved) | geometric supervision capacity (support anchors) |
| read_out_inheritance | the extent condition is met but the per-cell/ranking gates are not | a lattice-agnostic occupancy read-out |
| dispersion | the lever's per-board error spread `>= 1.5` cells | per-board lattice refinement capacity |
| scaling | otherwise (extent moved, occupancy ranked, fences still short) | occupancy read-out at a second scale |

`scale` and `translation` have both been tried and their outcomes are inputs to this rule, so neither can
be returned again.

## Reconciliation the receipt must carry

The curated scope must report, from the two previous evidence files (reference only, never verdict
input): the geometry lever's occupancy attribution and placement cost, and the origin lever's occupancy
attribution and placement cost. This pass's baseline is the same configuration as both, so the receipt
states plainly whether extent supervision repaid either the +0.0781 occupancy gain or the 19.9% placement
cost, or added a third cost.

## Receipts this pass must produce

1. `runs/m3-occextent-{lever,baseline}-r1-profile-r1` (2 steps, 1,152 MiB / 300 s / 2 work items).
2. `runs/m3-occextent-{lever,baseline}-r1` (256 steps, 1,152 MiB / 1,800 s / 256 work items), each
   admitted only by its own profile.
3. `...-sweep-r1-profile-r1` (2 forwards) and `...-sweep-r1` (54 forwards) per arm, 640 MiB.
4. `...-probe-r1` per arm (36 forwards, 640 MiB / 300 s).
5. Curation `runs/m3-occextent-curation-r1-job.json` into
   `docs/evidence/m3-occupancy-extent-r1/scope.json`, re-deriving every published number and asserting
   each probe reproduces its own sweep's per-cell F1 (1e-6) and TP/FP/FN counts.
6. The owned family `m3-occextent-*` declared in `record_stream_run.OWNED_RUN_PATTERN`, bound by a test.

Any launch that dies, leaves its declared arithmetic or is superseded is preserved under
`runs/attempts/` with a written note and re-recorded under a fresh number; none is evidence.

## Limits, stated before the measurement

- Synthetic, training-fit population; nine of twelve groups never appear in the run's own ledger. No
  real-screenshot, generalization, calibration or complete-state claim follows.
- One budget point (256 steps), one checkpoint, one structural change, one declared weight, one seed.
- Placement is measured against the committed target lattice, so it says nothing about deployment
  situations where that lattice is unavailable.
- The term supervises `|log|` of the basis norms only; it cannot teach which of two axis-aligned bases is
  the intended one, and it says nothing about rotation or shear.


## Measured (2026-11-09): the extent term did not shrink the box, and it cost both scores

Both arms ran 256 steps from `9ae1fef9f649...` under one captured catalogue (`f50dee7a...`), each admitted
only by its own 2-step profile, and both stayed inside the declared envelope: lever **814.2 s**, peak
**1,066,991,616 B**; baseline **1,120.7 s**, peak **1,067,384,832 B** (ceiling 1,200,000,000 B). The
baseline reproduces the curated baseline of both earlier passes exactly - F1@0 **0.414649** with counts
351/675/316/4258, AUC **0.825539**, best swept F1 **0.423720** at `-0.30`, ratios **1.225688 / 1.180353**,
mean placement **1.981220**, spread **4.604344**, mapped share **0.650714** - so the pair is the same
configuration and the contrast is the declared change.

| condition | bar | lever | baseline | met |
|---|---|---|---|---|
| per-cell F1 at `0.0` | >= 0.70 | **0.331497** (361 TP / 1,150 FP / 306 FN) | 0.414649 (351 / 675 / 316) | **no** |
| ranking | AUC >= 0.90, swept F1 >= 0.70 | **0.752582**, best **0.455556** at `2.10` | 0.825539, 0.423720 at `-0.30` | **no** |
| attribution | best-F1 delta >= +0.05 | **+0.031836** | - | **no** |
| extent | both axes reduced >= 50% | u **-1.075204**, v **+0.174746** | 1.225688 / 1.180353 | **no** |

**Verdict: `not-cleared`, failing all four declared conditions.**

The extent condition fails in the direction the predeclaration did not anticipate: the term *inflated* the
column axis. Ratios moved **1.225688 -> 1.525480** on columns (|log| 0.2035 -> 0.4225, a reduction of
**-107.5%**) while rows moved 1.180353 -> 1.146643 (**+17.5%**, not the 50% required). At weight `1.0`, on
36 boards and 256 steps, the objective did not act as an extent regularizer: whatever descent path it took
was dominated by the geometry and occupancy terms, and the box grew along the axis it was meant to
constrain.

**Placement did not follow either.** Mean placement error rose **67.5%**, 1.981220 -> **3.318173** cells,
signed column offset +1.382965 -> **+3.686498**, row -2.191449 -> **-2.948318**, per-board spread 4.604344
-> **5.971017**, mapped share 0.650714 -> **0.514464**. Both falsifiers declared before the measurement
hold at once: the extent did not move as declared *and* the placement error did not improve, so neither
"extent is the binding placement constraint" nor "the placement residual was an extent residual" can be
credited to this lever.

**Attribution is negative where it counts.** The best swept F1 rises by **+0.031836** (below the +0.05
floor) at a threshold of `2.10` instead of `-0.30`, but that is an operating-point move: the twelve-cluster
paired bootstrap (400 draws) puts the delta at median **-0.084752**, interval **-0.134500 to -0.031271**,
and the standing point falls **0.414649 -> 0.331497** while AUC falls **0.825539 -> 0.752582**. The extra
true positives (351 -> 361) arrive with 475 more false positives (675 -> 1,150). The same matched baseline
appearing unchanged in three consecutive passes (geometry, origin, extent) is the control: the drift is
the levers', not the checkpoint's.

**Reconciliation.** Geometry supervision bought **+0.078064** occupancy at **19.9%** worse placement; the
box origin bought **+0.010235** at **11.0%** worse; the extent term buys **+0.031836** (an operating point,
not a ranking gain) at **67.5%** worse placement. The three costs add, and the third is the largest, so no
earlier gain is explained by a hidden extent correction and extent supervision is not the cheaper route to
either. The origin pass had already left the box wider than the baseline (1.3242 / 1.1709), so the box had
been drifting outward under both prior levers.

**Declared next suspect.** The non-circular rule takes its first branch: smallest extent reduction
**-1.075204 < 0.25** -> `geometry_capacity`, **geometric supervision capacity (support anchors)**. This is
a ranking, not a measurement; `read_out_inheritance`, `dispersion` (spread 5.971017) and `scaling` stay
behind it, and neither `scale` nor `translation` can be returned.

**Matched start, and one repair.** The first curation, `runs/m3-occextent-curation-r1`, was refused by
`matched_start.step_zero_identical` on `initial_training_probe`. The arms' saved initial weights were
bitwise identical and their step-0 predictions matched on every field; the only differences were `loss`
and `cases[*].loss`, which an objective lever changes by construction. That is **V-072**, an instrument
error of the V-071 class one layer lower (V-071 assumed the lever adds no parameter; this assumed it
changes no objective value). `step_zero_identical` now strips the objective's own fields recursively and
names the differing field, and `MatchedStartTests` binds both the accept case (a re-lossed probe) and the
refuse cases (moved `cell_accuracy`, missing probe). The refused curation and a profile launch killed by a
caller timeout are preserved under `runs/attempts/` with notes; neither is evidence.

**Receipts.** Profiles `runs/m3-occextent-{lever,baseline}-r1-profile-r1` (40.4 s / 39.5 s); studies
`runs/m3-occextent-{lever,baseline}-r1` (256 steps each); sweep profiles
`runs/m3-occextent-{lever,baseline}-r1-sweep-r1-profile-{r1,r2}` (2 forwards each - the baseline's first
launch was killed by a caller timeout and re-recorded as `-r2`, and each sweep now binds its profile
through its own `admission.run` record); sweeps `...-sweep-r1` (54 forwards, 73.8 s / 74.0 s, 640 MiB);
placement probes `...-probe-r1` (36 forwards, 51.3 s / 49.8 s, peaks 564,518,912 B / 556,527,616 B);
curation `runs/m3-occextent-curation-r2-job.json` into `docs/evidence/m3-occupancy-extent-r1/scope.json`,
which re-derived every published number and required each probe to reproduce its own sweep's per-cell F1
and TP/FP/FN counts at 1e-6. Focused tests `runs/m3-occextent-tests-r4-job.json` (50 tests).

**Limits.** Synthetic, training-fit, one budget point, one checkpoint, one declared weight, one seed,
committed lattice only; nine of twelve groups never appear in the run's own ledger. The extent term
supervises the magnitude of the basis norms only and cannot teach which axis-aligned basis is intended.
Nothing here is a real-image, complete-state, calibration or generalization claim, and no gate, threshold,
population or prior evidence byte moved.
