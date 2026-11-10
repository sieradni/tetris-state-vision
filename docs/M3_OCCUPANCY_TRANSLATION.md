# Occupancy translation lever — make the box origin depend on where the board is

Status: predeclared (2026-11-09) and committed before any profile, study, sweep, probe or evidence
directory for this instrument exists. The change, its matched baseline, the envelope, the population,
the four gates, the signed-offset condition and the conditional next-suspect rule below are fixed;
the measured section is appended only after the canonical receipts exist.

## Where this comes from

The geometry-stage pass ([M3_OCCUPANCY_GEOMETRY.md](M3_OCCUPANCY_GEOMETRY.md), verdict
`not-cleared`) raised the occupancy decision (best swept F1 0.4237 -> 0.5018) but moved placement
19.9% *further* from the target, and its predeclared decomposition returned `translation`:

| quantity | lever | baseline |
|---|---|---|
| signed column offset | +2.3243 cells | +1.3830 |
| signed row offset | -2.3848 cells | -2.1914 |
| extent ratio u / v | 1.2231 / 1.1386 | 1.2257 / 1.1804 |
| per-board error spread | 5.4583 cells | 4.6043 |

Both signed components exceed the 1.0-cell translation floor, so the declared ordering named
**box-origin supervision** next. Two facts shape the lever:

1. The box head is `initial_box(board[:, 0])` - a LayerNorm and a linear map on **one global token**.
   It has no spatially localized channel, so a board-dependent origin has to be inferred from a
   pooled summary of everything.
2. The residual is board-dependent, not a constant bias: per-board error spread is 4.60-5.46 cells
   while the *mean* signed offsets are ~2.3/2.4 cells. A constant offset would have been fitted away
   by the geometry term; a varying one needs an input that varies with the board.

## The single declared change

**A zero-initialized spatial-moment residual on the box head.**

| arm | `lattice_support` | meaning |
|---|---|---|
| lever | `1` | `raw_box += support_box(moments(coarse))`, where the moments are the coarse feature map's normalized energy centroid (x, y), its x/y spread and its log energy - five numbers - and `support_box` is a `Linear(5, 4)` with **zero** weight and bias |
| baseline | `0` | the published global-token box head, byte-identical to today |

Because the projection starts at zero, step zero is **exactly** the published function and both arms
start from identical saved tensors; the only difference is that the lever arm *can* learn to place
the box from spatially localized evidence. No new loss term, no label, no threshold, no read-out
feature, and the extent parameterization is untouched (that is the next branch, not this one).

Everything else is the campaign's matched 256-step configuration: same published checkpoint
`runs/m3-percell-balance-r1` (`9ae1fef9f649...`), 256 continuation steps, published center-sample
read-out (`occupancy_detail 1`, stencil `0`), `occupancy_context 0`, published geometry stage policy
(`geometry_final_only 0`), loss factor `4`, positive weight `4`, `seed 20261026`, learning rate
`.001`, curriculum `stream-v1`, capture `512`, capacity `64`, positional identity read-out, length
endpoint, 1,152 MiB, 1,800 s. Both arms run under this pass's own captured catalogue; earlier
256-step runs are unmatched cross-pass references and cannot enter this verdict.

## Envelope and declared work arithmetic

Per arm: **1,152 MiB** aggregate commit, **256 steps**, **1,800 s** runner wall, one heavy job at a
time. An arm that cannot finish inside that envelope is **NOT ADMISSIBLE AS DECLARED**; nothing is
relaxed or re-run.

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
occupied and 4,933 empty, twelve ancestry groups - at threshold `0.0`, plus the standing 81-point
grid on `[-6, 6]` step 0.15 for the best swept F1. The oracle pairing's co-scoreable subset is **not**
a judgement population: it is exactly where alignment is granted.

## The four declared conditions

**The lever CLEARS iff all four hold:**

1. **Per-cell gate** - occupied-cell F1 at `0.0` on the full population `>= 0.70`.
2. **Ranking gate** - `sweep_verdict` is `informative` (AUC `>= 0.90` and some swept point `>= 0.70`).
3. **Attribution floor** - best swept F1 exceeds the matched baseline's by `>= 0.05`, with the
   whole-ancestry-cluster paired bootstrap (400 draws, seed `20261026`) reported beside it; the point
   delta carries the floor, as in every preceding pass.
4. **Signed-offset condition** - **both** signed components fall to at most **half** their matched
   baseline magnitude (`|signed column| <= 0.5 * |baseline signed column|` and the same for the row
   axis). A box-origin lever that leaves the origin where it was has not done the thing it was ranked
   to do, whatever the occupancy score does. A sign reversal is not a reduction.

Otherwise the pass is **`not-cleared`** with the failing conditions named mechanically.

## Declared next-suspect rule (non-circular)

`r` is the smallest per-axis signed reduction, `1 - |lever|/|baseline|`. First match wins:

| rule | condition | next suspect |
|---|---|---|
| extent | `r < 0.25` (the origin barely moved) | box-extent supervision of the lattice box |
| read-out inheritance | the signed condition is met but the per-cell/ranking gates are not | a lattice-agnostic occupancy read-out |
| dispersion | the lever's per-board error spread `>= 1.5` cells | per-board lattice refinement capacity |
| extent | otherwise (the origin moved but not enough) | box-extent supervision of the lattice box |

This deliberately cannot return "translation" again: translation has been tried and its outcome is an
input to the rule. The extent branch is the next measurable one because the same receipt measures
extent ratios 1.22 / 1.14 (both past the predeclared 0.10 log-ratio floor) and an extent lever is a
single structural change to the same box head.

## Reconciliation the receipt must carry

The curated scope must report, from the previous evidence file (reference only, never verdict
input): the geometry lever's occupancy attribution and its placement cost. This pass's baseline is the
same configuration as that pass's baseline, so the receipt states plainly whether the translation
lever's occupancy number is above, at, or below the gain the geometry lever bought, and whether the
placement cost was repaid.

## Receipts this pass must produce

1. `runs/m3-occorigin-{lever,baseline}-r1-profile-r1` (2 steps, 1,152 MiB / 300 s / 2 work items).
2. `runs/m3-occorigin-{lever,baseline}-r1` (256 steps, 1,152 MiB / 1,800 s / 256 work items), each
   admitted only by its own profile.
3. `...-sweep-r1-profile-r1` (2 forwards) and `...-sweep-r1` (54 forwards) per arm, 640 MiB.
4. `...-probe-r1` per arm (36 forwards, 640 MiB / 300 s).
5. Curation `runs/m3-occorigin-curation-r1-job.json` into
   `docs/evidence/m3-occupancy-translation-r1/scope.json`, re-deriving every published number and
   asserting each probe reproduces its own sweep's per-cell F1 (1e-6) and TP/FP/FN counts.
6. The owned family `m3-occorigin-*` declared in `record_stream_run.OWNED_RUN_PATTERN`, bound by a test.

Any launch that dies, leaves its declared arithmetic or is superseded is preserved under
`runs/attempts/` with a written note and re-recorded under a fresh number; none is evidence.

## Limits, stated before the measurement

- Synthetic, training-fit population; nine of twelve groups never appear in the run's own ledger. No
  real-screenshot, generalization, calibration or complete-state claim follows.
- One budget point (256 steps), one checkpoint, one structural change. The lever adds a small
  zero-initialized projection; it does not add capacity to the encoder or a second box head.
- Placement is measured against the committed target lattice, so it says nothing about deployment
  situations where that lattice is unavailable.

## Measured (curated 2026-11-09)

Receipts, each passed inside its declared envelope: profiles
`runs/m3-occorigin-{lever,baseline}-r1-profile-r1` (2 work items), studies
`runs/m3-occorigin-{lever,baseline}-r1` (256 steps, 1,152 MiB / 1,800 s; 380.8 s / 380.7 s, peaks
1,085,526,016 B / 1,083,441,152 B), sweep profiles `...-sweep-r1-profile-r1` and sweeps `...-sweep-r1`
(54 forwards, 640 MiB), probes `...-probe-r1` (36 forwards, 640 MiB / 300 s), and curation
`runs/m3-occorigin-curation-r2-job.json` into `docs/evidence/m3-occupancy-translation-r1/scope.json`.
Every published number below is re-derived by the curator from the probes' own cells, and each probe
reproduces its own sweep's per-cell F1 at `0.0` and its TP/FP/FN counts before any verdict is written.

**One predeclaration sentence was imprecise and is corrected here.** "Both arms start from identical
saved tensors" cannot hold for a lever that *adds* a parameter: the lever's `initial-weights.pt` carries
four extra zero tensors, so its digest differs (`a5f71c8b...` against `eec1e10e...`). The matched
condition that was actually required - and is now enforced - is the one that matters: every published
tensor is bitwise identical between the two saved states, the added `support_box.{weight,bias}` are
exactly zero, and both arms' step-0 probes report the identical loss on the identical cases
(`12.281119876437717` training, `10.749210410647922` development), so step zero is the same function in
both arms. The first curation launch enforced the inherited file-digest guard instead and was refused by
it (V-071); that launch is preserved under `runs/attempts/` and is cited nowhere.

| quantity | lever | baseline |
|---|---|---|
| per-cell F1 at `0.0` | 0.4165 (622/1,698/45) | 0.4146 (351/675/316) |
| AUC | 0.8446 | 0.8255 |
| best swept F1 (threshold) | 0.4340 (`0.45`) | 0.4237 (`-0.30`) |
| signed column offset | +1.3998 | +1.3830 |
| signed row offset | -2.6181 | -2.1914 |
| mean placement error | 2.1988 | 1.9812 |
| extent ratio u / v | 1.3242 / 1.1709 | 1.2257 / 1.1804 |
| per-board error spread | 4.7870 | 4.6043 |
| mapped share / shift share | 0.6334 / 0.6020 | 0.6507 / 0.5375 |

**Verdict: `not-cleared`; all four declared conditions fail.**

1. `per_cell_f1` - 0.4165 < 0.70.
2. `ranking` - `uninformative`: AUC 0.8446 < 0.90 and no swept point reaches 0.70 (best 0.4340).
3. `attribution` - the best-F1 point delta **+0.0102** is below the declared +0.05. The
   whole-ancestry-cluster paired bootstrap (400 draws, seed 20261026, twelve clusters) puts that delta
   at median **-0.0378** with interval **-0.1700 to +0.0872**: at this budget the occupancy difference
   is not distinguishable from zero, and the resampling does not reproduce the sign of the point delta.
4. `signed_offsets` - both reductions are negative: columns `1 - 1.3998/1.3830 = -0.0122`, rows
   `1 - 2.6181/2.1914 = -0.1947`. The origin moved *outward* on both axes, not toward zero.

**The lever did not do the thing it was ranked to do.** Giving the box head spatially localized moment
evidence left the signed column offset essentially where it was (-1.2%) and pushed the row offset
**19.5% further out**, while mean placement error rose **11.0%** (1.9812 -> 2.1988) and the share of
cells mapping onto any target cell fell 0.6507 -> 0.6334. What occupancy movement it did produce is an
operating-point shift, not a better-ordered score: F1@0 gains 0.0018 by trading 271 false negatives for
1,023 false positives, and AUC gains 0.0191. Per-board spread rose 4.6043 -> 4.7870.

**Reconciliation with the geometry pass (reference only, never verdict input).** That lever bought
**+0.0781** best F1 (interval -0.0152 to +0.1243) and cost **19.9%** placement (1.9812 -> 2.3757). This
lever buys **+0.0102** - about an eighth of that occupancy gain - and adds a further **11.0%** placement
cost (to 2.1988). The two costs add rather than compensate, so the geometry lever's placement cost was
*not* repaid by moving the origin, and its occupancy gain was not a hidden origin correction: it was the
read-out absorbing the offset, exactly as that pass concluded.

**Next suspect, by the declared non-circular rule: `extent` -> box-extent supervision of the lattice
box.** The smallest per-axis signed reduction is **-0.1947 < 0.25**, so the first row of the declared
table matches: the origin barely moved. The rule cannot return `translation` again, by construction.
Extent is the next *measurable* branch because this pass measured it on the same receipts - ratios
**1.3242 / 1.1709**, both past the 0.10 log-ratio floor (log 0.2808 / 0.1578) - and because an extent
lever is one structural change to the same four box parameters (their scale, not their origin), so it
can be admitted, matched and read with this instrument unchanged. Dispersion (spread 4.7870, past 1.5
cells) and read-out inheritance remain ranked behind it.

**Limits.** Synthetic, training-fit population; nine of the twelve ancestry groups never appear in the
run's own ledger. One budget point (256 steps), one checkpoint (`9ae1fef9f649...`), one structural
change, one seed. The bootstrap resamples only twelve clusters, so its interval is coarse. Placement is
measured against the committed target lattice and says nothing about deployment situations where that
lattice is unavailable. Nothing here is a real-screenshot, generalization, calibration or complete-state
claim - and a cleared lever would not have been one either.
