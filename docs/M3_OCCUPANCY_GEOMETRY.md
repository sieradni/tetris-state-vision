# Occupancy geometry lever — spend the lattice gradient on the lattice the read-out reads

Status: predeclared (2026-11-09) and committed before any profile, study, sweep, probe or evidence
directory for this instrument exists. The change, the matched baseline, the envelope, the
population, the four gates, the placement measurement and the conditional next-suspect rule below
are fixed; the measured section is appended only after the canonical receipts exist.

## Where this comes from

The registration pass ([M3_OCCUPANCY_SCALED.md](M3_OCCUPANCY_SCALED.md), verdict
`registration_implicated`) measured the occupancy residual as cell placement: correcting placement
alone moved F1 0.4390 -> 0.5742 (+0.1352, interval +0.0460 to +0.2833) on the co-scoreable cells,
with a mean offset of 2.57 cells and 48.3% of cells on the wrong target cell. Two structural facts
connect that residual to the objective:

1. The occupancy head samples its features at the model's **own predicted lattice**
   (`model.py: sample_features(detail, lattice, ...)`), so every occupancy cell index inherits the
   placement error directly.
2. `joint_loss` averages the geometry term over **all six decoder stages**, while only the final
   stage's lattice is the one the read-out samples. The final lattice therefore receives one sixth
   of the placement gradient that the published objective spends, and the other five sixths are
   spent on boxes nothing downstream reads.

The `registration` term is unrelated to placement: it is a binary registered/unregistered
classifier. No previous pass has touched the geometry stage policy.

## The single declared change

**Geometry stage policy only.**

| arm | `geometry_final_only` | meaning |
|---|---|---|
| lever | `1` | the geometry term is the final stage's lattice error |
| baseline | `0` | the geometry term is the mean over all six stages (published objective) |

Everything else is identical to the campaign's matched 256-step configuration: the same published
checkpoint `runs/m3-percell-balance-r1` (`9ae1fef9f649...`), 256 continuation steps, the published
center-sample detail tap (`occupancy_detail 1`, `occupancy_detail_stencil 0`), `occupancy_context 0`,
loss factor `4`, positive weight `4`, `seed 20261026`, `learning_rate .001`, curriculum `stream-v1`,
capture side `512`, capacity `64`, positional identity read-out, length endpoint, 1,152 MiB and
1,800 s. Both arms run under this pass's own captured catalogue; the earlier 256-step runs are an
unmatched cross-pass reference only and cannot enter this verdict.

With `geometry_final_only = 0` the objective is byte-identical to the published one; the flag adds no
parameter, no loss term, no label and no read-out feature.

## Envelope and declared work arithmetic

Per arm: **1,152 MiB** aggregate commit, **256 steps**, **1,800 s** runner wall, one heavy job at a
time. If an arm cannot finish inside that envelope the outcome is **NOT ADMISSIBLE AS DECLARED**,
recorded with its measured peak and wall; nothing is relaxed or re-run.

| launch | work items | what is counted |
|---|---|---|
| each profile | 2 | 2 optimizer updates |
| each study | **256** | 256 optimizer updates |
| each sweep profile | 2 | 2 forwards |
| each measured sweep | **54** | 36 deciding + 18 development forwards |
| each placement probe | **36** | 36 deciding forwards, 640 MiB / 300 s |
| curation | **8** | eight receipts read (two studies, two sweeps, two probes, two profiles) |

## Scoring population: the full deciding population

**Every known cell of every nonempty deciding board** - 28 boards, 200 cells each, 5,600 known cells,
667 occupied and 4,933 empty, twelve ancestry groups - at the standing decision threshold `0.0` and
over the standing 81-point grid on `[-6, 6]` step 0.15. The oracle pairing's co-scoreable subset is
**not** a judgement population here: it is exactly where alignment is granted, so using it would
bank the effect the lever is supposed to produce. The placement instrument computes its offsets on
all 200 cells of the same boards.

## The four declared conditions

**The lever CLEARS iff all four hold:**

1. **Per-cell gate** - occupied-cell F1 at `0.0` on the full population `>= 0.70`.
2. **Ranking gate** - `sweep_verdict` is `informative` (pooled AUC `>= 0.90` and some swept
   threshold reaches F1 `>= 0.70`).
3. **Attribution floor** - the lever's best swept F1 exceeds **its matched-budget baseline's** best
   swept F1 by `>= 0.05`, with the whole-ancestry-cluster paired bootstrap (400 draws, seed
   `20261026`) reported beside it. As in every preceding pass, the point delta carries the floor and
   no gate reads the interval.
4. **Placement measurement** - the declared placement error, the mean over the 200 cells and two
   axes of `|offset|` in target cells, must fall by `>= 25%` relative to the matched baseline. A
   geometry lever that does not move placement has not done the thing it was ranked to do, whatever
   happens to the occupancy score.

Otherwise the pass is **`not-cleared`**, with the failing conditions named mechanically.

## Declared placement decomposition and next-suspect rule

The probe always reports the placement decomposition, and on a refusal it names the next measurable
suspect by this precedence (first match wins), fixed now:

| rule | condition on the lever's own residual | next suspect |
|---|---|---|
| translation | `|signed column offset| >= 1.0` or `|signed row offset| >= 1.0` cell | box-origin supervision of the lattice box |
| scale | `|log(mean |u| ratio)| >= 0.10` or `|log(mean |v| ratio)| >= 0.10` | box-extent supervision of the lattice box |
| dispersion | per-board error spread (max - min) `>= 1.5` cells | per-board lattice refinement capacity |
| read-out inheritance | placement improved `>= 25%` but the occupancy gates did not follow | a lattice-agnostic occupancy read-out |
| geometry capacity | none of the above | geometric supervision capacity (support anchors) |

The rule is a decision over measured quantities, not an assertion about cause: each branch names the
next lever to predeclare, with the residual it must move.

## Receipts this pass must produce

1. `runs/m3-occgeom-{lever,baseline}-r1-profile-r1` (2 steps, 1,152 MiB / 300 s / 2 work items).
2. `runs/m3-occgeom-{lever,baseline}-r1` (256 steps, 1,152 MiB / 1,800 s / 256 work items), each
   admitted only by its own profile.
3. `...-sweep-r1-profile-r1` (2 forwards) and `...-sweep-r1` (54 forwards) per arm, 640 MiB.
4. `...-probe-r1` per arm (36 forwards, 640 MiB / 300 s).
5. Curation `runs/m3-occgeom-curation-r1-job.json` into
   `docs/evidence/m3-occupancy-geometry-r1/scope.json`, re-deriving every published number, and
   asserting each probe reproduces its own sweep's published per-cell F1 (1e-6, published precision)
   and its three operating counts before any verdict is written.
6. The owned family `m3-occgeom-*` declared in `record_stream_run.OWNED_RUN_PATTERN`, bound by a test.

Any launch that dies to a harness limit, leaves its declared arithmetic, or is superseded is
preserved under `runs/attempts/` with a written note and re-recorded under a fresh number; none is
evidence.

## What each outcome licenses

- **Cleared** - placement was the binding constraint and the objective can move it; the occupancy
  head opens on the full deciding population, the fit gate is re-read, and the release-side chain is
  next (never a real-image claim from synthetic evidence).
- **Not cleared with placement improved** - the objective moves geometry but not enough, or the
  read-out cannot exploit it; the decomposition names which, and the next pass targets that.
- **Not cleared with placement unmoved** - the stage policy is not the geometry constraint; the
  decomposition names where the residual sits and the next lever is the matching one.
- **Not admissible** - the envelope is the result; nothing is relaxed.

## Limits, stated before the measurement

- Synthetic, training-fit population: the stream's own drawings, nine of whose twelve groups the run
  never saw in its own ledger. No real-screenshot, generalization, calibration or complete-state
  claim follows.
- One budget point (256 steps) and one checkpoint; the geometry flag is a single policy change, not a
  sweep over stage weightings.
- Placement error is measured against the committed target lattice; it says nothing about situations
  where the target lattice itself is unavailable in deployment.

## Measured: the stage policy moves occupancy but not placement, so the lever is refused

The declaration above is unchanged from commit `35ae175` and was committed before any receipt
for this instrument existed. Everything below is re-derived by the curator
(`scripts/record_geometry_lever.py`) from the two probes' own cells and published in
`docs/evidence/m3-occupancy-geometry-r1/scope.json`.

Both arms ran 256 steps from `9ae1fef9f649...` under one captured catalogue (`6879a8208ad2`) and one
identical initial tensor file, at 1,152 MiB / 1,800 s:

| arm | best swept F1 (threshold) | F1 @ 0.0 (TP/FP/FN) | AUC | placement (cells) | peak | wall |
|---|---|---|---|---|---|---|
| lever (final-stage geometry) | 0.5018 (0.45) | 0.4791 (459/790/208) | 0.8316 | 2.3757 | 1,087,176,704 B | 396.3 s |
| baseline (all six stages) | 0.4237 (-0.30) | 0.4146 (351/675/316) | 0.8255 | 1.9812 | 1,058,406,400 B | 399.7 s |

**Attribution is real: +0.0781 best swept F1** (cluster interval -0.0152 to +0.1243, 400 draws),
with +0.0645 per-cell F1 and +0.0061 AUC. **Placement is not: it got 19.9% worse**, 1.9812 -> 2.3757
cells, and the shift share rose from 0.5375 to 0.5816. The lever therefore fails the declared
test it was ranked on, and it also fails the per-cell gate (0.4791 < 0.70) and the ranking gate
(AUC 0.8316 < 0.90). **Verdict: `not-cleared`, failing `per_cell_f1`, `ranking`, `placement`** -
the stage policy buys occupancy decisions but does not fix where the cells are put.

That combination is itself the result: the occupancy head improved while the lattice it samples
got *further* from the target, so the gain came from the read-out absorbing the offset, not from
better geometry. Spending the whole placement gradient on the final lattice is therefore not the
binding constraint on placement.

### The refusal names the next suspect, by the declared rule

The measured decomposition of the lever's own residual hits the first declared branch:

| quantity | lever | baseline |
|---|---|---|
| signed column offset | **+2.3243 cells** | +1.3830 |
| signed row offset | **-2.3848 cells** | -2.1914 |
| cell-extent ratio u / v | 1.2231 / 1.1386 | 1.2257 / 1.1804 |
| per-board error spread | 5.4583 cells | 4.6043 |

Both signed offsets exceed the 1.0-cell translation floor, so the rule returns **`translation`** and
names **box-origin supervision of the lattice box** as the next measurable lever. The residual is
not a vague "geometry problem": it is a systematic ~2.3-cell right / ~2.4-cell down displacement of
the predicted board box, stable across the two objectives, with a 22% extent exaggeration in train.

### Re-ranked next step

**Predeclare a box-origin (translation) lever on the lattice box, and read it with these same four
conditions.** The origin displacement is the largest single measured component, it survives the
stage-policy change, and it is the cheapest of the remaining branches to instrument: the declared
quantity to move is `signed_column_offset_cells` and `signed_row_offset_cells` toward zero, the
occupancy gates stay as published, and the placement condition stays the 25% relative floor. If the
origin lever also refuses, the predeclared rule's next branch is extent (the u/v ratios are already
1.22/1.14), then per-board dispersion (spread 5.46 cells), then read-out inheritance.

### What this pass does and does not show

- It shows the geometry-stage policy is not the binding constraint on placement, and it quantifies
  the residual as a translation with an inflated extent on the full 28-board deciding population.
- It does **not** show that placement cannot be fixed from this objective, nor that the occupancy
  gain would survive a corrected lattice: the +0.0781 attribution is on the uncorrected population.
- Synthetic, training-fit, one checkpoint, one budget, one policy change; no real-image,
  generalization, calibration or complete-state claim follows.
