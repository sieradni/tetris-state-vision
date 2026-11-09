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
