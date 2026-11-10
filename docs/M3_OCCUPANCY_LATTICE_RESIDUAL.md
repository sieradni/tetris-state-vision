# Per-stage lattice residual capacity — give each decoder stage its own decoded-space box correction

Status: predeclared (2026-11-10) and committed before any profile, study, sweep, probe or evidence
directory for this instrument exists. The change, its parameter cost, the matched start, the envelope,
the population, the four conditions, the regression falsifier, the control-drift rule, the
predeclaration-ancestry rule and the non-circular next-suspect rule below are fixed; the measured
section is appended only after the canonical receipts exist.

## Where this comes from

The geometry-capacity pass was the last ranked suspect and it lost every score
([M3_OCCUPANCY_CAPACITY.md](M3_OCCUPANCY_CAPACITY.md), `not-cleared`: attribution **-0.0241**,
placement **-28.2%**). Its declared non-circular rule took its first branch - placement improvement
`-0.282 < 0.05` - and named **per-stage lattice residual capacity**. `translation`, `scale` and
`geometry_capacity` remain excluded as outputs.

The premises are in the published model and are checkable:

1. Every decoder stage decodes **one shared `raw_box`** - `initial_box(board_token)` plus
   `box_refine[index]` for the first `refinements = 2` of `blocks = 6` stages.
2. `decode_lattice` is a **single sigmoid decode** that ties the origin to the extent
   (`width = (1-left)*(.05+.95*sigmoid)`), so extent cannot move independently of where the box sits,
   and a stage cannot correct its own boundary at all without moving the shared box's four parameters.
3. Each stage's occupancy head samples at **that stage's own lattice**, and the published geometry term
   supervises **every** stage's lattice.

So four of six stages carry no box refinement, and no stage can express a decoded-space correction:
the only per-stage capacity is a pre-sigmoid nudge to four numbers shared by the whole decoder.

## The single declared change

**A per-stage, zero-initialized residual applied to the decoded lattice.**

| arm | `lattice_residual` | meaning |
|---|---|---|
| lever | `1` | each stage adds `nn.Linear(4, 4)` on that stage's refined `raw_box`, applied to the decoded basis |
| baseline | `0` | the published model, byte-identical to today |

For stage `i` with residual `r = Linear_i(raw_box)`:

```
value  = 0.25 * tanh(r)
offset = value[0:2] cells along the stage's own u and v
scale  = exp(value[2:4])
origin = lattice.origin + offset[0]*u + offset[1]*v
u, v   = u*scale[0], v*scale[1]
```

- **Declared parameter cost:** `blocks = 6` stages x `(4*4 + 4) = 20` = **120 new parameters**, 12 new
  tensors (`lattice_residual.{0..5}.{weight,bias}`), all exactly zero at initialization. That zero
  makes the residual the identity, so **step zero is exactly the published function**; the memory cost
  is under 1 KB above the published model, so the declared envelope is unchanged.
- **Why decoded space and not another pre-sigmoid nudge:** `box_refine` already nudges `raw_box` for two
  stages, and the sigmoid decode still ties origin to extent inside the legal range. The residual acts
  on the decoded basis in cell units and log-scale, so a stage can pin its own boundary while the
  shared box stays where the published head puts it, and it covers all six stages, not two.
- **What is not changed:** no new input, no change to the encoder, read-out, stage policy, objective or
  population. The published geometry term already supervises every stage's lattice, so the residual is
  supervised by the existing objective rather than by a new term - this is a capacity lever, not a
  supervision lever, which is what distinguishes it from the four refused geometry-side levers.

Everything else is the campaign's matched 256-step configuration: checkpoint `runs/m3-percell-balance-r1`
(`9ae1fef9f649...`), 256 continuation steps, read-out `occupancy_detail 1`, stencil `0`,
`occupancy_context 0`, `geometry_final_only 0`, `lattice_support 0`, `box_extent_weight 0.0`,
`support_anchor_weight 0.0`, factor `4`, positive weight `4`, `seed 20261026`, learning rate `.001`,
curriculum `stream-v1`, capture `512`, capacity `64`, positional identity read-out, length endpoint,
1,152 MiB, 1,800 s.

## Matched start: proven, not assumed

This lever **adds tensors**, so the shared contract uses its parameter-addition branch (V-071):
every published tensor must be bitwise identical between the arms, and the added tensors must be
**exactly the declared set** (`lattice_residual.{0..5}.weight`, `lattice_residual.{0..5}.bias`) and
**exactly zero** - anything else added, missing or non-zero refuses the curation. `step_zero_identical`
then compares the two arms' step-0 probe records with the objective's own value stripped (V-072), since
the model is the same function at step zero but the loss is not compared. A launch refused by the guard
is preserved under `runs/attempts/` with a note and re-recorded under a fresh number.

## Baseline control: the curated reference is a hard check

The baseline must reproduce the control curated by the four earlier passes:

| quantity | curated control |
|---|---|
| per-cell F1 at `0.0` | 0.4146485528647372 (351 TP / 675 FP / 316 FN) |
| pooled AUC | 0.8255386800822172 |
| best swept F1 / threshold | 0.4237196765498652 at `-0.30` |
| mean placement error | 1.9812198625 cells |
| extent ratios u / v | 1.2256880473012681 / 1.18035288054444 |
| per-board error spread | 4.604344325 cells |

Tolerance `1e-6`. Any disagreement is a **drift defect**: the curator refuses before computing a
lever attribution, and the pass reports a defect rather than a verdict.

## Envelope and declared work arithmetic

Per arm: **1,152 MiB** aggregate commit, **256 steps**, **1,800 s** runner wall, one heavy job at a
time; the 120 added parameters do not change the ceiling. An arm that cannot finish inside the
envelope is **NOT ADMISSIBLE AS DECLARED**; nothing is relaxed.

| launch | work items | what is counted |
|---|---|---|
| each profile | 2 | 2 optimizer updates |
| each study | **256** | 256 optimizer updates |
| each sweep profile | 2 | 2 forwards |
| each measured sweep | **54** | 36 deciding + 18 development forwards |
| each placement probe | **36** | 36 deciding forwards, 640 MiB / 300 s |
| curation | **8** | eight receipts read |

## Scoring population

Every known cell of every nonempty deciding board - 28 boards, 200 cells each, 5,600 known cells, 667
occupied and 4,933 empty, twelve ancestry groups - at threshold `0.0`, plus the standing 81-point grid
on `[-6, 6]` step 0.15. The placement instrument measures all 200 cells of the same boards. The oracle
pairing's co-scoreable subset is **not** a judgement population.

## The four declared conditions

**The lever CLEARS iff all four hold.** These are the campaign's standing criteria, unchanged, with the
geometry pass's own placement condition reused so all five geometry-side levers are comparable.

1. **Per-cell gate** - occupied-cell F1 at `0.0` on the full population `>= 0.70`.
2. **Ranking gate** - `sweep_verdict` is `informative` (AUC `>= 0.90` and some swept point `>= 0.70`).
3. **Attribution floor** - best swept F1 exceeds the matched baseline's by `>= 0.05`, with the
   whole-ancestry-cluster paired bootstrap (400 draws, seed `20261026`) reported beside it.
4. **Placement condition** - mean placement error over 200 cells and two axes must fall by `>= 25%`
   relative to the matched baseline (1.9812198625 -> at most 1.485914896875 cells). A lattice-capacity
   lever that does not move placement has not done what it was ranked to do, whatever the occupancy
   score does.

Otherwise the pass is **`not-cleared`** with the failing conditions named mechanically.

## Declared regression falsifiers

- **Placement unmoved** - improvement `< 5%`: per-stage box capacity did not act, so no occupancy
  change may be credited to it.
- **Placement regressed** - improvement `< 0`: the added capacity moved the box the wrong way; the
  write-up must say so and may not describe the residual as a refinement.
- **Extent or origin regress** - either extent ratio growing by more than 5%, or either signed offset
  growing by more than 1.0 cell, relative to the baseline.
- **Occupancy moved, placement not** - attribution `>= 0.05` with placement `< 5%`: the gain cannot be
  credited to the residual.

## Declared next-suspect rule (non-circular)

`improvement` is the declared placement reduction. First match wins; no branch can return `translation`,
`scale`, `geometry_capacity` or `placement_capacity` - those four levers are inputs, never outputs.

| rule | condition | next suspect |
|---|---|---|
| read_out_inheritance | `improvement < 0.05` (per-stage box capacity did not move placement either) | a lattice-agnostic occupancy read-out |
| dispersion | placement moved, but the lever's per-board error spread `>= 1.5` cells | per-board lattice refinement capacity |
| scaling | otherwise (placement moved, spread controlled, occupancy still short) | occupancy read-out at a second scale |

## Predeclaration ancestry

The curator resolves the commit that first added this file with `git log --diff-filter=A`, requires it to
be an ancestor of `HEAD`, and refuses unless **every** receipt it reads was written after that commit's
committer timestamp. Receipts are untracked artifacts, so time is the proof available; the rule is
stated here, implemented in the curator and exercised by a test.

## Reconciliation the receipt must carry

From the four earlier evidence files (reference only, never verdict input): geometry **+0.0781** at
19.9% placement cost, origin **+0.0102** at 11.0%, extent **+0.0318** at 67.5%, capacity **-0.0241** at
28.2%. The receipt states where this lever sits on that single control and whether per-stage decoded
capacity is a cheaper route to the occupancy score than any supervision-side change.

## Receipts this pass must produce

1. `runs/m3-occlatres-{lever,baseline}-r1-profile-r1` (2 steps, 1,152 MiB / 300 s / 2 work items).
2. `runs/m3-occlatres-{lever,baseline}-r1` (256 steps, 1,152 MiB / 1,800 s / 256 work items), each
   admitted only by its own profile.
3. `...-sweep-r1-profile-r1` (2 forwards) and `...-sweep-r1` (54 forwards) per arm, 640 MiB.
4. `...-probe-r1` per arm (36 forwards, 640 MiB / 300 s).
5. Curation `runs/m3-occlatres-curation-r1-job.json` into
   `docs/evidence/m3-occupancy-lattice-residual-r1/scope.json`, re-deriving every published number and
   asserting each probe reproduces its own sweep's per-cell F1 (1e-6) and TP/FP/FN counts.
6. The owned family `m3-occlatres-*` declared in `record_stream_run.OWNED_RUN_PATTERN`, bound by a test.

Any launch that dies, leaves its declared arithmetic or is superseded is preserved under
`runs/attempts/` with a written note and re-recorded under a fresh number; none is evidence.

## Limits, stated before the measurement

- Synthetic, training-fit population; nine of twelve groups never appear in the run's own ledger. No
  real-screenshot, generalization, calibration or complete-state claim follows.
- One budget point (256 steps), one checkpoint, one residual scale (`0.25`), one seed, committed lattice
  only.
- The residual is an axis-aligned correction in cell space: it cannot express rotation, shear or a
  non-rectangular board, and the 120 parameters are a declared cost that the occupancy score must repay.
- Placement is measured against the committed target lattice, so it says nothing about deployment
  situations where that lattice is unavailable.

---

# Measured (2026-11-10)

Both arms ran the declared 256 continuation steps from the published checkpoint
`9ae1fef9f649...` under one captured catalogue (`242e8c767c9f...`), inside the declared 1,152 MiB /
1,800 s envelope: lever **1,059,573,760 B / 826.1 s**, baseline **1,056,604,160 B / 1,017.6 s**.
Admission profiles passed at 77.4 s / 54.2 s (peaks 1,041,580,032 B and 1,041,199,104 B); the sweeps
(54 forwards, 640 MiB) took 60.6 s / 69.7 s and the placement probes (36 forwards, 640 MiB / 300 s)
42.8 s / 47.5 s, peaking at 561,954,816 B and 556,920,832 B. Curation
`runs/m3-occlatres-curation-r1-job.json` passed and published
`docs/evidence/m3-occupancy-lattice-residual-r1/scope.json`.

**Matched start.** The parameter-addition branch passed: the shared initial tensors are bitwise
identical (`f426e4061d60...`), the two saved states differ by exactly the twelve declared
`lattice_residual.{0..5}.{weight,bias}` tensors, all exactly zero, and the step-0 probes agree after the
objective's own value is stripped. The baseline's weights (`d149351cbd...`) and the lever's
(`6dfb6be736c9...`) differ only through the 256 steps.

**Verdict: `not-cleared`, failing all four declared conditions.** Per-cell F1 **0.3520**
(661 TP / 2,428 FP / 6 FN) against **0.4146** (351/675/316); AUC **0.8209** against **0.8255**
(-0.0046); best swept F1 **0.4122** at `2.25` against **0.4237** at `-0.30`, attribution **-0.0115**,
with the twelve-cluster paired bootstrap median **-0.0954** and interval **-0.2795 to +0.0300**.
Placement improved **-19.2%** (1.9812 -> 2.3615 cells): condition 4 fails in the wrong direction, so the
declared **placement-regressed** falsifier holds and this residual may not be described as a refinement
of the box.

**The residual moved the lattice decisively, and the move was the wrong one.** The column extent ratio
**shrank** 1.2257 -> **0.9084** while rows went 1.1804 -> 1.0781; signed offsets +1.3830 -> **+1.5856**
and -2.1914 -> **-2.9092**; spread 4.6043 -> **4.8346**, shifted share 0.5375 -> **0.7093** and mapped
share 0.6507 -> **0.7243**. So 71% of cells were shifted rather than the 54% the baseline shifted, and
the occupancy head answered by predicting every cell occupied: 661 of 667 occupied cells found with
only 6 misses, at 2,428 false positives and recall **0.991** at threshold `0.0`. The per-stage capacity
did not fail to act - it pushed the read-out into an almost all-positive regime, which is why the
per-cell gate and the ranking gate fail while AUC stays near the baseline.

**What the capacity reading means.** Six stages each given an independently correctable decoded box,
zero-initialized and supervised by the existing geometry term, produced the second-worst occupancy
result of the five geometry-side levers and the second-worst placement. Per-stage box capacity is
therefore not the binding constraint on the occupancy score; it is a channel through which the lattice
can move without the score following. The rule's first branch names the read-out, because placement
moved by 19.2% and occupancy did not follow it.

**Reconciliation.** Five geometry-side levers now sit on the same control: geometry **+0.0781** at
**19.9%** placement cost, origin **+0.0102** at **11.0%**, extent **+0.0318** at **67.5%**, capacity
**-0.0241** at **28.2%**, residual **-0.0115** at **19.2%**. No lever cleared the four conditions; the
two supervision-side gains were bought with placement, and the two capacity-side levers lost both.

| lever | attribution | placement cost | verdict |
|---|---|---|---|
| geometry stage | +0.0781 | 19.9% | not-cleared |
| box origin | +0.0102 | 11.0% | not-cleared |
| box extent | +0.0318 | 67.5% | not-cleared |
| geometry capacity | -0.0241 | 28.2% | not-cleared |
| per-stage residual | -0.0115 | 19.2% | not-cleared |

**Ranked next.** The declared non-circular rule takes its first branch - improvement **-0.192 < 0.05** -
and names **a lattice-agnostic occupancy read-out**. **Ranked next: predeclare a lattice-agnostic
read-out lever**, read with the same four conditions; dispersion (spread 4.8346, below the 1.5-cell
floor) and scaling stay behind it, and `translation`, `scale`, `geometry_capacity` and
`placement_capacity` cannot be returned by the rule.

**Defects and preserved launches.** The baseline study was killed at step 247 of 256 when the host went
offline; it is preserved with its receipt and log as `runs/attempts/m3-occlatres-baseline-r1.attempt1`
and re-recorded at the same declared name, admitted by the same passed profile. No measurement was taken
from the killed launch. Two defects in this pass's own files are logged as V-075 (the owned-run regex
replacement dropped the previously owned family, caught by the capacity lever's own test) and V-076 (the
new rule recomputed the placement reduction instead of using the declared decision value, caught by its
own test).

**Unchanged.** No gate widened or retuned, no residual scale or parameter count changed after the
receipts, no prior evidence or frozen manifest touched. Limits are as declared: synthetic training-fit,
one budget point, one checkpoint, one residual scale, one seed, committed lattice only.
