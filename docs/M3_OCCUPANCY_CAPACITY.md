# Geometry-capacity lever — spend geometry supervision on the anchors the box is pinned by

Status: predeclared (2026-11-10) and committed before any profile, study, sweep, probe or evidence
directory for this instrument exists. The change, its matched start, the envelope, the population, the
four conditions, the falsifiers, the control-drift rule, the predeclaration-ancestry rule and the
conditional next-suspect rule below are fixed; the measured section is appended only after the
canonical receipts exist.

## Where this comes from

Three occupancy levers have been spent and refused, each naming the next by a rule fixed in advance:

| pass | verdict | occupancy attribution | placement cost | what it moved |
|---|---|---|---|---|
| geometry stage ([M3_OCCUPANCY_GEOMETRY.md](M3_OCCUPANCY_GEOMETRY.md)) | `not-cleared` | +0.0781 (-0.0152..+0.1243) | -19.9% (1.9812 -> 2.3757) | read-out absorbed the offset |
| box origin ([M3_OCCUPANCY_TRANSLATION.md](M3_OCCUPANCY_TRANSLATION.md)) | `not-cleared` | +0.0102 (-0.1700..+0.0872) | -11.0% (1.9812 -> 2.1988) | origin drifted outward |
| box extent ([M3_OCCUPANCY_EXTENT.md](M3_OCCUPANCY_EXTENT.md)) | `not-cleared` | +0.0318 (-0.1345..-0.0313) | -67.5% (1.9812 -> 3.3182) | column extent inflated |

The extent pass's declared rule took its first branch - smallest extent reduction `-1.075 < 0.25` - and
named **geometric supervision capacity (support anchors)**. The same name is the geometry pass's own
last branch. Nothing about it has been measured yet.

The premises are in the published code and are checkable:

1. `losses.geometry_error` supervises **204** points: the 200 cell centers plus the four board corners.
   `stage_loss` takes their plain mean, so the boundary carries **4/204 = 2%** of the geometry term,
   and `joint_loss` averages the term over six decoder stages.
2. The occupancy head samples its features at the model's **own predicted lattice**, so the box is the
   geometric object every occupancy cell index depends on. It is pinned by 4 parameters produced from
   one global token, and the only supervision that constrains its boundary is that 2% share.

So the campaign's supervision spends almost all of its geometry weight where the box is least
identifiable - interior centers, which move with the box rather than pin it - and almost none on the
eight boundary points that determine where the box's corners and edges actually fall.

## The single declared change

**A boundary support-anchor geometry term on the final lattice.**

| arm | `support_anchor_weight` | meaning |
|---|---|---|
| lever | `1.0` | the published objective plus `1.0 * masked_mean(smooth_l1(anchor_error), registered)`, evaluated on the **final** decoder lattice |
| baseline | `0.0` | the published objective, byte-identical to today |

The term is the mean smooth-L1 error of **eight declared support anchors** in cell coordinates -

```
(0,0)  (5,0)  (10,0)  (0,10)  (10,10)  (0,20)  (5,20)  (10,20)
```

- the four board corners and the four edge midpoints - each mapped through the predicted lattice and
  through the target lattice, differenced, and divided by the target basis norms per axis, so the error
  is in the same normalized units as the published geometry term. Weight `1.0` is declared as *equal to
  the published geometry term's own weight*, not tuned; `0.0` and `1.0` is the whole declared set.

Why this is a **capacity** lever and not another origin or extent lever: it adds no parameter and
changes neither the box's position nor its scale directly. It changes **which points the geometry
gradient is spent on**, raising the boundary's share of the geometric supervision from 4/204 of one
term to a full-weight term of its own. The origin and extent levers both moved the four box parameters
without ever making the boundary itself the object of supervision; if the box cannot hold its support
anchors, neither could have worked.

Everything else is the campaign's matched 256-step configuration: the same published checkpoint
`runs/m3-percell-balance-r1` (`9ae1fef9f649...`), 256 continuation steps, the published center-sample
read-out (`occupancy_detail 1`, stencil `0`), `occupancy_context 0`, published geometry stage policy
(`geometry_final_only 0`), published box head (`lattice_support 0`), published objective
(`box_extent_weight 0.0`), loss factor `4`, positive weight `4`, `seed 20261026`, learning rate `.001`,
curriculum `stream-v1`, capture `512`, capacity `64`, positional identity read-out, length endpoint,
1,152 MiB, 1,800 s. Both arms run under this pass's own captured catalogue.

## Matched start: proven, not assumed

The lever adds no parameter, so the shared contract `scripts/matched_start.py` (V-071) is used on its
no-addition branch: every tensor in both saved initial states must be bitwise identical, and
`step_zero_identical` must find the two arms' step-0 probe records equal **after the objective's own
value is stripped** (V-072). The stripped fields are exactly `loss` and `cases[*].loss`, which this
lever changes at step zero by construction; every prediction field must match. A launch refused by the
guard is preserved under `runs/attempts/` with a note and re-recorded under a fresh number.

## Baseline control: the curated reference is a hard check

The baseline must reproduce the control curated by the three earlier passes:

| quantity | curated control |
|---|---|
| per-cell F1 at `0.0` | 0.4146485528647372 (351 TP / 675 FP / 316 FN) |
| pooled AUC | 0.8255386800822172 |
| best swept F1 / threshold | 0.4237196765498652 at `-0.30` |
| mean placement error | 1.9812198625 cells |
| extent ratios u / v | 1.2256880473012681 / 1.18035288054444 |
| per-board error spread | 4.604344325 cells |

Tolerance `1e-6`. Any disagreement is a **drift defect**: the curator refuses before computing a
lever attribution, and the pass reports a defect rather than a verdict until it is resolved.

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
occupied and 4,933 empty, twelve ancestry groups - at threshold `0.0`, plus the standing 81-point grid
on `[-6, 6]` step 0.15 for the best swept F1. The placement instrument measures all 200 cells of the
same boards. The oracle pairing's co-scoreable subset is **not** a judgement population.

## The four declared conditions

**The lever CLEARS iff all four hold.** The first three are the campaign's standing criteria, unchanged
since the pixel-ceiling pass; the fourth is the geometry pass's own placement condition, declared there
before its measurement and reused here so all four geometry-side levers are directly comparable.

1. **Per-cell gate** - occupied-cell F1 at `0.0` on the full population `>= 0.70`.
2. **Ranking gate** - `sweep_verdict` is `informative` (AUC `>= 0.90` and some swept point `>= 0.70`).
3. **Attribution floor** - best swept F1 exceeds the matched baseline's by `>= 0.05`, with the
   whole-ancestry-cluster paired bootstrap (400 draws, seed `20261026`) reported beside it; the point
   delta carries the floor, as in every preceding pass.
4. **Placement condition** - mean placement error over 200 cells and two axes must fall by `>= 25%`
   relative to the matched baseline (1.9812198625 -> at most 1.485914896875 cells). A geometry-capacity
   lever that does not move placement has not done what it was ranked to do, whatever the occupancy
   score does.

Otherwise the pass is **`not-cleared`** with the failing conditions named mechanically.

## Falsifiers, stated before the measurement

- **Placement unmoved** - if placement improves by `< 5%`, then boundary supervision did not act on the
  box, whatever occupancy does, and the capacity reading is falsified.
- **Anchors moved, occupancy not attributed** - if placement improves by `>= 25%` and attribution is
  `< 0.05`, then the box moved without the occupancy score following, which falsifies "boundary
  supervision capacity is what limits the occupancy score".
- **Occupancy moved, placement not** - if attribution is `>= 0.05` and placement improves by `< 5%`,
  the gain cannot be credited to the anchors and must be attributed to the operating point.
- **Extent or origin regress** - if either extent ratio grows by more than 5% or either signed offset
  grows by more than 1.0 cell relative to the baseline, the term moved the box the wrong way and the
  write-up may not describe it as boundary pinning.

## Declared next-suspect rule (non-circular)

`improvement` is the declared placement reduction. First match wins, and no branch can return
`translation`, `scale` or `geometry_capacity` - those three levers have been run and refused, so they
are inputs to this rule, never outputs.

| rule | condition | next suspect |
|---|---|---|
| placement_capacity | `improvement < 0.05` (boundary supervision did not move the box) | per-stage lattice residual capacity |
| read_out_inheritance | placement improved `>= 25%` and the per-cell/ranking gates did not follow | a lattice-agnostic occupancy read-out |
| dispersion | the lever's per-board error spread `>= 1.5` cells | per-board lattice refinement capacity |
| scaling | otherwise (placement moved, occupancy ranked, attribution short) | occupancy read-out at a second scale |

## Predeclaration ancestry

The curator resolves the commit that first added this file with `git log --diff-filter=A`, requires it
to be an ancestor of `HEAD`, and refuses the curation unless **every** receipt it reads was written after
that commit's committer timestamp. Receipts are untracked artifacts, so time is the proof available; the
rule is stated here, implemented in the curator and exercised by a test.

## Reconciliation the receipt must carry

From the three earlier evidence files (reference only, never verdict input): the geometry lever's
occupancy attribution and placement cost, the origin lever's both, and the extent lever's both. The
receipt states plainly whether boundary-capacity supervision repaid any of the geometry lever's +0.0781
occupancy gain, and at what placement cost relative to the recorded 19.9%, 11.0% and 67.5%.

## Receipts this pass must produce

1. `runs/m3-occgeocap-{lever,baseline}-r1-profile-r1` (2 steps, 1,152 MiB / 300 s / 2 work items).
2. `runs/m3-occgeocap-{lever,baseline}-r1` (256 steps, 1,152 MiB / 1,800 s / 256 work items), each
   admitted only by its own profile.
3. `...-sweep-r1-profile-r1` (2 forwards) and `...-sweep-r1` (54 forwards) per arm, 640 MiB.
4. `...-probe-r1` per arm (36 forwards, 640 MiB / 300 s).
5. Curation `runs/m3-occgeocap-curation-r1-job.json` into
   `docs/evidence/m3-occupancy-geometry-capacity-r1/scope.json`, re-deriving every published number and
   asserting each probe reproduces its own sweep's per-cell F1 (1e-6) and TP/FP/FN counts.
6. The owned family `m3-occgeocap-*` declared in `record_stream_run.OWNED_RUN_PATTERN`, bound by a test.

Any launch that dies, leaves its declared arithmetic or is superseded is preserved under
`runs/attempts/` with a written note and re-recorded under a fresh number; none is evidence.

## Limits, stated before the measurement

- Synthetic, training-fit population; nine of twelve groups never appear in the run's own ledger. No
  real-screenshot, generalization, calibration or complete-state claim follows.
- One budget point (256 steps), one checkpoint, one anchor set, one declared weight, one seed.
- The anchor set is eight boundary points at the board's own scale; it cannot supervise a boundary the
  board does not have, and it says nothing about rotation or shear.
- Placement is measured against the committed target lattice, so it says nothing about deployment
  situations where that lattice is unavailable.
