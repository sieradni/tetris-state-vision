# Occupancy registration lever: a bounded per-cell field on the detail tap's sample

Status: **predeclared (2026-11-12) and committed before any profile, study, sweep or evidence
directory for this instrument exists.** The change, the control, the envelope, the gates, the floor,
the population and the work-item arithmetic below are fixed; the measured section is appended only
after the canonical receipts exist.

## Where this comes from

The campaign's largest measured residual is **placement**, not information. At 4x budget, pairing
each model cell with the label of the target cell it physically covers raised occupied-cell F1
`0.4390 -> 0.5742` (+0.1352, AUC 0.9331 -> 0.9702) with a mean placement offset of 2.57 cells and
48.3% of model cells covering the wrong target cell
([M3_OCCUPANCY_SCALED.md](M3_OCCUPANCY_SCALED.md)). Independently, a logistic read-out over 30
per-cell pixel features that never sees the model recovers the targets at F1 0.978 under oracle
registration ([M3_OCCUPANCY_PIXEL_CEILING.md](M3_OCCUPANCY_PIXEL_CEILING.md)). So the information is
present at the cell and the failure is alignment.

Every geometry-side lever so far corrected the **lattice**: the shared box origin, the box extent, a
per-stage residual, a board-conditioned four-vector, boundary support anchors. Each moved placement
and occupancy against each other, and the board-conditioned correction never engaged at all
(0.0014 cells). A global correction has to be right for every cell at once; a per-cell error field
does not.

## The single declared change

`ModelConfig.occupancy_deform` (default `0`, published). It requires `occupancy_detail == 1` because
it moves the detail tap's sample, and it is declared at `occupancy_detail_stencil == 0`,
`occupancy_resolution == 0` and `occupancy_context == 0` so the pass compares registration with
registration, not with sampling geometry or a second residual.

When `1`, a **per-cell deformation field** is predicted from the same first-stage feature the tap
reads and applied to that cell's tap sample:

```
field     = Conv2d(widths[0] -> 2, 3x3, padding 1)(first)          # its own zero-init conv
shift     = sample(field at the decoded cell centres)              # 2 numbers per cell
offset    = clamp_tanh(shift) * 1.0                                # bounded in cell units
centers   = sample(first at cell centre + offset)                  # bilinear, differentiable
occupancy = cells + W_d . centers,     W_d = 0 at step zero
```

**Why this mechanism.** A per-cell field can express local misalignment that a single rigid box
correction cannot, while the bound of one cell keeps every sample inside the neighbourhood of the cell
it decides, so it cannot turn into a re-detection of a different cell; and because the field is a
plain convolution with bilinear sampling it is trained by the existing occupancy term with no new
supervision and no new loss.

The field's convolution is **zero-initialized**, so at step zero every offset is exactly zero, the
sample points are exactly the published ones and both arms compute the same function. The 2x3x3x48
= 866 weights and 2 biases exist only on the levered arm, so they are the single difference; the
projection `W_d` is the same module with the same zero initialization in both arms. Constructed last
in `__init__`, so every published tensor keeps the seeded value it has in the control.

Three declared consequences, bound by `tests/test_occupancy_deform.py`:

1. Both arms compute **exactly** the published function at step zero.
2. Every shared tensor is drawn from the same seeded initialization and loaded from the same
   checkpoint bytes; the only added keys are `occupancy_deform.*`.
3. With the projection at zero, any field weights whatsoever give the same image-supported output, so
   a hidden alignment state the pixels cannot distinguish cannot change the decision.

The declared bound: `DEFORM_LIMIT = 1.0` cell per axis, applied as `tanh(x) * 1.0`, so the field can
never leave `(-1, 1)` cells and cannot sample a different cell's interior at typical cell sizes.

## Matched pair, one budget, one catalogue

Both arms are **continuations** of the committed published checkpoint `runs/m3-percell-balance-r1`
(weights `9ae1fef9f649...`) under the same captured source catalogue:

| arm | `occupancy_detail` | `occupancy_deform` | `occupancy_resolution` | `occupancy_detail_stencil` | `occupancy_context` |
|---|---|---|---|---|---|
| lever | `1` | `1` | `0` | `0` | `0` |
| control | `1` | `0` | `0` | `0` | `0` |

Everything else identical: same initial weights, `seed 20261026`, `learning_rate .001`, curriculum
`stream-v1`, capture side `512`, component capacity `64` with the positional identity read-out and
length endpoint, occupancy loss factor `4` and positive weight `4`, **64 further steps**, the same
image order, gates, floor and population.

## Envelope, declared before admission

| quantity | value |
|---|---|
| aggregate commit ceiling, per arm | **1,879,048,192 B (1,792 MiB)** |
| steps per arm | 64 |
| runner wall / trainer wall | 1,200 s / 900 s |
| sweep envelope | 640 MiB, 54 forwards |
| heavy jobs | one at a time |

The ceiling is the one the resolution pass measured: its control peaked at 1,061,023,744 B and its
stride-1 branch at 1,500,233,728 B, and this field is a small convolution plus 200 sampled points, so
the admitted envelope is expected to cost little above the control and to sit far under 1,792 MiB.
Refusing on a tighter self-chosen ceiling is what left this question unmeasured once already.

**Declared refusal rule:** if either arm cannot be admitted inside 1,792 MiB at the matched budget,
the outcome is **NOT ADMISSIBLE AS DECLARED**, recorded with the measured peak and the receipt; no
ceiling is raised, no smaller variant is substituted, and no run is repeated to pass.

## Declared work-item arithmetic

| launch | work items | what is counted |
|---|---|---|
| each profile | 2 | 2 optimizer updates |
| each study | 64 | 64 optimizer updates (`steps`); evaluations are not work items |
| each sweep profile | 2 | 2 forwards |
| each measured sweep | **54** | 54 forwards = 36 deciding + 18 development |
| curation | **4** | four receipts read (two studies, two sweeps) |

The launch, the receipt and the curator are bound to these numbers: the curator refuses unless
`job.work_items == result.forwards == 54`, the deciding population is exactly 36 images / 28 nonempty
boards / 667 occupied / 4,933 empty known cells / twelve ancestry clusters, and the sweep evaluated
twelve groups.

## Scoring, on the standing widened gates

Each arm is scored read-only on the **widened deciding population** with the standing 81-point grid
on `[-6, 6]` step 0.15, under its own 640 MiB sweep profile. Population, grid and thresholds are
unchanged from every pass in this neighborhood.

**The lever CLEARS iff all three hold:**

1. **Per-cell gate** - occupied-cell F1 at the replay's operating point (`0.0`) `>= 0.70`.
2. **Ranking gate** - `sweep_verdict` is `informative` (pooled AUC `>= 0.90` **and** some swept
   threshold reaches F1 `>= 0.70`).
3. **Attribution floor** - lever best swept F1 exceeds control best swept F1 by **`>= 0.05`**.

**Otherwise NOT CLEARED**, with the failing condition named mechanically.

The calibration-neutral report of [M3_MINIMUM_DETECTABLE_EFFECT.md](M3_MINIMUM_DETECTABLE_EFFECT.md)
is reported beside the verdict and read by nothing: ranking quality at the arm's own best boundary
(no interval, because that boundary is fitted on the cells it scores), the fixed-boundary component
with its whole-cluster interval and minimum detectable effect, and the out-of-fold component with its
own, plus the zero-boundary cost, transfer cost and precision at `0.0`.

## Receipts this pass must produce

1. `runs/m3-occdeform-lever-r1-profile-r1-job.json` and `runs/m3-occdeform-control-r1-profile-r1-job.json`,
   2 steps, 1,792 MiB / 300 s / 2 work items.
2. `runs/m3-occdeform-lever-r1-job.json` and `runs/m3-occdeform-control-r1-job.json`, 64 steps,
   1,792 MiB / 1,200 s / 64 work items, each admitted only by its own profile.
3. `...-sweep-r1-profile-r1` (2 forwards) and `...-sweep-r1` (54 forwards) per arm, 640 MiB.
4. Curation `runs/m3-occdeform-curation-r1-job.json` into
   `docs/evidence/m3-occupancy-deformation-r1/scope.json`, re-deriving every published number.
5. The owned family `m3-occdeform-*` declared in `record_stream_run.OWNED_RUN_PATTERN` and bound by
   `tests/test_occupancy_deform.py`.

## What each outcome licenses

- **Cleared** - local misalignment was the constraint; the occupancy question moves to scale.
- **Not cleared, attribution met and the fixed-boundary component resolvable** - the mechanism
  registers cells and the level is still far from fit; the next pass is the same mechanism at a
  matched multiple of this budget, priced by the power document.
- **Not cleared, field inert** - record the applied offset magnitude; an inert field means the
  occupancy term does not find per-cell alignment useful at this budget, and the registration residual
  must be attacked through supervision (a placed target), not through capacity.
- **Not cleared, field active and worse** - the field is fitting noise; report the shift the way the
  box levers were reported, with placement, and close the deformation family with that number.
- **Not admissible** - the constraint is the measured result, recorded with its peak.
- **Unchanged either way** - the fit gate stays FALSE unless 3 of 28 exact nonempty boards appear, the
  identity heads are untouched, and no real-image, complete-state or quality claim follows.

## Audit

The recorder re-derives every number from the four receipts, refuses a mismatched pair (different
catalogue, different start bytes, identical logits), asserts the deciding arithmetic (28 boards, 12
clusters, 667 occupied) before writing, and refuses to publish unless each arm's component block
equals the sweep's own deciding rows. It is a copy of the resolution pass's instrument with one
declared difference - the field instead of the stride-1 branch - so the two passes' numbers are
directly comparable as unmatched references.

## Outcome (2026-11-12) - measured: NOT CLEARED, attribution met, and the first lever to move ranking toward the oracle-registration level

Both arms were admitted inside the declared envelope, completed 64 steps under one catalogue from the
published checkpoint bytes, and were scored read-only on the widened deciding population (36 images,
28 nonempty boards, 667 occupied, 4,933 empty known cells, twelve ancestry clusters) with the declared
grid and gates. Evidence: `docs/evidence/m3-occupancy-deformation-r1/scope.json`, curated by
`runs/m3-occdeform-curation-r2-job.json` (passed, 11.1 s, peak 38,891,520 B), every number re-derived
from the four receipts.

| quantity | lever (field) | control (detail, no field) |
|---|---|---|
| deciding AUC | **0.866566** | 0.822643 |
| F1 at `0.0` | 0.369420 | **0.381640** |
| best swept F1 | **0.490752** at `3.15` | 0.412034 at `0.90` |
| fixed-boundary precision | 0.226945 | 0.236325 |
| best-boundary precision | **0.416754** | 0.263488 |
| out-of-fold F1 | **0.456054** | 0.400399 |
| zero-boundary cost | -0.121332 | -0.030394 |
| transfer cost | 0.086634 | 0.018759 |
| exact nonempty boards | 0 of 28 | 0 of 28 |
| study peak / wall | 1,090,600,960 B / 142.5 s | 1,058,295,808 B / 246.7 s |
| sweep peak | 563,838,976 B | 562,388,992 B |

The field costs **32,305,152 B** (30.8 MiB) above the control at 64 steps, so the mechanism was admitted
with **751.9 MiB** to spare: the refusal this declaration was written to avoid did not occur.

**Verdict `not-cleared`**: the per-cell gate (`0.369420` < `0.70`) and the ranking gate (pooled AUC
`0.866566` < `0.90`; best F1 `0.490752` < `0.70`) both fail, and the attribution floor is **met**: best
swept F1 **+0.078718** against a `0.05` floor. The field re-scaled the head: the best boundary moved
from `0.90` to `3.15`, and the out-of-fold boundaries chosen on the other eleven clusters sat at `3.15`
in ten of twelve groups (the control's sat at `0.90` in ten), so the shift is a consistent scale change,
not a fitted artefact of one group.

### What this design can and cannot resolve

| reading | delta | interval (12 clusters, 400 draws) | MDE at 80% | resolvable |
|---|---|---|---|---|
| ranking, pooled AUC | **+0.043922** | +0.017963 .. +0.063903 | 0.032257 | **yes** |
| fixed-boundary F1 | -0.012220 | -0.021661 .. -0.003297 | 0.013660 | **yes** |
| out-of-fold F1 | +0.044328 | -0.105165 .. +0.290645 | 0.315147 | **no** |
| fixed-boundary precision | -0.009380 | -0.020459 .. -0.001968 | 0.013542 | **yes** |
| best-boundary precision | +0.153266 | none (boundary fitted on the scored cells) | - | reported, never read |
| best swept F1 | +0.078718 | none (same reason) | - | attribution only |

- **Resolvable**: the field **improves ranking** by `+0.044` AUC with an interval clear of zero, and
  costs `-0.012` F1 and `-0.009` precision at the fixed `0.0` boundary with intervals also clear of
  zero. Both are real at this design.
- **Not resolvable**: how much of the `+0.044` out-of-fold F1 transfers. Its interval spans
  -0.105 .. +0.291 and its MDE is `0.315`, so this block cannot confirm or deny transfer; the
  fixed-boundary read is the only boundary-neutral component this population can resolve.
- **Never read**: the best swept F1 and the best-boundary precision, because both boundaries are fitted
  on the cells they score. The `+0.078718` attribution is a consequence of the ranking gain, not an
  independent confirmation of it.

### Where this leaves the registration residual

The control's best swept F1 is `0.412034` and oracle registration was worth `0.5742`; the lever closes
`0.078718` of that `0.162166` distance, i.e. **48.5%** of the gap to the oracle-registration level at
the same 64-step budget, and 58% of the `+0.1352` the oracle pairing was worth at 4x. That is measured
progress on the campaign's largest attributed residual, from a local mechanism with 866 added weights,
and it is the largest attribution any read-out lever in this family has measured.

It is also **not** the fix. The level is still `0.0834` best F1 and `0.2048` F1-at-`0.0` below oracle
registration, no board is exact, and the lever's own success created the next obstacle: its ranking is
better but its **zero-boundary cost quadrupled** (`-0.030394` -> `-0.121332`) because the field's logit
scale no longer matches the replay point. Four of the five resolvable readings say the same thing -
the deformed sample separates cells better and the fixed decision boundary reads it worse.

**Next strongest lever, honestly**: read the deformation arm at a calibrated boundary instead of a hard
`0.0` (the standing V-091/V-095 rule applied to the arm that moves the scale most), then repeat this
exact matched pair at a matched multiple of the budget. Both are cheap and attributable; a wider trunk
is not indicated by anything measured here, and the out-of-fold component will stay unresolvable until
the population has more than twelve ancestry clusters.

### Fit gate

**Unchanged: FALSE.** 0 of 28 exact nonempty boards on both arms, identity heads untouched, no
real-image, complete-state, calibrated or quality claim. A `+0.078718` best-F1 attribution on a
population where the best arm still emits every cell wrong on every board is progress in ranking, not
fit.

### Defects found in this pass's own record

- **V-103**: the first canonical curation read its mechanism precision at the fixed `0.0` boundary
  that this lever moves (`mechanism.precision_improved = false` from `-0.009380`), the V-091 pattern
  reintroduced through a helper written for levers that do not re-scale logits. The defective scope and
  receipt are preserved unedited under `runs/attempts/m3-occdeform-curation-r1-defective/`; the read
  now happens at each arm's own boundary, with the fixed-boundary delta kept beside it as a labelled
  cost and its interval asserted against the statistic it names. The published scope is the r2
  curation.
- **A killed supervisor, preserved**: the control study's first launch was killed with its shell at
  the host's 295 s command cap, leaving a `running` receipt and a partial output. Both are preserved
  under `runs/attempts/m3-occdeform-control-killed-r1/` and the arm was re-run under the declared name;
  the published control receipt is the re-run.
