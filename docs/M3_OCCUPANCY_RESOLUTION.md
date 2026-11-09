# Occupancy resolution lever — declared stride-1 sample, matched continuation control

Status: predeclared (2026-11-09) and committed before any profile, study, sweep or evidence directory
for this instrument exists. The change, the control, the envelope, the gates, the floor, the
population and the work-item arithmetic below are fixed; the measured section is appended only after
the canonical receipts exist.

## Where this comes from

`M3_OCCUPANCY_RANKING.md` ranks one step after the stencil result
([M3_OCCUPANCY_STENCIL.md](M3_OCCUPANCY_STENCIL.md)): *move upstream to encoder resolution — a finer
native tap than stride-2*. The three read-out levers so far converted extra capacity into ordering
(AUC 0.7427 -> 0.8244 -> 0.8226 -> 0.8804) while the swept optimum stayed 0.21-0.29 short of the 0.70
bar, so the open question is whether the information the tap reads is sampled too coarsely to
separate cells, rather than the read-out being too small.

## The single declared change

`ModelConfig.occupancy_resolution` (default `0`, published). It is a declared *source* for the
existing detail tap, not a new read-out: it requires `occupancy_detail == 1`, and is mutually
exclusive with the declared stencil form (`occupancy_detail_stencil == 0`) so the pass compares
resolution with resolution, not with sampling geometry.

When `1`, a **stride-1 convolution of the image** supplies the tap's sample in place of the encoder's
stride-2 first-stage feature:

```
fine      = Conv2d(3 -> widths[0], 3x3, padding 1)((2*pixels-1)*valid)     # stride 1
sample    = sample(stride-2 first stage) + sample(fine)                    # same cell centers
occupancy = cells + W_d . sample,        W_d = 0 at step zero
```

The projection is the same module, same shape (48 -> 192) and the same zero initialization in both
arms; the only new tensors are the branch's 3 x 48 x 3 x 3 = 1,296 weights and 48 bias. The branch is
**not** zero-initialized — a zero branch would receive no gradient through the zero projection and
could never train — but the projection is, so step zero is exactly the published function in both
arms. The branch exists only on the levered arm, so it is the single difference.

Three declared consequences, bound by `tests/test_occupancy_resolution.py`:

1. Both arms compute **exactly** the published function at step zero.
2. Every shared tensor is drawn from the same seeded initialization and loaded from the same
   checkpoint bytes; the only added keys are `occupancy_resolution_branch.*`.
3. With the projection at zero, any branch weights whatsoever give the same image-supported output,
   so a hidden branch state the pixels cannot distinguish cannot change the decision.

What this does and does not change: the encoder backbone, decoder, losses and every other lever stay
exactly as published. This pass answers one question — does sampling the cell at pixel resolution
instead of stride-2 resolution separate occupied from empty cells better at this budget?

## Matched pair, one budget, one catalogue

Both arms are **continuations** of the committed published checkpoint `runs/m3-percell-balance-r1`
(weights `9ae1fef9f649...`) under the same captured source catalogue:

| arm | `occupancy_detail` | `occupancy_detail_stencil` | `occupancy_resolution` |
|---|---|---|---|
| lever | `1` | `0` | `1` |
| control | `1` | `0` | `0` |

Everything else identical: same initial weights, `seed 20261026`, `learning_rate .001`, curriculum
`stream-v1`, capture side `512`, component capacity `64` with the positional identity read-out and
length endpoint, occupancy loss factor `4` and positive weight `4`, **64 further steps**, the same
image order, gates, floor and population.

The control is therefore the same configuration as the stencil pass's control, so its numbers also
serve as an independent repetition check of that baseline; the stencil arm itself (catalogue
`27a0e917`) is reported as an unmatched **reference** and cannot enter this verdict.

Envelope per arm, declared before admission: **1,152 MiB** aggregate commit, **64 steps**, **1,200 s**
runner wall / 900 s trainer wall, one heavy job at a time. The branch adds 1,344 parameters and, at capture side 512, four retained full-resolution 48-channel
feature maps - input, output, grad output and grad input - plus the masked 3-channel image, a census of
204,472,320 B, so the 1,152 MiB ceiling is not certain to hold. **Declared refusal rule:** if either arm cannot be admitted inside 1,152 MiB at
the matched budget, the outcome is **NOT ADMISSIBLE AS DECLARED**, recorded with the measured peak and
the receipt; no ceiling is raised, no smaller variant is substituted, and no run is repeated to pass.

## Declared work-item arithmetic

Declared here because V-061 exists for exactly this: a receipt must certify the work it performed.

| launch | work items | what is counted |
|---|---|---|
| each profile | 2 | 2 optimizer updates |
| each study | 64 | 64 optimizer updates (`steps`); evaluations are not work items |
| each sweep profile | 2 | 2 forwards |
| each measured sweep | **54** | 54 forwards = 36 deciding + 18 development |
| curation | **4** | four receipts read (two studies, two sweeps) |

The launch, the receipt and the curator are bound to these numbers: the curator refuses unless
`job.work_items == result.forwards == 54`, the deciding population is exactly 36 images / 28 nonempty
boards / 667 occupied / 4,933 empty cells / 12 ancestry clusters, and the sweep evaluated 12 groups.

## Scoring, on the standing widened gates

Each arm is scored read-only on the **widened deciding population** (training, canonical twelve
groups / 36 images / 28 nonempty boards / 667 occupied and 4,933 empty known cells) with the standing
81-point grid on `[-6, 6]` step 0.15, under its own 640 MiB sweep profile. Population, grid and
thresholds are unchanged from the neighborhood, detail and stencil passes.

**The lever CLEARS iff all three hold:**

1. **Per-cell gate** — occupied-cell F1 at the replay's operating point (`0.0`) `>= 0.70`.
2. **Ranking gate** — `sweep_verdict` is `informative` (pooled AUC `>= 0.90` **and** some swept
   threshold reaches F1 `>= 0.70`).
3. **Attribution floor** — lever best swept F1 exceeds control best swept F1 by **`>= 0.05`**.

**Otherwise NOT CLEARED**, with the failing condition named mechanically.

Additional declared **mechanism quantity**, which cannot move the verdict: cell-level **precision** at
the standing point (`0.0`). `precision_improved` iff the lever exceeds the control by `>= 0.05` **and**
the whole-ancestry-cluster interval on that delta excludes zero.

Intervals: Hanley-McNeil beside each pooled AUC, Wilson beside each precision and recall, and the
whole-ancestry-cluster paired bootstrap (400 draws, seed `20261026`, twelve clusters). No gate reads
an interval.

## Receipts this pass must produce

1. `runs/m3-occres-lever-r1-profile-r1-job.json` and `runs/m3-occres-control-r1-profile-r1-job.json`,
   2 steps, 1,152 MiB / 300 s / 2 work items.
2. `runs/m3-occres-lever-r1-job.json` and `runs/m3-occres-control-r1-job.json`, 64 steps, 1,152 MiB /
   1,200 s / 64 work items, each admitted only by its own profile.
3. `...-sweep-r1-profile-r1` (2 forwards) and `...-sweep-r1` (54 forwards) per arm, 640 MiB.
4. Curation `runs/m3-occres-curation-r1-job.json` into
   `docs/evidence/m3-occupancy-resolution-r1/scope.json`, re-deriving every published number, with the
   deciding arithmetic asserted before writing.
5. The owned family `m3-occres-*` declared in `record_stream_run.OWNED_RUN_PATTERN` and bound by a test.

Any launch that dies to a harness limit leaves its attempt under `runs/attempts/` with a written note
and is re-recorded under a fresh number; no ceiling, gate, population or floor moves.

## What each outcome licenses

- **Cleared** — the sampling resolution was the constraint; the occupancy question moves to scale.
- **Not cleared, precision improved** — finer sampling carries precision but not separation; scaling
  the matched budget on this pair is the next step.
- **Not cleared, precision flat or worse** — resolution is not the constraint either; the next step is
  the untested stacking of `occupancy_context` with `occupancy_detail`, which is the only remaining
  lever that changes information rather than sampling.
- **Not admissible** — the constraint itself is the measured result, recorded with its peak.

## Outcome — NOT ADMISSIBLE AS DECLARED (measured refusal)

The two-step admission profile ran before any study, as V-057 requires, and the lever arm breached the
declared ceiling on its first launch. Nothing was trained, so nothing can be scored.

| quantity | value |
|---|---|
| declared ceiling | 1,207,959,552 B (1,152 MiB) |
| lever peak (resolution 1) | 1,231,880,192 B |
| control peak (resolution 0) | 1,034,313,728 B |
| measured increment of the branch | 197,566,464 B (188.4 MiB) |
| excess over the ceiling | 23,920,640 B (22.8 MiB) |
| declaration estimate | 50,331,648 B (one retained map) |
| estimate understated by | 3.93x |
| lever receipt | `runs/m3-occres-lever-r1-profile-r1-job.json`, failed, exit 3221225477, 46.0 s |
| control receipt | `runs/m3-occres-control-r1-profile-r1-job.json`, passed, 88.8 s |

The guard terminated the child at the ceiling; the receipt and launch log are preserved under
`runs/attempts/m3-occres-lever-refused-r1/` with a written note, and
`docs/evidence/m3-occupancy-resolution-r1/refusal.json` binds every number above to both receipts.
The control profile was run in the same session under the same ceiling, so the increment is
attributable to the branch alone rather than to drift in a replay.

**Mechanism.** A trained stride-1 convolution over the whole captured image keeps four
full-resolution 48-channel maps alive through backward, so the branch costs 3.9x what a
single-activation estimate says.

**What was not done.** No ceiling was raised, no smaller branch substituted, no capture side reduced,
no cumulative-cursor variant tried, no study, sweep or curation launched, and no re-run under a fresh
number. The declaration's own refusal rule is the whole disposition.

**Verdict.** `refused`; `precision_improved = false` because precision was never measured, not because
it failed to improve. The gates, floor, population and grid are untouched and unanswered: this pass
says nothing about whether pixel-resolution sampling separates cells, only that the declared envelope
cannot pay for it.

## What this licenses

Resolution stays open only as a *separately declared* configuration - a smaller branch, a reduced
capture side, or a computed-at-the-cell formulation are three different instruments with their own
predeclarations, and none of them is a substitution inside this one. The ranked next step is the
untested stacking of `occupancy_context` with `occupancy_detail`: the 8 x 8 context tap retains two
2 x 2 48-channel maps (about 0.02 MiB), so it fits the declared envelope, and it is the last lever
that changes the information the read-out sees rather than the sampling of it.

## Audit

30 numeric claims on the lines this pass added, across the five touched documents, were re-derived from the two profile receipts and the refusal record, with 16 receipt joins re-derived - **0 drift**. A whole-file sweep flags prior-pass history figures in these same files, so the instrument is scoped to added lines (`docs/evidence/m3-occupancy-resolution-r1/audit.json`) rather than weakening the documents.
