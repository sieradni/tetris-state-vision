# Occupancy resolution lever, re-declared at an admitted envelope

Status: **predeclared (2026-11-12) and committed before any profile, study, sweep or evidence directory
for this instrument exists.** The change, the control, the envelope, the gates, the floor, the
population and the work-item arithmetic below are fixed; the measured section is appended only after
the canonical receipts exist.

This is a *separately declared configuration*, which is what
[M3_OCCUPANCY_RESOLUTION.md](M3_OCCUPANCY_RESOLUTION.md) licensed when it refused its own run. The
refusal verdict stands: that pass raised no ceiling and its gates remain unanswered. This document
declares a new, larger envelope and nothing else. No gate, floor, population, grid, budget, seed or
source checkpoint moves.

## Why the question is still open

The refused pass measured its branch at **197,566,464 B (188.4 MiB)** of aggregate commit against a
same-session control, and breached its self-imposed **1,152 MiB** ceiling by **23,920,640 B
(22.8 MiB)**. Nothing about the lever was measured as ineffective - the arm never reached one
optimizer step under its declared envelope. The three read-out levers that ran converted extra
capacity into ordering (AUC 0.7427 -> 0.8244 -> 0.8226 -> 0.8804) while the swept optimum stayed
0.21-0.29 short of the 0.70 bar, so the open question is unchanged: is the information the tap reads
**sampled too coarsely** to separate cells, rather than the read-out being too small?

The host publishes **33,515,835,392 B** physical and the repository's admission rule permits
`min(physical/3, 8 GiB)` = 8 GiB, with the usual ceiling well below 6 GiB. The refused pass's own
number therefore sat 22.8 MiB under a limit that was a declaration, not a capability.

## The single declared change

`ModelConfig.occupancy_resolution` (default `0`, published), byte-identical to the refused
declaration. It is a declared *source* for the existing detail tap, not a new read-out: it requires
`occupancy_detail == 1`, and is mutually exclusive with the declared stencil form
(`occupancy_detail_stencil == 0`) so the pass compares resolution with resolution, not with sampling
geometry.

When `1`, a **stride-1 convolution of the image** supplies the tap's sample in place of the encoder's
stride-2 first-stage feature:

```
fine      = Conv2d(3 -> widths[0], 3x3, padding 1)((2*pixels-1)*valid)     # stride 1
sample    = sample(stride-2 first stage) + sample(fine)                    # same cell centers
occupancy = cells + W_d . sample,        W_d = 0 at step zero
```

The projection is the same module, same shape (48 -> 192) and the same zero initialization in both
arms; the only new tensors are the branch's 3 x 48 x 3 x 3 = 1,296 weights and 48 bias. The branch is
**not** zero-initialized - a zero branch would receive no gradient through the zero projection and
could never train - but the projection is, so step zero is exactly the published function in both
arms. The branch exists only on the levered arm, so it is the single difference.

Three declared consequences, bound by `tests/test_occupancy_resolution.py` (unchanged):
both arms compute exactly the published function at step zero; every shared tensor is drawn from the
same seeded initialization and loaded from the same checkpoint bytes; and with the projection at zero
any branch weights whatsoever give the same image-supported output.

## Envelope, declared before admission

| quantity | value |
|---|---|
| aggregate commit ceiling, per arm | **1,879,048,192 B (1,792 MiB)** |
| measured 2-step peak of this branch (refused pass) | 1,231,880,192 B |
| headroom over that measurement | 647,168,000 B (617.2 MiB) |
| steps per arm | 64 |
| runner wall / trainer wall | 1,200 s / 900 s |
| sweep envelope | 640 MiB, 54 forwards |
| heavy jobs | one at a time |

The declaration is deliberately above the measured peak rather than at the measured minimum: the
refused pass's own error was estimating an autograd-retained branch from a single-activation count
(3.93x understatement, V-062). Four retained full-resolution 48-channel maps at capture side 512 are
204,472,320 B before the masked image, and a 64-step run adds checkpointing traffic on top of the
2-step profile. 1,792 MiB is the next round envelope that pays that bill on measurement rather than
on another estimate.

**Declared refusal rule:** if either arm cannot be admitted inside 1,792 MiB at the matched budget,
the outcome is **NOT ADMISSIBLE AS DECLARED**, recorded with the measured peak and the receipt; no
ceiling is raised again, no smaller variant is substituted, and no run is repeated to pass. This
document's envelope is not revised after a receipt.

## Matched pair, one budget, one catalogue

Both arms are **continuations** of the committed published checkpoint `runs/m3-percell-balance-r1`
(weights `9ae1fef9f649...`) under the same captured source catalogue:

| arm | `occupancy_detail` | `occupancy_detail_stencil` | `occupancy_resolution` |
|---|---|---|---|
| lever | `1` | `0` | `1` |
| control | `1` | `0` | `0` |

Everything else identical: `seed 20261026`, `learning_rate .001`, curriculum `stream-v1`, capture
side `512`, component capacity `64` with the positional identity read-out and length endpoint,
occupancy loss factor `4` and positive weight `4`, **64 further steps**, the same image order, gates,
floor and population. Each arm runs its own checkpoint-every-32 policy from the same start bytes.

The control is therefore the same configuration as the stencil and stacking passes' control, so its
numbers also serve as an independent repetition check of that baseline; those arms ran under other
catalogues and are **not** matched references here.

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
boards / 667 occupied / 4,933 empty known cells / 12 ancestry clusters, and the sweep evaluated 12
groups.

## Scoring, on the standing widened gates

Each arm is scored read-only on the **widened deciding population** (training, canonical twelve
groups / 36 images / 28 nonempty boards / 667 occupied and 4,933 empty known cells) with the standing
81-point grid on `[-6, 6]` step 0.15, under its own 640 MiB sweep profile. Population, grid and
thresholds are unchanged from every pass in this neighborhood.

**The lever CLEARS iff all three hold:**

1. **Per-cell gate** - occupied-cell F1 at the replay's operating point (`0.0`) `>= 0.70`.
2. **Ranking gate** - `sweep_verdict` is `informative` (pooled AUC `>= 0.90` **and** some swept
   threshold reaches F1 `>= 0.70`).
3. **Attribution floor** - lever best swept F1 exceeds control best swept F1 by **`>= 0.05`**.

**Otherwise NOT CLEARED**, with the failing condition named mechanically.

## The calibration-neutral report this pass adds

[M3_MINIMUM_DETECTABLE_EFFECT.md](M3_MINIMUM_DETECTABLE_EFFECT.md) declares, for what comes next, that
a primary read is **three components, not one F1**: ranking quality at the arm's own best boundary,
the zero-boundary cost `F1(0.0) - F1(best)`, and the transfer cost `F1(out-of-fold) - F1(0.0)`. This
pass is the first to report all three.

**Reported, never read.** The verdict is the three gates above, unchanged. The report adds:

- `out_of_fold`: the same grid, criterion and tie-break as the standing sweep, with the threshold
  chosen for each ancestry group **on the other eleven only** (highest F1, then higher precision, then
  lower threshold) and applied to the held-out group; the twelve held-out count sets are pooled. It is
  invariant to any monotone shift of an arm's logits, so it cannot inherit a boundary the lever moved.
- A paired whole-ancestry-cluster bootstrap (400 draws, seed `20261026`, twelve clusters) of both the
  operating-point F1 delta and the out-of-fold F1 delta, reported with its standard error and the
  minimum detectable effect at 80% power (`2.802 x SE`) as declared in the power document.
- The mechanism quantity this family already carries: precision at `0.0`, with `precision_improved`
  iff the lever exceeds control by `>= 0.05` **and** the whole-cluster interval excludes zero. It
  cannot move the verdict.

No interval and no component is read by a gate or a verdict.

## Receipts this pass must produce

1. `runs/m3-occres2-lever-r1-profile-r1-job.json` and `runs/m3-occres2-control-r1-profile-r1-job.json`,
   2 steps, 1,792 MiB / 300 s / 2 work items.
2. `runs/m3-occres2-lever-r1-job.json` and `runs/m3-occres2-control-r1-job.json`, 64 steps, 1,792 MiB /
   1,200 s / 64 work items, each admitted only by its own profile.
3. `...-sweep-r1-profile-r1` (2 forwards) and `...-sweep-r1` (54 forwards) per arm, 640 MiB.
4. Curation `runs/m3-occres2-curation-r1-job.json` into
   `docs/evidence/m3-occupancy-resolution-r2/scope.json`, re-deriving every published number, with the
   deciding arithmetic asserted before writing.
5. The owned family `m3-occres2-*` declared in `record_stream_run.OWNED_RUN_PATTERN` and bound by
   `tests/test_resolution_r2.py`.

Any launch that dies to a harness limit leaves its attempt under `runs/attempts/` with a written note
and is re-recorded under a fresh number; no ceiling, gate, population or floor moves.

## What each outcome licenses

- **Cleared** - pixel-resolution sampling was the constraint; the occupancy question moves to scale,
  and the next block is priced by the power document rather than by another single trajectory.
- **Not cleared, precision improved** - finer sampling carries precision but not separation; the same
  branch is then priced at a matched multiple of this budget.
- **Not cleared, precision flat or worse** - resolution is not the constraint either; the sampling
  family is closed with a measured number instead of an unmeasured refusal, and the residual is
  registration, which the 4x-budget fallback already implicated at +0.1352 F1.
- **Not admissible** - the constraint itself is the measured result, recorded with its peak; and
  because this ceiling is 1.56x the refused one, a second refusal would mean the branch is not
  affordable on this host at all, which is a stronger statement than the first.
- **Unchanged either way** - the fit gate stays FALSE unless 3 of 28 exact nonempty boards appear, the
  identity heads are untouched, and no real-image, complete-state or quality claim follows.

## Outcome — NOT CLEARED, and the first read-out lever to meet the attribution floor

Both arms ran 64 steps from `9ae1fef9f649...` under one catalogue, each admitted by its own 2-step
profile at the declared ceiling, and both sweeps scored the widened deciding population with the
standing grid. Nothing moved after the receipts existed: the gates, floor, population, grid, ceiling
and source checkpoint are the ones declared above.

| quantity | lever (stride-1) | control (stride-2) |
|---|---|---|
| deciding AUC | **0.868612** | 0.822643 |
| occupied-cell F1 at `0.0` | **0.413858** | 0.381640 |
| best swept F1 | **0.472012** at `1.65` | 0.412034 at `0.90` |
| out-of-fold F1 | **0.460754** | 0.400399 |
| exact nonempty boards | 0 of 28 | 0 of 28 |
| study peak commit | 1,500,233,728 B | 1,061,023,744 B |
| study wall | 217.6 s | 122.7 s |

The envelope held on both arms - the lever's peak sits **378,814,464 B (361.3 MiB)** under the
declared 1,879,048,192 B - and the two-step profile had already measured 1,485,766,656 B against
the refused pass's 1,231,880,192 B, so the re-declared ceiling was necessary rather than liberal.

**Verdict `not-cleared`.** Per-cell gate fails (`0.413858` against `0.70`), ranking gate fails
(`sweep_verdict uninformative`: AUC `0.868612` is **0.031388** short of `0.90` and the best swept F1
is **0.227988** short of `0.70`), and the attribution floor is **met**: best swept F1 rises
**+0.059978**, AUC **+0.045968**, the fixed-boundary F1 **+0.032218** and precision **+0.025637**
(below its own `0.05` floor, so `precision_improved = false`). This is the first matched
continuation in the read-out family to clear the `+0.05` attribution floor, and it does so while
raising ranking rather than trading it away.

**The calibration-neutral report is what makes that readable.** The three components separate:

| component | delta | whole-cluster interval | MDE at 80% power | resolvable |
|---|---|---|---|---|
| best swept F1 (boundary fitted on the scored cells) | +0.059978 | none claimed | - | reported only |
| fixed-boundary F1 | +0.032218 | +0.016476 .. +0.039566 | 0.016226 | **yes** |
| out-of-fold F1 | +0.060355 | -0.138621 .. +0.179634 | 0.222851 | **no** |

So the gain is real against the design's floor where the boundary is fixed, and it is **not
resolvable** once the boundary has to be chosen on other ancestry groups: the transfer interval is
wide enough to contain zero and its minimum detectable effect is `0.223`, nearly four times the
observed `0.060`. The lever also moves its own zero-boundary cost (-0.058154 against -0.030394), so
part of its fixed-boundary advantage is the offset it shifts rather than separation alone. At twelve
clusters this pass can say finer sampling helps and cannot say how much of it transfers.

**What this settles.** Sampling resolution was never measured; it is measured now, and it is not the
binding constraint on the fit gate - it buys ordering and a resolvable fixed-boundary gain, well
short of the separation `0.70` requires. The sampling family is closed with a number rather than a
refusal, and the residual the campaign has already attributed is registration: at 4x budget, pairing
model cells with the labels of the cells they physically cover moved F1 `+0.1352`, three times this
lever's gain.

## Audit

The recorder re-derives every published number from the four receipts and the two sweep results,
refuses a mismatched pair (different catalogue, different start bytes, identical logits), and asserts
the deciding arithmetic (28 boards, 12 clusters, 667 occupied) before writing. The out-of-fold
component and its interval are computed by the same rule the verdict uses for its boundary, and
`tests/test_resolution_r2.py` checks the out-of-fold selector against a hand-worked toy case and
against a monotone shift, so a component that could be inflated by moving the boundary is refused.
