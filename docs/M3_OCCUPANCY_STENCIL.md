# Occupancy detail stencil — declared 2x2 in-cell sample, matched continuation control

Status: predeclared (2026-11-09) and committed before any profile, study, sweep or evidence
directory for this instrument exists. The change, the control, the envelope, the gates, the floor,
the population and the work-item arithmetic below are fixed; the measured section is appended only
after the canonical receipts exist.

## Where this comes from

`M3_OCCUPANCY_RANKING.md` closes every threshold, conditioned-threshold and class-budget family and
ranks one step after the precision result ([M3_OCCUPANCY_PRECISION.md](M3_OCCUPANCY_PRECISION.md)):
*carry the demonstrated signal with more capacity — a 2x2 in-cell detail stencil per cell through the
same zero-initialized projection*. The precise wording of that entry is "four stride-2 samples
instead of one". That pass's declared mechanism moved as intended (precision at `0.0`
0.1752 -> 0.2363, cluster interval 0.0104-0.0939, recall also up, standing-point F1
0.2848 -> 0.3816) while AUC (0.8226) and the swept optimum (0.4120) stayed short, so the open
question is whether the precision signal is limited by *how much of the cell is sampled*.

This pass declares that one change and nothing broader. The untested stacking of the two read-out
taps (`occupancy_context` and `occupancy_detail`) is **not** part of it and remains open.

## The single declared change

`ModelConfig.occupancy_detail_stencil` (default `0`, published). It is a declared *form of the
detail tap*, not a new read-out: it requires `occupancy_detail == 1` and is refused otherwise. When
`1`, the tap reads the **four quarter-cell samples of a 2x2 in-cell grid** instead of the single
cell-center sample:

```
occupancy_features = cells + W_s . concat(s(u+-1/4, v+-1/4) for the four quadrant points),
W_s : 4 * widths[0] -> dimension,   W_s = 0 at step zero
```

The projection is the same module (`occupancy_detail_tap`), initialized to **zero** exactly as
before: 4 x 48 = 192 inputs to 192, i.e. 36,864 weights plus 192 bias, four times the single tap's
9,216 plus 192, and one grid sample of 4 x 260 = 1,040 points instead of 260. The stencil offsets
are a non-persistent buffer, so both arms carry identical state-dict keys apart from the projection
itself.

Three declared consequences, bound by `tests/test_detail_stencil.py`:

1. Each arm computes **exactly** the published function at step zero (`W_s = 0`), so the change is
   additive rather than a reinitialization.
2. Every shared tensor is drawn from the same seeded initialization and loaded from the same
   checkpoint bytes; the arms differ only by the stencil form of the tap.
3. Relabelling the hidden sample order (permuting the four stencil points *and* their projection
   input blocks together) leaves the image-supported occupancy output unchanged, so a hidden state
   the pixels cannot distinguish cannot manufacture a different decision.

`occupancy_context` stays `0` in both arms: this pass declares one change, not a combination.

One declared consequence of the entry taken literally: the stencil *replaces* the center sample, it
does not add to it. The control therefore keeps the center sample the previous pass demonstrated,
and the measured contrast asks whether the four quadrant appearances carry that signal further
than the center alone does. If precision falls, the measured outcome is that the center sample
carried more than the quadrants; that is a result, not a defect.

## Matched pair, one budget, one catalogue

Both arms are **continuations** of the committed published checkpoint
`runs/m3-percell-balance-r1` (weights `9ae1fef9f649...`) under the same captured source catalogue:

| arm | `occupancy_detail` | `occupancy_detail_stencil` | everything else |
|---|---|---|---|
| lever | `1` | `1` | identical |
| control | `1` | `0` | identical |

Identical means the same initial weights, `seed 20261026`, `learning_rate .001`, curriculum
`stream-v1`, capture side `512`, component capacity `64` with the positional identity read-out and
length endpoint, occupancy loss factor `4` and positive weight `4`, **64 further steps**, the same
image order and the same envelope. Both arms are fresh continuations under this pass's own
catalogue, so the pair is matched by the repo's own source-identity standard.

Because both arms carry the declared detail tap and differ only in its stencil form, the measured
contrast is the **increment of the extra in-cell capacity over the demonstrated tap**, not "detail
versus no detail". The previous pass's arms (catalogue `27a0e917`) are reported as an unmatched
**reference** only: they cannot enter this verdict.

Envelope per arm, declared before admission: **1,152 MiB** aggregate commit, **64 steps**,
**1,200 s** runner wall / 900 s trainer wall, one heavy job at a time. The stencil adds 27,648
parameters over the single tap, so the ceiling is not expected to move; if it does, the outcome is
"not admissible as declared" and nothing is relaxed.

## Declared work-item arithmetic

Declared here because a prior pass declared 36 while performing 54 and its receipt was refused:

| launch | work items | what is counted |
|---|---|---|
| each profile | 2 | 2 optimizer updates |
| each study | 64 | 64 optimizer updates (`steps`); evaluations are not work items |
| each sweep profile | 2 | 2 forwards |
| each measured sweep | **54** | 54 forwards = 36 deciding + 18 development |
| curation | **4** | four receipts read (two studies, two sweeps) |

## Scoring, on the standing widened gates

Each arm is scored read-only on the **widened deciding population** (training, canonical twelve
groups / 36 images / 28 nonempty boards / 667 occupied and 4,933 empty known cells) with the
standing 81-point grid on `[-6, 6]` step 0.15, under its own 640 MiB sweep profile. Population, grid
and thresholds are unchanged from the neighborhood and detail passes, so the numbers sit beside the
standing table.

**The lever CLEARS iff all three hold** (identical to the two preceding passes):

1. **Per-cell gate** — occupied-cell F1 at the replay's operating point (`0.0`) `>= 0.70`.
2. **Ranking gate** — `sweep_verdict` is `informative` (pooled AUC `>= 0.90` **and** some swept
   threshold reaches F1 `>= 0.70`).
3. **Attribution floor** — lever best swept F1 exceeds control best swept F1 by **`>= 0.05`**.

**Otherwise NOT CLEARED**, with the failing condition named mechanically. If either arm's study or
sweep cannot complete inside the declared envelope, the verdict is **NOT ADMISSIBLE AS DECLARED**,
quoting the receipt; no ceiling is raised, no gate widened, no population shrunk, no run repeated
until it passes.

Additional declared **mechanism quantity**, which cannot move the verdict: cell-level **precision**
at the standing point (`0.0`). `precision_improved` iff the lever exceeds the control by `>= 0.05`
**and** the whole-ancestry-cluster interval on that delta excludes zero. Precision, recall, AUC and
F1 deltas are otherwise reported as costs.

Intervals: Hanley-McNeil beside each pooled AUC, Wilson beside each precision and recall, and the
whole-ancestry-cluster paired bootstrap (400 draws, seed `20261026`, twelve clusters) for both arms
and every delta. No gate reads an interval.

## Receipts this pass must produce

1. `runs/m3-occdstencil-lever-r1-profile-r1-job.json` and
   `runs/m3-occdstencil-control-r1-profile-r1-job.json`, 2 steps, 1,152 MiB / 300 s / 2 work items.
2. `runs/m3-occdstencil-lever-r1-job.json` and `runs/m3-occdstencil-control-r1-job.json`, 64 steps,
   1,152 MiB / 1,200 s / 64 work items, each admitted only by its own profile.
3. `...-sweep-r1-profile-r1` (2 forwards) and `...-sweep-r1` (54 forwards) per arm, 640 MiB.
4. Curation `runs/m3-occdstencil-curation-r1-job.json` into
   `docs/evidence/m3-occupancy-stencil-r1/scope.json`, re-deriving every published number, with the
   deciding arithmetic (36 cases, 28 nonempty boards, 667/4,933 cells, 12 clusters) asserted before
   writing.
5. The owned family `m3-occdstencil-*` declared in `record_stream_run.OWNED_RUN_PATTERN` and bound
   by a test.

Any launch that dies to the platform's 295 s command cap or leaves its declared arithmetic is
preserved under `runs/attempts/` with a written defect note and re-recorded canonically; no ceiling,
gate, population or floor moves.

## What each outcome licenses

- **Cleared** — extra in-cell capacity is the precision lever at this budget; the occupancy
  question moves to scale, and the next step is a longer matched budget on this pair.
- **Not cleared, precision improved** — more in-cell sampling carries the demonstrated gain further
  without reaching the bar; the next step moves upstream to encoder resolution.
- **Not cleared, precision flat or worse** — the center sample was already the better in-cell
  statistic; extra quadrant samples do not add separation, and the next step is encoder resolution
  rather than more read-out detail. If precision is *worse*, the reference comparison against the
  prior pass must say so explicitly.
- **Not admissible** — the constraint itself is the measured result.
