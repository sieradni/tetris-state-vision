# Occupancy precision lever — declared high-resolution detail tap, matched control

Status: predeclared (2026-11-09) and committed before any profile, study, sweep or evidence
directory for this instrument exists. The change, the control, the envelope, the gates, the floor,
the population and the work-item arithmetic below are fixed; the measured section is appended only
after the canonical receipts exist.

## Where this comes from

`M3_OCCUPANCY_RANKING.md` closes every threshold, conditioned-threshold and class-budget family and
ranks exactly one next step after the neighborhood-residual result
([M3_OCCUPANCY_REPRESENTATION.md](M3_OCCUPANCY_REPRESENTATION.md)): *a representation that targets
cell-level precision rather than ordering at this budget*. That pass raised pooled AUC
0.7427 -> 0.8244 and standing-point F1 0.2848 -> 0.3600 but made the read-out permissive (recall at
`0.0` 0.73-0.79 -> 0.985-0.998), did not beat its control's best swept F1 (0.4200 against 0.4338),
and cleared no gate. This document declares the one change that directly attacks that precision
failure, and nothing broader.

## The single declared change

`ModelConfig.occupancy_detail` (default `0`, published). When `1`, the occupancy read-out also sees
a **higher-resolution sample at the cell center**: the first image-encoder stage's stride-2 feature
map (the same map the decoder path only reaches after average-pooling) is sampled at each lattice
cell's center point and projected into the read-out dimension by a linear map whose weights and bias
are initialized to **zero**:

```
occupancy_features = cells + W_d * first_stage(cell centers),   W_d = 0 at step zero
```

Why this and not the neighborhood residual: the previous lever mixed decoder latents after the
axial decoder had already spread information across cells, which raised ordering and permissiveness.
The failure this pass targets is *precision* — an empty cell predicted occupied — and the
information that separates a filled cell from empty background at that scale is the pixel-level
appearance at the cell itself, which the stride-2 map still carries and the pooled decoder path does
not. The tap adds 48 x 192 = 9,216 parameters and one grid sample of 260 points; nothing else
changes.

Two declared consequences, bound by `tests/test_occupancy_detail.py`:

1. At step zero the levered model computes **exactly** the published function (`W_d = 0`), so the
   change is additive rather than a reinitialization.
2. Every shared tensor is drawn from the same seeded initialization and loaded from the same
   checkpoint, so the arms differ only by the tap.

`occupancy_context` stays `0` in both arms: this pass declares one change, not a combination.

## Matched pair, one budget, one catalogue

Both arms are **continuations** of the committed published checkpoint `runs/m3-percell-balance-r1`
(weights `9ae1fef9f649...`) under the same captured source catalogue:

| arm | `occupancy_detail` | everything else |
|---|---|---|
| lever | `1` | identical |
| control | `0` | identical |

Identical means the same initial weights, `seed 20261026`, `learning_rate .001`, curriculum
`stream-v1`, capture side `512`, component capacity `64` with the positional identity read-out and
length endpoint, occupancy loss factor `4` and positive weight `4`, **64 further steps**, the same
image order and the same envelope. The control is a fresh continuation under this pass's own
catalogue, not the previous pass's control, precisely so the pair is matched by the repo's own
source-identity standard.

Envelope per arm, declared before admission: **1,152 MiB** aggregate commit, **64 steps**, **1,200 s**
runner wall / 900 s trainer wall, one heavy job at a time. The tap adds 9,216 parameters, so the
ceiling is not expected to move; if it does, the outcome is "not admissible as declared" and nothing
is relaxed.

## Declared work-item arithmetic

Declared here because a prior pass declared 36 while performing 54 and its receipt was refused:

| launch | work items | what is counted |
|---|---|---|
| each profile | 2 | 2 optimizer updates |
| each study | 64 | 64 optimizer updates (`steps`); evaluations are not work items |
| each sweep profile | 2 | 2 forwards |
| each measured sweep | **54** | 54 forwards = 36 deciding + 18 development |
| curation | 4 | four receipts read |

## Scoring, on the standing widened gates

Each arm is scored read-only on the **widened deciding population** (training, canonical twelve
groups / 36 images / 28 nonempty boards / 667 occupied and 4,933 empty known cells) with the
standing 81-point grid on `[-6, 6]` step 0.15, under its own 640 MiB sweep profile. Population, grid
and thresholds are unchanged from the neighborhood pass, so the numbers sit beside the standing
table.

**The lever CLEARS iff all three hold** (identical to the previous pass):

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

1. `runs/m3-occdetail-lever-r1-profile-r1-job.json` and
   `runs/m3-occdetail-control-r1-profile-r1-job.json`, 2 steps, 1,152 MiB / 300 s / 2 work items.
2. `runs/m3-occdetail-lever-r1-job.json` and `runs/m3-occdetail-control-r1-job.json`, 64 steps,
   1,152 MiB / 1,200 s / 64 work items, each admitted only by its own profile.
3. `...-sweep-r1-profile-r1` (2 forwards) and `...-sweep-r1` (54 forwards) per arm, 640 MiB.
4. Curation `runs/m3-occdetail-curation-r1-job.json` into
   `docs/evidence/m3-occupancy-precision-r1/scope.json`, re-deriving every published number, with
   the deciding arithmetic (36 cases, 28 nonempty boards, 12 clusters) asserted before writing.
5. The owned family `m3-occdetail-*` declared in `record_stream_run.OWNED_RUN_PATTERN` and bound by
   a test.

## What each outcome licenses

- **Cleared** — the read-out detail tap is a precision lever at this budget; the occupancy question
  moves to scale, and the next step is a longer matched budget on this pair.
- **Not cleared, precision improved** — the tap does the mechanism work (empties predicted occupied
  fall) without reaching the bar; the next step is a representation that carries the same signal
  with more capacity or at more cells.
- **Not cleared, precision flat** — read-out detail at the cell center is not the precision
  bottleneck either; the next step moves upstream into the encoder's resolution or the target
  policy, and the remaining families stay closed.
- **Not admissible** — the constraint itself is the measured result.

## Measured result

(appended after the canonical receipts exist)
