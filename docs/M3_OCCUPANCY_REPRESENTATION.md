# Occupancy representation lever — declared read-out neighborhood, matched control

Status: predeclared (2026-11-09) and committed before any profile, study, sweep or evidence
directory for this instrument exists. The rule, the envelope and the floor below are fixed; the
measured section is appended only after the canonical receipts exist.

## Where this comes from

The occupancy head is closed from four measured sides: the scalar weight, the class split, the
global threshold family and — most recently — the **conditioned** family
([M3_OCCUPANCY_CONDITIONING.md](M3_OCCUPANCY_CONDITIONING.md)). That last pass found 0 of 28
nonempty deciding boards reach F1 0.70 at their own oracle grid point, an oracle-pooled F1 of
0.4305, and per-board AUC of 0.291 / **0.829** / 0.970. The ordering is real; the separation is
not. The record therefore ranks **the representation** first, and until now that ranking had no
experiment behind it.

This pass is that experiment: one declared change to what the occupancy read-out sees, one matched
control, one budget, the standing gates.

## The single declared change

`ModelConfig.occupancy_context` (default `0`, published). When `1`, the cell-latent grid
`(26 rows x 10 columns)` passes through a residual block before the occupancy head:

```
occupancy_features = cells + W2 * SiLU(GroupNorm8(W1 * cells_neighborhood))
```

with a 3x3 convolution over the lattice (so each cell sees its eight neighbors), `W1` a plain
convolution and `W2` a 1x1 projection initialized to **zero**. Two consequences are declared here
because the whole comparison rests on them:

1. At step zero the levered model computes **exactly** the published function — the block adds
   nothing — so the lever is additive, not a reinitialization.
2. Every shared tensor (`encoder`, `decoders`, `occupancy`, `semantic`, `components`, ...) is drawn
   from the same seeded initialization and loaded from the same checkpoint, so the only difference
   between arms is the block itself. `tests/test_occupancy_context.py` binds both properties.

No other model, loss, data, seed, capture-side or threshold change is made. The occupancy head,
the semantic head, the component heads and the trunk all keep their published weights at start.

## Matched pair, one budget

Both arms are **continuations** of the committed published checkpoint `runs/m3-percell-balance-r1`
(weights `9ae1fef9f649...`, 128 steps, 458.08 s):

| arm | `occupancy_context` | everything else |
|---|---|---|
| lever | `1` | identical |
| control | `0` | identical |

Identical means: the same initial weights, `seed 20261026`, `learning_rate .001`, curriculum
`stream-v1`, capture side `512`, component capacity `64` with the published positional identity
read-out and length endpoint, occupancy loss factor `4` and positive weight `4`, **64 further
steps**, the same image order, and the same resource envelope. The control is a continuation too:
it isolates *the block*, not "more training", which is why the published 128-step numbers are
quoted as context and never as the control.

Envelope, per arm, declared before admission: **1,152 MiB** aggregate commit (the source study's
own ceiling), **64 steps**, **1,200 s** runner wall / 900 s trainer wall, one heavy job at a time.
The lever adds one 3x3 convolution, one 1x1 projection and one GroupNorm at 192 channels — under
0.2 M parameters — so the ceiling is not expected to move; if it does, the outcome is
"not admissible as declared" and nothing is relaxed.

## Scoring, on the standing widened gates

Each arm is scored read-only by `scripts/sweep_stream_occupancy.py` on the **widened deciding
population** (training, canonical twelve groups / 36 images / 28 nonempty boards, 667 occupied and
4,933 empty known cells), under its own 640 MiB sweep profile, with the standing 81-point grid and
the standing thresholds. Nothing about the population, grid or thresholds changes.

**The lever CLEARS iff all three hold:**

1. **Per-cell gate** — occupied-cell F1 at the replay's operating point (`0.0`) is `>= 0.70`
   (`CELL_F1_THRESHOLD`, unchanged).
2. **Ranking gate** — `occupancy_sweep.sweep_verdict` on the swept grid is `informative`
   (pooled AUC `>= 0.90` **and** some swept threshold reaches F1 `>= 0.70`; unchanged).
3. **Attribution floor** — lever best swept F1 exceeds control best swept F1 by **`>= 0.05`**.
   Without this the lever is not attributable even if the absolute bar were met by both arms; it
   is declared here, not chosen after seeing a delta.

**Otherwise the lever is NOT CLEARED**, and the failing condition is named mechanically from the
counts. If either arm's study or sweep fails to complete inside the declared envelope, the verdict
is **NOT ADMISSIBLE AS DECLARED**, quoting the receipt; no ceiling is raised, no gate widened, and
no run is repeated until it passes. The exact-board counts, the empty-population rate, development
rows and every per-board quantity are costs: reported, inert for the verdict.

Intervals respect the clustering already measured
([M3_WIDENED_DECIDING_POPULATION.md](M3_WIDENED_DECIDING_POPULATION.md) addendum): Hanley-McNeil
beside each pooled AUC, Wilson beside each precision and recall, and a whole-**ancestry-cluster**
paired bootstrap (400 draws, seed `20261026`, twelve clusters) for the AUCs, the F1s and the
lever-vs-control delta. No gate reads an interval, and no interval can move the verdict.

## Receipts this pass must produce

1. Lever profile: `runs/m3-occctx-lever-r1-profile-r1-job.json`, 2 steps, 1,152 MiB / 300 s / 2
   work items. Control profile: `runs/m3-occctx-control-r1-profile-r1-job.json`, same envelope.
2. Studies, each admitted only by its own profile: `runs/m3-occctx-lever-r1-job.json` and
   `runs/m3-occctx-control-r1-job.json`, 64 steps, 1,152 MiB / 1,200 s / 64 work items.
3. Sweep profile and sweep per arm: `...-sweep-profile-r1` (2 forwards, 640 MiB) and `...-sweep-r1`
   (36 forwards, 640 MiB / 600 s / 36 work items).
4. Curation: `runs/m3-occctx-curation-r1-job.json` into
   `docs/evidence/m3-occupancy-representation-r1/scope.json`, re-deriving every published number.
5. The owned family `m3-occctx-*` is declared in `record_stream_run.OWNED_RUN_PATTERN` and bound by
   a test, so no run can hide outside the declared names.

## What each outcome licenses

- **Cleared** — the read-out neighborhood is a representation lever at this budget; the occupancy
  question moves from "needs a representation" to "this representation, then scale", and the next
  step is a longer matched budget on the same pair.
- **Not cleared** — the recorded mechanism says whether the block failed to improve separation
  (delta indistinguishable from zero), improved it without reaching the bar, or improved one arm's
  fit while costing precision; the ranking doc then names the next representation question in
  priority order, and the boundary family stays closed.
- **Not admissible** — the lever is recorded as exceeding its declared envelope, which is itself
  the measured constraint on this read-out at this budget.

## Measured result

(appended after the canonical receipts exist)
