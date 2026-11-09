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

## Measured result: NOT CLEARED — the block buys ordering and coverage, not separation

Receipts: lever profile `runs/m3-occctx-lever-r1-profile-r1-job.json` (passed, 51.0 s,
1,045,307,392 B) and control profile `runs/m3-occctx-control-r1-profile-r1-job.json` (passed,
49.7 s, 1,015,898,112 B), both under the declared 1,152 MiB; lever study
`runs/m3-occctx-lever-r1-job.json` (passed, 64 steps, 206.4 s, 1,059,500,032 B) and control study
`runs/m3-occctx-control-r1-job.json` (passed, 64 steps, 193.8 s, 1,043,173,376 B); both sweeps 54
forwards under 640 MiB; curation `runs/m3-occctx-curation-r3-job.json` into
`docs/evidence/m3-occupancy-representation-r1/scope.json`. Both arms start from
`9ae1fef9f649...` and differ in `occupancy_context` alone.

| widened deciding population | lever (`1`) | control (`0`) | published 128-step |
|---|---|---|---|
| pooled AUC | **0.8244** (CI 0.8045-0.8442; cluster 0.8199 [0.7941, 0.8390]) | 0.7427 (CI 0.7204-0.7651; cluster 0.7491 [0.6498, 0.8261]) | 0.7852 |
| occupied-cell F1 at `0.0` | **0.3600** (recall 0.985-0.998) | 0.2848 (recall 0.728-0.792) | 0.3455 |
| best swept F1 (threshold) | 0.4200 (2.70) | **0.4338** (3.90) | 0.3704 (-0.90) |
| exact nonempty boards at best | 0/28 | 0/28 | 0/28 |
| ranking verdict | `uninformative` | `uninformative` | `uninformative` |

Deltas (lever minus control): best swept F1 **-0.0138**; pooled AUC **+0.0816** (cluster interval
-0.0252 to +0.1794); F1 at the standing point **+0.0752** (cluster interval 0.0142 to 0.1106,
median 0.0580).

**Verdict: NOT CLEARED**, failing all three declared conditions — per-cell F1 0.3600 against 0.70,
ranking `uninformative` (AUC 0.076 short of 0.90 and best F1 0.28 short of 0.70), and attribution
-0.0138 against the +0.05 floor.

The block does something real, and the record should say exactly what. At matched budget it
improves **ordering**: pooled AUC rises 0.7427 -> 0.8244, and the cluster-respecting interval on
the standing-point F1 excludes zero. What it does not improve is **separation**. Recall at the
standing `0.0` decision jumps from 0.73-0.79 to 0.985-0.998 — the read-out becomes permissive,
flagging nearly every occupied cell — while the swept F1 optimum falls slightly below the control's
(0.4200 at `2.70` against 0.4338 at `3.90`). The neighborhood term therefore moved the read-out's
operating behaviour and its ranking, not its ability to separate occupied from empty at a point
that clears the bar; 0/28 boards are exact in either arm.

Relative to the conditioning pass, this sharpens the mechanism rather than contradicting it:
separation is still the binding constraint, and a local-context term added at the read-out does
not create it at this budget. The next ranked representation step is therefore one that targets
**cell-level precision** — a contrastive or boundary-aware detail tap at the lattice, or a
higher-resolution detail feature at the sampled points — not another threshold, conditioning or
class-budget pass, all of which remain closed.

Provenance, preserved rather than hidden: the first lever study was killed by a runner restart at
step 46 (`runs/attempts/m3-occctx-lever-r1-runner-restart/` with its receipt), the first pair of
sweeps declared a 36 work-item cap while performing 54 forwards
(`runs/attempts/m3-occctx-*-r1-sweep-r1-workcap36*`), and the first curation bootstrapped all 54
cases instead of the declared 36 (`runs/attempts/evidence-m3-occupancy-representation-r1-allsplit-bootstrap/`).
Each was preserved, refused by a declared check, and re-recorded canonically; neither the
ceiling, the gates, the population nor the floor moved.

`tests/test_representation_lever.py` re-derives both arms, the paired twelve-cluster bootstrap,
the population and the verdict from the committed receipts.
