# M30: moving the level the winning read lands on

Status: **predeclared before any M30 run.** Mechanisms, signatures, floors, gate rule and receipts
are fixed here; the measured section is appended only after the canonical receipts exist.

## What is already measured (not re-derived here)

On the fixed matched 2,986-cell population with both label vectors fixed by the reference receipt
(`docs/evidence/m29-pixel-form-r2/scope.json`):

| read | F1 at `0.0` | AUC | best swept F1 |
|---|---|---|---|
| pixel ceiling, same cells | 0.974763 | 0.999195 | 0.987179 |
| reference arm, own lattice | 0.439024 | 0.933138 | 0.543974 |
| M29 winner, form 3 (30-64-192, zero-init output) | **0.291464** | **0.946604** | **0.610101** |
| unchanged control | 0.129870 | 0.574178 | 0.219626 |

The block ranks (`0.946604` AUC, above the `0.90` bar) and its best swept F1 is `0.610101`, but its
level at the fixed `0.0` boundary is far from where the labels sit: `0.291464` against the `0.70`
per-cell bar. This pass declares three mechanisms that can only move **where the read sits**, not
what it can order, and asks which one moves the level.

## Declared mechanisms, one single change each

Every mechanism is exactly identity at step zero, so step zero stays bit-identical to the M29 winner
(form 3). All are trained at the unchanged 64-step protocol from `runs/m3-percell-balance-r1`, seed
`20261026`, 1,152 MiB ceiling, one job at a time.

| mechanism | flag | implementation | identity at step zero |
|---|---|---|---|
| **A. learned level freedom** | `--pixel-query-level 1` | the projection's output gets a scale `1+tanh(delta)` and a bias `beta`, both parameters, added before the head | `delta=beta=0` gives scale `1`, bias `0` |
| **B. supervision-side level** | `--occupancy-balance 1` | the occupancy loss's class budget becomes the image's own prevalence ratio `negative/positive` clamped to `[0.1, 10]`, recomputed per image; the labelled cells and their labels are unchanged | the number is data-derived, not a chosen constant; no parameter |
| **C. monotone read calibration** | `--occupancy-read-standardise 1` | the read becomes `logits + tanh(g)*(board_standardised - logits)`, a label-free per-board affine map through a learned gate | `g=0` gives exactly the published logits |

## Signatures and the floor

The anchor is the M29 winner: F1 at `0.0` `0.291464`, AUC `0.946604`, best `0.610101`. A mechanism
is **confirmed** only if its F1 at `0.0` on the fixed population rises by at least **`0.05`** over the
anchor; below that floor it is **excluded** as unable to explain the level.

| mechanism | confirming signature | additionally reported |
|---|---|---|
| A | F1 at `0.0` `>= 0.341464` with AUC **bit-identical** to `0.946604`, because a single global positive affine map cannot reorder pooled cells | the learned scale and bias, and whether the scale stayed positive |
| B | F1 at `0.0` `>= 0.341464` | the AUC movement, and the prevalence ratios the loss actually used |
| C | F1 at `0.0` `>= 0.341464` | the learned gate, and the AUC movement |

**A monotone calibration that lifts F1 at `0.0` while leaving AUC unchanged is evidence about the
gate's operating point, not a pass of the ranking claim.** The receipt records that distinction in
its cause field instead of treating the two as the same result.

## Predeclared gate rule

The best confirmed mechanism is evaluated on all three campaign gates on the same fixed population:

1. per-cell F1 at `0.0` `>= 0.70`;
2. ranking: pooled AUC `>= 0.90` **and** best swept F1 `>= 0.70`;
3. attribution over the unchanged control (`0.219626` best) `>= 0.05`.

`cleared` iff all three hold; otherwise the receipt names the gate that remains and the measured
numbers that say what the true remaining limit is. The span closure (`best - 0.543974` over
`0.443205`) and the placement/representation split are re-derived from the winner's numbers, and the
points and fractions are asserted to sum exactly before the receipt is written.

## Receipts

`runs/m30-level-{bias,balance,calibrate}-r1-profile-r1` (2 steps, 1,152 MiB), the studies
`runs/m30-level-*-r1` (64 steps, 1,200 s runner wall), the probes
`runs/m30-level-*-r1-registration-r1` (36 forwards, 640 MiB), and the reader
`runs/m30-operating-point-r1-job.json` (128 MiB) writing `docs/evidence/m30-operating-point-r1/`.
The `m30-level-` family is declared in the owned-run guard. Any refused or destroyed attempt keeps its
receipt; only completed runs are reported.

## Limits, stated before the measurement

- One seed, 64 steps, synthetic training-fit boards; a null result bounds these mechanisms at this
  budget, not at all budgets.
- The gate is measured at the fixed `0.0` boundary by the standing instrument; a mechanism that moves
  the read without moving F1 at `0.0` is not the same as a mechanism that improves separation.
- No population, metric, threshold or gate is changed, and the control and reference arms are the
  committed ones, not retrained.
