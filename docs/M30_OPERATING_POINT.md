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

## Measured

Three declared forms, one change each, all trained at the unchanged 64-step protocol from
`runs/m3-percell-balance-r1` and all scored on the identical 2,986 cells with both fixed label
vectors (`docs/evidence/m30-operating-point-r1/scope.json`). The anchor is the M29 winner,
re-scored here from its own probe rather than quoted: F1 at `0.0` `0.291464`,
AUC `0.946604`, best `0.610101`.

| mechanism | F1 at `0.0` | AUC | best swept F1 | delta F1 at `0.0` | delta best | verdict |
|---|---|---|---|---|---|---|
| bias | 0.285714 | 0.943155 | 0.621315 | -0.005750 | +0.011214 | excluded |
| balance | 0.275772 | 0.903305 | 0.449123 | -0.015693 | -0.160978 | excluded |
| calibrate | 0.372881 | 0.954529 | 0.689775 | +0.081417 | +0.079674 | confirmed |

The unchanged control is `0.219626` best and the reference arm
`0.543974`; the pixel read-out on the same cells holds
`0.987179`.

### What each mechanism did

- **Learned level freedom (A, bias): excluded.** The learned scale settled at
  `0.995428` with bias `0.000636` - essentially
  identity, because the loss has no reason to move a parameter that does not change the ordering and the
  batch already balances classes. F1 at `0.0` moved `-0.005750`, best
  `+0.011214`. The mechanism is real but too weak to explain the
  level, at this budget and on this loss.
- **Data-derived class budget (B, balance): excluded, and negative.** F1 at `0.0`
  `-0.015693`, best `-0.160978`,
  AUC `-0.043298`. The already class-balanced loss means the
  prevalence ratio mostly re-weights a term that was balanced by construction, and it cost ranking.
- **Gated board standardisation (C, calibrate): confirmed.** F1 at `0.0`
  `0.372881` (`+0.081417`), best swept
  `0.689775` (`+0.079674`), AUC
  `0.954529` (`+0.007925`), attribution over the control
  `+0.470149`. The learned gate is `0.022809` - a small,
  strictly positive, label-free affine map per board - and it moves the level far more than that number
  suggests, because the read's across-board offsets are what the F1-at-`0.0` gate samples.

### The gate on the confirmed mechanism

| gate | bar | calibrate | result |
|---|---|---|---|
| per-cell F1 at `0.0` | `>= 0.70` | `0.372881` | fail |
| ranking (AUC `>= 0.90` and best `>= 0.70`) | `0.90` / `0.70` | `0.954529` / `0.689775` | AUC pass, best fail |
| attribution over the unchanged control | `>= 0.05` | `+0.470149` | pass |

`cleared` is `False` and the cause field reads `level-inconclusive:calibrate`: one of three gates
passes. Span closure for this read is `+0.145801` (`0.328969`),
placement `-0.114151` (`-0.257557`), representation
`+0.411555` (`0.928588`), summing to the
`0.443205` span in points and to one in fractions, asserted in code.

### The reading, stated as predeclared

The predeclaration says a monotone calibration that lifts F1 at `0.0` is evidence about the gate's
operating point, not a pass of the ranking claim. Here it is more than that and less than a clearance:
the AUC also moved `+0.007925` - the map is monotone inside each board, so the
pooled AUC could only move through the across-board level, and it did - and the best swept F1 rose
`+0.079674` to `0.689775`, `0.010225` short of the
`0.70` ranking bar. Per-cell F1 at the fixed `0.0` boundary remains `0.327119`
short, and that is the gate the level cannot buy: standardisation re-centres each board but the fixed
boundary is global, so a per-board map moves each board's level against a threshold it does not own.

### What this establishes, and what it does not

- The read's level *is* adjustable by a label-free structural mechanism, and the mechanism that works is
  the across-board normalisation, not a learned scalar, not a class budget.
- The remaining per-cell gap is a boundary problem: the gate scores every cell against one global
  threshold while the read's correct level differs per board. That is a property of the gate the campaign
  measures, and since the predeclaration forbids changing it, this receipt reports the gap rather than
  closing it by redefinition.
- Not shown: whether a per-board boundary would clear the per-cell gate (that would be a metric change and
  is out of scope here), whether wider capacity compounds with this mechanism, or any real-image claim.
  One seed, 64 steps, training-fit boards.

## M31 update: the boundary was never the limit

The question this receipt left open is settled on the same frozen arm
(`docs/M31_DECISION_RULE.md`, `docs/evidence/m31-decision-rule-r2/scope.json`): the campaign's declared
label-free per-board crossing scores `0.278699` F1 at `0.0` against the global
read's `0.372881`, so the boundary is not the deficit, and the standing gate is
unchanged. A label-fitted per-board boundary is reported only as an oracle bound
(`0.763916`, excluded from every verdict). The branch's declared structural
follow-up, a context-carrying hidden projection on this arm, measured `0.518657` best swept F1
against `0.689775` (`-0.171118`), so that step is refused too.

