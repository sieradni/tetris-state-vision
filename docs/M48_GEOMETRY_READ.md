# M48 - the committed read-out at the predicted lattice: statistics or geometry?

Status: **predeclared before any scoring run.** The read-out and its provenance, the substitution its missing
coefficients force, the two samplings, the protocol, the branch thresholds and the falsifier are below. The measured
section is added after the run and is not part of the declaration.

## The question this arm answers

M47 removed every latent path from the occupancy decision and lost `0.239131` F1 at `0.0` and `0.094155` AUC, while
the campaign's committed per-cell read-out scores `0.974763` F1 at `0.0` and `0.999195` AUC on the **same** `2,986`
cells from the **same** thirty statistics. Both cannot be about the numbers alone. The difference between them is
where those numbers are sampled: the committed read-out was granted the **target lattice** (the committed board
geometry), while the model's read samples at the lattice it **predicts**.

This arm measures that difference directly, with no training and no parameter fitted on its own inputs.

## The read-out, imported and its provenance

- **Artifact**: `runs/m3-occpix-ceiling-r1-r2/result.json`, produced by `scripts/probe_pixel_ceiling.py` joining the
  committed sweep `m3-occstack-lever-r2-sweep-r1` at weights `c0b5b783fc11…`, declared in
  `docs/M3_OCCUPANCY_PIXEL_CEILING.md`.
- **Definition**: per cell, thirty statistics - a 3x3 interior block mean per channel (27) plus the per-channel
  interior standard deviation (3), sampled over `+/-0.3` cell units - read by a **logistic regression with ridge
  `1.0`**, leave-one-ancestry-group-out, standardised by the training folds, clipped to `+/-8` and iterated `40`
  times (`scripts/pixel_ceiling.py::logistic_fit`).
- **Committed numbers on the `2,986` fixed cells**: F1 at `0.0` `0.974763`, AUC `0.999195`, best swept `0.987179`.

**The substitution this arm must declare.** The committed artifact ships per-board **out-of-fold logits**, not
coefficients, so the read-out cannot be applied to another sampling by loading a matrix. The faithful derivation is
therefore: run the read-out's **own committed estimator** (`scripts/pixel_ceiling.py`, unchanged constants) on the
**committed features, labels and folds**, keep the fitted per-fold transform, and **accept the reconstruction only if
it reproduces the committed logits to `1e-6` on all twenty-eight nonempty boards**; only then transform the
predicted-lattice features with those same folds. No parameter is fitted on this arm's inputs, no threshold is fitted
(this arm scores at the fixed `0.0` and also reports the sweep the committed read-out was reported with), and the
features the read-out is fitted on are the committed target-lattice ones.

The provenance is pinned twice, and both pins are checked *before* any predicted-lattice number exists:

1. **features**: the target-lattice statistics recomputed here have the committed artifact's own `features_sha256`
   per board (verified for the two profiled boards at admission, and for all twenty-eight in the run);
2. **read-out**: the reconstructed folds reproduce the committed `pixel_logits` to `1e-6`.

## The two samplings, and what changes

Exactly one thing changes between the two reads: the lattice the same `cell_features` function reads.

- **Target lattice** (the committed read): `case.frame.state.board.lattice.transformed(scale_x, scale_y)` - the
  committed board geometry in the prepared raster's pixel units.
- **Predicted lattice** (this arm): the committed winner checkpoint's own final-stage `lattice` output, converted to
  image pixels by the model's own convention (origin top-left, columns rightward, rows upward, multiplied by the
  capture's width and height) - no training, no label, no annotation.

Everything else is held fixed: the same twenty-eight nonempty deciding boards, the same `200` cells per board, the
same labels, the same ancestry-group folds, the same `2,986`-cell fixed population, the same evaluator, and the fixed
`0.0` crossing.

## Protocol, isolation and falsifier

- **No training**, no weight change (the script asserts the model's tensors are unchanged after the forwards), no new
  learned parameter, no threshold fitting. The model is used read-only for its geometry.
- **Single seed inherited**: none is drawn; the committed folds, labels and weights are the control's own.
- **Non-inertness**: perturbing the statistics changes the read-out's logits (test); the read-out applied to its own
  target features reproduces the committed logits exactly (test), and applied to shifted features it does not
  (anti-shortcut test).
- **Falsifier, from already-committed numbers**: the committed ceiling read-out on these cells (`0.974763` F1 at
  `0.0`, `0.999195` AUC, best `0.987179`), the winner control (`0.372881` F1 at `0.0`, `0.954529` AUC, precision
  `0.230430`), and the absence of any committed seed-spread replicate for the winner - so deltas above `0.02` AUC are
  treated as resolvable and nothing smaller is claimed.

**Branch selection, fixed now, by F1 at `0.0` on the predicted lattice:**

| predicted-lattice F1 at `0.0` | branch | reading |
|---|---|---|
| `>= 0.70` (the campaign's per-cell gate) | **`readout-binds`** | the thirty statistics order the cells even where the model thinks they are; the head's **training** is the binding constraint, and the successor is head-only training from the committed initialisation |
| `>= 0.382881` and `< 0.70` | **`geometry-costs`** | placement takes a measurable share of the ceiling but does not remove the evidence; the successor must address both |
| `< 0.382881` (not better than the control) | **`geometry-binds`** | at the predicted lattice the evidence no longer separates; **sampling geometry** is the binding constraint, and the successor is a registration-first arm |

Also reported, whatever the branch: AUC, best swept F1 with its threshold, the complete TP/FN/FP/TN table, precision
and recall at `0.0`, the row-wise false-positive profile for rows 10-19 and 0-9 reconciled to the control's own
counts, and the per-family margin distributions of both samplings so that a degradation is attributed either to
ordering (AUC) or to threshold placement, and never asserted.

**Prediction, declared before the run**: the read-out is a fixed linear function of thirty statistics and the
predicted lattice is a few tensor-widths' displacement from the target lattice, so the evidence should survive with a
cost: predicted-lattice F1 at `0.0` `0.80-0.96`, AUC `0.97-0.999`. The predicted branch is **`readout-binds`**.

## Pre-committed reading of every branch

- **`readout-binds`** - the statistics are not the constraint, so the campaign's next arm trains the head alone from
  the committed initialisation with the trunk frozen, and its falsifier is a return to `>= 0.382881` F1 at `0.0`
  without losing `0.02` AUC.
- **`geometry-costs`** - the next arm is a client-side decision: head-only training **and** registration must both be
  budgeted, and the cheaper of the two is measured first.
- **`geometry-binds`** - the next arm samples the decision's statistics at a support-derived lattice (registration
  first); its falsifier is an AUC that stays under `0.93`.
- **A failed provenance pin** (features digest mismatch or logit reconstruction above `1e-6`) is not an arm reading:
  it is recorded as the measurement's finding, the closest faithful derivation is reported, and no predicted-lattice
  number is published.

No branch is re-run for a better number.

## Measured

`docs/evidence/m48-geometry-read-r2/scope.json` and its `diagnosis.json`: the committed read-out applied to the same
thirty statistics at two samplings on the same `2,986` cells, one execution, no training. (The `-r1` name was
refused by the runner after an earlier attempt consumed its receipt; every attempt is preserved.)

| read | F1 at `0.0` | AUC | best swept |
|---|---|---|---|
| committed read-out at the **target** lattice (reconstruction) | `0.974763` | `0.999195` | `0.987179` |
| committed read-out at the **predicted** lattice (this arm) | **`0.037152`** | **`0.442341`** | `0.037152` (at `-6.00`, the grid's edge) |
| winner control (its own learned head) | `0.372881` | `0.954529` | `0.689775` |

**Verdict: `geometry-binds`, decisively.** The reconstruction reproduces the committed artifact exactly - same F1 at
`0.0` (`0.974763`), same AUC (`0.999195`) - so the read-out is the committed one; moved to the lattice the model
predicts, the *same* read-out loses `0.937611` F1 at `0.0` and `0.556854` AUC, and its ordering falls **below
chance**.

### Provenance pins (checked before any predicted-lattice number existed)

- **Features**: every one of the twenty-eight nonempty boards' target-lattice statistics reproduce the committed
  artifact's own `features_sha256` (asserted in the run; the profile pinned two of them at admission).
- **Read-out**: the reconstructed per-fold transform reproduces the committed `pixel_logits` with a maximum absolute
  difference of `4.98282851424392e-07`, i.e. inside the declared `1e-6`.
- **Substitution declared**: the committed artifact ships out-of-fold logits, not coefficients; the read-out was
  therefore re-executed from its own committed estimator on the committed features, labels and folds, and accepted
  only because it reproduces those logits. No parameter was fitted on this arm's inputs; the model's tensors are
  asserted unchanged after the forwards.

### Complete contingency table at the fixed `0.0` crossing

| at `0.0` | target lattice | predicted lattice |
|---|---|---|
| true positives | `309` | `6` |
| false negatives | `0` | `303` |
| false positives | `41` | `8` |
| true negatives | `2,636` | `2,669` |
| precision | `0.882857` | `0.428571` |
| recall | `1.000000` | `0.019417` |

The predicted lattice does not over-fire - it under-fires catastrophically: `303` of `309` occupied cells are called
empty, and its best swept F1 (`0.037152`) sits at the sweep grid's own lower boundary (`-6.00`), so no threshold
recovers it.

### The late-count profile, reconciled to the control

| row band | control (its own head) | predicted-lattice read-out |
|---|---|---|
| rows 0-9 false positives | `400` | `8` |
| rows 10-19 false positives | `140` | `0` |
| total false positives | `698` | `8` |

Per row: `1` at row 0, `4` at row 2, `3` at row 6, and zero everywhere else. **This is not an improvement in the
late mass** - the arm barely predicts positive at all, so its false-positive count is small for the same reason its
recall is `0.019`: comparing it to the control's `140` as if it were a like-for-like late-row reading would be wrong,
and the reconciliation is stated here so it cannot be read that way.

### Decomposing the degradation: ordering, and a distribution shift

- **Margin scale**: at the predicted lattice the read's logits collapse away from the fitted scale - false negatives
  sit at median `-50.568729` and true negatives at `-18.937260`, while the `6` true positives sit at `+2.820802` and
  the `8` false positives at `+2.168406`. The target-lattice read produces margins of order `+/-1`.
- **Ordering**: AUC `0.442341` is *below* chance, so the loss is not a threshold sitting in the wrong place - a shift
  of the boundary cannot change AUC at all. The same 30 statistics, sampled where the model thinks the cells are,
  anti-correlate with the anchored labels under this read-out.
- **Geometry, measured on the same boards** (`docs/evidence/m48-lattice-displacement/scope.json`, read-only): the
  committed model's final-stage lattice sits a **mean `2.479843` cells** from the target lattice (median
  `2.048453`, p90 `5.050841`), overwhelmingly vertical - mean `-1.116965` cells along the board's row axis and
  `-0.129167` along its column axis. The thirty statistics drift from their target-lattice distribution by a mean
  `0.764765` standard deviations, with `3` of the `30` features beyond `3` standard deviations.
- **Concentration**: the disagreement is not localised in rows or columns - the arm's handful of positives are spread
  over rows 0, 2 and 6 of the low band, and the discussion above shows the loss is a whole-population shift, not a
  band.

So the degradation is both distributional (a two-cell displacement moves the patch statistics off the fitted scale)
and ordinal (the ordering inverts), and the displacement is the measured cause: a per-cell colour read with a
`+/-0.3`-cell footprint cannot survive a two-cell registration error.

## Statistics or geometry? The measured answer

**Geometry.** The same estimator on the same thirty statistics scores `0.974763` F1 at `0.0` / `0.999195` AUC where
the cells actually are, and `0.037152` / `0.442341` where the committed model believes they are, with a measured mean
displacement of `2.479843` cells. The ceiling is a property of **where the numbers are sampled**, not of the numbers
alone, and the campaign's learned read has been fighting that displacement at every arm: M47's pixel-sole head,
trained *at* the displaced lattice, reached `0.860374` AUC - far above this fixed read-out's `0.442341` but far below
the winner's `0.954529`.

## Re-ranked successors

| rank | next arm | headroom | cost | falsifier |
|---|---|---|---|---|
| 1 | **registration-first**: the decision's statistics sampled at a support- or anchor-derived lattice instead of the predicted one, with the committed read-out and the head unchanged | the measured displacement (`2.479843` cells mean) is the largest single defect this campaign has quantified, and the same read-out worth `0.974763` F1 at `0.0` lives on the other side of it | one read-only registration pass plus one `64`-step run, no new parameters | AUC stays under `0.93` at the registered lattice - sampling geometry is then not the binding term either and the campaign's read is at its structural limit |
| 2 | **head-only training from the committed initialisation**, measured *at the registered geometry* (rank 1's outcome, not a substitute for it) | high once the geometry is fixed: the head's training is then the only remaining term | one `64`-step run, no new parameters | F1 at `0.0` `< 0.382881` or AUC `< 0.954529` at registered geometry: training is not the residual term |
| 3 | **seed-spread replicate of the winner** (measurement) | bounds every future delta | three `64`-step runs | spread in AUC `> 0.02` |
| R | **retired**: every additive raster term (M44-M46), the pixel-sole routing (M47), M42/M43/M16 decision-side levels, M14/M31 levels, M9-M13 objective terms, M15/M17, M34-M41 geometry variants | closed by measurement | - | not offered |

**Named next arm: registration-first.** This measurement makes it the only step whose falsifier is still open, and
its own number (`2.479843` cells of measured displacement) is the reason.

## Receipts

`runs/m48-geometry-read-profile{,-job.json}`, `runs/m48-geometry-read-r1-job.json` (the refused attempt preserved by
V-181), `runs/m48-geometry-read-r2{,-job.json}`, `docs/evidence/m48-geometry-read-r2/{scope,diagnosis}.json`,
`runs/m48-lattice-displacement-profile{,-job.json}`, `runs/m48-lattice-displacement{,-job.json}`,
`docs/evidence/m48-lattice-displacement/scope.json`.
