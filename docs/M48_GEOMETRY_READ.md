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

## Receipts

`runs/m48-geometry-read-profile{,-job.json}`, `runs/m48-geometry-read{,-job.json}`,
`docs/evidence/m48-geometry-read-r1/{scope,diagnosis}.json`.
