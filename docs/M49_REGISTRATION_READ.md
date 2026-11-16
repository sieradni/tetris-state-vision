# M49: does a label-free raster registration buy the read back?

Status: **predeclared before any scoring run.** The registered lattice, the protocol, the three comparable reads, the
falsifier and every branch threshold are fixed below. The measured section is added after the run and is not part of
the declaration.

## The question this arm answers

M48 (`docs/M48_GEOMETRY_READ.md`, `docs/evidence/m48-geometry-read-r2/scope.json`) applied the campaign's committed
per-cell read-out unchanged to the same thirty statistics at two samplings and measured that the ceiling is a property
of *where* the numbers are sampled: `0.974763` F1 at the fixed `0.0` crossing at the **target** lattice, `0.037152` at
the **predicted** one, with the committed model's final lattice a measured mean `2.479843` cells away from the target
lattice (median `2.048453`, p90 `5.050841`). Its named next arm was registration-first.

This arm runs that arm. It samples the same thirty statistics at a lattice **registered from the board's own raster
support** - no training target, no label, no annotation, no fitted constant, and no use of the committed model's
drifted prediction - and reads it with the same committed read-out, on the same `2,986` cells, the same splits, the
same evaluator and the same fixed `0.0` crossing. The three reads are then directly comparable for the first time.

## The read-out, imported and its provenance

Identical to M48's, and pinned by the same two checks before either non-target number is produced.

- **artifact**: `runs/m3-occpix-ceiling-r1-r2/result.json` (producer `scripts/probe_pixel_ceiling.py`, committed sweep
  `m3-occstack-lever-r2-sweep-r1`, weights `c0b5b783fc11…`).
- **definition**: per cell, thirty statistics - a 3x3 interior block mean per channel (27) plus the per-channel
  interior standard deviation (3) over `+/-0.3` cell units - read by a logistic regression with ridge `1.0`,
  leave-one-ancestry-group-out, standardised by the training folds, clipped to `+/-8`, `40` iterations
  (`scripts/pixel_ceiling.py::logistic_fit`).
- **committed numbers on the fixed cells**: F1 at `0.0` `0.974763`, AUC `0.999195`, best swept `0.987179`.
- **substitution (declared, unchanged from M48)**: the artifact ships out-of-fold logits rather than coefficients, so
  the read-out is re-executed from its own committed estimator on the committed features, labels and folds and is
  accepted **only** if it reproduces the committed logits to `1e-6`; the same fitted folds then transform the
  registered and predicted features. No parameter is fitted on this arm's inputs, and no threshold is fitted.
- **provenance pins, checked before any registered-lattice number exists**: (1) every nonempty board's target-lattice
  statistics reproduce the committed `features_sha256`; (2) the reconstruction reproduces the committed
  `pixel_logits` to `1e-6`.

## The registered lattice, exactly

`pixel_support_lattice(valid, rows=20, columns=10, ...)` from the geometry source's frozen tree
(`runs/m47-control-winner-config-r6/code/tetris_vision/model.py`), the campaign's own declared M34 registration
recipe (`docs/M34_SUPPORT_LATTICE.md`): per board, the bounding box `[left, right) x [top, bottom)` of the **valid**
raster pixels - the mask of pixels the capture actually contains, not padding - mapped onto the board's twenty rows
and ten columns bottom-first:

`origin = (left, bottom)`, `column = ((right-left)/10, 0)`, `row = (0, -(bottom-top)/20)`.

It reads the raster mask only: no label, no prevalence, no scored cell, no fitted, learned or board-derived-from-labels
quantity, and it never sees the committed model's predicted lattice. The function is imported from the frozen geometry
tree and its file is asserted to live inside that tree, and the lattice is converted to image pixels by the same
diagonal map M48 used for the predicted lattice, so the two transfers are directly comparable.

## The three reads, and what each one is

| read | sampling geometry | provenance |
|---|---|---|
| target lattice | the committed board geometry (`state.board.lattice.transformed`) | the ceiling read-out's own |
| predicted lattice | the committed winner's final-stage `lattice` output | M48's measurement |
| **registered lattice** | the raster support's bounding box, label-free | **this arm** |

Exactly one thing differs across the three: the lattice `pixel_ceiling.cell_features` reads. Same twenty-eight
nonempty deciding boards, same `200` cells per board, same `2,986`-cell fixed population, same anchored labels, same
ancestry-group folds, same evaluator, fixed `0.0`.

## Protocol, isolation and falsifier

- **no training**, no weight change (the script asserts the geometry source's tensors are unchanged after its
  forwards), no new learned parameter, no threshold fitting, no per-board rule.
- **single seed inherited**: none is drawn; the committed folds, labels and weights are the control's own.
- **non-inertness and anti-shortcut**: the registered lattice's statistics change the read-out's logits relative to
  both other samplings (more than the perturbation floor), and the registered read does not reproduce either of the
  other two reads' logits (tests).
- **falsifier, taken from already-committed numbers** - the ceiling read-out on these cells (`0.974763` / `0.999195` /
  `0.987179`), the winner control (`0.372881` F1 at `0.0`, `0.954529` AUC, precision `0.230430`), M48's predicted
  lattice (`0.037152` / `0.442341`) and M48's measured displacement (`2.479843` cells). Nothing here is derived from
  this arm's rows.

**Branch selection, fixed now, on the registered read alone:**

| registered read | branch | reading | named next arm |
|---|---|---|---|
| F1 at `0.0` `>= 0.70` **and** AUC `>= 0.90` | **`registration-recovers`** | a label-free raster registration places the statistics well enough that the committed read-out keeps ceiling-level quality; the defect is the *placement the model predicts*, and registration is the fix | feed the registered lattice to the decision (the read samples the support lattice instead of the predicted one) and re-measure the winner's own head there |
| F1 at `0.0` `>= 0.382881`, or AUC `>= 0.86`, but not both above the first row | **`registration-partial`** | registration recovers the ordering but not the fixed crossing: the placement is close but not exact, and the residual is a scale/offset the read cannot absorb | measure the registered-vs-target displacement in cell units, and only then decide whether a placement correction or a crossing re-centring is the next test |
| F1 at `0.0` `< 0.382881` **and** AUC `< 0.86` | **`registration-fails`** | the support bounding box does not place the statistics; label-free raster registration is ruled out as the campaign's geometry fix and the next arm must address placement inside the model | placement training inside the model (the geometry it predicts), not another registration recipe |

**Prediction, declared before the run**: M34 already measured the support lattice's *placement* residual at
`-0.004778` points (its smallest term, against `-0.114151` for the predicted lattice), so the registered lattice
should sit close to the target lattice and the read should mostly return: registered F1 at `0.0` `0.90-0.99`, AUC
`0.985-0.999`, and registered-vs-target displacement `<= 0.10` cells mean. The predicted branch is
**`registration-recovers`**.

## Pre-committed reading of every branch

- **`registration-recovers`** - geometry registration is the campaign's fix and it is label-free; the next arm gives
  the decision the registered lattice (no new parameter) and its falsifier is the winner's own head scoring below
  `0.372881` F1 at `0.0` or below `0.954529` AUC when it samples there.
- **`registration-partial`** - the registered lattice is measurably better than the predicted one but not exact; the
  next arm is decided by the measured residual, and a scale/offset correction is only proposed if the residual is
  large enough to explain the loss.
- **`registration-fails`** - label-free raster registration cannot place the read; the campaign stops proposing
  registration recipes and the next arm moves placement inside the model.
- **a failed provenance pin** (features digest mismatch or logit reconstruction above `1e-6`) is not an arm reading:
  it is recorded as the measurement's finding and no registered-lattice number is published.

No branch is re-run for a better number.

## Receipts

`runs/m49-registration-read-profile{,-job.json}`, `runs/m49-registration-read{,-job.json}`,
`docs/evidence/m49-registration-read/{scope,diagnosis}.json`. A refused attempt keeps its receipt under its own name;
only completed runs are quoted.
