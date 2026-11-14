# M40: the predicted lattice, corrected by the support

Status: **predeclared before any M40 scoring run.**

## What this pass is for, and a defect it caught before running

The campaign's recorded next step (M39, `docs/M39_DUAL_PLACEMENT.md`) was to keep the model's predicted
lattice as the sampled geometry and apply the support only as a declared alignment composed with it, so the
read stays a function of what the network predicts while gaining the raster's correction.

**The literal formulation of that is a duplicate, and this document records why before spending a run on
it.** An affine lattice mapped affinely onto the support box *is* the support lattice: the predicted box
becomes the support box, the column and row vectors scale to it, and the result equals
`pixel_support_lattice` for every board, whatever the predicted lattice was. The composition as first
declared therefore collapses onto M34's arm, and measuring it would have produced a second copy of
`0.274869` / `0.883260` / `0.385925` while claiming to test something new. V-165 records the defect.

The form below is the same idea made non-degenerate: a declared **convex combination**, so the sampled
geometry is half the predicted lattice and half the aligned one.

## The mechanism, exactly (form 14)

`--occupancy-pixel-query 14`, one head block, no second selector, no new parameter:

- **Sampled geometry**: `pixel_composed_lattice(lattice, valid, rows, columns, device, dtype)`, which
  returns, per board and per lattice vector, `(1-w) * predicted + w * aligned`, with
  `PIXEL_COMPOSED_WEIGHT = 0.5` declared as a fixed constant (not learned, not selected, not tuned).
  `aligned` is the affine lattice that carries the predicted board box onto the bounding box of the valid
  pixels, computed exactly as M34's helper does.
- **Properties the tests pin**: it is the identity when the support already fills the capture (so step zero
  and any full-frame board are untouched), and on a padded capture it is neither the predicted lattice nor
  the support lattice - the two endpoints M39 measured.
- **Features and projection**: unchanged - the `30` statistics, `Linear(30, 64) -> SiLU -> Linear(64,
  dimension)` with a zero-initialised output. Against the winner the form adds **no parameter at all**
  (`653,243` either way): `0.5*(predicted+aligned)` is arithmetic, and the gradient still reaches the
  decoder's lattice through the predicted half.
- **Boundary**: the consumer's fixed global `0.0`. Nothing fitted, no label, no prevalence, no scored cell.

## Protocol

One profile, one 64-step study, one 36-board probe under the `m40-composed` owned family (1,152 MiB study,
640 MiB probe), scored by `scripts/m37_stencil_read.py --inputs 30 --instrument m40-composed-r1
--predeclaration docs/M40_COMPOSED_ALIGNMENT.md` on the same 2,986 cells and both fixed label vectors, the
winner re-scored in the same pass.

## Gate thresholds, unchanged

per-cell `0.70` on both fixed label vectors; ranking (AUC `>= 0.90` and best swept `>= 0.70`); attribution
over the unchanged control `>= 0.05`.

## The deciding bar, fixed before scoring

New operating point only if, against the winner's re-scored numbers:

- F1 at `0.0` `>= 0.372881`, and
- best swept F1 `>= 0.689775`.

**A refusal means** a half-strength exogenous correction is not learnable at this depth either, and the
campaign can then state - across the two endpoints already measured and this arm between them - whether the
placement degradation is a property of *any* exogenous component in the sampling geometry, rather than of
how strongly it is applied.

## Branch interpretations, pre-committed

- **PASS**: the arm becomes the operating point; re-run its probe and reproduce weights and logits (not an
  earlier delta by assumption), and record the next structural step.
- **FAIL**: report the gate flags, deltas and split, and interpret the result **between the measured
  endpoints** - M34/M7 support lattice (`0.274869` / `0.883260` / `0.385925`) and the winner's predicted
  lattice (`0.372881` / `0.954529` / `0.689775`) - to say whether the exogenous correction is
  present-but-insufficient, absent, or swamped, using false-positive counts, margins and the same row bands
  as M37-M39. Capacity is not available as an explanation and will not be offered.

## Receipts

`runs/m40-composed-{profile,study,probe}` and `docs/evidence/m40-composed-alignment-r1/{scope,diagnosis}.json`.
Refused attempts keep their receipts; only completed runs are quoted.
