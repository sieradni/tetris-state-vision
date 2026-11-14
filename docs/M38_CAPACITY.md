# M38: the same fused read at twice the hidden width

Status: **predeclared before any M38 scoring run.** This document was written after M37 was measured, and
the diagnosis below is what selects the change.

## What M37 measured, and why the next change is capacity

`docs/evidence/m37-stencil-fusion-r1/scope.json` (form 11: `30` statistics + the ten-number coarse
`stencil_contrast` summary, `40` inputs, one head block): F1 at `0.0` `0.270096`, AUC `0.919512`, best swept
`0.589286` at threshold `3.15`; gate per-cell/ranking false, attribution true; **bar missed** on both
declared comparisons (`0.372881`, `0.689775`).

The comparison with M35's recorded diagnosis is decisive for what to do next:

| measurement | M35 (60 inputs) | M37 (40 inputs, fused) |
|---|---|---|
| false positives at `0.0` | `1088` | `1131` |
| false-positive median margin | `1.352717` | `1.558202` |
| weakest true-positive margin | `0.936171` | `0.804033` |
| upper-band false positives (rows 0-9) | `933` | `932` |
| `textured_empty_separated` | - | **`false`** |

The coarse channel did **not** separate the textured-empty cells: the false-positive count is unchanged,
their margins are *higher* than M35's, and they still sit above the weakest true positive. Three feature
families (absolute statistics, their sample-mean deviations, and now the coarse contrast taps) have each
plateaued between `0.570` and `0.589` best swept, and the best swept point is computed at the arm's own
optimal threshold - so the shortfall is **not** the fixed boundary: even a perfectly placed boundary cannot
reach the winner's `0.689775`. AUC fell to `0.919512` against the winner's `0.954529`.

**Named binding constraint: the read's discrimination at this width, not placement, not the boundary, not
the feature family.** The projection is `Linear(C, 64) -> SiLU -> Linear(64, dimension)`: one 64-wide layer
has to map both the level statistics and the coarse contrast summary, and with 64 training steps that is
the narrowest part of the arm.

## The declared mechanism (form 12)

`--occupancy-pixel-query 12`, one declared form, one head block:

- **Inputs**: unchanged from form 11 - the `30` statistics plus the ten-number coarse `stencil_contrast`
  summary, `40` inputs. No new feature family, no new sampling, no new label.
- **Capacity**: the hidden width doubles, `Linear(40, 128) -> SiLU -> Linear(128, dimension)`, with the
  output projection still zero-initialised, so step zero is the published function and the arm adds no
  output parameter. Against form 11 this is `40*64 + 64` more hidden parameters; against the winner it is
  `40*128 + 128` in the hidden input width.
- **Placement and boundary**: unchanged - `pixel_support_lattice`, and the consumer's fixed global `0.0`.
  Nothing is fitted, no threshold is swept, no label or prevalence is read.

## Protocol

One profile, one 64-step study, one 36-board probe under the `m38-capacity` owned family (1,152 MiB study,
640 MiB probe), scored by `scripts/m37_stencil_read.py` with `--instrument m38-capacity-r1` on the same
2,986 cells and both fixed label vectors, the winner re-scored in the same pass.

## Gate thresholds, unchanged

per-cell `0.70` on both fixed label vectors; ranking (AUC `>= 0.90` and best swept `>= 0.70`); attribution
over the unchanged control `>= 0.05`.

## The deciding bar, fixed before scoring

New operating point only if, against the winner's re-scored numbers:

- F1 at `0.0` `>= 0.372881`, and
- best swept F1 `>= 0.689775`.

**A refusal means** the width is not the binding constraint either, and the campaign records that three
feature families and a doubled read all plateau below the winner's discrimination at this budget - which is
a finding about the arm class, not an invitation to tune.

## Branch interpretations, pre-committed

- **PASS**: promote, re-run the probe and reproduce weights and logits, and record the next structural step.
- **FAIL**: report the gate block, the split and the ladder, state plainly that capacity at this budget is
  refused as well, and record the measured next step as the one the evidence supports - a deliberate
  departure from the support-lattice arm class, since every variant of it now sits below the winner's
  ranking.

## Receipts

`runs/m38-capacity-{profile,study,probe}` and `docs/evidence/m38-capacity-r1/{scope,diagnosis}.json`.
Refused attempts keep their receipts; only completed runs are quoted.
