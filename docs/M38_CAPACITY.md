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

## Measured

Study `m38-capacity` (probe `-registration`), form 12 - form 11's `40` inputs through a
`128`-wide hidden layer - under the unchanged 64-step protocol, peak `1,026,883,584` bytes. Scored by
`scripts/m37_stencil_read.py` (`docs/evidence/m38-capacity-r1/scope.json`), the winner re-scored in the same
pass.

| read | F1 at `0.0` | AUC | best swept F1 |
|---|---|---|---|
| M30/M31 winner (re-scored) | `0.372881` | `0.954529` | `0.689775` |
| doubled read, identity vector | `0.273082` | `0.895041` | `0.463551` (at `2.4`) |
| delta | -0.099799 | -0.059488 | -0.226223 |
| doubled read, anchored vector | `0.279241` | `0.771979` | `0.387302` |
| same inputs at 64 wide (M37) | `0.270096` | `0.919512` | `0.589286` |

Gate block: per-cell `false`, ranking
`false`, attribution `true`
(+0.243925 over control). **Bar missed on both comparisons; capacity is refused.**

Doubling the width made the same read **worse**, not better: best swept
-0.125735 against M37's `40`-input arm, AUC -0.024472
against M37, and the false positives moved the wrong way (`1114`
against `1131`, median margin `1.385334` against `1.558202`, upper bands
`932` -> `933`). At a fixed 64-step budget the extra `128`-wide layer is not trained, so this is a
measurement of the budget, not of the read's expressiveness.

## The span split

| term | points | fraction of the `0.443205` span |
|---|---|---|
| closure | -0.080423 | -0.181457 |
| placement | -0.076250 | -0.172042 |
| representation | +0.599877 | 1.353499 |

Sums asserted in code; `short_stage: closure`.

## The ladder now

| arm | inputs | hidden | F1 at `0.0` | AUC | best swept |
|---|---|---|---|---|---|
| **M30/M31 winner (predicted lattice)** | 30 | 64 | **`0.372881`** | **`0.954529`** | **`0.689775`** |
| M34 support lattice | 30 | 64 | `0.274869` | `0.883260` | `0.385925` |
| M35 contrast | 60 | 64 | `0.277778` | `0.945596` | `0.570048` |
| M36 monotone contrast | 60 | 64 | `0.313667` | `0.886350` | `0.420935` |
| M37 fused coarse channel | 40 | 64 | `0.270096` | `0.919512` | `0.589286` |
| M38 doubled capacity | 40 | 128 | `0.273082` | `0.895041` | `0.463551` |

**What the ladder says.** Every arm that samples the raster at the support lattice ranks worse than the
winner, whatever its features or width; the winner is the only arm whose statistics are sampled at *the
model's own predicted lattice*, so its features and its placement are consistent with each other. Three
feature families (absolute, sample-mean deviation, coarse contrast) and a doubled width have been measured
and refused; the best support-lattice arm is `0.100489` below the winner on best swept and `0.035016` below
on AUC, and its errors are `1114` confident false positives
whose margins (`0.005197`-`4.816195`, median
`1.385334`) sit above the weakest true positive
(`0.622427`).

## The next structural step the evidence now implies

Read the **same cell at both placements in one arm**: the `30` statistics sampled at the model's predicted
lattice beside the `30` sampled at the support lattice (`60` inputs, one head block, no second selector).
The difference between the two is a *placement-consistency* signal that reads only the raster and the
model's own lattice - no label, no fitted constant, no threshold - and it is the one quantity that
distinguishes the winner's arm class from every arm this campaign has since measured. That is recorded here
and predeclared for the next pass; it is not run in this one.

## Receipts

`runs/m38-capacity-job.json` (study, peak `1,026,883,584`), `runs/m38-capacity-registration-job.json` (probe, peak `563,929,088`), `runs/m38-capacity-r1-job.json` (this score). The runner restart interrupted this pass mid-profile and the driver was relaunched: the surviving pair produced a second complete study (`runs/m38-capacity-r2-job.json`, peak `1,029,341,184`) and probe (`runs/m38-capacity-r2-registration-job.json`), which is kept as a **reproducibility check** - identical `weights_sha256` (`74d80e8e...`) and a maximum absolute logit difference of `0.0` across all 36 boards - and as the record of a duplicate launch (V-163).
