# M34: a support-derived lattice, and the question of whether placement is reducible

Status: **predeclared before any M34 scoring run.** Mechanism, protocol, unchanged gate thresholds, both
branch interpretations and the placement-reducibility rule are fixed here.

## The hypothesis

The M33 receipt (`docs/evidence/m33-fixed-lattice-r1/scope.json`) measured a constant lattice over the
whole capture as the worst placement recipe so far (`-0.118649` points, `-0.267707` of span) and recorded
this pass's step: derive the sample lattice from the capture's **valid support** instead of the padded
capture, since a padded capture puts the board somewhere inside it. If padding-driven placement is the
residual, mapping the board onto the support's bounding box should recover the placement points.

## Three placement recipes already measured

| recipe | placement points | fraction of span | receipt |
|---|---|---|---|
| sampled at the predicted lattice (winner arm) | `-0.114151` | `-0.257557` | `m30-operating-point-r1` |
| sampled at the predicted lattice (mass arm) | `-0.103647` | `-0.233858` | `m32-mass-match-r1` |
| constant lattice over the full capture | `-0.118649` | `-0.267707` | `m33-fixed-lattice-r1` |

All three sit in a `0.015002`-wide band, which is why this pass also has to settle whether placement is
reducible at this capacity rather than assume it.

## The mechanism, exactly

`--occupancy-pixel-query 7`, one declared form of the existing pixel read:

- **Sampling placement**: `pixel_support_lattice(valid, rows, columns, device, dtype)` in
  `src/tetris_vision/model.py`. Per board it takes the bounding box of the valid pixels
  (`[left, right) x [top, bottom)` in image fractions) and maps the board's rows and columns onto it:
  `origin = (x0, y1)`, `column = ((x1-x0)/columns, 0)`, `row = (0, -(y1-y0)/rows)` — bottom-first, the
  conventions `decode_lattice` uses. It reads the raster mask only: no label, no prevalence, no scored
  cell, no fitted, learned or board-derived-from-labels quantity.
- **Statistics and projection**: unchanged — nine sub-cell samples, `30` features, and the M29/M30 form-3
  hidden layer `Linear(30, 64) -> SiLU -> Linear(64, dimension)` with a zero-initialized output, so step
  zero is the published function and the form adds no parameter.
- **Single change** against the M30/M31 winner (`m30-level-calibrate-r5`): same seed, same source
  checkpoint, same 64 steps, same 1,152 MiB ceiling. The consumer's boundary stays the fixed global `0.0`.

## Protocol

One profile, one 64-step study, one 36-board probe under the `m34-support` owned family (1,152 MiB study,
640 MiB probe). Scored by `scripts/m34_support_lattice_read.py` on the same 2,986 cells and both fixed
label vectors, joining the unchanged control `0.219626`, the reference `0.543974` and the pixel ceiling
`0.987179`; the winner anchor is re-scored in the same pass.

## Gate thresholds, unchanged

1. per-cell F1 at `0.0` `>= 0.70`, reported on both fixed label vectors;
2. ranking: pooled AUC `>= 0.90` **and** best swept F1 `>= 0.70`;
3. attribution over the unchanged control `>= 0.05`.

## Branch interpretations, pre-committed

- **PASS** (per-cell clears on both vectors): state plainly that this is a change in the read and quantify
  every moved condition against the winner; keep the split as the account of the gain.
- **FAIL**: report the same gate block and the split, name the short stage, and record the concrete next
  structural step that stage implies.

## The placement-reducibility rule, pre-committed before scoring

Placement is declared **reducible** by a sampling recipe only if this arm's placement points exceed the
best previously measured placement (`-0.103647`) by at least the declared floor **`0.02`**, i.e.
`>= -0.083647`.

- If it does, the path is live and the next step is whatever the split's next-shortest stage implies.
- If it does **not**, the conclusion is recorded as: at this capacity and on this population, placement is
  not reducible by choosing where the statistics are sampled — three independent recipes, including one
  derived from the raster's own support, land inside a `0.015002`-wide band and none exceeds the floor.
  The campaign then stops spending passes on placement and the live path is representation, which carries
  more than a full span in every arm measured so far.

## Receipts

`runs/m34-support-{profile,study,probe}` receipts and `docs/evidence/m34-support-lattice-r1/scope.json`.
Refused attempts keep their receipts; only completed runs are quoted.

## Measured

Study `m34-support` (probe `-registration`) on the M30/M31 winner's arm with form 7,
the unchanged 64-step protocol, peak `1,057,873,920` bytes. Scored by
`scripts/m34_support_lattice_read.py` on the fixed 2,986 cells and both label vectors
(`docs/evidence/m34-support-lattice-r1/scope.json`), the winner anchor re-scored in the same pass, the
placement comparison joined by the recorded fields of the three earlier receipts. No oracle, no
per-board rule, no fitted threshold.

| read | F1 at `0.0` | AUC | best swept F1 |
|---|---|---|---|
| M30/M31 winner (anchor, re-scored) | `0.372881` | `0.954529` | `0.689775` |
| support-lattice form, identity vector | `0.274869` | `0.883260` | `0.385925` |
| delta | -0.098012 | -0.071268 | -0.303850 |
| support-lattice form, anchored vector | `0.277264` | `0.756305` | `0.381148` |

Gate block, unchanged thresholds: identity per-cell `false`, ranking
`false`, attribution `true`
(+0.166299 over the unchanged control `0.219626`); anchored per-cell
`false`, ranking
`false`, attribution
`true`. `cleared: false`.

**The gate branch is fail, and the placement question is answered yes.** The support-derived lattice
does what it was declared to test - it removes the placement term - but it does so at a cost the gate
sees: F1 at `0.0` fell -0.098012 and best swept fell
-0.303850 against the winner, with AUC -0.071268.

## The placement term across every recipe the campaign has measured

| recipe | placement points | fraction of span | receipt |
|---|---|---|---|
| sampled at the predicted lattice (winner arm) | `-0.114151` | `-0.257557` | `m30-operating-point-r1` |
| sampled at the predicted lattice (mass arm) | `-0.103647` | `-0.233858` | `m32-mass-match-r1` |
| constant lattice over the full capture | `-0.118649` | `-0.267707` | `m33-fixed-lattice-r1` |
| **support-derived lattice (this arm)** | **`-0.004778`** | **`-0.010780`** | `m34-support-lattice-r1` |

The rule fixed before scoring: placement counts as reducible only if it reaches the best previous value
(`-0.103647`) plus the declared floor
(`0.020000`), i.e. `-0.083647`.
This arm reaches `-0.004778`, so **`placement_reducible: true`** - and the margin is not
marginal: the term shrinks from -0.118649 to -0.004778, a
0.113871 improvement, and the placement residual is now
`0.004778` points, which is `0.010781` of the span.

**What the support-extent statistics imply.** The three earlier recipes all sampled inside the *padded*
capture, and each paid a placement cost of `0.10`-`0.12` points; the one recipe that uses the raster's own
valid extent pays `0.004778`. So the placement error measured in M30-M33 was
padding-driven, exactly as the M33 receipt's recorded step supposed - a padded capture puts the board
somewhere inside the frame, and a sample grid that assumes the frame is the board misreads where the
cells are. That is a reducible error, and it is now reduced.

## The span split: placement leaves the negative side, closure becomes the short stage

| term | points | fraction of the `0.443205` span |
|---|---|---|
| closure | -0.158049 | -0.356605 |
| placement | -0.004778 | -0.010780 |
| representation | +0.606031 | 1.367384 |

The split sums to the span in points and to one in fractions, asserted in code. `short_stage:
closure`. Placement is no longer the short stage - it is now the *smallest* term in the split,
`0.010780` of a span where it used to be `0.23`-`0.27`. The cost moved into
closure: the same 64-step protocol trained on support-lattice statistics reaches
`0.385925` best swept F1 against the winner's `0.689775`, so correcting where the
read looks left the per-cell decision itself under-fitted, and representation carries
`1.367384` of the span.

## The concrete next structural step this stage implies

Closure, at the placement that is now correct: the read's decision needs the statistics' *within-sample
contrast* rather than their absolute level. Declared step for the next pass: keep form 7's support
lattice and the same nine-sample statistics, and give the hidden projection both the `30` absolute
statistics and their deviation from the sample's own mean along the lattice axes (`30` more) - computed
from the sample itself, no fitted constant, no label, no scored cell, and no additional output parameter.
The winner's absolute-level read then has to compete with the same information expressed relative to the
board it is reading.

## What this establishes, and what it does not

- Placement is reducible at this capacity, by a mechanism that reads only the raster: `-0.103647` to
  `-0.004778`. The campaign's placement error was padding-driven, and the campaign does not need to stop
  spending passes on placement - it needs to stop spending them on *sampling recipes*.
- The winner is still the M30/M31 arm: this pass buys the placement term at `0.303850`
  best swept F1, which no gate accepts, so it is a diagnostic arm, not a new operating point.
- Not shown: whether the contrast read recovers closure; whether closure at a correct placement is
  reachable at this capacity at all. One seed, 64 steps, training-fit boards, synthetic gates only.

## Receipts

`runs/m34-support-{profile,study,probe}` receipts and `docs/evidence/m34-support-lattice-r1/scope.json`. Refused attempts keep their receipts; only completed runs are quoted.

## M35 update: the contrast read was measured, and it moves closure

The step recorded here was run (`docs/M35_CONTRAST_READ.md`,
`docs/evidence/m35-contrast-lattice-r1/scope.json`): at this arm's placement, giving the projection the
statistics' sample-mean deviation lifts best swept F1 from `0.385925` to `0.570048`
(`+0.184123`) and closure from `-0.158049` to `+0.026074`, though the
predeclared bar (`0.372881` / `0.689775`) is still not met and precision at the fixed boundary is
`0.161787`. So the under-fit recorded here was a feature-level problem, and the next step is structural:
the M36 monotone contrast read.

