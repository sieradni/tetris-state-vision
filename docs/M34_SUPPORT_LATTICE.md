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
- **Statistics and projection**: unchanged — nine sub-cell samples, `31` features, and the M29/M30 form-3
  hidden layer `Linear(31, 64) -> SiLU -> Linear(64, dimension)` with a zero-initialized output, so step
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
