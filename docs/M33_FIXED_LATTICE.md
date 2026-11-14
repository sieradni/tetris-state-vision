# M33: a fixed-lattice placement for the pixel-statistics read

Status: **predeclared before any M33 scoring run.** Mechanism, protocol, unchanged gate thresholds
and both branch interpretations are fixed here; the measured section is appended from the final
receipt's own fields.

## The hypothesis, from the M32 receipt's fields

`docs/evidence/m32-mass-match-r1/scope.json` measured the mass/occupancy-matching term as negative
(F1 at `0.0` `0.277778` against the winner's `0.372881`; best swept `0.602198` against `0.689775`) and
its span split put **placement** as the only negative term (`-0.103647` points, `-0.233858` of span)
while representation still carries more than a full span (`1.102488`). The read's pixel statistics are
sampled at the *predicted* lattice, so every placement error the decoder makes is inherited by the
features that are supposed to be immune to it. The hypothesis is that sampling the same statistics at a
fixed lattice over the capture removes that inheritance by construction and recovers the placement
points.

## The mechanism, exactly

`--occupancy-pixel-query 6`, one declared form of the existing pixel read:

- **Sampling placement**: `pixel_fixed_lattice(rows, columns, device, dtype)` in
  `src/tetris_vision/model.py` — three vectors mapping cell coordinates to image coordinates over the
  whole capture (`origin = (0, 1)`, `column = (1/columns, 0)`, `row = (0, -1/rows)`), the same
  conventions `decode_lattice` uses. It contains no learned, fitted, board-derived or scored-cell
  quantity; it is a constant of the capture's geometry.
- **Statistics**: unchanged `cell_pixel_statistics` — nine sub-cell samples over the declared interior
  extent, twenty-seven channel values and three per-channel spreads (`31` features).
- **Projection**: the M29/M30 form-3 hidden layer, `Linear(31, 64) -> SiLU -> Linear(64, dimension)`,
  output zero-initialized so step zero is the published function (V-143 rule).
- **Constants**: none new; the declared sub-cell extent and sample count are the existing ones. No
  weight is introduced — the change is spatial, not a loss term.
- **Single change against the M30/M31 winner** (`m30-level-calibrate-r5`: form 3 + read
  standardisation). Same seed, same source checkpoint, same 64 steps, same 1,152 MiB ceiling. Inference
  boundary stays the fixed global `0.0`; nothing is fitted on scored cells.

The helper is a pure function of the board's row and column counts, so the read cannot depend on the
decoded lattice it is measured against — that invariance is the point and is tested directly.

## Protocol

One profile, one 64-step study, one 36-board attribution probe under the `m33-fixed-` owned family
(1,152 MiB study ceiling, 640 MiB probe). Scoring by `scripts/m33_fixed_lattice_read.py` on the same
2,986 cells and both fixed label vectors, joining the unchanged control `0.219626`, the reference
`0.543974` and the pixel ceiling `0.987179`. The winner anchor is re-scored in the same pass, not
quoted.

## Gate thresholds, unchanged

1. per-cell F1 at `0.0` `>= 0.70` on the identity vector (and reported for the anchored vector);
2. ranking: pooled AUC `>= 0.90` **and** best swept F1 `>= 0.70`;
3. attribution over the unchanged control `>= 0.05`.

No threshold, population, metric, boundary or per-board rule changes.

## Both branches, pre-committed

- **PASS** (per-cell clears on both fixed label vectors): record plainly that this is a change in the
  read; quantify every moved condition (per-cell, AUC, best, attribution) against the winner and keep
  the span split as the account of where the gain came from.
- **FAIL**: report the same gate block and the span split, name the stage that is now short, and record
  in the receipt the concrete next structural step that stage implies — not a hyperparameter sweep.

## Receipts

`runs/m33-fixed-{profile,study,probe}` receipts plus `docs/evidence/m33-fixed-lattice-r1/scope.json`.
Refused attempts keep their receipts; only completed runs are quoted.
