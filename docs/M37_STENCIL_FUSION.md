# M37: the coarse contrast channel fused into the pixel read

Status: **predeclared before any M37 scoring run.** Mechanism, declared widths, protocol, the unchanged
gate thresholds, the deciding bar and both branch interpretations are fixed here.

## Where this comes from

`docs/evidence/m35-contrast-lattice-r1/diagnosis.json` measured the contrast read's remaining failure:
precision `0.161787` at recall `0.981308`, `1,088` false positives whose margins (`0.003737`-`4.629019`,
median `1.352717`) overlap the true positives' (`0.936171`-`5.464106`), concentrated in the dense upper
bands (`405` in rows 0-4, `528` in rows 5-9). `docs/evidence/m36-monotone-contrast-r1/scope.json` then
refused the structural answer (best swept `0.420935` against M35's `0.570048`). Both receipts record the
same next step, and this pass takes it: the campaign's one measured instrument for telling a filled cell
from a *textured empty* one is the coarse lattice-free channel, whose `stencil_contrast` form is exact
under a board-wide gain and offset (`docs/evidence/m32-mass-match-r1`, the M22/M23 stencil line), and no
support-lattice arm has yet fused it with the pixel statistics.

## The mechanism, exactly (form 11)

`--occupancy-pixel-query 11`, one declared form, one head block, no second occupancy selector:

- **Placement**: unchanged from form 7 - `pixel_support_lattice`, the bounding box of the capture's valid
  pixels, bottom-first.
- **Pixel half**: the unchanged nine sub-cell samples over the declared interior extent, `30` statistics
  (`27` channel values + `3` per-channel spreads).
- **Coarse half**: the encoder's coarse feature map pooled to a fixed `(rows+2) x (columns+2)` box lattice,
  each box passed through the already-declared `stencil_contrast` transform (each box against the mean and
  spread of its own `5x5` neighbourhood, so a board-wide gain or offset cancels by construction), and then
  summarised per cell over the five declared taps (own, above, below, left, right) by **channel mean and
  channel spread** - a fixed ten-number summary, `PIXEL_QUERY_STENCIL_FEATURES = 10`.
- **Fusion**: `torch.cat` of the two halves, `40` inputs, through the unchanged M29/M30 hidden projection
  `Linear(40, 64) -> SiLU -> Linear(64, dimension)` with a zero-initialised output. Step zero is therefore
  the published function, no new output parameter exists, and against the winner the form adds
  `10*64 = 640` parameters, all in the hidden input width.
- **Boundary**: the consumer's fixed global `0.0`. Nothing is fitted, no label, no prevalence, no scored
  cell, no threshold.

## Protocol

One profile, one 64-step study, one 36-board probe under the `m37-stencil` owned family (1,152 MiB study,
640 MiB probe), scored by `scripts/m37_stencil_read.py` on the same 2,986 cells and both fixed label
vectors, joining the unchanged control `0.219626`, the reference `0.543974` and the pixel ceiling
`0.987179`; the winner is re-scored in the same pass. The reader also compares this arm's error structure
with M35's recorded diagnosis.

## Gate thresholds, unchanged

per-cell F1 at `0.0` `>= 0.70` on both fixed label vectors; ranking (AUC `>= 0.90` and best swept
`>= 0.70`); attribution over the unchanged control `>= 0.05`.

## The deciding bar, fixed before scoring

The arm is the **new operating point** only if it matches or beats the winner's re-scored numbers:

- F1 at `0.0` `>= 0.372881`, and
- best swept F1 `>= 0.689775`.

**A refusal means**: this fusion does not earn the operating point, whatever it does to the diagnosis, and
the campaign's next change must come from the measured error structure - not from another feature family,
another sampling recipe, another threshold, or a re-swept operating point.

## Branch interpretations, pre-committed

- **PASS**: promote the arm, re-run its probe and reproduce weights and logits (not the earlier all-zero
  delta by assumption), and record the next structural step.
- **FAIL**: diagnose from the measured evidence which cells fail, at what support, with what margins, and
  whether the textured-empty false positives that carried M35's `1,088` (median margin `1.35`) are now
  separated; then take one structural change attacking that cause (capacity or decision structure if the
  representation now separates), predeclared with its own bar, implemented and measured in this pass.

## Receipts

`runs/m37-stencil-{profile,study,probe}` and `docs/evidence/m37-stencil-fusion-r1/{scope,diagnosis}.json`.
Refused attempts keep their receipts; only completed runs are quoted.

## Measured

Study `m37-stencil` (probe `-registration`), form 11 with `40`
declared inputs, the unchanged 64-step protocol, peak `1,028,612,096` bytes. Scored by
`scripts/m37_stencil_read.py` (`docs/evidence/m37-stencil-fusion-r2/scope.json`, diagnosis in
`diagnosis.json`), the winner re-scored in the same pass.

| read | F1 at `0.0` | AUC | best swept F1 |
|---|---|---|---|
| M30/M31 winner (re-scored) | `0.372881` | `0.954529` | `0.689775` |
| fused coarse channel, identity vector | `0.270096` | `0.919512` | `0.589286` (at `3.15`) |
| delta | -0.102785 | -0.035016 | -0.100489 |
| fused coarse channel, anchored vector | `0.282424` | `0.757793` | `0.495697` |

Gate block: identity per-cell `false`, ranking
`false`, attribution `true`
(+0.369660 over control `0.219626`).

**The fusion is refused.** It lifts the best swept point over M35's by
+0.019238 and moves closure to `+0.045312`, but AUC falls
-0.035016 and the bar (`0.372881` / `0.689775`) is missed on both comparisons.

## The diagnosis: the coarse channel does not separate the textured-empty cells

| measurement | M35 (60 inputs) | this arm (40 inputs) |
|---|---|---|
| false positives at `0.0` | `1088` | `1131` |
| false-positive median margin | `1.352717` | `1.558202` |
| weakest true-positive margin | `0.936171` | `0.804033` |
| upper-band false positives (rows 0-9) | `933` | `932` |
| `textured_empty_separated` | - | **`false`** |

Per band (before -> now): rows 0-4 `405` ->
`405`, rows 5-9 `528` ->
`527`, rows 10-14 `11` ->
`16`, rows 15-19 `144` ->
`183`. Precision `0.1566` at recall
`0.981308`; true positives hold margins `0.804033`-
`5.765948` (median `3.551605`), false positives
`9.3e-05`-`5.563735`.

So the coarse contrast channel adds **nothing** to the separation this arm needs: the false positives are
the same cells, their margins are higher than M35's, and they still outrank the weakest true positive. The
declared exact-invariance of the transform is real but irrelevant here - these captures' textured-empty
cells are not a board-wide gain or offset away from filled ones.

## The span split

| term | points | fraction of the `0.443205` span |
|---|---|---|
| closure | +0.045312 | 0.102236 |
| placement | -0.093589 | -0.211163 |
| representation | +0.491482 | 1.108927 |

Sums asserted in code; `short_stage: placement`.

## The ladder, and what the numbers now say about the constraint

| arm | inputs | F1 at `0.0` | AUC | best swept |
|---|---|---|---|---|
| M30/M31 winner (predicted lattice) | 30 | `0.372881` | `0.954529` | `0.689775` |
| M34 support lattice | 30 | `0.274869` | `0.883260` | `0.385925` |
| M35 contrast | 60 | `0.277778` | `0.945596` | `0.570048` |
| M36 monotone contrast | 60 | `0.313667` | `0.886350` | `0.420935` |
| M37 fused coarse channel | 40 | `0.270096` | `0.919512` | `0.589286` |

Every support-lattice arm's best swept point is computed at its own optimal threshold, so the shortfall is
not the fixed boundary: at `0.689775` the winner's arm is simply better at ranking cells than
any of these four, whatever threshold they are given. Three feature families have been tried; the
remaining untried lever inside this arm class is the read's own capacity (a 64-wide hidden layer for
`40`-`60` inputs), which `docs/M38_CAPACITY.md` predeclares and measures in the same pass.

## Receipts

`runs/m37-stencil-job.json` (study, peak `1,028,612,096`), `runs/m37-stencil-registration-job.json` (probe), `runs/m37-stencil-profile-job.json` (admissibility) and `runs/m37-stencil-fusion-r2-job.json` (this score). The first score, `runs/m37-stencil-fusion-r1-job.json` with its evidence directory, is preserved as a superseded record: V-162, the reader reported M35's inherited input width.
