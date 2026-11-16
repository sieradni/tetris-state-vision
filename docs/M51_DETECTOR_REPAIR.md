# M51: does a repaired board-box detector buy the read back?

Status: **predeclared before any scoring run.** The detector, its fixed constants, the protocol, the four comparable
reads, the falsifier, the no-regression rule and every branch are fixed below. The measured section is added after the
run and is not part of the declaration.

## The question this arm answers

M50 (`docs/M50_BOARD_BOX.md`, `docs/evidence/m50-board-box-read/scope.json`) ran M49's named successor and landed
`box-partial`: a label-free wall-run detector put the committed read-out back to **`0.979200`** F1 at the fixed `0.0`
crossing and **`0.999150`** AUC, but the declared falsifier still fired - the population mean registered-vs-target
displacement was **`0.802227`** cells against the declared `0.25` - and the tail was two boards:
`training-20261035-v2` at `13.509043` cells and `training-20261030-v1` at `7.389043`. Both failures had the same cause,
named in M50's receipt: the detector's primary path needs a **full-height wall pair** covering all twenty rows, and on
those two captures the pairing locked onto a ten-by-twenty rectangle formed by a HUD panel edge and a well border, so
the whole lattice shifted by six or more columns.

M50 named one successor: replace or guard that single generator with an a-priori candidate search that picks a box from
label-free evidence, evaluated identically on **all 28 boards**, with no branch keying off capture identity. This arm
runs that successor.

## Leakage disclosure, stated before the run

The fallback this arm tests was **motivated by the two named outlier captures**. That is a real leak of information
from the measured population into the instrument's design, and it is disclosed here rather than hidden: a reader is
entitled to treat any improvement on those two boards as partly fitted to knowing which boards failed. Three things
bound the damage and are declared now:

- the detector is a **single declared objective** over candidates, with fixed constants, applied in the same order and
  with the same weights to every capture - there is no branch on capture identity, sample id, group, split, count of
  blocks, orientation or aspect of the capture;
- the falsifier is judged on the **full 28-board population**, not on the repaired subset, and every board's
  displacement is reported so a reader can see whether the repair generalises or merely patched the two;
- the no-regression check below is judged on the 26 boards M50 already placed, so a repair that buys the two outliers
  by degrading the rest is a failure, not a success.

If the arm clears, the measured section must say plainly whether the clean number comes from a general mechanism or
from knowledge of which boards failed; this declaration does not pre-empt that judgement.

## The detector, exactly

`scripts/box_search.py::detect_box`, fixed constants: `EDGE = 12.0`, `WALL_SHARE = 0.5`, `CLUSTER = 6`,
`MIN_WIDTH = 60`, `MAX_WIDTH = 400`, `ASPECT_SLACK = 0.06`, `SHARE_FLOOR = 0.25`, `SHARE_STEP = 0.05`,
`BOARD_SPAN = 0.3`, `CELL_MIN = 10.0`, `CELL_MAX = 22.0`, `CELL_STEP = 0.1`, `CELL_PHASE_STEP = 0.5`,
`SATURATION = 60.0`, `STACK_WEIGHT = 0.5`, `FLOOR_BAND = 0.25`.

1. **Edges**: per-pixel channel-mean difference above `EDGE` along each axis gives the vertical and horizontal edge
   masks and the per-column and per-row edge energy.
2. **Wall candidate** (M50's generator, kept as one candidate): line candidates are the columns and rows whose longest
   edge run reaches `BOARD_SPAN` of the axis - a board border runs across most of the capture, so this finds a faint
   border that a share of the brightest line would miss - relaxed to `WALL_SHARE` of the strongest run (stepping down
   to `SHARE_FLOOR`) only when that finds no pair. The box is the wall pair bracketing a width in
   `[MIN_WIDTH, MAX_WIDTH]` with a row pair within `ASPECT_SLACK` of `height = 2 * width`, ranked by the weaker of the
   four runs then by area; the declared fallback when no rectangle is aspect-valid is the pairing ranked by area then
   by the weakest run. The box is read off the **inner** edges of the four clusters.
3. **Grid candidate** (new): for each pitch in `[CELL_MIN, CELL_MAX]` at `CELL_STEP`, the eleven column lines and the
   twenty-one row lines are placed at that pitch over every phase in `CELL_PHASE_STEP`, and the contiguous window of
   those lines is scored by **line energy over the summed energy at the cell midpoints**. That contrast, not a plain
   sum or a mean, is what separates the true cell pitch from a fraction of it (which can always land every line on a
   block edge) and from a sparse pitch that only touches the board's border. The pitch maximising the two axes' scores
   is used for both axes - the cells are square - and its windows are the box.
4. **The objective, one for every candidate**: `wall_support + grid_alignment + (1 - cell_uniformity) +
   STACK_WEIGHT * stack_containment`, where `wall_support` is the weakest of the four boundary runs over the longest
   run on either axis, `grid_alignment` is the mean edge energy at the ten interior column lines and nineteen interior
   row lines over the axis maxima, `cell_uniformity` is the mean interior standard deviation over the cells' central
   70 % normalised by the frame's own, and `stack_containment` is the share of saturated pixels that fall inside the
   box. The arg max wins; if a candidate generator raises, its note is recorded and the other candidates are scored.
5. **Lattice**: `origin = (left, bottom)`, `u = (width/10, 0)`, `v = (0, -height/20)` - the campaign's conventions,
   the same diagonal map into image pixels M48/M49/M50 applied.

It reads the raster and the declared constants only: **no label, no prevalence, no scored cell, and no use of the
target lattice or the model's predicted lattice at detection time.** The target lattice is used only *after*
detection, to report the detector's own accuracy.

Two constants were considered during design and are **not** part of the instrument because their declared terms were
measured to make the population worse and were reverted: an area term (`+1.0 * box area / frame area`) and a
gravity/floor term; the border-share guard inside the pitch search was reverted for the same reason. They are recorded
here so the reader knows the instrument was not selected by searching over weights on the measured population beyond
what is disclosed above - the objective above is the one that was declared, and the study is the first scoring run
under it.

## Protocol, unchanged from M49/M50

- **read-out**: `runs/m3-occpix-ceiling-r1-r2/result.json`, logistic, ridge `1.0`, forty iterations, clip `+/-8`,
  leave-one-ancestry-group-out, the thirty statistics, accepted **only** if it reproduces the committed logits to
  `1e-6`; then the same fitted folds transform the detected-box features. Nothing is fitted on this arm's inputs and
  no threshold is fitted.
- **population**: the same `2,986` cells on the same `28` nonempty boards, the same folds, the fixed `0.0` crossing.
- **four reads, one pass**: `target_lattice` (the ceiling's own lattice - scoring only, never registration),
  `predicted_lattice` (M48's committed predicted boxes), `detected_box` (this arm's candidate search), and the winner
  control `m30-level-calibrate-r5` at checkpoint `eddec1fd4af2…`.

## The falsifier and the branches, fixed now

**Falsifier (inherited from M49/M50, judged on the full 28-board population):** the population mean registered-vs-target
displacement is greater than **`0.25` cells, or** the registered AUC is below **`0.90`**. Displacement is the same
per-cell measure M49/M50 used: the mean over the twenty rows of the mean over each row's ten cells of the distance from
the detected cell centre to the target cell centre, in detected-cell units.

Branches, exactly one of which the receipt reports:

- **`repair-general`** - mean displacement `<= 0.25` cells **and** AUC `>= 0.90`: the candidate search places the
  statistics at the declared bars on the full population, so the read comes back and the geometry fix exists in the
  raster as a general mechanism.
- **`repair-under-specified`** - the median of the per-board mean displacements is `<= 0.25` cells **and** AUC
  `>= 0.90`, but the population mean is not: the median board is placed and a minority of captures carry more than one
  defensible box, so the search is under-determined there rather than wrong everywhere.
- **`repair-fails`** - neither of the above: most boards are misplaced, so a label-free candidate search over the
  raster is ruled out as the geometry source for this population.

**Prediction, standing as written whatever the outcome.** From the instrument check below I predict the study lands
**mean displacement in `[0.30, 0.50]` cells, median of per-board means `<= 0.10` cells, F1 at `0.0` in
`[0.95, 0.99]`, AUC `>= 0.998`** - i.e. the **falsifier fires on the displacement leg** and the branch is
**`repair-under-specified`**, with `repair-general` requiring the residual outliers to fall to within a quarter cell.
The prediction is a commitment, not a description: if the study lands elsewhere it lands elsewhere and the measured
section says so.

**No-regression rule, fixed now.** For every board whose M50 mean displacement was `<= 0.25` cells (the 26 boards M50
placed), this arm's mean must stay `<= max(0.25, M50 value + 0.10)` cells. Breaches are listed by sample id in the
receipt. The rule is judged over the whole 26, not a subset.

## Instrument check (declared, run before the study)

`scripts/m51_board_box_read.py --detect-only`, receipt `runs/m51-detector-check-r9-job.json`, output
`runs/m51-detector-check-r9/detection.json` - detection only, no scoring, no model:

| statistic | this arm | M50 |
| --- | --- | --- |
| mean of per-board mean displacement | `0.383358` cells | `0.802227` |
| median of the same | `0.383519` cells | `0.803901` |
| median of per-board medians | `0.066438` cells | `0.061328` |
| worst board mean | `5.209784` cells | `13.509043` |
| worst single-cell displacement | `5.432274` cells | `17.582818` |
| mean width error | `0.006892` cells | `0.043312` |
| mean height error | `0.004720` cells | `0.017508` |

Two former outliers, as this arm measures them: `training-20261035-v2` `1.9285` cells (M50 `13.509043`) and
`training-20261030-v1` `5.2098` cells (M50 `7.389043`). The declared no-regression rule is **not** cleared by the
instrument: `training-20261031-v1` and `training-20261031-v2` move from `0.0588` to `0.9241` cells. That is declared
here, before scoring, because it is exactly the cost the leakage disclosure warns about - buying the two outliers by
degrading two previously-good boards.

The prediction above is made from this table, not from the study.
