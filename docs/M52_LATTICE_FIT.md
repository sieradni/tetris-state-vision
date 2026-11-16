# M52: does an interior-fit lattice refinement buy the last of the read back?

Status: **predeclared before any scoring run.** The refinement, its fixed constants, the protocol, the four
comparable reads, the falsifier, the no-regression rule and every branch are fixed below. The measured section is
added after the run and is not part of the declaration.

## The question this arm answers

M51 (`docs/M51_DETECTOR_REPAIR.md`, `docs/evidence/m51-board-box-read/scope.json`) ran M50's named successor and landed
`repair-under-specified`: one declared objective over wall and cell-grid candidates found the board box on every
capture to within `0.006892` cells in width and `0.004720` in height, cut both of M50's outliers (`13.509043` ->
`1.9285` and `7.389043` -> `5.2098`), and still missed the falsifier - population mean displacement `0.383358` cells
against the declared `0.25` - while the read fell to `0.904376` F1 at the fixed `0.0` crossing and `0.933275` AUC, and
two boards M50 had placed (`training-20261031-v1` / `-v2`) degraded from `0.0588` to `0.9241` cells. M51's own measured
diagnosis was that the box is right and the **pitch** is not: a pitch error of a few percent accumulates across ten
columns, which is why the read lost `0.075` F1 while the box was nearly the right size.

M51 named one successor: refine the lattice by fitting the interior block edges instead of trusting the windowed
search. This arm runs that successor.

## Motivation disclosure, stated before the run

The refinement is **motivated by M51's measured residual** - the four boards over the gate, the pitch values the fit
now corrects, and the read loss attributed to pitch. That is information from the measured population leaking into the
instrument's *choice of question*, and it is disclosed rather than hidden. Three things bound the damage:

- the refinement is one declared procedure with fixed constants, applied in the same order to every capture: no branch
  on capture identity, sample id, group, split, orientation, block count or aspect, and no per-board parameter choice;
- the falsifier is judged on the **full 28-board population**, not on the four boards that motivated the arm, and every
  board's displacement is reported;
- the no-regression rule is judged against **both** M50's and M51's committed per-board tables, so a refinement that
  buys the outliers by degrading the boards either earlier arm placed is a failure.

One stage was designed and **withdrawn before this declaration**, and it is recorded here so the instrument is not
mistaken for a search over weights: a global phase alignment (the offset within one cell, at `0.25`-pixel steps, where
the fitted lines carry most edge energy) measured *worse* on the population - mean `0.400312` cells and
median-of-medians `0.101008`, against `0.380890` / `0.074716` without it - and was removed. The instrument declared
below is the one that survived that comparison, which is itself a measured choice and is disclosed as such.

## The refinement, exactly

`scripts/lattice_fit.py::refine`, fixed constants: `SEARCH_SHARE = 0.35`, `MISSING_SHARE = 0.35`,
`OUTLIER_CELLS = 0.25`, `MIN_LINES = {"x": 6, "y": 12}`.

1. **Input**: the raster and M51's detected box (`scripts/box_search.py::detect_box`), unchanged. Nothing else.
2. **Profiles**: per-column and per-row edge energy inside the capture (`|difference|` along the perpendicular axis,
   summed).
3. **Per-boundary localisation**: for each expected boundary `origin + k * pitch` (`k = 0..10` in x, `0..20` in y),
   the strongest profile peak within `SEARCH_SHARE` of a cell is found and its centre is the profile's centroid over a
   three-sample window around that peak - a sub-pixel line centre.
4. **Missing boundaries**: a boundary whose peak is below `MISSING_SHARE` of the median boundary peak is *missing* (a
   board drawn without a grid line there) and is not used.
5. **Robust fit**: the pitch is the **Theil-Sen** slope, the median of all pairwise slopes over the used boundaries, so
   one displaced line cannot move it; the origin is the box origin plus the mean residual of the used boundaries
   against the box's own expectation, so the box's pixel convention is preserved exactly.
6. **Outlier rule**: boundaries whose residual differs from the median residual by more than `OUTLIER_CELLS` cells are
   dropped once and the fit is repeated.
7. **Declared fallback**: if fewer than `MIN_LINES` boundaries survive on an axis, that axis keeps the detected box's
   pitch and origin unchanged.
8. **Lattice**: `origin = (left, top + 20 * pitch_y)`, `u = (pitch_x, 0)`, `v = (0, -pitch_y)` - the campaign's
   conventions, the same diagonal map into image pixels every arm applies.

It reads the raster and M51's label-free box only: **no label, no prevalence, no scored cell, no use of the target
lattice or the model's predicted lattice at refinement time.**

## Protocol, unchanged from M49/M50/M51

- **read-out**: `runs/m3-occpix-ceiling-r1-r2/result.json`, logistic, ridge `1.0`, forty iterations, clip `+/-8`,
  leave-one-ancestry-group-out, the thirty statistics, accepted **only** if it reproduces the committed logits to
  `1e-6`; then the same fitted folds transform the refined features. Nothing is fitted on this arm's inputs and no
  threshold is fitted.
- **population**: the same `2,986` cells on the same `28` nonempty boards, the same folds, the fixed `0.0` crossing.
- **four reads, one pass**: `target_lattice` (scoring only, never registration), `predicted_lattice` (M48),
  `detected_box` (M51's box, re-measured here), `refined_lattice` (this arm), plus the winner control
  `m30-level-calibrate-r5` at checkpoint `eddec1fd4af2…` from its committed scope.

## The falsifier and the branches, fixed now

**Falsifier (inherited from M49/M50/M51, judged on the full 28-board population):** the population mean
refined-vs-target displacement is greater than **`0.25` cells, or** the refined AUC is below **`0.90`**.

Branches, exactly one of which the receipt reports:

- **`fit-general`** - mean displacement `<= 0.25` cells **and** AUC `>= 0.90`: the pitch is recoverable from the
  raster's interior edges on every board and the read comes back.
- **`fit-under-specified`** - the median of the per-board mean displacements is `<= 0.25` cells **and** AUC `>= 0.90`,
  but the population mean is not: the median board is placed and the residual lives on a minority of captures where
  the pitch, not the box, is under-determined.
- **`fit-fails`** - neither of the above: most boards are misplaced, so fitting interior cell lines is ruled out as the
  geometry source for this population.

**Prediction, standing as written whatever the outcome.** From the instrument check below I predict the study lands
**mean displacement in `[0.30, 0.45]` cells - i.e. the falsifier still fires on the displacement leg - with
median-of-medians in `[0.06, 0.10]` cells, F1 at `0.0` in `[0.93, 0.99]` and AUC `>= 0.96`**, and therefore the branch
**`fit-under-specified`**. The pitch fix should raise the read above M51's `0.904376`; if it does not, that is the
measurement that the pitch was not the read's binding term, and the measured section must say so.

**No-regression rule, fixed now.** Two tables, both judged over the whole set:

- against **M50**: every board M50 placed within `0.25` cells (`26` boards) must stay `<= max(0.25, M50 + 0.10)`;
- against **M51**: every board M51 placed within `0.25` cells (`24` boards) must stay `<= max(0.25, M51 + 0.10)`;

and, named explicitly because M51 degraded them, `training-20261031-v1` and `training-20261031-v2` are judged against
their **M50** values (`0.0588`), so their allowance is `0.1588` cells - not M51's `0.9241`. Breaches are listed by
sample id in the receipt.

## Instrument check (declared, run before the study)

`scripts/m52_lattice_fit_read.py --detect-only`, receipt `runs/m52-lattice-fit-detect-r3-job.json`, output
`runs/m52-lattice-fit-detect-r3/detection.json` - detection and refinement only, no scoring, no model:

| statistic | M52 refined | M51 box |
| --- | --- | --- |
| mean of per-board mean displacement | `0.380890` cells | `0.383358` |
| median of per-board means | `0.380191` cells | `0.383519` |
| median of per-board medians | `0.074716` cells | `0.066438` |
| worst board mean | `5.177659` cells | `5.209784` |
| worst single-cell displacement | `5.368246` cells | `5.432274` |
| mean width error | `0.001776` cells | `0.006892` |
| mean height error | `0.002611` cells | `0.004720` |
| boards where both axes kept the fit | `22` / `28` | - |

Per-board means, as this arm measures them: `training-20261030-v1` `5.1777` (M51 box `5.2098`),
`training-20261035-v2` `1.9260` (`1.9285`), `training-20261031-v1` `0.9497` (`0.9241`), `training-20261031-v2`
`0.9496` (`0.9241`). The refinement tightens the scale by a factor of `3.9` in width and `1.8` in height, moves the
population mean by `0.002468` cells, and *costs* `0.008` cells on the median board - a trade that the prediction above
and the no-regression rule are written to expose rather than hide.

The prediction is made from this table, not from the study.
