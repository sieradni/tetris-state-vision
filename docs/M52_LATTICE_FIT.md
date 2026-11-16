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
## Measured

Run: `scripts/m52_lattice_fit_read.py --admission runs/m52-lattice-fit-read-profile --output
docs/evidence/m52-lattice-fit-read`, receipt `runs/m52-lattice-fit-read-job.json` (`status: passed`, peak job commit
`740,196,352` B against the `848` MiB ceiling of `1,024` MiB), `28` boards, `2,986` cells, the fixed `0.0` crossing.
The declaration above is unchanged: the prediction is judged against it, not restated to fit it.

### The four reads, one pass, same cells and folds

| read | F1 @ `0.0` | AUC | best swept | TP / FN / FP / TN at `0.0` | precision | recall |
| --- | --- | --- | --- | --- | --- | --- |
| `target_lattice` (ceiling) | `0.974763` | `0.999195` | `0.987179` @ `1.05` | `309 / 0 / 16 / 2,661` | `0.950769` | `1.000000` |
| `refined_lattice` (this arm) | **`0.893268`** | **`0.908824`** | `0.896211` @ `0.90` | `272 / 37 / 28 / 2,649` | `0.906667` | `0.880259` |
| `detected_box` (M51's box, re-measured) | `0.904376` | `0.933275` | `0.908497` | `279 / 30 / 29 / 2,648` | `0.905844` | `0.902913` |
| `predicted_lattice` (M48) | `0.037152` | `0.442341` | `0.037152` | `6 / 303 / 8 / 2,669` | `0.428571` | `0.019417` |
| winner control `m30-level-calibrate-r5` | `0.372881` | `0.954529` | `0.689775` | - | `0.230430` | - |

M51's box reproduces its committed receipt exactly (`0.904376` / `0.933275` / `279-30-29`), so the two arms are
compared on one evaluator. The refinement is **worse** on the read: `-0.011108` F1 at `0.0` and `-0.024451` AUC
against the box, and `0.045705` AUC below the winner's own head.

### Per-row profiles at the fixed crossing

False positives by row (`refined_lattice`): `0: 4, 1: 0, 2: 1, 3: 5, 4: 2, 5: 2, 6: 4, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0,
12: 0, 13: 0, 14: 0, 15: 2, 16: 6, 17: 2, 18: 0, 19: 0` - bands `rows_0_4 = 12`, `rows_5_9 = 6`, `rows_10_14 = 0`,
`rows_15_19 = 10`. The early-row mass M51 measured is unchanged (`12` / `6`), so the refinement did not touch the
phase error it was aimed at.

### Margin distributions at the fixed crossing

| class | count | lowest margin | median margin |
| --- | --- | --- | --- |
| true positive | `272` | `1.491581` | `6.411599` |
| false positive | `28` | `0.311084` | `2.820802` |
| false negative | `37` | `-55.801426` | `-11.933784` |
| true negative | `2,649` | `-58.742410` | `-11.596236` |

### Displacement, all 28 boards

Population mean **`0.380890`** cells, median of per-board means `0.380191`, **median of per-board medians
`0.074716`**, mean of per-board medians `0.380191`, p90 `0.399150`, worst single-cell `0.405117`; worst board
`5.177659`; **`4` boards above the `0.25`-cell gate**; `22` / `28` boards kept the fit on both axes (six fall back to
the box on one axis). Against M51's box (`0.383358` mean, `0.066438` median-of-medians) the refinement moves the
population mean by `-0.002468` cells and the median board by `+0.008278` cells: it is a wash on location and a cost on
the typical board.

Per-board values for all 28 boards are in the receipt under `detector.boards` and `no_regression.*.boards`, each with
its M51 box value beside it; the four gate boards are `training-20261030-v1` `5.1777`, `training-20261035-v2` `1.9260`,
`training-20261031-v1` `0.9497`, `training-20261031-v2` `0.9496`.

### Scale and tightness

Mean width error **`0.001776`** cells and height error **`0.002611`** cells against M51's `0.006892` / `0.004720` - a
factor of `3.9` and `1.8` tighter, and against M49's frame-extent recipe (`3.325530`x the board's width, `1.736611`x
its height) the fitted lattice is the board to within `0.2 %` of a cell in width. **The scale part of the declaration
is met in full; the location part is not.**

### Falsifier, judged on the full population

The inherited falsifier is **met**: the population mean refined-vs-target displacement is `0.380890` cells, above the
declared `0.25`. The AUC leg (`0.908824 >= 0.90`) is not met, by `0.008824`. The branch is therefore
**`fit-under-specified`**, exactly as predicted - the median board is placed and the residual lives on four captures.

### The prediction, judged

| declared | predicted | measured | verdict |
| --- | --- | --- | --- |
| population mean | `[0.30, 0.45]` cells | `0.380890` | met |
| median of per-board medians | `[0.06, 0.10]` cells | `0.074716` | met |
| F1 at `0.0` | `[0.93, 0.99]` | `0.893268` | **missed low** |
| AUC | `>= 0.96` | `0.908824` | **missed low** |
| branch | `fit-under-specified` | `fit-under-specified` | met |

The F1 and AUC predictions were wrong, and in the direction that matters: I predicted the pitch fix would *raise* the
read above M51's `0.904376`, and it lowered it. That is the measured refutation of M51's diagnosis, and it is this
arm's main result.

### No-regression check, both baselines

**Not cleared.** Against M50's `26` placed boards the breaches are `training-20261031-v1` and `training-20261031-v2`
(`0.058827` -> `0.949682` / `0.949620`, allowance `0.158827`). Against M51's `24` placed boards there are no further
breaches - the two named boards are M51's own over-gate boards. The pair's numbers, as the receipt records them:

| board | M50 | M51 box | M52 box | M52 refined |
| --- | --- | --- | --- | --- |
| `training-20261031-v1` | `0.058827` | `0.924144` | `0.924144` | `0.949682` |
| `training-20261031-v2` | `0.058827` | `0.924144` | `0.924144` | `0.949620` |

So the refinement does not repair M51's regression; it deepens it by `0.0255` cells on both boards.

### The estimator's own limitation, measured

The boundary localiser is a three-sample profile centroid: sub-pixel, but biased toward the nearest integer with a
sawtooth of at most **`0.2997`** pixels (measured on a `1.2`-pixel-wide synthetic peak at offsets `0.1` to `0.9`, and
pinned by `tests/test_m52_lattice_fit.py::LocalisationTest`). At the captures' `12`-`18`-pixel pitches that is up to
`0.025` cells of phase error per boundary, which is the same order as the `+0.008278` cells the median board lost. The
bias is a property of the estimator, not of the population, and it is the concrete reason this arm's location gain is a
wash even though its pitch gain is real; a parabolic peak interpolation would remove it and is the natural first change
to any successor that keeps this fit.

### What holds: `under-specified`, and what it rules out

**Not `wrong`**: the fit does what it claims - pitch is recovered to `0.18 %` of a cell, `22` of `28` boards keep it on
both axes, and the median board sits at `0.0747` cells of median error.

**Not `registration-failing`**: the AUC leg passes (`0.908824 >= 0.90`) and the displacement is `21.7`x smaller than
M49's registered `8.256949` and `2.1`x smaller than M50's `0.802227`.

**`under-specified`, and the under-specification is now located.** The four gate boards are *box relocations*, not
pitch errors: `training-20261030-v1` sits `5.18` cells out because M51's box locked onto a panel edge six rows above the
well, and a refinement whose windows are `0.35` of a cell around the box's own expected lines **cannot reach a
six-cell offset by construction**. The pitch hypothesis M51 named is refuted on the read: with pitch `3.9`x tighter, F1
fell `0.011` and AUC fell `0.024` while the median board got `0.008` cells worse. The read is therefore not
pitch-limited; it is limited by per-board phase and location, and the population mean is limited by two gross
relocations.

**Ruled out for the next arm**: any further local refinement of a lattice around a box (this arm: a local fit cannot
reach a gross offset); more interior-line localisation stages; and the phase-alignment search withdrawn before the
declaration. What is left is a mechanism that can place the lattice *without* trusting the box's phase: **M53 - a
stack-anchored lattice fit**, which uses the visible blocks' own axis-aligned edges (they lie exactly on cell
boundaries, and there are several per board) as absolute anchors, refits pitch and origin by Theil-Sen over those
anchors, and falls back to M51's box where a board has too few blocks. Its falsifier is the same inherited one:
population mean `> 0.25` cells or AUC `< 0.90`. It is a different instrument because its anchors are the pieces, not
the grid lines - the one source of cell-boundary evidence present on the two captures this arm could not place.

**no training**, **no threshold fitting**, **no new learned parameter**: this arm refits a lattice from the raster and
applies the committed read-out unchanged; nothing is fitted on its rows.
