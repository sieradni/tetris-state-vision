# M53: do the visible blocks' own edges place the board where a box-relative fit cannot?

Status: **predeclared before any scoring run.** The instrument, its constants, the protocol, the five reads, the
falsifiers, the no-regression rule and every branch are fixed below. The measured section is added after the run and is
not part of the declaration.

## The question this arm answers

M52 (`docs/M52_LATTICE_FIT.md`) fitted the lattice to the interior cell lines *around* M51's detected box, tightened the
scale to `0.001776` / `0.002611` cells, and landed `fit-under-specified`: the population mean stayed at `0.380890`
cells, the read fell to `0.893268` F1 at `0.0` / `0.908824` AUC (M51's box `0.904376` / `0.933275`), and M52 located the
cause - **the remaining error is box relocation, not pitch** - because a refinement window of `0.35` cells cannot reach
a six-cell offset.

M52 named this arm: use the visible blocks' own cell-boundary edges as **absolute** anchors, which do not inherit the
box's phase.

## Motivation disclosure, stated before the run

The instrument is motivated by M52's located failure and by the two named captures (`training-20261030-v1` `5.18` cells,
`training-20261035-v2` `1.93`) whose *boxes* are misplaced while their blocks are visible. That is information from the
measured population shaping the instrument's question, and it is disclosed. The bounds are the same as M51's and M52's:
one declared procedure with fixed constants applied identically to every capture, no branch on capture identity,
sample id, group, split, orientation or block count; the falsifier judged on the full 28-board population; and a
no-regression rule that fails an arm which buys the outliers by degrading the boards an earlier arm placed.

Three instrument builds were measured before this declaration and are disclosed, because they are design changes made
against the population:

- r1: the floor was taken as the raw lowest block edge and used directly as the vertical origin - population mean
  `1.729204`, `13` boards over the gate;
- r2: the floor was used as a whole-cell correction of the box's bottom - `1.720100`;
- r3: the block set was restricted to components whose centre lies inside the box, so merged stacks and HUD panels
  were handled better - `1.676500`;
- r4 (**the declared instrument**): the correction is applied only when the lowest block edge is within
  `FLOOR_TOLERANCE` of the box's own bottom, because the lowest visible block is not the floor when a piece is falling.
  That guard moves the population mean to `0.470500`.

The declared instrument is r4. The earlier three are recorded so nobody mistakes this arm for a single-shot a-priori
design: its constants are fixed, but its *rule* was chosen after seeing that the unguarded version loses on most boards.

## The instrument, exactly

`scripts/stack_anchor.py::anchor_lattice`, fixed constants: `SATURATION = 60.0`, `MIN_BLOCK_CELLS = 0.25`,
`MAX_BLOCK_CELLS = 5.0`, `RATIO_TOLERANCE = 0.35`, `MIN_PAIRS = 3`, `MIN_ANCHORS = 2`, `FLOOR_TOLERANCE = 1.0`.

1. **Input**: the raster and M51's detected box (`scripts/box_search.py::detect_box`). Nothing else.
2. **Blocks**: the saturated mask (`max channel - min channel > SATURATION`) is segmented by row runs and union-find;
   every component at least a quarter cell on a side whose centre falls inside the box grown by one cell is kept. No
   upper size bound: same-coloured blocks merge, and the merged stack's bounding box still has cell boundaries as its
   edges.
3. **Anchors**: the kept components' left/right and top/bottom edges - absolute cell-boundary positions that do not
   depend on the box's phase.
4. **Pitch**: per axis, the median of `|a - b| / round(|a - b| / pitch_seed)` over anchor pairs the seed explains to
   within `RATIO_TOLERANCE`, requiring at least `MIN_PAIRS` pairs. A median, so one displaced anchor cannot move it.
5. **Vertical origin**: the stack's lowest edge is the floor; it moves the box's bottom by **whole cells** only
   (`rows = round((floor_edge - bottom) / pitch)`), and only when the anchor is floor-consistent, i.e.
   `|floor_edge - bottom| <= FLOOR_TOLERANCE * pitch`. Otherwise the box's bottom is kept unchanged.
6. **Horizontal origin**: the box's left snapped to the anchors' circular median phase, moved by the nearest whole
   multiple of the fitted pitch.
7. **Declared fallback, per axis**: fewer than `MIN_ANCHORS` anchors or fewer than `MIN_PAIRS` pairs keeps the box's
   pitch and origin for that axis, unchanged.
8. **Lattice**: `origin = (left, top + 20 * pitch_y)`, `u = (pitch_x, 0)`, `v = (0, -pitch_y)` - the campaign's
   conventions.

It reads the raster and M51's label-free box only: **no label, no prevalence, no scored cell, no use of the target
lattice or the model's predicted lattice at anchoring time.**

## Protocol, unchanged from M49-M52

- **read-out**: `runs/m3-occpix-ceiling-r1-r2/result.json`, logistic, ridge `1.0`, forty iterations, clip `+/-8`,
  leave-one-ancestry-group-out, the thirty statistics, accepted **only** if it reproduces the committed logits to
  `1e-6`; the same fitted folds then transform the anchored features.
- **population**: the same `2,986` cells on the same `28` nonempty boards, the same folds, the fixed `0.0` crossing.
- **five reads, one pass**: `target_lattice`, `predicted_lattice` (M48), `detected_box` (M51), `refined_lattice` (M52),
  `stack_anchor` (this arm), plus the winner control `m30-level-calibrate-r5` from its committed scope.

## Falsifiers, branches and what counts as a win, fixed now

**Inherited falsifiers (judged on the full 28-board population):** the population mean anchored-vs-target displacement
must be **at or below `0.25` cells** and the anchored **AUC at or above `0.90`**.

**In addition, this arm must beat its predecessors to count as a win:** anchored F1 at `0.0` **above M51's `0.904376`**
and anchored AUC **above M52's `0.908824`**, with the population mean **below M51's `0.383358`** and **no worse than
M52's `0.380890`**. These thresholds are declared here, before the study.

Branches, exactly one of which the receipt reports:

- **`anchor-accepted`** - mean `<= 0.25` cells **and** AUC `>= 0.90`: the blocks place the statistics at the declared
  bars on the full population.
- **`anchor-starved`** - median of per-board means `<= 0.25` **and** AUC `>= 0.90` **and** mean `<= 0.383358` (M51's
  box): the typical board is placed and the population is no worse than the box, but a minority still sits outside the
  gate.
- **`anchor-worse`** - anything else: the anchors lose to M51's box on the population, and stack-anchored placement is
  ruled out as the geometry source.

**Prediction, standing as written whatever the outcome.** From the instrument check below I predict the study lands
**F1 at `0.0` in `[0.80, 0.90]`, AUC in `[0.85, 0.93]`, population mean in `[0.42, 0.55]` cells,
median-of-medians in `[0.06, 0.10]` cells**, and therefore the branch **`anchor-worse`**: the anchors help
`training-20261030-v1` (`5.1777` -> `4.113`) and hurt several boards the box already placed, and they cannot reach
`training-20261035-v2` because its floor anchor is inconsistent and its x phase is not the well's.

**No-regression rule, fixed now.** Three tables, each judged over its whole placed set: against **M50** (`26` boards at
`<= 0.25`), against **M51** (`24`), and against **M52** (its own placed set), each with allowance
`max(0.25, baseline + 0.10)`. `training-20261031-v1` and `training-20261031-v2` are also named explicitly against their
**M50** values (`0.058827`).

## Instrument check (declared, run before the study)

`scripts/m53_stack_anchor_read.py --detect-only`, receipt `runs/m53-stack-anchor-detect-r4-job.json`, output
`runs/m53-stack-anchor-detect-r4/detection.json` - detection, refinement and anchoring only, no scoring, no model:

| statistic | M53 anchored | M52 refined | M51 box |
| --- | --- | --- | --- |
| mean of per-board mean displacement | `0.470500` cells | `0.380890` | `0.383358` |
| median of per-board means | `0.469100` cells | `0.380191` | `0.383519` |
| median of per-board medians | `0.083300` cells | `0.074716` | `0.066438` |
| worst board mean | `4.113000` cells | `5.177659` | `5.209784` |
| boards over the `0.25` gate | `8` / `28` | `4` | `4` |
| mean width error | `0.005700` cells | `0.001776` | `0.006892` |
| mean height error | `0.006100` cells | `0.002611` | `0.004720` |
| boards with both axes kept | `28` / `28` | `22` / `28` | - |

The instrument check already shows what the study will confirm: the anchors lower the worst board (`5.21` -> `4.11`)
and raise the population mean (`0.383358` -> `0.470500`), with eight boards over the gate instead of four. The
prediction above is made from this table, not from the study.
## Measured

Receipt: `docs/evidence/m53-stack-anchor-read/scope.json` (status `passed`, `elapsed_seconds` `63.419`), bounded run
`runs/m53-stack-anchor-read-job.json` (status `passed`, `peak_job_commit_bytes` `736555008`), profile
`runs/m53-stack-anchor-read-profile` (`job_ceiling_bytes` `1024` MiB, `target_features_match` true on both reads).

**Verdict: `anchor-worse`.** The anchors lose to M51's box on the population (`0.470531` cells against `0.383358`) and
break the no-regression rule on six boards, so stack-anchored placement is ruled out as the geometry source for this
population - exactly the branch the declaration predicted. The read nevertheless lands the **highest F1 at the fixed
`0.0` crossing of any geometry arm (`0.919192`)**, which is a real, and unpredicted, gain.

### The read

| sampling | F1 at `0.0` | AUC | TP | FP | FN | precision | recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| target lattice | `0.974763` | `0.999195` | 309 | 16 | 0 | `0.950769` | `1.000000` |
| M48 predicted lattice | `0.037152` | `0.442341` | 6 | 8 | 303 | `0.428571` | `0.019417` |
| M51 detected box | `0.904376` | `0.933275` | 279 | 29 | 30 | `0.905844` | `0.902913` |
| M52 interior fit | `0.893268` | `0.908824` | 272 | 28 | 37 | `0.906667` | `0.880259` |
| **M53 stack anchor** | **`0.919192`** | `0.910164` | 273 | 12 | 36 | `0.957895` | `0.883495` |
| winner head (`m30-level-calibrate-r5`) | `0.372881` | `0.954529` | - | - | - | `0.230430` | - |

On the declared bars: the inherited falsifier is **missed** - the population mean is `0.470531` cells, above the `0.25`
gate, though the AUC gate (`>= 0.90`) is cleared at `0.910164`. The declared predecessor bars are **split**: F1 at `0.0`
beats M51's `0.904376` by `+0.014816` and M52's `0.893268` by `+0.025924`; AUC beats M52's `0.908824` by
`+0.001341` but **misses** M51's `0.933275` by `0.023111`; the population mean is worse than both
(`0.383358` / `0.380890`). So this is **not a win**: the anchor moves twenty-nine false positives down to twelve and
pays for it with six displaced boards, and the fixed crossing sees the gain while the rank statistic sees the loss.

The anchored read's own best operating point is `1.5` at F1 `0.923858` (precision `0.968085`, recall `0.883495`) - the
only arm in the campaign whose best F1 is above its `0.0` F1, i.e. the anchored statistics separate the classes more
cleanly but sit at a slightly higher offset at `0.0`.

### Displacement, all 28 boards

Gate `0.25` cells: per-board mean displacement of the anchored sampling against the target lattice, with M51's box and
M52's fit beside it.

| board | M53 anchored | M51 box | M52 refined | rows corrected | floor consistent |
| --- | --- | --- | --- | --- | --- |
| `training-20261026-v2` | `0.033800` | `0.065500` | `0.041600` | 0 | yes |
| `training-20261027-v0` | `0.654800` | `0.101000` | `0.067300` | 0 | no |
| `training-20261027-v1` | `0.654800` | `0.054600` | `0.051100` | 0 | no |
| `training-20261027-v2` | `1.071400` | `0.054600` | `0.094400` | +1 | yes |
| `training-20261028-v0` | `0.083300` | `0.118700` | `0.074900` | 0 | yes |
| `training-20261028-v1` | `0.083300` | `0.063700` | `0.106300` | 0 | yes |
| `training-20261028-v2` | `0.083300` | `0.063700` | `0.083200` | 0 | yes |
| `training-20261029-v2` | `0.062500` | `0.047800` | `0.065200` | 0 | no |
| `training-20261030-v0` | `0.087900` | `0.059200` | `0.026800` | 0 | no |
| `training-20261030-v1` | `4.113000` | `5.209784` | `5.177659` | +1 | yes |
| `training-20261030-v2` | `0.088800` | `0.059200` | `0.025700` | 0 | no |
| `training-20261031-v0` | `0.153800` | `0.091100` | `0.092900` | 0 | yes |
| `training-20261031-v1` | **`0.000000`** | `0.924144` | `0.949682` | +1 | yes |
| `training-20261031-v2` | **`0.000000`** | `0.924144` | `0.949620` | +1 | yes |
| `training-20261032-v2` | `0.080900` | `0.065300` | `0.060600` | 0 | no |
| `training-20261033-v0` | `1.076900` | `0.058800` | `0.026400` | +1 | yes |
| `training-20261033-v1` | `1.076900` | `0.058800` | `0.079500` | +1 | yes |
| `training-20261033-v2` | `1.076900` | `0.058800` | `0.073200` | +1 | yes |
| `training-20261034-v0` | `0.062500` | `0.088400` | `0.096900` | 0 | yes |
| `training-20261034-v1` | `0.062500` | `0.047800` | `0.048500` | 0 | yes |
| `training-20261034-v2` | `0.062500` | `0.088400` | `0.096800` | 0 | yes |
| `training-20261035-v2` | `2.034900` | `1.928500` | `1.926000` | 0 | no |
| `training-20261036-v0` | `0.072200` | `0.101000` | `0.076200` | 0 | yes |
| `training-20261036-v1` | `0.072200` | `0.054600` | `0.070600` | 0 | yes |
| `training-20261036-v2` | `0.075500` | `0.101000` | `0.075500` | 0 | no |
| `training-20261037-v0` | `0.083300` | `0.117900` | `0.095500` | 0 | yes |
| `training-20261037-v1` | `0.083300` | `0.063700` | `0.066400` | 0 | yes |
| `training-20261037-v2` | `0.083300` | `0.063700` | `0.066400` | 0 | yes |

Population: mean `0.470531`, mean of per-board medians `0.469082`, median of per-board means `0.469082`, **median of
per-board medians `0.083333`**, `p90` `0.522702`, worst board `4.112968` (`training-20261030-v1`), **`8` of `28`
boards over the gate**, and **`28` of `28` boards kept by the fit on both axes** - the fallback branch fired on no
board, because every capture pins at least two block edges per axis; the floor guard, not the fallback, is what
declined the correction on eight boards.

Pitch and extent, against both predecessors:

| dispersion | M53 anchored | M51 box | M52 refined |
| --- | --- | --- | --- |
| mean width error (cells) | `0.005696` | `0.006892` | `0.001776` |
| mean height error (cells) | `0.006134` | `0.004720` | `0.002611` |
| mean displacement (cells) | `0.470531` | `0.383358` | `0.380890` |
| median of per-board medians | `0.083333` | `0.066438` | `0.074716` |
| worst board | `4.112968` | `5.209784` | `5.177659` |

### No-regression, both predecessors and the named pair

Rule, fixed in the declaration: every board an earlier arm placed within the gate must stay within it, or within its
own previous value plus `0.10` cells. Judged against all three baselines - M50, M51 and M52 - **`cleared`: false**,
with the same six breaches in every table: `training-20261027-v0`, `training-20261027-v1`, `training-20261027-v2`,
`training-20261033-v0`, `training-20261033-v1`, `training-20261033-v2`.

| board | M50 | M51 | M52 | M53 anchored |
| --- | --- | --- | --- | --- |
| `training-20261027-v0` | `0.058827` | `0.101000` | `0.067300` | `0.654800` |
| `training-20261027-v1` | `0.058827` | `0.054600` | `0.051100` | `0.654800` |
| `training-20261027-v2` | `0.058827` | `0.054600` | `0.094400` | `1.071400` |
| `training-20261033-v0` | `0.058827` | `0.058800` | `0.026400` | `1.076900` |
| `training-20261033-v1` | `0.058827` | `0.058800` | `0.079500` | `1.076900` |
| `training-20261033-v2` | `0.058827` | `0.058800` | `0.073200` | `1.076900` |

The declared named pair, judged explicitly against M50:

| board | M50 | M51 | M52 refined | M53 box | **M53 anchored** |
| --- | --- | --- | --- | --- | --- |
| `training-20261031-v1` | `0.058827` | `0.924144` | `0.949682` | `0.924144` | **`0.000000`** |
| `training-20261031-v2` | `0.058827` | `0.924144` | `0.949620` | `0.924144` | **`0.000000`** |

**The two boards that M51 and M52 both degraded to `0.92`-`0.95` cells are placed exactly by the anchors: `0.000000`
cells on both, to the pixel.** The motivating claim is confirmed in the strongest possible form: those captures'
blocks are visible at the right place, a box-relative geometry can never reach them, and a stack anchor lands on them.

### What the misses prove

The six breaches are all **one whole cell, from a `+1` row correction or a one-cell x snap**, on boards whose box was
already correct. The `rows_corrected = +1` boards are `training-20261027-v2`, `training-20261031-v1`,
`training-20261031-v2` and the `training-20261033` trio; on the trio and `20261027-v2` that correction is wrong, on the
`20261031` pair it is right. The distinction this rule cannot make is:

- **`20261031-v1/-v2`**: the box's bottom is one row *high*, so the stack's floor edge sits one pitch below it, and the
  lowest block edge really is the floor. Correct.
- **`20261027-v2`, `20261033-*`**: the box's bottom is right, but the stack's bottom row is **empty**, so the lowest
  visible block edge sits one pitch *above* the floor. The lowest block edge is *not* the floor.

Both look identical to a single lowest-edge anchor. The remedy is multiplicity: a real floor is pinned by *several*
blocks sharing the same bottom y, while an empty bottom row leaves at most one bounding-box edge there. The arm already
holds the components; it needs to count them against the floor instead of trusting the single maximum.

`training-20261035-v2` (`1.928500` -> `2.034900`) is unaffected by that: its floor anchor is inconsistent
(`floor_consistent` false), so the vertical axis keeps M51's box, and its remaining error is the box's *horizontal*
phase, which the x snap narrows but does not remove.

### Prediction, judged in both directions

| quantity | predicted | measured | outcome |
| --- | --- | --- | --- |
| F1 at `0.0` | `[0.80, 0.90]` | `0.919192` | **missed high** - the anchors beat both predecessors' F1 |
| AUC | `[0.85, 0.93]` | `0.910164` | hit |
| population mean displacement | `[0.42, 0.55]` cells | `0.470531` | hit |
| median of per-board medians | `[0.06, 0.10]` cells | `0.083333` | hit |
| branch | `anchor-worse` | `anchor-worse` | hit |

The one band the declaration got wrong, it got wrong in the arm's favour: I predicted the anchors would cost the read
as well as the geometry, and they did not - they raised F1 at `0.0` above every predecessor. The declared predecessor
bar (F1 above M51's `0.904376`) is therefore **met**; the inherited falsifier (mean `<= 0.25`) is **not**, and the
branch stays `anchor-worse` by the arithmetic fixed before the run. Both statements stand as written.

## Successors this arm names

1. **M54 - floor consensus by block count.** Keep the whole-cell correction, but require the floor to be *pinned*: at
   least `MIN_FLOOR_BLOCKS = 2` components whose bottom edges agree within `0.15` cells, and take the modal bottom over
   all components rather than the maximum. That separates the two cases above by construction. Targets, in this
   receipt's shape: keep `training-20261031-v1/-v2` at `0.000000`, restore `training-20261027-v2` and the `20261033`
   trio to `<= 0.10`, bring the population mean from `0.470531` to `<= 0.25` with `<= 2` boards over the gate, and hold
   anchored F1 above `0.91` while beating M51's `0.904376` / `0.933275`.
2. **M55 - x phase from column occupancy.** The x snap moves boards by a column because the circular median is taken
   over block *edges*, which are sparse and asymmetric. Counting occupied columns per phase bin over the whole stack
   and snapping to the bin with maximum occupancy is M54's floor consensus on the horizontal axis, and is the direct
   remedy for `training-20261035-v2`'s remaining `2.03` cells.
3. **M56 - the fixed-crossing question.** M53 is the first arm whose geometry is *worse* while its F1 at `0.0` is
   *better*, and whose best F1 (`0.923858` at `1.5`) is above its `0.0` F1. The read's offset at the fixed crossing is
   now measurably a function of geometry, not of the classifier: recalibrating on M53's own margins would tell whether
   `0.0` is the right crossing for geometry arms at all.

## Files this arm touches

- `scripts/stack_anchor.py` - the instrument.
- `scripts/m53_stack_anchor_read.py` - the arm; `scripts/record_stream_run.py` - the two accepted run families.
- `runs/m53_build.py` - the builder that derived the arm from M52's, asserting every anchor it replaced.
- `docs/evidence/m53-stack-anchor-read/` - the receipt (`scope.json`, `diagnosis.json`, `progress.json`).
- `tests/test_m53_stack_anchor.py`, `tests/test_m53_receipt.py` - the synthetic-stack and receipt controls.

### Instrument limitation this run exposed

The pitch estimator pools *all* block edges per axis - tops with bottoms, lefts with rights - so on the drawn blocks
(with their inset) the spacings it sees mix one cell with one cell minus twice the inset. The median is still the pitch
on this population (mean width error `0.005696` cells, height `0.006134`), but it is `1.9`x and `2.3`x looser than M52's
interior fit, and the synthetic controls had to draw blocks flush to the cell boundary to recover the pitch exactly.
Same-side edges, or M52's line localiser inside each block, would remove that bias - it is part of why this arm's scale
is not where its improvement lives.

## Verification this landing carries

- full suite: `scripts/run_bounded.py --memory-mib 1024 --wall-seconds 1800 --work-items 1990`, status `passed`, `584.299` s, peak `628,793,344` B, `Ran 2017 tests`, `OK`; receipt committed at `docs/evidence/m53-stack-anchor-read/suite-job.json` (log `runs/suite-m53.log`).
- mutation guards: four fired, each with the restored file byte-identical and the literal read back (`FLOOR_TOLERANCE` `1.0` -> `5.0`; the floor guard line; `MIN_PAIRS` -> `0`; `BOX_MEAN_CELLS` `0.383358` -> `1.5`).