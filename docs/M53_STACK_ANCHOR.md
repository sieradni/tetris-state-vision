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
