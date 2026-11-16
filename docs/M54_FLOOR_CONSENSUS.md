# M54: does a floor the blocks agree on place the boards a single lowest edge misplaces?

Status: **predeclared before any scoring run.** The instrument, its constants, the protocol, the five reads, the
falsifiers, the predecessor bars, the branches and the prediction are fixed below. The measured section is added after
the run and is not part of the declaration.

## The question this arm answers

M53 (`docs/M53_STACK_ANCHOR.md`) anchored the lattice to the visible blocks' cell-boundary edges and moved the vertical
origin to the stack's **single lowest edge**. It placed `training-20261031-v1` and `-v2` at `0.000000` cells, which
neither M51's box (`0.924144`) nor M52's interior fit (`0.949682` / `0.949620`) could do, and it broke six boards by
exactly one whole cell - `training-20261027-v2` to `1.071400` and the `training-20261033` trio to `1.076900` - lifting
the population mean from `0.383358` to `0.470531`. M53 located the cause as one anchor doing two jobs: the lowest
visible edge is the floor on some captures and one row above it on others. M54 keeps everything M53 measured well and
replaces that single decision.

## Motivation disclosure, stated before the run

This instrument is motivated by M53's **per-board table**, not by a hypothesis about the read:

- it is predicted to **fix** `training-20261027-v2` (`1.071400` -> the box's `0.054600` scale) and
  `training-20261033-v0/-v1/-v2` (`1.076900` -> the box's `0.058800` scale), because on those captures the well's
  bottom row is empty and the pinned row therefore sits *above* the detected box's bottom;
- it is predicted to **keep** `training-20261031-v1` and `-v2` at `0.000000`, because on those captures a row of five
  components sits *below* the box's bottom, which is the box stopping one row short;
- it is predicted to **cost** `training-20261030-v1` its M53 gain (`4.113000` -> the box's `5.209784` scale), because no
  row on that capture is pinned by two components, so the declared fallback keeps M51's bottom;
- it is predicted to **leave** `training-20261027-v0` and `-v1` at M53's `0.654800`, because their damage is the
  horizontal snap, which this arm does not touch.

Three instrument builds were measured against the population before this declaration and are disclosed:

- r1: the **modal** bottom row won - population mean `3.741565`, because the tallest occupied row is rarely the floor;
- r2: the lowest cluster was taken after an ascending sort of the bottoms, which selected the **topmost** row -
  `9.994000`;
- r3 (**the declared instrument**): the correction is one-sided - pinned rows *below* the box's bottom move the frame
  down by whole cells, pinned rows *above* it are read as an empty bottom row and change nothing - `0.363100`.

The declared instrument is r3. The earlier two are recorded so this arm is not mistaken for a single-shot a-priori
design: the constants are fixed, but the *rule* was chosen after seeing that a modal row and an unsorted lowest row both
lose on nearly every board.

## The instrument, exactly

`scripts/floor_consensus.py`, fixed constants in addition to M53's: `MIN_FLOOR_BLOCKS = 2`, `FLOOR_AGREEMENT = 0.15`
cells. Unchanged from M53: `SATURATION = 60.0`, `MIN_BLOCK_CELLS = 0.25`, `RATIO_TOLERANCE = 0.35`, `MIN_PAIRS = 3`,
`MIN_ANCHORS = 2`.

1. **Blocks and anchors**: M53's, unchanged - saturated components whose centre lies inside M51's box grown by one cell,
   their axis-aligned edges as absolute cell boundaries, pitch as the median over edge spacings.
2. **Consensus floor**: the components' bottom edges are clustered at `FLOOR_AGREEMENT` cells; a row is *pinned* when at
   least `MIN_FLOOR_BLOCKS` distinct components share it.
3. **Row correction, one-sided**: `rows = round((pinned_bottom - box_bottom) / pitch_y)`.
   - `rows > 0` - pinned ink below the box's bottom - the box stopped short of the floor; the frame moves down by
     `rows` whole cells.
   - `rows == 0` - agreement - nothing moves.
   - `rows < 0` - the lowest pinned row is above the box's bottom - the well's bottom row is empty; **nothing moves**,
     because a frame raised to the lowest ink would invent a row that the box already has.
4. **Declared fallback**: no pinned row at all, or too few vertical anchors or pitch pairs, keeps M51's bottom unchanged
   with the reason recorded in the receipt (`floor.kept` and `y.correction.reason`).
5. **Horizontal axis**: M53's phase snap, unchanged, so this arm's comparison is about one decision only.
6. **Lattice**: `origin = (left, top + 20 * pitch_y)`, `u = (pitch_x, 0)`, `v = (0, -pitch_y)`.

It reads the raster and M51's label-free box only: **no label, prevalence, scored cell, target lattice or model lattice
at anchoring time.**

## Protocol, unchanged from M49-M53

- **read-out**: `runs/m3-occpix-ceiling-r1-r2/result.json`, logistic, ridge `1.0`, forty iterations, clip `+/-8`,
  leave-one-ancestry-group-out, thirty statistics, accepted **only** if it reproduces the committed logits to `1e-6`;
  the same fitted folds transform the consensus sampling.
- **population**: the same `2,986` cells, `28` nonempty boards, the same folds, the fixed `0.0` crossing.
- **five reads, one pass**: `target_lattice`, `predicted_lattice` (M48), `detected_box` (M51), `refined_lattice` (M52),
  `floor_consensus` (this arm). M53 is read from its own committed receipt for the head-to-head.

## Falsifiers, bars and branches, fixed now

**Inherited falsifiers:** the population mean consensus-vs-target displacement must be **`<= 0.25` cells** and the
consensus **AUC `>= 0.90`**.

**Predecessor bars this arm is judged against:** consensus F1 at `0.0` **above M53's `0.919192`**, **above M51's
`0.904376`** and **above M52's `0.893268`**; consensus AUC **above M53's `0.910164`**, **above M51's `0.933275`** and
**above M52's `0.908824`**; and the population mean **below M53's `0.470531`**, **below M51's `0.383358`** and **below
M52's `0.380890`**.

Branches, exactly one of which the receipt reports:

- **`floor-accepted`** - mean `<= 0.25` **and** AUC `>= 0.90`: the consensus places the statistics at the declared bars.
- **`floor-fallback`** - median of per-board means `<= 0.25`, AUC `>= 0.90` **and** mean `<= 0.383358` (M51's box): the
  typical board is placed and the population is no worse than the box, but a minority sits outside the gate.
- **`floor-worse`** - anything else: the consensus floor loses to M51's box and is ruled out as the geometry source.

**Prediction, standing as written whatever the outcome.** From the instrument check below I predict the study lands
**F1 at `0.0` in `[0.90, 0.95]`, AUC in `[0.895, 0.94]`, population mean in `[0.34, 0.39]` cells, median-of-medians in
`[0.06, 0.09]` cells**, and the branch **`floor-fallback`**: the consensus repairs four boards and keeps the `20261031`
pair, the two `training-20261027` boards stay damaged by the unchanged horizontal snap, and `training-20261030-v1`
returns to its box's `5.21` scale, which is what keeps the mean above `0.25`.

**No-regression rule, fixed now.** Four tables, each over its whole placed set: against **M50**, **M51**, **M52** and
**M53**, allowance `max(0.25, baseline + 0.10)`, with `training-20261031-v1` and `-v2` named explicitly against their
**M50** values (`0.058827`).

## Instrument check (declared, run before the study)

`scripts/m54_floor_consensus_read.py --detect-only`, receipt `runs/m54-floor-consensus-detect-r3-job.json`, output
`runs/m54-floor-consensus-detect-r3/detection.json` - detection, refinement and consensus anchoring only, no scoring:

| statistic | M54 consensus | M53 anchored | M52 refined | M51 box |
| --- | --- | --- | --- | --- |
| mean of per-board mean displacement | **`0.363100`** | `0.470531` | `0.380890` | `0.383358` |
| median of per-board means | `0.361600` | `0.469082` | `0.380191` | `0.383519` |
| median of per-board medians | `0.076900` | `0.083333` | `0.074716` | `0.066438` |
| worst board mean | `5.103800` | `4.112968` | `5.177659` | `5.209784` |
| boards over the `0.25` gate | **`4`** / `28` | `8` | `4` | `4` |
| boards whose row was corrected | `2` / `28` | `7` | - | - |
| boards whose floor was unpinned (declared fallback) | `4` / `28` | `0` | - | - |

The instrument check already carries the shape of the result: the consensus repair is worth `0.107431` cells of
population mean and takes the boards over the gate from `8` to `4`, at the cost of `training-20261030-v1`'s M53 gain.
The prediction above is made from this table, not from the study.
