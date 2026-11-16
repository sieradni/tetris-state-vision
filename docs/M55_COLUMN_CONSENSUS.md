# M55: can the blocks place the columns, or can they only certify them?

Status: **predeclared before any scoring run.** The instrument, its constants, the protocol, the five reads, the
falsifiers, the predecessor bars, the branches and the prediction are fixed below. The measured section is added after
the run and is not part of the declaration.

## The question this arm answers

M54 (`docs/M54_FLOOR_CONSENSUS.md`) named the horizontal axis: its two surviving `training-20261027` boards sat at
`0.654800` while their boxes were accurate (`0.101000` / `0.054600`), and the blame fell on M53's sub-pixel phase snap.
M55 asks whether a consensus over the components' column edges can place that axis where a phase median could not.

## Motivation disclosure, stated before the run

M54's per-board table, read line by line, is the whole argument:

| board | components | box pitch | anchored pitch | M54 snap | M54 displacement | box displacement |
| --- | --- | --- | --- | --- | --- | --- |
| `training-20261027-v0` | 2 | `14.0` | `13.666667` | `2.833333` px | `0.654800` | `0.101000` |
| `training-20261027-v1` | 2 | `13.9` | `13.666667` | `1.833333` px | `0.654800` | `0.054600` |
| `training-20261031-v0` | 54 | `12.8` | `12.8` | `-1.0` px | `0.153846` | `0.091118` |

Two facts follow, and both are properties of the axis rather than of the estimator. **First**, with two components the
anchored pitch is off by `0.33` px - `2.4%` - and over twenty rows that error alone is `0.6` of a cell; M54's `0.654800`
is mostly a *scale* error, not a phase error. **Second**, the many-component boards move by exactly the negative of the
drawing inset (`-1.0` px, the inset measured from both sides), which means the phase median is fitting the blocks'
padding. And underneath both: a stack occupies *some* columns of the well and nothing in the raster says which, so every
horizontal observable - pitch, inset, phase - is invariant under sliding the stack sideways. **The blocks can certify the
box's columns; they cannot place them.**

What M55 predicts, board by board, before measuring:

- **fixes** `training-20261027-v0` and `-v1` (`0.654800` -> their boxes' `0.101000` / `0.054600`), because a pitch fitted
  from two components will be refused and no phase snap will be applied;
- **keeps** `training-20261031-v1/-v2` at `0.000000`, because their floor consensus is untouched and their anchors span
  the whole well (so their pitch is pinned);
- **may regress** `training-20261026-v2`, `training-20261030-v0/-v2`, `training-20261034-v0/-v2`,
  `training-20261036-v0/-v2` and `training-20261037-v0` by `0.02`-`0.06` cells, where M54's snap happened to land a
  fraction of a cell better than the box does;
- **cannot move** `training-20261030-v1` (`5.10`) and `training-20261035-v2` (`1.93`), whose problem is the box, not the
  columns.

The risk this arm takes is explicit: the population mean should improve by about `0.03` cells while a dozen individual
boards get slightly worse, so the declared no-regression rule against M54 may fail even though the falsifier's *mean*
leg improves. That trade is what this arm is for - it is the axis where the campaign's remaining error lives - and the
receipt reports both sides.

Four instrument builds were measured before this declaration:

- r1: a whole-column shift chosen by edge support - ties everywhere (edges are symmetric), so every board fell back and
  the anchored pitch still stretched the frame: mean `0.376600`;
- r2: certification only, anchored pitch everywhere: `0.370900`, `training-20261027-v0` still `0.655900`;
- r3: the pitch-span rule with the box's pitch on both axes: `0.329800`, two boards over the gate;
- r4 (**the declared instrument**): the pitch-span rule with the anchored pitch used only where the span pins it:
  `0.329300`.

The declared instrument is r4. The earlier three are disclosed because they are rule changes made against the
population; the constants are fixed, but the rule was found by measurement, in the order M53 and M54 were too.

## The instrument, exactly

`scripts/column_consensus.py`, fixed constants: `MIN_COLUMN_BLOCKS = 2`, `COLUMN_TOLERANCE = 0.15` cells,
`COLUMN_INSET_MAX = 0.20` cells, `MIN_PITCH_SPAN_CELLS = 3.0`; inherited from M54/M53: `SATURATION = 60.0`,
`MIN_BLOCK_CELLS = 0.25`, `RATIO_TOLERANCE = 0.35`, `MIN_PAIRS = 3`, `MIN_ANCHORS = 2`,
`MIN_FLOOR_BLOCKS = 2`, `FLOOR_AGREEMENT = 0.15`.

1. **Blocks, anchors, pitch median**: M53's, unchanged.
2. **Pitch certification (both axes)**: a pitch fitted from edges spanning fewer than `MIN_PITCH_SPAN_CELLS` cells is
   not pinned - two or three edges quantise the pitch worse than the pitch itself - and the box's pitch is kept for that
   axis. Where the span pins it, the anchored pitch is used.
3. **Column consensus (x)**: at least `MIN_COLUMN_BLOCKS` components must agree, each within `COLUMN_TOLERANCE` cells of
   the circular median, on both their left and their right edge phases; and those phases must sit inside the drawing
   band - left phase `<= COLUMN_INSET_MAX`, right phase `>= 1 - COLUMN_INSET_MAX`. That certifies the box's columns.
   **No phase shift is ever applied**, because this axis has no absolute anchor: the frame's left stays exactly M51's.
4. **Declared fallbacks**: no visible blocks, too few components, phases disagreeing beyond tolerance, a phase outside
   the drawing band, or too few anchors all keep M51's box on that axis and record the reason.
5. **Vertical axis**: M54's consensus floor and its one-sided whole-cell row correction, unchanged, now also subject to
   the pitch certification of clause 2.
6. **Lattice**: `origin = (left, top + 20 * pitch_y)`, `u = (pitch_x, 0)`, `v = (0, -pitch_y)`.

It reads the raster and M51's label-free box only: **no label, prevalence, scored cell, target lattice or model lattice
at anchoring time.**

## Protocol, unchanged from M49-M54

- **read-out**: `runs/m3-occpix-ceiling-r1-r2/result.json`, logistic, ridge `1.0`, forty iterations, clip `+/-8`,
  leave-one-ancestry-group-out, thirty statistics, accepted **only** if it reproduces the committed logits to `1e-6`;
  the same fitted folds transform the column-consensus sampling.
- **population**: the same `2,986` cells, `28` nonempty boards, the same folds, the fixed `0.0` crossing.
- **five reads, one pass**: `target_lattice`, `predicted_lattice` (M48), `detected_box` (M51), `refined_lattice` (M52),
  `column_consensus` (this arm). M53 and M54 are read from their own committed receipts for the head-to-head.

## Falsifiers, bars and branches, fixed now

**Inherited falsifiers:** the population mean consensus-vs-target displacement must be **`<= 0.25` cells** and the
consensus **AUC `>= 0.90`**.

**Predecessor bars this arm is judged against:** F1 at `0.0` above **M54's `0.944724`**, **M53's `0.919192`**, **M51's
`0.904376`** and **M52's `0.893268`**; AUC above **M54's `0.922077`**, **M53's `0.910164`**, **M51's `0.933275`** and
**M52's `0.908824`**; and the population mean below **M54's `0.363060`**, **M51's `0.383358`** and **M52's `0.380890`**
(M53's `0.470531` is below all three and is reported for completeness).

Branches, exactly one of which the receipt reports:

- **`column-accepted`** - mean `<= 0.25` **and** AUC `>= 0.90`.
- **`column-fallback`** - median of per-board means `<= 0.25`, AUC `>= 0.90` **and** mean `<= 0.383358` (M51's box).
- **`column-worse`** - anything else.

**Prediction, standing as written whatever the outcome.** From the instrument check below I predict the study lands
**F1 at `0.0` in `[0.93, 0.97]`, AUC in `[0.91, 0.95]`, population mean in `[0.30, 0.35]` cells, median-of-medians in
`[0.07, 0.10]` cells**, the branch **`column-fallback`**, and **two** boards over the gate
(`training-20261030-v1`, `training-20261035-v2`).

**No-regression rule, fixed now.** Five tables, each over its whole placed set: against **M50**, **M51**, **M52**,
**M53** and **M54**, allowance `max(0.25, baseline + 0.10)`, with `training-20261031-v1` and `-v2` named explicitly
against their **M50** values (`0.058827`).

## Instrument check (declared, run before the study)

`scripts/m55_column_consensus_read.py --detect-only`, receipt `runs/m55-column-consensus-detect-r4-job.json`, output
`runs/m55-column-consensus-detect-r4/detection.json` - detection, refinement and consensus anchoring only, no scoring:

| statistic | M55 columns | M54 consensus | M53 anchored | M52 refined | M51 box |
| --- | --- | --- | --- | --- | --- |
| mean of per-board mean displacement | **`0.329300`** | `0.363060` | `0.470531` | `0.380890` | `0.383358` |
| mean of per-board medians | `0.329300` | `0.361604` | `0.469082` | `0.380191` | `0.383519` |
| median of per-board medians | `0.083300` | `0.076923` | `0.083333` | `0.074716` | `0.066438` |
| worst board mean | `5.085774` | `5.103766` | `4.112968` | `5.177659` | `5.209784` |
| boards over the `0.25` gate | **`2`** / `28` | `4` | `8` | `4` | `4` |
| mean width error | `0.003700` | `0.005696` | `0.005696` | `0.001776` | `0.006892` |
| mean height error | **`0.001200`** | `0.006134` | `0.006134` | `0.002611` | `0.004720` |

The instrument check already carries the shape of the result: the certification and the pitch-span rule remove
`0.033760` cells from the population mean and take the boards over the gate from `4` to `2`, they put
`training-20261027-v0/-v1` back on their boxes (`0.101015` / `0.054625`) and hold the `20261031` pair at `0.000000`, and
they leave `training-20261030-v1` (`5.085774`) and `training-20261035-v2` (`1.974197`) as the only two out. The
prediction above is made from this table, not from the study.
