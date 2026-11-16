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

## Measured

Receipt: `docs/evidence/m54-floor-consensus-read/scope.json` (status `passed`, `elapsed_seconds` `63.453`), bounded run `runs/m54-floor-consensus-read-job.json` (status `passed`, `peak_job_commit_bytes` `725991424`), profile `runs/m54-floor-consensus-read-profile` (`job_ceiling_bytes` `1024` MiB, `target_features_match` true on both reads). Declaration committed as `e706fa3`, strictly before this measurement.

**Verdict: `floor-fallback`** - the branch predicted from the instrument check. The consensus floor is the best population geometry of any arm so far (`0.363060` cells against M53's `0.470531`, M51's `0.383358` and M52's `0.380890`), it takes the boards over the gate from `8` back to `4`, it clears the no-regression rule **against M53** and lands the **highest F1 at `0.0` of any read in this campaign outside the target itself (`0.944724`)** - and it still misses the inherited falsifier, because the two `training-20261027` boards remain at `0.654800` and the mean cannot reach `0.25` while they do.

### The read

| sampling | F1 at `0.0` | AUC | TP | FP | FN | precision | recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| target lattice | `0.974763` | `0.999195` | 309 | 16 | 0 | `0.950769` | `1.000000` |
| M48 predicted lattice | `0.037152` | `0.442341` | 6 | 8 | 303 | `0.428571` | `0.019417` |
| M51 detected box | `0.904376` | `0.933275` | 279 | 29 | 30 | `0.905844` | `0.902913` |
| M52 interior fit | `0.893268` | `0.908824` | 272 | 28 | 37 | `0.906667` | `0.880259` |
| M53 stack anchor | `0.919192` | `0.910164` | 273 | 12 | 36 | `0.957895` | `0.883495` |
| **M54 consensus floor** | `0.944724` | `0.922077` | 282 | 6 | 27 | `0.979167` | `0.912621` |
| winner head (`m30-level-calibrate-r5`) | `0.372881` | `0.954529` | - | - | - | `0.230430` | - |

The consensus read beats every predecessor's F1 at `0.0`: M53 by `+0.025532`, M51 by `+0.040348`, M52 by `+0.051456`, with six false positives where M51 had twenty-nine and M53 had twelve, and one false negative fewer than M51 (`27` against `30`) with eight more put down than M53 (`282` against `273`). It clears M53's AUC (`0.922077` against `0.910164`) and M52's (`0.908824`), and misses M51's `0.933275` by `0.011198`: the consensus geometry is the best of the three at the fixed crossing and second-best at ranking, behind the box it started from.

### Displacement, all 28 boards

Per-board mean displacement against the target lattice, gate `0.25` cells, with the row decision the consensus made:

| board | M54 consensus | M53 anchored | M51 box | M52 refined | rows | floor |
| --- | --- | --- | --- | --- | --- | --- |
| `training-20261026-v2` | `0.033850` | `0.033850` | `0.065500` | `0.041607` | +0 | agrees |
| `training-20261027-v0` | `0.654811` | `0.654811` | `0.101015` | `0.067303` | +0 | unpinned |
| `training-20261027-v1` | `0.654811` | `0.654811` | `0.054625` | `0.051116` | +0 | unpinned |
| `training-20261027-v2` | `0.071429` | `1.071429` | `0.054625` | `0.094379` | +0 | unpinned |
| `training-20261028-v0` | `0.083333` | `0.083333` | `0.118682` | `0.074947` | +0 | agrees |
| `training-20261028-v1` | `0.083333` | `0.083333` | `0.063729` | `0.106322` | +0 | agrees |
| `training-20261028-v2` | `0.083333` | `0.083333` | `0.063729` | `0.083210` | +0 | agrees |
| `training-20261029-v2` | `0.062500` | `0.062500` | `0.047797` | `0.065170` | +0 | agrees |
| `training-20261030-v0` | `0.087869` | `0.087869` | `0.059175` | `0.026829` | +0 | agrees |
| `training-20261030-v1` | `5.103766` | `4.112968` | `5.209784` | `5.177659` | +0 | agrees |
| `training-20261030-v2` | `0.088845` | `0.088845` | `0.059175` | `0.025742` | +0 | agrees |
| `training-20261031-v0` | `0.153846` | `0.153846` | `0.091118` | `0.092873` | +0 | agrees |
| `training-20261031-v1` | `0.000000` | `0.000000` | `0.924144` | `0.949682` | +1 | pinned x5 |
| `training-20261031-v2` | `0.000000` | `0.000000` | `0.924144` | `0.949620` | +1 | pinned x5 |
| `training-20261032-v2` | `0.080886` | `0.080886` | `0.065262` | `0.060618` | +0 | agrees |
| `training-20261033-v0` | `0.076923` | `1.076923` | `0.058827` | `0.026440` | +0 | agrees |
| `training-20261033-v1` | `0.076923` | `1.076923` | `0.058827` | `0.079467` | +0 | agrees |
| `training-20261033-v2` | `0.076923` | `1.076923` | `0.058827` | `0.073153` | +0 | agrees |
| `training-20261034-v0` | `0.062500` | `0.062500` | `0.088388` | `0.096858` | +0 | agrees |
| `training-20261034-v1` | `0.062500` | `0.062500` | `0.047797` | `0.048537` | +0 | agrees |
| `training-20261034-v2` | `0.062500` | `0.062500` | `0.088388` | `0.096785` | +0 | agrees |
| `training-20261035-v2` | `2.034873` | `2.034873` | `1.928501` | `1.926006` | +0 | unpinned |
| `training-20261036-v0` | `0.072211` | `0.072211` | `0.101015` | `0.076159` | +0 | agrees |
| `training-20261036-v1` | `0.072211` | `0.072211` | `0.054625` | `0.070647` | +0 | agrees |
| `training-20261036-v2` | `0.075498` | `0.075498` | `0.101015` | `0.075542` | +0 | agrees |
| `training-20261037-v0` | `0.083333` | `0.083333` | `0.117851` | `0.095451` | +0 | agrees |
| `training-20261037-v1` | `0.083333` | `0.083333` | `0.063729` | `0.066401` | +0 | agrees |
| `training-20261037-v2` | `0.083333` | `0.083333` | `0.063729` | `0.066401` | +0 | agrees |

Population: mean **`0.363060`**, mean of per-board medians `0.361604`, median of per-board means **`0.361604`**, **median of per-board medians `0.076923`**, `p90` `0.415227`, worst board **`5.103766`** (`training-20261030-v1`), **`4` of `28` boards over the gate**, `28` of `28` kept on both axes, **`2` of `28` rows corrected** (`training-20261031-v1/-v2`) and **`4` of `28` floors unpinned** (`training-20261027-v0`, `-v1`, `-v2`, `training-20261035-v2`).

Pitch and extent against all three predecessors:

| dispersion | M54 consensus | M53 anchored | M52 refined | M51 box |
| --- | --- | --- | --- | --- |
| mean width error (cells) | `0.005696` | `0.005696` | `0.001776` | `0.006892` |
| mean height error (cells) | `0.006134` | `0.006134` | `0.002611` | `0.004720` |
| mean displacement (cells) | **`0.363060`** | `0.470531` | `0.380890` | `0.383358` |
| median of per-board medians | `0.076923` | `0.083333` | `0.074716` | `0.066438` |
| worst board | `5.103766` | `4.112968` | `5.177659` | `5.209784` |
| boards over the gate | `4` | `8` | `4` | `4` |

### No-regression, all three predecessors and the named pair

Rule, fixed in the declaration: every board an earlier arm placed within the gate must stay within it, or within its own
previous value plus `0.10` cells. Judged against **M50**, **M51**, **M52** and **M53**:

| baseline | cleared | breaches |
| --- | --- | --- |
| M50 | **false** | `training-20261027-v0`, `training-20261027-v1` |
| M51 | **false** | `training-20261027-v0`, `training-20261027-v1` |
| M52 | **false** | `training-20261027-v0`, `training-20261027-v1` |
| **M53** | **true** | none |

The two breaches are not a regression this arm introduces: they are M53's own `0.654800`, inherited unchanged, and they
come from the **horizontal** snap that M54 deliberately did not touch. Against M53 - the arm it replaces - the consensus
clears every board on every table, which is the comparison that matters for "did replacing the single lowest edge do
what it promised".

| board | M50 | M51 | M52 | M53 | **M54** |
| --- | --- | --- | --- | --- | --- |
| `training-20261027-v0` | `0.058827` | `0.101000` | `0.067300` | `0.654800` | `0.654800` |
| `training-20261027-v1` | `0.058827` | `0.054600` | `0.051100` | `0.654800` | `0.654800` |

The declared named pair, judged explicitly against M50:

| board | M50 | M51 | M52 refined | M53 anchored | **M54 consensus** |
| --- | --- | --- | --- | --- | --- |
| `training-20261031-v1` | `0.058827` | `0.924144` | `0.949682` | `0.000000` | **`0.000000`** |
| `training-20261031-v2` | `0.058827` | `0.924144` | `0.949620` | `0.000000` | **`0.000000`** |

Both stay exactly on the floor the consensus found: five components pin a row one pitch below the box's bottom, the
one-sided correction moves the frame down one cell, and the lattice lands on the target to the pixel. The mechanism M53
found by accident is now the mechanism this arm applies by rule.

### What the miss proves, and the boards it comes from

The falsifier is missed: mean `0.363060 > 0.25`. The four boards carrying the population are exactly the ones the
declaration predicted would not move, plus the two that cannot:

| board | M54 | why it is still out |
| --- | --- | --- |
| `training-20261030-v1` | `5.103766` | its box is `5.21` cells out and **no row is pinned** (`floor.unpinned`): 20 scattered components, no two sharing a bottom, so the declared fallback keeps M51's bottom and this arm's only lever does nothing |
| `training-20261035-v2` | `2.034900` | same shape - five components at five distinct bottoms - and its error is the box's *horizontal* phase, which M54 does not touch |
| `training-20261027-v0` | `0.654800` | floor unpinned, and the damage is the horizontal snap, unchanged since M53 |
| `training-20261027-v1` | `0.654800` | as above |

Those four sum to `8.448266` cells of the `10.165680` the population carries; the other 24 boards average `0.071559`.
So the mean is not a scale problem and not a row problem any more - **`0.25` is now blocked by two horizontal snaps and
two captures whose box is wrong and whose blocks pin no floor at all.** The row instrument has done its job: it
recovered four boards (`training-20261027-v2` `1.071400` -> `0.071429`, the `training-20261033` trio `1.076900` ->
`0.058800`) and kept both `20261031` boards at zero.

### Prediction, judged in both directions

| quantity | predicted | measured | outcome |
| --- | --- | --- | --- |
| F1 at `0.0` | `[0.90, 0.95]` | `0.944724` | hit |
| AUC | `[0.895, 0.94]` | `0.922077` | hit |
| population mean displacement | `[0.34, 0.39]` cells | `0.363060` | hit |
| median of per-board medians | `[0.06, 0.09]` cells | `0.076923` | hit |
| branch | `floor-fallback` | `floor-fallback` | hit |
| `20261031` pair kept at zero | yes | `0.000000` / `0.000000` | hit |
| `20261027-v2` and `20261033` trio repaired | yes | `0.071429` / `0.058800` | hit |
| `20261030-v1` returns to box scale | yes | `5.103766` (box `5.209784`) | hit |
| `20261027-v0/-v1` unchanged | yes | `0.654800` / `0.654800` | hit |

Every predicted band and every predicted board movement landed as declared. The declaration's own win bars are
**split**: F1 beats all three predecessors; the mean beats M53's `0.470531`, M51's `0.383358` and M52's `0.380890`; AUC
beats M53's and M52's and misses M51's `0.933275` by `0.011198`. By the arithmetic fixed before the run the branch is
`floor-fallback`, and the falsifier is missed. Both statements stand as written.

## Successors this arm names

1. **M55 - x phase from column occupancy.** The two surviving `training-20261027` boards sit at `0.654800`, and their
   error is one column: M53's and M54's horizontal snap takes the circular median over block *edges*, which is sparse
   and asymmetric. Counting, per phase bin, how many *cells* the stack occupies and snapping to the heaviest bin is the
   row argument applied to the columns. Numbers: those two boards to `<= 0.10` takes the population mean from
   `0.363060` to `0.316000` and leaves `training-20261030-v1` and `training-20261035-v2` as the only boards over the
   gate. It must hold M54's F1 (`0.944724`) and its `20261031` pair at `0.000000`.
2. **M56 - the well frame, not the cell grid.** Both remaining boards are box failures where the blocks pin no floor
   (`5.103766` and `2.034900`, `12.5` cells between them). Their captures do show the well: a bordered rectangle whose
   four walls are a stronger, lower-frequency signal than any cell line. Fitting that rectangle - two vertical and two
   horizontal wall edges, robust to the pieces inside it - is the instrument the two boards actually need. Numbers:
   both to `<= 0.5` takes the population mean from `0.316000` (after M55) to `<= 0.22` and clears the inherited
   `0.25`/`0.90` falsifier for the first time on geometry alone.
3. **M57 - the fixed crossing, restated.** With F1 at `0.0` now `0.944724` and the best swept F1 `0.949495` at `1.5`,
   the read's own offset is the last non-geometric term: a recalibration fitted on the consensus margins would say
   whether `0.0` is still the right crossing, or whether the campaign should report a geometry arm at its own declared
   operating point.

## Files this arm touches

- `scripts/floor_consensus.py` - the instrument (`CONSTANTS` above).
- `scripts/m54_floor_consensus_read.py` - the arm; `scripts/record_stream_run.py` - the two accepted run families.
- `runs/m54_build.py` - the builder that derived the arm from M53's, asserting every anchor it replaced.
- `docs/evidence/m54-floor-consensus-read/` - the receipt (`scope.json`, `diagnosis.json`, `progress.json`).
- `tests/test_m54_floor_consensus.py`, `tests/test_m54_receipt.py` - the synthetic-stack and receipt controls.

## Verification this landing carries

- full suite: `scripts/run_bounded.py --memory-mib 1024 --wall-seconds 1800 --work-items 2010`, status `passed`, `536.193` s, peak `631,840,768` B, `Ran 2053 tests`, `OK`; receipt committed at `docs/evidence/m54-floor-consensus-read/suite-job.json` (log `runs/suite-m54.log`).
- mutation guards: four fired, each with the restored file byte-identical and the literal read back (`MIN_FLOOR_BLOCKS` `2` -> `1`; `FLOOR_AGREEMENT` `0.15` -> `0.5`; the one-sided `if rows < 0` delete; `clusters[pinned[-1]]` -> `clusters[pinned[0]]`).
- frozen gate: `305` directories / `3707` files match `docs/frozen-evidence-manifest.json`.
