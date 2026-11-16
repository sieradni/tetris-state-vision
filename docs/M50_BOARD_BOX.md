# M50: does a board-box registration buy the read back?

Status: **predeclared before any scoring run.** The detector, its fixed constants, the protocol, the three comparable
reads, the falsifier and every branch are fixed below. The measured section is added after the run and is not part of
the declaration.

## The question this arm answers

M49 (`docs/M49_REGISTRATION_READ.md`, `docs/evidence/m49-registration-read/scope.json`) ran M48's named next arm and
landed `registration-fails`: the campaign's declared support registration maps the board onto the capture's *valid
extent*, which is `3.325530`x the board's width and `1.736611`x its height, so the committed read-out fell to
`0.000000` F1 at the fixed `0.0` crossing at a mean `8.256949`-cell displacement. M49 named one successor: a label-free
detection of the board's **own** box.

This arm runs that successor. It locates each board's box from the raster alone - the well the capture itself draws -
and scores the same thirty statistics, read by the same committed per-cell read-out, on the same `2,986` cells, the
same folds, the same evaluator and the fixed `0.0` crossing. M49's numbers are not taken on trust: the detection is
re-derived here from the board inputs, and the read-out's provenance and feature hashes are re-pinned exactly as M49's
scope did.

## The read-out, imported and its provenance

Identical to M48/M49, pinned by the same two checks before any detected-box number is produced.

- **artifact**: `runs/m3-occpix-ceiling-r1-r2/result.json` (producer `scripts/probe_pixel_ceiling.py`, committed sweep
  `m3-occstack-lever-r2-sweep-r1`, weights `c0b5b783fc11…`).
- **definition**: per cell, thirty statistics - a 3x3 interior block mean per channel (27) plus the per-channel
  interior standard deviation (3) over `+/-0.3` cell units - read by a logistic regression with ridge `1.0`,
  leave-one-ancestry-group-out, standardised by the training folds, clipped to `+/-8`, `40` iterations.
- **committed numbers on the fixed cells**: F1 at `0.0` `0.974763`, AUC `0.999195`, best swept `0.987179`.
- **substitution (declared, unchanged)**: the artifact ships out-of-fold logits, not coefficients, so the read-out is
  re-executed from its own committed estimator on the committed features, labels and folds and is accepted **only** if
  it reproduces the committed logits to `1e-6`; the same fitted folds then transform the detected-box features. No
  parameter is fitted on this arm's inputs, no threshold is fitted.
- **provenance pins, checked before any detected-box number exists**: every nonempty board's target-lattice statistics
  reproduce the committed `features_sha256`, and the reconstruction reproduces the committed `pixel_logits` to `1e-6`.

## The detector, exactly

`scripts/m50_board_box_read.py::detect_box`, fixed constants: `EDGE = 12.0`, `WALL_SHARE = 0.5`, `CLUSTER = 6`,
`MIN_WIDTH = 60`, `MAX_WIDTH = 400`, `ASPECT_SLACK = 0.06`, `SHARE_FLOOR = 0.25`, `SHARE_STEP = 0.05`.

1. **Edges**: per-pixel channel-mean difference above `EDGE` along each axis gives the vertical and horizontal edge
   masks.
2. **Wall runs**: each column's longest vertical edge run, and each row's longest horizontal edge run. The discriminator
   is run *length* rather than brightness, because the captures draw the well's left and right walls over the full
   twenty rows while the HOLD and NEXT panels are framed far shorter, and the cell grid - where it is drawn at all - is
   much fainter.
3. **Candidates**: every column and row whose run reaches `WALL_SHARE` of that axis' longest run, relaxed in `0.05`
   steps down to `SHARE_FLOOR` only when an axis does not bracket a board.
4. **Box**: the wall pair whose width lies in `[MIN_WIDTH, MAX_WIDTH]` and for which a row pair exists within
   `ASPECT_SLACK` of the declared ten-by-twenty aspect (`height = 2 * width`), ranked by the weaker of the four wall
   runs, then by area. If no rectangle satisfies the strict aspect, the declared fallback is the best-effort pairing
   ranked by area then by the weakest wall run. The box is read off the **inner** edges of the four wall clusters.
5. **Lattice**: `origin = (left, bottom)`, `u = (width/10, 0)`, `v = (0, -height/20)` - the same conventions every
   lattice in this campaign uses, and the same diagonal map into image pixels M48/M49 applied.

It reads the raster and the declared constants only: **no label, no prevalence, no scored cell, and no use of the
target lattice or the model's predicted lattice at detection time.** The target lattice is used only *after*
detection, to report the detector's own accuracy.

## Instrument check, before any scoring run

`runs/m50-board-box-detect-r9/detection.json` (bounded, 28 boards, replays only, no forward, no read, no score): over
the `28` nonempty deciding boards the per-board mean displacement of the detected box from the board's own lattice is
**`0.802227`** cells, the **median board's median displacement is `0.061328`** cells, the worst board is
`13.509043` cells, and the per-board scale error is `0.043312` (width) and `0.017508` (height) cells. So the detector
places most boards to a small fraction of a cell and misses badly on two captures:
`training-20261035-v2` (`13.509043` cells - the walls it finds are a panel's, because the well's walls are below the
candidate share) and `training-20261030-v1` (`7.389043` cells - its well's left wall is dim enough to fall below
`EDGE`, so the box lands on the panel-plus-right-wall).

The detector's constants were fixed during this instrument phase, against detection accuracy only; no read-out output
was inspected at any point before the declaration below.

## The three reads, and what each one is

| read | sampling geometry | provenance |
|---|---|---|
| target lattice | the committed board geometry (`state.board.lattice.transformed`) | the ceiling read-out's own |
| predicted lattice | the committed winner's final-stage `lattice` output | M48's measurement |
| **detected box** | the well's own walls, label-free | **this arm** |

Exactly one thing differs: the lattice `pixel_ceiling.cell_features` reads. Same twenty-eight nonempty deciding boards,
same `200` cells per board, same `2,986`-cell fixed population, same anchored labels, same ancestry-group folds, same
evaluator, fixed `0.0`.

## Protocol, isolation and falsifier

- **no training**, no weight change (the script asserts the geometry source's tensors are unchanged after its
  forwards), no new learned parameter, no threshold fitting, no per-board rule beyond the declared detector.
- **single seed inherited**: none is drawn; the committed folds, labels and weights are the control's own.
- **non-inertness and anti-shortcut**: perturbing the detected lattice's statistics changes the read-out's logits, and
  the detected-box read does not reproduce the target or predicted logits (tests).
- **falsifier, declared before the run, on M49's stated bar**: the arm clears only if **mean detected-vs-target
  displacement `<= 0.25` cells**, **detected-box AUC `>= 0.90`** and **F1 at `0.0` `>= 0.70`**. The `0.25`-cell gate is
  the one M49 named for this arm; it is judged against the measured `3.325530`x / `1.736611`x frame-extent error M49
  recorded, not against a hope.

**Branch selection, fixed now:**

| condition | branch | reading |
|---|---|---|
| mean displacement `<= 0.25` and AUC `>= 0.90` and F1 at `0.0` `>= 0.70` | **`box-recovers`** | a label-free board box places the statistics; the read comes back and the campaign has its geometry fix |
| mean displacement `<= 0.25`, or AUC `>= 0.86` | **`box-partial`** | either the detection is close but the crossing does not return, or the read returns where the detector works while the population mean misses the gate; the residual is measured and named |
| otherwise | **`box-fails`** | the detected box is too far out and the read does not separate; raster-only box detection is ruled out as the geometry source |

**Prediction, declared before the run**: the instrument check puts 26 of 28 boards at roughly `0.06` cells and two
boards at `7`-`13` cells, so the population mean displacement will miss the `0.25` gate while the read returns on the
boards that are detected: detected-box F1 at `0.0` `0.70`-`0.90`, AUC `0.93`-`0.97`. The predicted branch is
**`box-partial`**.

## Pre-committed reading of every branch

- **`box-recovers`** - the box is the campaign's geometry fix; the next arm gives the decision this detector's lattice
  (no new parameter) and its falsifier is the winner's own head scoring below `0.372881` F1 at `0.0` or `0.954529` AUC
  when it samples there.
- **`box-partial`** - the detector works where it detects and misses on named captures; the next arm is a detector
  repair with the failing captures named, and its falsifier is a population mean displacement above `0.25` cells
  again after the repair.
- **`box-fails`** - raster-only box detection cannot place the read; the campaign stops proposing raster registration
  and placement moves inside the model.
- **a failed provenance pin** (features digest mismatch or logit reconstruction above `1e-6`) is not an arm reading:
  it is recorded as the measurement's finding and no detected-box number is published.

No branch is re-run for a better number.

## Measured

`docs/evidence/m50-board-box-read/scope.json` and its `diagnosis.json`: one execution, one pass, the committed
read-out on the same `2,986` cells at three samplings, no training and no refit on this arm's inputs. The
reconstruction reproduces the committed `pixel_logits` to `4.98e-07`, and every board's target-lattice statistics
reproduce the committed `features_sha256`.

| read (same 30 statistics, same cells, fixed `0.0`) | F1@0 | AUC | best swept |
|---|---|---|---|
| committed read-out at the **target** lattice | `0.974763` | `0.999195` | `0.987179` |
| committed read-out at the **predicted** lattice (M48) | `0.037152` | `0.442341` | `0.037152` |
| committed read-out at the **detected board box** (this arm) | **`0.979200`** | **`0.999150`** | `0.985507` (at `0.90`) |
| winner control (its own learned head) | `0.372881` | `0.954529` | `0.689775` |

**Verdict: the declared branch `box-partial` lands, as predicted, and the reason is the displacement gate alone.** The
read is not partially recovered - it is fully recovered and slightly ahead of the target lattice at the fixed
crossing (`0.979200` against `0.974763`), with AUC within `0.000045` of it. What misses is the population **mean**
detected-vs-target displacement (`0.802227` cells against the declared `0.25`), and that mean is two captures.

### Complete contingency table at the fixed `0.0` crossing

| at `0.0` | target | predicted | **detected box** |
|---|---|---|---|
| true positives | `309` | `6` | **`306`** |
| false negatives | `0` | `303` | **`3`** |
| false positives | `16` | `8` | **`10`** |
| true negatives | `2,661` | `2,669` | **`2,667`** |
| precision | `0.950769` | `0.428571` | **`0.968354`** |
| recall | `1.000000` | `0.019417` | **`0.990291`** |

### The late-count profile, reconciled to the control

| row band | control (its own head) | detected-box read-out |
|---|---|---|
| rows 0-9 false positives | `400` | `0` |
| rows 10-19 false positives | `140` | `10` |
| total false positives | `698` | `10` |

Per row: `8` at row 16, `2` at row 17, zero elsewhere. Unlike M49 this is a like-for-like reading - the arm predicts
positively on `306` of `309` occupied cells - so the late band's ten false positives are real errors, not the shadow of
an arm that never fires, and they are `92.9%` below the control's `140`.

### Margin distributions

| class | count | median margin | lowest margin |
|---|---|---|---|
| true positive | `306` | `+6.411599` | `+1.564623` |
| false positive | `10` | `+2.532209` | `+0.707313` |
| false negative | `3` | `-8.792206` | `-8.792206` |
| true negative | `2,667` | `-11.661230` | `-68.717728` |

The `3` false negatives sit at `-8.79`, far below the crossing rather than beside it: they are cells sampled outside the
board, which is what the two mis-detected captures do.

### The detector, measured on the same boards

| statistic | value |
|---|---|
| mean detected-vs-target displacement (population, cells) | `0.802227` |
| median board's median displacement (cells) | `0.061328` |
| worst board's mean displacement (cells) | `13.509043` |
| p90 of the pooled per-cell displacement (cells) | `0.988148` |
| mean width / height scale error (cells) | `0.043312` / `0.017508` |

Against M49's measured frame-extent error (`3.325530`x width, `1.736611`x height, `8.256949` cells), the box detector
removes both errors: the scale error falls to `0.043312` cells in width and `0.017508` in height, and the mean
displacement falls from `8.256949` to `0.802227` cells - `89.7%` of it, and `99.1%` of the *median* removed.

The residual is concentrated, exactly where the instrument check said it would be: `training-20261035-v2`
(`13.509043` cells) and `training-20261030-v1` (`7.389043` cells). Both are captures whose well is drawn without a
usable pair of full-height bright walls - one's walls fall below the candidate share, the other's left wall is dim
enough to fall below `EDGE` - so the detector selects a panel-plus-wall rectangle instead. Every other board is
placed to a fraction of a cell.

**Label defect, recorded rather than rewritten.** The frozen receipt's `geometry.detection.recipe` string reads
`bright-frame interior of the captured well` - the label of the detector's first draft, which the declared recipe in
this document replaced (`tallest ten-by-twenty wall rectangle in the capture`). The evidence was produced by the
wall-run detector, and the receipt pins that code by digest in `sources`; the string is stale, the numbers are not.
It is logged as V-187 rather than edited into the frozen receipt.

## Answering the campaign's question

**A board-box lattice buys the read back.** The same thirty statistics and the same committed read-out score
`0.979200` F1 at the fixed `0.0` crossing and `0.999150` AUC when sampled at a label-free detection of each capture's
own well - against `0.974763` / `0.999195` at the target lattice, `0.037152` / `0.442341` at the model's predicted
lattice, and `0.372881` / `0.954529` for the winner's own learned head. The geometry axis is closed on the raster side:
the ceiling is reachable from the raster alone, with no training, no label and no fitted constant, and the
`0.25`-cell gate the campaign set for itself is missed only by two named captures.

## Re-ranked successors

| rank | next arm | headroom | cost | falsifier |
|---|---|---|---|---|
| 1 | **detector repair on the two named captures**: a wall-free fallback (grid-periodic spacing or a stack-containment constraint) for captures whose well has no usable wall pair | the read already measures `0.979200` F1 at `0.0`; the only remaining geometry defect is `2` of `28` boards, and repairing them is the difference between a `0.802227` and a `~0.06`-cell population mean | one read-only detection pass, no forward, no read | population mean displacement `> 0.25` cells after the repair: the detector family cannot cover these captures and the box axis is closed at `26`/`28` |
| 2 | **give the decision the detected box**: the winner's own head reads at the detected lattice instead of its predicted one, no new parameter | the winner's read is `0.372881` F1 at `0.0` against the ceiling's `0.974763`, and the geometry that reaches it is now available per board, label-free | one `64`-step run, or a read-only re-score first | head F1 at `0.0` `< 0.382881` or AUC `< 0.954529` at the detected geometry: the head cannot use a correct lattice and placement moves inside the model |
| 3 | placement inside the model (M49's declared successor) | unchanged, and now strictly worse than rank 2 because the raster side already supplies the geometry | one `64`-step run | as declared in M49 |
| R | **retired**: the valid-mask/frame-extent registration (M49); every additive raster term (M44-M46); pixel-sole routing (M47); M42/M43 decision-side levels; M34-M41 geometry variants as *training* recipes | closed by measurement | - | not offered |

**Named next arm: detector repair on the two named captures.** It is the only open falsifier that costs a read-only
pass, and its target - a `0.25`-cell population mean - is now the sole obstacle between the campaign and a geometry it
can trust on every board. The step after it is rank 2, the first arm in this campaign whose geometry is not the
model's own.

## What this arm establishes, and what it does not

- Established: a label-free board box exists in these captures and it places the thirty statistics well enough that
  the committed read-out returns to `0.979200` F1 at `0.0` / `0.999150` AUC, slightly ahead of the target lattice at
  the fixed crossing.
- Established: the scale error M49 measured (`3.325530`x) is a property of the frame-extent recipe, not of the raster:
  the box detector's own scale error is `0.043312` cells in width.
- Not established: that the detector covers every capture - it misses two, and their failure modes are named with
  numbers.
- Not shown: any effect on the winner's own head (`0.954529` AUC untouched), or generality beyond these synthetic
  captures, one seed and the training-fit boards.

## Receipts

`runs/m50-board-box-read-profile{,-job.json}`, `runs/m50-board-box-read{,-job.json}`,
`docs/evidence/m50-board-box-read/{scope,diagnosis}.json`, and the instrument check
`runs/m50-board-box-detect-r9{,-job.json}`. Refused attempts keep their receipts under their own names; only completed
runs are quoted.
