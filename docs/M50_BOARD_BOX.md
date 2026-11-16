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

## Receipts

`runs/m50-board-box-read-profile{,-job.json}`, `runs/m50-board-box-read{,-job.json}`,
`docs/evidence/m50-board-box-read/{scope,diagnosis}.json`, and the instrument check
`runs/m50-board-box-detect-r9{,-job.json}`. Refused attempts keep their receipts under their own names; only completed
runs are quoted.
