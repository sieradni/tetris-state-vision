# M27: the occupancy deficit is three factors, measured on one population

Status: **predeclared before this pass has any receipt of its own.** This document fixes the
question, the two committed receipts it joins, the four join conditions, the declared reads, the
reading rule and the limits. The measured section is appended only after the canonical receipt
exists. Nothing is trained, no weight, threshold, gate, label or prior evidence byte changes.

## Where this comes from

M26 closed the pair-supervision axis with strict exact still `0 / 28`, and left the residual where
M24 put it: within-board discrimination. That framing is now incomplete, because two committed
measurements that were never read together say the deficit has more than one first-order term:

- **The pixel ceiling** ([M3_OCCUPANCY_PIXEL_CEILING.md](M3_OCCUPANCY_PIXEL_CEILING.md),
  `docs/evidence/m3-occupancy-pixel-ceiling-r1/scope.json`) fits a 30-feature colour read-out at the
  **target** lattice and scores F1 `0.9780`, best swept F1 `0.9859`, fold-mean AUC `0.9976` on the
  same 28 boards. The representation the task needs is nearly free from the pixels.
- **Registration attribution** ([M3_OCCUPANCY_SCALED.md](M3_OCCUPANCY_SCALED.md),
  `runs/m3-occscale-registration-r2/result.json`) pairs each model cell with the target cell its
  centre actually covers and reports F1 `0.4390 -> 0.5742` and AUC `0.9331 -> 0.9702` on the
  `2,986` co-scoreable cells, with a mean absolute offset of `2.575` target cells and `48.25%` of
  cells shifted.

Each was reported on its own axis. Neither was read as a *budget*: how much F1 is sitting in
registration, how much in the learned read, and how much at the boundary. That budget is what the
last twenty-three single-lever arms lacked, and it is measurable read-only from bytes that already
exist.

## The one question

**On one population, with three reads side by side, how much of the locked-occupancy deficit is
registration, how much is the representation, and how much is the operating point?**

## Instrument: one population, four joins, no forward pass

`scripts/oracle_pairing_read.py`. It reads exactly two committed receipts and replays nothing:

| receipt | what it contributes |
|---|---|
| `runs/m3-occscale-registration-r2/result.json` | per-case model logits, labels, known mask and the model-cell -> target-cell `mapping` for the 36 deciding boards |
| `runs/m3-occpix-ceiling-r1-r2/result.json` | per-case pixel read-out logits on the same boards and the same 200 cells |

The pixel read-out is a function of the raster and the committed labels only, so it is
**arm-independent**; the model side is the arm whose mapping is committed
(`m3-occscale-lever-r1`). That difference is declared here rather than glossed: the pixel number is
a property of the capture, the model number is a property of one arm.

Three reads are scored on **identical cells** - the `2,986` co-scoreable cells where a model cell
maps inside the target board and the target cell is known:

| read | pairing | what it measures |
|---|---|---|
| model, as-is | model cell `i` vs label `i` | the standing read at the model's own lattice |
| model, oracle | model cell `i` vs label `mapping[i]` | the same weights with registration granted |
| pixel, same cells | target cell `j` vs label `j`, restricted to `j` in the mapping's image | a sufficient read-out on the same cells |

The full-population reads (`5,600` cells, `667` occupied, `4,933` empty, 28 boards) are reported
beside them so the matched subset cannot be mistaken for the deciding population.

Scoring is the campaign's own: the `81`-point grid `-6.0 .. 6.0` step `0.15`, a cell occupied iff
`logit > threshold`, pooled AUC with ties counted one half, best F1 with the sweep's tie-break
(highest F1, then higher precision, then lower threshold).

### Four join conditions, checked before any number is published

The instrument **refuses** rather than reports if any of these fails:

1. both receipts are `status: passed`, 36 deciding boards, 28 nonempty, `667` occupied and `4,933`
   empty known cells, 12 ancestry groups;
2. the matched population is exactly `2,986` cells;
3. the recomputed **model as-is** matched read reproduces the registration receipt's own
   `summary.as_is` F1 and AUC (`0.439024`, `0.933138`) to `1e-6`;
4. the recomputed **model oracle** read reproduces its `summary.oracle` (`0.574181`, `0.970233`),
   and the recomputed **pixel full-population** read reproduces the ceiling receipt's own
   `measured.operating.f1` and `pooled_auc` (`0.978006`, `0.998765`).

Condition 4's two halves are a cross-receipt join in the standing V-004 pattern: the pass is only
admitted if it reproduces a published number from a receipt it did not write.

## Declared reads and the reading rule

- `registration_gain` = matched-oracle best swept F1 - matched-as-is best swept F1.
- `representation_gap` = matched-pixel best swept F1 - matched-oracle best swept F1.
- `boundary_gap` = matched-oracle best swept F1 - matched-oracle F1 at `0.0`.

The reading is fixed now, by the first matching branch:

| reading | condition |
|---|---|
| **`two-factor-first-order`** | `registration_gain >= 0.10` **and** `representation_gap >= 0.10`: no single lever can clear the `0.70` at `0.0` gate, because two independent terms each exceed it |
| **`registration-first-order`** | `registration_gain >= 0.10` and `representation_gap < 0.10` |
| **`representation-first-order`** | `representation_gap >= 0.10` and `registration_gain < 0.10` |
| **`single-deficit`** | neither reaches `0.10`; the campaign's read-side framing stands |

`boundary_gap` is reported beside every branch and **cannot move the verdict**: the level family is
already closed by three measured negatives ([M14](M14_READ_CENTRING.md),
[M15](M15_STACK_CONTIGUITY.md), [M21](M21_LEVEL_COMPOSITION.md)), and this pass does not re-open it.

No branch is a quality, accuracy, real-image or complete-state claim. No gate moves: the fit gate
stays `FALSE` at `0 / 28`, and no threshold is proposed for any arm.

## Envelope and receipts

One read-only CPU job, no Torch import and no forward pass: ceiling **128 MiB**, wall **120 s**,
**3 work items** (three receipts read). Any launch that dies to a harness limit is preserved under
`runs/attempts/` with a note and re-recorded under a fresh number.

Receipts this pass must produce:

1. `runs/m27-threefactor-r1-job.json`, written by `scripts/run_bounded.py`;
2. `docs/evidence/m27-three-factor-r1/scope.json`, written by the instrument only after all four
   join conditions hold.

## Limits, stated before the measurement

- **The matched subset is not the deciding population.** `2,986` of `5,600` cells (`53.3%`) are
  co-scoreable, and the exclusion is not random: it removes cells whose model centre falls outside
  the target board, i.e. the boundary band. Every matched number is reported with that population
  beside it and is never substituted for the deciding population.
- **Different arms.** The model side is the `4x`-budget arm; the pixel read-out is arm-independent.
  The pixel reference is therefore a property of the capture, not of that arm's weights.
- **The pixel read-out is granted the target lattice**, so the pixel number is a ceiling, not a
  model, and not a decoder.
- **One seed, one synthetic population, training-fit boards.** No generalization, real-image,
  calibration or complete-state claim follows.

## What each outcome licenses

- **`two-factor-first-order`** - the read-side lever family was bounded by construction: any single
  read-side lever can move at most the `representation_gap` term, so a gate at `0.70` at `0.0` was
  unreachable even by a perfect read-out change. The next lever must attack the representation
  *and* leave registration to its own declared arm, and the pixel ceiling names the feature class
  that carries the missing information.
- **`registration-first-order`** - registration is the single dominant term, and the next lever is a
  registration-robust read, not a richer one.
- **`representation-first-order`** - the pixel ceiling's advantage is the thing to import, and the
  next lever is a read-out that sees what that 30-feature read-out sees.
- **`single-deficit`** - none of the three terms is first-order at this population; the residual is
  not decomposable this way and the next step is a larger, group-diverse population.

## Measured result: `two-factor-first-order`, and the gate a single lever could not reach

Receipt `runs/m27-three-factor-r3-job.json` (passed, 3 work items, 0.39 s, peak 38,686,720 B of the
declared 128 MiB, kernel Job guard). Every one of the four join conditions held before the evidence
directory was written: the two receipts agree on the 28 nonempty boards and the 36 sample ids, the
matched population is exactly `2,986` cells, and the recomputations reproduce
`0.439024 / 0.933138` (model as-is matched), `0.574181 / 0.970233` (model oracle),
`0.379147 / 0.800604` (model as-is full) and `0.978006 / 0.998765` (pixel full).

| read | population | AUC | F1 at `0.0` | best swept F1 (threshold) |
|---|---|---|---|---|
| model, as-is | 5,600 deciding cells | 0.800604 | 0.379147 | 0.392496 (`0.60`) |
| model, as-is | 2,986 matched cells | 0.933138 | 0.439024 | 0.543974 (`2.10`) |
| model, oracle | 2,986 matched cells | 0.970233 | 0.574181 | **0.762590** (`2.25`) |
| pixel, same cells | 2,986 matched cells | 0.999195 | 0.974763 | **0.987179** (`1.05`) |
| pixel, full | 5,600 deciding cells | 0.998765 | 0.978006 | 0.985936 (`1.05`) |

**Verdict `two-factor-first-order`.** On identical cells:

- **registration `+0.218616`** best swept F1 (`0.543974 -> 0.762590`), with AUC `0.933138 -> 0.970233`:
  granting the correspondence the model's own cells need is worth more than any single lever this
  campaign has measured;
- **representation `+0.224590`** (`0.762590 -> 0.987179`, AUC `0.970233 -> 0.999195`): even correctly
  registered, the learned read is a fifth of F1 short of a 30-feature colour read-out on the same
  cells;
- **boundary `+0.188409`** for the model's own oracle read (`0.574181` at `0.0` against `0.762590` at
  `2.25`), reported and inert, because the level family is closed.

The three terms are measured on one population and each is threshold-free except the boundary term;
**they are not claimed to be additive**, and the matched population is not the deciding population.

### What this explains, and what it licenses

The standing gate is occupied-cell F1 `>= 0.70` at `0.0`, from a published `0.379147`. The
representation term alone is `0.224590` - so **no read-side lever of any kind could have cleared
that gate**, even one that perfectly closed the representation gap, because the gate is also paying
registration and level. That is the structural reason the twenty-three single-lever arms all
returned a number between `0.40` and `0.48` at `0.0`: each was moving one of three first-order
terms while the gate was the sum.

The measured consequences, in the order the numbers rank them:

1. **Registration is not a settled question.** It is worth `+0.219` best swept F1 on identical cells,
   and the geometry family never moved placement by its own declared `25%` floor. A read-out that is
   *tolerant* of a `2.575`-cell mean offset is a lever with a measured prize, and it has not been
   built in the form this decomposition asks for (the free channel reads `0.607085` alone).
2. **The representation has a proven-sufficient target.** The pixel read-out reaches `0.987179` on the
   same cells, so the missing information is in the raster at the cell and not in the label policy -
   the label hypothesis is refuted twice now. A read-out that sees what that 30-feature read sees, at
   the model's own lattice, is a declared representation lever with `0.224590` of measured headroom.
3. **The boundary is the smallest term and stays closed.** `0.188409` at the model's own optimum is
   real, but `0.0` is where the standing gate lives and three prior passes measured the level family
   out; this pass does not re-open it.

No lever is ranked by preference here: the numbers say registration and representation are equal
first-order terms, so the honest next move is to build the representation lever (it is additive,
zero-initialized, needs no new data and is measurable at the standing `64`-step protocol) and to
carry a registration-robust read as its separately declared sibling rather than a composition.

## Receipt note

Three launches preceded the canonical one and none is evidence. `runs/m27-three-factor-r1-job.json`
and `-r2-job.json` failed inside the instrument (the first read its own durable receipt before argv
had finished, the second restricted `as_is` to all known cells instead of the matched population),
and both are preserved with their logs; the canonical run is `runs/m27-three-factor-r3-job.json`
writing `docs/evidence/m27-three-factor-r3/scope.json`. The instrument tests are
`runs/m27-tests-r4-job.json` (19 tests, passed); the two failures before them are preserved as
`runs/m27-tests-r1..r3-job.json` and recorded as V-149.

## Limits

- The matched population is `53.3%` of the deciding cells and is biased away from the board edge,
  which is exactly where a `2.575`-cell offset spills. Every matched number is reported beside the
  full-population number and is never substituted for it.
- The pixel read-out is arm-independent (raster plus committed labels); the model side is the
  `4x`-budget arm whose mapping is committed. The two are therefore not two arms of one experiment.
- The pixel read-out is granted the target lattice and is a ceiling, not a model or a decoder.
- One seed, one synthetic population, training-fit boards; the fit gate stays `FALSE` at `0 / 28`
  and no accuracy, real-image, calibration or complete-state claim follows.
