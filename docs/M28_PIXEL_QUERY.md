# M28: the per-cell pixel-statistics read-out

Status: **predeclared before this pass has any receipt of its own.** The lever, the matched
control, the budget, the gates, the attribution and the failure criterion below are fixed; the
measured section is appended only after the canonical receipts exist.

## Where this comes from

M27 measured the occupancy deficit on one population
([M27_THREE_FACTOR.md](M27_THREE_FACTOR.md), `docs/evidence/m27-three-factor-r3/scope.json`):

| read | AUC | F1 at `0.0` | best swept F1 |
|---|---|---|---|
| model, own lattice (2,986 matched cells) | 0.933138 | 0.439024 | 0.5440 |
| model, registration granted | 0.970233 | 0.574181 | 0.7626 |
| 30-feature pixel read-out, same cells | 0.999195 | 0.974763 | 0.9872 |

Registration and representation are equal first-order terms (`+0.218616` and `+0.224590` best
swept F1), and the boundary term is `+0.188409`. The read-side family is therefore bounded: no
read-side lever can close more than the representation term, and this pass tests the largest
measured one.

The question is deliberately narrow. The pixel ceiling was **granted the target lattice**. The
model must read at its **own predicted** lattice, whose mean absolute placement offset is `2.575`
target cells. So the open quantity is not "are the colours informative" - that is measured - but
**how much of the `0.2246` a learned head recovers when the same feature class is computed at the
model's own placement**.

## The single declared change

`ModelConfig.occupancy_pixel_query` (default `0`, published). When `1`, the occupancy read-out is
given 30 deterministic per-cell statistics computed from the prepared raster at the lattice the
stage already decodes:

- nine sub-cell samples on a 3x3 grid over the declared interior extent (`0.3` of a cell in each
  lattice axis), giving twenty-seven channel values, plus three per-channel spreads;
- `occupancy_features = features + W * statistics + b`, with `W`, `b` of a `Linear(30, 192)`
  **zero-initialized**, so at step zero the block adds exactly zero and the arm is strictly
  additive against its matched control;
- the sampler reads `pixels` and the model's own `lattice` only: **no label, no target lattice,
  no decoded state, no threshold and no prevalence**.

The lever is refused beside every other occupancy read-out selector (`--occupancy-context`,
`--occupancy-detail`, stencil, resolution, deform, column, lattice-free, offset rows, and the
five level/supervision transforms), so it is one declared statement about what the read sees.

## Matched pair, one budget

Both arms are **continuations** of the published checkpoint `runs/m3-percell-balance-r1`, at the
campaign's standing 64-step protocol: `--model accuracy --component-capacity 64
--component-identity-readout positional --component-endpoint-term length --occupancy-factor 4
--occupancy-positive-weight 4 --curriculum stream-v1 --capture-max-side 512 --seed 20261026
--steps 64 --learning-rate 0.001 --evaluate-every 64 --development-groups 6`, 1,152 MiB aggregate
commit, 1,200 s runner wall, one heavy job at a time.

| arm | run | `occupancy_pixel_query` |
|---|---|---|
| control | `runs/m28-pixelread-base-r1` | `0` |
| lever | `runs/m28-pixelread-lever-r1` | `1` |

Nothing else differs. The control is a continuation too, so the comparison isolates the block, not
"more training".

## Scoring, on the unchanged population and metric

Each arm is scored read-only by `scripts/sweep_stream_occupancy.py` on the deciding population
(36 training images, 28 nonempty boards, 667 occupied and 4,933 empty known cells, 12 ancestry
groups), standing 81-point grid `-6.0 .. 6.0` step `0.15`, occupied iff `logit > threshold`, pooled
AUC with ties counted one half, best F1 with the sweep's tie-break. The arm's own registration
mapping is measured by `scripts/probe_registration_attribution.py` (36 forwards), and the same
three reads M27 defined - own lattice, oracle registration, pixel read-out on the same cells - are
then recomputed by `scripts/pixel_query_read.py` from the arm's own receipts.

**The lever CLEARS iff, at its own lattice on the full deciding population:**

1. **per-cell gate** - occupied-cell F1 at `0.0` is `>= 0.70` (`percell_gate.CELL_F1_THRESHOLD`);
2. **ranking gate** - `occupancy_sweep.sweep_verdict` is `informative` (pooled AUC `>= 0.90` and
   some swept threshold reaches F1 `>= 0.70`);
3. **attribution floor** - its best swept F1 exceeds the matched control's by `>= 0.05`.

**Otherwise the lever is NOT CLEARED.** If either arm cannot finish inside the declared envelope
the outcome is **NOT ADMISSIBLE AS DECLARED**, recorded with its measured peak and wall; no ceiling
is raised and no run is repeated to pass.

## Declared attribution, fixed before the run

The pass reports, for each arm, three reads on the arm's own co-scoreable cells:

- `registration_gain` = oracle best swept F1 - own-lattice best swept F1 (the same arm, one
  population);
- `representation_gap` = pixel best swept F1 - oracle best swept F1 (the same cells);
- the fraction of the M27 `0.5440 -> 0.9872` span the lever closes, computed as
  `(lever own-lattice best - 0.5440) / (0.9872 - 0.5440)` on the M27 matched population.

**If the lever does not improve**, the pass must record which of two causes holds, from the
committed numbers rather than from narrative:

| cause | condition |
|---|---|
| **`features-unavailable-at-own-lattice`** | the pixel read-out on this arm's co-scoreable cells keeps its M27-level F1 (>= `0.90`), while the arm's own read does not move: the information exists at the placement the arm used, and the learned head still failed to exploit it |
| **`head-cannot-exploit`** | the arm's own read moves by less than the `0.05` floor while its oracle read also stays within `0.05` of the control's oracle read |
| **`placement-only`** | the arm's own read does not improve but its oracle read beats the control's oracle read by `>= 0.05`: the added features help only once registration is granted |

A missing or failed receipt yields no attribution.

## Receipts this pass must produce

1. `runs/m28-pixelread-{base,lever}-r1-profile-r1` (2 steps, 1,152 MiB / 300 s / 2 work items).
2. `runs/m28-pixelread-{base,lever}-r1` (64 steps, 1,152 MiB / 1,200 s / 64 work items), each
   admitted only by its own profile.
3. `runs/m28-pixelread-*-r1-sweep-r1{-profile-r1,}` (2 then 54 forwards, 640 MiB).
4. `runs/m28-pixelread-*-r1-registration-r1` (36 forwards, 640 MiB).
5. `runs/m28-pixelread-r1-read-r1-job.json` (the reader, 128 MiB) writing
   `docs/evidence/m28-pixel-query-r1/scope.json` after its joins hold.
6. The owned family `m28-pixelread-*` declared in `record_stream_run.OWNED_RUN_PATTERN` and bound
   by a test.

## Limits, stated before the measurement

- Synthetic, single seed, training-fit boards; no real-image, calibration, generalization or
  complete-state claim, and the fit gate is not re-opened.
- The matched population is a subset of the deciding cells; every matched number is reported beside
  its full-population counterpart.
- The pixel read-out is a ceiling with the target lattice granted; it is not a candidate model.
- A 64-step continuation can only show that the block is used at this budget, not that the
  representation is sufficient at a longer one.

## Measured

Receipts: `docs/evidence/m28-pixel-query-deciding-r3/scope.json` (the deciding read, 128 MiB, one work
item, table annotated per population) and `docs/evidence/m28-reference-provenance-r1/scope.json` (the
provenance read that pins each cited baseline). They join the committed reference receipts
`runs/m3-occscale-registration-r2` and `runs/m3-occpix-ceiling-r1-r2` with the two trained arms' own
probes (`runs/m28-pixelread-{base,lever}-r1-registration-r1`) and their study job receipts. No
retraining, no forward pass, no population, protocol, metric, threshold or gate change.

### Provenance of the cited baselines (V-151)

The four figures the lever is scored against were previously quoted as one row of prose, which reads
as a single population. They are not one population. Read from the receipts and recomputed by
`scripts/m28_reference_provenance.py` - which refuses to write unless each figure reproduces its
source block exactly:

| figure | measured as | population | arm | source (block) |
|---|---|---|---|---|
| `0.3791` = `0.379147` | F1 at threshold 0.0, own lattice | full deciding set, 5600 known cells, 667 occupied | `m3-occscale-registration-r2` | `runs/m3-occscale-registration-r2/result.json` (summary.as_is_full) |
| `0.439024` = `0.439024` | F1 at threshold 0.0, own lattice | matched set, 2986 cells, 214 occupied | `m3-occscale-registration-r2` | `runs/m3-occscale-registration-r2/result.json` (summary.as_is) |
| `0.5440` = `0.543974` | best swept F1 over the standing grid | matched set, 2986 cells, 214 occupied | `m3-occscale-registration-r2` | `docs/evidence/m27-three-factor-r3/scope.json` (reads.model_matched_as_is.best.f1) |
| `0.7626` = `0.76259` | best swept F1 with the committed registration granted | matched set, 2986 cells, 309 occupied | `m3-occscale-registration-r2` | `docs/evidence/m27-three-factor-r3/scope.json` (reads.model_oracle.best.f1) |
| `0.9872` = `0.987179` | best swept F1 of the 30-feature pixel read-out | matched set, 2986 cells, 309 occupied | `m3-occpix-ceiling-r1-r2 (arm-independent read)` | `runs/m3-occpix-ceiling-r1-r2/result.json` (cases[].pixel_logits with the reference labels) |
| `0.978006` = `0.978006` | F1 at threshold 0.0 of the pixel read-out | full deciding set, 5600 known cells, 667 occupied | `m3-occpix-ceiling-r1-r2 (arm-independent read)` | `runs/m3-occpix-ceiling-r1-r2/result.json` (measured.operating) |

`0.3791` is therefore the **reference arm's own-lattice F1 at `0.0` on the full 5,600-cell deciding
set**, not a property of the M28 control and not the same population as `0.5440` / `0.7626` /
`0.9872`, which are all measured on the matched 2,986-cell co-scoreable set. The reference arm is
`m3-occscale-lever-r1`, weights `150b3ca3...`, job
`Local\TetrisStateVision-45973ed3bb344119b4dfbe7d2cda2204`; the pixel read is job `Local\TetrisStateVision-97361e7d76db43d696bc16f1ca6959b4`.

The M28 control is a **different checkpoint** (`m28-pixelread-base-r1`, weights
`75fc5a91a4de3571...`), and its own committed full-population read is reproduced
exactly by this pass: `0.284833` F1 at `0.0` and `0.742741` AUC on
5600 known cells, 667 occupied, matching its sweep `runs/m28-pixelread-base-r1-sweep-r1/result.json`
(job `Local\TetrisStateVision-f5e8ffce8e8d41929393d357f1455809`). Its matched-population value of `0.12987`
is lower because those cells are the reference arm's co-scoreable subset - it is a population effect,
not a defect in the control read, and no control figure was ever `0.3791`.

### The deciding table, with each row's population stated

| population | read | arm | cells (occupied) | AUC | F1 at `0.0` | best swept F1 |
|---|---|---|---|---|---|---|
| full-5600 | reference own lattice | `m3-occscale-lever-r1` | 5,600 (667) | 0.800604 | **0.379147** | 0.392496 |
| matched-2986 | pixel read-out, arm-independent | pixel ceiling | 2,986 (309) | 0.999195 | 0.974763 | **0.987179** |
| matched-2986 | reference own lattice | `m3-occscale-lever-r1` | 2,986 (214) | 0.933138 | **0.439024** | **0.543974** |
| matched-2986 | reference, registration granted | `m3-occscale-lever-r1` | 2,986 (309) | 0.970233 | 0.574181 | 0.762590 |
| matched-2986 | control own lattice | `m28-pixelread-base-r1` | 2,986 (214) | 0.574178 | 0.129870 | 0.219626 |
| matched-2986 | **lever own lattice** | `m28-pixelread-lever-r1` | 2,986 (214) | **0.898191** | **0.282638** | **0.425703** |
| matched-2986 | control, registration granted | `m28-pixelread-base-r1` | 2,986 (309) | 0.477262 | 0.172477 | 0.186075 |
| matched-2986 | lever, registration granted | `m28-pixelread-lever-r1` | 2,986 (309) | 0.760646 | 0.271980 | 0.422803 |

Every arm row above is scored on the same 2,986 rows with the same two label vectors, fixed from the
reference receipt (identity: 214 occupied; anchored: 309). On that population the lever beats its
matched control by `+0.324013` AUC, `+0.152768` F1 at `0.0` and `+0.206077` best swept F1. The
comparable F1-at-`0.0` reference on these cells is `0.439024`; the `0.379147` full-population figure
belongs to the reference arm on 5,600 cells and is shown here only so it can no longer be mistaken
for a matched-population baseline.

### Span closure and the split, in points and in span fractions

Span = `0.987179 - 0.543974 = 0.443205`, both endpoints on the matched 2,986-cell population. Each
arm's three terms sum to that span; the reader asserts the point sum and the fraction sum before it
writes.

| arm | closed `own - 0.5440` | placement `anchored - own` | representation `0.9872 - anchored` |
|---|---|---|---|
| control | `0.219626 - 0.543974 = -0.324348` (-0.7318) | `0.186075 - 0.219626 = -0.033551` (-0.0757) | `0.987179 - 0.186075 = +0.801104` (+1.8075) |
| lever | `0.425703 - 0.543974 = -0.118271` (-0.2669) | `0.422803 - 0.425703 = -0.002900` (-0.0065) | `0.987179 - 0.422803 = +0.564376` (+1.2734) |

Parenthesised values are fractions of the span; the three add to exactly one span
(`-0.118271 - 0.002900 + 0.564376 = 0.443205`). Relative to its **control** the lever closes
`0.206077 / 0.443205 = 0.464970` of the span; relative to the `0.543974` **reference arm** it ends
`0.118271` below that anchor.

**The residual is representation for both arms, and placement is measured at zero.** Granting the
reference registration moves the lever's best swept F1 by `-0.002900` (0.65% of span) and the
control's by `-0.033551`; the pixel read-out on the same placement reaches `0.987179`. The features
are present where the arm reads, and `1.2734` span fractions sit between its registration-granted
read and the pixel read.

### Verdict on the deciding population: **NOT CLEARED**, attribution gate passed

| gate | bar | lever | result |
|---|---|---|---|
| per-cell F1 at `0.0` | `>= 0.70` | `0.282638` | fail |
| ranking (AUC `>= 0.90` and best `>= 0.70`) | `0.90` / `0.70` | `0.898191` / `0.425703` | fail |
| attribution over its control | `>= 0.05` | `+0.206077` | **pass** |

Two of three fail, so the lever remains **not cleared**; the attribution gate passes with a positive
measurement, correcting the earlier "all three conditions fail". The ranking gate misses by `0.001809`
AUC while the per-cell bar is `0.417362` away, so the block buys ordering and very little absolute
operating point.

`cause = representation-limited-partial-gain`: the arm's own read moves `0.464970` of span over its
control, the features are available at its placement (`0.987179`), and the placement term is
`-0.0065` of span. The cause is read from the within-arm placement term, never from a cross-arm
anchored delta, because anchoring also swaps the label vector (214 against 309 occupied).

### Secondary read: the per-arm 3,316-cell population (retained, not comparable)

`docs/evidence/m28-pixel-query-r2/scope.json` keeps the first M28 verdict, which scored each arm on
its own co-scoreable cells. It is **per-arm**, so it is not cross-arm comparable and is not used for
any verdict here.

| arm | own pairing | registration granted | pixel, same cells |
|---|---|---|---|
| control | 0.2801 | 0.2194 | 0.9889 |
| lever | 0.5079 | 0.5345 | 0.9771 |

### What this establishes, and what it does not

- The feature class is available at the model's own lattice and the block is used: `+0.324` AUC and
  `+0.206` best swept F1 over a matched control on one fixed population.
- It is **not** sufficient: the lever stops `0.118271` below the `0.543974` reference arm on the same
  cells and `0.561476` below the `0.987179` pixel read, with the placement term at zero.
- The `0.3791` figure is the reference arm's full-population read and is never the control's; no
  deciding-population claim rests on it.
- Not shown: a longer or wider block, a different budget, another seed, real images, or that the
  representation is sufficient at any budget. One seed, training-fit boards, no generalization claim.

## M29 update: the form-1 verdict is superseded as the campaign's best read

`docs/M29_PIXEL_QUERY_FORM.md` and `docs/evidence/m29-pixel-form-r2/scope.json` separate the causes
of this arm's `0.425703` own read. Amplitude is excluded (`0.703541` of the head's gain),
scale/offset is excluded (`-0.012545`), local context is excluded
(`+0.036428`), and **capacity is confirmed**: the same
statistics through a 64-wide hidden layer reach `0.610101` best swept F1 and
`0.946604` AUC on the same 2,986 cells, `+0.184398`
over this arm and above the `0.543974` reference. The form-1 numbers above remain the record of this
arm; they are no longer the campaign's best read of the pixel-query design, and the `0.149202`
positive span closure now belongs to form 3.
