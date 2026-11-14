# M29: why the pixel-query read collapses, decided by mechanism

Status: **predeclared before any M29 run.** The variants, their predicted signatures, the
population, the metric, the gates and the reporting form are fixed here; the measured section is
appended only after the canonical receipts exist.

## The fact this pass has to explain

On the fixed matched-2,986 deciding population (`docs/evidence/m28-pixel-query-deciding-r3/`):

| read | F1 at `0.0` | best swept F1 |
|---|---|---|
| pixel read-out on the same cells | 0.974763 | **0.987179** |
| reference arm, own lattice | 0.439024 | 0.543974 |
| **M28 lever (`occupancy_pixel_query = 1`), own lattice** | **0.282638** | **0.425703** |
| M28 control, own lattice | 0.129870 | 0.219626 |

The same 30 deterministic statistics that a fixed read scores at `0.987179` are handed to a learned
projection and buy only `0.425703`; the placement term is `-0.002900` of a `0.443205` span. The
information arrives and is then not converted. This pass separates the competing mechanisms with
one instrument instead of a hyperparameter sweep.

## The instrument: four mechanisms, one variant each, one predicted signature

Each variant is a **single declared change** to the same block, zero-initialized so step zero is
bit-identical to the published function, and trained at the unchanged 64-step protocol from
`runs/m3-percell-balance-r1` with the same seed and budget as the M28 arms.

| cause | variant | predicted signature that would confirm it | what excludes it |
|---|---|---|---|
| **gradient starvation / optimizer dynamics** | no training run: read `runs/m28-pixelread-lever-r1/checkpoint/weights.pt` and measure the projection's weight norm and its contribution norm against the cell-feature norm | weights stay at a negligible fraction of the cell-feature scale, so the block never had the amplitude to move the read | weights reach a comparable scale while the read stays low |
| **scale/offset mismatch between raw features and model inputs** | `occupancy_pixel_query = 2`: the statistics are put on the encoder's own convention (colour `2x-1`, spread doubled) before the same zero-initialized projection | F1 at `0.0` on the fixed population rises by `>= 0.05` over form 1 | rise `< 0.05` |
| **projection capacity** | `occupancy_pixel_query = 3`: the same projection with one 64-wide SiLU hidden layer and a zero-initialized output | rise `>= 0.05` over form 1 | rise `< 0.05` |
| **receptive field / local context** | `occupancy_pixel_query = 4`: the 30 statistics beside the mean of the four in-board neighbours, same zero-initialized projection on 60 inputs | rise `>= 0.05` over form 1 | rise `< 0.05` |

**Supervision/attribution is excluded before the run by the committed record**, not by a variant:
the form-1 arm moved the deciding read by `+0.206077` best swept F1 over its matched control
(`docs/evidence/m28-pixel-query-deciding-r3/`), so the occupancy loss demonstrably reaches the
block's parameters; a read-out that receives no gradient could not have moved at all.

## Predeclared decision rule

Let `d(form) = ` best swept F1 on the fixed 2,986-cell population minus form 1's `0.425703`.

1. If the starvation diagnostic shows the projection at a negligible fraction of the cell-feature
   scale, the cause is recorded as `gradient-starvation` and the amplitude result is reported beside
   every variant, because an amplitude defect makes the other three variants unidentifiable.
2. Otherwise the cause is the variant with the largest `d >= 0.05`; ties are recorded as joint and
   each tie's signature reported. If no variant reaches `0.05`, the residue is recorded as
   `additive-form-expressivity`: the features must enter the read-out somewhere other than an
   additive term on the head's input, and no fixable form of this block explains the collapse.
3. If a variant wins, the M28 gates are re-evaluated **on that variant** on the deciding population
   (`per-cell F1 at 0.0 >= 0.70`, ranking `AUC >= 0.90` and best `>= 0.70`, attribution over the
   unchanged control `>= 0.05`), and the span closure and placement/representation split are
   re-derived from its numbers with the arithmetic asserted in code.

## Population, metric, receipts

Every variant is scored read-only on the **identical 2,986 rows and the same two fixed label
vectors** as `m28-pixel-query-deciding-r3` (identity 214 occupied, anchored 309), by the same
`sweep`-free path: the 36-board registration probe supplies per-cell logits, `score` supplies AUC,
F1 at `0.0` and the best point of the standing grid. The per-arm 3,316-cell read stays a labelled
secondary and is not used for any decision.

Receipts, each a bounded job: `runs/m29-pixelform-{centred,hidden,neighbours}-r1-profile-r1` (2
steps, 1,152 MiB / 300 s), the studies `runs/m29-pixelform-*-r1` (64 steps, 1,152 MiB / 1,200 s),
the probes `runs/m29-pixelform-*-r1-registration-r1` (36 forwards, 640 MiB), and the reader
`runs/m29-pixelform-read-r1-job.json` (128 MiB) writing `docs/evidence/m29-pixel-form-r1/`.

## Limits, stated before the measurement

- One seed, 64 steps, synthetic training-fit boards; a null result bounds these mechanisms at this
  budget, not at all budgets.
- The starvation diagnostic is a read of one committed checkpoint, so it can confirm amplitude
  starvation but cannot by itself show the block would have moved at a longer budget.
- No protocol, population, metric, threshold or gate is changed; the control and reference arms are
  the committed ones and are not retrained.

## Measured

Receipts: `docs/evidence/m29-pixel-form-r2/scope.json` (this read, 128 MiB, one work item),
`docs/evidence/m29-amplitude-r1/scope.json` (the checkpoint-tensor diagnostic, 640 MiB, one work
item), and the three variant probes `runs/m29-pixelform-{centred-r3,hidden-r2,neighbours-r2}-registration-*`
(36 forwards each, 640 MiB). Studies: `runs/m29-pixelform-{centred-r3,hidden-r2,neighbours-r2}-job.json`,
64 steps each at the unchanged protocol, peaks 1,050,759,168 / 1,008,046,080 / 994,127,872 bytes
inside the 1,152 MiB ceiling. No retraining of the control or reference arms, and no change to the
population, metric, threshold or gate.

### Amplitude is excluded before any variant is read

The committed form-1 checkpoint carries a projection of weight norm `0.795252`
and bias norm `0.035453` against the occupancy head's `0.404851`.
Composed, the statistics reach the logit with gain `0.284829`, which is
`0.703541` of the head's own gain - far above the predeclared `0.05`
floor. The statistics reach the logit at a comparable fraction of the head's own gain, so amplitude cannot explain a collapse. Gradient starvation and optimizer starvation of the block are
therefore **excluded by measurement**, not by argument.

### Every variant on the identical 2,986-cell population

Same rows, same two fixed label vectors, unchanged control (`m28-pixelread-base-r1`, best
`0.219626`)
and reference (`m3-occscale-lever-r1`, best `0.543974`).

| variant | F1 at `0.0` | AUC | best swept F1 | delta vs form 1 | delta vs control |
|---|---|---|---|---|---|
| form 1, raw single layer (anchor) | 0.282638 | 0.898191 | 0.425703 | -- | +0.206077 |
| form 2, centred inputs | 0.223739 | 0.882089 | 0.413158 | -0.012545 | +0.193532 |
| form 3, 64-wide hidden layer | 0.291464 | 0.946604 | 0.610101 | +0.184398 | +0.390475 |
| form 4, neighbourhood statistics | 0.193578 | 0.914689 | 0.462131 | +0.036428 | +0.242505 |
| pixel read-out, same cells | 0.974763 | 0.999195 | 0.987179 | -- | -- |
| reference arm, own lattice | 0.439024 | 0.933138 | 0.543974 | -- | -- |

### Decision

`cause = form:hidden-r2`. Confirmed: `hidden-r2`. Excluded:
`centred-r3, neighbours-r2`.

- **capacity is confirmed**: the hidden-layer form gains `+0.184398`
  best swept F1 over the raw single layer, clears the `0.05` floor, and reaches
  `0.610101` - above the reference arm's `0.543974` on the
  same cells. A single `Linear(30, 192)` cannot express what the same inputs support with one
  hidden layer; that, not placement and not the features, is the collapse.
- **scale/offset mismatch is excluded**: putting the statistics on the encoder's own convention
  (`2x-1`) changes the read by `-0.012545`, below the floor and
  in the wrong direction.
- **receptive field / local context is excluded**: adding the four-neighbour mean gains
  `+0.036428`, below the `0.05` floor. The
  per-cell feature class is sufficient, as the pixel ceiling already implied.
- **supervision/attribution was excluded before the run**: form 1 moved the deciding read by
  `+0.206077` over its control, so the loss reaches the block.

### Gates re-evaluated on the winning form, and the re-derived split

Span `0.987179 - 0.543974 = 0.443205`, both endpoints on the matched 2,986-cell population.
For the hidden-layer form the three terms are
`+0.066127` closed (`+0.149202` of span),
`-0.089852` placement (`-0.202732`),
`+0.466930` representation
(`+1.053530`); they sum to the span in points and to one in
fractions, asserted in code before the receipt was written. This is the **first positive span
closure** against the `0.543974` reference in the campaign.

| gate | bar | hidden-layer form | result |
|---|---|---|---|
| per-cell F1 at `0.0` | `>= 0.70` | `0.291464` | fail |
| ranking (AUC `>= 0.90` and best `>= 0.70`) | `0.90` / `0.70` | `0.946604` / `0.610101` | AUC pass, best fail |
| attribution over the unchanged control | `>= 0.05` | `+0.390475` | pass |

So the lever is **still not cleared** - two of three gates fail - but the mechanism is now known and
the fix is real: capacity alone recovers `+0.184398` best swept F1
over the same features, the same placement and the same budget. The remaining shortfall is an
operating-point problem (F1 at `0.0` `0.291464` against `0.70`),
not an information or placement problem.

### What this establishes, and what it does not

- The collapse of the M28 read was a **capacity** collapse: the single zero-initialized projection
  was amplitude-healthy (`0.703541` of the head's gain) and could not express the
  mapping its own inputs support.
- Not shown: whether a wider or deeper block continues to improve, whether the operating point can
  be fixed by anything other than a decision rule, and any real-image or generalization claim. One
  seed, 64 steps, training-fit boards.

## M30 update: the level moves, but the per-cell boundary does not

The M29 winner's operating point was attacked directly with three predeclared label-free level
mechanisms (`docs/M30_OPERATING_POINT.md`, `docs/evidence/m30-operating-point-r1/scope.json`).
A learned scale and bias on the projection is too weak (`-0.005750`
F1 at `0.0`) and a data-derived class budget is negative (`-0.015693`);
a gated board standardisation is confirmed at `+0.081417`, lifting best swept F1 to
`0.689775` (`0.954529` AUC) while attribution holds at
`+0.470149`. The read therefore improves, and the ranking gate is
`0.010225` short while the per-cell gate at the fixed `0.0` boundary stays
`0.327119` short. The form-1 and form-3 records above stand as measured; what
changes is that the remaining deficit is now located in the gate's single global boundary, not in
capacity or in the read's ability to order cells.
