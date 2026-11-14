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
