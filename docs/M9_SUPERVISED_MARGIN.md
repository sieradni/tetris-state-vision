# M9 - can the boundary be supervised instead of guessed?

**Status: predeclared before any receipt of this pass.** Reference numbers come from committed records
(`docs/evidence/m8-normalised-boundary-r1/scope.json`, `docs/evidence/m7-calibration-r1/scope.json`). Nothing
here may change once a receipt exists.

## What two failed passes established

The occupancy read has never been told where its decision boundary should be. Everything the project has tried
was chosen *after* training:

- **M7** - a per-board Otsu threshold from the read's own logits, no labels, translation-equivariant: `down`'s
  F1 `0.380898` against control's `0.381640`, reading `calibration-fails`. All arms landed at recall `0.99` /
  precision `0.24` because between-class variance is the wrong criterion at `11.9%` prevalence.
- **M8** - a declared output parameterisation instead of a rule, z-scoring each board's logits inside the model:
  F1 `0.384661` (`+0.003021`), AUC `-0.034181`, reading `normalisation-moves-without-clearing`. Zero became the
  board's median, so `48.2%` of cells sat above it and precision could not pass `0.247`; the arm hit the ceiling
  of its own parameterisation.

Both passes leave the same hole: the objective never distinguishes "these cells are occupied" from "this board's
occupied and empty logits are separated". This pass puts that distinction into the loss.

## The mechanism: a within-board margin hinge

**Declared lever (`--occupancy-margin-hinge 1`).** For every board in the batch that carries **both** occupied and
empty *known* visible cells:

```
hinge(board) = max(0, margin - (lowest_occupied_logit - highest_empty_logit))
term         = mean over qualifying boards
```

with the margin and the weight **fixed here, before any receipt**:

| constant | value | why this value, declared now |
|---|---|---|
| `MARGIN_HINGE_MARGIN` | `1.0` | one logit of separation: the two modes may touch but must not overlap. It is a unit of the read's own scale, not a threshold - it says nothing about where on the logit axis the gap sits. |
| `MARGIN_HINGE_WEIGHT` | `0.5` | one eighth of the occupancy likelihood term's `4.0` factor, so the term shapes the boundary without outvoting the fit. |

Declared properties:

1. **It is supervision, not a boundary rule.** It names no threshold, chooses no operating point at inference,
   adds no parameter, buffer or state key, and cannot change the ranking of cells *within* a mode.
2. **It uses only labels the occupancy term already uses**, and it excludes `unknown` cells from both sides, so an
   invisible cell is never read as empty (the standing visibility contract). Boards with one class absent are not
   counted rather than being scored as zero.
3. **It reads no prevalence.** The margin is a margin; nothing counts cells, and no quantile of the logits is
   taken.
4. **It applies to every arm that enables it, identically.** There is nothing arm-specific in the term.

### The operating point stops being hand-chosen, including for the offsets

This supersedes the way the project has been treating the boundary. The `-1` and `-2` row offsets were declared
sampling-geometry levers that happened to *move* the best threshold (`0.90 -> 2.70`); any gain they showed at their
own hard-coded best boundary was a boundary effect as much as a sampling effect, which is exactly what M7 could not
separate. From this pass on, **the boundary is a supervised quantity**: an offset arm may only be admitted beside
the hinge (so both arms are trained under the same boundary supervision), and never re-scored at a threshold
chosen for it after the fact. Nothing about this block measures an offset - the declaration fixes the policy so the
next offset study starts from a supervised boundary instead of a chosen one.

## Arms - one catalogue, one block

| arm | boundary supervision | role |
|---|---|---|
| `m9-margin-control-r1` | `--occupancy-margin-hinge 0` | must reproduce the recorded `0.8226433610683003` and `0.381640` |
| `m9-margin-hinge-r1` | `--occupancy-margin-hinge 1` | the arm this pass measures |

Both are 64-step continuations from `runs/m3-percell-balance-r1` (so the arm continues the *published* function's
weights, not a normalised or offset read), re-captured in one catalogue, with the M7/M8 envelope unchanged:
capacity `64`, positional identity, length endpoint, detail tap `1`, stencil `0`, resolution `0`, context `0`,
column `0`, offset rows `0`, normalise `0`, placed `0`, factor `4`, positive weight `4`, `512` capture, seed
`20261026`, rate `.001`, 12 sweep groups, 1,792 MiB studies with 64 work items, 640 MiB sweeps with 54 forwards,
1,900 MiB curation, 81-point grid on `[-6, 6]`.

**Composition is refused, deliberately.** The trainer rejects the hinge beside any lever that changes what is read,
where it is sampled, or where zero sits (`context`, `stencil`, `resolution`, `deform`, `placed`, `column`, `offset
rows`, `lattice-free`, `normalise`). One arm, one question; there is no sweep and no second margin.

## What is read, and how

**Primary: plain-binary per-cell F1 at the standing `0.0` boundary** - the sweep's own `per_cell_f1`: threshold at
zero, score the cells, no calibration and no post-processing. **Secondary: deciding AUC**, reported with M8's
declared tolerance so the two passes are directly comparable. Alongside both: best swept F1 (context only, never
the metric), out-of-fold F1, precision/recall at `0.0`, the row-error diagnostic, the operative hinge value at the
end of training if the study records it, exact-board count and per-board F1.

## Declared readings - fixed now

Primary delta is `hinge F1@0 - control F1@0`, with control required to reproduce `0.381640` and `0.8226433610683003`
exactly first; if it does not, the comparison does not count and the record must say so.

1. **`margin-clears-the-boundary`**: delta `>= 0.05` (the attribution floor). Supervision moves the operating point
   by more than any post-hoc rule or parameterisation could, and clears M7's `0.431640` bar too.
2. **`margin-moves-without-clearing`**: `0 < delta < 0.05`. The loss can move the boundary, but not far enough to
   matter.
3. **`margin-fails`**: delta `<= 0`. Telling the network to separate does not make `0.0` an operating point.

Secondary reading, M8's vocabulary and tolerance: **`ranking-preserved`** if `hinge AUC >= control AUC - 0.02`, else
**`ranking-degraded`**.

Gates unchanged: per-cell F1 `0.70`, ranking AUC `0.90`, ranking F1 `0.70`, attribution floor `0.05`, 400 bootstrap
draws at `20261026`, MDE `2.802 x cluster SE`. Verdict vocabulary unchanged. Fit gate reported on exact boards and
FALSE until all 28 non-empty boards are exact.

## Declared limitation, and what a negative result would mean

The hinge constrains the **separation** between the modes, not the **location** of the gap on the logit axis. A read
whose occupied mode sits at `-2` and empty mode at `-4` satisfies it perfectly and predicts nothing above `0.0`. The
location is currently supplied by the likelihood term (positive weight `4`), and this pass tests whether that is
enough once separation is required.

So if the reading is `margin-fails`, the honest conclusion is not "supervision cannot work" but "supervision of
separation cannot work without supervision of location", and the next structural idea is declared now so a negative
result cannot drift into thresholds:

- **Two-sided anchor (preferred follow-up):** `max(0, margin - lowest_occupied) + max(0, margin + highest_empty)` -
  the same within-board hinge, but anchored to zero on both sides, so `0.0` becomes a *supervised* boundary rather
  than an inherited one. Still supervision, still no threshold, no prevalence, no parameter.
- **Supervised boundary parameter (alternative):** one learned scalar bias per model, applied to the occupancy
  logits and committed to `0` at inference, so the boundary is a fitted quantity like any other; admitted only if it
  is supervised by the loss rather than fitted after it.

Neither is measured here, and no margin or weight will be re-tuned to flip this reading.

## Limits, declared

One dataset, 36 deciding images, 28 non-empty boards, 12 clusters, one seed, one 64-step budget, one architecture,
no held-out population. A margin cannot add information: it constrains a difference of two logits that the loss
already sees, so any AUC change is a change in what was learned, and the record reports it as such. Nothing here is
a real-screenshot, complete-state or quality claim.

## Measured

Receipts: `docs/evidence/m9-supervised-margin-r1/scope.json` (curation, 1,900 MiB, receipt
`runs/m9-margin-curation-inner-r2.json`), studies `runs/m9-margin-<arm>-r1/` (64 steps, 1,792 MiB), sweeps
`runs/m9-margin-<arm>-r1-sweep-r1/` (54 forwards, 640 MiB). One catalogue (`eae71837...`), one block, both
studies `complete` at 64 steps (`164.7 s`, `158.2 s`).

**Reproduction first.** Control came back bit-identical in this catalogue: AUC `0.8226433610683003` and plain
F1 at `0.0` `0.381640`, exact string equality on both, so the comparison counts.

| arm | AUC | plain-binary F1 at `0.0` | best swept F1 | precision / recall at `0.0` | out-of-fold F1 | row error |
|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 0.2363 / 0.9910 | 0.400399 | +7.819 |
| margin hinge | 0.864667 | **0.251701** | 0.474210 @ `2.70` | 0.1440 / 0.9985 | 0.429457 | +10.691 |

Operating counts at `0.0`: control `661 TP / 2,136 FP / 6 FN`, hinge `666 / 3,959 / 1`, of `667` occupied and
`4,933` empty cells. The hinge the arm actually paid, from its own step log: mean `2.403` over 64 steps, `0.0` on
16 of them (a quarter of the steps were already separated), `0.0` on the final step.

### Reading: `margin-fails`, and `ranking-preserved`

Primary delta **`-0.129939`** (bar `+0.05`) - the declared reading is **`margin-fails`**. Secondary AUC delta
**`+0.042024`**, so `ranking-preserved` and in fact far above the tolerance. The rest of the block:
best swept F1 **`+0.062176`**, out-of-fold **`+0.029058`**, row error **`+2.872`**. Verdict `not-cleared` (per-cell
F1 `0.251701` against `0.70`; AUC `0.864667` against `0.90`; attribution not met because the primary is
negative), fit gate **FALSE** (`0` of `28` exact boards).

### What actually happened - the declared limitation, measured

The hinge did its declared job and it did not do the job the pass was named for. Separation was reached: the
term fell to `0.0` on the final step, i.e. the board's lowest occupied logit cleared its highest empty logit by
more than the margin on that batch. Ranking improved more than any occupancy lever has managed without a
sampling change: `+0.042024` AUC, `+0.062176` best swept F1 (clearing the `0.05` attribution floor on its own),
`+0.029058` out-of-fold. And the operating point got **worse**: `-0.129939` F1 at `0.0`, because the boundary
drifted *up* - the best swept threshold moved `0.90 -> 2.70`, the read now calls `4,625` of `5,600` cells
occupied at zero (recall `0.9985`, precision `0.1440`), and the row diagnostic worsens to `+10.691`.

That is precisely the limitation declared before the block: the hinge constrains the **gap** between the modes,
not the **location** of the gap on the logit axis, and nothing in the term prefers a gap that straddles zero.
The loss raised the occupied mode (positive weight `4` plus a margin) and satisfied the margin from above; a read
that separates its modes at `+2.70` is a better ranker and a worse thresholded detector at `0.0`. So the honest
conclusion is the declared one: **supervising separation cannot place the operating point, and the objective
still does not control location.**

### Next structural step, now supported by the data

The predeclaration named it and this block makes the case concrete: the **two-sided anchor**,

```
term = mean over boards [ max(0, margin - lowest_occupied) + max(0, margin + highest_empty) ]
```

which keeps the separation the one-sided hinge found (`+0.042024` AUC) and additionally requires the highest
empty logit to sit at least `margin` below zero, so `0.0` becomes a *supervised* boundary instead of an
inherited one. Its margin and weight must be fixed in a predeclaration before its receipt, measured with the same
arm pair (control plus anchor) in one catalogue, and read with this pass's primary metric and reading vocabulary.
The one-sided lever stays in the engine as a measured ranking result and as `0` in every other arm.

Two consequences the record should carry forward:

1. **The offset policy declared above now has a baseline.** The `-1` offset's `+0.048086` AUC was measured against
   control's `0.822643`. The hinge now reaches `0.864667` with no sampling change, so the offset's gain must be
   re-measured on top of a supervised boundary rather than assumed additive - it may already be inside it.
2. **A ranking gain is not an operating point, again.** M7, M8 and M9 now agree from three directions: ranking
   moves cheaply, the boundary does not. The next arm has to supervise location, and if the two-sided anchor also
   fails, the alternative named in the predeclaration is a single learned boundary bias supervised by the loss and
   committed to `0` at inference.

Defects of this pass: none in the engine or the read path - both studies and all six read stages passed on the
first launch, the run family was registered before training, and the curator's only repair was to record the paid
hinge it had promised to report (V-130).
