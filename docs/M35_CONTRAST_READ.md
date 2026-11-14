# M35: the contrast read at the support lattice

Status: **predeclared before any M35 scoring run.** Mechanism, protocol, unchanged gate thresholds, the
bar that decides the arm and both branch interpretations are fixed here.

## Where this comes from

`docs/evidence/m34-support-lattice-r1/scope.json` settled placement: the support-derived lattice took the
placement term from `-0.103647` to `-0.004778`, clearing the predeclared reducibility bar. It also moved
the short stage to **closure** - `-0.158049` points, `-0.356605` of span - and recorded this pass's step:
give the hidden projection the statistics' within-sample contrast beside their absolute values.

## The correction this document makes to its predecessors

The pixel read is `30` statistics per cell - nine sub-cell samples x three channels (`27`) plus three
per-channel spreads - not `31` as the M33 and M34 documents said (`PIXEL_QUERY_FEATURES = 9*3+3 = 30`,
the value the model has used since the read was introduced). Nothing measured changes: the arm, the
weights and the receipts are identical, only the description of the input width was wrong. This arm's
input is therefore `60`: `30` absolute + `30` deviations. V-158 records the correction.

## The hypothesis

M34's arm places the read correctly but over-predicts: at the fixed boundary it holds recall `0.981308`
with precision `0.159817` (`210` true positives, `1104` false positives, `4` false negatives) and its
ranking fell to AUC `0.883260` from the winner's `0.954529`. Its best swept F1, `0.385925`, occurs where
precision is already `0.254873` - so the per-cell decision is not separating inside the board, it is
riding a level. A projection that sees each statistic together with its deviation from the board's own
mean has the board's internal contrast available and can place the decision inside the board instead of
on a level. If that is the binding constraint, closure should recover.

## The mechanism, exactly

`--occupancy-pixel-query 8`, one declared form of the existing pixel read:

- **Placement**: unchanged from form 7 - `pixel_support_lattice(valid, rows, columns, device, dtype)`,
  the bounding box of the capture's valid pixels, mapped bottom-first.
- **Input**: the unchanged nine sub-cell samples over the declared interior extent, giving the `30`
  statistics, concatenated with `statistics - statistics.mean(dim=1, keepdim=True)`, i.e. each statistic's
  deviation from the mean of that statistic over the sample's own `260` lattice cells. `60` inputs.
  Both moments come from the sample itself: no population, no label, no scored cell, no fitted constant.
- **Projection**: the unchanged M29/M30 form-3 hidden layer `Linear(60, 64) -> SiLU -> Linear(64,
  dimension)`, zero-initialised output. The hidden width, the activation, the output width and the
  zero-init are the same; only the input width follows from the declared `60` features, which is
  `+1,920` parameters (`30*64`) and no new output parameter.
- **Single change** against the M30/M31 winner (`m30-level-calibrate-r5`): same seed, same source
  checkpoint, same 64 steps, same 1,152 MiB ceiling. The consumer's boundary stays the fixed global `0.0`.

## Protocol

One profile, one 64-step study, one 36-board probe under the `m35-contrast` owned family (1,152 MiB study,
640 MiB probe), scored by `scripts/m35_contrast_read.py` on the same 2,986 cells and both fixed label
vectors, joining the unchanged control `0.219626`, the reference `0.543974` and the pixel ceiling
`0.987179`; the winner is re-scored in the same pass.

## Gate thresholds, unchanged

1. per-cell F1 at `0.0` `>= 0.70`, on both fixed label vectors;
2. ranking: pooled AUC `>= 0.90` **and** best swept F1 `>= 0.70`;
3. attribution over the unchanged control `>= 0.05`.

## The bar that decides the arm, fixed before scoring

This arm is the **new operating point** only if it beats the winner's re-scored numbers on both declared
comparisons:

- F1 at `0.0` `>= 0.372881` (the winner's re-scored value), and
- best swept F1 `>= 0.689775` (the winner's re-scored value).

Anything less is a diagnostic arm, however much placement it keeps.

## Branch interpretations, pre-committed

- **PASS** (both comparisons met): the arm becomes the operating point; re-verify its artifacts, keep the
  split as the account, and record the next structural step.
- **FAIL**: diagnose *why* closure under-fits from the measured evidence - which cells fail at the fixed
  boundary, by how much they miss, where on the board they sit, and whether precision or recall is
  missing - name the binding constraint (capacity, decision rule, or representation), and take one further
  structural change attacking that cause, predeclared the same way, implemented and measured in this pass.
  No fallback to sampling recipes and no scalar tuning.

## Receipts

`runs/m35-contrast-{profile,study,probe}` and `docs/evidence/m35-contrast-lattice-r1/scope.json`, with the
diagnosis (if needed) in `docs/evidence/m35-contrast-lattice-r1/diagnosis.json`. Refused attempts keep
their receipts; only completed runs are quoted.

## Measured

Study `m35-contrast` (probe `-registration`) on the winner's arm with form 8, the
unchanged 64-step protocol, peak `1,059,209,216` bytes. Scored by `scripts/m35_contrast_read.py` on
the fixed 2,986 cells and both label vectors (`docs/evidence/m35-contrast-lattice-r1/scope.json`), the
winner re-scored in the same pass, the diagnosis reading the arm's own logits
(`diagnosis.json`). No oracle, no per-board rule, no fitted threshold.

| read | F1 at `0.0` | AUC | best swept F1 |
|---|---|---|---|
| M30/M31 winner (anchor, re-scored) | `0.372881` | `0.954529` | `0.689775` |
| contrast read, identity vector | `0.277778` | `0.945596` | `0.570048` (at `3.15`) |
| delta | -0.095104 | -0.008933 | -0.119726 |
| contrast read, anchored vector | `0.280025` | `0.793416` | `0.455863` |

Gate block, unchanged thresholds: identity per-cell `false`, ranking
`false`, attribution `true`
(+0.350422 over the unchanged control `0.219626`); anchored per-cell
`false`, ranking
`false`, attribution
`true`.

**The bar is not met** (`F1 at 0.0 >= 0.372881` and `best >= 0.689775`):
this is a diagnostic arm, not an operating point. But it is the first arm in the campaign to move closure
off the floor - best swept F1 `0.570048` against M34's `0.385925`, `+0.184123`
-and the first whose AUC (`0.945596`) is within `0.008933` of the
winner.

## The diagnosis: the read separates, then over-calls

| class at `0.0` | count | margins | median |
|---|---|---|---|
| true positive | 210 | `0.936171`-`5.464106` | `3.307301` |
| false positive | 1088 | `0.003737`-`4.629019` | `1.352717` |
| false negative | 4 | `-2.998682`-`-1.187106` | `-1.59634` |
| true negative | 1684 | `-4.460561`-`-0.027747` | `-2.596762` |

Precision `0.161787`, recall `0.981308` at the fixed boundary, and the best
swept point sits at `3.15` - so the ordering is informative (AUC
`0.945596`) while the fixed boundary lands inside the wrong part of the margin distribution.
The false positives are not marginal: `1088` of them with median margin
`1.352717` and a maximum of `4.629019`. They
cluster where the board is dense: `405` in rows 0-4,
`528` in rows 5-9, `11` in rows
10-14, `144` in rows 15-19, against
`214` occupied cells among the scored rows.

**Named constraint: decision structure, not capacity and not the sample.** The features carry the
information (AUC `0.945596`, best swept `0.570048`), the arm is not starved (`60` inputs,
`655,163` parameters), and placement is not the loss in this arm. What fails is that a per-cell read with
no notion of the board's shape can call any colour-contrasting cell occupied - which is why the false
positives are confident and sit in bands.

## The span split

| term | points | fraction of the `0.443205` span |
|---|---|---|
| closure | +0.026074 | 0.058831 |
| placement | -0.114185 | -0.257635 |
| representation | +0.531316 | 1.198804 |

The sums are asserted in code. Closure is positive for the first time - the contrast features do close
`0.058831` of the span - while the negative term is now the anchored vector's own gap
(placement `-0.114185`), which is this arm's ordering favouring the identity vector rather
than a sampling loss.

## The concrete next structural step this stage implies

The M36 pass declares it and is measured separately: keep form 8's features and projection, and make the
read carry the stack's own bottom-first shape - one crossing per column via the campaign's declared
`stack_contiguous` envelope - so a confident positive cannot sit above an empty cell in its own column.
`docs/M36_MONOTONE_CONTRAST.md` fixes that bar before its run.

## What this establishes, and what it does not

- Within-sample contrast is worth `+0.184123` best swept F1 at the same placement and the
  same budget: the M34 under-fit was a feature-level problem, and closure responds to feature structure.
- The fixed boundary, not the ranking, is what remains out of reach at this capacity: precision
  `0.161787` at recall `0.981308`.
- Not shown: whether a structural constraint recovers precision without losing the ranking; whether the
  remaining `1.198804`-span representation term is reachable at `64` steps. One
  seed, training-fit boards, synthetic gates only.

## Receipts

`runs/m35-contrast-{profile,study,probe}` and `docs/evidence/m35-contrast-lattice-r1/{scope,diagnosis}.json`. Reader attempts that were refused before a completed read are kept as `runs/m35-contrast-lattice-r{1,2}-job.json` (wrong case key, then wrong cell binding) and `r{3,4,5}` (diagnosis keystone defects, V-159); the completed reads are `r6`-`r9`, with `r9` writing this receipt.
