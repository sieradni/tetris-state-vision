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
