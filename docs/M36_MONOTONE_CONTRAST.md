# M36: the contrast read under the stack's own shape

Status: **predeclared before any M36 scoring run.** This document was written after M35 was measured, and
only after: it declares the one structural change the M35 diagnosis calls for, with the same bar, the same
gate and the same discipline.

## What M35 measured, and why the next step is structural

`docs/evidence/m35-contrast-lattice-r1/scope.json`: the contrast read (form 8, 60 inputs) reached F1 at
`0.0` `0.277778` and best swept F1 `0.570048` at threshold `3.15`; AUC `0.945596`, `0.008933` below the
winner. The bar was not met on either comparison. The split reads closure `+0.026074`
(`0.058831` of span), placement `-0.114185` (`-0.257635`), representation `+0.531316` (`1.198804`) -
so closure moved off the floor for the first time in this campaign, and the negative term is now the
label-vector gap rather than placement.

The diagnosis (`docs/evidence/m35-contrast-lattice-r1/diagnosis.json`) says why the gate is still out of
reach, and it is not the ranking:

| class at `0.0` | count | margin range | median |
|---|---|---|---|
| true positive | 210 | `0.936171`-`5.464106` | `3.307301` |
| **false positive** | **1088** | `0.003737`-`4.629019` | `1.352717` |
| false negative | 4 | `-2.998682`-`-1.187106` | `-1.596340` |
| true negative | 1684 | `-4.460561`-`-0.027747` | `-2.596762` |

With the best swept point at `3.15`, the read separates - but at the fixed boundary it calls 1298 cells
occupied against 210 correct ones, and its false positives are *confidently* wrong (`1,088` of them,
median margin `1.35`, up to `4.63`), not marginal noise. False positives sit where the board is dense:
`405` in rows `0-4`, `528` in rows `5-9`, `144` in rows `15-19`. That is the signature of a read that has
no notion of the board's own shape: any cell whose colour statistics resemble a filled cell is called
filled, wherever it sits.

So the binding constraint is the decision structure, not capacity: the per-cell margin is informative
(AUC `0.945596`) and the read is not starved (60 inputs, 655,163 parameters).

## The declared mechanism: form 8 features, one crossing per column

`--occupancy-pixel-query 8 --occupancy-stack-monotone 1`, unchanged features and unchanged projection,
with the campaign's already-declared `stack_contiguous` envelope applied after the read and before the
consumer's boundary:

- the per-column occupancy profile is replaced by its suffix maximum in the project's bottom-first row
  order, so a column crosses `0.0` at most once and that crossing is the column's height;
- a locked stack rests on the floor, so this is the domain's own constraint, not a fitted assumption: no
  label, no prevalence, no threshold, no parameter, and nothing read about the scored cells;
- every cell below a high read is raised with it and every cell above a low read stays empty, which is
  exactly what removes an isolated confident positive in the middle of an empty column - the diagnosed
  failure - and it cannot create a hole inside a stack.

## The single change, and what stays fixed

One change against M35's arm (`m35-contrast`): the monotone envelope. Placement (`pixel_support_lattice`),
the `30` statistics, their `30` sample-mean deviations, the `60`-input hidden projection, the seed, the
source checkpoint, the `64` steps and the `1,152` MiB ceiling are untouched. The consumer's boundary stays
the fixed global `0.0`.

## The bar, fixed before scoring (identical to M35's)

New operating point only if both hold against the winner's re-scored numbers:

- F1 at `0.0` `>= 0.372881`, and
- best swept F1 `>= 0.689775`.

## Gate thresholds, unchanged

per-cell `0.70` on both fixed label vectors; ranking (AUC `>= 0.90` **and** best swept `>= 0.70`);
attribution over the unchanged control `>= 0.05`.

## Branch interpretations, pre-committed

- **PASS**: the arm becomes the operating point; re-verify its artifacts and record the next step.
- **FAIL**: report the gate block and the split, state plainly that the decision-structure hypothesis is
  refused too, and record the measured next step as the representation, which the split then names as the
  short stage; no further sampling or threshold work.

## Receipts

`runs/m36-monotone-{profile,study,probe}` and `docs/evidence/m36-monotone-contrast-r1/scope.json`.
Refused attempts keep their receipts; only completed runs are quoted.
