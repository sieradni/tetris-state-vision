# M32: shaping the read's score distribution toward occupancy

Status: **predeclared before any M32 scoring run.** Hypothesis, mechanism, protocol, unchanged gate
thresholds and both branch interpretations are fixed here; the measured section is appended from the
final receipt's own fields.

## The hypothesis, from the M31 receipt's fields

The M31 receipt (`docs/evidence/m31-decision-rule-r2/scope.json`) measured, on the M30 winner
(`m30-level-calibrate-r5`) and the fixed 2,986 cells with both label vectors:

| read | F1 at `0.0` | best swept F1 |
|---|---|---|
| unchanged global `0.0` gate | `0.372881` | `0.689775` |
| declared label-free per-board `otsu_board` | `0.278699` | `0.581359` |
| label-fitted per-board boundary (oracle, excluded) | `0.763916` | `0.763916` |

The read *ranks* well enough for the per-cell bar when a boundary is fitted, but its score mass sits
where no label-free rule finds the split. One cell of neighbourhood context made it worse
(`-0.171118` best swept F1). So the missing thing is not the inputs, the width or the boundary: it is
**what the read's score map is trained to represent**. The hypothesis is that the read's predicted
positive mass per board does not track the occupancy it is looking at, and that charging it directly
places the fixed `0.0` boundary where the split actually is.

## The mechanism, exactly

`--occupancy-mass 1` adds one auxiliary term to the objective, with a fixed declared weight
`MASS_WEIGHT = 1.0` (the occupancy term's own scale is ~0.5, so this is comparable and is not tuned):

```
mean over boards [ (mean_known sigmoid(logit) - mean_known occupancy)^2 ]
```

- Known cells only; `unknown` is never counted as empty. No parameter, no threshold, no grid.
- It is supervision of the *distribution shape*, not of the boundary's location and not of any single
  cell: it says "your scores should put this much mass above `0`", which is what `sigmoid > 0.5`
  already means.
- It reads the training labels' own occupancy fraction per board. **Nothing is fitted at inference**:
  the consumer's boundary stays the fixed `0.0`, and no scored-cell statistic enters the model.
- Single change against the M30 winner: the winner's arm (form 3 + read standardisation) is otherwise
  unchanged, at the same seed, budget and 64-step protocol from `runs/m3-percell-balance-r1`.

## Protocol

One profile, one 64-step study, one 36-board attribution probe under the `m32-mass-` owned family,
1,152 MiB study ceiling, 640 MiB probe ceiling, one heavy job at a time. Scoring is the campaign's
fixed-population read on the same 2,986 cells and both fixed label vectors, joined to the unchanged
control `0.219626`, reference `0.543974` and pixel ceiling `0.987179`. No oracle is used in this pass.

## Gate thresholds, unchanged

1. per-cell F1 at `0.0` `>= 0.70`;
2. ranking: pooled AUC `>= 0.90` **and** best swept F1 `>= 0.70`;
3. attribution over the unchanged control `>= 0.05`.

No threshold, boundary, population, metric or gate is redefined, and no per-board rule is used in any
verdict.

## Both branches, pre-committed

- **PASS** (per-cell clears): record plainly that this is a change in the read, not a decision-rule
  change; report which gate conditions moved and by how much (per-cell, AUC, best, attribution) against
  the M30/M31 winner, and keep the span split as the report of where the improvement came from.
- **FAIL**: report the same gate block and the span split (`closed` / `placement` / `representation`),
  name the stage that is short, and record in the receipt the concrete next structural step that stage
  implies - not a hyperparameter sweep.

## What would make this wrong

- If the term improved only the swept best and not F1 at `0.0`, it would have sharpened the ranking
  rather than shaped the mass, and the pass would be reported as a ranking result, not a gate pass.
- If the mass per board matched but the labels did not follow, the term would be satisfiable by
  degenerate boards; the span split and the attribution gate are the checks against that.

## Receipts

`runs/m32-mass-{profile,study,probe}` receipts and `docs/evidence/m32-mass-match-r1/scope.json` written
by `scripts/m32_mass_match_read.py`. Refused attempts keep their receipts and are reported; only
completed runs are quoted.
