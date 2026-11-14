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

## Measured

The declared study is `m32-mass-v4`: the M30 winner's arm with the mass term added, trained at
the unchanged 64-step protocol and scored with `scripts/m32_mass_match_read.py` on the fixed 2,986
cells and both label vectors (`docs/evidence/m32-mass-match-r1/scope.json`, peak
`1,061,507,072` bytes). No oracle, no per-board rule and no fitted threshold appears anywhere in
this receipt.

| read | F1 at `0.0` | AUC | best swept F1 |
|---|---|---|---|
| M30/M31 winner (anchor, re-scored) | `0.372881` | `0.954529` | `0.689775` |
| mass-shaped arm | `0.277778` | `0.942324` | `0.602198` |
| delta | -0.095104 | -0.012205 | -0.087577 |
| unchanged control `0.219626` / reference `0.543974` | - | - | attribution +0.382572 |

## The branch resolves to fail

Per-cell F1 at `0.0` `0.277778` against the `0.70` bar, AUC
`0.942324` against `0.90`, best swept `0.602198` against `0.70`. Gate block:
per-cell `false`, ranking `false`,
attribution `true`; `cleared: false`.

The mechanism moved the read in the wrong direction on every separation measure: F1 at `0.0`
-0.095104, best swept -0.087577, AUC
-0.012205. Charging a board's predicted mass toward its occupancy forces the
read's level to a prevalence the ranking does not support at that threshold, so the cells it needed
above `0.0` are pushed below it. The hypothesis that mass placement alone was the obstacle is
refuted on this protocol.

## The span split, and the stage it names

| term | points | fraction of the `0.443205` span |
|---|---|---|
| closure | +0.058224 | 0.131370 |
| placement | -0.103647 | -0.233858 |
| representation | +0.488628 | 1.102488 |

Points and fractions sum to the span and to one (asserted in code before the receipt was written).
Placement is the short stage and the only negative term: the read loses `0.103647`
of best swept F1 when the reference registration is granted, while the representation term still
carries `1.102488` of the span. Since the arm's own lattice read is what the
pixel statistics are sampled from, the deficit this pass can act on next is **where the read samples**,
not how much mass it puts there.

### The concrete next structural step this stage implies

Give the pixel-statistics read a **fixed-lattice** placement so its scores cannot inherit the
predicted lattice's misplacement: the campaign's declared lattice-free channel
(`--occupancy-lattice-free`, reading a coarse map over equal divisions of the capture plus the cell's
own row/column embeddings) combined with the form-3 projection, which removes the sampled placement
from the gate by construction rather than correcting it. That is the step recorded here for the next
pass; it is not run in this one.

## What this establishes, and what it does not

- The M31 reading - that the read's score *shape* was the obstacle - is **not** fixable by matching
  per-board mass; that mechanism is measured and refused.
- Placement, not representation, is now the largest single loss in the split, so the next declared
  step is a placement-free read rather than another supervision term on the score map.
- Not shown: whether the fixed-lattice read recovers the placement loss at this capacity; whether any
  supervision term on mass can help on a read with better placement. One seed, 64 steps, training-fit
  boards, synthetic gates only.

## M33 update: the fixed lattice is refused, and placement is still the stage

The step this receipt recorded was run (`docs/M33_FIXED_LATTICE.md`,
`docs/evidence/m33-fixed-lattice-r1/scope.json`): the same pixel statistics sampled at a constant
lattice over the capture. It costs -0.117251 F1 at `0.0` and
-0.057448 best swept F1 against this receipt's arm, so a constant placement does
not recover the placement term - placement is now measured at `-0.118649` and is still
the only negative term. The next step is recorded as a support-derived lattice rather than another
constant.

