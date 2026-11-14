# M31: is the per-cell deficit the read or the boundary?

Status: **predeclared before any M31 scoring run.** The rule, the branch test, the oracle's exclusion
and both branches' structural follow-ups are fixed here; the measured section is appended from the
receipt's own fields.

## The question

The M30 receipt (`docs/evidence/m30-operating-point-r1/scope.json`) leaves the calibrated read at
`0.372881` F1 at `0.0` and `0.689775` best swept F1 - 0.010225 short of the ranking bar and 0.327119
short of the per-cell bar - while its AUC is `0.954529`. Two explanations remain, and they imply
opposite next steps:

1. **Decision rule.** The read separates cells well; the campaign's single global `0.0` boundary is
   the wrong place to threshold a read whose correct level is per board.
2. **Representation.** The read's per-cell scores are themselves wrong, and no boundary would clear
   the gate.

## The three reads, reported side by side

All three use the **same** committed arm's logits (`runs/m30-level-calibrate-r5-registration`,
the M30 winner), the same 2,986 fixed cells, and both fixed label vectors. Nothing is retrained here;
this is a scoring read of one frozen model.

| read | boundary | labels used | role |
|---|---|---|---|
| (a) global | fixed `0.0`, exactly as the standing gate scores it | labels only for scoring | the gate; unchanged, not redefined |
| (b) per-board, label-free | each board's own exact Otsu crossing over its visible logits, the campaign's declared `occupancy_crossing_threshold` rule | **none** in the offset | the question |
| (c) per-board, oracle | each board's own best F1 threshold from the standing grid | labels choose the offset | **diagnostic upper bound only** |

(b) is the declared rule and is computed by the committed `otsu_board` itself, applied to each
board's own `20x10` logits, so the offset takes no label, no prevalence, no grid and no parameter.
(c) is an oracle upper bound: it is reported to bound how much of the gap any per-board boundary
could recover, and it is **excluded from every verdict and every gate result**.

## Branch test, declared in advance

- If (b) reaches `>= 0.70` per-cell F1 at `0.0`, the branch is **decision-rule**: the deficit is the
  boundary, and this will be stated plainly as a decision-rule change, **not** as evidence that the
  read improved. The unchanged gate keeps failing on (a).
- Otherwise the branch is **representation**: the deficit is the read's per-cell scores.

## The structural follow-up each branch implies, also declared in advance

- **decision-rule branch**: `--occupancy-read-level 1`, a mechanism change inside the read that
  defines each board's level from the board's own standardised read instead of a free offset, so the
  unchanged global `0.0` boundary lands at each board's own level by construction. One 64-step study
  at the unchanged protocol, scored on (a) only: the gate is not relaxed, and a form whose read level
  is defined by the board must clear the same `0.70` bars.
- **representation branch**: `--occupancy-pixel-query 5`, the M29 form-3 hidden layer (`30 -> 64 ->
  192`) given one cell of spatial context by concatenating the four-neighbour mean into its input
  (the M29 form-4 context, which was excluded only because its single linear layer had no capacity
  for it). One 64-step study at the unchanged protocol, scored on (a) only.

Both follow-ups are single declared changes, trained from the same source checkpoint
(`runs/m3-percell-balance-r1`), same seed, same 64 steps, same 1,152 MiB ceiling, and are scored by
the same fixed-population read. Exactly one runs - the branch the measurement selects.

## What would make this instrument wrong

- If (b)'s offset were fitted on the scored cells, it would be an oracle; the implementation calls
  the committed rule, and a test asserts the offset is invariant to the label vector.
- If (a) were recomputed with any other boundary, it would no longer be the gate; (a) is
  `score(rows_for(arm, fixed, pairing))` verbatim.
- If (b) passed and were reported as showing the read improved, that would be false; the branch text
  says so explicitly.

## Receipts

`scripts/m31_decision_rule_read.py` (640 MiB, the committed `otsu_board` and `torch` are imported) and,
for whichever branch runs, one profile, one 64-step study and one attribution probe under the
`m31-` owned family. Refused attempts keep their receipts; only completed runs are reported.

## Measured

One frozen arm (`m30-level-calibrate-r5`, the M30 winner), the same 2,986 cells and both fixed label
vectors, three boundaries (`docs/evidence/m31-decision-rule-r2/scope.json`). Nothing was
retrained for the comparison.

| read | identity: F1 at `0.0` | identity: best | anchored: F1 at `0.0` | anchored: best |
|---|---|---|---|---|
| (a) global `0.0`, the standing gate | 0.372881 | 0.689775 | 0.373355 | 0.575624 |
| (b) per-board `otsu_board`, **label-free** | 0.278699 | 0.581359 | 0.289638 | 0.491003 |
| (c) per-board, label-fitted: **oracle only** | 0.763916 | 0.763916 | 0.667678 | 0.667678 |

AUC is unchanged by construction inside a board: the identity-vector global read is
`0.954529` and the standing gate's gates are per-cell
`false`, ranking `false`, attribution
`true` - the same verdict the M30 receipt recorded.

### The branch

`(b)` is `0.278699`, far below the `0.7` per-cell bar,
so the branch is `representation`: the deficit is the read's own per-cell scores, not the boundary.
The declared label-free rule *lowers* the read rather than lifting it - the crossing it finds on these
boards sits where almost nothing is predicted, which is a statement about the shape of the read's score
distribution, not about the information in it.

The oracle row is the important qualifier and it is deliberately **not** a verdict: with per-board
boundaries fitted on the labels, the same logits reach `0.763916` F1 at `0.0` on the
identity vector - above the `0.70` bar - and `0.667678` on the anchored vector. So the
information is close to sufficient, the standing gate is not unreachable in principle, and the failure is
that no label-free rule the campaign declares can find the split the read actually makes. `(c)` is
excluded from the branch, from the gate and from every conclusion here.

### The follow-up this branch implies, measured

The declared representation-branch step was the M29 hidden layer given one cell of spatial context
(`--occupancy-pixel-query 5`) on the winning arm, at the unchanged 64-step protocol
(`docs/evidence/m31-followup-r1/scope.json`, study `m31-context-r5`, peak
`1,026,760,704` bytes):

| read | F1 at `0.0` | AUC | best swept F1 | delta best vs M30 winner |
|---|---|---|---|---|
| M30 winner | `0.372881` | `0.954529` | `0.689775` | - |
| context-carrying form | `0.255139` | `0.916663` | `0.518657` | -0.171118 |

It is **negative**: attribution still passes
(`+0.299031` over the unchanged control `0.219626`), but per-cell
F1 at `0.0` and best swept F1 both fall, and span closure is `-0.025317`
(`-0.057123` of the `0.443205` span) with placement
`-0.119742` and representation `+0.588264`. Adding neighbour
context to the same 64-wide hidden layer therefore does not improve the per-cell representation; the
declared step is measured and refused, not deferred.

### What this establishes, and what it does not

- The per-cell deficit is **not** the global boundary: the campaign's own declared label-free per-board
  rule scores worse than the global read, and the gate is unchanged.
- The read carries nearly enough information for the per-cell gate (oracle `0.763916`),
  but its score distribution does not place the meaningful split where a label-free rule looks, and the
  declared context-carrying form made that worse rather than better.
- Not shown: whether a supervision-side mechanism that shapes the read's level (rather than its inputs)
  would let a label-free rule find the split; whether a different boundary family would; any real-image
  claim. One seed, 64 steps, training-fit boards, and the anonymised label vectors the campaign scores.
