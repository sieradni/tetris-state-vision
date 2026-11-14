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
