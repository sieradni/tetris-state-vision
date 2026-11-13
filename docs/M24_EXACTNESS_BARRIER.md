# M24: what the exactness barrier actually is

The record held 18 measured arms and not one strict exact board. This pass stops guessing at the cause and takes the
near-miss apart cell by cell, row by row, over the six arms that matter: the record best (`stencil`), the best normalised
arms (`norm`, `norm-cross`), the de-meaned M22 form (`centred`), the M20 envelope, and the control.

Evidence: `docs/evidence/m24-exactness-r1/scope.json` - per board (missed cells, direction, missed rows, row totals, row
signed margins, within-board AUC, inversions, exact-level search) and per arm (aggregates). Read-only: every number comes
from the frozen sweeps the catalogue already scored, no checkpoint, no forward, no threshold admitted into the record.

## The finding

**No level would have made any board exact, and the reason is ordering, not level.**

| arm | mean missed cells at 0.0 | false positives | false negatives | within-board AUC | boards exact at any exact level | fewest wrong cells at the best level | median inverted pairs |
|---|---|---|---|---|---|---|---|
| stencil | 49.68 | 48.86 | 0.82 | 0.818993 | 0 | 1 | 151.5 |
| centred | 66.61 | 66.39 | 0.21 | 0.802217 | 0 | 1 | 171.5 |
| norm | 66.61 | 66.39 | 0.21 | 0.806938 | 0 | 1 | 168.0 |
| norm-cross | 76.57 | 75.93 | 0.64 | 0.774361 | 0 | 1 | 204.5 |
| envelope | 87.57 | 87.39 | 0.18 | 0.753676 | 0 | 1 | 252.5 |
| control | 76.50 | 76.29 | 0.21 | 0.775698 | 0 | 1 | 168.5 |

Three mechanical results, each of which rules out a different explanation:

1. **It is not the threshold, and it is not the grid.** Searching *every distinct level the read can be cut at* - not the
   81-point grid - makes **0 of 28 boards exact in every arm**. The best board in the whole record still has exactly
   **1** wrong cell at its best level. A level can only fix a board whose occupied cells all outrank its empty ones;
   none of the 168 boards in this table does.
2. **It is not the composition.** Applying the record's own consistency projections - the monotone envelope, and the
   envelope followed by the per-board crossing rule - and then searching every exact level on the projected read leaves
   **0 of 28 boards exact under every projection in every arm**. A projection can move a cell the read already orders
   above or below a level; it cannot invent separation the read does not have.
3. **It is ordering, and the size of the deficit is measured.** The within-board AUC is `0.818993` for the record best and
   `0.753676`-`0.806938` for the others, which is an **inversion rate of 18.95%** of all (occupied, empty) pairs on the
   best arm and `21.2%`-`26.3%` on the rest. Exactness on a board requires an inversion rate of exactly `0.0`. The best
   board already has one inverted pair; the median board has about 150.

The direction of the misses explains the shape of every earlier result in the record. At the `0.0` boundary the read
declares far more cells occupied than the labels do - `48.86` false positives against `0.82` false negatives on the best
arm - so every arm's problem *at the boundary* looks like a level, and every level mechanism in M14-M23 moved it. What
the level mechanisms could not do, and what the crossing rule could not do, was remove the inversions underneath: a board
with `151` inverted pairs cannot be made exact by moving a line.

## What this means for the exactness rule, stated plainly

- **The barrier is not the operating point.** No per-board level, on the exact level set, rescues a single board, so
  tuning, calibration, centring, de-meaning, contrast normalisation and the crossing rule were all attacking a
  component that is *necessary but not sufficient*.
- **The barrier is the per-cell read's within-board discrimination.** To produce one strict exact board the read must
  rank every occupied cell above every empty one on that board - an inversion rate of `0.0`. The record's best is `0.19`.
- **The requirement is therefore quantitative.** Closing it by composition is impossible (proven above), so the only
  path is a read whose ordering is perfect on whole boards. Nothing in the record is within a factor of the precision
  this needs: `18.95%` of pairs inverted against `0%`, on 28 boards simultaneously.

This is a negative result and it is reported as one. No mechanism was invented to move the number: the decomposition rules
out the level family that every previous pass extended, and inventing a constraint that cannot restore ordering would have
been a way of dressing a near-miss as progress.

## What is still decidable, and what is not

Decidable, and now measured: whether a level exists (no), whether a projection can restore exactness (no), how far the
ordering is from exactness (18.95% -> 0%), and where the miss sits (false positives above the stack, `48.86` per board on
the best arm, with the false negatives near zero).

Not decidable from this evidence, and named rather than guessed: whether a read trained with an objective that is
*explicitly* the ordering the rule needs - a pairwise within-board ranking term over occupied and empty cells - could
push the inversion rate far enough down. That is the only mechanism the decomposition leaves standing, because it is the
only one that acts on the quantity that is actually wrong. It is not measured here, and this document does not claim it
would work.

## Positive control on the decomposition itself

A negative result is only as good as its instrument, so the checker is controlled:

- a synthetic board whose read is separable, cut at any level between its classes, is reported exact (`exact_at_any_level`
  true, `fewest_wrong_cells` zero);
- the same board with a **single** pair inverted is reported not exact at any level, with `fewest_wrong_cells` one -
  exactly the case the record's best board is in;
- adding a constant to every cell of a board changes no verdict, because the search is over the read's own values;
- the counts the decomposition reports for the six arms equal the counts the curator recorded (`exact_boards_at_zero` is
  zero in both), so the two instruments agree on the record rather than on a re-derivation.

The instrument is `tests/test_exactness_barrier.py`, and it holds the evidence file to those claims rather than to a
sentence.

## Limits

- Six arms of the record, 28 boards, one seed. The decomposition uses the frozen sweeps those arms were scored from, so
  it cannot be an artefact of re-running the model.
- The exact-level search is over the read's own distinct values per board, which is the complete set of levels that can
  change any decision; it is not a claim about levels outside that set, which cannot change a decision.
- The projections tested are the ones the record contains (monotone envelope, crossing rule). Other decoders exist; none
  of them can restore ordering, which is the argument, not the enumeration.
