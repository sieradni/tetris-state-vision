# M26: the symmetric pair margin, and the closure of the pair-supervision axis

M25 charged one direction of every occupied/empty pair and was paid by lifting empties - 99.93% positive share, precision
0.119, AUC below control. This pass replaces it with a symmetric charge, proves what symmetry does and does not buy, and
measures it under the standing protocol next to a control that reproduces the published numbers exactly.

Evidence: `docs/evidence/m26-gap-r1/scope.json`, curated by the shared recorder `scripts/record_rank_read.py --family
m26-gap` (receipt `runs/m26-curate-r1-job.json`, 1900 MiB ceiling). Constants were declared before any receipt:
`PAIR_GAP_MARGIN = 1.0`, `PAIR_GAP_WEIGHT = .5`.

## The mechanism

`pair_gap_term` charges each cross-class pair by the *closer* of its two directional violations,
`min(relu(margin - (occupied - empty)), relu(margin + (occupied - empty)))`, i.e. `relu(margin - |gap|)`, over the complete
product of known visible cells. It is:

- **symmetric**, so an arm cannot lower it by calling one mode the other;
- **exactly zero only when the pair is separated by the declared margin**, and charged the full margin on every pair when
  the two modes are collapsed together - the failure mode M25 measured;
- **invariant to a global additive shift**, so it adds no level term and no tensor, and cannot absorb its charge by moving
  a board's level.

And it pays a price that is stated in the code rather than discovered afterwards: **a charge that is the same whichever
class is called occupied is blind to orientation**. An inverted-but-separated board is free. The symmetric term is
therefore a guard on the pair gap, not an orientation teacher, and this pass measures exactly that.

Wiring: admission join `require_occupancy_pair_gap`, weight `PAIR_GAP_WEIGHT` in `joint_loss`, contract line, CLI flag
`--occupancy-pair-gap` refused without `--occupancy-lattice-free 2`, refused beside the one-sided `--occupancy-pair-rank`
(one statement per arm about the same pairs) and beside the five boundary/margin terms and structural centring (which
supervise a level this term is invariant to); owned-run family `m26-gap-`; the curator and launch script are shared with
M25 rather than copied.

## Mechanical properties, held by tests

`tests/test_pair_gap.py` (23 tests) proves: exactly zero on a margin-separated board; exactly the margin on a fully
collapsed board and the per-pair charge on a partly collapsed one; the same value whichever class is called occupied and
under a mirrored read; invariance to pair iteration order; invariance to any global shift; blindness to unknown cells and
to spawn rows; a live gradient with one step moving the read once it is off the tie; and no added parameter with a
bitwise-identical forward pass plus a gradient that reaches the real occupancy head.

Two properties are pinned rather than claimed, because no implementation can have both:

- **the exact tie is a symmetric saddle**: at an all-equal read the charge is exactly the margin and its slope is exactly
  zero, because a charge even in the pair gap has zero derivative at zero. A swap-symmetric charge cannot also have a
  non-zero slope at the tie;
- **the two terms differ exactly on inverted pairs**: on a board with no inversion the symmetric and one-sided charges
  coincide, which is why this lever can only change an arm through inversions and collapse - and why the measured arms are
  read against those two quantities specifically.

## What was measured

Same seed, 64 steps, 54 forwards, the standing 81-point grid and `0.0` boundary, the curator at its 1900 MiB ceiling.

| arm | declared levers | AUC | F1@0 | strict exact | row error @0.0 | positive share @0.0 | within-board AUC | inversion rate | lattice / free / local AUC |
|---|---|---|---|---|---|---|---|---|---|
| repro | none (published control) | 0.822643 | 0.381640 | 0 / 28 | +7.819 | 49.95% | 0.775698 | 0.224302 | - / - / - |
| gap | free channel + symmetric pair margin | 0.866499 | 0.332749 | 0 / 28 | +9.040 | 59.36% | 0.797969 | 0.202031 | 0.868533 / 0.754066 / - |
| gap-stencil | free channel + envelope + stencil + symmetric pair margin | 0.881989 | 0.403049 | 0 / 28 | +1.403 | 46.66% | 0.818553 | 0.190075 | 0.817134 / 0.649526 / 0.639399 |

`repro` reproduces the published control to the last digit (`0.8226433610683003` AUC, `0.381640` F1@0), so the harness is
unchanged. Frozen comparisons, copied and not re-run: stencil `0.9170200020605955` / `0.480776` / 0 exact / +0.404; norm
`0.8562806069091949` / `0.414810` / 0 exact / +1.007; norm-cross `0.8357421228570795` / `0.377106` / 0 exact / +1.925;
envelope `0.8980661706446594` / `0.350636` / 0 exact / +3.322; centred `0.859823433104044` / `0.414810` / 0 exact /
+1.007; control as in the reproduction row.

## The verdict

**Strict exact is still 0 of 28. The record-best arm remains `stencil` (`0.9170200020605955` / `0.480776`).**

The symmetric lever does what M25's could not, and still does not reach the bar:

- **On the bare read it is a real gain and the collapse is gone.** `gap` scores `0.866499` against control's
  `0.8226433610683003` (**+0.043856**, beyond the catalogue's 0.02 attribution tolerance), and its inversion rate falls
  from `0.224302` to `0.202031` (**-0.022271**, beyond the 0.01 bar). The positive share at the boundary is 59.36%, not
  M25's 99.93%: the read is no longer collapsing its two modes, which was the whole point of charging both directions.
- **But the composition regresses the record-best read.** `gap-stencil` scores `0.881989` against stencil's
  `0.9170200020605955` (**-0.035031**) and F1@0 `0.403049` against `0.480776` (**-0.077727**), with its boundary share
  pushed from 27.94% to 46.66%. Charging the gap of a read that already separates better spends capacity on pairs that were
  mostly not inverted, and the level is where the cost lands.
- **Exactness did not move at all.** `gap-stencil`'s inversion rate is `0.190075` against stencil's `0.189525`; within-board
  AUC is `0.818553`. Exactness needs 0.0 inversions, and the lever's own definition cannot supply orientation, so no board
  can become exact through it.

**The axis is closed.** Both readings of pair supervision have now been measured on the standing protocol: the one-sided
charge failed by collapse, the symmetric charge avoided the collapse and improved the bare read, but it is orientation-blind
by construction and its composition with the record-best read is a regression. Neither produces a strict exact board, and
further variants of this family would be tuning a lever whose definition excludes the quantity exactness needs. The residual
is where M24 put it: the record-best read's within-board discrimination (`0.818553`-`0.828472` within-board AUC, ~19% of
cross-class pairs inverted), not the decision boundary, not the composition, and not the direction of the pair charge.

## Limits

- One seed, one population, 28 boards, three arms; the composition is measured against frozen numbers from the catalogue,
  not re-run.
- The margin and weight were declared before the receipt and not searched afterwards; no second value was tried, because a
  sweep of the weight would have been tuning rather than testing the mechanism.
- The positive-share comparison against M25 is a comparison of two separately trained runs, not a paired ablation; it is
  cited because the two arms share every setting except the charge's symmetry.
