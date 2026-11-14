# M25: charging the read for its pairs, and what happened

M24 showed the exactness barrier is ordering: no level in the complete decision set makes any board exact, the record's
own projections leave every board inexact, and the record-best arm carries an 18.95% inversion rate where exactness needs
0.0. This pass builds the one mechanism that decomposition leaves standing - a declared within-board pairwise ordering
term - proves its properties mechanically, and measures it under the unchanged protocol.

Evidence: `docs/evidence/m25-rank-r1/scope.json`. Instrument: `scripts/record_rank_read.py`, receipt
`runs/m25-curate-r1-job.json` at the curator's 1900 MiB ceiling. Predeclared arms, constants and refusals were fixed before
any receipt: `PAIR_RANK_MARGIN = 1.0`, `PAIR_RANK_WEIGHT = .5`.

## The mechanism

`pair_rank_term` charges, for every board carrying both classes among its *known* visible cells,
`mean(relu(margin - (occupied - empty)))` over the complete cross-class product: no sampling, no pair ordering, no class
weighting, exactly zero when the read separates the two classes by the declared margin.

It is an objective term, not a tensor. It adds no parameter, no module and no configuration field, so construction order
is uninfluenced by it and the base parameter values are the ones the base arm draws - there is nothing to re-seed. What it
does change is the gradient the read receives, and that is the whole lever.

Declared refusals (`scripts/train_m2_stream.py`): refused without `--occupancy-lattice-free 2` (it supervises the free
channel's own read), and refused beside the five boundary/margin terms and structural centring, because those supervise
the level while this term is invariant to it, so an arm carrying both could not attribute a change to either. The envelope
and the crossing rule compose with it by declaration: they decide at the read's own split rather than moving it, which is
exactly the composition this pass asks about.

## Mechanical properties, held by tests

`tests/test_pair_order.py` (20 tests) proves, in the established style:

- **exactly zero on a board separated by the margin**, and positive on an inverted board charged exactly its overlap;
- **complete and order-free**: the value equals the mean over every occupied/empty pair computed by hand, the pair count
  is `occupied × empty`, and permuting the cells leaves it unchanged (no sampling, no double counting);
- **no board-level component**: adding any constant to a whole board's read leaves the term unchanged, so it cannot pay a
  pair's charge by moving the level of the arm it is declared on;
- **blind to unknown cells and to spawn rows**: dropping an unknown cell equals removing it, and junk above the labelled
  board changes nothing;
- **non-zero gradient at the all-zero read**: both classes receive gradient, and one optimizer step moves the read, so the
  zero fixed point is left on the first step;
- **no tensor and an identical read**: the model gains no parameter, its forward output is bitwise unchanged when the term
  is switched on, and the term's gradient does reach the real occupancy head.

## What was measured

Same seed, same population, 64 steps, 54 forwards, the standing 81-point grid, the standing `0.0` boundary, and the
curator at its 1900 MiB ceiling. `repro` is the published read re-run through this harness: it reproduces the published
control to the last digit (`0.8226433610683003` AUC, `0.381640` F1@0), which is the harness's own identity check.

| arm | declared levers | AUC | F1@0 | strict exact | row error @0.0 | positive share @0.0 | within-board AUC | inversion rate | lattice / free / local AUC |
|---|---|---|---|---|---|---|---|---|---|
| repro | none (published control) | 0.822643 | 0.381640 | 0 / 28 | +7.819 | 49.95% | 0.775698 | 0.224302 | - / - / - |
| rank | free channel + pair order | 0.786891 | 0.212997 | 0 / 28 | +12.007 | 99.93% | 0.753922 | 0.246080 | 0.776508 / 0.725182 / - |
| rank-stencil | free channel + envelope + stencil + pair order | 0.918588 | 0.430619 | 0 / 28 | +1.443 | 42.91% | 0.828472 | 0.183009 | 0.815097 / 0.654808 / 0.607662 |

Frozen comparison arms from the catalogue, unchanged and not re-measured: control `0.8226433610683003` / `0.381640` /
0 exact / row error +7.819; stencil `0.9170200020605955` / `0.480776` / 0 exact / +0.404; norm `0.8562806069091949` /
`0.414810` / 0 exact / +1.007; norm-cross `0.8357421228570795` / `0.377106` / 0 exact / +1.925; envelope
`0.8980661706446594` / `0.350636` / 0 exact / +3.322; centred `0.859823433104044` / `0.414810` / 0 exact / +1.007.

## The verdict, in the record's own vocabulary

**`pair-order-fails`. Strict exact is still 0 of 28 for every arm, and this pass did not clear it.**

The composition came closest and still fell short of a clearance by every measure the record uses:

- **Ranking barely moved.** `rank-stencil` scores `0.918588` AUC against the record-best stencil's `0.9170200020605955`
  (+0.001568, under the catalogue's 0.02 attribution tolerance), and its inversion rate is `0.183009` against the frozen
  `0.189525` - a fall of 0.006516, below the 0.01 fall this pass declared as the bar for "moves". So it neither cleared
  exactness nor moved the quantity it was built to move by a margin the record would call a move.
- **The boundary paid for it.** The term is invariant to a board's level, so the likelihood term keeps choosing the level
  and the ordering term spends its gradient elsewhere. `rank-stencil`'s positive share rose from 27.94% to 42.91%, F1@0
  fell `0.480776` -> `0.430619` (-0.050157) and row error rose +0.404 -> +1.443. `rank` alone is worse still: the bare
  read declares 99.93% of cells occupied at the boundary (precision 0.119, recall 1.000), AUC `0.786891` against the
  control's `0.8226433610683003`, and its inversion rate *rises* to `0.246080` against `0.224302`.
- **The residual is still ordering.** `rank-stencil`'s within-board AUC is `0.828472` and it still has no board exact at
  any level. The mechanism was aimed at the right quantity and, at this capacity and budget, moved it by 0.0065 of the
  0.19 it would have to remove.

So the honest statement of where the barrier now stands: M24 showed no level or projection can repair an inexact board;
M25 shows that supervising the pairs directly - the one lever that acts on the quantity that is actually wrong - does not
buy the separation either, at 64 steps. The term's own gradient is one-sided and shift-invariant, and the cheapest way
for it to reduce its charge on a board where the occupied cells are not separable is to lift the empties toward them; the
observed collapse of precision at the boundary is that failure mode, not a tuning artefact, and it is why the record-best
arm stays `stencil` (`0.9170200020605955` / `0.480776`).

## Limits

- One seed, one population, 28 boards, three new arms. The two compositions reuse the frozen arms' exact protocol and the
  reproduction arm proves the harness is unchanged, but a negative at this budget is not a claim about a longer fit.
- The `rank` arm's first study hit its 1200-second wall cap in finalisation and was re-run under the same flags and cap
  (`runs/attempts/m25-rank-rank-walltimeout`); no guard, cap or step count was relaxed.
- The margin and weight were declared before the receipt and not searched afterwards. No second weight was tried: with a
  boundary cost this large, searching the weight would have been tuning the arm to a number rather than testing the
  mechanism.
