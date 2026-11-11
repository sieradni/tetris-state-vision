# What this design can resolve: power and minimum detectable effect

Status: **derived after the replicate read, from committed receipts only. It changes no verdict, no
threshold and no arm, and it adds no training.** It answers the question the campaign had been
answering by judgement: with the design it actually ran, what is the smallest effect it could
resolve, and were the observed deltas even above that floor?

**No figure in this document is typed by hand.** The block between the `power-figures` markers is
written by `scripts/power_analysis.py --render` from `docs/evidence/m3-power-mde-r3/result.json`, and
`tests/test_power_evidence.py` recomputes the entire analysis from the nine committed sweep receipts
and fails if this block, that evidence, or the headline figures quoted in ROADMAP and
CONTINUATION_STATE disagree with it by one digit. Evidence `-r1` and `-r2` are preserved unedited as
superseded passes; `-r3` carries the band.

<!-- power-figures:begin -->

| lever | bootstrap SE | delta-method SE | ratio | MDE | observed | inside the floor | power at a true +0.05 |
|---|---|---|---|---|---|---|---|
| `budget192` | 0.029549 | 0.029680 | 1.0044 | **0.083151** | -0.079599 | yes | 0.3917 |
| `cosine128` | 0.046307 | 0.045144 | 0.9749 | **0.126475** | +0.021276 | yes | 0.1981 |

Design effect of the 36 clusters: `budget192` 9.5 against boards and 16.8 against cells; `cosine128` 12.3 and 29.2.

Audit pair `fit_budget`, 12 groups: SE `0.070036`, MDE `0.196212`, power against its own claim `0.5665`, resolvable at its own design: no.

| target | seeds | clusters | arm-hours | SE | MDE |
|---|---|---|---|---|---|
| `budget192` +0.149, point estimate | 5 | 60 | 1.99 | 0.052417 | 0.146852 |
| `budget192` +0.149, band low | 3 | 36 | 1.20 | 0.044112 | 0.123583 |
| `budget192` +0.149, band high | 83 | 996 | 33.11 | 0.052992 | 0.148462 |
| `budget192` the +0.05 floor | 44 | 528 | 17.55 | 0.017670 | 0.049504 |
| `cosine128` +0.149, point estimate | 4 | 48 | 1.60 | 0.049209 | 0.137864 |
| `cosine128` +0.149, band low | 2 | 24 | 0.80 | 0.050216 | 0.140684 |
| `cosine128` +0.149, band high | 49 | 588 | 19.55 | 0.053179 | 0.148986 |
| `cosine128` the +0.05 floor | 31 | 372 | 12.37 | 0.017676 | 0.049522 |

2 df band on the between-population variance: `0.003959` to `0.231199` (90% scaled chi-square, df 2), point `0.011859`.

| seed | lever | dAUC | d F1 at best boundary | d cost of `0.0` | d transfer cost | d out-of-fold F1 |
|---|---|---|---|---|---|---|
| `20261101` | `budget192` | +0.043 | +0.022 | -0.006 | -0.045 | -0.030 |
| `20261102` | `budget192` | -0.076 | -0.063 | -0.095 | +0.167 | +0.009 |
| `20261103` | `budget192` | -0.030 | -0.169 | +0.062 | -0.150 | -0.257 |
| `20261101` | `cosine128` | +0.102 | +0.137 | -0.123 | +0.049 | +0.063 |
| `20261102` | `cosine128` | +0.041 | +0.061 | +0.004 | +0.040 | +0.105 |
| `20261103` | `cosine128` | -0.005 | -0.097 | -0.079 | +0.055 | -0.121 |

<!-- power-figures:end -->

## Reading it

**Both observed magnitudes sit inside the floor** (`0.083151` and `0.126475` at 80%
power). At a true effect of the campaign's `+0.05` floor this design had power `0.3917` and
`0.1981`. So the published `unconfirmed` verdicts mean "not resolvable here", not "no
effect" - and the rule that produced them was followed exactly; nothing here re-reads them.

**The cells are not the sample.** The 36 clusters carry a design effect against cells of
`16.8` and `29.2`, so the honest sample size is the cluster count, not the
cell count.

**The audit's own claim was inside its own floor.** Its fit-budget pair has `0.196212` as its
minimum detectable effect, and the power it had against its own claimed `+0.1494` was
`0.5665`. The reversal that motivated the re-read could not have been resolved by the design
that produced it.

**Three seeds are not enough once populations count as random.** Treating the three populations as
fixed makes a three-seed `256/384` block look powered; that inference is wrong for a claim about the
method, and the population-random row in the figure block shows why. The decision-relevant reading is
the one where the between-population variance is a random effect.

**The variance component rests on two degrees of freedom, so the recommendation is a band.** Its 90%
scaled chi-square interval widens the required seed count to `3 to 83` seeds
(`1.20 to 33.11` arm-hours) for the audit's `+0.149`, against `44 seeds, 528 clusters, 17.55 arm-hours` for the
campaign's own `+0.05` floor. The point estimate is the middle of that band, not a commitment: a
block sized to it is powered only if the between-population variance happens to sit near its point
estimate, and the band's weak end is the number to plan against if the claim must actually be
settled.

## The primary read is now two components, not F1 alone

The out-of-fold F1 conflates two things, and the block's own receipts show them separating. For every
arm the analysis reports three parts that add exactly to the primary (tested):

- **ranking quality** - F1 at the arm's *own* best boundary (and AUC, which is boundary-free);
- **zero-boundary cost** - `F1(0.0) - F1(best)`, what a fixed `0.0` boundary costs that arm;
- **transfer cost** - `F1(out-of-fold) - F1(0.0)`, what a boundary chosen on other groups costs.

The worked example is seed `20261103` on the budget lever, in the decomposition table above: its AUC
delta is small, its F1 at its own best boundary is far below control, the `0.0` boundary penalises it
*less* than control, and the transfer cost finishes the gap. An earlier revision of
[M3_COMPUTE_REPLICATION.md](M3_COMPUTE_REPLICATION.md) read that seed as "mostly boundary transfer"
from the AUC gap alone; the decomposition refutes it, and that document now says so.

**Policy for any future block:** report both components - ranking and the two boundary terms - each
with a cluster-respecting interval and its own detectable floor, instead of a single F1 number. The
existing read stays as published; this is the declared primary for what comes next.

## The campaign conclusion

The retire-the-compute-closure decision now has a quantitative basis, and it is not the comfortable
one: the levers were judged against a floor this design cannot resolve, the audit's `+0.149` was
inside its own design's floor, and settling either at population level costs the designs in the
figure block. The honest statement is that **the campaign's lever reads were underpowered by
construction, and the compute closure is retired because it could not be resolved at the design that
was run - not because it was measured to be null.** Either fund a design of the size the band implies,
or stop ranking levers on point estimates and treat every future lever as exploratory.

## What this is not

Not a verdict change for either lever, not a threshold or arm substitution, not a new training run,
not a fit-gate reopening, and not a claim that the compute levers are null - only that this design
could not have shown them to be real, and the power to do so is now quantified and checked.
