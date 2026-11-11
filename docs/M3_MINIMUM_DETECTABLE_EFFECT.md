# What this design can resolve: power and minimum detectable effect

Status: **derived after the replicate read, from committed receipts only. It changes no verdict, no
threshold and no arm, and it adds no training.** It answers the question the campaign had been
answering by judgement: with the design it actually ran, what is the smallest effect it could
resolve, and were the observed deltas even above that floor?

Evidence: `docs/evidence/m3-power-mde-r2/result.json` (job `runs/m3-power-r2-job.json`). An earlier
pass, `-r1`, lacked the audit comparison and the floor-cost table and is preserved unedited; `-r2`
supersedes it. Instrument: `scripts/power_analysis.py`, pinned by `tests/test_power_analysis.py`.

## It reproduces the published read before it is trusted

The analysis rebuilds the same 36 `(seed, group)` clusters from the nine committed sweeps and refuses
to run unless it reproduces `docs/evidence/m3-compute-replication-r1/result.json`: pooled F1 per arm
to `1e-6`, both paired point deltas to `1e-6`, and both bootstrap intervals **exactly** (same draws,
same seed, so the interval dicts are compared whole). It then recomputes the paired standard error a
second, independent way - a delta-method linearisation of pooled F1, whose per-cluster influences sum
to zero by construction - and the two routes agree closely:

| lever | bootstrap SE | delta-method SE | ratio | MDE at 80% power |
|---|---|---|---|---|
| `budget192` | 0.029549 | 0.029679 | 1.0044 | **0.083151** |
| `cosine128` | 0.046307 | 0.045144 | 0.9749 | **0.126475** |

The width is therefore not an artifact of resampling: a resampling-free linearisation of the same
pooled ratio lands on the same number. `MDE = (z_0.975 + z_0.80) * SE = 2.8016 * SE`, two-sided
alpha 0.05, power 0.80.

## Both observed deltas were inside the floor

| lever | observed | magnitude | MDE | inside the floor? | power at a true +0.05 |
|---|---|---|---|---|---|
| 1.5x fit budget | -0.0796 | 0.0796 | 0.0832 | **yes** | **0.392** |
| cosine schedule | +0.0213 | 0.0213 | 0.1265 | **yes** | **0.198** |

Stated plainly: the block could not resolve an effect of its own declared `+0.05` floor. At a true
floor-sized effect it had 39% power on the budget lever and 20% on the schedule. So its `unconfirmed`
verdicts mean "no effect resolvable at this design", not "no effect" - and the rule that produced
them was followed exactly; nothing here re-reads them.

**Design effect of the clusters.** Comparing the cluster-level standard error with the standard error
the same read would show if boards, then cells, were independent: **9.5** (boards) and **16.8**
(cells) on the budget lever, **12.3** and **29.2** on the schedule. The 36 clusters are the real
sample size; the 7,200 cells are worth roughly 1/17 to 1/29 of what their count suggests. That is the
same clustering the campaign's deciding-power note flagged at the cell level, now measured on the
paired delta rather than on one arm's level.

## Was the audit's +0.149 resolvable at its own design?

No. The committed audit's fit-budget pair (`docs/evidence/m3-operating-point-r2/result.json`) reports
`d_cv_interval.standard_error = 0.070036` over **12 groups, one population**. Its own detectable floor
is therefore `0.196212`, and the power it had against its own claimed `+0.1494` was **0.567**. A
single-trajectory, twelve-cluster design could not have resolved `+0.149` even if the effect were
exactly that size - which is also why its interval's lower bound sat at `+0.006`.

That is the precise sense in which the audit's reversal never became a lever: at its own design the
claim was inside the noise floor, so its `refusal_reversed` state was a statement about a point
estimate, not a resolvable effect.

## Would a three-seed 256/384 block settle it?

Two inferences, both computed from these receipts, and they disagree - which is the finding.

**Fixed-population (the three seeds are the universe).** Within-seed standard errors are
`0.0203 / 0.0331 / 0.0642` on the budget lever and `0.0609 / 0.0423 / 0.0516` on the schedule,
so three seeds would give `SE = 0.0250` and `0.0301`, an MDE of `0.070` and `0.084`, and `+0.149`
would be resolved. This inference is wrong for a stopping decision, because the claim is about the
method, not about these three populations.

**Population-random (the decision-relevant one).** The three per-seed deltas differ far more than
their own sampling error explains: the between-population variance, corrected for that error, is
`0.01186` (budget) and `0.00697` (cosine). With three seeds the standard error is then `0.0677` and
`0.0568`, so the minimum detectable effect is **`0.190` and `0.159` - both above `+0.149`**, with
power `0.596` and `0.746`. **A three-seed 256/384 block is underpowered for the audit's own claim.**

**Counts and cost** (measured cost: 2.24375 s per step, the median of the nine committed arms;
256+384 = 640 steps per seed):

| target | seeds | clusters | arm-hours | read |
|---|---|---|---|---|
| `+0.149`, budget variance | 5 | 60 | 1.99 | clears by 0.002 only |
| `+0.149`, schedule variance | 4 | 48 | 1.60 | clears by ~0.01 |
| `+0.149`, recommended margin | 8 | 96 | 3.19 | `SE 0.0414`, MDE `0.1161` |
| the campaign's own `+0.05` floor | 44 | 528 | 17.55 | `SE 0.0177`, MDE `0.0495` |

So the answer to "would three seeds settle `+0.149`" is **no**; five seeds clear it on the point
estimate and eight give real margin. The answer to "can this campaign resolve the floor it judges
every lever against" is **44 seeds, 528 clusters, about 17.6 arm-hours of training** - and that is a
scope decision for the user, not a licence to launch it.

**Limitations, stated rather than buried.** The between-population variance rests on three population
draws (2 df); its 90% upper limit is about 9.5x the point estimate, which would push the required seed
count for `+0.149` as high as ~41. The 5- and 8-seed numbers are point-estimate answers, not
commitments. The per-step cost is measured at 128/192 steps and is used as a proxy for 256/384; a
longer budget may cost more per step. And the 256/384 contrast may carry a different cluster variance
than 128/192, which only running it would reveal.

## The primary read is now two components, not F1 alone

The out-of-fold F1 conflates two things, and the block's own receipts show them separating. For every
arm the analysis reports three parts that add exactly to the primary (tested):

- **ranking quality** - F1 at the arm's *own* best boundary (and AUC, which is boundary-free);
- **zero-boundary cost** - `F1(0.0) - F1(best)`, what a fixed `0.0` boundary costs that arm;
- **transfer cost** - `F1(out-of-fold) - F1(0.0)`, what a boundary chosen on other groups costs.

The worked example is the one that overturned an earlier reading of this block:

| seed `20261103` (budget minus control) | value |
|---|---|
| delta AUC | -0.030 |
| delta F1 at the arm's own best boundary | **-0.169** |
| delta cost of the `0.0` boundary | +0.062 |
| delta transfer cost | -0.150 |
| delta out-of-fold F1 (the primary) | -0.257 |

An AUC-only comparison suggests "mostly a boundary effect". The decomposition refutes it: at that
seed the budget arm's F1 at its own best boundary is `0.169` *below* control, so its ranking's
operating-point quality collapsed even though its AUC barely moved, the `0.0` boundary actually
penalised it *less* than control, and the out-of-fold transfer cost finished the job. For the
schedule the same split is `+0.137 / -0.123 / +0.049`, `+0.061 / +0.004 / +0.040` and
`-0.097 / -0.079 / +0.055`: seed 1 gains at its best boundary and pays it back at `0.0`.

**Policy for any future block:** report both components - ranking and the two boundary terms - each
with a cluster-respecting interval and its own detectable floor, instead of a single F1 number. The
existing read stays as published; this is the declared primary for what comes next.

## The campaign conclusion

The retire-the-compute-closure decision now has a quantitative basis, and it is not the comfortable
one:

1. The two levers were judged against a `+0.05` floor that this three-seed, 36-cluster design
   **cannot resolve** (MDE 0.083 and 0.126; power at the floor 0.39 and 0.20).
2. The audit's `+0.149` was judged against its own design's MDE of `0.196`, so the reversal that
   motivated the whole re-read was **underpowered by construction**.
3. Resolving the floor at population level would need about **44 seeds (528 clusters, ~17.6
   arm-hours)**; resolving the audit's `+0.149` needs about **8 seeds (~3.2 arm-hours)**.

So the honest stopping decision is: **the campaign's lever reads were underpowered by construction,
and the compute closure is retired because it could not be resolved at the design that was run - not
because it was measured to be null.** Either fund a design of the size in the table above, or stop
ranking levers on point estimates and treat every future lever as exploratory. Both are defensible;
continuing to read point deltas against a floor the design cannot reach is not.

## What this is not

Not a verdict change for either lever, not a threshold or arm substitution, not a new training run,
not a fit-gate reopening, and not a claim that the compute levers are null - only that this design
could not have shown them to be real, and the power to do so is now quantified.
