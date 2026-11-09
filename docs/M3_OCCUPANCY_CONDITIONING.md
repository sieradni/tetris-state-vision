# Occupancy conditioning — does the failure live in the boards or in the boundary?

Status: predeclared (2026-11-09), committed before any conditioning receipt exists. Rule, bar and
population are fixed below; the measured section is appended only after the run.

## The question this answers

The occupancy head is closed from three sides: the weight scale and class split by measurement,
and the **decision boundary** by a sweep whose predeclared rule returned `uninformative` on the
widened population (AUC 0.7852, best global F1 0.3704 against the 0.70 bar, 0/28 exact boards).
Every published statement since then reads that as "a representation that separates occupied from
empty cells at this budget" being the open question — but that ranking is an *inference*, not a
measurement. The sweep exhausted the family of **unconditioned, global** thresholds. It did not
measure whether the failure is per-board.

Those are different mechanisms with different next levers:

- If each board's own ranking is weak, no decision rule can rescue it: the next lever is the
  representation (features), and the boundary family is closed even when conditioned.
- If boards rank well individually but need very different operating points, then the threshold
  family was closed only **unconditioned**, and the next lever is a deployable conditioned rule
  (for example by density), not a representation change.

This pass measures which one it is, from the committed widened sweep cases, with no training, no
model and no new threshold.

## Population and inputs

The deciding population is unchanged: the **training** split's 28 nonempty boards, 667 occupied
and 4,933 empty known cells, from the committed widened sweep
(`docs/evidence/m3-widened-deciding-population-r1/sweep-result.json`), whose bytes are verified
against the widened scope's `artifact_sha256` before use. Density is joined from the committed
replay rows when those rows carry a `density_recipe`, and is reported as unavailable otherwise —
never inferred.

## Predeclared rule — no new threshold

The grid is the standing 81-point grid on `[-6, 6]` step 0.15; a logit exactly at a threshold
counts empty; F1 is `2TP/(2TP+FP+FN)` on that board's own known cells. For each nonempty deciding
board:

1. **Per-board oracle F1** — the maximum over the standing grid. A board **reaches** iff that is
   `>= 0.70`, the standing F1 bar; an exact tie reaches (the same convention as
   `percell_gate_opens`).
2. **Oracle threshold** — the grid point attaining that maximum, taking the **lowest** threshold on
   ties.
3. **Oracle threshold spread** — `max - min` over the thresholds of reaching boards (0 when fewer
   than two reach).

The population classifies as:

| verdict | condition | meaning |
|---|---|---|
| `representation-limited` | fewer than **4** of 28 boards reach (the predeclared floor) | no board ranking survives conditioning; the boundary is not the binding constraint |
| `conditioned-boundary` | >= 4 reach **and** spread **> 1.5** logits | boards rank, operating points disagree; the family was closed only unconditioned |
| `inconsistent` | >= 4 reach **and** spread **<= 1.5** | boards rank and agree on an operating point, so a global threshold should have found them; that contradiction is itself the finding and is reported, not smoothed |

Additionally measured and reported (descriptive, never decisive): the per-board AUC distribution,
each board's F1 at the standing `-0.90` operating point, the **oracle pooled F1** (every board at
its own oracle threshold — an upper bound, not a deployable rule), and, when density is joined,
the median oracle threshold per density with its spread.

The bar, the grid, the population and every published gate number are unchanged. This pass does
not re-open the fit gate or the per-cell gate, and it cannot move the ranking verdict: that
verdict is defined by `occupancy_sweep.sweep_verdict` on the global rule and stays `uninformative`
whatever this decomposition shows.

## What each outcome licenses

- `representation-limited` — ranks the representation question with evidence and closes the
  conditioned-boundary family too. No new model work is licensed by this pass alone.
- `conditioned-boundary` — the ranked next step becomes a **density-conditioned decision rule**,
  which must be predeclared and measured in its own pass, with a matched unconditioned control and
  the same bars. The phrase "the operating-point family is exhausted" is then qualified in
  `M3_OCCUPANCY_RANKING.md`, the roadmap and the handoff: exhausted over *unconditioned* global
  thresholds only.
- `inconsistent` — reported as a defect of the join or of the pooled arithmetic and investigated
  before any lever is ranked; V-054 is the precedent for exactly this kind of finding.

## Receipts, resource plan and discipline

One bounded read-only pass: `scripts/record_board_conditioning.py` at **384 MiB** ceiling and
**300 s** wall, four work items (the widened scope, sweep result, replay rows, and the evidence
write), absolute interpreter path (V-030), canonical receipt
`runs/m3-occupancy-conditioning-r1-job.json`, evidence directory
`docs/evidence/m3-occupancy-conditioning-r1/scope.json`. No training, no model import, no GPU, no
artefact written outside that directory. Prior receipts, evidence bytes and the frozen manifest are
untouched, and the artifact-dependent test re-derives every published number from the committed
bytes.

## Measured result

(appended after the receipt exists)
