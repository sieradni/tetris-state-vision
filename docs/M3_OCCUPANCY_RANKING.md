# M3 occupancy ranking — read-only threshold sweep (predeclaration)

Status: predeclared. Committed before any sweep admission profile, sweep receipt or sweep
evidence directory existed in the tree.

## The question

The per-cell class budget (`docs/M3_PERCELL_DECISION.md`) moved the locked decision's errors
without opening fit: recall rose while precision fell, F1 rose only 0.2605 -> 0.3109, and
empty-population false positives tripled. That leaves exactly one question about the locked
occupancy head that counts and labels can settle: **do its logits rank occupied cells above
empty ones at all, or is only the operating point wrong?**

This pass answers it with a read-only instrument. No training, no weight change, no new
model: the committed `m3-percell-balance-r1` checkpoint is reloaded, one forward recomputes
the occupancy logits for the same nine training and eighteen development saved probe boards,
and a fixed threshold grid is swept over the binary decision.

## The instrument

`scripts/sweep_stream_occupancy.py` (with the rule and metrics in
`scripts/occupancy_sweep.py`), executed against the checkpoint's own frozen source archive.

- **Grid**: the whole fixed grid `-6.0 .. 6.0` in steps of `0.15`, 81 thresholds, declared
  here and in code (`threshold_grid()`), never read from the run's current operating point.
- **Decision convention**: a known cell is predicted occupied when `logit > threshold`, the
  same strict comparison the replay's ordinary metrics use. A logit exactly at a threshold
  counts as empty. The grid therefore contains the replay's operating point at `0.0`, and
  the receipt's zero row must reproduce the replay's locked counts (the `reference_join` the
  curator checks); this is a consistency join, not the sweep's result.
- **AUC**: `P(logit of an occupied known cell > logit of an empty known cell)` pooled over
  the known cells of the seven nonempty training boards, ties counted one half
  (Mann-Whitney). Chance is `0.5`.
- **Curve**: for every grid threshold, the known-cell true/false positives and negatives, the
  derived precision/recall/F1, and the exact-board counts of the nonempty and empty
  populations. The **best operating point** is the grid threshold with the highest F1; ties
  go to the higher precision, then to the lower threshold.

## Predeclared rule — one split, two conditions, both required

The deciding split is **training**; development is a diagnostic that cannot change the
verdict.

**The ranking is INFORMATIVE iff both hold:**

1. the training nonempty pooled AUC is **>= 0.90** (chance `0.5`), and
2. **some** swept threshold reaches occupied-cell F1 **>= 0.70** on the seven nonempty
   training boards (the bar the per-cell lever had to meet).

**Otherwise the ranking is UNINFORMATIVE**, and in that case the occupancy head of this
representation is **closed** and the operating-point family is recorded as **exhausted** so it
is never re-swept.

A value exactly at either boundary counts as met (`>=`). Development AUC, its curve, the
empty-population false-positive rate at the best point, exact-board counts, per-board detail
and every other side result are **costs**: reported in full, inert for the verdict. Tuning the
threshold into the report as if it were a new result is explicitly not a finding: the best
point is named as an operating point of this checkpoint, nothing more.

If the ranking is informative, the precision/recall budget that the best point implies is an
**owner decision** - this document names that choice and does not pick one.

What the receipt cannot license: no real-image, complete-state, calibration or quality claim,
no re-opening of the closed fit-gate, class-weight or scale levers, and no claim that a swept
operating point is a model improvement.

## Receipts this pass must produce

1. Admission profile: two forwards, bounded at 640 MiB / 300 s / 2 work items, receipt
   `runs/m3-sweep-occupancy-r1-profile-r1-job.json`, summary
   `runs/m3-sweep-occupancy-r1-profile-r1/summary.json`.
2. The measurement, admitted only by that profile, 27 forwards, bounded at 640 MiB / 600 s /
   27 work items, receipt `runs/m3-sweep-occupancy-r1-ranking-r1-job.json` and logs
   `runs/m3-sweep-occupancy-r1-ranking-r1.log` / `.stderr.log`.
3. Curation at 128 MiB / 300 s, receipt `runs/m3-sweep-occupancy-r1-ranking-r1-curation-job.json`,
   into `docs/evidence/m3-sweep-occupancy-r1-ranking-r1/scope.json`.
4. Instrument tests, canonical receipt `runs/m3-sweep-occupancy-tests-r1.json` (re-run under a
   fresh number if a cap kills it).

The sweep needs its own owned name family, so `m3-sweep-*` is declared in
`record_stream_run.OWNED_RUN_PATTERN` and bound by a test.

## Resource plan and discipline

The read-only replay of this same checkpoint peaked 568,070,144 B; this sweep performs the
same 27 forwards and adds floating-point bookkeeping over at most 5,400 stored logits, so the
expected peak is unchanged. Declared ceiling 640 MiB under the engine's 2 GiB bounded-job cap.
One heavy job at a time, absolute interpreter paths, canonical receipts and sidecar names,
killed stages preserved and re-run to a passing canonical receipt, and any defect this pass
exposes fixed at its root with a regression test and recorded in `docs/BUG_LOG.md` - only if
something genuinely breaks.

## Measured result

(Measured below this predeclaration, in a later commit; the rule above is not revised by the
result.)
