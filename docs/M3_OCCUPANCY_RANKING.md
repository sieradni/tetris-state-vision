# M3 occupancy ranking — read-only threshold sweep (predeclaration)

Status: measured (2026-11-09). The rule below was predeclared and committed at `9ad3598`,
before any sweep admission profile, sweep receipt or sweep evidence directory
existed in the tree; the measured result is the last section and does not revise it.

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

## Measured result: the ranking is UNINFORMATIVE, and the operating-point family is exhausted

Receipts: admission `runs/m3-sweep-occupancy-r1-profile-r1-job.json` (passed, 2 forwards),
the sweep `runs/m3-sweep-occupancy-r1-ranking-r1-job.json` (passed, 27 forwards, 25.6 s,
under the declared 640 MiB ceiling), its curation
`runs/m3-sweep-occupancy-r1-ranking-r1-curation-job.json` (passed) into
`docs/evidence/m3-sweep-occupancy-r1-ranking-r1/scope.json`, and the instrument tests
`runs/m3-sweep-occupancy-tests-r4-job.json`. The sweep reloads the committed
`m3-percell-balance-r1` checkpoint (weights `9ae1fef9f649...`), changes no weight, and its
curator re-derives every number below from the receipt's own per-cell logits.

| quantity (sweep receipt) | training (deciding) | development |
|---|---|---|
| pooled AUC over known cells | **0.7491** | 0.8712 |
| chance reference | 0.5 | 0.5 |
| occupied / empty known cells on nonempty boards | 168 / 1,232 | 401 / 2,399 |
| best swept threshold | **-0.90** | -0.45 |
| best true / false positive / false negative | 106 / 337 / 62 | 335 / 492 / 66 |
| best precision / recall / F1 | 0.2393 / 0.6310 / **0.3470** | 0.4051 / 0.8354 / 0.5456 |
| exact nonempty boards at the best point | 0/7 | 0/14 |
| empty-population false positive rate at the best point | 122/400 = **0.3050** | 189/800 = 0.23625 |
| grid zero row (join with the replay) | 67 / 196 / 101 | 275 / 352 / 126 |

**The verdict is the second of the two predeclared forms: the ranking is UNINFORMATIVE.** No
swept threshold on the predeclared 81-point grid reaches the required occupied-cell F1 of
0.70 - the best of all 81 points is **0.3470** on training - and the training AUC of 0.7491
is below the predeclared informativity bar of 0.90. Both conditions fail, so the occupancy
head of this representation is **closed** and the operating-point family is recorded as
**exhausted**: no threshold of this checkpoint's locked occupancy logits, at any point of the
declared grid, approaches fit, and nobody should re-sweep it.

What the numbers do say, precisely and without upgrading the verdict: the head ranks better
than chance (0.7491 on training, 0.8712 in development), so the logits carry some ordering
signal about occupied cells. What they do not carry is enough separation to matter. Even the
best training point trades 62 remaining misses for 337 inventions and 122 false positives on
the 400 known empty training cells, and the favourite operating point of the previous run
(`0.0`, the replay's own decision) records the same 67/196/101 counts the replay published -
the reference join that makes this receipt a consistency check as well as a measurement. The
class-budget run's 0.3109 F1 sits 0.036 below the swept maximum of 0.3470, so the whole
operating-point family buys a few percent of F1 and nothing else; the ceiling of what a
threshold can do here is 0.3470, and the predeclared bar was 0.70.

Costs, all inert and all reported: exact nonempty boards are 0/7 and 0/14 at the best point,
exact empty boards are 0/2 and 0/4, development never enters the verdict, and the empty
population's false-positive rate at the best point (0.3050 training) is the price of the
boundary the sweep chose. Tuning any of these into the report is exactly what the
predeclaration forbids: the swept best point is an operating point of this checkpoint, not a
model improvement and not a new result.

`scripts/occupancy_sweep.py` holds the rule as code (`threshold_grid`, `pooled_auc`,
`evaluate`, `sweep_verdict`) and `tests/test_sweep_rule.py` reproduces it from the committed
receipt: the grid is 81 fixed points including zero, logits exactly at a threshold count as
empty, the AUC counts ties as one half, both conditions are required with boundary ties met,
counts come from the deciding population only, a missing population is refused, and the
recorded sweep re-derives as `uninformative` with AUC 0.7491 and best F1 0.3470.

What this does not license: no real-image, complete-state, calibration or quality claim, no
re-opening of the fit-gate, class-weight or scale levers, and no further occupancy threshold
work - this family is spent. The occupancy question of this representation is now closed
from both sides, the scalar weight and the class split by measurement, and the decision
boundary by this sweep. What remains open is out of scope for a threshold: a representation
that separates occupied from empty cells at this budget is a different model question.

## Widened-population update: the same verdict on a population 3.8x larger

The numbers above are this sweep's own scope and stand unchanged. The deciding population was
later widened from three groups / nine images to the canonical twelve / thirty-six and the same
81-point grid and rule re-measured on the same checkpoint
([M3_WIDENED_DECIDING_POPULATION.md](M3_WIDENED_DECIDING_POPULATION.md),
`docs/evidence/m3-widened-deciding-population-r1/scope.json`). The narrow numbers are
**superseded without invalidation**; old and new side by side: AUC 0.7491 (CI 0.7049-0.7934)
against 0.90 becomes **0.7852 (CI 0.7640-0.8064)** over 667 occupied and 4,933 empty known
cells; best swept F1 0.3470 becomes **0.3704** at the same `-0.90` threshold against the same
0.70 bar; exact nonempty boards stay 0/28. The verdict is still **UNINFORMATIVE**, and the
interval now shows the gap to the bar is not sampling noise: its upper bound is 0.094 short. The *conditioned* family is measured closed as well: giving every board its own grid point, 0 of 28 reach F1 0.70 and the oracle-pooled F1 is 0.4305 ([M3_OCCUPANCY_CONDITIONING.md](M3_OCCUPANCY_CONDITIONING.md)), so "exhausted" now holds for conditioned thresholds too. The *representation* question that this left ranked has since been measured: a declared zero-initialized read-out neighborhood, continued for 64 matched steps against a control continuation from the same checkpoint, raised pooled AUC 0.7427 -> 0.8244 and the standing-point F1 0.2848 -> 0.3600 but lowered the best swept F1 0.4338 -> 0.4200 and cleared no gate (0/28 exact boards, AUC 0.076 short, best F1 0.28 short), so it is **not cleared** ([M3_OCCUPANCY_REPRESENTATION.md](M3_OCCUPANCY_REPRESENTATION.md)). The next-ranked step is a representation that targets cell-level *precision* rather than ordering at this budget. That lever has now also been measured: a declared zero-initialized stride-2 detail tap at the cell centers, 64 matched steps against a fresh control under one catalogue, **not cleared** either — but its declared mechanism moved as intended, with precision at `0.0` 0.1752 -> 0.2363 (cluster delta 0.0104-0.0939), recall *also* up (0.7616 -> 0.9910) and standing-point F1 0.2848 -> 0.3816, while AUC (0.8226) stays 0.077 short of 0.90, the best swept F1 (0.4120) stays 0.288 short of 0.70, and 0/28 boards are exact ([M3_OCCUPANCY_PRECISION.md](M3_OCCUPANCY_PRECISION.md)). So precision is addressable from the read-out, but read-out information at this budget is not enough separation. **Next-ranked step:** carry the same demonstrated signal with more capacity (a 2x2 in-cell detail stencil per cell through the same zero-initialized projection), then encoder resolution, then scale; every threshold, conditioning and class-budget family stays closed.

**The stencil step this paragraph ranked has since been measured.** A declared 2x2 in-cell detail
stencil - four quarter-cell stride-2 samples instead of the single center sample, through the same
zero-initialized projection - continued 64 matched steps from the same published checkpoint against a
matched single-sample control under one catalogue, both inside 1,152 MiB. It is **not cleared** (0/28
exact boards; AUC 0.8804, **0.020 short** of 0.90; best swept F1 0.4913, **0.209 short** of 0.70), but
for the first time an arm **meets the attribution floor**: its swept optimum beats the single-sample
control's 0.4120 by **+0.0793** against the declared +0.05 floor, and its pooled AUC is the highest of
the campaign (cluster delta +0.0385 to +0.0943). Its declared mechanism, standing-point precision,
moved only 0.2363 -> 0.2474 - below the 0.05 floor, so `precision_improved = false`. The implication
is specific: **extra in-cell capacity carries ordering and the swept optimum, not the standing-point
precision, and separation is now the binding constraint** - the ordering deficit has fallen from
0.076/0.077 to 0.020 since the neighborhood pass while the swept-F1 deficit held at 0.21-0.29
throughout. **Next-ranked step:** encoder resolution, a finer native tap than stride-2, because every
read-out lever has converted capacity into ordering rather than separation; then scale the matched
budget on whichever carries the gain. Stacking the two read-out taps remains untested. See
[M3_OCCUPANCY_STENCIL.md](M3_OCCUPANCY_STENCIL.md) and
`docs/evidence/m3-occupancy-stencil-r1/scope.json`.

**The encoder-resolution step that paragraph ranked has since been tested and refused.** The declared
single change was a stride-1 convolution of the captured image whose cell-center sample joins the same
zero-initialized detail projection, against the same single-sample control, at the identical 64-step
budget, gates, 1,152 MiB ceiling, population and catalogue identity ([M3_OCCUPANCY_RESOLUTION.md](M3_OCCUPANCY_RESOLUTION.md),
predeclared at `eee9362`). Its 2-step admission profile breached the ceiling before any study: lever
peak **1,231,880,192 B** against 1,207,959,552 B, same-session control 1,034,313,728 B, so the branch
costs **197,566,464 B (188.4 MiB)** - 3.93x the one-map estimate the declaration carried - and the
envelope is short by 23,920,640 B (22.8 MiB). Under the declaration's own refusal rule nothing was
relaxed, substituted or re-run, so the verdict is **refused** with `precision_improved = false` because
precision was never measured ([refusal.json](evidence/m3-occupancy-resolution-r1/refusal.json)). The
result answers only one thing: a learned pixel-resolution source is not affordable inside this
envelope, so **the ranked next step is stacking `occupancy_context` with `occupancy_detail`** - the
only remaining lever that changes what the read-out sees rather than how finely it samples it, at
about 0.02 MiB of retained activation - with scale after it and resolution reopenable only as a
separately declared smaller configuration.
