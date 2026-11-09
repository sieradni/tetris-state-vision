# Widened deciding population — instrument change (predeclaration)

Status: measured (2026-11-09). The rule below was predeclared and committed at `111aa9e`, before
any widened-population admission profile, replay, sweep receipt or evidence directory existed in
the tree; the measured result is the last section and does not revise it.

## The question

Every M3 verdict since the first joint fit is decided on the **training** probe population,
and that population is nine images from three ancestry groups: `scripts/train_m2_stream.py`
evaluates `("training", 3)` at both ends of a run while `--development-groups` already admits
twelve. The deciding populations are therefore the smaller, noisier ones — seven nonempty
boards and 168 occupied known cells decide the fit gate, the per-cell F1 and the occupancy
ranking, against 401 occupied cells in development.

This pass changes the instrument, not the question. It widens the training deciding
population to the canonical twelve groups (thirty-six images) and re-derives the two standing
verdicts — the occupancy ranking (`UNINFORMATIVE`) and the identity discrimination (constant
predictor, disjunction closed) — from the same committed checkpoints, with a 95% confidence
interval printed beside every gate number.

## What changes and what does not

| element | before | after |
|---|---|---|
| deciding population (training) | 3 groups / 9 images | **12 groups / 36 images** |
| fit-gate threshold | >= 3 exact nonempty training boards | **unchanged** |
| per-cell F1 bar | 0.70 | **unchanged** |
| ranking AUC bar (chance .5) | 0.90 | **unchanged** |
| swept threshold grid | 81 points on `[-6, 6]`, step 0.15 | **unchanged** |
| sweep decision convention | `logit > threshold`, a tie counts empty | **unchanged** |
| identity class bar | >= 3 distinct classes per head | **unchanged** |
| development population | 6 groups / 18 images | **unchanged** |
| model, loss, optimizer, seeds, checkpoints, frozen archives | frozen | **unchanged** |

Explicitly **not** this pass: no model, loss, optimizer, seed, capture or data change; no new
seed policy; no representation lever; no real-image work; no threshold re-sweep of the closed
occupancy family (same 81-point grid, same rule, same zero-row join, only the population it is
measured on changes); no re-opening of the occupancy weight, class-budget, identity read-out,
endpoint or queue-length families; and no edit to any frozen evidence byte or manifest entry.

## Why widening is not new data

`m2_curriculum.training_spec` maps a training step to `(group, view)`, so a run's training
probe population is a prefix of its own training stream. Group indices 3..11 under the run's
recorded base seed are that stream's steps 10..36: the drawings the model already trained on,
resampled deterministically by the run's frozen sampler. The widened population therefore adds
no appearance family, no seed and no annotation. Like the recorded nine, these are
**training-fit** drawings, so the gates remain memorization instruments and nothing here
becomes a generalization, complete-state, calibrated or real-image estimate.

Validation is structural: the recorded ledger must contain exactly groups 0..2 with views
0..2; the extension must reach exactly the canonical twelve groups; and the recorded nine-case
subset of the widened replay must reproduce the earlier published counts exactly, which is the
join that makes this supersession rather than invalidation.

## Predeclared rule — reuse, never restate

There is no new acceptance rule and no new threshold. The pass re-derives, on the widened
population, the rules already in code and already published:

1. **Fit gate** — `scripts/fit_gate.py`: opens iff the independently replayed run reports
   >= 3 exact nonempty training boards.
2. **Per-cell decision** — `scripts/percell_gate.py`: opens iff occupied-cell F1 on the
   deciding nonempty population is >= 0.70; `decision_mechanism` names the pinning error mass.
3. **Occupancy ranking** — `scripts/occupancy_sweep.py`: `informative` iff the deciding AUC
   is >= 0.90 and some swept threshold reaches F1 >= 0.70; otherwise `uninformative`, and the
   operating-point family stays exhausted.
4. **Identity discrimination** — `scripts/identity_gate.py`: `opens` iff the queue head and
   the hold head each emit >= 3 distinct predicted classes and neither is a constant.

A verdict that flips is the finding and is written up as such; a verdict that tightens is also
the finding and is written up as such. Neither is smoothed over, and neither re-opens a closed
family by itself.

## Confidence intervals — reported, never decisive

`scripts/gate_power.py` supplies the intervals, and no rule consults it:

- **AUC**: Hanley-McNeil standard error, 95% interval, beside the informativity bar.
- **Precision and recall** (the two proportions behind F1): Wilson score intervals, beside the
  F1 point and its signed gap to the 0.70 bar, at both the sweep's best operating point and
  the replay's own `0.0` decision.
- **Identity exact rates**: Wilson interval on the exact piece-token rate.
- **Count-based gates** (exact boards, distinct classes) are a census of the population, not a
  sample: they are reported as counts with their denominators and carry no interval.

## Populations and receipts this pass must produce

Deciding split: **training**, twelve groups, thirty-six images. Development stays at its
recorded six groups / eighteen images as a diagnostic.

1. Sweep admission profile (2 forwards), bounded at 640 MiB / 300 s / 2 work items, receipt
   `runs/m3-sweep-widened-occupancy-r1-profile-r1-job.json`.
2. Occupancy replay of the committed `m3-percell-balance-r1` checkpoint, 54 forwards (36
   training + 18 development), bounded at 640 MiB / 900 s / 54 work items, receipt
   `runs/m3-percell-widened-r1-job.json`, result `runs/m3-percell-widened-r1/result.json`.
3. Identity replay of the committed `m3-identity-factor-length-r1` checkpoint, the same 54
   forwards and bounds, receipt `runs/m3-identity-factor-widened-r1-job.json`.
4. Widened occupancy sweep, 54 forwards, bounded at 640 MiB / 900 s / 54 work items, receipt
   `runs/m3-sweep-widened-occupancy-r1-job.json`, admitted only by receipt 1.
5. Curation at 128 MiB / 300 s into one new atomic evidence directory,
   `docs/evidence/m3-widened-deciding-population-r1/scope.json`, which copies its receipts and
   re-derives every number from their bytes.

## Resource plan and discipline

Both checkpoint replays of twenty-seven forwards peaked between 560 and 569 MB, and a forward
cost is unchanged: the added cases are twenty-seven more single-image forwards per replay, so
the expected peak is unchanged and the declared ceiling stays 640 MiB under the engine's 2 GiB
bounded-job cap. One heavy job at a time, absolute interpreter paths (V-030), canonical
receipts (V-047), preserved attempts (V-050), objective identity bound by `arguments` (V-051),
declarations held once (V-052, and `probe_population` here), and curator sidecars read from a
declaration (V-053). Every prior receipt, evidence byte and manifest entry is preserved
untouched; the earlier narrow-population numbers are marked **superseded without
invalidation**, old and new side by side.

## Measured result: both verdicts re-derive unchanged, and the intervals close the question

Receipts: admission `runs/m3-sweep-widened-occupancy-r1-profile-r1-job.json` (passed, 2
forwards, 6.3 s, 545,607,680 B), the occupancy replay `runs/m3-percell-widened-r1-job.json`
(passed, 54 forwards, 45.0 s, 559,869,952 B, weights `9ae1fef9...` unchanged and every saved
metric reproduced), the identity replay `runs/m3-identity-factor-widened-r1-job.json` (passed,
54 forwards, 45.7 s, 555,425,792 B, weights `8042c288...` unchanged), the widened sweep
`runs/m3-sweep-widened-occupancy-r1-job.json` (passed, 54 forwards, 44.3 s, 585,424,896 B), and
curation `runs/m3-widened-population-curation-r2-job.json` (passed) into
`docs/evidence/m3-widened-deciding-population-r1/scope.json`.

| population (deciding split) | narrow (superseded) | widened |
|---|---|---|
| ancestry groups / images | 3 / 9 | **12 / 36** |
| nonempty boards | 7 | **28** |
| occupied known cells | 168 | **667** |
| empty known cells on nonempty boards | 1,232 | **4,933** |
| empty boards | 2 | 8 |
| development images (unchanged) | 18 | 18 |

**The occupancy verdict is unchanged and is no longer a small-population artifact.**

| gate number | narrow (superseded) | widened, with 95% interval | verdict |
|---|---|---|---|
| ranking AUC (bar 0.90) | 0.7491, CI 0.7049-0.7934 | **0.7852, CI 0.7640-0.8064** (SE 0.0108) | UNINFORMATIVE |
| best swept F1 (bar 0.70) | 0.3470 | **0.3704** at the same `-0.90`; precision 0.2522 [0.2329, 0.2725], recall 0.6972 [0.6612, 0.7308] | UNINFORMATIVE |
| operating-point F1 at `0.0` (bar 0.70) | 0.3109 | **0.3455**, gap 0.3545; precision 0.2810 [0.2548, 0.3088], recall 0.4483 [0.4109, 0.4862] | STANDS |
| exact nonempty training boards (bar 3) | 0/7 | **0/28** | FALSE |
| queue distinct classes (bar 3) | `EOS`x54, 1 | **`EOS`x216, 1** | STANDS |
| hold distinct classes (bar 3) | `J`x9, 1 | **`J`x33, 1** | STANDS |

The widened population is 3.8x larger in occupied cells (3.97x in nonempty boards) and the
point estimates move *toward* the bars — AUC 0.7491 -> 0.7852, best F1 0.3470 -> 0.3704,
operating-point F1 0.3109 -> 0.3455 — while the verdicts do not move at all:

- The AUC interval narrowed from +-0.044 to **+-0.021**, and its upper bound (0.8064) is still
  **0.094 below the 0.90 bar**. The ranking is better than chance and decisively below the
  informativity bar; that is now a property of 5,600 known cells, not of 1,400.
- The best swept point recovers 465 of 667 occupied cells and invents 1,379. At that fixed
  recall, reaching F1 0.70 would require **at most 196** false positives — a 7.0x precision gain
  that no threshold on this checkpoint's logits can supply, which is what "the operating-point
  family is exhausted" meant.
- The per-cell gate still stands, still `specificity-limited` (inventions 765 exceed misses
  368), and the operating-point precision interval [0.2548, 0.3088] excludes any value near the
  0.70 F1 the bar needs at that recall.
- The identity heads emit one class each — queue `EOS` at all **216** supervised training
  positions and hold `J` at all 33 cases — so the constant-predictor finding reproduces at four
  times the supervised positions. Hold exactness (15/33, CI [0.2984, 0.6201]) is exactly `J`'s
  target frequency, the signature V-048 described; the queue is 0/216 (CI [0.0, 0.0175]).

Both directions are reported as required: the widened population is *easier* per cell (AUC up)
and the fit gate is *harder* at the same threshold (3 of 28 rather than 3 of 7). Neither
movement changes a verdict, so nothing in the earlier record needs revisiting: every earlier
verdict stands, with its narrow-population numbers marked **superseded without invalidation**.

The join is exact, not asserted: the recorded nine-case subset of each widened replay reproduces
the earlier published bytes — 67 true positive / 196 false positive / 101 missed occupied cells,
0 exact nonempty boards, empty-population false positives 72, and the queue `EOS`x54 / hold `J`x9
identity classes. `scripts/probe_population.py` holds the population, `scripts/gate_power.py`
the intervals (which no verdict reads), `scripts/identity_gate.py` the discrimination rule, and
`scripts/record_widened_gates.py` the curator; `tests/test_widened_gates.py` re-derives both
verdicts and every interval from the committed evidence, including the artifact hashes.

What this does not license: no representation, model, loss or data claim; no re-opening of any
closed family; no real-image, complete-state, calibration or quality claim. The one open model
question is unchanged — a representation that separates occupied from empty cells at this
budget — and this pass deliberately does not rank or start it.

## Methodological addendum: the intervals were cell-count intervals, and the cells are clustered

Status: measured (2026-11-09) from the committed widened artifacts; recorded in
`docs/evidence/m3-widened-deciding-population-clustering-r1/scope.json`. This is an addendum to
the predeclaration above, not a new predeclaration, sweep or lever: no gate, threshold, point
estimate, model, loss, seed or prior evidence byte changes, and the widening is not re-litigated.

### The defect this measures

Every interval in the section above (and in the state docs) is a cell-count interval: Hanley-
McNeil for the AUC and Wilson for precision, recall and identity exactness. Both formulas treat
all 5,600 known cells — 667 occupied across 28 nonempty boards inside 12 ancestry groups, plus
4,933 empty — as independent observations. They are not: each board contributes 200 cells, and
three views share an ancestry. The repo's own scope lines already said "matched views share
ancestry; no independent-sample interval", so the published widths were an untested assumption.

### Measured clustering (per-cell error indicator, one-way ICC(1))

| operating point | level | clusters | mean cells | ICC | design effect | effective cells | effective occupied |
|---|---|---|---|---|---|---|---|
| `0.0` | board | 28 | 200.0 | 0.000941 | **1.187** | 4,716.5 | 561.8 |
| `0.0` | ancestry group | 12 | 466.7 | 0.004024 | **2.874** | 1,948.8 | 232.1 |
| `-0.90` | board | 28 | 200.0 | 0.005906 | **2.175** | 2,574.4 | 306.6 |
| `-0.90` | ancestry group | 12 | 466.7 | 0.009233 | **5.300** | 1,056.7 | 125.9 |

The pooled per-cell error indicator is **only weakly clustered**: design effects of 1.19-2.87 at
the operating point and 2.18-5.30 at the swept best, so the instrument has roughly **232-562
effective occupied cells instead of 667**, and 1,057-4,716 effective cells instead of 5,600.
That is the honest effective sample size: between one third and one fifth of the nominal
occupied count, and between one sixth and one tenth of the nominal cell count at the swept best.

### The clustering is not in the error rate — it is in precision

The pooled ICC mixes two regimes: 88% of known cells are empty, and empty-cell behaviour is
stable. Conditioning on each class separates them, and the asymmetry is the finding:

| quantity | cell-level width | board-cluster width | ancestry-cluster width | design effect (group) |
|---|---|---|---|---|
| precision at `0.0` | 0.0539 | 0.2053 | 0.3630 | **45.85** |
| recall at `0.0` | 0.0753 | 0.0385 | 0.0619 | **0.99** |
| precision at `-0.90` | 0.0396 | 0.1905 | 0.3434 | **76.02** |
| recall at `-0.90` | 0.0696 | 0.0578 | 0.0927 | **1.99** |

Recall is essentially binomial (design effect 0.99-1.99, sometimes *narrower* than the cell
formula). Precision is 14-76x more variable than the cell formula assumes, because how many of a
board's predicted cells are wrong is a board-level property, not a cell-level one. The published
precision interval was therefore the one overstated claim in the record: its true width is 3.4x
to 6.7x larger than printed. The F1 point and the specificity-limited mechanism are unaffected,
because both are decided by the point estimate.

### Corrected intervals beside the published ones

Whole-cluster percentile bootstrap (seed 20261026, 400 draws for the AUC, 1,000 for counts and
rates, degenerate draws skipped), ancestry group as the primary sampling unit because matched
views share ancestry, board as the finer partition.

| gate number | point | cell-level 95% (published) | board-cluster 95% | ancestry-cluster 95% |
|---|---|---|---|---|
| ranking AUC (bar 0.90) | 0.785186 | 0.763959-0.806412 | 0.765266-0.803474 | **0.747438-0.818608** |
| precision at `0.0` | 0.281015 | 0.254831-0.308774 | 0.175435-0.380699 | **0.090827-0.453809** |
| recall at `0.0` | 0.448276 | 0.410938-0.486206 | 0.427775-0.466249 | **0.407616-0.469513** |
| F1 at `0.0` (gap 0.3545) | 0.345465 | — (F1 is scored as a point) | 0.251614-0.414611 | **0.148758-0.453838** |
| precision at `-0.90` | 0.252169 | 0.232878-0.272491 | 0.156852-0.347351 | **0.080246-0.423657** |
| recall at `-0.90` | 0.697151 | 0.661233-0.730812 | 0.668616-0.726374 | **0.647292-0.740023** |
| F1 at `-0.90` (gap 0.3296) | 0.370370 | — | 0.255580-0.465002 | **0.143740-0.529574** |
| queue exactness | 0/216 | 0.0-0.017474 | — | **0.0-0.017474** (design effect 1.0) |
| hold exactness | 15/33 | 0.298429-0.620141 | — | **0.181818-0.727273** (bootstrap), 0.204541-0.729780 (Wilson at design effect 3.3) |

The F1 rows are supplementary empirical intervals: the published rule still scores F1 as a point
and the predeclaration's "no interval is invented for F1" stands; the bootstrap is reported
because it resamples the boards rather than assuming a distribution for the ratio.

### Does any verdict change?

**No. All three decided verdicts are confirmed, none is flipped, and none is weakened against its
bar.** The corrected intervals are wider, and in every case the bar is still excluded at 95%:

- **Occupancy ranking stays UNINFORMATIVE.** The ancestry-cluster interval is 1.68x wider than the
  cell interval, and its upper bound **0.8186** is still **0.081 below the 0.90 bar** (board level
  0.8035, 0.096 below). The board-level resampling gives a design effect of 0.85, i.e. the cell
  formula was *not* optimistic for the AUC — it was mildly conservative. The verdict does not rest
  on the earlier narrow interval: it rests on the point estimate (0.7852) against the bar, and no
  threshold on the 81-point grid reaches F1 0.70 either (best 0.3704, empirical F1 upper 0.5296,
  0.170 below the bar).
- **Per-cell gate still STANDS**, and its mechanism is *strengthened*: the corrected precision
  interval (upper bound 0.454 at `0.0`, 0.424 at `-0.90`) cannot support the precision F1 0.70
  needs at this recall, so `specificity-limited` is now a claim about the sampling distribution and
  not only about the point.
- **Identity still STANDS.** The queue is a 0/216 all-failure census: a proportion at 0 has no
  estimable between-board variance, so the measured design effect is exactly 1.0 and clustering
  *cannot* widen that interval — the finding rests on the count, and it is unchanged. The hold
  design effect is 3.3 over 11 ancestry groups with supervised hold, so its interval widens from
  0.298-0.620 to 0.182-0.727 (bootstrap) / 0.205-0.730 (Wilson at reduced n); the point 15/33 is
  still exactly `J`'s target frequency, which is an identity of frequencies, not an interval claim.
- **Fit gate unchanged**: 0/28 is a census of boards, carries no interval, and no interval method
  applies to a count gate.

### Does "superseded without invalidation" still hold?

Yes, and the like-for-like recomputation supports it. Re-running the same *method* on the narrow
sweep's own committed cases (7 nonempty boards over 3 ancestry groups) gives point 0.749116,
cell-level 0.704851-0.793380 and board-cluster 0.703924-0.779389: the narrow interval was already
essentially honest at board level, so the narrow numbers are superseded because the *population* is
larger and less noisy, not because their intervals were wrong. Only the precision intervals change
materially under correction, and they change in both the narrow and widened records, which is a
method correction rather than a scope boundary.

The single claim in the record that needs qualifying is therefore narrow and now stated: **the
cell-level precision intervals were over-confident by 3.4-6.7x**. No verdict, threshold, point
estimate, gate, closed family, model, loss, seed or frozen evidence byte moves, and
`docs/frozen-evidence-manifest.json` is untouched.
