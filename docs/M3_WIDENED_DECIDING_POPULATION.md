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
