# Widened deciding population — instrument change (predeclaration)

Status: predeclared (2026-11-09), committed before any widened-population admission profile,
replay, sweep receipt or evidence directory exists in the tree.

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
