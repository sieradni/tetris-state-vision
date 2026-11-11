# The optimiser regime was never audited, and every committed step is clipped

Status: **analysis and instrument only.** Nothing was trained, no model, checkpoint, threshold,
label, gate or lever-table entry changed, and no earlier verdict is reopened. Every number below
is a census of committed bytes or a citation of an existing receipt.

## Why this question outranks the standing fork

What the campaign has established is not in doubt: the contracts, the bounded runner, the
21-forward replays and the fit gate all reproduce from committed artifacts, the occupancy head is
closed from every measured direction (weight scale, class split, decision boundary, read-out form,
tap stacking, encoder resolution at the envelope, a 4x budget, origin, extent and supervision
capacity), and the fit gate reads **0 of 28** exact nonempty boards at capacity 64. What is left
at the frozen architecture is a user decision priced in
[M3_TRUNK_WIDTH_PRICING.md](M3_TRUNK_WIDTH_PRICING.md).

Before that fork is treated as the only move, one assumption deserves an audit because **no lever
in the campaign ever varied it**: the optimiser schedule. Every arm was matched at
`learning_rate .001` with `clip_grad_norm_(..., 1.)`, and V-085 has already measured what that
regime does on the pilot harness - a peak, a drift, and recovery once the step size decays.

## The measurement

`scripts/stability_regime.py` reads the `gradient_norm` every trainer step already writes
(pre-clip, returned by `clip_grad_norm_`) from all `*.jsonl` curves under `docs/evidence`, and
censors nothing:

| quantity | measured |
|---|---|
| curve files / run directories | 106 / 90 |
| recorded steps | 40,796 |
| steps above the 1.0 clip | **40,796 (100.00%)** |
| median / min / max pre-clip norm | 23.89 / **1.657** / 1,785.62 |
| runs where every step is clipped | **90 of 90** |
| files containing even one unclipped step | **0** |

The lowest norm in the entire record, 1.657 (in `accumulation-arm-r1`), is still 1.66x the clip.
The longest run, `m2-stream-accuracy4096-s10`, is 4,096 steps with median 28.25 and maximum
1,785.62 - all clipped. Reproduce with:

```
.venv/Scripts/python.exe scripts/stability_regime.py docs/evidence
```

There is no learning-rate schedule anywhere in the tree: no `lr_scheduler`, no
`CosineAnnealing`, no `StepLR`, no `cosine` reference in `scripts/*.py` or
`src/tetris_vision/*.py`. The three clip sites are `scripts/train_m2_pilot.py:171`,
`scripts/train_m2_stream.py:842` and `scripts/accumulation_contract.py:81`, each
`clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)` immediately before
`optimizer.step()`.

## What that regime is

`clip_grad_norm_` does not only cap the norm, it *rescales the gradient to the cap*. When the
pre-clip norm exceeds the clip on every step, as it does in all 40,796 recorded steps here, each
update is

```
delta = lr * unit_direction
```

— constant magnitude at constant `lr`, independent of curvature, gradient scale or proximity to an
optimum. A 256-step arm at `lr 1e-3` walks 0.256 in accumulated unit displacement; the 4,096-step
baseline walks 4.096. This is exactly the regime V-085 diagnosed on the pilot: 100% clipped, the
gate peaking and then drifting (0.8044 at 1,000 steps, 0.6373 at 12,600 with 4.52-cell geometry),
and cosine decay 1e-3 -> 1e-4 reaching **0.8337** at step 1,150 and ending with 0.5983-cell
geometry against 0.933 for the undecayed arm.

## What this licenses, and what it does not

Licensed:

- **The campaign's absolute claims were measured under one schedule and no other.** Statements of
  the form "this representation does not separate occupied from empty cells at this budget" hold
  for a fixed-magnitude walk; they have never been tested under a decaying step.
- **A cheap, zero-parameter instrument is available and untried.** A schedule change adds no
  tensor, so a matched start from `9ae1fef9f649...` is exact - unlike the trunk-width fork, which
  changes every trunk tensor shape.

Not licensed:

- **No lever verdict is overturned.** Each verdict compared two arms at the same `lr`, clip,
  population, seed, catalogue and step count. A regime shared by both arms does not bias the
  pairwise attribution, and re-reading them as regime artefacts would be a claim the data does not
  support.
- **Decay is not predicted to pass the fit gate.** The pilot's decayed arm still produced **0
  exact nonempty boards** at 1,150 steps while scoring 0.8337; its pool and gate population are
  not the campaign's 28 boards, so it says nothing about them either way.
- **The 100% saturation is not shown to be harmful.** It is shown to be uniform, unvaried and
  unexamined. Harm is what the arm below would measure.

## Ranked next steps

1. **A declared schedule lever on the published architecture - no user decision needed.**
   Two arms from `9ae1fef9f649...`, 256 matched steps each, identical population, seed, catalogue,
   objective and thresholds, differing only in the schedule: baseline at the standing constant
   `1e-3`, lever cosine `1e-3 -> 1e-4`. Zero added parameters, so the matched-start contract is
   satisfied by loading the same bytes; envelope 1,152 MiB / 1,800 s, which the last three arms
   measured at 701.7-1,522.4 s. Read by the standing criteria, not new ones: the fit gate
   (**3 of 28** exact nonempty boards at threshold `0.0`), the +0.05 per-cell F1 attribution floor,
   AUC, and placement non-regression. Predeclare the rule and its falsifiers before launch, as
   every other lever did.
2. **The trunk-width fork - user decision, already priced.** Wide A (`[64,128,224,256]`/192,
   8,844,777 parameters) fits the settled 1,152 MiB profile at 1.7082 s per model step, 346 steps
   inside the 1,800 s wall; wide B (`[96,192,256,256]`/240) does not fit and yields 228 steps.
   Authorizing it means a new architecture identity, a fresh baseline and its own resource
   profile, and its numbers are not comparable to the capacity-64 table.
3. **Corpus and real-image admission.** Parked pending explicit authorization; unchanged by this
   pass.

The order matters: step 1 costs one bounded job pair at the existing envelope and can retire the
last untested non-structural suspect, whereas step 2 is a new line of evidence that the user has
to authorize and that would be measured under the same unaudited regime unless step 1 runs first.

## The instrument

`scripts/stability_regime.py` (stdlib only, no Torch, no GPU, no thread policy) with
`tests/test_stability_regime.py`. It is a **standing instrument for the detection class V-085
opened** - "a training regime read from endpoints rather than from its own per-step record" - which
already has a second instance here (90 runs admitted without ever asking whether the clip released).
Per the repository's three-instance rule it is added before a third occurrence, and it censuses
committed bytes rather than trusting a summary document.

Limits unchanged: synthetic training fit, one seed, committed lattice, no real-image, generalization
or complete-state claim, and no change to the default configuration or the published checkpoint.

## Measured outcome (2026-11-11): the schedule lever is spent, and it is a null

The first ranked step above has now been run end to end
([M3_OPTIMISER_SCHEDULE.md](M3_OPTIMISER_SCHEDULE.md), `docs/evidence/m3-optimiser-schedule-r1/scope.json`,
curation receipt `runs/m3-occsched-curation-r1-job.json`). Two 256-step arms from `9ae1fef9f649...`
differing only in the schedule (constant 1e-3 against cosine 1e-3 -> 1e-4, zero added parameters)
completed inside 1,152 MiB / 1,800 s with matched start proved from the records.

**Verdict `null`:** per-cell F1 at the standing `0.0` point rose +0.005041 against a **+0.05**
floor, placement improved 39.9%, pooled AUC 0.8457 -> 0.8864, and the
fit gate stayed **0 of 28** in both arms. So the audited regime is **not** the missing occupancy
separation at this budget: the schedule shortens the walk and reorders logits, and the standing point
does not move. Step 1 is spent; **step 2 (trunk width) remains the user's decision** and step 3 stays
parked.
