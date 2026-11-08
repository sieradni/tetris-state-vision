# M3 fit gate — locked-occupancy scale lever (predeclaration)

Status: predeclared. Committed before any fit-gate admission profile, study or probe; every
receipt named below is created afterwards.

## The question

The identity disjunction is closed (`docs/M3_IDENTITY_QUEUE_LENGTH.md`): class-side
conditioning, position-side conditioning, endpoint factorization and one queue-length
decision were each measured at the fixed matched budget, each returned a NEGATIVE, and no
variant of that disjunction is ranked. What the last run named as the ranked next lever is
the **fit gate**: every M3 stream run reports `0/9` training and `0/18` development exact
locked boards, role exactness that swings between runs (`9/9` and `0/9` within the same
family), and `0/9`, `0/18` exact readable components.

That gate has never been measured as a gate. This pass states the rule first, then measures
it with one declared lever.

## The lever

`--occupancy-factor 4`: the locked-occupancy term's weight rises from `1.` to `4.`, the
same kind of declared scale intervention the identity factor used. The reason is the
measured shape of the failure: the last run's nonempty training boards report
`true_positive 0, false_negative 168`, i.e. the occupancy head emits a constant "empty"
across every supervised locked cell, so the available question is whether the term is
gradient-starved at this budget.

Nothing else belongs to the fit question and nothing else changes: capacity 64, component
identity factor 4, capture side 512, seed `20261026`, learning rate `.001`, 128 updates,
evaluation/checkpoint 64, six development groups, the positional read-out with the
four-query hold bank, the length endpoint objective, and every other loss weight.

## Predeclared rule — one artifact, one split, one quantity, one threshold

- **Artifact**: `docs/evidence/m3-fit-occupancy-r1-errors-r1/result.json`, written by the
  independent replay of the new run at its final checkpoint, field
  `totals.training.locked_populations.nonempty.exact_boards`.
- **Split**: training only (nine probes, three ancestries). Development is a diagnostic.
- **Quantity**: the number of **nonempty** training images whose locked board is exactly
  right on every known cell. It is bounded by 7 here, since two of the nine training
  images have empty boards.
- **Threshold**: **3**.

**The gate OPENS iff that count is >= 3.** Otherwise the gate **STANDS (FALSE)**, whatever
else moves.

Ties, partial fits and everything else are recorded as costs and can never upgrade a
negative: exact empty boards (a constant-empty predictor gets these for free and they are
not evidence of fit), development boards, role exactness for the active and ghost layers,
mean geometry error, exact readable components, and classroom/class-frequency artefacts are
all reported and all inert for the verdict. Exact-match counts of the *empty* population,
and any single split's role counts, are never a criterion by themselves; the criterion is
discrimination between occupied and empty cells, which only exact nonempty boards
evidence.

The run is a matched single-seed diagnostic at a fixed budget: opening the gate would say
the locked read-out can fit nine synthetic training views at this budget and nothing more —
no real-image, complete-state, calibration or quality claim, and no licence to re-open the
identity disjunction.

## Receipts this pass must produce

1. Admission, 2 steps, `--profile-square`, bounded at 1,152 MiB / 700 s / 2 work items,
   receipt `runs/m3-fit-occupancy-r1-profile-r1-job.json`.
2. The matched study, admitted only by that profile, bounded at 1,152 MiB / 1,900 s /
   4,096 work items, receipt `runs/m3-fit-occupancy-r1-job.json`.
3. Independent replay and curation at 640 MiB / 600 s / 27 forwards and 128 MiB, with the
   canonical receipt and log names the recorder reads:
   `runs/m3-fit-occupancy-r1-errors-r1-job.json`, `...-errors-r1-curation.json`.

The fit-gate run needs its own owned name family, so `m3-fit-*` is declared in
`record_stream_run.OWNED_RUN_PATTERN` and bound by a test rather than left to a hand-edited
call.

## Resource plan and discipline

The length run peaked 1,011,888,128 B with the same model and input shape; an occupancy
weight is a scalar on an existing term, so the expected peak is unchanged. Declared ceiling
1,152 MiB under the engine's 2 GiB bounded-job cap (V-045). One heavy job at a time,
absolute interpreter paths (V-030), canonical receipts (V-047), killed stages preserved and
re-run to a passing canonical receipt (V-050), objective provenance bound by
`arguments` (V-051) and declared once (V-052). Any defect this pass exposes is fixed at its
root with a regression test and recorded in `docs/BUG_LOG.md`.
