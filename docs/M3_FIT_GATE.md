# M3 fit gate — locked-occupancy scale lever (predeclaration)

Status: measured (2026-11-08). The rule below was predeclared in its own commit `5065811`,
before any fit-gate admission profile, study or probe existed; the measured result is
the section that follows it.

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

## Measured result: the gate STANDS, and the occupancy term is not the starved term

Receipts: admission `runs/m3-fit-occupancy-r1-profile-r1-job.json` (passed, 2 work items,
`--occupancy-factor 4`), the matched study `runs/m3-fit-occupancy-r1-job.json` (passed,
128/128 updates, 250.6 s, 990,515,200 B peak under the declared 1,152 MiB ceiling,
`arguments.occupancy_factor: 4`), the independent replay
`runs/m3-fit-occupancy-r1-errors-r1-job.json` (passed, 27 forwards, weights `d0bb45ce...`
unchanged, every saved metric reproduced, `occupancy_factor: 4`,
`component_endpoint_term: length`), its curation
`runs/m3-fit-occupancy-r1-errors-r1-curation.json` (passed) into
`docs/evidence/m3-fit-occupancy-r1-errors-r1/scope.json`, and the instrument tests
`runs/m3-fit-occupancy-tests-r4.json` (7 modules, 101 tests, `OK`).

**The gate STANDS on the predeclared rule.** The deciding field,
`totals.training.locked_populations.nonempty.exact_boards`, is **0 of 7**, below the
threshold of 3. Both training empty boards are wrong too (`0/2`), development is `0/14`
nonempty and `0/4` empty, and no other field can upgrade the negative.

| quantity (independent replay) | training | development |
|---|---|---|
| exact nonempty boards | 0/7 | 0/14 |
| occupied cells recovered (TP / known, nonempty) | 34/1400 | 230/2800 |
| nonempty false positives / false negatives | 59 / 134 | 130 / 171 |
| exact empty boards | 0/2 | 0/4 |
| exact roles (active, ghost) | 0/9, 0/9 | 0/18, 0/18 |
| exact readable components | 0/9 | 3/18 (empty-queue cases) |
| queue piece tokens | 15/54 | 9/90 |
| queue EOS | 0/9 | 0/15 |
| queue slot status | 54/54 | 90/90 |
| queue presence | 54/63 | 90/105 |
| hold identity | 0/9 (`T`x9) | 3/12 (`T`x12) |
| geometry mean cells | 3.3397 | 2.6864 |
| position logit spread (max) | 0.0641 | 0.0379 |

**The mechanism is measured, and it is not the one this lever was ranked on.** The motive
was the length run's constant-empty read-out (`true_positive 0, false_negative 168` on the
nonempty training boards). At four times the weight that read-out is no longer constant:
the replay records 34 of 1,400 training occupied cells recovered and 230 of 2,800 in
development, so the occupancy term does receive gradient and does move. It still
reproduces no board, because its errors are per-cell and bidirectional (59 false positives
and 134 misses on 1,400 training cells; 2.4% of occupied cells recovered), and the scale is
paid for elsewhere: queue EOS fell from 9/9 and 15/15 to 0/9 and 0/15, presence from 63/63
and 105/105 to 54/63 and 90/105, and the position logit spread fell to 0.0641 and 0.0379,
the smallest in the identity family. The fit gate is pinned by the locked read-out's
per-cell decision, not by a gradient-starved scalar weight. Scaling the weight changed
which errors the head makes, exactly as the class-frequency measurement (V-048) predicted
for a scaled term; it did not buy fit.

`scripts/fit_gate.py` holds the rule as code (`FIT_GATE_THRESHOLD`, `fit_gate_opens`,
`occupancy_mechanism`) and `tests/test_fit_gate.py` reproduces both the rule (empty-board
successes alone never open it; a missing population is refused rather than read as fit) and
this verdict from the committed bytes: the recorded run reports `occupancy_factor: 4`,
`0/7` exact nonempty training boards, `true_positive > 0`, and the mechanism label
`per-cell-error` rather than `constant-empty`.

Comparison to the identity family's instability: the ghost head was 9/9 exact in the
prototype run, 3/9 in the endpoint run and 0/9 here; roles are 0/9 and 0/18 again; the
three development exact-readable cases are the same empty-queue pattern as before. None of
that instability is fit and none of it is a criterion here.

What this does not license: no real-image, complete-state, calibration or quality claim, no
re-opening of the closed identity disjunction, and no budget extension of this lever (the
term already moves at this budget). The launcher must write the probe's captured stdout and
stderr to `runs/<name>.log` and `runs/<name>.stderr.log` beside the `<name>-job.json`
receipt, which is what the curator copies; V-053 records that trap and its fix.
