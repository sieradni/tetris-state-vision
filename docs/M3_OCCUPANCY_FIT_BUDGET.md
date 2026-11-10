# The M2 fit gate at capacity 64 — is the model under-trained or under-specified? (predeclaration)

Status: **declared before any receipt exists.** This file fixes one lever, one criterion, one verdict. It
was committed before the first admission profile, study, sweep or probe for the `m3-occfit-*` family was
launched, and the curator refuses any receipt written before that commit's timestamp.

Previous lever: per-board lattice refinement capacity
([M3_OCCUPANCY_BOARD_REFINEMENT.md](M3_OCCUPANCY_BOARD_REFINEMENT.md)) failed its own activation condition
(applied correction 0.0014 cells) and regressed placement 19.2%, so the declared non-circular rule took its
`fit_gate` branch and named **the M2 fit gate at capacity 64**. This pass spends exactly that step: it turns
the standing FIT-GATE-FALSE sentence into a measured lever rather than an argument.

## The question, stated as a lever

Every M3 stream run reports **0 of 28** exact nonempty training boards: the model recovers individual cells
but never reproduces a whole board. Two explanations remain, and they rank different next work:

1. **Under-trained.** The shared trunk is fine but 256 continuation steps at this budget are not enough to
   fit even the supervised population; more compute would move fit.
2. **Under-specified.** Fit is already saturated for this architecture at capacity 64; more compute buys
   nothing and the limiter is representational.

This pass answers that question with one declared compute intervention, not with prose.

## The change under test

**Training budget only.** The lever arm continues the published checkpoint for **1,024 optimizer steps**
(4x the settled 256-step continuation, 8x the source checkpoint's own 128 steps); the baseline arm runs the
**256 steps** every reconciled number in this campaign was produced at. Nothing else differs:

- **No new parameters.** Explicit statement: the lever adds **0 tensors and 0 parameters**; both arms are
  byte-identical in architecture, and the baseline arm is bitwise the reconciled configuration
  (`occupancy_factor 4`, positive weight `4`, detail tap `1`, stencil `0`, context `0`,
  `geometry_final_only 0`, `lattice_support 0`, `box_extent_weight 0.0`, `support_anchor_weight 0.0`,
  `lattice_residual 0`, `occupancy_lattice_free 0`, `lattice_board_refine 0`).
- **No objective, label, read-out, stage-policy, seed, capture-size or population change.** Same catalogue,
  same curriculum, same learning rate `.001`, same seed `20261026`.
- **Why this is not the spent 4x-budget pass.** The earlier `4x` lever
  ([M3_OCCUPANCY_SCALED.md](M3_OCCUPANCY_SCALED.md)) ran 256 steps, i.e. 4x of the 64-step stencil runs, on
  a different read-out form. This pass is the first budget lever on the reconciled read-out and takes it to
  1,024 steps, the budget that read-out has never seen.

## The fitting criterion (numeric, executable)

The criterion is the repository's own fit gate, not a new one:

- **Artifact**: the lever arm's own sweep result, `runs/m3-occfit-lever-r1-sweep-r1/result.json`, field
  `deciding.rows` at the standing decision threshold `0.0`.
- **Split**: training (the 28 nonempty deciding boards plus 8 empty boards; empty boards are recorded and
  never counted).
- **Quantity**: `exact_nonempty_boards` — boards whose 200 known cells are all predicted correctly at the
  threshold.
- **Threshold**: `fit_gate.FIT_GATE_THRESHOLD = 3`, applied through `fit_gate.fit_gate_opens` so the rule
  exists in one place.

**The fit gate OPENS iff the lever arm reports at least 3 of 28 exact nonempty training boards.** Otherwise
it **STANDS (FALSE)**, whatever else moves. Empty-board exactness, development counts, role exactness,
geometry error and component counts are costs and can never open it. The mechanism label
(`fit_gate.occupancy_mechanism`: `constant-empty` or `per-cell-error`) is reported with the verdict so a
negative names what pinned it.

## The activation condition (declared, with a floor)

A 4x budget that never reaches the model cannot answer a fit question. The declared engaged quantity is
**behavioural**, measured on the same deciding population as the verdict:

- **Quantity**: the share of known deciding cells whose thresholded decision at `0.0` **flips** between the
  baseline arm and the lever arm, computed from the two probes' own stored logits and labels
  (`decision_flip_rate`).
- **Floor**: **0.05** (5% of known cells). Declared here; it cannot move after the receipt.
- **Falsifier**: a flip rate below the floor fails the activation condition, the verdict records
  `activation: not-engaged`, and this pass may not be read as evidence about training adequacy — the rule
  then ranks a larger declared budget instead of an architecture change.
- **Reported beside it, not a condition**: the relative L2 displacement between the two trained checkpoints
  and the number of differing tensors, so the reader can see how far the extra compute actually moved the
  weights.

## The four standing conditions (reported as costs)

1. Per-cell F1 at `0.0` on the full deciding population `>= 0.70`.
2. Ranking (`percell_gate`) `informative`.
3. Attribution: best swept F1 minus the control's best swept F1 `>= +0.05` (400-draw whole-ancestry-cluster
   bootstrap, seed 20261026, 12 clusters).
4. Placement improvement `>= 25%` in mean absolute offset in target cells against the control's `1.9812`.

The verdict is `cleared` only if the fit gate opens **and** all four conditions and the activation condition
hold. Otherwise it is `not-cleared`, and the failing names are published.

## Declared next-suspect rule (non-circular)

First match wins. No branch can return `translation`, `scale`, `geometry_capacity`, `placement_capacity`,
`lattice_residual`, `lattice_free` or `board_refinement` — those seven levers are inputs, never outputs.

| rule | condition | next suspect |
|---|---|---|
| under_trained | activation flip rate `< 0.05` | a larger declared continuation budget than 4x |
| capacity_limit | engaged, fit gate still closed | a wider shared trunk than capacity 64 |
| fit_opens | engaged and the gate opens | the occupancy label family at the opened gate |

## Matched start

Both arms continue from `m3-percell-balance-r1`
(`9ae1fef9f649fae5c79643061fe04d592b2798692fe103ad23d15f9e497d4b4d`) under one catalogue. Because the
lever adds no tensor, the start is proven the strict way: the two saved initial state dictionaries must be
**bitwise identical**, the declared added-tensor set is exactly empty, and the step-0 probes must agree after
the objective's own value is stripped.

## Run envelope

| stage | work items | memory | wall clock |
|---|---|---|---|
| admission profile, per arm | 2 steps | 1,152 MiB | 300 s |
| baseline study | 256 steps | 1,152 MiB | 1,800 s |
| lever study | 1,024 steps | 1,152 MiB | 5,400 s |
| sweep profile, per arm | 2 forwards | 640 MiB | 300 s |
| sweep, per arm | 54 forwards | 640 MiB | 600 s |
| placement probe, per arm | 36 forwards | 640 MiB | 300 s |
| curation | 8 items | 640 MiB | 300 s |

One heavy job at a time through `scripts/run_bounded.py`. A launch that dies, breaches its envelope or is
superseded is preserved under `runs/attempts/` with a note and is not evidence.

## Predeclaration ancestry

The curator resolves the commit that first added this file with `git log --diff-filter=A`, requires it to be
an ancestor of `HEAD`, and refuses unless **every** receipt it reads was written after that commit's
committer timestamp.

## Reconciliation the receipt must carry

From the seven earlier evidence files (reference only, never verdict input): geometry **+0.0781** at 19.9%
cost, origin **+0.0102** at 11.0%, extent **+0.0318** at 67.5%, capacity **-0.0241** at 28.2%, residual
**-0.0115** at 19.2%, read-out **+0.0147** at 22.3% gain (gate `0.0742`), per-board refinement **+0.0539** at
19.2% worse (activation `0.0014` cells, off). This row adds the fit-gate answer and its activation flip rate.

## Receipts this pass must produce

1. `runs/m3-occfit-{lever,baseline}-r1-profile-r1` (2 steps, 1,152 MiB / 300 s / 2 work items).
2. `runs/m3-occfit-baseline-r1` (256 steps, 1,152 MiB / 1,800 s) and `runs/m3-occfit-lever-r1` (1,024 steps,
   1,152 MiB / 5,400 s), each admitted only by its own profile.
3. `...-sweep-r1-profile-r1` (2 forwards) and `...-sweep-r1` (54 forwards) per arm, 640 MiB.
4. `...-probe-r1` per arm (36 forwards, 640 MiB / 300 s).
5. Curation `runs/m3-occfit-curation-r1-job.json` into
   `docs/evidence/m3-occupancy-fit-budget-r1/scope.json`, re-deriving every number, asserting each probe
   reproduces its own sweep's per-cell F1 (1e-6) and identical TP/FP/FN counts, and asserting the baseline
   reproduces the curated control to 1e-6.
6. The owned family `m3-occfit-*` declared in `record_stream_run.OWNED_RUN_PATTERN`, bound by a test.

## Limits, stated before the measurement

- Synthetic, training-fit population. Opening the gate would say this architecture can fit 28 synthetic
  training views at 4x budget and nothing more: no real-image, complete-state, calibration or quality claim.
- One budget point (1,024 steps), one seed, one continuation, committed catalogue.
- The measurement is a single-seed diagnostic: a null at 4x closes 4x, not "any budget".
- The activation floor is behavioural (prediction flips); a budget can flip decisions without that making it
  adequate, which is a `not-cleared` result, not evidence that compute is irrelevant.
