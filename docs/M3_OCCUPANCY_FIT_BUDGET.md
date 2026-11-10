# The M2 fit gate at capacity 64 — is the model under-trained or under-specified? (predeclaration)

Status: **declared before any receipt exists.** This file fixes one lever, one criterion, one verdict. It
was committed before the first admission profile, study, sweep or probe for the `m3-occfit-*` family was
launched, and the curator refuses any receipt written before that commit's timestamp.

Amendment (2026-11-10, before any study receipt): the declared 5,400 s lever envelope is **inadmissible**.
`scripts/run_bounded.py` enforces a hard 3,600 s cap on itself and refused the launch with
`run_bounded.py: error: command and wall cap in (0, 3600] required`; the rejected launch is preserved as
`runs/attempts/m3-occfit-lever-r1.attempt1.log` with a note and is not evidence. The lever is therefore
declared at **768 steps (3x the settled 256)** inside **3,600 s**, the largest single job the runner admits.
Raising the guard was not considered. The fitting criterion, the activation floor, the matched start and the
zero-parameter declaration are unchanged; only the budget and its wall ceiling move.
Second amendment (2026-11-10, before any study receipt): the 3,600 s ceiling is itself inadmissible. The
training driver refuses any wall above **1,800 s**
(`train_m2_stream.py: error: wall budget (0,1800] and learning rate (0,.01] required`), and its
`--resume-from` completes only an *unfinished* stream, so a finished 256-step arm cannot be extended in
segments. At the measured **4.09 s/step** (256 steps in 1,047.5 s) one admitted job reaches about 430 steps,
so the lever is declared at **384 steps (1.5x the settled 256)** inside **1,800 s**, leaving headroom rather
than a race against the cap. The rejected launch is preserved as
`runs/attempts/m3-occfit-lever-r1.attempt2.log`. Criterion, activation floor, matched start, zero added
parameters and the rule are unchanged; when the gate stays closed at the largest admissible budget the rule
ranks the wider trunk, which is the branch this truncated budget is designed to decide.

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

**Training budget only.** The lever arm continues the published checkpoint for **384 optimizer steps**
(1.5x the settled 256-step continuation, 3x the source checkpoint's own 128 steps); the baseline arm runs the
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
  384 steps, the largest budget one admitted job can hold under the driver's own 1,800 s cap.

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

A 1.5x budget that never reaches the model cannot answer a fit question. The declared engaged quantity is
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
| under_trained | activation flip rate `< 0.05` | a larger declared continuation budget than 1.5x |
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
| lever study | 384 steps | 1,152 MiB | 1,800 s |
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
2. `runs/m3-occfit-baseline-r1` (256 steps, 1,152 MiB / 1,800 s) and `runs/m3-occfit-lever-r1` (384 steps,
   1,152 MiB / 1,800 s), each admitted only by its own profile.
3. `...-sweep-r1-profile-r1` (2 forwards) and `...-sweep-r1` (54 forwards) per arm, 640 MiB.
4. `...-probe-r1` per arm (36 forwards, 640 MiB / 300 s).
5. Curation `runs/m3-occfit-curation-r1-job.json` into
   `docs/evidence/m3-occupancy-fit-budget-r1/scope.json`, re-deriving every number, asserting each probe
   reproduces its own sweep's per-cell F1 (1e-6) and identical TP/FP/FN counts, and asserting the baseline
   reproduces the curated control to 1e-6.
6. The owned family `m3-occfit-*` declared in `record_stream_run.OWNED_RUN_PATTERN`, bound by a test.

## Limits, stated before the measurement

- Synthetic, training-fit population. Opening the gate would say this architecture can fit 28 synthetic
  training views at 1.5x budget and nothing more: no real-image, complete-state, calibration or quality claim.
- One budget point (384 steps), one seed, one continuation, committed catalogue.
- The measurement is a single-seed diagnostic: a null at 1.5x closes 1.5x, not "any budget".
- The activation floor is behavioural (prediction flips); a budget can flip decisions without that making it
  adequate, which is a `not-cleared` result, not evidence that compute is irrelevant.

---

# Measured (2026-11-10)

Both arms continued the published checkpoint `9ae1fef9f649...` under one captured catalogue (`f0c833eb9221...`) inside
their declared envelopes: lever **384 steps, 1,033,187,328 B /
1522.4 s** (declared 1,152 MiB / 1,800 s) and baseline **256 steps,
1,060,024,320 B / 1046.2 s** (declared
1,152 MiB / 1,800 s). The sweeps (54 forwards, 640 MiB) took 67.8 s / 68.1 s
and the placement probes (36 forwards, 640 MiB / 300 s) 45.3 s / 46.7 s. Curation `runs/m3-occfit-curation-r1-job.json` passed and
published `docs/evidence/m3-occupancy-fit-budget-r1/scope.json`; every measurement receipt it read - study, sweep, sweep profile, probe - postdates the
predeclaration commit `875543d`. One receipt does predate it: `runs/m3-occfit-focused-r1-job.json`, the bounded job
of the authoring test run, which measures no arm and carries no study, sweep or probe output.

**Matched start.** With no declared tensor the guard is the strict form: the two saved initial state
dictionaries are **bitwise identical** (`eec1e10ee4d1...`), the declared added-tensor set is exactly empty, and
the step-0 probes agree after the objective's own value is stripped. The trained weights differ only
through the extra steps: relative L2 displacement **0.0724** across
**266 of 271** tensors.

**Activation: ENGAGED.**
The budget **engaged**: **0.5481** of known deciding cells (3946/7200) changed decision between the arms, against the declared floor **0.05**.

**Fit gate: STANDS (FALSE).**
The fit gate **STANDS (FALSE)**: **0 of 28** exact nonempty training boards at threshold `0.0`, below the declared threshold **3**.
The mechanism label is **`per-cell-error`** (the repository's own
`fit_gate.occupancy_mechanism`): the read-out does predict occupied cells and still reproduces no board, so the errors are per-cell and bidirectional. The baseline arm reports
**0 of 28** at the same threshold, mechanism
`per-cell-error`.

**Verdict: `not-cleared`, failing per_cell_f1, ranking, placement, fit_gate.** Per-cell F1@0 **0.2808**
(661 TP / 3380 FP /
6 FN) against **0.4146** (351 /
675 / 316); AUC **0.8998** against
**0.8255**; best swept F1 **0.5459** at `4.35` against
**0.4237** at `-0.3`, attribution **+0.1221** against the plus-0.05
floor, and the twelve-cluster interval on the per-cell F1 delta **-0.2589 to -0.0326** - the gain is
in ranking toward a stricter operating point (the swept optimum moved from `-0.3` to `4.35`), not at the
standing `0.0` decision where the counts moved to 661 TP / 3380 FP / 6 FN. Placement moved
1.9812 -> 1.8419 cells
(+7.0% improvement), and the per-board spread 4.6043 ->
2.6965.

**What the 1.5x budget did.** The extra compute moved the head's decisions and still bought no exact board, so at this budget the limit is not compute alone.
The deciding occupancy numbers moved by per-cell F1 **-0.1338** and
AUC **+0.0743**, while the deciding-threshold counts went
351/675/316 ->
661/3380/6.
Read against the campaign's other levers, this is a budget lever whose activation number is published:
it engaged and still stands the gate.

**Reconciliation.** Eight levers now sit on the same curated control:

| lever | attribution | placement | activation | verdict |
|---|---|---|---|---|
| geometry stage | +0.0781 | 19.9% cost | n/a | not-cleared |
| box origin | +0.0102 | 11.0% cost | n/a | not-cleared |
| box extent | +0.0318 | 67.5% cost | n/a | not-cleared |
| geometry capacity | -0.0241 | 28.2% cost | n/a | not-cleared |
| per-stage residual | -0.0115 | 19.2% cost | n/a | not-cleared |
| lattice-free read-out | +0.0147 | 22.3% gain | gate 0.0742 | not-cleared |
| per-board refinement | +0.0539 | 19.2% worse | 0.0014 cells (off) | not-cleared |
| fit budget (1.5x) | +0.1221 | +7.0% | 0.5481 flips | not-cleared |

**Ranked next.** The declared non-circular rule takes its **`capacity_limit`** branch and names
**a wider shared trunk than capacity 64**. **Ranked next: predeclare a wider shared trunk than capacity 64**, read with the same conditions and the same activation
discipline; the other branch stays behind it, and the seven spent levers cannot be returned by the rule.

**Unchanged.** No gate widened or retuned, no ceiling, population, label or prior-evidence change, no frozen
manifest touched. Limits are as declared: synthetic training-fit, one budget point, one seed, committed
lattice only. No real-image, generalization or complete-state claim follows.
