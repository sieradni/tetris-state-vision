# Per-board lattice refinement capacity — predeclaration

Status: **declared before any receipt exists.** This file fixes one lever, one measurement, one verdict.
It was committed before the first study, sweep, probe or curation job for the `m3-occbref-*` family was
launched, and the curator refuses any receipt written before that commit's timestamp.

Previous lever: the lattice-agnostic occupancy read-out
([M3_OCCUPANCY_LATTICE_FREE.md](M3_OCCUPANCY_LATTICE_FREE.md)) did not clear, its gate reached only
`|tanh| = 0.0742`, and the declared non-circular rule took its spread branch (lever spread `3.3544`,
floor `1.5` cells) and named **per-board lattice refinement capacity**. This pass spends exactly that
lever.

## The declared change

The published decode is one shared affine function of the box head's four sigmoid outputs: origin in
normalized boundaries plus two axis vectors (`decode_lattice`). Two earlier levers changed *where* that
decode is corrected — one global support-moment residual on the raw box (origin), one per-stage
four-vector in decoded cell space (residual) — and both lost the score. Neither let the correction differ
between boards in a way the shared head cannot already express per stage.

This lever adds **one board-conditioned correction of the decoded lattice**: a two-layer projection of the
board token (`dimension -> 64 -> 4`) that produces one four-vector per board, applied through the
published `apply_lattice_residual` map — an origin offset in cells along the decoded axes plus a
log-scale per axis, bounded by `scale = 0.25` and `tanh`.

- **Stage-invariant by construction.** The same four-vector is applied to every decoder stage's lattice,
  so all six stages share one per-board correction. This is capacity *per board*, not *per stage*: the
  residual lever's six independent per-stage corrections are not reintroduced.
- **No new input, labels, objective term, stage policy or read-out.** The geometry term already
  supervises every stage's decoded lattice, so the new vector is trained by the published objective.
- **Step zero is bitwise the published function.** Every tensor of the projection is initialized to
  exactly zero, so the correction is identically zero at step 0 and the paired control starts from the
  same tensors.

## Declared parameter budget

| tensor | shape | count | initial value |
|---|---|---|---|
| `lattice_board_refine.0.weight` | 64 x 192 | 12,288 | exact zero |
| `lattice_board_refine.0.bias` | 64 | 64 | exact zero |
| `lattice_board_refine.2.weight` | 4 x 64 | 256 | exact zero |
| `lattice_board_refine.2.bias` | 4 | 4 | exact zero |

**12,612 parameters in four tensors**, all exactly zero at initialization, constructed last in
`__init__` so every published tensor keeps the same seeded value. No other parameter is added, removed
or re-initialized; the declared cost is 12,612 parameters that the occupancy score must repay.

## The activation condition (new in this pass)

The previous lever's null was ambiguous because its gate never opened: a channel measured at 7% of the
logits cannot support a claim about what that channel could do. This pass therefore declares, **before
the measurement**, that the capacity knob must be shown to have engaged, using a quantity measured on the
same population that decides the verdict:

- **Engaged quantity.** For every board in the deciding population, the probe records the applied
  correction — `0.25*tanh(raw)` per origin axis and `|exp(0.25*tanh(raw)) - 1|` per axis scale — and
  reports their means over all boards and both axes. These are the cells and relative axis changes the
  model actually applied, not a weight norm.
- **Floor.** The lever is **engaged** only if `mean_abs_origin_cells >= 0.10` **and**
  `mean_abs_log_scale >= 0.02` (2% axis scale). Both floors are declared here and cannot move after the
  receipt.
- **Falsifier.** If either mean is below its floor, the activation condition fails, the verdict records
  `activation: not-engaged`, and this pass may **not** be read as evidence about per-board refinement
  capacity — the channel would have been measured while switched off. The rule then takes its fit-gate
  branch rather than pretending a channel was tested.

## The four conditions (unchanged from the five prior levers)

All four must hold on the widened deciding population; a tie at any boundary counts as met.

1. Per-cell F1 at threshold `0.0` on the **full** deciding population (28 nonempty boards x 200 cells)
   `>= 0.70`.
2. Ranking (`percell_gate`) `informative`.
3. Attribution: best swept F1 minus the control's best swept F1 `>= +0.05`, with a 400-draw
   whole-ancestry-cluster bootstrap (seed 20261026, 12 clusters) reported beside it.
4. Placement improvement `>= 25%` in mean absolute offset in target cells.

## Declared regression falsifiers

- **Activation** (above): either engaged mean below its floor — the channel did not switch on.
- **Placement regressed**: mean placement error above the control's `1.9812` cells (improvement `< 0`).
- **Operating-point-only gain**: attribution positive only through the swept threshold while per-cell
  F1 at `0.0` does not move.
- **Capacity moved, score did not**: placement improvement above `5%` while per-cell F1 stays below the
  `0.70` bar and attribution below `+0.05`.

## Declared next-suspect rule (non-circular, two branches)

First match wins. No branch can return `translation`, `scale`, `geometry_capacity`, `placement_capacity`,
`lattice_residual` or `lattice_free` — those six levers are inputs, never outputs.

| rule | condition | next suspect |
|---|---|---|
| fit_gate | activation not engaged, **or** the lever's per-board error spread `>= 1.5` cells | the M2 fit gate at capacity 64 |
| scaling | otherwise (engaged, spread controlled) | occupancy read-out at a second scale |

## Matched start

Both arms continue from the published checkpoint `m3-percell-balance-r1`
(`9ae1fef9f649fae5c79643061fe04d592b2798692fe103ad23d15f9e497d4b4d`), the same start every prior lever
branched from, under one captured catalogue and the same seed. The only declared difference is
`lattice_board_refine` in the model configuration; the objective, curriculum, capture size, population,
stage policy (`geometry_final_only 0`, `lattice_support 0`, `box_extent_weight 0.0`,
`support_anchor_weight 0.0`, `lattice_residual 0`, `occupancy_lattice_free 0`), read-out
(`occupancy_detail 1`, stencil `0`, context `0`), factor `4` and positive weight `4` are shared.

The start is **proven, not assumed**: the curator loads both arms' saved initial weights and requires
(a) the shared tensors bitwise identical, (b) the difference to be exactly the four declared tensors, all
exactly zero, and (c) the step-0 probes equal after stripping the objective's own value.

## Predeclaration ancestry

The curator resolves the commit that first added this file with `git log --diff-filter=A`, requires it to
be an ancestor of `HEAD`, and refuses unless **every** receipt it reads was written after that commit's
committer timestamp. Receipts are untracked artifacts, so time is the proof available; the rule is stated
here, implemented in the curator and exercised by a test.

## Run envelope

| stage | work items | memory | wall clock |
|---|---|---|---|
| admission profile, per arm | 2 steps | 1,152 MiB | 300 s |
| study, per arm | 256 steps | 1,152 MiB | 1,800 s |
| sweep profile, per arm | 2 forwards | 640 MiB | 300 s |
| sweep, per arm | 54 forwards | 640 MiB | 600 s |
| placement probe, per arm | 36 forwards | 640 MiB | 300 s |
| curation | 8 items | 640 MiB | 300 s |

One heavy job at a time, each through `scripts/run_bounded.py`. Any launch that dies, breaches its
envelope or is superseded is preserved under `runs/attempts/` with a note and is not evidence.

## Reconciliation the receipt must carry

From the six earlier evidence files (reference only, never verdict input): geometry **+0.0781** at 19.9%
placement cost, origin **+0.0102** at 11.0%, extent **+0.0318** at 67.5%, capacity **-0.0241** at 28.2%,
residual **-0.0115** at 19.2%, read-out **+0.0147** at 22.3% gain (activation `0.0742`). The receipt
states where this lever sits on that single control, with its own activation magnitude beside it.

## Receipts this pass must produce

1. `runs/m3-occbref-{lever,baseline}-r1-profile-r1` (2 steps, 1,152 MiB / 300 s / 2 work items).
2. `runs/m3-occbref-{lever,baseline}-r1` (256 steps, 1,152 MiB / 1,800 s / 256 work items), each admitted
   only by its own profile.
3. `...-sweep-r1-profile-r1` (2 forwards) and `...-sweep-r1` (54 forwards) per arm, 640 MiB.
4. `...-probe-r1` per arm (36 forwards, 640 MiB / 300 s), carrying the activation diagnostic.
5. Curation `runs/m3-occbref-curation-r1-job.json` into
   `docs/evidence/m3-occupancy-board-refinement-r1/scope.json`, re-deriving every published number and
   asserting each probe reproduces its own sweep's per-cell F1 (1e-6) and TP/FP/FN counts.
6. The owned family `m3-occbref-*` declared in `record_stream_run.OWNED_RUN_PATTERN`, bound by a test.

## Limits, stated before the measurement

- Synthetic, training-fit population; nine of twelve groups never appear in the run's own ledger. No
  real-screenshot, generalization, calibration or complete-state claim follows.
- One budget point (256 steps), one checkpoint, one refinement form (`0.25*tanh`, four numbers, shared
  across stages), one seed, committed lattice only.
- The refinement is axis-aligned in decoded cell space: it cannot express rotation, shear or a
  non-rectangular board.
- The activation floor is a floor on applied magnitude, not on usefulness: a channel can engage and still
  fail, which is a `not-cleared` result and not a null about the channel's existence.
- Placement is measured against the committed target lattice, so it says nothing about deployment
  situations where that lattice is unavailable.
