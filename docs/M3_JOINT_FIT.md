# M3 joint learning fit — predeclaration

Status: predeclared; no receipt for this study exists yet. This document was
committed before the admission profile and before the training run; every
receipt named below is created afterwards, so commit ordering is the proof that
the criterion was not chosen from the result.

## Why this item, and why the real-image chain is not the one running

`docs/ROADMAP.md` ranks two open items above plain model work: item 4 (admitted
reviewed real labels, controlled decoder comparison) and item 5 (the >=95%
exact complete-state release gate). Both need labelled real screenshots. The
repository holds 166 real drafts with **zero training/scoring eligibility** — no
independent review, no permission record, no label artifact
(`docs/REAL_STRESS_SET.md`, `docs/evidence/real-drafts/scope.json`). Nothing in
the tree converts those pixels into admitted labels, so item 4/5 cannot run from
repo data. Unblocking them needs reviewed, permissioned real labels plus honest
session/ancestry/family grouping; that is an owner decision, not a code change.
The skip is recorded here and in `docs/CONTINUATION_STATE.md` rather than
restated as a blocker.

The next ranked item that repo data does support is item 3's open half:
**"Joint learning ... remains pending"** (`docs/M3_COMPONENT_TARGETS.md`). Every
M3 receipt so far covers contracts, resource envelopes and recovery on a
two-update profile checkpoint. The joint heads have never been trained. The
two-update profile's `final_training_probe` reports 0/9 exact readable
components, which is plumbing, not learning.

## What is measured

One bounded joint-fit run of the accuracy model with the optional M3 heads at
the 64-slot capacity, on the existing `stream-v1` synthetic curriculum. Nothing
new is generated, relabelled or resampled: the curriculum, observer, capture
policy, targets, losses and development generator are the committed ones. Queue
and hold supervision therefore arrive only where the public observer supports
them; unknown slots stay masked and never shift later indices.

Fixed configuration: seed `20261026`, AdamW learning rate .001, layer evidence
off, evaluation every 64 updates, checkpoint every 64 updates,
`--development-groups 6` (the canonical development set),
`--curriculum stream-v1`, `--component-capacity 64`, `--model accuracy`.

Budget: 128 updates, trainer wall cap 1,800 s (the trainer accepts only
(0, 1800]). A wall-partial segment is resumed with the admitted recovery
mechanism and never reported as complete. This is the first finite
joint-learning point, deliberately comparable to the M2 `learn16`/`learn128`
first-step practice. It is not a release-model selection and does not extend
the completed M2 1,024-update studies.

## Predeclared criterion

At the final checkpoint, over the run's own 9 saved training probe images:

- 9/9 exact locked boards,
- 18/18 exact known role layers (active + ghost),
- 9/9 exact readable components (every supported queue slot/token/presence,
  panel status, observed start/end and hold status/type correct on images whose
  component is fully readable),
- mean geometry error <= 0.25 source cells.

Passing is a **memorisation/fit** statement about nine synthetic views. Failing
is a measured negative that must be reported with its decomposition (queue
token vs presence vs slot status vs start/end vs hold status vs hold type, board
and role counts) — never as a loss value alone. Development stays a diagnostic:
it is reported on the canonical 18 images and is not part of the gate.

## Admission, run and replay receipts

Predeclared commands, both under `scripts/run_bounded.py` (kernel Job with a
hard memory ceiling, wall cap, work-item cap and receipt):

1. Fresh capacity-64 admission:

   ```
   .venv/Scripts/python.exe scripts/train_m2_stream.py --output runs/m3-stream-joint-fit-profile-r1 \
     --model accuracy --component-capacity 64 --steps 2 --profile-square \
     --wall-seconds 600 --development-groups 6
   ```

   bounded at 1,152 MiB / 700 s / 2 work items (the admission guard requires the
   profile receipt's declared work-item bound to be exactly 2), receipt
   `runs/m3-stream-joint-fit-profile-r2-job.json`.

2. The study itself, admitted only by that profile:

   ```
   .venv/Scripts/python.exe scripts/train_m2_stream.py --output runs/m3-stream-joint-fit-r1 \
     --model accuracy --component-capacity 64 --steps 128 \
     --admission runs/m3-stream-joint-fit-profile-r1 --seed 20261026 \
     --learning-rate 0.001 --evaluate-every 64 --checkpoint-every 64 \
     --development-groups 6 --wall-seconds 1800
   ```

   bounded at 1,152 MiB / 1,900 s / 4,096 work items, receipt
   `runs/m3-stream-joint-fit-r1-job.json`.

3. Independent ordinary replay of the resulting checkpoint (no training, weights
   untouched), then curation through the existing gate:

   ```
   .venv/Scripts/python.exe scripts/probe_stream_errors.py \
     --run runs/m3-stream-joint-fit-r1 --output runs/m3-stream-joint-fit-r1-errors-r1
   .venv/Scripts/python.exe scripts/record_stream_errors.py \
     --name m3-stream-joint-fit-r1-errors-r1
   ```

The replay must reproduce every saved per-case metric and leave the weights
identical; the curator must pass into
`docs/evidence/m3-stream-joint-fit-r1-errors-r1/scope.json`.

## Discipline carried into this study

- No retraining of any closed study, no re-tuning of the accumulation negative,
  no weight or arm changes: the M2 artifacts named in `docs/CONTINUATION_STATE.md`
  stay byte-identical.
- One heavy job at a time; expected peak <= 1 GiB at the admitted capacity-64
  envelope, ceiling 1,152 MiB, one CPU thread, more than 40 GB free before the
  long window.
- Any harness, fixture or gate defect this run surfaces is fixed at its root in
  the working tree and recorded in `docs/BUG_LOG.md` on a named receipt.
- A wall-partial run never reports complete; if the cap is reached the partial
  prefix is retained and reported as such, and the fit criterion is evaluated
  only on a complete checkpoint.
- Bookkeeping correction before any step executed: the first study launch was
  refused by the trainer's `(0, 1800]` wall-budget bound, since the draft above
  had declared 2,700 s. It ran zero steps and wrote no run directory. The refused
  receipt is retained as `runs/m3-stream-joint-fit-r1-launch-refused-job.json`
  with its stderr log; only the wall budget and the admission work-item cap were
  corrected, and nothing in the configuration, data, criterion or replay plan
  changed.

## Comparison stated up front

The staged head refinement is a measured negative and stays so: 36/64 exact
training roles at 1,024 updates against a 64/64 gate. Development is now
comparable across stages on the canonical 18 images — stage 1 (ordinary replay
of its artifact) and stage 2 both reach 4/18 exact boards, 0/36 exact roles,
.901944 cell accuracy and 3.530471-cell geometry, with development loss 11.0885
versus 11.8129. This M3 study is a separate lever and changes neither number.

## Result: measured negative (2026-11-08)

The study ran complete and inside its envelope. Predeclaration commit `90218b3`
(and the budget/bound corrections `c3dc407`) precede every run receipt below;
the run directory was created afterwards.

| Receipt | Status | Numbers |
|---|---|---|
| `runs/m3-stream-joint-fit-profile-r2-job.json` | passed | 82.759 s, 943,411,200 B peak under 1,152 MiB, work-item bound 2 |
| `runs/m3-stream-joint-fit-r1-job.json` | passed | 417.353 s outer / 360.29 s of updates, 984,649,728 B peak, 128 steps |
| `runs/m3-stream-joint-fit-r1-errors-r2-job.json` | passed | 27 forwards, 35.184 s, 546,545,664 B, weights unchanged |
| `runs/m3-stream-joint-fit-r1-curation-r2-job.json` | passed | curator `docs/evidence/m3-stream-joint-fit-r1-errors-r2/scope.json` |

Checkpoint `b2e44c9c...`, source catalogue `7074e6a6...`, component capacity 64,
curriculum `stream-v1`, seed `20261026`, 6 development groups. The independent
replay reproduces every saved per-case metric and leaves the weights identical
(`automatic_metrics_reproduced`, `weights_unchanged`).

| Population | Exact boards | Exact roles | Cell accuracy | Geometry, cells | Loss |
|---|---:|---:|---:|---:|---:|
| Training probes, 9 images (2 updates, profile) | n/a | 3/18 | .9067 | 19.8217 | 29.972 |
| Training probes, 9 images (128 updates) | 0/9 | 12/18 | .8494 | 4.0417 | 7.739 |
| Development, 18 images (2 updates) | n/a | 0/36 | .8886 | 16.1776 | 30.556 |
| Development, 18 images (128 updates) | 0/18 | 0/36 | .8481 | 2.0294 | 7.332 |

The predeclared criterion is **FALSE**: 0/9 exact boards (gate 9/9), 12/18 exact
roles (gate 18/18), 0/9 exact readable components (gate 9/9) and 4.0417-cell
geometry (gate <= 0.25). The 128 updates are not a fit of the nine synthetic
views.

The decomposition is the informative part, and it is not a flat failure:

- Geometry is the term that learned most: mean error fell 19.82 -> 4.04 cells on
  training and 16.18 -> 2.03 on development, with per-column/.24 and per-row/.13
  basis errors, so the residue is a ~4-cell origin offset, not a scale failure.
- Role structure partially learned: ghost identity is 9/9 exact on the training
  probes, active is 3/9 (6 type, 3 column, 6 row and 3 equivalent-rotation
  errors). Development roles stay 0/36, so this is fit, not transfer.
- Component *structure* is exact and component *identity* is absent. Training:
  queue EOS 9/9, slot status 54/54, presence 63/63, but piece tokens 0/54 and
  hold identity 0/9. Development: EOS 15/15, slot status 90/90, presence 90/105,
  status errors 3 (queue) and 6 (hold), pieces 0/90 and 0/12.
- Every exact readable component therefore fails on identity alone. This is the
  same class-collapse signature the M2 ordered fit showed for locked semantics,
  now localized to the queue-token and hold-type heads while their structural
  heads converge.
- No locked cell is a missed known occupancy in the training probes
  (0 false negatives on 1,800 known cells) but 20 true positives against 148
  false negatives on the nonempty development images: the model over-predicts
  occupancy where it has not fit the state.

Development remains 0/18 boards and 0/36 roles, unchanged in kind from the
two-update profile. This study is a fit diagnostic on nine synthetic views. It
does not measure real-image accuracy, does not select a model, and does not
promote M3 to a quality admission. The next M3 lever is component identity
(targets, loss weighting or a longer admitted fit at the same criterion); it
needs its own predeclaration and admission, and none of it is claimed here.

## Bookkeeping corrections retained

Three launches were refused before any training step executed, each for a
declared bound that was authored instead of derived from the contract:

- `runs/m3-stream-joint-fit-r1-launch-refused-job.json`: wall budget 2,700 s, but
  the trainer accepts only (0, 1800].
- `runs/m3-stream-joint-fit-r1-admission-refused-s1*`: admission work-item bound
  256, but the admission guard requires exactly 2. That attempt wrote an empty
  run directory and no steps; it is retained rather than deleted.
- `runs/m3-stream-joint-fit-r1-errors-r1-workitem-refused-job.json`: probe
  work-item bound 64, but the curator requires it to equal the replay's forward
  count exactly (27).

All three receipts and their stderr logs are preserved. Nothing in the
configuration, data, criterion, replay or curation path changed, and no code,
fixture or gate defect was found: the guards behaved as designed. The lesson
recorded is that every declared bound in a predeclaration must be the bound the
guard actually enforces.
