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

Budget: 128 updates, trainer wall cap 2,700 s. This is the first finite
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

   bounded at 1,152 MiB / 700 s / 64 work items, receipt
   `runs/m3-stream-joint-fit-profile-r1-job.json`.

2. The study itself, admitted only by that profile:

   ```
   .venv/Scripts/python.exe scripts/train_m2_stream.py --output runs/m3-stream-joint-fit-r1 \
     --model accuracy --component-capacity 64 --steps 128 \
     --admission runs/m3-stream-joint-fit-profile-r1 --seed 20261026 \
     --learning-rate 0.001 --evaluate-every 64 --checkpoint-every 64 \
     --development-groups 6 --wall-seconds 2700
   ```

   bounded at 1,152 MiB / 2,800 s / 4,096 work items, receipt
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

## Comparison stated up front

The staged head refinement is a measured negative and stays so: 36/64 exact
training roles at 1,024 updates against a 64/64 gate. Development is now
comparable across stages on the canonical 18 images — stage 1 (ordinary replay
of its artifact) and stage 2 both reach 4/18 exact boards, 0/36 exact roles,
.901944 cell accuracy and 3.530471-cell geometry, with development loss 11.0885
versus 11.8129. This M3 study is a separate lever and changes neither number.
