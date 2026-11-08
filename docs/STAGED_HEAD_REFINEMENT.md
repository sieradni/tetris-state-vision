# Staged type-head refinement with a frozen backbone

Status: **predeclared before any run** (see the commit that adds this file; the
run receipts and their evidence folder are added afterwards, and their base
commit is strictly later than this document's commit).

## Question

Every recorded learning failure on the ordered all-class fit now sits in one
place. The ordered 1,024-update fit reaches 32/32 exact locked boards, 128/128
supported semantic cells and .21059-cell geometry, and fails only its 64 role
labels (20/64). At the same time a copied linear head, fitted on that run's saved
frozen feature vectors, reaches 62/64 training types (`TYPE_HEAD_FIT.md`), and the
checkpoint diagnostic finds recurring type-correctness turnover with local
class-gradient conflicts (`TYPE_FORGETTING_DIAGNOSTIC.md`).

Those two facts leave one untested question: can the joint model's own type head
reach, on actual pixels, what the copied head reaches on saved vectors? If it
can, the residual failure is optimizer interference between the shared
representation and the type head, not missing class information. If it cannot,
the frozen-vector fit exploited a feature state the joint forward never presents.

This is a memorization/fit diagnostic, not a release model or real-image test.

## Lever

One bounded second stage. Stage 1 is the existing, untouched ordered artifact
`runs/m2-stream-headrate-control-s3` (source catalogue `18ce798e...`, weights
`1f20007c...`, 1,024 updates, seed 20261026, AdamW .001/decay .01). Stage 2:

- loads that artifact's inference weights as its **recorded initial state**
  (`--init-weights-from`); the saved file hash and that run's checkpoint bytes
  must agree, and the source is read-only;
- freezes every parameter except the shared type head's weight and bias
  (`--freeze-backbone`), so exactly 585 trainable scalars remain;
- starts a **fresh** AdamW at .001/decay .01. No stage-1 optimizer state, RNG or
  cursor is resumed: the vector-fit precedent and the recovered-cursor rule both
  say a new diagnostic optimizer is declared, not inherited;
- consumes the same 1,024 ordered image/target pairs (32 complete cycles, one
  image per update) with seed 20261026 and the same joint loss, global norm
  clipping at 1, evaluation every 256 images and checkpoints every 128 images.

The unchanged criterion is the standing controlled fit: 32/32 exact locked
boards, 64/64 supported active/ghost type-and-pose roles including eight
absences, 128/128 supported semantic cells, mean geometry error below .25 cells.
Non-type outputs are expected to stay exact by construction and are reported as
such; the verdict rests on the role labels.

## Falsifiable predictions, declared in advance

1. If the mechanism is optimizer interference, stage 2 raises roles materially
   above 20/64 — the copied head shows 62/64 is representable — while boards,
   cells and geometry stay at their stage-1 values.
2. If the representation does not present type information in the joint forward,
   roles stay near 20/64 and the vector fit is not a valid target.
3. In both cases development must be reported unchanged in kind: stage 2 cannot
   create generalization from 32 memorized pairs, so a development improvement
   would itself be evidence the stage-1 development miss was an optimization
   artifact. Neither outcome is a generalization claim.

## Admission and gates, before any output is inspected

- Fresh native warm/reload admission for supervisor `d9382fa2...` on the exact
  staged mode: `train_m2_stream.py --profile-square --steps 2 --freeze-backbone
  --init-weights-from runs/m2-stream-headrate-control-s3`, 640 MiB / 300 s /
  work 2. Admission pins the staged mode, so a plain profile cannot admit it.
- The study then runs only through `--admission`, 640 MiB / 600 s / work 1,024,
  as one bounded segment if wall permits, otherwise two bounded segments with
  exact snapshot resume; a `step_partial`/`wall_partial` run keeps its status.
- Independent ordinary replay (`probe_stream_errors.py`, 50 forwards: 32 training
  and 18 development probes) must reproduce every saved per-case metric with
  unchanged weights, then an independent curator (`record_stream_errors.py`)
  must write `scope.json`. The run curator must additionally re-read the
  stage-1 weight bytes from its own run directory.
- The stage-1 artifact is never resumed, extended or relabeled.

Expected peak memory is the measured pilot envelope (below 600 MiB), capped at
640 MiB by the kernel Job with the OS pressure guard armed; one heavy job at a
time.

## Interpretation limits fixed now

One seed, 16 training ancestries, six development ancestries with three distinct
appearance families. A pass would establish that the joint head can fit the
training types it previously missed; it would not establish development,
unseen-family, complete-state or real-image quality. A failure would retire the
"copied head is reachable from pixels" hypothesis and leave architecture,
initialization and data expansion untouched and open.
