# Staged type-head refinement with a frozen backbone

Status: **predeclared before any run, then run to a measured negative** (the
commit adding this document is `e601182`; every staged receipt and its evidence
folder is strictly later, recorded under "Result" below).

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

## Result — measured negative on the role gate (2026-11-08)

Predeclaration commit `e601182`; the staged admission and every training receipt
are later commits. Two defects had to be repaired before the first step, and both
are closed in `BUG_LOG.md` (V-034, V-035).

Admission. Fresh native warm/reload admission on the exact staged mode passed for
supervisor `d9382fa2...`: `runs/staged-head-refine-profile-r3-job.json`, status
`passed`, peak 442,085,376 B under the 671,088,640 B ceiling, 36.287 s, work 2,
source catalogue `7074e6a6...`, warm backward over exactly 2 parameter tensors (the
shared type head).

Segment history, retained in full rather than rewritten as one clean run:

1. A first attempt was killed by the outer 295 s command cap at step 716 with the
   last committed boundary at 640. It is kept as
   `runs/staged-head-refine-interrupted-s1` with its own receipt and logs.
2. A fresh 1,024-step segment then ran and the supervisor's OS commit guard
   stopped it at step 512 (boundary 384) after processes outside this job drove
   system commit to 31.37 GB; the job's own peak was 430,661,632 B
   (`runs/m2-stream-staged-head-r1-job.json`, status `memory_pressure`).
3. The first resume, `runs/m2-stream-staged-head-r1-s2`, was stopped the same way
   at step 404 (429,887,488 B, 27.843 s).
4. `runs/m2-stream-staged-head-r1-s3` restored the step-384 snapshot and ran the
   remaining 640 updates to step 1,024: `passed`, peak 433,860,608 B, 268.64 s,
   trainer 267.22 s, status `complete`, checkpoint `e02a02e7...`, initial state
   `1f20007c...`, `freeze_backbone: true`, catalogue `7074e6a6...`.

Training fit on the same 32 images and 16 ancestries:

| Quantity | Stage 1 (`m2-stream-headrate-control-s3`) | Stage 2 (`m2-stream-staged-head-r1-s3`) | Gate |
|---|---|---|---|
| Exact locked boards | 32/32 | 32/32 | pass |
| Supported semantic cells | 128/128 | 128/128 | pass |
| Mean geometry error (cells) | .21058594 | .21058594 | pass (< .25) |
| Exact supported roles | 20/64 | 36/64 | **fail (64/64)** |
| Exact observed roles | 12/56 | 28/56 | fail (56/56) |
| Exact absences | 8/8 | 8/8 | pass |
| Copied-head reference | 62/64 | 62/64 | not a gate |

Non-type outputs are identical to the digit, which is what the frozen backbone
predicts. Roles move by +16 (observed +16) and the criterion still fails, so
prediction 1 is only partially supported and prediction 2 is rejected: the
residual is neither pure optimizer interference nor a representation that cannot
present type information. Head-only refinement recovers about a quarter of the
gap to the copied-head fit and then stalls 26 roles short of it. This lever is
therefore closed as a measured negative; re-tuning its rate, optimizer or budget
needs a new predeclared hypothesis, not an extension of this one.

Development (6 groups, 18 images): 4/18 exact boards, all four empty, 0/36 roles,
cell accuracy .9019, geometry 3.5305 cells, loss 11.8129. Stage 1 recorded its
development row on a different denominator (9 images, 3 groups: 2/9 boards, 0/18
roles), so the two are not directly comparable; stage 2 cannot create
generalization and none is claimed.

Independent evidence. `runs/m2-stream-staged-head-r1-s3-errors-r1/result.json`
replays 50 forwards (32 training, 18 development) with unchanged weights
`e02a02e7...`, reproduces every saved per-case metric and passes at 425,635,840 B /
20.032 s; the independent curator then wrote
`docs/evidence/m2-stream-staged-head-r1-s3-errors-r1/scope.json`, re-reading the
stage-1 origin bytes and the source catalogue rather than trusting the printed
origin.

## Interpretation limits fixed now

One seed, 16 training ancestries, six development ancestries with three distinct
appearance families. A pass would establish that the joint head can fit the
training types it previously missed; it would not establish development,
unseen-family, complete-state or real-image quality. A failure would retire the
"copied head is reachable from pixels" hypothesis and leave architecture,
initialization and data expansion untouched and open.
