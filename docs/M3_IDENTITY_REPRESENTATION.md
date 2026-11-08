# M3 identity representation lever — predeclaration

Status: predeclared. This file is committed before any admission profile, before the
study and before any replay for this lever, so commit ordering proves the criterion
was not chosen from the result.

## Where this sits

`docs/M3_IDENTITY_BUDGET.md` closed the budget question as a declared miss: four times
the updates raised exact queue piece tokens 3/54 -> 12/54 training and 9/90 -> 12/90
development, but training queue EOS fell 8/9 -> 5/9, development hold identity fell
3/12 -> 0/12 and training roles fell 12/18 -> 9/18. The roadmap therefore ranks the
input representation next and explicitly does not rank another budget extension.

## The one thing this lever changes

**The model capture side**, i.e. the resolution at which the synthetic frame is
rendered for the model. `model_capture_style` already accepts `max_side`; today it
is pinned to 512 in three places (the stream sampler's final capture, the trainer's
`prepare_image`, and the trainer's summary/admission identity). This lever threads
one explicit value through those places and sets it to **640** for the study.

Why 640 and not something else: the source canvases are about 540x650 px, so at
`max_side=512` the capture scale is `512/650 = 0.79` and drawn icons of 6-12 px
arrive at 4.7-9.5 px, with the smallest below the observer's 5 px strong-icon
threshold. At 640 the same canvases are captured at `640/650 = 0.98`, so the same
icons arrive at 5.9-11.8 px and the smallest cross that threshold. Exactly one value
is declared and tried; there is no sweep.

What this does **not** change: capacity 64, `--component-identity-factor 4`, the
separate read-out, the loss terms and their coefficients, the component target
tables and mask rules, the curriculum and its order, the development set, the seed,
the learning rate, the evaluation/checkpoint cadence, the optimizer, the model
architecture, and the number of updates. `observe.py` decides strong/weak icons by
the same rule as before; because the capture is finer, more of the same icons
qualify, which is the lever's mechanism and not a separate mask or target policy.
Every identity target still comes from the same observer at the actual model frame,
and unknown stays unknown.

Consequence the predeclaration must state plainly: the number of supervised identity
labels per probe can rise, because previously weak icons become strong at the finer
capture. That is the change under test, so the predeclared criterion is stated on the
counts as the independent replay measures them, not against a fixed label budget.

## Matched control

`m3-identity-factor-r1` (committed evidence
`docs/evidence/m3-identity-factor-r1-errors-r3/scope.json`): 128 updates, capacity 64,
factor 4, separate read-out, seed `20261026`, learning rate `.001`, curriculum
`stream-v1`, evaluation and checkpoint every 64, six canonical development groups. Its
final-checkpoint counts are the control column in the verdict rule:
queue tokens 3/54 training and 9/90 development, hold identity 0/9 and 3/12, queue EOS
8/9 and 3/15, slot status 54/54 and 90/90, presence 54/63 and 90/105.

## Predeclared criterion and TOTAL verdict rule

All numbers are read from the independent 27-forward replay of the new run at its
final checkpoint, on that run's own nine training probes and the canonical eighteen
development images. No judgement calls, no post-hoc choice of population.

Definitions, fixed before the run:

- **primary rose** := exact queue piece tokens strictly above 3/54 training **and**
  strictly above 9/90 development.
- **structure held** := training queue EOS >= 8/9, training slot status 54/54,
  training presence >= 54/63, development queue EOS >= 3/15, development slot status
  90/90, development presence >= 90/105, training hold identity >= 0/9 and development
  hold identity >= 3/12.

The rule is total over the two independent conditions:

| Primary | Structure | Verdict | What is recorded |
|---|---|---|---|
| rose | held | **POSITIVE** | representation is sufficient to move identity without structural cost; next lever is scaling this representation and data breadth |
| rose | regressed | **DECLARED MISS** | identity moved at a structural cost; the representation is not sufficient on its own and this pass does not rank another representation value |
| did not rise | held | **NEGATIVE** | representation is not the limiter; the executable M3 identity levers are exhausted and what remains is the owner-decision real-image chain plus the unchanged fit gate |
| did not rise | regressed | **DECLARED MISS (regression)** | no identity gain and a structural cost; the same exhaustion statement as the negative branch applies, with the regression reported separately |

"Rose" and "held" are the two conditions above and nothing else; the declared-miss
branch exists because V-044 found that the previous rule set had no branch for a rise
with a structural regression. The fit gate (9/9 boards, 18/18 roles, 9/9 readable
components, <= 0.25 cells) is reported but is not this lever's criterion.

A miss is reported as a miss with its receipts. No value is retuned, no second capture
side is tried, and the budget is not extended.

## Receipts this pass must produce

1. Fresh admission on the exact representation, because the source catalogue changes:

   ```
   .venv/Scripts/python.exe scripts/train_m2_stream.py --output runs/m3-identity-representation-profile-r1 \
     --model accuracy --component-capacity 64 --component-identity-factor 4 \
     --capture-max-side 640 --steps 2 --profile-square --wall-seconds 600 --development-groups 6
   ```

   bounded at a ceiling large enough for the finer capture (see below), wall 700 s,
   work items 2, receipt `runs/m3-identity-representation-profile-r1-job.json`.

2. The matched study:

   ```
   .venv/Scripts/python.exe scripts/train_m2_stream.py --output runs/m3-identity-factor-representation-r1 \
     --model accuracy --component-capacity 64 --component-identity-factor 4 \
     --capture-max-side 640 --steps 128 --admission runs/m3-identity-representation-profile-r1 \
     --seed 20261026 --learning-rate 0.001 --evaluate-every 64 --checkpoint-every 64 \
     --development-groups 6 --wall-seconds 1800
   ```

   bounded at the measured ceiling, wall 1,900 s, work items 4,096, receipt
   `runs/m3-identity-factor-representation-r1-job.json`.

3. Independent ordinary replay and curation into a new evidence directory:

   ```
   .venv/Scripts/python.exe scripts/probe_stream_errors.py \
     --run m3-identity-factor-representation-r1 --output runs/m3-identity-factor-representation-r1-errors-r1
   .venv/Scripts/python.exe scripts/record_stream_errors.py \
     --name m3-identity-factor-representation-r1-errors-r1
   ```

   The replay must reproduce every saved per-case metric and leave the weights
   unchanged; it reads the capture side from the run summary rather than assuming 512.

## Resource plan

The factor-4 run peaked at 982,077,440 B at capture 512. A capture of 640 raises the
pixel count by (640/512)^2 = 1.56, so the expected peak is about 1.5 GB and a ceiling
of 1,152 MiB cannot admit it. The declared ceiling for both the admission and the
study is therefore **2,304 MiB**, which is under the min(physical RAM / 3, 8 GiB) cap.
The profile receipt is the measurement: if it exceeds 2,304 MiB the ceiling is raised
to the measured need (bounded by that cap) and the change is recorded; the capture side
is never reduced to fit a ceiling, because that would shrink the lever.

## Discipline

- Every frozen run and evidence directory stays byte-identical; all new artifacts use
  new names under the already admitted `m3-identity-factor-*` family.
- The code change is confined to threading one capture-side value through the sampler,
  curriculum, trainer, admission identity and replay. Default 512 must reproduce every
  previous recipe byte-for-byte, which the existing evidence-checking tests and a
  replay of an untouched run verify.
- Any harness, admission or gate defect this pass surfaces is fixed at its root and
  recorded in `docs/BUG_LOG.md` on a named receipt.
