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

## Result: a declared negative — 2026-11-08

Receipts, all later than the predeclaration at `b229972`:

- Code path (`a892dfb`): one capture-side value is threaded through the sampler, the
  curriculum, the trainer's `prepare_image`/input budget, the admission identity and
  the replay. Default 512 still reproduces every previous recipe byte-for-byte (two
  frozen factor-4 recipes replay identically; the capture-side contract tests assert
  record identity, pixel identity at the declared side and refusal of another side).
- Admission: the predeclared 2,304 MiB ceiling was refused by the engine rather than by
  pressure — `probe_model.bounded_job_snapshot` caps a bounded job at 2 GiB
  (`RuntimeError: use run_bounded.py with <=2 GiB aggregate commit and a wall cap`), so
  the ceiling was lowered to the engine's maximum 2,048 MiB (V-045); the capture side
  was not reduced. The first 640 profile then refused at the trainer's own admission
  because the observed-pixel identity compared the profile maximum to the literal
  `512**2` while this side's second profile case is a real 544x640 canvas (348,160 px);
  fixed at the root to `512**2 <= observed <= max_side**2` (V-046, commit `73f0452`).
  The admission catalogue hashes the frozen source, so the profile was retaken after
  that fix: `runs/m3-identity-representation-profile-r3-job.json` passed (2 steps,
  87.4 s, 944,017,408 B peak under 2,048 MiB, checkpoint `e1ba82fc...`, identical to
  the pre-fix profile, so the fix changed only the guard).
- Study: `runs/m3-identity-factor-representation-r1-job.json` passed, 128/128 steps,
  542.6 s, peak 986,275,840 B, checkpoint `acb6a8a9c2fcdbd922dbf5985499c946beb62da9c806fcec4ddc24ec35737a67`,
  catalogue `2d3623a14a84dadd7c4f2fa71db02ce4a1c3b5bc890de1037f36551b78a3b15c`,
  `max_side` 640, observed maximum 348,160 px.
- Independent replay: `runs/m3-identity-factor-representation-r1-errors-r1-job.json`
  passed, 27 forwards, 39.0 s, 631,767,040 B, `automatic_metrics_reproduced` true and
  the weights unchanged; curated in
  `docs/evidence/m3-identity-factor-representation-r1-errors-r1/scope.json` (`passed`).

Measured against the rule exactly as predeclared:

| Term | Declared | Measured | Condition |
|---|---|---|---|
| training queue tokens | > 3/54 | 3/54 | not risen |
| development queue tokens | > 9/90 | 9/90 | not risen |
| training queue EOS | >= 8/9 | 9/9 | held |
| training slot status | 54/54 | 54/54 | held |
| training presence | >= 54/63 | 54/63 | held |
| training hold identity | >= 0/9 | 0/9 | held |
| development queue EOS | >= 3/15 | 3/15 | held |
| development slot status | 90/90 | 90/90 | held |
| development presence | >= 90/105 | 90/105 | held |
| development hold identity | >= 3/12 | 3/12 | held |

Primary did not rise and structure held, so the verdict is **NEGATIVE** by the
predeclared table: the representation is not the limiter. Reported but not the
criterion, the fit gate is unchanged and FALSE — 0/9 and 0/18 exact boards, 9/18 and
0/36 exact roles, 0/9 and 0/18 exact readable components, mean geometry 3.82 and 2.28
cells — and the replay measures exactly the control's 54 training and 90 development
queue tokens and 9/12 hold cases, so the finer capture did not turn any previously weak
icon into a known component on this probe population.

Deviations from the predeclaration, all declared before their receipts were read: the
2,304 MiB ceiling became 2,048 MiB (engine cap, V-045); the profile is r3 rather than r1
(retaken after V-046, same checkpoint); receipts of refused attempts are preserved with
suffixes while the passing receipt occupies the canonical `<name>-job.json` (V-047). No
other capture side was tried, nothing was retuned, and the budget was not extended.
