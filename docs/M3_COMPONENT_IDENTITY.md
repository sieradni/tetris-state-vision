# M3 component-identity lever — predeclaration

Status: predeclared; no receipt for this lever exists yet. This file is committed
before the admission profile, before the study and before any replay, so commit
ordering proves the criterion was not chosen from the result.

## Where this sits

`docs/M3_JOINT_FIT.md` closed the first M3 joint learning pass as a measured
negative and localized it: at capacity 64 over 128 updates the structural heads
converge (queue EOS 9/9, slot status 54/54, presence 63/63 exact) while piece
identity never leaves collapse (piece tokens 0/54, hold identity 0/9), so every
readable component fails on identity alone. That is the highest-ranked item repo
data supports. Roadmap items 4–5 (admitted real labels, the 95% release gate)
remain unreachable: the 166 real drafts have zero training/scoring eligibility,
and admitting them needs independent review plus permission, which is an owner
decision rather than a code change.

## The lever

One declared change, matched everywhere else: the component objective keeps its
published coefficients, except the queue-token and hold-piece terms are scaled by
`--component-identity-factor 4` instead of 1. Data, curriculum, order, seed,
optimizer, capacity, budget, evaluation and checkpoint cadence, development set
and criterion are identical to `m3-stream-joint-fit-r1`. No architecture, target,
capture, mask or sampling change; structure supervision is untouched.

Hypothesis under test: identity collapse at factor 1 is gradient starvation of two
cross-entropy terms against an already-converged structural objective, not a
missing representation. The factor is a loss scale, not a tuning sweep: exactly
one matched comparison at 4, declared in advance. No other value is tried in this
pass, and a negative is not extended by trying more values.

## Predeclared criterion and decision rule

Primary, on the run's own 9 training probe images at the final checkpoint:

- exact queue piece tokens above 0/54, and
- exact hold identity above 0/9,

with the previous pass's structural terms required not to regress (queue EOS 9/9,
slot status 54/54, presence 63/63).

Full fit criterion, unchanged from the previous pass: 9/9 exact locked boards,
18/18 exact roles, 9/9 exact readable components and mean geometry
<= 0.25 cells.

Decision rule, declared before the run:

- **Measured positive** if identity leaves collapse (either primary count > 0)
  while the structural counts hold: the localization is loss-scale starvation and
  the next lever is identity-focused data/targets at that scale.
- **Measured negative** if both identity counts stay 0 while the structural terms
  stay exact: the collapse is not loss scale, and the next lever is component
  representation or a longer admitted fit — not another coefficient.
- Any structural regression is reported as its own finding, not folded into the
  identity verdict.

Development stays a diagnostic on the canonical 18 images and is reported, not
gated. Passing or failing is a fit statement on nine synthetic views, never
real-image accuracy or a release model.

## Receipts this pass must produce

All under `scripts/run_bounded.py` with a kernel-enforced memory ceiling, wall
cap, work-item cap and receipt.

1. Fresh admission on the exact lever mode:

   ```
   .venv/Scripts/python.exe scripts/train_m2_stream.py --output runs/m3-identity-factor-profile-r1 \
     --model accuracy --component-capacity 64 --component-identity-factor 4 \
     --steps 2 --profile-square --wall-seconds 600 --development-groups 6
   ```

   bounded at 1,152 MiB / 700 s / 2 work items (the admission guard requires the
   profile receipt's declared work-item bound to be exactly 2), receipt
   `runs/m3-identity-factor-profile-r1-job.json`.

2. The matched study:

   ```
   .venv/Scripts/python.exe scripts/train_m2_stream.py --output runs/m3-identity-factor-r1 \
     --model accuracy --component-capacity 64 --component-identity-factor 4 \
     --steps 128 --admission runs/m3-identity-factor-profile-r1 --seed 20261026 \
     --learning-rate 0.001 --evaluate-every 64 --checkpoint-every 64 \
     --development-groups 6 --wall-seconds 1800
   ```

   bounded at 1,152 MiB / 1,900 s / 4,096 work items, receipt
   `runs/m3-identity-factor-r1-job.json`. A wall-partial segment is resumed, never
   reported complete.

3. Independent ordinary replay and curation:

   ```
   .venv/Scripts/python.exe scripts/probe_stream_errors.py \
     --run m3-identity-factor-r1 --output runs/m3-identity-factor-r1-errors-r1
   .venv/Scripts/python.exe scripts/record_stream_errors.py \
     --name m3-identity-factor-r1-errors-r1
   ```

   The probe runs with its declared work-item bound exactly equal to its 27
   forwards; the curator must pass into
   `docs/evidence/m3-identity-factor-r1-errors-r1/scope.json`.

## Discipline

- No retraining of a closed study, no re-tuning of the accumulation negative, no
  weight or arm changes: `m3-stream-joint-fit-r1` and every M2 artifact stay
  byte-identical, and this study is a new artifact.
- Code change is confined to a declared loss factor: an argument, its validation,
  its recovery/argument identity, the admission comparison, and contract tests.
  The default of 1 must reproduce the previous pass's losses exactly.
- One heavy job at a time; capacity-64 envelope expected under 1 GiB, ceiling
  1,152 MiB, one CPU thread, more than 40 GB free.
- Any harness, fixture, admission or gate defect this pass surfaces is fixed at
  its root in the working tree and recorded in `docs/BUG_LOG.md` on a named
  receipt. Every declared bound must be the bound the guard enforces.
- The staged head refinement stays a measured negative, and development stays
  comparable across stages on the canonical 18 images: 4/18 exact boards, 0/36
  exact roles, .901944 cell accuracy and 3.530471-cell geometry for both stages,
  with development loss 11.0885 versus 11.8129. This lever changes neither.
