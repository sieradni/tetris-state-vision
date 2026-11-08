# M3 identity budget lever — predeclaration

Status: predeclared. This file is committed before any admission profile, before the
study and before any replay for this lever, so commit ordering proves the criterion
was not chosen from the result.

## Where this sits

`docs/M3_IDENTITY_EXPOSURE.md` measured the scored probes of the positive
identity-scale run: coverage is complete (all seven piece classes supervised in
both splits), and the only identities ever exact are queue `S` (3/3 training,
9/18 development) and hold `T` (0/3 training, 3/3 development). That measurement
also showed the scored probes are not the training stream: `training_spec` draws a
new ancestry every three steps, so the factor-4 run's 128 updates saw 128 distinct
images from 43 ancestries. The identity question therefore has two live readings —
optimization budget (more updates over more distinct draws) and the input
representation (pixels per icon) — and this pass opens exactly the first one.

## The lever

One declared change, matched everywhere else: **512 updates instead of 128**.
Data policy, order, capacity 64, `--component-identity-factor 4`, separate
read-out, seed `20261026`, learning rate `.001`, evaluation and checkpoint cadence
64, and the six canonical development groups are identical to
`m3-identity-factor-r1`. There is no change to targets, masks, ordering, loss
coefficients, architecture, capture, representation or evaluation.

The lever adds two things at once and cannot separate them: four times as many
optimizer updates and, because the curriculum draws a new ancestry every three
steps, roughly four times as many distinct images (128 -> 512 updates, 43 -> 171
ancestries). That is the curriculum's own definition of budget; the predeclaration
therefore tests "is identity budget-limited?" and makes no claim about presentations
versus diversity separately.

## Predeclared criterion and decision rule

Primary, at the final checkpoint, on the run's own nine training probes and the
canonical eighteen development images:

- exact queue piece tokens strictly above **3/54** training and strictly above
  **9/90** development (the factor-4 run's counts), and
- hold identity not below the factor-4 run: >= 0/9 training and >= 3/12 development.

Structure held, required for a positive verdict:

- training queue EOS >= 8/9, slot status 54/54, presence >= 54/63;
- development queue EOS >= 3/15, slot status 90/90, presence >= 90/105.

Fit gate, unchanged from `M3_JOINT_FIT.md` and reported but not used as this
lever's verdict: 9/9 exact locked boards, 18/18 exact roles, 9/9 exact readable
components and mean geometry <= 0.25 cells. It is expected FALSE and a FALSE gate
is not a failure of this lever.

Decision rule, declared before any receipt:

- **Budget-limited (positive)** if both primary counts rise and the structure-held
  counts hold. The next lever is then a longer or wider admitted fit with the
  structural repair, not a representation change.
- **Not budget-limited (negative)** if neither primary count rises. The budget
  question is then closed, and the next lever is the input representation (pixels
  per icon, i.e. the model capture side), predeclared separately.
- **Partial** (exactly one count rises with structure held) is reported as partial
  with both counts and does not select a next lever by itself.

Any structural regression is its own finding and is not folded into the identity
verdict. A miss is reported as a miss with its receipts; no value is retuned and no
second budget is tried in this pass.

## Receipts this pass must produce

All under `scripts/run_bounded.py` with a kernel-enforced memory ceiling, wall cap,
work-item cap and receipt.

1. Fresh admission, because the source catalogue and supervisor changed since the
   factor-4 admission:

   ```
   .venv/Scripts/python.exe scripts/train_m2_stream.py --output runs/m3-identity-budget-profile-r1 \
     --model accuracy --component-capacity 64 --component-identity-factor 4 \
     --steps 2 --profile-square --wall-seconds 600 --development-groups 6
   ```

   bounded at 1,152 MiB / 700 s / 2 work items, receipt
   `runs/m3-identity-budget-profile-r1-job.json`.

2. The matched study:

   ```
   .venv/Scripts/python.exe scripts/train_m2_stream.py --output runs/m3-identity-factor-budget-r1 \
     --model accuracy --component-capacity 64 --component-identity-factor 4 \
     --steps 512 --admission runs/m3-identity-budget-profile-r1 --seed 20261026 \
     --learning-rate 0.001 --evaluate-every 64 --checkpoint-every 64 \
     --development-groups 6 --wall-seconds 1800
   ```

   bounded at 1,152 MiB / 1,900 s / 16,384 work items, receipt
   `runs/m3-identity-factor-budget-r1-job.json`. A wall-partial segment is resumed
   from its own checkpoint, never reported complete.

3. Independent ordinary replay and curation:

   ```
   .venv/Scripts/python.exe scripts/probe_stream_errors.py \
     --run m3-identity-factor-budget-r1 --output runs/m3-identity-factor-budget-r1-errors-r1
   .venv/Scripts/python.exe scripts/record_stream_errors.py \
     --name m3-identity-factor-budget-r1-errors-r1
   ```

   The replay's work-item bound must be exactly its 27 forwards; the curator must
   pass into `docs/evidence/m3-identity-factor-budget-r1-errors-r1/scope.json`.

## Discipline

- Every frozen run and evidence directory stays byte-identical; this study is a new
  artifact under a new name, and the owned-family guard already admits
  `m3-identity-factor-*`.
- No retraining or re-scoring of a closed study, no coefficient change, no target,
  mask or order change, and no retune of this lever after its result.
- One heavy job at a time; expected peak well under 1 GiB (the factor-4 run peaked
  at 982,077,440 B), ceiling 1,152 MiB, one CPU thread, more than 40 GB free.
- Any harness, admission or gate defect this pass surfaces is fixed at its root in
  the working tree and recorded in `docs/BUG_LOG.md` on a named receipt.
