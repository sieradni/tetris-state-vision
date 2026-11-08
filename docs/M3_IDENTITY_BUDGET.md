# M3 identity budget lever — predeclaration

Status: executed and evaluated; the predeclared criterion was not met (see Result).
The predeclaration text below is unchanged from its committed form, which was written
before any receipt for this lever existed, so commit ordering still proves the criterion
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

## Result: identity rises with budget, the declared structure condition fails (2026-11-08)

Predeclaration `9970e5a` (2026-11-08 02:38:37 -0800) precedes every receipt below; the first
receipt for this lever is the admission profile written after that commit.

| Receipt | Status | Numbers |
|---|---|---|
| `runs/m3-identity-budget-profile-r1-job.json` | passed | 2 steps, 943,157,248 B peak under 1,152 MiB |
| `runs/m3-identity-factor-budget-r1-job.json` | passed | 512/512 steps, 1729.7 s, 985366528 B peak under 1,152 MiB |
| `runs/m3-identity-factor-budget-r1-errors-r1-job.json` | passed | 27 forwards, 34.9 s, 563867648 B, weights `9b79863b` unchanged, every saved metric reproduced |
| `runs/m3-identity-factor-budget-r1-curation-r1-job.json` | passed | `docs/evidence/m3-identity-factor-budget-r1-errors-r1/scope.json` (`passed`) |

Checkpoint `9b79863b`, source catalogue `ebde6a1b`,
capacity 64, factor 4, separate read-out, curriculum `stream-v1`, seed `20261026`,
canonical six development groups. Matched against `m3-identity-factor-r1` in
everything except the 512-update budget.

| Measure | Training 128 | Training 512 | Development 128 | Development 512 |
|---|---:|---:|---:|---:|
| Exact queue piece tokens | 3/54 | 12/54 | 9/90 | 12/90 |
| Hold identity | 0/9 | 0/9 | 3/12 | 0/12 |
| Queue EOS | 8/9 | 5/9 | 3/15 | 3/15 |
| Queue slot status | 54/54 | 54/54 | 90/90 | 90/90 |
| Queue presence | 54/63 | 54/63 | 90/105 | 90/105 |
| Exact locked boards | 0/9 | 0/9 | 0/18 | 0/18 |
| Exact roles (active+ghost) | 12/18 | 9/18 | 0/36 | 0/36 |
| Mean geometry error (cells) | 3.83 | 3.03 | 2.21 | 2.07 |
| Exact readable components | 0/9 | 0/9 | 0/18 | 0/18 |

Verdict against the predeclared rule, applied as written: a **miss on the declared
criterion**. Both primary counts rise — exact queue piece tokens 3/54
-> 12/54 training and 9/90 ->
12/90 development — and the primary signal is therefore in the
positive direction. The structure-held condition that the same rule requires for a
positive verdict does not hold: training queue EOS falls 8/9 ->
5/9 against the declared >= 8/9, and development hold identity
falls 3/12 -> 0/12 against the declared
>= 3/12. Slot status (54/54 and 90/90) and
presence (54/63 and 90/105) hold, and
the fit gate stays FALSE as expected: 0/9 and
0/18 exact boards, 9/18 and
0/36 exact roles, 0/9 and
0/18 exact readable components, geometry
3.03 and 2.07 cells.

The run also does not match the declared negative branch (neither count rose) or the
declared partial branch (one count with structure held), so the observed state fell
outside the rule set — a predeclaration defect recorded as V-044 and fixed by making
the rule total over the primary-rise and structure-held square. No value is retuned
and no second budget is tried.

What the numbers support: identity is budget-sensitive, since four times the updates
over four times the ancestry range nearly quadruples the exact training tokens
(3/54 -> 12/54) and raises development
(9/90 -> 12/90), while the same
budget costs training EOS, all development hold identity and three of the twelve
training roles (12/18 -> 9/18). What they do
not support: a fit claim, a readable component claim, or the reading that budget alone
is the ceiling — the trade-off moved with the budget rather than clearing.

Ranked next: the input representation (pixels per icon, i.e. the model capture side),
because budget is shown not to be the sole limit and a further budget extension would
repeat the same structural trade-off. That lever needs its own predeclaration,
admission and criteria, and nothing about it is admitted here.
