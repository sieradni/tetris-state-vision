# M3 identity read-out tying — predeclaration

Status: predeclared and measured (result appended below, 2026-11-08). This file was
committed before the admission profile, before the study and before any replay, so
commit ordering proves the criterion was not chosen from the result.

## Where this sits

`docs/M3_COMPONENT_IDENTITY.md` closed the loss-coefficient question: a declared 4x
identity scale moved queue piece tokens 0/54 -> 3/54 (training) and 0/90 -> 9/90
(development) and hold identity 0/12 -> 3/12, while development EOS fell 15/15 ->
3/15 and no readable component appeared. Identity at the published scale is
gradient-starved, but more weight buys partial identity at a structural cost and
does not make a component readable. The loss-coefficient lever is closed and is not
re-tested here.

Roadmap items 4–5 (admitted real labels, the 95% release gate) remain unreachable
from repo data: the 166 real drafts still have zero training/scoring eligibility and
admitting them is an owner decision (independent review plus permission), not a code
change. They are skipped here for that stated reason.

The narrowest open item is therefore the **identity target/representation** question.

## The lever (parameterization, not a weight)

The published M3 heads give piece identity its own two classifiers:
`components.queue_tokens` (8 classes: seven pieces plus EOS) and
`components.hold_piece` (seven pieces), both randomly initialized and supervised only
by the sparse identity targets — 54 queue tokens and 9 hold cases per pass, against
1,800 supervised cells per pass.

Declared change: with `--component-identity-readout shared`, those two heads are
replaced by **the cell-type classifier the network already uses for locked cells**,
plus one new logit for EOS. The class vocabularies already coincide exactly
(`TYPE_NAMES = IOTSZJLG`, and queue/hold identity ids are `PIECES = IOTSZJL` in that
same order), so the tying needs no remapping: queue logits are
`cell_type(values)[..., :7]` concatenated with one learned EOS logit, and hold logits
are `cell_type(role)[..., :7]`.

Consequence worth stating: identity supervision now also trains the cell-type
read-out, and cell-type supervision now trains the identity read-out. That is the
whole hypothesis — identity collapse is a read-out parameterization problem (two
sparse, unshared heads), not a missing representation in the encoder.

Everything else is matched: capacity 64, 128 updates, seed `20261026`, learning rate
.001, `stream-v1`, evaluation and checkpoint every 64, canonical 6 development
groups, identity loss factor 1 (unchanged). The default `separate` mode must
reproduce the published objective bit-for-bit and is pinned by tests.

## Predeclared criterion and decision rule

Primary, on the run's own 9 training probe images at the final checkpoint:

- exact queue piece tokens above 0/54, or exact hold identity above 0/9,
- with the structural terms held: queue EOS 9/9, slot status 54/54, presence 63/63.

Full fit criterion (unchanged): 9/9 exact locked boards, 18/18 exact roles, 9/9 exact
readable components, mean geometry <= 0.25 cells.

Decision rule, declared before the run:

- **Measured positive** if identity leaves collapse with structure held: tying the
  read-out is the lever.
- **Measured negative** if identity stays 0/54 and 0/9 while structure is exact: the
  collapse is neither scale nor read-out sharing, and the next lever is the input
  representation (pixels per icon) or the component targets themselves.
- A structural regression is reported as its own finding and never folded into the
  identity verdict.

Development is reported, not gated. A pass or fail here is a fit statement about nine
synthetic views, never real-image accuracy or a release model.

## Receipts this pass must produce

All under `scripts/run_bounded.py` with a kernel-enforced memory ceiling, wall cap,
work-item cap and receipt.

1. Fresh admission on the exact shared-read-out mode:

   ```
   .venv/Scripts/python.exe scripts/train_m2_stream.py --output runs/m3-stream-identity-shared-profile-r1 \
     --model accuracy --component-capacity 64 --component-identity-readout shared \
     --steps 2 --profile-square --wall-seconds 600 --development-groups 6
   ```

   bounded at 1,152 MiB / 700 s / 2 work items, receipt
   `runs/m3-stream-identity-shared-profile-r1-job.json`.

2. The matched study against the `m3-stream-joint-fit-r1` baseline:

   ```
   .venv/Scripts/python.exe scripts/train_m2_stream.py --output runs/m3-stream-identity-shared-r1 \
     --model accuracy --component-capacity 64 --component-identity-readout shared \
     --steps 128 --admission runs/m3-stream-identity-shared-profile-r1 --seed 20261026 \
     --learning-rate 0.001 --evaluate-every 64 --checkpoint-every 64 \
     --development-groups 6 --wall-seconds 1800
   ```

   bounded at 1,152 MiB / 1,900 s / 4,096 work items, receipt
   `runs/m3-stream-identity-shared-r1-job.json`. A wall-partial segment is resumed,
   never reported complete.

3. Independent replay and curation:

   ```
   .venv/Scripts/python.exe scripts/probe_stream_errors.py \
     --run m3-stream-identity-shared-r1 --output runs/m3-stream-identity-shared-r1-errors-r1
   .venv/Scripts/python.exe scripts/record_stream_errors.py \
     --name m3-stream-identity-shared-r1-errors-r1
   ```

   The probe runs with its declared work-item bound exactly equal to its 27 forwards;
   the curator must pass into
   `docs/evidence/m3-stream-identity-shared-r1-errors-r1/scope.json`.

## Discipline

- No retraining of a closed study, no re-tuning of the accumulation negative, no
  weight or arm changes: `m3-stream-joint-fit-r1`, `m3-identity-factor-r1` and every
  M2 artifact stay byte-identical.
- The change is confined to the declared read-out mode: a config field, the module
  wiring, its validation, its recovery-contract record, the admission comparison
  (config identity already covers it) and contract tests. Default `separate` keeps the
  published heads and objective bit-for-bit.
- One heavy job at a time; capacity-64 envelope expected under 1 GiB, ceiling
  1,152 MiB, one CPU thread, more than 40 GB free.
- Any harness, admission or gate defect this pass surfaces is fixed at its root and
  recorded in `docs/BUG_LOG.md` on a named receipt. Every declared bound must be the
  bound the guard enforces.
- The staged head refinement stays a measured negative, and development stays
  comparable across stages on the canonical 18 images: 4/18 exact boards, 0/36 exact
  roles, .901944 cell accuracy and 3.530471-cell geometry for both stages, with
  development loss 11.0885 versus 11.8129. This lever changes neither.

## Result — read-out sharing moves identity, costs the role head: a measured negative (2026-11-08)

Status: measured negative on the predeclared criterion. The predeclaration above was
committed at `e83d70e` and nothing in it was revised after the run.

Receipts, in order:

- Admission r1 was refused by the resident-Adam guard because the shared mode still
  carried the two identity heads it never uses (`b80d82d`, V-039; refusal preserved as
  `runs/m3-stream-identity-shared-profile-r1-adam-refused-s1*`).
- Admission r2 on the exact shared mode: `runs/m3-stream-identity-shared-profile-r2-job.json`
  (passed, 80.218 s, 945,143,808 B under 1,152 MiB, checkpoint `c9e8f3f8...`, catalogue `2f69689a...`).
- Matched study against `m3-stream-joint-fit-r1`: `runs/m3-stream-identity-shared-r1-job.json`
  (passed, 128 steps, 574.770 s outer, peak 949,555,200 B, checkpoint
  `55a352433de0966e0a2fc998d362e8bdffbe83462853f0335582acf9f13bb29c`, catalogue
  `2f69689a...`, read-out `shared`, identity factor 1).
- Independent replay and curator: `runs/m3-stream-identity-shared-r1-errors-r1-job.json`
  (27 forwards, 35.867 s, 554,905,600 B) and `docs/evidence/m3-stream-identity-shared-r1-errors-r1/scope.json`
  (`passed`, weights `55a35243...` unchanged, automatic metrics reproduced).

Per-component identity decomposition (baseline = published `m3-stream-joint-fit-r1`,
separate heads, identical capacity/steps/seed/rate/factor):

| Population | Term | Baseline (separate) | Shared read-out |
|---|---|---|---|
| Training | Exact locked boards | 0/9 | **0/9** |
| Training | Exact roles | 12/18 | **0/18** |
| Training | Exact readable components | 0/9 | **0/9** |
| Training | Queue piece tokens | 0/54 | **9/54** |
| Training | Hold identity | 0/9 | **0/9** |
| Training | Queue EOS | 9/9 | **9/9** |
| Training | Slot status | 54/54 | **54/54** |
| Training | Presence | 63/63 | **54/63** |
| Training | Geometry (cells) | 4.0417 | **3.3222** |
| Training | Cells / loss | .8494 / 7.739 | **0.8228 / 7.6864** |
| Development | Exact locked boards | 0/18 | **0/18** |
| Development | Exact roles | 0/36 | **0/36** |
| Development | Exact readable components | 0/18 | **0/18** |
| Development | Queue piece tokens | 0/90 | **6/90** |
| Development | Hold identity | 0/12 | **3/12** |
| Development | Queue EOS | 15/15 | **13/15** |
| Development | Slot status | 90/90 | **90/90** |
| Development | Presence | 90/105 | **90/105** |
| Development | Geometry (cells) | 2.0294 | **2.5458** |
| Development | Cells / loss | .8481 / 7.332 | **0.8389 / 8.1584** |

Hold-status errors: 0 training, 6 development (presence held at 90/105 development).

Reading:

- **Identity moves.** Tying the read-out is not inert: training exact queue piece tokens
  go 0/54 -> 9/54 and development 0/90 -> 6/90, with development hold identity 0/12 -> 3/12.
  Any account of identity collapse as purely a scale problem is now incomplete.
- **Structure does not hold.** The predeclared condition required EOS 9/9, slot 54/54 and
  presence 63/63 training; slot stays exact (54/54 and 90/90) and training EOS stays 9/9,
  but training presence falls to 54/63 and development EOS to 13/15.
- **New cost.** Training exact roles collapse 12/18 -> 0/18, which the 4x coefficient lever
  did not cost (it held 12/18). Geometry improves on training (4.0417 -> 3.3222) and worsens
  on development (2.0294 -> 2.5458). No readable component appears in either population.
- **Fit gate FALSE**, unchanged: 0/9 and 0/18 exact boards, 0/9 and 0/18 exact readable components.

Decision by the predeclared rule: negative. Sharing the cell classifier with the identity
targets raises identity counts but neither holds the declared structure nor makes a
component readable, so read-out parameterization is not the identity lever. Combined with
`M3_COMPONENT_IDENTITY.md`, both identity-side levers reachable from the current data are
now measured: the coefficient and the read-out. What remains is the supervision signal
itself — 54 queue tokens and 9 hold cases against ~1,800 supervised cells per pass — so the
predeclared next lever is the input representation (pixels per icon) or the component
targets, not another re-weighting. Real-image items 4–5 stay an owner decision.

Discipline: the frozen artifacts (`m3-stream-joint-fit-r1`, `m3-identity-factor-r1`, every
M2 run) are byte-identical, the default `separate` mode is pinned by tests to reproduce the
published objective, and no receipt above was weakened or re-scored.

## Verification — the whole suite is green on this tree (2026-11-08)

`docs/evidence/m3-identity-readout-fullsuite-r11/` holds the preserved job receipt,
log and summary of a complete bounded contract run on this tree: 407 tests collected,
`Ran 407 tests in 72.290s`, `OK`, exit 0, peak 545,845,248 B, admitted at
38,347,567,104 B commit against a 45,626,134,118 B stop line. That count includes the
identity-read-out contract tests and the default-`separate` objective pin, so the
published bit-for-bit objective is covered by the same green run. It is derived rather
than assumed: `git diff 5c6b41f HEAD -- tests/` adds exactly 9 test methods and removes
none, so 407 = the previous green run's 398 (its log, preserved at
`docs/evidence/audit-398-baseline-derivation/identity-factor-fullsuite-r1.stderr.log`,
records `Ran 398 tests in 59.453s` and `OK`) + 4 identity-read-out tests + 5 pressure-decision tests, with the
32 component-identity/component/joint-stream tests inside that total.

Getting there fixed the supervisor, not the tests (V-040): the admission guard derived
its commit stop line from physical RAM (`physical − 2 GiB` = 31.4 GB) while the
exhaustible budget is the commit limit (50.7 GB here), so a page-file host's ordinary
steady state refused five runs of a job that peaked at 343 MiB. The line is now the commit
limit's 10% headroom, the unchanged 2 GiB available-physical floor still refuses genuine
physical pressure, and the decision matrix is pinned by
`tests/test_resources.PressureDecisionTests` plus the synthetic-exhaustion cases in
`tests/test_pressure_guard.py`. Every refusal receipt (`-r1`…`-r5`) and the intermediate
failed suite run `-r10` are preserved; nothing was re-scored.

