# M3 identity read-out tying — predeclaration

Status: predeclared; no receipt for this lever exists yet. This file is committed
before the admission profile, before the study and before any replay, so commit
ordering proves the criterion was not chosen from the result.

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
