# M3 identity lever — per-piece image queries (predeclaration)

Status: predeclared. This file is committed before any admission profile and before the
study; every receipt named below is created afterwards, so commit ordering proves the
rule was not chosen from the result.

## What the previous pass proved

`docs/M3_IDENTITY_DISCRIMINATION.md` measured that every closed M3 run reports one
class per head: the hold-piece head emits `T` for all 12 development cases and all 9
training cases in three of four runs, and the training queue-token head is provably a
single constant on three of four (`S`, `S`, `I`). Its exact counts were that class's
frequency in the targets. The separate read-out maps one pooled vector through a fixed
linear classifier, and the hold head reads exactly one role vector, `roles[:, 1]`.

## The one thing this lever changes

`--component-identity-readout prototype`, a third identity read-out that replaces the
two sparse heads:

- **Queue tokens:** eight learned class queries (`piece_queries`, seven pieces plus
  EOS) attend the observed component memory and the board. A position's class logits
  are its query read-out compared against those image-conditioned class vectors. The
  classifier is therefore not one fixed weight matrix shared by every position and
  every frame: what each class looks like is a function of the frame.
- **Hold piece:** a bank of four learned hold queries (not one role vector) attends the
  same memory, and the hold identity is the pooled bank compared against the same
  seven class vectors.

Everything else is held: capacity 64, `--component-identity-factor 4`, capture side
512, `stream-v1`, seed `20261026`, learning rate `.001`, evaluation/checkpoint 64,
six development groups, 128 updates, the optimizer, the loss terms and weights, the
component target tables and masks, the board/role/type heads. The queue EOS position
keeps its own learned logit so the endpoint is not forced through the class bank.

This is a read-out conditioning change, and it is the only change. Discrimination is
possible by construction in the sense that nothing about the architecture forces a
single emitted class; whether 128 updates actually separate classes is the measurement.

## Control

`m3-identity-factor-r1` (evidence `docs/evidence/m3-identity-factor-r1-errors-r3/`):
training queue tokens 3/54 with provable constant class `S`, hold 0/9 predicted `T`
for every case, development 9/90 and 3/12. Its structural terms are training EOS 8/9,
slot status 54/54, presence 54/63, hold status 12 cases correct; development EOS 3/15,
slot 90/90, presence 90/105.

## Predeclared criterion — discrimination only, total over two conditions

All numbers come from the independent 27-forward replay of the new run at its final
checkpoint, on that run's own nine training probes, read from the per-split
`predicted_classes`, `distinct_predicted_classes` and `constant_predictor` aggregates
that V-048 added. The development split is a diagnostic and is not part of the verdict.

- **queue discriminates** := training `queue.distinct_predicted_classes >= 3` (of the
  7 piece classes) and `queue.constant_predictor` is false.
- **hold discriminates** := training `hold.distinct_predicted_classes >= 3` (of the
  7 piece classes across the 9 hold cases).

Exact/masked counts are **reported only**. They are not pass conditions, because the
audit already showed an exact count is a class frequency when a head is constant.

| Queue | Hold | Verdict | What is recorded |
|---|---|---|---|
| discriminates | discriminates | **POSITIVE** | both identity heads separate classes; next lever is budget/data at this read-out |
| discriminates | not | **PARTIAL (hold)** | the queue read-out separates classes and the four-query hold bank does not; the hold path is the named obstacle |
| not | discriminates | **PARTIAL (queue)** | the hold bank separates classes and the per-position comparison does not |
| not | not | **NEGATIVE** | the read-out is not what pins identity; the mechanism must be named from measured evidence |

A NEGATIVE is a legitimate result and is reported as such, with the concrete mechanism
that pins it stated from the run's own artifacts (for example: which class the queries
collapse to, how many distinct classes appear at all, and whether the structural heads
moved). A PARTIAL or POSITIVE is not an accuracy claim: this is a 9-image synthetic
fit diagnostic, not a real-image, complete-state or release measurement.

Structural cost is recorded, never a verdict branch: training/development EOS, slot
status and presence are reported beside the control and any fall is named in the
record. It cannot upgrade a NEGATIVE.

## Receipts this pass must produce

1. Fresh admission on the new read-out (the source catalogue changes), 2 steps,
   `--profile-square`, wall 600 s, bounded at 1,152 MiB / 700 s / exactly 2 work items,
   receipt `runs/m3-identity-factor-prototype-profile-r1-job.json`:

   ```
   .venv/Scripts/python.exe scripts/train_m2_stream.py --output runs/m3-identity-factor-prototype-profile-r1 \
     --model accuracy --component-capacity 64 --component-identity-factor 4 \
     --component-identity-readout prototype --steps 2 --profile-square \
     --wall-seconds 600 --development-groups 6
   ```

2. The matched study, admitted only by that profile, bounded at 1,152 MiB / 1,900 s /
   4,096 work items, receipt `runs/m3-identity-factor-prototype-r1-job.json`:

   ```
   .venv/Scripts/python.exe scripts/train_m2_stream.py --output runs/m3-identity-factor-prototype-r1 \
     --model accuracy --component-capacity 64 --component-identity-factor 4 \
     --component-identity-readout prototype --steps 128 \
     --admission runs/m3-identity-factor-prototype-profile-r1 --seed 20261026 \
     --learning-rate 0.001 --evaluate-every 64 --checkpoint-every 64 \
     --development-groups 6 --wall-seconds 1800
   ```

3. Independent ordinary replay and curation into a new evidence directory:

   ```
   .venv/Scripts/python.exe scripts/probe_stream_errors.py \
     --run m3-identity-factor-prototype-r1 --output runs/m3-identity-factor-prototype-r1-errors-r1
   .venv/Scripts/python.exe scripts/record_stream_errors.py \
     --name m3-identity-factor-prototype-r1-errors-r1
   ```

   bounded at 640 MiB / 600 s / 27 forwards and 128 MiB for the curator. The replay
   must reproduce every saved per-case metric, leave the weights unchanged, and its
   result must carry the discrimination fields and per-split aggregates.

## Resource plan

The factor-4 separated read-out peaked 982,077,440 B at capture 512 with 128 updates.
The prototype read-out adds 12 query vectors (8 class + 4 hold) at dimension 192 and
four extra cross-attention sequences per stage, so the expected peak is about the same
1.0 GB. Declared ceiling 1,152 MiB, which is under the engine's 2 GiB bounded-job cap
(V-045) and under min(physical/3, 8 GiB). If the profile exceeds it the ceiling is
raised to the measured need and the change is recorded; the lever is never shrunk to
fit.

## Discipline

- Frozen runs, evidence directories and every closed study stay byte-identical; all new
  artifacts use new names in the owned `m3-identity-factor-*` family.
- Default `separate` must reproduce every previous recipe byte-for-byte; the existing
  read-out tests plus the shared-mode regression stay green.
- One heavy job at a time; interpreter paths absolute (V-030); receipts canonical
  (V-047); every declared bound is the bound the guard enforces.
- Any harness, admission, recorder or receipt defect this pass exposes is fixed at its
  root and recorded in `docs/BUG_LOG.md` on a named receipt.
