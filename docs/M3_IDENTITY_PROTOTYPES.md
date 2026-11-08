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

## Result: NEGATIVE — the class side was conditioned, the position side was not

The lever ran complete inside its envelope and is judged against the rule above,
which was committed at `bdae6e1` before the admission profile existed.

| Receipt | Status | Numbers |
|---|---|---|
| `runs/m3-identity-factor-prototype-profile-r1-job.json` | passed | 2 steps, 39.055 s, 967,761,920 B under 1,152 MiB |
| `runs/m3-identity-factor-prototype-r1-job.json` | passed | 128/128 steps, 223.786 s, 1,013,481,472 B, checkpoint `f818567d...` |
| `runs/m3-identity-factor-prototype-r1-errors-r1-job.json` | passed | 27 forwards, 15.933 s, 561,905,664 B, weights unchanged, saved metrics reproduced |
| `runs/m3-identity-factor-prototype-r1-errors-r1-curation-r2.json` | passed | curator into `docs/evidence/m3-identity-factor-prototype-r1-errors-r1/scope.json` |

Measured over the run's own nine training probes and eighteen development images:

| Term | Training | Development | Control (factor 4, separate) |
|---|---:|---:|---:|
| Queue piece tokens exact | 0/54 | 0/90 | 3/54, 9/90 |
| Queue `predicted_classes` | `EOS` x54 | `EOS` x90 | `S` x3 (provable constant) |
| Queue `distinct_predicted_classes` | **1** | **1** | 1 on training |
| Queue `queue_constant_predictor` | **true** | **true** | true |
| Hold `predicted_classes` | `L`x3, `O`x6 | `L`x2, `O`x9, `S`x1 | `T` x9 / `T` x12 |
| Hold `distinct_predicted_classes` | **2** | 3 | 1 |
| Hold exact | 3/9 | 0/12 | 0/9, 3/12 |
| Queue EOS exact | 9/9 | 15/15 | 8/9, 3/15 |
| Slot status exact | 54/54 | 90/90 | 54/54, 90/90 |
| Presence exact | 39/63 | 48/105 | 54/63, 90/105 |
| Exact roles (active + ghost) | 0/18 | 0/36 | 12/18, 0/36 |
| Mean geometry, cells | 3.5626 | 2.3223 | 3.83, 2.21 |
| Exact boards | 2/9 (both empty) | 0/18 | 0/9, 0/18 |

By the predeclared rule the verdict is **NEGATIVE**: queue discrimination needs >= 3
distinct training classes and measures **1**, hold discrimination needs >= 3 and
measures **2**. Neither head reached the threshold, so the table's fourth row applies.

### The mechanism the artifacts pin

The conditioning change worked on the side it was applied to and not on the side that
decides. The class vectors are now image-derived — the regression test shows identical
queries producing different class vectors on different frames — but the **emitted**
class is invariant across all 54 supervised positions from nine different frames, and it
is `EOS`, the one class whose own term was already satisfied (EOS exact 9/9 training and
15/15 development, better than the control's 8/9 and 3/15). Predicting `EOS` everywhere
keeps the endpoint and slot-status terms perfect and pays only the masked piece term, so
it is the cheapest constant available to the head. What remains position-varying in the
query is only the index encoding plus the shared queue role vector; the per-class bank
conditions *what a class looks like*, never *which position is which*. Discrimination is
therefore not reachable through the class side at this budget: the position read-out
never separates, and the head takes the constant whose structural terms are free.

The hold path is the exception and the useful signal: its four-query bank moved off a
single constant (`T` everywhere in the control) to two classes on training and three on
development, with 3/9 exact training pieces against 0/9. It still fails the >= 3
training-class threshold, but per-query conditioning is the only mechanism in this pass
that produced any class separation at all.

Structural cost, recorded and not a verdict branch: presence falls 54/63 -> 39/63
training and 90/105 -> 48/105 development, roles fall 12/18 -> 0/18 (active 0/9 and ghost
0/9, all type errors), while EOS rises and geometry is comparable (3.83 -> 3.56 training,
2.03 -> 2.32 development). The fit gate is unchanged and still **FALSE** — 2/9 and 0/18
exact boards (the two training boards are empty), 0/18 and 0/36 exact roles, 0/9 and 0/18
exact readable components.

### Defect exposed and fixed

V-049: the replay crashed in the discrimination tally because a supervised piece
position predicted as `EOS` indexed past the seven-name piece vocabulary. The tally now
carries the endpoint as an emitted class, refuses any class outside the declared
vocabulary, and has a regression test. The crash is also positive evidence for the
mechanism above: the head emitted `EOS` at a piece position on the very first replay.

### Two attempts retained, and why

The first replay attempt crashed on V-049 and left `runs/m3-identity-factor-prototype-r1-errors-r1`
with only `progress.json`; it is preserved as `runs/attempts/m3-identity-factor-prototype-r1-errors-r1-attempt1`
and the passing output occupies the canonical path. Its receipt and logs are kept as
`-job-attempt1`. The first curation attempt failed because the curator copies
`runs/<name>.log`, not `runs/<name>-job.log`; the partial evidence directory it had
already created is preserved as `runs/attempts/evidence-<name>-attempt1`, the missing
logs were supplied under the names the curator reads, and the recorded curation is
`-curation-r2`. Nothing was deleted or re-scored.

### Limits and next

Nine synthetic training probes, six development ancestries, one seed, 128 updates at
capacity 64, factor 4, capture 512. This is a fit/discrimination diagnostic on synthetic
views: no real-image, complete-state, calibration or release claim follows, and the
earlier identity verdicts are unchanged.

Ranked next is the **position side** of the identity read-out, which this pass leaves as
the named obstacle: give each supervised position its own image-conditioned query (or a
spatial anchor for the panel icon) so the read-out can differ between positions, and keep
the four-query hold bank, which is the one mechanism that separated classes. Re-running
the same class-side lever at a larger budget is not ranked: the emitted class is
the cheapest constant, and budget does not change that economics.
