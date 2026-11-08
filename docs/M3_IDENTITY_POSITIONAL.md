# M3 identity lever — per-position image queries (predeclaration)

Status: predeclared. This file is committed before any admission profile and before the
study; every receipt named below is created afterwards, so commit ordering proves the
rule was not chosen from the result.

## The obstacle this addresses

`docs/M3_IDENTITY_PROTOTYPES.md` measured a NEGATIVE with the mechanism named: the
class bank made class *appearance* image-conditioned, but the emitted class was still
invariant across all supervised positions, because the only position-varying input
reaching the decision was a shared index code plus one queue role vector. On
`m3-identity-factor-prototype-r1` every supervised piece position emitted `EOS`
(distinct 1 on training and development), the class whose own term was already exact.

The class side is therefore closed. This lever changes the **position side only**.

## The one thing this lever changes

`--component-identity-readout positional`. It keeps the validated read-out from the
previous run — eight image-derived class queries, positions compared against them, and
the four-query hold bank — and adds one learned query per queue position (`capacity+1`
vectors) that attends the same observed component memory and board:

```
positioned = read(position_queries[:capacity+1] + board + queue_role)
tokens     = positioned @ prototypes.T / sqrt(dimension)
```

Each position therefore reads the frame through its own query, so a position's own
content can move its class logits. Discrimination across positions is possible by
construction; whether 128 updates produce it is the measurement.

Everything else is held identical to `m3-identity-factor-prototype-r1`: capacity 64,
`--component-identity-factor 4`, capture side 512, `stream-v1`, seed `20261026`,
learning rate `.001`, evaluation/checkpoint 64, six development groups, 128 updates,
the optimizer, every loss term and weight, the component target tables and masks, and
the four-query hold bank. The structure heads still read their own `values` path, so
their behaviour is unchanged by construction.

## Control

`m3-identity-factor-prototype-r1` (evidence
`docs/evidence/m3-identity-factor-prototype-r1-errors-r1/`): training queue tokens 0/54
with `EOS` x54 (distinct 1, constant true), hold 3/9 with `L`x3/`O`x6 (distinct 2);
development 0/90 `EOS` x90 and 0/12 `L`x2/`O`x9/`S`x1. Structure: EOS 9/9 and 15/15,
slot 54/54 and 90/90, presence 39/63 and 48/105, roles 0/18 and 0/36, boards 2/9 (both
empty) and 0/18.

## Predeclared criterion — discrimination only, total over two conditions

All numbers come from the independent 27-forward replay of the new run at its final
checkpoint, on that run's own nine training probes, from the per-split
`predicted_classes`, `distinct_predicted_classes` and `constant_predictor` aggregates.
Development is a diagnostic and is not part of the verdict. Exact-match counts are never
a criterion.

- **queue discriminates** := training `queue.distinct_predicted_classes >= 3` (of the
  7 piece classes plus the endpoint) and `queue.constant_predictor` is false.
- **hold discriminates** := training `hold.distinct_predicted_classes >= 3`.

| Queue | Hold | Verdict | What is recorded |
|---|---|---|---|
| discriminates | discriminates | **POSITIVE** | both identity heads separate classes across positions; next lever is budget/data at this read-out |
| discriminates | not | **PARTIAL (hold)** | the position side separates classes and the hold bank does not |
| not | discriminates | **PARTIAL (queue)** | the hold bank separates and the position side does not |
| not | not | **NEGATIVE** | the position side is not what pins identity; the mechanism is named from measured evidence |

A NEGATIVE stands only with the mechanism stated from the run's own artifacts. Because
this lever specifically claims the position axis carries information, a NEGATIVE must
also report the measured **position-axis spread** (`queue.position_logit_spread_max`
and `_mean`, `queue.position_axis_reaches_logits`) and, when that spread is positive
while `distinct_predicted_classes` is 1, record it as the demonstration that
position-varying information reaches the logits and still fails to separate — the
analogue, one level down, of the class-vector regression test. The recorded mechanism
must then name whether the remaining constraint is the read-out, the loss masking or the
data's independence of position, using the measured tables rather than a hypothesis.

Structure and class-frequency results are recorded as a cost and can never upgrade a
NEGATIVE. This is a 9-image synthetic fit/discrimination diagnostic: no real-image,
complete-state, calibration or release claim follows.

## Receipts this pass must produce

1. Fresh admission on the new read-out, 2 steps, `--profile-square`, wall 600 s, bounded
   at 1,152 MiB / 700 s / exactly 2 work items, receipt
   `runs/m3-identity-factor-positional-profile-r1-job.json`:

   ```
   .venv/Scripts/python.exe scripts/train_m2_stream.py --output runs/m3-identity-factor-positional-profile-r1 \
     --model accuracy --component-capacity 64 --component-identity-factor 4 \
     --component-identity-readout positional --steps 2 --profile-square \
     --wall-seconds 600 --development-groups 6
   ```

2. The matched study, admitted only by that profile, bounded at 1,152 MiB / 1,900 s /
   4,096 work items, receipt `runs/m3-identity-factor-positional-r1-job.json`:

   ```
   .venv/Scripts/python.exe scripts/train_m2_stream.py --output runs/m3-identity-factor-positional-r1 \
     --model accuracy --component-capacity 64 --component-identity-factor 4 \
     --component-identity-readout positional --steps 128 \
     --admission runs/m3-identity-factor-positional-profile-r1 --seed 20261026 \
     --learning-rate 0.001 --evaluate-every 64 --checkpoint-every 64 \
     --development-groups 6 --wall-seconds 1800
   ```

3. Independent ordinary replay and curation, bounded at 640 MiB / 600 s / 27 forwards
   and 128 MiB for the curator, with the canonical receipt and log names the recorder
   reads (`runs/<name>-job.json`, `runs/<name>.log`, `runs/<name>.stderr.log`):

   ```
   .venv/Scripts/python.exe scripts/probe_stream_errors.py \
     --run m3-identity-factor-positional-r1 --output runs/m3-identity-factor-positional-r1-errors-r1
   .venv/Scripts/python.exe scripts/record_stream_errors.py \
     --name m3-identity-factor-positional-r1-errors-r1
   ```

## Resource plan

The prototype read-out peaked 1,013,481,472 B (967 MiB) with 128 updates at capture 512.
This lever adds 65 query vectors at dimension 192 — about 12.5 k parameters and one more
attention sequence of length 65 per stage — so the expected peak is the same envelope.
Declared ceiling 1,152 MiB, under the engine's 2 GiB bounded-job cap (V-045) and under
min(physical/3, 8 GiB). If the profile exceeds it the ceiling is raised to the measured
need and the change is recorded; the lever is never shrunk to fit.

## Discipline

- Frozen runs, evidence directories and closed studies stay byte-identical; new
  artifacts use new names in the owned `m3-identity-factor-*` family.
- Default `separate` and the prototype mode itself must keep reproducing their tests;
  the position queries exist only in the declared positional mode.
- One heavy job at a time, absolute interpreter paths (V-030), canonical receipts
  (V-047), and every declared bound is the bound the guard enforces.
- Any defect this pass exposes is fixed at its root with a regression test and recorded
  in `docs/BUG_LOG.md` on a named receipt.
