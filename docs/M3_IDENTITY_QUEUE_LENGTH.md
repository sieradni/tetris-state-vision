# M3 identity lever — one queue-length decision (predeclaration)

Status: predeclared. This file is committed before any admission profile and before the
study; every receipt named below is created afterwards, so the commit ordering proves the
rule was not chosen from the result.

## The obstacle this addresses

`docs/M3_IDENTITY_ENDPOINT_LOSS.md` measured a NEGATIVE and named the mechanism precisely.
With the endpoint factored out of the piece softmax it still appeared at every supervised
position, because the endpoint term was charged as a **per-position binary question** at
all 63 training and 105 development known positions (9 and 15 positives, exactly 1/7), so a
constant "endpoint here" answer cost only 0.410 nats and preserved EOS 9/9, slot 54/54 and
presence 54/63 while `p(k) = softmax7(k)·(1 - p_end)` suppressed every piece class.

The endpoint is a per-sequence count, not a per-position property. This lever makes the
objective say that.

## The one thing this lever changes

`--component-endpoint-term length`. The positional read-out is unchanged (per-piece class
queries, per-position queries, four-query hold bank). The read-out's per-position scalar
becomes a **hazard**, and the endpoint becomes one decision per case:

```
p(length = k) = h_k · prod_{j<k} (1 - h_j)          k < capacity
p(length = capacity) = prod_{j<capacity} (1 - h_j)
```

which is a normalised distribution over `0..capacity`. The emitted 8-way
`queue_tokens` are derived from it, so every position remains a proper class
distribution and the ordinary argmax metrics and position-spread diagnostic keep their
meaning: at position `k`, `p(EOS) = p(length = k) / (1 - p(length < k))` and the seven
piece shares are `softmax7(k) · (1 - p(length <= k)) / (1 - p(length < k))`.

The objective becomes one 7-way piece cross-entropy at piece positions plus **one**
length cross-entropy per case (weight `1 x identity_factor` each, masked to cases whose
queue end is observed). The per-position endpoint binary cross-entropy is gone, and with
it the admissible "endpoint everywhere" answer: constant endpoint now means length 0 for
every case, which is wrong wherever the observed endpoint is elsewhere.

Everything else is held identical to `m3-identity-factor-endpoint-r1` and
`m3-identity-factor-positional-r1`: capacity 64, `--component-identity-factor 4`, capture
side 512, `stream-v1`, seed `20261026`, learning rate `.001`, evaluation/checkpoint 64,
six development groups, 128 updates, the same optimizer, every other loss term and weight,
the component targets and masks, and the positional read-out with its four-query hold bank.
The position-spread fields are reported unchanged.

## Control

`m3-identity-factor-endpoint-r1` (evidence
`docs/evidence/m3-identity-factor-endpoint-r1-errors-r1/`): training queue tokens 0/54 with
`EOS` x54 (distinct 1, constant true), hold 1/9 with `S` x7/`T` x2 (distinct 2, constant
false), position spread max 1.2338; development 0/90 `EOS` x90, hold 4/12 `S` x11/`T` x1,
spread max 1.2304. Structure: EOS 9/9 and 15/15, slot 54/54 and 90/90, presence 54/63 and
90/105, roles 3/9 and 9/9 training (0/18 and 0/18 development), boards 0/9 and 0/18,
geometry 4.0672 and 2.3110.

## Predeclared criterion — discrimination only, total over two conditions

Identical to the three previous runs. All numbers come from the independent 27-forward
replay of the new run at its final checkpoint, on that run's own nine training probes, from
the per-split `predicted_classes`, `distinct_predicted_classes` and `constant_predictor`
aggregates. Development is a diagnostic and is not part of the verdict. Exact-match counts
are never a criterion.

- **queue discriminates** := training `queue.distinct_predicted_classes >= 3` and
  `queue.constant_predictor` is false.
- **hold discriminates** := training `hold.distinct_predicted_classes >= 3`.

| Queue | Hold | Verdict |
|---|---|---|
| discriminates | discriminates | **POSITIVE** |
| discriminates | not | **PARTIAL (hold)** |
| not | discriminates | **PARTIAL (queue)** |
| not | not | **NEGATIVE** |

Structure, class frequency, board and role results are recorded as a cost and can never
upgrade a NEGATIVE.

A NEGATIVE stands only with its mechanism named from the run's own artifacts. It must
report the position-axis spread, report whether the earlier runs' initialization-dependent
hold separations reproduced, and state which candidate constraint remains — the read-out,
the loss masking, or the data's independence of position — with the measured artifact that
rules each other candidate in or out. It must be paired with a passing regression test
that the new objective is active (the length head is charged, the per-position endpoint
term is gone, mismatched joins are refused) and still fails to separate. It must also state
explicitly whether the class-side / position-side / loss-side disjunction is closed by this
lever or still open, and on what evidence.

This is a 9-image synthetic fit/discrimination diagnostic: no real-image, complete-state,
calibration or release claim follows.

## Receipts this pass must produce

1. Fresh admission on the new objective, 2 steps, `--profile-square`, wall 600 s, bounded
   at 1,152 MiB / 700 s / exactly 2 work items, receipt
   `runs/m3-identity-factor-length-r1-profile-r1-job.json`:

   ```
   .venv/Scripts/python.exe scripts/train_m2_stream.py --output runs/m3-identity-factor-length-r1-profile-r1 \
     --model accuracy --component-capacity 64 --component-identity-factor 4 \
     --component-identity-readout positional --component-endpoint-term length \
     --steps 2 --profile-square --wall-seconds 600 --development-groups 6
   ```

2. The matched study, admitted only by that profile, bounded at 1,152 MiB / 1,900 s /
   4,096 work items, receipt `runs/m3-identity-factor-length-r1-job.json`:

   ```
   .venv/Scripts/python.exe scripts/train_m2_stream.py --output runs/m3-identity-factor-length-r1 \
     --model accuracy --component-capacity 64 --component-identity-factor 4 \
     --component-identity-readout positional --component-endpoint-term length --steps 128 \
     --admission runs/m3-identity-factor-length-r1-profile-r1 --seed 20261026 \
     --learning-rate 0.001 --evaluate-every 64 --checkpoint-every 64 \
     --development-groups 6 --wall-seconds 1800
   ```

3. Independent replay and curation, bounded at 640 MiB / 600 s / 27 forwards and 128 MiB
   for the curator, with the canonical receipt and log names the recorder reads:

   ```
   .venv/Scripts/python.exe scripts/probe_stream_errors.py \
     --run m3-identity-factor-length-r1 --output runs/m3-identity-factor-length-r1-errors-r1
   .venv/Scripts/python.exe scripts/record_stream_errors.py \
     --name m3-identity-factor-length-r1-errors-r1
   ```

## Resource plan

The endpoint-loss read-out peaked 1,024,016,384 B (977 MiB) with 128 updates at capture
512. This lever removes 65 binary logits and adds 65 hazard logits plus one cumsum pair, so
the expected peak is the same envelope. Declared ceiling 1,152 MiB, under the engine's 2 GiB
bounded-job cap (V-045) and under min(physical/3, 8 GiB). If the profile exceeds it the
ceiling is raised to the measured need and the change is recorded; the lever is never
shrunk to fit.

## Discipline

- Frozen runs, evidence directories and closed studies stay byte-identical; new artifacts
  use new names in the owned `m3-identity-factor-*` family.
- The published `joint` objective and both earlier declared objectives keep reproducing
  their tests and their frozen receipts; the length parameterisation exists only in the
  declared mode, and the two declared endpoint objectives are mutually exclusive.
- The probe and the curator replay and record `arguments.component_endpoint_term`, so a
  mismatched objective join fails loudly instead of silently re-scoring a run (V-051).
- One heavy job at a time, absolute interpreter paths (V-030), canonical receipts (V-047),
  and any stage killed by a tooling cap is preserved and re-run to a passing canonical
  receipt (V-050).
- Any defect this pass exposes is fixed at its root with a regression test and recorded in
  `docs/BUG_LOG.md` on a named receipt.
