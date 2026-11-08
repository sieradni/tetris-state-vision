# M3 identity lever — the endpoint leaves the piece softmax (predeclaration)

Status: predeclared. This file is committed before any admission profile and before the
study; every receipt named below is created afterwards, so the commit ordering proves the
rule was not chosen from the result.

## The obstacle this addresses

Two conditioning levers are closed by measurement. `docs/M3_IDENTITY_PROTOTYPES.md` gave
the class side image-derived class vectors and still emitted one class at every supervised
position. `docs/M3_IDENTITY_POSITIONAL.md` gave the position side its own image queries and
measured the position axis reaching the decision (logit spread 0.3627 training, 0.5758
development) while the argmax stayed one class. In both runs, and in the four frozen runs
before them, the emitted class was `EOS`, whose own term is already exact (9/9 training,
15/15 development, and 8/9, 3/15 in the earliest control).

The reason is the objective, not the read-out: the endpoint is one class of the same
8-way masked piece softmax, and it is the only label that is always present and always
predictable. Emitting it at every supervised position keeps the endpoint, slot-status and
presence terms at or above control and pays only the position-spread piece term, which is
the cheapest descent available.

## The one thing this lever changes

`--component-endpoint-term separate`. The decoder keeps its positional read-out (per-piece
class queries plus per-position queries plus the four-query hold bank) and emits, in
addition, one **binary endpoint logit per position**, `queue_endpoint`. The emitted 8-way
`queue_tokens` become log-probabilities of the factorization

```
p(class k) = softmax7(piece logits)[k] * (1 - p_end)      k < 7
p(class 7) = p_end = sigmoid(queue_endpoint)
```

so the endpoint can never again be one of the alternatives that absorb a piece identity,
and `argmax` still reports the maximum a-posteriori class for the ordinary metrics.

The objective becomes two identity terms instead of one: a 7-way cross-entropy at piece
positions (weight `1 x identity_factor`), and the endpoint's own masked binary
cross-entropy over every known position (weight `1 x identity_factor`). Nothing else in
the objective or the model changes.

Everything else is held identical to `m3-identity-factor-positional-r1`: capacity 64,
`--component-identity-factor 4`, capture side 512, `stream-v1`, seed `20261026`, learning
rate `.001`, evaluation/checkpoint 64, six development groups, 128 updates, the same
optimizer, every other loss term and weight, the component targets and masks, and the
positional read-out with its four-query hold bank. The position-spread diagnostic is
reported unchanged so any movement on that axis is visible.

## Control

`m3-identity-factor-positional-r1` (evidence
`docs/evidence/m3-identity-factor-positional-r1-errors-r1/`): training queue tokens 0/54
with `EOS` x54 (distinct 1, constant true), hold 0/9 with `T` x9 (distinct 1), position
spread max 0.3627; development 0/90 `EOS` x90, hold 3/12 `T` x12, spread max 0.5758.
Structure: EOS 9/9 and 15/15, slot 54/54 and 90/90, presence 54/63 and 90/105, roles
0/18 and 0/36, boards 2/9 (both empty) and 0/18.

## Predeclared criterion — discrimination only, total over two conditions

Identical to the two previous runs. All numbers come from the independent 27-forward replay
of the new run at its final checkpoint, on that run's own nine training probes, from the
per-split `predicted_classes`, `distinct_predicted_classes` and `constant_predictor`
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

A NEGATIVE stands only with its mechanism named from the run's own artifacts, and it must
say which candidate constraint remains — the read-out, the loss masking, or the data's
independence of position — with the measured artifact that rules each other candidate in
or out. It must also report the position-axis spread and whether the prior runs'
initialization-dependent hold separations reproduced, and it must be paired with a passing
regression test that the loss change was active in the run's objective (the factorized
output contract, the two charged terms, and the refusal of a mismatched join).

This is a 9-image synthetic fit/discrimination diagnostic: no real-image, complete-state,
calibration or release claim follows.

## Receipts this pass must produce

1. Fresh admission on the new objective, 2 steps, `--profile-square`, wall 600 s, bounded
   at 1,152 MiB / 700 s / exactly 2 work items, receipt
   `runs/m3-identity-factor-endpoint-profile-r1-job.json`:

   ```
   .venv/Scripts/python.exe scripts/train_m2_stream.py --output runs/m3-identity-factor-endpoint-profile-r1 \
     --model accuracy --component-capacity 64 --component-identity-factor 4 \
     --component-identity-readout positional --component-endpoint-term separate \
     --steps 2 --profile-square --wall-seconds 600 --development-groups 6
   ```

2. The matched study, admitted only by that profile, bounded at 1,152 MiB / 1,900 s /
   4,096 work items, receipt `runs/m3-identity-factor-endpoint-r1-job.json`:

   ```
   .venv/Scripts/python.exe scripts/train_m2_stream.py --output runs/m3-identity-factor-endpoint-r1 \
     --model accuracy --component-capacity 64 --component-identity-factor 4 \
     --component-identity-readout positional --component-endpoint-term separate --steps 128 \
     --admission runs/m3-identity-factor-endpoint-profile-r1 --seed 20261026 \
     --learning-rate 0.001 --evaluate-every 64 --checkpoint-every 64 \
     --development-groups 6 --wall-seconds 1800
   ```

3. Independent replay and curation, bounded at 640 MiB / 600 s / 27 forwards and 128 MiB
   for the curator, with the canonical receipt and log names the recorder reads:

   ```
   .venv/Scripts/python.exe scripts/probe_stream_errors.py \
     --run m3-identity-factor-endpoint-r1 --output runs/m3-identity-factor-endpoint-r1-errors-r1
   .venv/Scripts/python.exe scripts/record_stream_errors.py \
     --name m3-identity-factor-endpoint-r1-errors-r1
   ```

## Resource plan

The positional read-out peaked 978,120,704 B (933 MiB) with 128 updates at capture 512.
This lever adds one scalar logit per position and replaces one 8-way term with an
equivalent 7-way plus binary pair, so the expected peak is the same envelope. Declared
ceiling 1,152 MiB, under the engine's 2 GiB bounded-job cap (V-045) and under
min(physical/3, 8 GiB). If the profile exceeds it the ceiling is raised to the measured
need and the change is recorded; the lever is never shrunk to fit.

## Discipline

- Frozen runs, evidence directories and closed studies stay byte-identical; new artifacts
  use new names in the owned `m3-identity-factor-*` family.
- The published `joint` objective and every earlier read-out must keep reproducing their
  tests and their frozen receipts; the factorization exists only in the declared mode.
- The probe and the curator must replay and record the objective the run trained with
  (`arguments.component_endpoint_term`), so a mismatched loss join fails loudly instead of
  silently re-scoring a run under a different objective.
- One heavy job at a time, absolute interpreter paths (V-030), canonical receipts (V-047),
  and any stage killed by a tooling cap is preserved and re-run to a passing canonical
  receipt (V-050).
- Any defect this pass exposes is fixed at its root with a regression test and recorded in
  `docs/BUG_LOG.md` on a named receipt.
