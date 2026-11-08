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
- Any defect this pass exposes is fixed at its root with a regression test and recorded
  in `docs/BUG_LOG.md` on a named receipt.

## Result: NEGATIVE — the disjunction is closed at this budget

Predeclared at `d2895de`; every receipt below postdates that commit.

| Receipt | Status | Numbers |
|---|---|---|
| `runs/m3-identity-factor-length-r1-profile-r1-job.json` | passed | 2 steps, 34.2 s, 981,618,688 B under 1,152 MiB |
| `runs/m3-identity-factor-length-r1-job.json` | passed | 128/128 steps, 228.0 s, 1,011,888,128 B, checkpoint `8042c288...` |
| `runs/m3-identity-factor-length-r1-errors-r1-job.json` | passed | 27 forwards, 16.2 s, 602,841,088 B, weights unchanged, saved metrics reproduced under `component_endpoint_term: length` |
| `runs/m3-identity-factor-length-r1-errors-r1-curation.json` | passed | curator into `docs/evidence/m3-identity-factor-length-r1-errors-r1/scope.json` |
| `runs/m3-identity-length-tests-r3.json` | passed | 6 modules, `OK`, 92 tests |

Two refusals are part of the record, both preserved rather than re-scored:
`runs/attempts/m3-identity-factor-length-r1-attempt1` with its failed receipt (the
admission gate refused the new mode, V-052), and
`runs/attempts/m3-identity-factor-length-r1-attempt2` (after the gate fix the study was
correctly refused because the superseded profile's source catalogue predated that fix, so
the profile was re-run against the fixed tree; it is kept as
`runs/attempts/...-profile-r1-superseded`).

| Term | Training | Development | Endpoint-loss control |
|---|---:|---:|---:|
| Queue `predicted_classes` | `EOS` x54 | `EOS` x90 | `EOS` x54 / x90 |
| Queue `distinct_predicted_classes` | **1** | **1** | 1 |
| `queue_constant_predictor` | **true** | **true** | true |
| Queue piece tokens exact | 0/54 | 0/90 | 0/54, 0/90 |
| `position_logit_spread_max` | **0.0877** | **0.0838** | 1.2338, 1.2304 |
| `position_logit_spread_mean` | **0.0876** | **0.0487** | 1.0166, 0.5894 |
| `position_axis_reaches_logits` | true | true | true |
| Hold `predicted_classes` | `J` x9 | `J` x12 | `S`x7/`T`x2, `S`x11/`T`x1 |
| `hold_distinct_predicted_classes` | **1** | **1** | 2, 2 |
| Hold exact | 3/9 | 3/12 | 1/9, 4/12 |
| Queue EOS exact | 9/9 | 15/15 | 9/9, 15/15 |
| Slot status exact | 54/54 | 90/90 | 54/54, 90/90 |
| Presence exact | **9/63** | **15/105** | 54/63, 90/105 |
| Roles exact (active, ghost) | 0/9, 0/9 | 0/18, 0/18 | 3/9, 9/9 / 0/18, 0/18 |
| Mean geometry, cells | 3.6709 | 3.1173 | 4.0672, 2.3110 |
| Exact boards | 0/9 (0 empty, 0 nonempty) | 0/4 empty, **1/14 nonempty** | 0/9, 0/18 |

Predeclared rule: queue discrimination needs >= 3 distinct training classes and measures
**1**; hold needs >= 3 and measures **1**. Verdict **NEGATIVE**.

### The mechanism, pinned from the artifacts

Replacing 63 per-position endpoint questions with one length decision removed the last
per-position pressure the identity path had. The measured consequence is a representation
collapse: the per-position logit spread fell from 1.2338 and 1.2304 to **0.0877** and
**0.0838** (means 0.0876 and 0.0487), an order of magnitude smaller than the run before,
so the positional read-outs became almost identical across positions. With one global
length answer available, the objective no longer asks any position to differ, and the head
stops making them differ. The emitted class is `EOS` at all 54 and all 90 supervised piece
positions while the endpoint and slot terms stay exact (9/9, 15/15, 54/54, 90/90) and the
piece term stays 0/54 and 0/90.

That is also visible in the terms that were *not* part of this lever: presence, a per-position
term the identity path shares the position read-out with, collapsed from 54/63 and 90/105
to **9/63 and 15/105**. The identity head did not merely fail to use position
information; the objective stopped requiring it anywhere, and the shared read-out obeyed.
The hold bank went back to a single class (`J` x9 and x12) with 3/9 and 3/12 exact — the
fourth of four runs to produce a different constant or a two-class split, never three.

### Which constraint remains — and why the disjunction is closed

Measured, ruling one candidate in and the other two out:

- **Read-out: ruled out.** Three read-out variants have now been run: image-derived class
  queries, per-position image queries, and both together under two endpoint objectives.
  The position axis demonstrably carries information (spread 0.3627 in the prototype run,
  1.2338 in the endpoint run) and the class vectors are image-dependent. This run shows
  the collapse tracks the *objective*, not the read-out: change the loss and the same
  architecture's spread falls by a factor of 14.
- **The data's independence of position: ruled out.** 54 training positions carry seven
  classes across six slots, and presence — the same positions, a coarser target — was at
  54/63 when the objective asked per position, so the data supports per-position answers.
- **The loss masking: what remains, in its final form.** Across four objectives the only
  invariant is that a single class everywhere is an admissible minimiser: the piece term is
  a masked mean over positions whose optimum under a position-independent read-out is the
  marginal class distribution, and neither the endpoint class, its own binary term, nor one
  length decision per case changes that. The identity term never becomes a constraint that
  *requires* a per-position answer.

The class-side / position-side / loss-side disjunction is therefore **CLOSED on evidence**:
all three sides have been measured at the fixed matched budget (capacity 64, factor 4,
capture 512, seed `20261026`, 128 updates), each was predeclared and receipted, each
returned a NEGATIVE, and the last one names the residual mechanism. No further variant of
this disjunction is ranked: repeating any of the three at this budget cannot change the
cheapest-constant argument, and changing the budget would not change the objective's shape.

### Demonstration that the objective was active

`test_the_length_objective_charges_one_decision_and_no_position_endpoint` asserts that the
length objective returns a `queue_length` term, that no `queue_endpoint` term exists, that
the 7-way piece term still charges a constant piece class above 0.5, and that all three
mismatched joins are refused (joint over a length output, length over a per-position
endpoint output, length over a missing distribution).
`test_the_length_head_replaces_the_per_position_endpoint_question` asserts the decoder emits
one length distribution summing to one and no `queue_endpoint` key, with every emitted
position still a proper distribution. `test_every_declared_endpoint_objective_is_accepted_by_the_driver_and_the_gate`
binds the driver, the model configuration and the admission gate to the single
declaration (V-052). The run's `result.json` records `component_endpoint_term: "length"` and
reproduces every saved per-case loss exactly.

### Cost, limits and next

Structure got worse, not better, and none of it can upgrade the negative: geometry 3.6709
and 3.1173, roles 0/9, 0/9 and 0/18, 0/18, exact readable components 0/9 and 0/18. The one
development nonempty board (1/14) and the 3/9, 3/12 hold exact are class-frequency
artefacts of a constant, not reads (V-048). The fit gate remains **FALSE**.

Nine synthetic training probes, six ancestries, one seed, 128 updates at capacity 64: a
fit/discrimination diagnostic on synthetic views, with no real-image, complete-state,
calibration or release claim, and no revision of any earlier verdict.

With the identity disjunction closed, the ranked next lever is the **fit gate itself** —
the locked-occupancy and role read-out that stands at 0/9 and 0/18 exact boards and 0/18
roles, unstable across runs (ghost 9/9 in one run, 0/9 here) — because an identity head
sitting on a representation that does not register the board cannot be expected to read a
seven-class queue from it. This is a statement about ranking, not a measured claim. Not
ranked: any further class-side, position-side or endpoint-loss variant, or a budget
extension of the same objective. The owner-decision real-image chain is unchanged.
