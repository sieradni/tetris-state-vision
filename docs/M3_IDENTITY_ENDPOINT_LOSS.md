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
- Any defect this pass exposes is fixed at its root with a regression test and recorded
  in `docs/BUG_LOG.md` on a named receipt.

## Result: NEGATIVE — the constant migrated into the endpoint term

Predeclared at `008be3d`; every receipt below postdates that commit and the run directory
was created afterwards.

| Receipt | Status | Numbers |
|---|---|---|
| `runs/m3-identity-factor-endpoint-r1-profile-r1-job.json` | passed | 2 steps, 35.1 s, 981,151,744 B under 1,152 MiB |
| `runs/m3-identity-factor-endpoint-r1-job.json` | passed | 128/128 steps, 256.6 s, 1,024,016,384 B, checkpoint `84faf263...` |
| `runs/m3-identity-factor-endpoint-r1-errors-r1-job.json` | passed | 27 forwards, 16.2 s, 565,972,992 B, weights unchanged, saved metrics reproduced |
| `runs/m3-identity-factor-endpoint-r1-errors-r1-curation.json` | passed | curator into `docs/evidence/m3-identity-factor-endpoint-r1-errors-r1/scope.json` |
| `runs/m3-identity-endpoint-tests-r2.json` | passed | 6 modules, `OK`, 88 tests, 3.4 s under 512 MiB |

Executed-name note: the admission actually ran under the run's own family name
(`runs/m3-identity-factor-endpoint-r1-profile-r1`, receipt `...-profile-r1-job.json`);
the command block above shows the `...-endpoint-profile-r1` spelling it was predeclared
with. Both artifacts are retained; the canonical receipts are the names in this table.

| Term | Training | Development | Positional control |
|---|---:|---:|---:|
| Queue `predicted_classes` | `EOS` x54 | `EOS` x90 | `EOS` x54 / x90 |
| Queue `distinct_predicted_classes` | **1** | **1** | 1 |
| `queue_constant_predictor` | **true** | **true** | true |
| Queue piece tokens exact | 0/54 | 0/90 | 0/54, 0/90 |
| `position_logit_spread_max` | **1.2338** | **1.2304** | 0.3627, 0.5758 |
| `position_logit_spread_mean` | **1.0166** | **0.5894** | 0.3594, 0.3347 |
| `position_axis_reaches_logits` | **true** | **true** | true |
| Hold `predicted_classes` | `S`x7, `T`x2 | `S`x11, `T`x1 | `T`x9 / `T`x12 |
| Hold `distinct_predicted_classes` | **2** | **2** | 1 |
| `hold_constant_predictor` | false | false | true |
| Hold exact | 1/9 | 4/12 | 0/9, 3/12 |
| Queue EOS exact | 9/9 | 15/15 | 9/9, 15/15 |
| Slot status exact | 54/54 | 90/90 | 54/54, 90/90 |
| Presence exact | 54/63 | 90/105 | 54/63, 90/105 |
| Roles exact (active, ghost) | 3/9, **9/9** | 0/18, 0/18 | 0/9, 0/9 / 0/18, 0/18 |
| Mean geometry, cells | 4.0672 | 2.3110 | 3.5398, 2.4024 |
| Exact boards | 0/9 (2 empty, 7 nonempty) | 0/18 | 2/9, 0/18 |

Predeclared rule: queue discrimination needs >= 3 distinct training classes and measures
**1**; hold needs >= 3 and measures **2**. Verdict **NEGATIVE**.

### The mechanism, pinned from the artifacts

Taking the endpoint out of the piece softmax did not remove the constant; it moved it.
The factorization makes the emitted class the product of a per-position binary endpoint
decision and a 7-way piece decision, and the endpoint term is charged at **every known
position** — 63 training and 105 development, of which only 9 and 15 are endpoints, a
positive rate of exactly 1/7 in both splits. A constant "endpoint here" answer therefore
costs only the binary entropy of that rate, `0.410` nats in both splits, and it preserves
every term that was already exact: EOS 9/9 and 15/15, slot 54/54 and 90/90, presence
54/63 and 90/105 — identical to the control. Because the two factors multiply, a high
endpoint probability suppresses every piece class at every position, so the argmax still
reports `EOS` at all 54 and all 90 supervised piece positions while the piece term stays
0/54 and 0/90.

The position axis is not the problem, and this run measures it more strongly than any
before: per-position logit spread rose from 0.3627 to **1.2338** training (mean 1.0166)
and from 0.5758 to **1.2304** development (mean 0.5894), with
`position_axis_reaches_logits` true in both splits. Position-varying information reaches
the decision more forcefully than under the previous read-out and is still collapsed.

### Which constraint remains

- **Read-out: ruled out.** Two conditioning axes are wired in and the measured spread is
  positive and larger than under either previous lever.
- **The data's independence of position: ruled out.** 54 training positions carry seven
  classes across six slots.
- **The loss: still the constraint, one level lower.** The endpoint is not a per-position
  property — it is a property of queue length — but the objective asks a per-position
  binary question and scores the constant answer as cheap. Removing it from the piece
  softmax was necessary and not sufficient; the admissible-constant structure survives in
  the endpoint term itself, and V-051 records that the run ledger cannot by itself show
  which component objective a run trained under.

So the class-side / position-side / loss-side disjunction is **not closed by this lever**.
What it does close is the narrower question of where the constant lives: not in the class
read-out, not in the position read-out, and not in the endpoint's membership of the piece
softmax, but in the endpoint being *scored per position while it is a per-sequence count*.

Hold: the head left a constant again in this run (`S`x7/`T`x2, distinct 2) and reached
1/9 exact training. Across the four identity runs the hold head has produced 2, 1, 1 and 2
distinct classes with three different class identities (`L`/`O`, `T`, `S`/`T`), so the
prior separations reappear only as an initialization-dependent two-class split and never
reach the predeclared three-class bar.

### Demonstration that the change was active

`test_the_separate_endpoint_term_charges_a_constant_piece_class` builds one image's
targets and a constant piece prediction, and asserts that under `endpoint_term="separate"`
the objective returns a `queue_endpoint` term, charges the 7-way piece term above 0.5
(so a constant piece class is charged as a piece class), and refuses both mismatched
joins — a factorized output under `joint`, and a joint output under `separate`.
`test_separate_endpoint_term_takes_the_endpoint_out_of_the_piece_softmax` asserts the
emitted columns are log-probabilities of the factorization
(`exp().sum(-1) == 1`, column 7 equals `logsigmoid(queue_endpoint)`), so the ordinary
argmax metrics and the position-spread diagnostic keep their meaning. The run's own
`result.json` records `component_endpoint_term: "separate"` and reproduces every saved
per-case loss exactly, which is only possible under the same objective: the joint and
factorized objectives produce different scalars.

Structural cost: ghost exactness rose to 9/9 training (from 0/9) and active to 3/9, but
locked boards are 0/9 and 0/18 (both training successes of the control, the empty boards,
are lost), roles are 0/18 on development, geometry is 4.0672 and 2.3110, and exact
readable components are 0/9 and 0/18. The fit gate stays **FALSE**.

### Limits and next

Nine synthetic training probes, six development ancestries, one seed, 128 updates at
capacity 64, factor 4, capture 512: a fit/discrimination diagnostic on synthetic views.
No real-image, complete-state, calibration or release claim, and no earlier verdict is
revised.

Ranked next, and now the only read-out/loss lever left standing, is the **queue length as
one global decision**: predict where the queue ends with a single distribution over
`0..capacity` (or one classification over positions) instead of asking a per-position
binary question, so "endpoint everywhere" is not in the hypothesis space, keeping the
class and position read-outs and the same discrimination rule. Not ranked: more class or
position conditioning, budget, or further re-weighting of the same per-position endpoint
term. The FALSE fit gate and the owner-decision real-image chain are unchanged.
