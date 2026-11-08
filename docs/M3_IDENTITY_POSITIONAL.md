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

## Result: NEGATIVE — position information reaches the decision and still does not separate it

Predeclared at `aa5bcc0`; every receipt below postdates that commit and the run directory
was created afterwards.

| Receipt | Status | Numbers |
|---|---|---|
| `runs/m3-identity-factor-positional-r1-profile-r1-job.json` | passed | 2 steps, 37.522 s, 980,447,232 B under 1,152 MiB |
| `runs/m3-identity-factor-positional-r1-job.json` | passed | 128/128 steps, 238.495 s, 978,120,704 B, checkpoint `060ee296...` |
| `runs/m3-identity-factor-positional-r1-errors-r1-job.json` | passed | 27 forwards, 15.844 s, 566,697,984 B, weights unchanged, saved metrics reproduced |
| `runs/m3-identity-factor-positional-r1-errors-r1-curation.json` | passed | curator into `docs/evidence/m3-identity-factor-positional-r1-errors-r1/scope.json` |
| `runs/m3-identity-positional-tests-r1.json` | passed | 5 modules, `OK`, 5.993 s, 461,545,472 B under 512 MiB |

Measured over the run's own nine training probes and eighteen development images, with
the prototype run as the matched control:

| Term | Training | Development | Prototype control |
|---|---:|---:|---:|
| Queue `predicted_classes` | `EOS` x54 | `EOS` x90 | `EOS` x54 / x90 |
| Queue `distinct_predicted_classes` | **1** | **1** | 1 |
| `queue_constant_predictor` | **true** | **true** | true |
| Queue piece tokens exact | 0/54 | 0/90 | 0/54, 0/90 |
| `position_logit_spread_max` | **0.3627** | **0.5758** | not measured |
| `position_logit_spread_mean` | **0.3594** | **0.3347** | not measured |
| `position_axis_reaches_logits` | **true** | **true** | not measured |
| Hold `predicted_classes` | `T` x9 | `T` x12 | `L`x3/`O`x6, `L`x2/`O`x9/`S`x1 |
| Hold `distinct_predicted_classes` | **1** | **1** | 2, 3 |
| Hold exact | 0/9 | 3/12 | 3/9, 0/12 |
| Queue EOS exact | 9/9 | 15/15 | 9/9, 15/15 |
| Slot status exact | 54/54 | 90/90 | 54/54, 90/90 |
| Presence exact | 54/63 | 90/105 | 39/63, 48/105 |
| Exact roles (active + ghost) | 0/18 | 0/36 | 0/18, 0/36 |
| Mean geometry, cells | 3.5398 | 2.4024 | 3.5626, 2.3223 |
| Exact boards | 2/9 (both empty) | 0/18 | 2/9, 0/18 |

The predeclared rule gives **NEGATIVE**: queue discrimination needs >= 3 distinct
training classes and measures **1**, hold discrimination needs >= 3 and measures **1**.

### The position axis reaches the decision and the decision ignores it

This is the demonstration the lever was built to make possible, and it is recorded in the
run's own evidence rather than asserted: every supervised position receives different
logits from the same frame (max spread 0.3627 training, 0.5758 development, mean 0.3594
and 0.3347, `position_axis_reaches_logits` true in both splits) while the emitted class is
the same `EOS` at all 54 and all 90 supervised positions. Two regression tests hold the
two halves: `test_positional_readout_gives_every_position_its_own_image_query` shows two
positions of one frame differing and one position reading different frames differently,
and `test_position_varying_logits_are_recorded_without_forcing_separation` shows the
diagnostic carrying a positive spread while `distinct_predicted_classes` stays 1 and
`constant_predictor` stays true.

### Which constraint remains

The pass was designed to discriminate between three candidates, and the artifacts choose
one:

- **Read-out: ruled out.** Two independent conditioning axes are now wired in — image
  queries for the class side and per-position queries for the position side — and the
  position axis demonstrably reaches the logits. A read-out that cannot carry the
  information would show zero spread.
- **The data's independence of position: ruled out.** The targets are position-dependent
  by construction: 54 supervised training positions carry seven classes across six
  slots, and every position's target is the icon drawn at that slot.
- **The loss formulation: what remains.** The endpoint label is a class in the same
  8-way identity softmax and the same masked mean as the piece classes, and it is the
  only label that is always present and always at one known position: EOS is exact 9/9
  training and 15/15 development in this run, in the prototype run and in the control,
  while piece identity is never separated. Emitting `EOS` everywhere therefore keeps the
  endpoint, slot-status and (here) presence terms at or above control level
  (54/63 and 90/105, control 54/63 and 90/105) and pays only the piece term, which is
  spread over positions and cannot pull the shared class vectors apart on its own. The
  single-class solution is the cheapest descent that satisfies the terms that are already
  exactly predictable.

A second measured fact belongs in the record: the hold bank's two-class split in the
prototype run did **not** reproduce here. The bank is structurally identical, but this
run's RNG draws differ (the position queries are drawn after the class and hold banks),
and the head returned to the constant `T` with 0/9 exact training pieces. The earlier
separation was initialization-dependent rather than a robust mechanism.

The three development `exact_readable_components` are empty-queue frames (zero supervised
piece tokens) whose hold target equals the emitted constant `T`; they are the
class-frequency artifact V-048 describes, not component reading.

Structural cost: presence improves over the prototype run to control level, roles remain
0/18 and 0/36 (active and ghost all type errors), geometry is 3.5398 and 2.4024 against
the prototype's 3.5626 and 2.3223, and boards are 2/9 (both empty) and 0/18. The fit gate
stays **FALSE**: 2/9 and 0/18 exact boards, 0/18 and 0/36 exact roles, 0/9 and 0/18 exact
readable components.

### Defect exposed

V-050: the caller's shell cap killed the supervisor during the first replay, leaving a
`running` receipt, no `exit_code`, and an unusable partial output directory. The attempt
is preserved (`runs/attempts/...-job-killed.json` and the partial directory) and the
canonical receipt now holds the passing replay. Nothing was deleted or re-scored.

### Limits and next

Nine synthetic training probes, six development ancestries, one seed, 128 updates at
capacity 64, factor 4, capture 512: a fit/discrimination diagnostic on synthetic views,
with no real-image, complete-state, calibration or release claim. The earlier identity
verdicts stand unchanged.

Ranked next is the **loss formulation**, judged on this same discrimination rule:
separate the endpoint decision from piece identity (its own binary endpoint term, or a
piece softmax without the endpoint class) so that the always-predictable label stops
being the cheapest constant, and report the position-axis spread unchanged. Not ranked:
more class-side or position-side read-out conditioning (both closed here), or another
budget extension (the economics of the cheapest constant do not change with budget). The
owner-decision real-image chain and the FALSE fit gate are unchanged.
