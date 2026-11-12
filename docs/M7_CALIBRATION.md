# M7 - can the shifted read carry a threshold fixed without labels?

**Status: predeclared before any receipt of this pass.** Every number it quotes as a reference comes from a
committed record (`docs/evidence/m6-row-offset-r1/scope.json`). Nothing here may change once a receipt
exists.

## The open question, restated with its receipts

M6 measured a declared one-row sampling offset and found the sign nobody predicted: sampling one row
**below** the decoded cell reaches deciding AUC **0.870729** (`+0.048086` over control, interval
`+0.013560 .. +0.071269`) and best swept F1 **0.469259** (`+0.057225`, clearing the attribution floor) with
**no tensor added**. It also found the cost: F1 at the standing `0.0` boundary falls to **0.345873**
(`-0.035767` against control's `0.381640`), the best boundary moves `0.90 -> 2.70`, and out-of-fold F1 falls
`0.031591`. The ranking gain is real; the operating point is not usable.

So the question this pass answers: **is the shifted read's decision boundary a determinable property of the
population's own statistics, or does it drift board by board?** If it is determinable without labels, the
offset becomes an operational read (`AUC 0.822643 -> 0.870729` with an F1 that beats control's); if it
drifts, the offset is a ranking fix that cannot be made operational at a fixed threshold, and that is the
result.

## The mechanism: a per-board threshold from the read's own logits

**Declared rule (`--calibration otsu-logits`).** For each board, take the 200 occupancy logits at the
labelled visible cells - the read's own output, no labels, no swept F1, no held-out boundary - and choose the
threshold that maximises the between-class variance of that board's logit distribution (Otsu's criterion,
candidates at the midpoints of consecutive distinct logits, ties to the lower candidate, and a constant
board chooses its own value so nothing is decided). A cell is occupied when its logit is strictly above its
board's threshold.

Why this mechanism, in two sentences. An additive misregistration shifts every logit of a board by
approximately the same amount, so a threshold that is defined relative to the board's own distribution is
**translation-equivariant** and absorbs exactly that shift - which is what the `0.90 -> 2.70` drift is -
while a fixed global constant cannot. And it uses only the tap's outputs, so it is a property of the read,
measurable on any arm, and it is the same rule for every arm, which is what makes "does the offset's
advantage survive calibration" a question with an answer.

It is a scoring rule, not a training change: it adds no parameter, touches no weight, and `calibration
none` is exactly the published scoring. Its identity at "zero" is therefore exact and testable: with the
rule disabled, every number must equal the sweep's own `0.0` operating point.

## Arms - one catalogue, one block

| arm | sampling | operating point | role |
|---|---|---|---|
| `control` | published (`0` offset) | `none`, then `otsu-logits` | must reproduce `0.8226433610683003` / `0.41203400915631133` |
| `wide` | deformation field | `none`, then `otsu-logits` | must reproduce `0.8822623454135491` / `0.514820592823713` |
| `down` | `--occupancy-offset-rows -1` | `none`, then `otsu-logits` | the arm whose operating point moved |

Three 64-step continuations are re-captured in one catalogue (the catalogue moves because this pass adds
scripts, V-112), so every arm is measured under the same digest as the calibration rule that reads it.
Shared settings, envelope, gates, grid, population and bootstrap are exactly M6's: detail tap `1`, stencil
`0`, resolution `0`, context `0`, column `0`, placed `0`, `factor 4`, `positive weight 4`, capacity `64`,
positional identity, length endpoint, `512` capture, seed `20261026`, rate `.001`, 12 sweep groups, 1,792
MiB studies, 640 MiB sweeps, 1,900 MiB curation, 81-point grid on `[-6, 6]`.

**The `-2` offset arm is omitted, deliberately.** It needs a fourth 64-step training round, which the
instruction forbids in this harness, so it is named here as unmeasured rather than guessed at; there is no
receipt for an arm that was not run, and this declaration says so before the block starts.

## What is read, and how

For every arm and both operating rules: deciding AUC, F1 at `0.0`, F1 at the arm's **own label-free
thresholds**, precision, recall, the distribution of per-board thresholds (mean, sd, min, max), the fraction
of boards whose threshold lies within `0.5` of the arm's label-chosen best, and the out-of-fold F1 from the
standing sweep. AUC is **unchanged by construction** under a scoring rule - the ranking is not touched - and
that is declared here so the record cannot present it as an improvement.

**Held-out honesty.** Calibration is a scoring rule defined per board and computed from that board's own
logits at evaluation time, so it uses no label and no cross-board statistic; the record reports the
out-of-fold F1 beside it, which is the number that would suffer if the rule were fitted to the population.

## Gates, thresholds and verdict vocabulary - fixed now

Unchanged: per-cell F1 `0.70`, ranking AUC `0.90`, ranking F1 `0.70`, attribution floor `0.05`, 400
bootstrap draws at `20261026`, MDE `2.802 x cluster SE`.

Reference points declared before the receipts (from the M6 record): control `AUC 0.822643`, `F1@0
0.381640`, best `0.412034`; wide `AUC 0.882262`, `F1@0 0.385902`, best `0.514821`; down `AUC 0.870729`,
`F1@0 0.345873`, best `0.469259` at `2.70`.

### Primary reading - is the operating point determinable without labels?

On the `down` arm, with the label-free rule:

1. **`calibration-recovers-the-operating-point`**: calibrated F1 `>= 0.381640 + 0.05 = 0.431640` **and**
   `>= 0.85 x 0.469259 = 0.398870` (at least 85% of the label-chosen gain over control).
2. **`calibration-moves-without-recovering`**: calibrated F1 `> 0.381640` but either clause above is missed.
3. **`calibration-fails`**: calibrated F1 `<= 0.381640`. The offset cannot be made operational at a
   label-free threshold, and the drift is the finding.

### Secondary reading - was the offset's advantage a ranking effect or a threshold effect?

Comparing the two calibrated arms at their own label-free thresholds (`down_cal - control_cal`):

- **`offset-advantage-survives-calibration`** if the calibrated F1 delta `>= 0.05`;
- **`offset-advantage-was-a-threshold-effect`** if it is below `0.05`.

Both readings are computed from the printed numbers by the curator and must match the rule, or the record is
refused. The fit gate is reported on exact boards and stays FALSE until all 28 non-empty boards are exact.

## Limits, declared

One dataset, 36 deciding images, 28 non-empty boards, 12 clusters, one seed, one 64-step budget, one
architecture. A scoring rule cannot change ranking, so any AUC this pass prints for the calibrated arms is
the uncalibrated arm's. Nothing here is a real-screenshot, complete-state, calibrated-in-the-deployment
sense or quality claim.

## Measured

*(appended after the receipts exist - see `docs/evidence/m7-calibration-r1/`)*
