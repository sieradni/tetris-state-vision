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

Receipts: `docs/evidence/m7-calibration-r1/scope.json` (curation, 1,900 MiB, receipt
`runs/m7-calibration-curation-inner-r2.json`), instrument `runs/m7-calibration-r1/result.json` (3 work items,
640 MiB), sweeps `runs/m7-calibration-<arm>-r1-sweep-r1/` (54 forwards, 640 MiB), studies
`runs/m7-calibration-<arm>-r1/` (64 steps, 1,792 MiB). One catalogue (`771d7fc8...`), one block.

**Reproduction first.** Both published arms came back bit-identical, so the block is the frame the earlier
numbers were measured in: control `0.8226433610683003` and best `0.41203400915631133`, wide
`0.8822623454135491` and best `0.514820592823713`, and the offset arm reproduces its own M6 record exactly
(`0.8707292410960544`, best `0.46925858951175414`).

### The two operating rules, all three arms

| arm | AUC | F1 at `0.0` | best swept F1 | F1 at its own label-free thresholds | threshold mean (sd) | out-of-fold F1 |
|---|---|---|---|---|---|---|
| `control` | 0.822643 | 0.381640 | 0.412034 @ `0.90` | **0.381117** | `-0.188` (0.140) | 0.400399 |
| `wide` | 0.882262 | 0.385902 | 0.514821 @ `1.95` | **0.358234** | `-0.362` (0.360) | 0.498413 |
| `down` (`-1`) | 0.870729 | 0.345873 | 0.469259 @ `2.70` | **0.380898** | `1.145` (0.252) | 0.368808 |

AUC is the uncalibrated arm's by construction and is printed only so no reader mistakes a scoring rule for a
ranking change. Row error is the diagnostic of `docs/evidence/m5-row-error-r1/`, recomputed at each rule: at
`0.0` control reads `+7.819` rows above its label row, wide `+5.872`, `down` `+8.879`; at the label-free
threshold control `+8.020`, wide `+6.624`, `down` `+5.523` (still above, never corrected).

### Reading: `calibration-fails`, and `offset-advantage-was-a-threshold-effect`

The declaration fixed the bar at `0.381640` (control's own `F1@0`) and, for full credit, `0.431640` with
`0.398870` recovery. The shifted read reaches **0.380898** - it lifts itself `+0.035025` off its own `0.0`
boundary, which is most of the `0.087386` the label-chosen boundary found, but it lands `0.000742` **below**
control and so misses the bar. `down_cal - control_cal` is `-0.000219`, nowhere near the `0.05` attribution
floor: at a boundary the read can supply, the offset is not ahead of control at all. Both clauses are computed
by the curator from the printed counters, not asserted in prose.

So on the predeclared rule: the operating point the offset needs is **not** delivered by the read's own
statistics, and the `+0.048086` AUC gain does not survive the move from the label-chosen boundary to a
label-free one. That is the answer to this pass's question as it was posed.

### Why the rule failed - the honest part, and it is not the read's fault alone

Every arm at its own label-free threshold sits at **recall `0.992504` (control) / `0.967016` (wide) /
0.992504` (down) and precision `0.235839` / `0.219836` / `0.235671`** - the calibrated rule degenerates to
"almost every cell is occupied", recovering 662 of 667 cells while admitting roughly 2,150 false positives out
of 4,933 empty cells. The reason is structural: cells are `11.9%` occupied, and Otsu maximises *between-class
variance*, which under that base rate is maximised by a cut that separates the bulk of the empty logits from a
tail containing nearly all occupied ones - not by the cut that maximises F1. The evidence is the thresholds
themselves: `-0.188` for control against its label-best `0.90`, `-0.362` for wide against `1.95`. The rule is
translation-equivariant, as declared, and it did absorb the shift - `down`'s threshold sits at `1.145`, by far
the highest of the three - but it absorbed it into a saturated detector.

Consequence for how this result may be cited: **the offset's boundary is not recovered by an imbalance-blind
label-free rule**, and the wide arm shows the rule is not free even for an arm that does not need it
(`-0.027668` F1 against its own `0.0` boundary). It is *not* evidence that no label-free statistic of the read
can carry the boundary; that would need a rule that is aware of the base rate, and this pass did not test one.

### Where this leaves the engine

The ranking finding survives untouched: the `-1` row offset buys `+0.048086` AUC (interval
`+0.013560 .. +0.071269`, MDE `0.043683`) with no tensor added, and it is the only lever measured so far that
moves deciding AUC on this population. What it does not buy is an *operating point*, and this pass shows the
cheap post-hoc route to one does not work: the boundary has to come from something other than the read's own
marginal distribution - either supervision inside training (a head trained in a normalised parameterisation,
so the `0.0` boundary is a learned quantity) or a rule that models the occupied rate. Neither is measured
here.

Verdict `not-cleared` on every condition (per-cell F1 `0.70`, ranking AUC `0.90`, attribution `0.05`), fit gate
**FALSE**, `0` of `28` exact non-empty boards.

Defects of this pass: V-126 (the run family was not declared in `record_stream_run.OWNED_RUN_PATTERN`, so every
sweep and the instrument were refused after the studies had already run), V-127 (the curator was invoked twice
without its required `--receipt` and then with a forward-slash interpreter path, both refused), V-128 (the
`-2` arm stays unmeasured and the frame is one seed, one 64-step budget, one architecture).
