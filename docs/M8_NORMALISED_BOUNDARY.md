# M8 - can training place the boundary the post-hoc rule could not?

**Status: predeclared before any receipt of this pass.** Every reference number comes from a committed record
(`docs/evidence/m7-calibration-r1/scope.json` and `docs/evidence/m6-row-offset-r1/scope.json`). Nothing here may
change once a receipt exists.

## Where the last pass left it

M7 measured three arms at a per-board Otsu threshold derived from each arm's own occupancy logits - no labels, no
swept F1, translation-equivariant by construction. The `-1` row offset, which buys `+0.048086` deciding AUC over
control with no tensor added, reached F1 **0.380898** at that threshold: `0.000742` below control's own
`0.381640`. Reading `calibration-fails`, secondary `offset-advantage-was-a-threshold-effect`. The mechanism was
diagnosable and is the starting point of this pass: cells are `11.9%` occupied and Otsu maximises
*between-class variance*, so every arm landed at recall `0.99` / precision `0.24` - a saturated detector - and
the rule even cost the best arm `-0.027668` F1.

So the ranking information is there and a **post-hoc rule cannot convert it**. This pass stops choosing a
boundary after training and makes the parameterisation of the read do the work.

## The mechanism: the read is trained in normalised space

**Declared lever (`--occupancy-normalise 1`).** Inside the model, after every occupancy tap and head has
contributed and before the map leaves the forward pass, the board's occupancy logits are z-scored over the
board's own **visible** cells:

```
z = (logit - mean_visible) / sd_visible          (sd clamped at 1e-6, unbiased=False)
```

Three properties, declared rather than claimed:

1. It is **inside the model**, so the tensor the objective differentiates and the tensor a scorer thresholds at
   `0.0` are *the same tensor*. This is what makes `0.0` the operating point **by construction**: there is no
   stage after training where a boundary is selected, and a checkpoint carries the parameterisation in its
   config, so a consumer cannot silently apply a different one.
2. It is **label-free and parameter-free**. The statistics are the board's own logits at its own visible cells:
   no occupancy label, no score, no prevalence estimate, no cross-board statistic, and no tensor, buffer or
   state key is added. `0` is **exactly** the published function, which is testable bit-for-bit.
3. It is **translation-equivariant in the sense that matters here**: a board-wide additive shift - the
   registration error the declared row offset exposed - cancels exactly, so the head cannot spend its capacity
   re-centring the read. What the head must now do is separate occupied from empty *by spread*, and a spread is
   what a `0.0` boundary with a unit divisor actually controls.

It is a change to what is supervised - the geometry of the objective's output - not a flag that rewrites a
result, and not a hyperparameter. There is exactly one arm and no sweep.

## Arms - one catalogue, one block

| arm | parameterisation | role |
|---|---|---|
| `m8-normbound-control-r1` | published (`--occupancy-normalise 0`) | must reproduce the recorded `0.8226433610683003` and `0.381640` |
| `m8-normbound-norm-r1` | `--occupancy-normalise 1` | the arm this pass measures |

Both are 64-step continuations from `runs/m3-percell-balance-r1`, re-captured in one catalogue (the catalogue
moves because this pass edits the package), with the M7 envelope unchanged: capacity `64`, positional identity,
length endpoint, detail tap `1`, stencil `0`, resolution `0`, context `0`, column `0`, offset rows `0`, placed
`0`, factor `4`, positive weight `4`, `512` capture, seed `20261026`, rate `.001`, 12 sweep groups, 1,792 MiB
studies with 64 work items, 640 MiB sweeps with 54 forwards, 1,900 MiB curation, 81-point grid on `[-6, 6]`.

**Composition is refused, deliberately.** The trainer rejects `--occupancy-normalise 1` beside any lever that
changes *what* is read or *where* it is sampled (context, stencil, resolution, deform, placed, column, row
offset, lattice-free). The arm therefore answers one question; normalising the offset arm is the follow-up, not
this pass.

## What is read, and how

**Primary: plain-binary per-cell F1 at the standing `0.0` boundary** - the sweep's own `per_cell_f1`, which is
exactly "threshold at zero and score the cells" with no calibration, no sweep and no post-processing. It is the
metric a boundary controls directly, and the one M7's rule failed to move usefully.

**Secondary: deciding AUC**, which the parameterisation can damage even though it cannot inflate it, so a
degradation is a real cost and is reported as one. Alongside both: best swept F1 (the label-chosen ceiling, for
context only - it is not the metric this pass declares), out-of-fold F1, the row-error diagnostic, exact-board
count and per-board F1.

## Declared readings - fixed now

The primary delta is `norm F1@0 - control F1@0`, with control required to reproduce `0.381640` exactly first.

1. **`normalisation-clears-the-boundary`**: delta `>= 0.05` (the attribution floor). The parameterisation
   converts the read's ranking into an operating point that clears M7's `0.431640` bar as well.
2. **`normalisation-moves-without-clearing`**: `0 < delta < 0.05`. The objective moved the boundary; it did not
   move it far enough to matter.
3. **`normalisation-fails`**: delta `<= 0`. Training in normalised space does not place the boundary either, and
   the drift is not reachable from the parameterisation.

Secondary reading on AUC, declared with its tolerance so it cannot be read post hoc: **`ranking-preserved`** if
`norm AUC >= control AUC - 0.02`, otherwise **`ranking-degraded`**. The tolerance is half of the smallest gain
this project has ever resolved (`0.048086` at MDE `0.043683`), so anything past it is a loss larger than any
effect this harness has been able to detect in the other direction.

Gates unchanged: per-cell F1 `0.70`, ranking AUC `0.90`, ranking F1 `0.70`, attribution floor `0.05`, 400
bootstrap draws at `20261026`, MDE `2.802 x cluster SE`. Verdict vocabulary unchanged (`cleared` /
`not-cleared`, all three conditions). The fit gate is reported on exact boards and stays FALSE until all 28
non-empty boards are exact.

## If it fails

Declared now, so a negative result cannot drift into a search: the next structural idea is to **supervise the
boundary directly** rather than infer it - a declared within-board ranking term (a hinge on the margin between
each board's occupied and empty logits, weight fixed before the receipt) so the objective is told what a margin
is, instead of the parameterisation only fixing where zero sits. That is a change to what is supervised. A
threshold or weight sweep is not, and will not be offered as one.

## Limits, declared

One dataset, 36 deciding images, 28 non-empty boards, 12 clusters, one seed, one 64-step budget, one
architecture, no held-out population. Normalisation does not add information - it cannot: it is a monotone
per-board transform of the read - so any AUC this pass reports higher than control's would be the *parameterisation*
changing what was learned, not new information entering, and that is exactly what is being tested.
Nothing here is a real-screenshot, complete-state or quality claim.

## Measured

Receipts: `docs/evidence/m8-normalised-boundary-r1/scope.json` (curation, 1,900 MiB, receipt
`runs/m8-normbound-curation-inner-r1.json`), studies `runs/m8-normbound-<arm>-r1/` (64 steps, 1,792 MiB), sweeps
`runs/m8-normbound-<arm>-r1-sweep-r1/` (54 forwards, 640 MiB). One catalogue (`efa02b84...`), one block, both
studies `complete` at 64 steps (`161.2 s` and `148.4 s`).

**Reproduction first.** Control came back bit-identical in this catalogue: AUC `0.8226433610683003` and
`F1@0 0.381640`, exact string equality on both, which is what makes the two-arm comparison honest.

| arm | AUC | plain-binary F1 at `0.0` | best swept F1 | precision / recall at `0.0` | out-of-fold F1 | row error at `0.0` |
|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 0.2363 / 0.9910 | 0.400399 | +7.819 |
| normalised | 0.788463 | **0.384661** | 0.393300 @ `0.30` | 0.2399 / 0.9700 | 0.393367 | +5.134 |

Operating counts at `0.0`: control `661 TP / 2,136 FP / 6 FN`, normalised `647 / 2,050 / 20`, out of `667`
occupied and `4,933` empty cells.

### Reading: `normalisation-moves-without-clearing`, and `ranking-degraded`

Primary delta `+0.003021` (bar `+0.05`), AUC delta `-0.034181` (tolerance `-0.02`), best swept F1 `-0.018734`,
out-of-fold `-0.007032`. Reading `normalisation-moves-without-clearing`, secondary `ranking-degraded`, verdict
`not-cleared` on all three conditions, fit gate **FALSE** (`0` of `28`). The parameterisation did not place an
operating point, and it cost ranking to try.

### Why it could not have worked - the ceiling is geometric, not a training failure

This is the part worth carrying forward. A per-board z-score gives the visible logits mean `0` and unit spread,
so the `0.0` boundary necessarily cuts the board near its **centre of mass**, and the share of cells above it is
approximately one half whatever the head does: measured `2,697 / 5,600 = 48.2%`. At `11.9%` prevalence a cut
that admits `48%` of cells has precision no better than `0.119 / 0.482 = 0.247` even with perfect recall, so F1
is bounded near `2 x 0.247 x 1 / (0.247 + 1) = 0.396`. The arm measured precision `0.2399` and F1 `0.384661` -
**it reached the ceiling of its own parameterisation**. The `+0.003021` is that ceiling being approached, not a
lever to build on: standardising the output makes zero a *median*, and a median is not an operating point when
`88%` of the cells are empty.

The cost is equally informative. AUC fell `0.034181` and out-of-fold F1 fell `0.007032`: dividing out the
board's global scale removed magnitude information the ranking was using (the M6 row offset works precisely
because the shifted read's logits are systematically larger, and this transform deletes exactly that). The one
thing that improved is the row diagnostic, `+7.819 -> +5.134` rows, which is the same fact from the other side -
fewer cells are decided above the stack once the read cannot sit high as a whole.

So the answer to this pass's question is **no**: training in per-board normalised space cannot place the
boundary, because the normalisation fixes where zero sits (the median) more tightly than training can move it.
Choosing a different label-free statistic does not rescue this - subtracting a low quantile pushes the boundary
the wrong way, and subtracting a quantile matched to `11.9%` would inject a prevalence prior, which this pass
declared it would not do.

### Next structural step, named at the same specificity

Supervise the boundary **directly** rather than infer it, on the *published* (unnormalised) read so ranking is
not damaged: a declared within-board ranking term - a hinge on the margin between each board's lowest occupied
logit and its highest empty logit, `max(0, margin - (lowest_occupied - highest_empty))` with `margin` and its
weight fixed before the receipt - so the objective is told what separation is instead of being handed a location
for zero. That is a change to what is supervised; a threshold or weight sweep is not, and this record does not
offer one.

What this pass does **not** touch: the `-1` row offset's `+0.048086` AUC gain stands, and the M7 conclusion that
no imbalance-blind post-hoc rule converts it stands. Between them, the operating point is now blocked from both
cheap directions - a rule chosen after training (M7) and a parameterisation chosen instead of a rule (M8) - which
is why the next arm has to put the margin into the loss.

Defect of this pass: V-129 (the curator's first draft would have recorded a Wilson interval where a point recall
was declared).
