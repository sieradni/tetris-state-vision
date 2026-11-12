# M6 - the boundary sits above the labels: a declared sampling offset

**Status: predeclared before any receipt of this pass.** The diagnostic below runs first, and this
document fixes its reading rule and both branches' arms, gates and verdict vocabulary *before* either
exists. Nothing here may change after the first receipt.

## The diagnostic, and the rule fixed before it runs

**`scripts/measure_row_error.py`** re-analyses the four frozen M5 sweeps with no checkpoint and no
forward. For every column of every deciding board (28 boards x 10 columns) it takes the top-most decided
row and the top-most occupied row, both bottom-first, and reports `decided - label` in rows, at the
standing `0.0` boundary and at each arm's own best swept boundary, with the full distribution, the
fraction exactly equal and within one row, and the census of columns where one side is empty.

Declared reading rule, evaluated on the mean over columns at `0.0`:

1. **`systematic-row-offset`**: `|mean| >= 0.5` rows for **both** `control` and `wide`, with the same sign.
2. **`row-offset-arm-specific`**: `|mean| >= 0.5` for at least one of them but not both consistently.
3. **`no-row-offset`**: `|mean| < 0.5` for all four arms.

The reading is published as its own evidence scope (`docs/evidence/m5-row-error-r1/`) before the arm is
declared admissible, and the arm this pass runs follows from it.

### Diagnostic, measured (2026-11-12)

`runs/m5-rowerror-r1-job.json` (passed, 4 work items), curated into `docs/evidence/m5-row-error-r1/`.

| arm | mean error at `0.0` | mean at best | median at best | exact at best | within one row | label-only cols | decided-only cols |
|---|---|---|---|---|---|---|---|
| control | **+7.819** | +2.886 | +1 | 0.013 | 0.040 | 0 | 131 |
| wide | **+5.872** | +0.716 | 0 | 0.094 | 0.195 | 0 | 131 |
| de-meaned | +5.651 | +1.973 | +1 | 0.094 | 0.208 | 0 | 131 |
| column | +6.832 | +1.081 | 0 | 0.067 | 0.148 | 0 | 131 |

**Reading: `systematic-row-offset`, positive.** Every arm decides the stack taller than the label says.
At the standing boundary the bias is enormous - the head marks cells occupied far above any stack, which
is why 131 of 280 columns have no label at all yet still carry decisions - and at each arm's own best
boundary it is still `+0.716` to `+2.886` rows, with only 1.3% of control's columns exactly on the label.

This is a **geometry** finding, not a label-objective one: the label is not asking for a different
question, the read is looking one place and deciding about another. The arm that follows perturbs exactly
that: where the tap samples, held constant across the board.

## The arm: a declared constant sampling offset, in rows

**`--occupancy-offset-rows`** adds a declared constant `(0, rows)` to every cell's tap offset, in cell
units, rows bottom-first, so `+1` samples one row **above** the decoded cell and `-1` one row below. It
touches no weight, adds no tensor, and is applied after any deformation field, so the field keeps meaning
"what was learned" while the offset stays "what was declared".

Direction is declared from the diagnostic, not searched: the bias is **positive** (decisions above the
truth), so the arm that should move the boundary down is **`+1`**, because sampling above the cell puts
the feature on material that is empty more often and tempers the decision. The **`-1`** arm is measured
beside it as the declared control for the sign - if down also helps, the sign is not the mechanism and
the finding is that any displacement helps, which is worth knowing.

`--occupancy-offset-rows 0` is the published function exactly: the added constant is zero, so every
tensor and every sample is unchanged. That is the identity this lever can and does hold at zero; at
`+/-1` it is a declared change of sampling geometry and makes no identity claim.

## Arms - one catalogue, one budget, one population

| arm | `occupancy_deform` | `occupancy_offset_rows` | role |
|---|---|---|---|
| `control` | 0 | 0 | published read-out; must reproduce `0.8226433610683003` |
| `wide` | 2 | 0 | family best; must reproduce `0.8822623454135491` |
| `up` | 0 | **+1** | samples one row above the cell |
| `down` | 0 | **-1** | samples one row below the cell (sign control) |

Shared and unchanged: seed `20261026`, learning rate `.001`, `capture_max_side 512`, curriculum
`stream-v1`, 6 development groups, evaluation and checkpoint every 64/32, `--occupancy-detail 1`,
`--occupancy-detail-stencil 0`, `--occupancy-resolution 0`, `--occupancy-context 0`, `--occupancy-column 0`,
`--occupancy-placed 0`, `--occupancy-factor 4`, `--occupancy-positive-weight 4`, `--component-capacity 64`,
positional identity read-out, length endpoint term, 12 training groups for the sweeps, 64 steps, and the
published checkpoint as the start.

Declared envelope, unchanged from the last two blocks: study **1,792 MiB** / 64 work items / 900 s wall;
sweep profile **640 MiB** / 2 items; sweep **640 MiB** / 54 forwards; curation **1,900 MiB** / 9 items.
One catalogue digest across all four arms is required before anything is published.

## Gates, thresholds, verdict - unchanged

per-cell F1 gate `0.70`, ranking AUC `0.90`, ranking F1 `0.70`, attribution floor `0.05`, precision floor
`0.05`, 400 bootstrap draws at seed `20261026`, minimum detectable effect `2.802 x cluster SE`, 81-point
grid on `[-6, 6]`. Attribution is measured **against `control`** for the offset arm - the lever is the
sampling geometry, so the published geometry is the baseline - with the arm's relation to `wide` reported
beside it.

### Declared readings, in this order

Evaluated on the arm whose sign matches the diagnostic (`up`), with `down` reported beside it:

1. **`offset-is-the-lever`**: `up - control` best swept F1 `>= 0.05`, its AUC interval clear of zero, and
   its mean row error at `0.0` at least `0.5` rows closer to zero than control's.
2. **`offset-moves-without-ranking`**: the row error moves at least `0.5` rows toward zero but the
   attribution floor or the AUC interval is not cleared.
3. **`offset-class-fails`**: neither the boundary nor the ranking moves.

The reading must match the rule applied to the printed numbers; the record is refused otherwise. If
`down` outperforms `up`, the sign is reported as not being the mechanism, with both numbers printed.

**Fit gate**: reported on its own numbers and stays FALSE unless all 28 non-empty boards are exact.

## Limits, declared

One dataset, 36 deciding images, 28 non-empty boards, 12 clusters, one seed, one 64-step budget. The
diagnostic reads where decisions sit, not why; the arm changes a declared constant and nothing else. No
real-screenshot, complete-state, calibrated or quality claim is made, and no arm is declared better than
another outside this population and budget.

## Measured (2026-11-12) - the geometry hypothesis holds, and the declared sign was wrong

Four arms ran inside 1,792 MiB and completed 64 steps under one catalogue from the published checkpoint.
Control and wide reproduced the previous record exactly (`0.8226433610683003`, `0.8822623454135491`).
Evidence: `docs/evidence/m6-row-offset-r1/scope.json`, curated by `runs/m6-rowoffset-curation-r1-job.json`
(passed, 9 work items).

| quantity | control | wide | **up (+1)** | **down (-1)** |
|---|---|---|---|---|
| deciding AUC | 0.822643 | **0.882262** | 0.814109 | **0.870729** |
| F1 at `0.0` | 0.381640 | **0.385902** | 0.378286 | 0.345873 |
| best swept F1 | 0.412034 @ `0.90` | **0.514821** @ `1.95` | 0.393958 @ `1.20` | **0.469259** @ `2.70` |
| precision at `0.0` | 0.236325 | **0.240955** | 0.233675 | 0.209427 |
| out-of-fold F1 | 0.400399 | **0.498413** | 0.326910 | 0.368808 |
| mean row error at `0.0` | +7.819 | +5.872 | +8.101 | +8.879 |
| mean row error at own best | +2.886 | **+0.716** | +4.624 | **+1.342** |
| exact on the label row | 0.013 | 0.094 | 0.007 | 0.000 |
| study peak / wall | 1,021,194,240 B / 285.7 s | 1,047,932,928 B / 387.7 s | 1,020,035,072 B / 334.1 s | 1,054,584,832 B / 212.0 s |

**Declared reading on the predicted direction: `offset-class-fails`.** Sampling one row *above* the cell
does not correct the bias and does not help: `-0.018076` best F1 and `-0.008535` AUC against control, its
boundary error **worsens** (`+8.101` against `+7.819`, a `-0.282`-row move away from zero), and it is
`-0.120862` behind wide. The declared direction is falsified by its own arm.

**The geometry hypothesis is confirmed in the opposite direction.**

| contrast | best swept F1 | AUC | F1 at `0.0` | out-of-fold F1 | attribution |
|---|---|---|---|---|---|
| **down vs control** | **+0.057225** | **+0.048086** (+0.013560 .. +0.071269, MDE 0.043683) | -0.035767 | -0.031591 | **met** |
| up vs control | -0.018076 | -0.008535 (-0.018244 .. +0.000396, MDE 0.013671) | -0.003354 | -0.073489 | not met |
| up vs wide | -0.120862 | -0.068154 (-0.103914 .. -0.034951) | -0.007616 | -0.171503 | not met |
| wide vs control | +0.102787 | +0.059619 (+0.029411 .. +0.094463) | +0.004262 | +0.098014 | met |

Sampling one row **below** the decoded cell clears the `0.05` attribution floor (`+0.057225`), its AUC gain
is **resolvable** (`+0.048086`, interval clear of zero), and the mechanism shows in the diagnostic: at its
own best boundary the decision sits `+1.342` rows above the labels against control's `+2.886`, so more than
half of the row bias is removed by a constant that adds **no tensor at all**. It recovers `+0.048086` of the
wide arm's `+0.059619` AUC, i.e. **four fifths of the free field's gain is a one-row registration
error**, not a shifted-window exploit.

What it does not do, and the record says so: at the standing `0.0` boundary it is *worse* (`-0.035767` F1,
`-0.027` precision), its best boundary is `2.70` rather than `0.90`, and the out-of-fold number - the only
one that says a boundary chosen on other clusters transfers - falls `0.031591`. So the arm ranks better and
pays for it in calibration: the shift moves the whole response, including its operating point. It is still
`0.045562` best F1 and `0.011533` AUC behind wide, so the free field is not explained away.

**Verdict `not-cleared`** (per-cell F1, ranking and attribution all short of their gates on the declared
arm). **Fit gate: FALSE** - 0 of 28 exact boards on all four arms.

### What this pass establishes, and what it leaves

1. **The row bias is real and is a sampling geometry problem.** A constant, undeclared-in-weights one-row
   shift reproduces most of the best arm's ranking gain, which no supervision or read-out change managed.
2. **The direction is measurable, not guessable.** The diagnostic's sign was read as "decisions are too
   high, so look higher"; the truth is the opposite, and only the paired sign arm found it. Any future
   offset must be declared with both signs.
3. **Ranking is not calibration.** The gain comes with a threshold that moves from `0.90` to `2.70` and a
   transferred-boundary F1 below control's, so the next instrument is not another offset - it is the
   **operating point**: a declared, label-free calibration of the shifted read (per-column offset removal
   at scoring time, or training the boundary itself), measured on the same population. The offset trend
   also suggests `-2` is worth one declared arm, but only after the calibration question is answered,
   because a shift that keeps moving the threshold is not yet a usable read.

The \(`-1`\) offset is now the engine's best non-field configuration: same tensors, same budget, AUC
`0.822643 -> 0.870729` and best swept F1 `0.412034 -> 0.469259`.
