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

## Measured

*(appended after the receipts exist - see `docs/evidence/m6-row-offset-r1/`)*
