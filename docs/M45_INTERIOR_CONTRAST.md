# M45 - interior contrast: reading a cell against its own surroundings

Status: **predeclared before any scoring run.** The change, its matched control, every bar element, its prediction
and the pre-committed reading of every branch are below. The measured section is added after the run and is not part
of the declaration.

## The axis, and the row set this pass reconciles against

M44 closed the pixel edge ring **harmful**, and its error mix located the cause precisely: true positives (`209`)
and false negatives (`5`) were bit-identical to the control's, so recall did not move and the whole loss was `251`
extra false positives at a slightly higher median margin. The mechanism is confound, not capacity: above a stack a
cell's top edge coincides with the board's own background transition, so an *absolute* boundary value rises there
and the `4x`-weighted objective raises the empty region instead of lowering it.

**Row-set reconciliation, stated before the run.** Three counts are in circulation and they are not the same
quantity: M43's `173` was the **M43 arm's own** late-row count; the control re-scored in that pass had `140`; M44
re-measured the same bit-exact control at `140` and its own arm at `240`. This pass therefore compares against
**the control's `140` false positives in rows 10-19 at the fixed `0.0` boundary**, on the same `2,986` fixed cells
and the same identity vector, and reports the arm's own per-row counts beside it.

## Ranked candidates

| rank | candidate | expected headroom | cost | falsifier that retires it |
|---|---|---|---|---|
| 1 | **interior contrast** (chosen): each cell's four edge midpoints *relative to that cell's own interior mean per channel*, through a zero-initialised projection | high - it is the exact statistic M44's negative names, and the confound it removes is what produced M44's `251` false positives; the cancellation is mechanical (a board-wide lift cancels in the difference, asserted in the arm's tests) | ~30 lines, `2,496` parameters, one extra sample call | F1 at `0.0` under control `+0.010`, or AUC below the `0.954529` reference, or the late-row mass not below `140` |
| 2 | a contrast-*gated* or relative form of the existing read (scale the read by local contrast) | moderate - it changes each cell's magnitude, so it moves the level the read sits at and the cross-board ordering | small | AUC below the reference floor; a per-cell rescaling is a level mechanism in disguise |
| 3 | per-cell residual / error-focus feature (a cell against a patch reference) | moderate but not separable - the board-wise form is M14/M31's closed level machinery, and the cell-local form is what rank 1 already computes with a better-defined reference | moderate | it reduces to a level or to rank 1 |
| 4 | decision-rule change at the fixed crossing (margin-conditioned precision handling) | large numerically, but it is threshold selection in disguise: any band chosen to raise precision at `0.0` is chosen from the scored population | small | the band would have to be picked from data, which is the per-arm tuning the campaign forbids |
| R | **retired, must not be re-run**: every offset/width/sample-count variant of the M44 ring; M42 support crossing; M43 column step; M16 board crossing; M14/M31 levels; M9-M13 objective level terms; M15/M17 monotone and skyline; M34-M41 geometry | closed by measurement | - | not offered |

**Why rank 1 ranks first.** It is the only candidate that is simultaneously (a) the statistic the previous negative
names, (b) mechanically free of the confound that caused that negative - a board-wide lift cancels exactly in
`ring - interior`, which the arm's own test asserts to `1e-5`, (c) additive at step zero by construction, and (d)
level-free: it never moves the read's offset, so the `0.954529` ranking it must not disturb has nothing to lose
except through genuinely better per-cell evidence.

## The change

`--occupancy-pixel-contrast 1` adds one zero-initialised projection and one sampling call, inside the declared
pixel-read block, after the existing projection:

1. **interior**: the declared in-cell grid (nine samples at `+/-0.3` cell units, the offsets the statistics already
   use) is sampled from the raster at the same declared placement, and averaged per channel - three numbers;
2. **ring**: the four edge midpoints at `+/-0.5` cell units, at the same placement - the same offsets M44 used;
3. **contrast**: for each of the `4 x 3 = 12` ring values, subtract the cell's own interior mean *for that
   channel*, so a board-wide lift or the board's own background transition cancels in the difference and what
   remains is the cell's step against its immediate surroundings;
4. **projection**: `nn.Linear(12, dimension)`, weights and bias initialised to zero, added to the read's cell
   features exactly as the existing pixel projection is. Parameters: `12 x 192 + 192 = 2,496` - the only parameter
   difference from the control, stated here because the arm carries its own projection.

At step zero the added term is exactly zero, so the arm starts bit-identically from the published function. The
geometry the model samples through (`sample_points`, `sample_offsets`) is untouched: this arm adds no sample to the
cell grid the model is trained through.

## Isolation

- **Control**: `m45-control-winner-config` - the winner's recorded configuration (`m30-level-calibrate-r5`:
  `occupancy_pixel_query 3`, `occupancy_read_standardise 1`, dimension `192`, blocks `6`, `component_capacity 64`)
  trained under the current code in this pass, same seed, same `64`-step protocol, same catalogue.
- **Reference**: the control must reproduce `m30-level-calibrate-r5`'s checkpoint `eddec1fd…` bit-exactly; three
  earlier controls of this configuration did (`m42`, `m43`, `m44`). A miss means the new code changed the published
  path and the pass is a defect, not an arm.
- **Arm**: `m45-interior-contrast`, differing from the control in exactly one recorded field plus exactly its
  `2,496` parameters, with a distinct checkpoint.

## Bar, gates and prediction

Against the control re-scored in the same pass at `0.372881` F1 at `0.0` and `0.954529` AUC:

- **F1 at `0.0`** `>= 0.382881` (control `+ 0.010`);
- **AUC** `>= 0.954529` - not merely inside a tolerance: the reference value itself, because an information arm that
  costs ranking is not the improvement this axis needs;
- **the diagnosed mass**: false positives in rows 10-19 strictly below the control's `140`, per row;
- **closure decomposition** (closure / placement / representation against the pixel ceiling, summing to the
  `0.443205` span) and the **per-cell** (`>= 0.70` F1 at `0.0`), **ranking** (`>= 0.90` AUC) and **attribution**
  (`>= 0.05` over the control's best swept) gates, all reported.

**Prediction, declared before the run:** contrast is the right statistic and the effect will be real but small -
F1 at `0.0` in `0.38-0.45`, AUC `0.950-0.958` (ranking held, possibly slightly up), false positives in rows 10-19
down from `140` to `100-145`. The predicted reading is *precision-moves-without-clearing* unless every element
above is met.

## Pre-committed reading of every branch

- **CLEARS** - all three bar elements and all three gates: interior contrast is the per-cell evidence the axis
  needed; the next arm extends it to a second ring radius or composes it with the ceiling read's stencil, and the
  measured trade is the evidence.
- **precision-moves-without-clearing** (the predicted branch) - F1 at `0.0` gains at least `0.010` but AUC is under
  `0.954529` or a gate stays false: the contrast carries real signal at a small ranking cost, and the next arm must
  keep the ranking while keeping the precision; **it must not be re-run with an adjusted radius, sample count or
  offset** - that is per-arm tuning.
- **harmful** - F1 at `0.0` loses `0.010` or more, or AUC falls below `0.949529`: interior contrast is retired as a
  per-cell precision lever, and the axis is recorded as closed for additive raster features at this budget.
- **defect** (not an arm reading) - the control does not reproduce `eddec1fd…`, or the arm's checkpoint equals the
  control's, or the arm's logits are bit-identical to the control's on the fixed population: the projection is
  inert; the cause is found, recorded in `docs/BUG_LOG.md`, and any re-run happens under a fresh name only after
  the cause is repaired.

No branch is re-run for a better number. A negative or neutral result is reported as the result.

## Receipts

`runs/m45-control-winner-config{,-profile}`, `runs/m45-interior-contrast{,-profile,-registration}`,
`runs/m45-interior-contrast-r1-job.json`, `docs/evidence/m45-interior-contrast-r1/{scope,diagnosis}.json`.
