# M44 - the pixel edge ring: giving the read the cell's own boundary

Status: **predeclared before any scoring run.** The change, its matched control, its bar, its prediction and the
pre-committed reading of every branch are below. The measured section is added after the run and is not part of the
declaration.

## Ranked next steps, and why this one ranks first

M43 closed the level axis at every granularity it had been tried (M14 board mean, M16 board crossing, M31 board
standardisation, M42 support, M43 column step) and closed it from the worst side: both arms aimed at the diagnosed
high-margin mass made it worse (`140` -> `173` late-row false positives in M43). The winner's ranking is
`0.954529` and its fixed-crossing F1 is `0.372881`, so what is left is per-cell precision *at a fixed crossing* -
information the read does not have, not a place to put `0.0`.

| rank | candidate | expected headroom | cost | falsifier that retires it |
|---|---|---|---|---|
| 1 | **pixel edge ring** (chosen): four samples on the cell's own edge midpoints, half a cell out - outside the declared `+/-0.3` in-cell grid - through a zero-initialised projection | high: the committed pixel-ceiling read scores `0.974763` F1 at the same fixed boundary on the matched population from deterministic raster statistics, so boundary evidence in the raster is sufficient for this task; the current read samples only *inside* the cell, so it cannot see where the fill stops | ~30 lines, `2,496` parameters, one sample call | F1 at `0.0` fails to beat the control by `0.010`, or AUC falls below `0.949529` |
| 2 | a second shallow head over the same `30` statistics | low - capacity, not information: the same numbers already enter one zero-initialised projection | small | no gain; the read already had this evidence |
| 3 | a denser in-cell grid (more samples inside `+/-0.3`) | low - it is the same region resampled, and sample count was already measured (M28's form) | small | no gain, and it is a variant of a measured form |
| 4 | learned per-row or per-column prior bias | low - a prior, not evidence, and the read already carries row and column embeddings | small | the bias stays near zero at the `64`-step budget |
| 5 | a column-prefix self-feature (the read's own height to the cell's left) | medium but not separable from M15/M43 - it is the monotone structure again, which is closed | moderate | it reduces to the M43 prediction and loses the same way |
| 6 | any level mechanism | closed by measurement (M9-M13, M14, M16, M31, M42, M43) | - | not re-run |

The chosen arm is first because it is the only candidate that adds **information the read cannot currently access**
- the raster on the cell's boundary, half a cell outside every sample the declared read takes - rather than
re-weighting, re-sampling or re-placing the statistics it already has. It changes no sample the model uses for
geometry, adds no annotation, and at step zero it is bit-identically the published function.

## The change

`--occupancy-pixel-edge 1` adds one zero-initialised projection and one sampling call:

1. the four offsets `(-0.5, 0)`, `(0.5, 0)`, `(0, -0.5)`, `(0, 0.5)` in cell units - the midpoints of the cell's
   left, right, bottom and top edges - are sampled from the raster at the *same* declared placement the statistics
   use (`lattice`, or whichever placement the declared pixel form selects);
2. the four samples x three channels are `12` features, passed through `nn.Linear(12, dimension)` whose weights and
   bias are initialised to zero, and added to the read's cell features exactly as the existing pixel projection is;
3. parameters: `12 * 192 + 192 = 2,496`, the only difference from the control, stated here because a
   representation arm has to carry its own projection (unlike M42/M43, which added none).

At step zero the added term is exactly zero, so the arm starts bit-identically from the published function and any
difference is learned. The cell samples the geometry uses - `sample_points`, `sample_offsets` - are untouched.

## Isolation

- **Control**: `m44-control-winner-config`, the winner's recorded configuration (`m30-level-calibrate-r5`:
  `occupancy_pixel_query 3`, `occupancy_read_standardise 1`, dimension `192`, blocks `6`, `component_capacity 64`)
  trained under the current code in this pass, same seed, same `64`-step protocol, same catalogue.
- **Reference**: the control must reproduce `m30-level-calibrate-r5`'s checkpoint `eddec1fd…` bit-exactly; two
  earlier controls of this configuration already did (`m42-control-winner-config`, `m43-control-winner-config`), so
  a failure here means the new code changed the published path and the pass is a defect, not an arm.
- **Arm**: `m44-pixel-edge`, differing from the control in exactly one recorded field plus its `2,496` parameters,
  with a distinct checkpoint.

## Bar, gates and prediction

Against the control re-scored in the same pass at `0.372881` F1 at `0.0` and `0.954529` AUC:

- **F1 at `0.0`** `>= 0.382881` (`+0.010`);
- **AUC** `>= 0.949529` (no more than `0.005` down) - the arm must not buy precision by disturbing the ordering;
- **the diagnosed mass**: false positives in rows 10-19 must fall below the control's `140`, per row;
- standing gates reported unchanged: per-cell `>= 0.70`, ranking `>= 0.90` AUC, attribution `>= 0.05` over the
  control's best swept; closure decomposition against the pixel ceiling reported as every arm reports it.

**Prediction, declared before the run:** the edge ring is the first arm since M31 to add information rather than
re-place a level, so the prediction is *a real gain that does not clear the gate*: F1 at `0.0` in `0.38-0.46`, AUC
`0.95-0.96` (ranking preserved or slightly up), late-row false positives down from `140` to `100-140`.

## Pre-committed reading of every branch

- **CLEARS** - all three bar elements and all three gates: boundary information is the missing per-cell evidence,
  and the next arm extends the ring (its two-ring form or the four quarter-cell corners) rather than replacing it.
- **precision-moves-without-clearing** (the predicted branch) - F1 at `0.0` gains `>= 0.010` but a gate stays false
  or AUC falls past `0.005`: the boundary is real evidence and insufficient alone; the next arm adds boundary
  samples at more scales or composes them with the ceiling read's contrast rule, and the measured trade is the
  evidence.
- **harmful** - F1 at `0.0` loses `0.010` or more, or AUC falls below `0.949529`: the edge ring is retired and no
  offset, width or sample-count variant of it is offered afterwards - that would be the per-arm tuning the
  declaration forbids.
- **defect** (not an arm reading) - the control does not reproduce `eddec1fd…`, or the arm's checkpoint equals the
  control's, or its logits are bit-identical to the control's on the fixed population: the projection is inert,
  the cause is found and recorded in `docs/BUG_LOG.md`, and the arm runs again under a fresh name only after the
  cause is repaired.

No branch is re-run for a better number. A negative is reported as the result.

## Receipts

`runs/m44-control-winner-config{,-profile}`, `runs/m44-pixel-edge{,-profile,-registration}`,
`runs/m44-pixel-edge-r1-job.json`, `docs/evidence/m44-pixel-edge-r1/{scope,diagnosis}.json`.
