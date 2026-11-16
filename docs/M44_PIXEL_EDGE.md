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

## Measured (scored `m44-pixel-edge-r1`)

| read | F1 at `0.0` | AUC | best swept |
|---|---|---|---|
| control / winner `m30-level-calibrate-r5` (re-scored in the same pass) | `0.372881` | `0.954529` | `0.689775` |
| **M44 `m44-pixel-edge`** | **`0.304665`** | **`0.934008`** | **`0.550622`** (at `3.15`) |
| delta | `-0.068217` | `-0.020521` | `-0.139153` |
| anchored variant | `0.312202` | `0.779490` | `0.486322` (at `3.15`) |

**Verdict: not cleared, and the branch that fired is harmful** - both falsifiers were met (`dF1@0 -0.068217 <=
-0.010`, `dAUC -0.020521`, below the declared `0.949529` minimum). Per-cell and ranking gates false, attribution
true (and false on the anchored variant). Split closure `+0.006648` / placement `-0.064299` / representation
`+0.500857` (`+0.014999` / `-0.145078` / `+1.130079`), sum asserted, `short_stage: placement`. Peaks: control
`1,059,082,240`, arm study `1,056,161,792` (against the declared `1,152` MiB), profile `1,020,018,688`, probe
`563,793,920`, reader `45,105,152`.

### Per-cell precision did not rise: the arm added false positives only

The error mix is exact and unambiguous. True positives are **`209` and false negatives `5` in both reads** -
recall is `0.976636` on either side, so the ring added no true positive and lost none. What it added is `251` false
positives (`698` -> `949`) at a slightly *higher* median margin (`1.633866` -> `1.695618`), so precision at the
fixed crossing fell from `0.230430` to `0.180484` (**`-0.049946`**). The feature made the read hotter, not sharper.

Per row, false positives at `0.0` on the `2,986` fixed cells (`comparison_to_m35.per_row`):

| row | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| control | 0 | 18 | 45 | 107 | 58 | 119 | 64 | 39 | 81 | 27 |
| arm | 40 | 50 | 59 | 89 | 76 | 114 | 88 | 80 | 86 | 27 |

| row | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|
| control | 0 | 0 | 0 | 0 | 1 | 111 | 2 | 26 | 0 | 0 |
| arm | 0 | 0 | 0 | 14 | 19 | 138 | 3 | 32 | 21 | 13 |

Rows 10-19: **`140` -> `240`**. The high-margin late-row mass the arm was aimed at went *up* by `100`, and it
gained cells in rows 13, 14, 18 and 19 that the control had entirely clean - the ring's positive evidence is
strongest exactly where the raster is empty above the stack, because there the cell's top edge sits on the board's
own background transition.

### Why it failed, as mechanism

The ring reads *brightness on the boundary*, and the read cannot distinguish "this edge is where a fill stops"
from "this edge is where any contrast sits": above a stack the cell's top edge coincides with the board's own
background edge, so the feature fires there too. Through a `4x`-weighted positive objective it therefore raises
the read in the empty region above the stack rather than lowering it, which is why recall is untouched and the
false positives are the only movement. The sample also moved the whole read's level up (`best swept` threshold
unchanged at `3.15` but its value down `0.139`), and the one positive is small: the closure component is
`+0.006648` over the control, the only stage that did not lose.

### What it settles

Boundary *brightness* is not the missing evidence: at this placement the cell's edge sample is confounded with the
background transition that marks the top of the board, so it cannot separate "filled cell" from "empty cell beside
a border". With recall already at `0.976636`, the remaining error is entirely precision at a fixed crossing, and
this arm's contribution to it is negative. Per the declaration, no offset, width, sample-count or brightness
variant of the ring is offered afterwards; what the negative leaves standing is that the missing per-cell evidence
has to be *contrast against the cell's own interior* (which cancels the background transition) rather than the
boundary value itself - and that is a different feature, to be declared as its own arm.

## Isolation, as measured

| assertion | measured |
|---|---|
| control's checkpoint | `eddec1fd…` - **bit-identical to `m30-level-calibrate-r5`'s**, with the new field present |
| parameter count, arm vs control | `8,023,562` -> `8,026,058`, exactly the declared `12 x 192 + 192 = 2,496` |
| differing recorded config fields | exactly `['occupancy_pixel_edge']` |
| arm's checkpoint | `d4ae2727…`, distinct from the control's |

## Defects found and repaired in this pass

`V-172`: the arm's first profile was refused by the initial-source allowlist
(`initial-weight source tensors differ from the declared representation`) because the new zero-initialised
projection is not a tensor the source checkpoint carries. Repaired by declaring `occupancy_pixel_edge.` in
`INITIAL_SOURCE_ADDED_PREFIXES`, the mechanism V-104 built for exactly this case, with the step-zero identity
asserted bit for bit by the arm's own test. Five refused profiles are preserved as
`runs/m44-pixel-edge-profile*-superseded-initial-source*`.

`V-173`: two detached drivers ran the same arm concurrently, so a second study was launched into a name whose
receipt already existed; it was stopped and preserved as
`runs/m44-pixel-edge-r2-superseded-duplicate-driver*`. The authoritative run is the one whose receipt and
checkpoint agree (`m44-pixel-edge`, `d4ae2727…`); the duplicate never reached a checkpoint.

## Receipts

`runs/m44-control-winner-config{,-profile}`, `runs/m44-pixel-edge{,-profile-r2,-registration}`,
`runs/m44-pixel-edge-r1-job.json`, `docs/evidence/m44-pixel-edge-r1/{scope,diagnosis}.json`.
