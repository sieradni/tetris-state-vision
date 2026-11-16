# M46 - the edge gate: contrast passes only where the cell has an edge

Status: **predeclared before any scoring run.** The gate rule, its threshold, the matched control, every bar element,
the prediction and the pre-committed reading of every branch are below. The measured section is added after the run
and is not part of the declaration.

## What M45 decided this arm has to be

M45 measured interior contrast as **harmful** and located its mechanism numerically rather than by assertion: the
projection's bias stayed small (`0.0418` norm), the empty-cell margin median barely moved (`1.633866` -> `1.662931`),
and the whole loss was the **tail below zero** - `1,186` of the control's most confidently negative cells crossed
up, because at a featureless background cell the contrast against its own ring is near zero by construction. So the
absolute cell statistics, which carry the level, were spent by an ungated contrast term.

The change this arm makes is therefore a **gate**, not a knob: the contrast enters only where a local edge actually
exists, and the absolute statistics stay untouched everywhere else.

## Ranked candidates considered

| rank | candidate | expected headroom | cost | falsifier that retires it |
|---|---|---|---|---|
| 1 | **edge-gated contrast** (chosen): the twelve ring contrasts, each multiplied by that cell's own edge gate built from the in-cell spread, plus the three gate values; absolute statistics untouched | high - it removes exactly the mechanism M45 measured as the loss (ungated contrast on featureless cells) while keeping the edge evidence | `15 x 192 + 192 = 3,072` parameters, one extra sampling call | F1 at `0.0` under control `+0.010`, or AUC below the `0.954529` reference |
| 2 | separate raster tower concatenated to the read instead of summed | low - same information, different credit assignment; nothing in M45's diagnosis is a capacity or mixing problem | `~30k` parameters | fails to move the tail below zero by more than the gate does |
| 3 | a two-sided in-cell residual (each of the nine interior samples against its own mean) | low - it is a rescaled interior statistic the read already receives as three spreads, so it adds ordering the twenty-seven raw samples already imply | none | no gain over the control at matched features |
| 4 | patch-referenced contrast (the reference being the board region around the cell, not the cell itself) | moderate but confounded - the board-wise reference is M14/M31's closed level machinery, and an off-board patch reference moves the sampled geometry, which M34-M41 measured at `-0.07` to `-0.35` best swept | moderate | reduces to a level or to moving the lattice |
| 5 | decision-rule change (margin-conditioned precision at the fixed crossing) | large numerically, but any band would be selected from the scored population | small | band selection is per-arm tuning, so it is not admissible |
| R | **retired, not offered**: every radius / width / sample-count variant of the M44 and M45 rings; M42 support crossing; M43 column step; M16 board crossing; M14/M31 levels; M9-M13 objective level terms; M15/M17 monotone and skyline; M34-M41 geometry | closed by measurement | - | not re-run |

Rank 1 ranks first because it is the one change that acts on the measured cause of the previous negative instead of
on a dial: M45's loss was the ungated pass on cells with no local edge, and the gate is identically zero there.

## The gate rule, fixed before the run and derived from the pipeline

For each cell and each colour channel `c`, over the **nine declared in-cell samples** at `+/-0.3` cell units that the
pixel statistics already take:

- `m_c` - their mean, already the cell's interior value;
- `s_c` - their standard deviation, already one of the three spread features the read receives and the statistic the
  standardised read divides by (`occupancy_read_standardise`, the winner's own configuration);
- `k_c,p` - the ring contrast at each of the four declared edge midpoints (`+/-0.5` cell units, as in M44/M45):
  `ring_c,p - m_c`.

The gate is

```
g_c = relu(|k_c,p| - s_c - phi) / (|k_c,p| + s_c + phi)          in [0, 1)
```

with `phi = 1e-6`, **the read's own division floor**, the same constant the standardised read already uses
(`clamp_min(1e-6)`) for the same division.

**The threshold is therefore `|k_c,p| > s_c + phi`**: a contrast passes only when it exceeds the cell's own in-cell
variation - its noise floor - and the same floor appears in the numerator so a cell whose contrast and spread are
both at the floor passes exactly `0` instead of amplifying float noise (this exact case was found while writing the
arm's own invariance test, before any run: at a uniform cell `|k|` and `s` are both ~`1e-7`, and the ungated ratio
returned `0.054` of pure noise).

- **No constant is fitted.** `s_c` is computed from the raster at the same placement, and `phi` is a pre-existing
  constant of the read. Nothing is selected from rows, scores, thresholds or any label vector.
- **Where a cell and its ring are both plain background**, `|k| ~ 0` and `s ~ 0`, so `g = 0` exactly: the contrast
  contributes nothing, which is the evidence M45 spent and this arm keeps.
- **Where a local edge exists**, `|k| > s` and `g` rises toward `1` with the excess, so the graded strength of the
  edge passes rather than a binary flag.

## The declared feature set, and the parameter delta

`--occupancy-pixel-edge-gate 1` adds, inside the declared pixel-read block, after the existing projection:

1. the twelve gated contrasted values `k_c,p * g_c` (four midpoints x three channels), flattened;
2. the three per-channel gate values `mean_p(g_c)`, so the read can see how much of its evidence passed;

fifteen numbers in total, through `nn.Linear(15, 192)` with weights and bias **zero-initialised**. Parameters added:
`15 x 192 + 192 = 3,072` - the only parameter difference from the control. Step zero is exactly zero, so the arm
starts bit-identically from the published function. The ring offsets are a non-persistent buffer; the geometry the
model samples through (`sample_points`, `sample_offsets`) is untouched. The gated and ungated contrast forms are
mutually exclusive and the configuration refuses both at once.

## Isolation

- **Control**: `m46-control-winner-config` - the winner's recorded configuration (`m30-level-calibrate-r5`:
  `occupancy_pixel_query 3`, `occupancy_read_standardise 1`, dimension `192`, blocks `6`, `component_capacity 64`)
  trained under this code in this pass, same seed, same `64`-step protocol, same catalogue.
- **Reference**: the control must reproduce `m30-level-calibrate-r5`'s checkpoint `eddec1fd…` bit-exactly; four
  earlier controls of this configuration did (`m42`, `m43`, `m44`, `m45`). A miss makes the pass a defect, not an arm.
- **Arm**: `m46-edge-gated-contrast`, differing from the control in exactly one recorded field plus exactly its
  `3,072` parameters, with a distinct checkpoint, and pinned non-inert by a test.

## Bar, gates and prediction

Against the control re-scored in the same pass at `0.372881` F1 at `0.0` and `0.954529` AUC:

- **F1 at `0.0`** `>= 0.382881` (control `+ 0.010`);
- **AUC** `>= 0.954529`, the reference itself, not a tolerance;
- **the diagnosed mass**: false positives in rows 10-19 strictly below the control's `140`, reported per row, with
  the row-set reconciliation stated (M43's `173` and M44's `240` were those arms' own counts; this pass compares
  against the control's `140`);
- **closure decomposition** against the `0.443205` span and the **per-cell** (`>= 0.70`), **ranking** (`>= 0.90` AUC)
  and **attribution** (`>= 0.05` over the control's best swept) gates.

**Prediction, declared before the run:** the gate removes M45's loss, so the confident-negative tail should be
preserved rather than spent - true negatives near the control's `2,074`, false positives at or below `698` - and the
edge evidence should carry some precision: F1 at `0.0` in `0.37-0.46`, AUC `0.950-0.962`, rows 10-19 in `100-150`.
The predicted reading is *precision-moves-without-clearing* unless every element above is met.

## Pre-committed reading of every branch

- **CLEARS** - every element and all three gates: the gate is the mechanism, and the family can raise fixed-crossing
  precision at the winner's ranking; the next arm scales the ring or adds a second, coarser edge radius.
- **NEUTRAL** (predicted) - F1 at `0.0` at or above `0.362881` but below `0.382881`, or AUC below `0.954529` while
  the tail is preserved: the gate works and the edge evidence is too thin at this budget to buy precision; the family
  answer is then "graded by cost: it can hold the ranking, not raise precision", and **no radius, width or
  sample-count variant is re-run**.
- **HARMFUL** - F1 at `0.0` at or below `0.362881`, or AUC below `0.949529`: additive raster terms are retired as a
  per-cell precision lever whatever their gating, and the axis moves to a structurally different arm.
- **DEFECT** (not an arm reading) - the control misses `eddec1fd…`, the arm's checkpoint equals the control's, or the
  arm's logits are bit-identical to the control's on the fixed population: the cause is found and recorded in
  `docs/BUG_LOG.md`, and any re-run happens under a fresh name only after the repair.

No branch is re-run for a better number.

## Receipts

`runs/m46-control-winner-config{,-profile}`, `runs/m46-edge-gated-contrast{,-profile,-registration}`,
`runs/m46-edge-gated-contrast-r1-job.json`, `docs/evidence/m46-edge-gated-contrast-r1/{scope,diagnosis}.json`.
