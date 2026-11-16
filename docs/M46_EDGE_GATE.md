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

## Measured

Recorded `docs/evidence/m46-edge-gated-contrast-r1/scope.json` and `diagnosis.json`, one seed, one run, `64` steps,
after the predeclaration committed before it.

| read | F1 at `0.0` | AUC | best swept |
|---|---|---|---|
| control = winner `m30-level-calibrate-r5` | `0.372881` | `0.954529` | `0.689775` |
| **M46 `m46-edge-gated-contrast`** | **`0.140733`** | **`0.757500`** | **`0.252362`** (at `3.00`) |
| delta | **`-0.232148`** | **`-0.197029`** | **`-0.437413`** |

Gates per-cell / ranking / attribution: false / false / **false** (registration admitted the anchored vector the same
way), `cleared` false, `short_stage` `closure`. Split against the `0.443205` span: closure `-0.291612`, placement
`+0.004104`, representation `+0.730713` (fractions `-0.657963` / `+0.009259` / `+1.648703`, sum asserted `1.0`).

**Verdict: `harmful`, and by a wider margin than the ungated form.**

### Per-cell precision at the fixed crossing: it fell, and recall went up

| at the fixed `0.0` boundary | control | M46 | change |
|---|---|---|---|
| true positives | `209` | `213` | `+4` |
| false negatives | `5` | `1` | `-4` |
| false positives | `698` | `2,600` | `+1,902` |
| true negatives | `2,074` | `172` | `-1,902` |
| precision | `0.230430` | `0.075720` | **`-0.154710`** |
| recall | `0.976636` | `0.995327` | `+0.018691` |

`2,600` of the `2,772` empty cells are positive at `0.0`: the fixed crossing sits inside the empty population rather
than between the populations.

### The late-row mass, reconciled and per row

Baseline is the **control's `140` false positives in rows 10-19** (the bit-exact winner; M43's `173` and M44's `240`
were those arms' own counts, and this pass compares against `140`). Per row, before -> now:

| row | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| before | 0 | 18 | 45 | 107 | 58 | 119 | 64 | 39 | 81 | 27 | 0 | 0 | 0 | 0 | 1 | 111 | 2 | 26 | 0 | 0 |
| now | 44 | 59 | 71 | 109 | 122 | 120 | 131 | 128 | 131 | 172 | 141 | 167 | 173 | 170 | 70 | 173 | 173 | 172 | 173 | 101 |

Rows 10-19 went **`140` -> `1,513`**; rows 0-9 went `400` -> `1,087`. The per-column profile
(`270`/`446`/`392`/`399`/`367`/`291`/`232`/`105`/`98`/`0`) shows the mass is not column-specific either.

### Mechanism, measured

- **Not an offset.** The projection's bias stayed small (`norm 0.063863`, mean `1.1e-4`) while its weight norm grew to
  `0.394220`, against M45's `0.0418` / `0.2520`.
- **Not one declared input.** Per-column weight norms are spread across the whole declared set - gated contrasts
  `0.029` to `0.176`, the three gate values `0.089` to `0.091` - so no single statistic carries the change; the
  effect is the projection's total output acting on the latent.
- **The empty distribution moved toward and across the boundary.** The empty-cell margin median is `-0.160` (control
  `-0.529`); only `172` empties remain below zero, and the positive cells sit at median `2.42` while true positives sit
  at `4.29`.
- **Ordering degraded, which a bias cannot do.** AUC is threshold-free: `-0.197029` says the read's ordering of empty
  against occupied cells collapsed, and the split attributes almost all of that to the representation term
  (`+0.730713` of the span, short stage `closure`).

So the gate did what it was declared to do - contrast passes only where the contrast exceeds the cell's own in-cell
spread - and the arm was still harmful: the optimiser spends the added projection on the empty region whatever its
inputs are gated to, because the term enters the same latent the read decides on and the `4x` positive weight rewards
lifting cells broadly.

## Can this family raise fixed-crossing precision? The measured answer

Three members of the additive raster family have now been measured against the same bit-exact control, on the same
`2,986` cells, at the same fixed crossing, under the same `64`-step single-seed protocol:

| arm | what the term is | F1 at `0.0` | AUC | precision at `0.0` | late rows 10-19 |
|---|---|---|---|---|---|
| control | - | `0.372881` | `0.954529` | `0.230430` | `140` |
| M44 | absolute edge-midpoint value | `0.304665` | `0.934008` | `0.180484` | `240` |
| M45 | edge value against the cell's interior | `0.183550` | `0.916312` | `0.101145` | `883` |
| M46 | the same contrast, gated by the cell's own spread | `0.140733` | `0.757500` | `0.075720` | `1,513` |

**No member of the family raises fixed-crossing precision, and the price is `-0.021` to `-0.197` AUC for `-0.068` to
`-0.232` F1 at `0.0`.** The best member (M44, the plain boundary value) still costs `-0.068` F1 and `-0.021` AUC, and
every step that made the statistic more like the thing the diagnosis asked for - relative, then edge-gated - lost
more, not less. The family is closed as a per-cell precision lever.

That closure is not the same as "the raster lacks the information": the committed pixel-ceiling read-out scores
`0.974763` F1 at `0.0` on exactly these cells from the same thirty statistics. What the three arms show is that
**adding raster evidence as a per-cell term to the read's latent does not let the read use it** - the joint objective
spends the term on the empty region instead. The open question is therefore structural: whether the raster's evidence
can be the decision rather than an addition to it.

## Structurally different next steps, ranked

| rank | next arm | headroom | cost | falsifier that retires it |
|---|---|---|---|---|
| 1 | **pixel-sole occupancy read**: the occupancy head reads the thirty statistics directly, with no decoder-latent route into that head | high - the same thirty numbers already score `0.974763` F1 at `0.0` in the committed ceiling read-out, so the information is present and only the route is in question | one head-input declaration, no new parameters, one `64`-step run | AUC `< 0.90` or F1 at `0.0` `< 0.30` - the trunk's latent context is required for ordering and the raster cannot be the decision |
| 2 | **frozen raster projection**: declared fixed random initialisation, only the head and downstream trained | moderate - separates optimisation freedom from information; if the term cannot move, the lift cannot be learned | one flag, no new parameters, one run | no AUC recovery over M46: freedom is not the binding constraint, so the family's information is |
| 3 | **the ceiling read as the decision**: the committed thirty-statistic read-out supplies the decision and only the boundary's level and registration are learned | high but possibly the ceiling itself | moderate - a new decision path, no raster training | F1 at `0.0` `< 0.95` or AUC `< 0.99` on this population - the ceiling's advantage is read- or population-specific |
| 4 | **seed-spread replicate of the winner** (measurement, not an improvement) | it bounds what any future arm delta means | three runs at the same protocol | seed spread in AUC `> 0.02`: every sub-`0.02` delta in the record is unresolvable and the campaign must report intervals from then on |
| R | **retired, must not be re-run**: every additive raster term (M44, M45, M46) and every radius/width/sample/gate variant of them; M42 support crossing; M43 column step; M16 board crossing; M14/M31 levels; M9-M13 objective level terms; M15/M17 monotone and skyline; M34-M41 geometry | closed by measurement | - | not offered |

## Receipts

`runs/m46-control-winner-config{,-profile}`, `runs/m46-edge-gated-contrast{,-profile,-registration}`,
`runs/m46-edge-gated-contrast-r1-job.json`, `docs/evidence/m46-edge-gated-contrast-r1/{scope,diagnosis}.json`.
