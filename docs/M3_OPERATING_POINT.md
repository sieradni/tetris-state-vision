# Is the standing occupancy attribution a model difference or an operating point?

Status: **predeclared before this audit has any receipt of its own.** Nothing is trained, no
weight, label, threshold, gate or prior verdict changes, and no new model is built. The
instrument reads committed sweep receipts only.

## The question the lever table never asked

Every occupancy verdict in the campaign is read at **threshold `0.0`** - per-cell F1 on the
deciding nonempty boards, with a logit above `0` counted occupied. That is a decision rule, not a
measurement of how well a model separates occupied from empty cells, and it is **sensitive to the
logit offset**. Two arms whose rankings are identical but whose logits are shifted by three units
score very differently at `0.0`, and one of them is not a worse model.

The campaign has already recorded that its levers move exactly this quantity. The 1.5x fit-budget
arm's best swept threshold is `4.35` against its control's `0.15`; the width lever's is `+0.75`
against `-0.75` with its visible logits ~3.3x smaller; the schedule lever's is `+1.95` against
`0.0`. Each of those passes recorded the shift and then refused the lever on its `0.0` F1, with
the sentence *"it moves the operating point, not the precision"*. That sentence is a claim about
the **published decision rule**. It is not a claim about the model, and it has never been tested
as one.

This matters because the refusal is then used to close families. "Budget scaling is not the
limiter", "read-out information is not separation" and "capacity is not the missing lever" all
rest on arms whose `0.0` reading was degraded by a boundary shift their own receipts report.

## What is measured, and why not the obvious alternative

The obvious calibration-free read is the **best swept F1**. The campaign correctly refuses it as a
verdict, because the threshold is fitted to the same cells it is scored on; a sweep can always
find a flattering point.

The audit therefore uses the same grid, the same criterion and the same tie-break, but selects the
threshold **out of fold**:

- the deciding population is the widened one every sweep receipt already commits - 36 training
  cases in **12 ancestry groups**, 667 occupied of 7,200 known cells, 200 known cells per board;
- for each group `g`, the threshold is chosen to maximise pooled F1 **on the other eleven groups
  only** (highest F1, then higher precision, then lower threshold, exactly the sweep's rule);
- that threshold is then applied to group `g`, and only its counts are kept;
- the twelve held-out count sets are pooled into one F1, `cv_f1`.

`cv_f1` cannot be inflated by choosing a boundary: the boundary never sees the cell it scores. It
is invariant to any monotone shift of an arm's logits, so it measures ranking-driven decision
quality at a transferable operating point, which is what the verdicts say they are about.

The F1 curve is computed over the **28 nonempty** boards, exactly as the sweep computes it - an
empty board's false positives are a separate population - so each receipt must reproduce its own
published `threshold 0.0` F1 under that convention or the audit refuses it. (The 36-case figure
above is the population; the scored boards are its 28 nonempty members.)

Four reads are recorded per arm, all from the arm's own committed per-cell logits:

| read | meaning |
|---|---|
| `f1_at_zero` | the standing read at threshold `0.0` |
| `auc` | pooled ranking quality, calibration-free |
| `cv_f1` | out-of-fold-threshold F1, defined above |
| `logit_offset` | median occupied logit minus median empty logit, with its change across the pair |

Alongside them: the count of exact nonempty boards at `0.0`, the count under the out-of-fold
threshold, and a whole-group percentile bootstrap (1,000 draws, seed `20261026`) over the twelve
held-out count triples for `cv_f1`. The bootstrap is on the **evaluation** population; it does not
capture the variance of the fold thresholds themselves, and that limitation is recorded rather
than hidden.

## The arms

The fourteen committed contiguous pairs, each two arms trained under one catalogue and differing
in the declared lever, resolved from `runs/<run>-sweep-r1/result.json`:

| pair | control run | lever run |
|---|---|---|
| representation | `m3-occctx-control-r1` | `m3-occctx-lever-r1` |
| precision (detail tap) | `m3-occctx-control-r1` | `m3-occdetail-lever-r1` |
| stencil | `m3-occdetail-lever-r1` | `m3-occdstencil-lever-r1` |
| stacking | `m3-occdstencil-lever-r1` | `m3-occstack-lever-r2` |
| scale (4x budget) | `m3-occscale-control-r1` | `m3-occscale-lever-r1` |
| fit budget (384 steps) | `m3-occfit-baseline-r1` | `m3-occfit-lever-r1` |
| geometry stage | `m3-occgeom-baseline-r1` | `m3-occgeom-lever-r1` |
| translation | `m3-occorigin-baseline-r1` | `m3-occorigin-lever-r1` |
| extent | `m3-occextent-baseline-r1` | `m3-occextent-lever-r1` |
| support anchors | `m3-occgeocap-baseline-r1` | `m3-occgeocap-lever-r1` |
| lattice residual | `m3-occlatres-baseline-r1` | `m3-occlatres-lever-r1` |
| lattice-free read-out | `m3-occlatfree-baseline-r1` | `m3-occlatfree-lever-r1` |
| board refinement | `m3-occbref-baseline-r1` | `m3-occbref-lever-r1` |
| schedule | `m3-occsched-constant-r1` | `m3-occsched-cosine-r1` |

The width pair is **not** in this audit: its receipt stores aggregates, not per-cell logits, so
the four reads cannot be recomputed from its committed bytes. It is named here as excluded rather
than silently dropped.

**Scope limit that binds the whole reading.** The verdict table's `f1_at_zero` is produced by
`percell_gate` from the 27-forward error replay, i.e. on the **narrow** nine training boards, while
this audit's population is the widened 36-case sweep population. The two are different
instruments. This audit therefore does **not** re-derive any verdict number. What it measures is
whether a threshold-`0.0` F1, computed the same way on one population, tracks an arm's ranking - a
property of the read that carries over to its narrower sibling by construction.

## The rule, fixed now

Attribution is the lever minus its control. For every pair:

- `d_zero` = `f1_at_zero(lever) - f1_at_zero(control)`
- `d_cv` = `cv_f1(lever) - cv_f1(control)`
- `d_auc` = `auc(lever) - auc(control)`

| state | condition |
|---|---|
| **`refusal_reversed`** | `d_zero < 0.05` **and** `d_cv >= 0.05`: the arm the standing read refused clears the standing attribution floor once the boundary is transferable |
| **`gain_reversed`** | `d_zero >= 0.05` **and** `d_cv < 0`: the gain the standing read credited is gone or reversed out of fold |
| **`shifted`** | neither reversal, but `|d_zero - d_cv| >= 0.05`: the two reads disagree by at least a floor |
| **`aligned`** | otherwise |

Overall verdict, in this order:

- **`standing_read_confounded`** - at least one pair is `refusal_reversed` or `gain_reversed`. The
  threshold-`0.0` read cannot separate a better model from a moved boundary, so the table's
  *mechanism* attributions are qualified wherever they rest on it.
- **`operating_point_matters`** - no reversal, but at least one `shifted` pair.
- **`standing_read_sufficient`** - every pair is `aligned`. Then the table's readings stand as
  written and this alternative explanation is closed.

No branch re-opens the fit gate. The gate is a count of exact boards at `0.0`; the audit reports
exact boards under the out-of-fold threshold as a secondary number so the reader can see whether a
transferable boundary would change that count, and no branch reads it as a gate.

## Guards, so the audit cannot join the wrong bytes

Every receipt must be `status: passed`; carry **36** training cases in **12** groups with **667**
occupied cells and **200** known cells per board; reproduce its own published `deciding.auc` and
its own `threshold 0.0` F1 from the case logits it commits (a cross-receipt join in the standing
V-004 pattern); and the two arms of a pair must carry different weights. Any failure refuses the
run instead of skipping a pair, so no gap is silent.

## Falsifiers

- Every pair `aligned` and every `d_cv` matching `d_zero` to within the floor: the confound is not
  material and the campaign's readings stand.
- A reversal produced only by degenerate folds (a group whose boards are all-empty or all-full):
  recorded against the fold table, and the verdict is not read on that pair alone.
- Any receipt failing a guard: no verdict at all.

## What this is not

Not a re-verdict on any lever, not a new threshold for any gate, not a quality or real-image claim,
and not a licence to choose an operating point after seeing the outcome. It measures one property
of one instrument on committed bytes.

## Addendum, predeclared before its own receipt: the delta interval

The states above read point estimates, like every gate in this repository. A point delta of `+0.05`
on 12 ancestry groups is not self-evidently resolvable, so a **paired whole-group bootstrap**
(1,000 draws, seed `20261026`) is reported for `d_cv`: one index draw is shared by both arms, so the
between-group variation the two arms share is removed from the difference, and the interval is a
percentile interval over the resampled held-out counts. The committed fold boundaries are held
fixed; re-selecting a threshold inside each resample is **not** done, and that is the interval's
main limitation.

The interval is **report-only**. No state and no verdict reads it, exactly as `gate_power`'s
intervals never move a gate. The receipt records `reversals_with_interval_excluding_zero` beside the
state tally so a reader can see how many reversals survive their own uncertainty; that count is
descriptive and is not a branch either.

## Result (receipt `docs/evidence/m3-operating-point-r2/result.json`)

Status: the pass ran under the bounded runner (`runs/m3-opaudit-r5-job.json`, passed, 88.7 s, peak
46,661,632 B of the 128 MiB ceiling, 28 work items). No model byte was read that is not in a
committed receipt, and every receipt reproduced its own published `deciding.auc` and `threshold 0.0`
F1 under the nonempty-board convention before any pair was classified; the shared control's
`0.414649` is the verdict table's own number.

**Verdict `standing_read_confounded`.** Fourteen declared pairs, state tally `refusal_reversed` **6**,
`gain_reversed` **1**, `shifted` **5**, `aligned` **2**.

| pair | `d f1@0` | `d cv_f1` | `d auc` | paired interval on `d cv_f1` | state |
|---|---:|---:|---:|---|---|
| representation | +0.075 | +0.001 | +0.082 | -0.135 .. +0.146 | shifted |
| precision | +0.097 | -0.018 | +0.080 | -0.152 .. +0.122 | gain_reversed |
| stencil | +0.014 | +0.067 | +0.058 | -0.033 .. +0.141 | refusal_reversed |
| stacking | +0.007 | -0.062 | -0.084 | -0.144 .. +0.048 | shifted |
| 4x scale | -0.036 | -0.024 | -0.025 | -0.103 .. +0.049 | aligned |
| **fit budget (384 steps)** | **-0.134** | **+0.149** | **+0.074** | **+0.006 .. +0.277** | refusal_reversed |
| geometry stage | +0.064 | +0.107 | +0.006 | -0.001 .. +0.190 | aligned |
| translation | +0.002 | +0.051 | +0.019 | -0.063 .. +0.159 | refusal_reversed |
| extent | -0.083 | +0.072 | -0.073 | -0.046 .. +0.197 | refusal_reversed |
| support anchors | -0.213 | -0.018 | -0.014 | -0.120 .. +0.082 | shifted |
| lattice residual | -0.063 | +0.011 | -0.005 | -0.122 .. +0.140 | shifted |
| lattice-free read-out | +0.014 | -0.043 | +0.015 | -0.172 .. +0.113 | shifted |
| board refinement | +0.009 | +0.053 | +0.034 | -0.042 .. +0.119 | refusal_reversed |
| **schedule (cosine)** | **+0.005** | **+0.084** | **+0.041** | **+0.010 .. +0.168** | refusal_reversed |

**Two reversals survive their own uncertainty**, and both are compute-side:

- the **1.5x fit budget** is `+0.149` out of fold with an interval clear of zero, `+0.074` AUC, and
  its own published placement improved (1.9812 -> 1.8419 cells) while the standing read refused it at
  **-0.134** - the largest standing regression in the table;
- the **cosine schedule** is `+0.084` with an interval clear of zero, `+0.041` AUC, and published
  placement improved 39.9%, while the standing read put it at **+0.005**, i.e. a null.

Five other reversals are **unresolved** at twelve groups: stencil, translation, extent, board
refinement, and the precision arm, whose credited `+0.097` becomes `-0.018` out of fold. Four
refusals are **supported** by the neutral read - stacking (-0.062), lattice-free (-0.043), 4x scale
(-0.024) and support anchors (-0.018) - so the read-out family's negative is not uniformly an
artifact either.

The mechanism is visible in the arms themselves: the shared control's occupied-minus-empty median
offset is `+4.16`, and the arms span `+1.88` to `+4.16` in offset and `1.56` to `3.40` in logit
scale. A `0.0` boundary is not the same decision across those arms.

**No gate moves.** Every arm has **0 of 28** exact nonempty boards at `0.0`; under the out-of-fold
boundary every arm still has **0 of 28** except the cosine arm's **1**, and no arm's `cv_f1` exceeds
`0.533`. "This representation is far from fit" is therefore not a calibration artifact, and the fit
gate stays FALSE.

## What this changes, and what it does not

- **Retired:** the *compute* closure. "Budget is not the limiter" and "more optimisation is not a
  substitute" were read at a boundary the levers moved; both compute-side levers reverse with an
  interval clear of zero.
- **Unchanged:** the fit gate (0 of 28), the level claim (no read approaches 0.70), every threshold,
  population and evidence byte, and the width `capacity_null` - that pass read best swept F1 and AUC,
  which are ranking quantities, not this boundary.
- **Qualified:** the read-out family's mechanism sentences that rest on `0.0` (notably "precision
  improved as declared", which reverses out of fold), while its stacking, lattice-free, 4x-scale and
  support-anchor refusals stand.
- **Ranked next:** a declared, **calibration-neutral re-read of the two compute-side arms** - same
  arms, primary = out-of-fold or group-mean AUC with a group-level interval, powered by replicate
  arms rather than the single deterministic pair - before any further lever is read at `0.0`.

Limits: the audit population is the sweep's 28 nonempty boards in 12 groups and cannot re-derive the
verdict primary, which is `percell_gate`'s F1 on the narrow nine-board replay; the interval is
`+/-0.15` wide at this group count and holds the committed fold boundaries fixed; the exact-board
count under the out-of-fold boundary is reported, never read as a gate; one seed, synthetic
population, no real-image or complete-state claim.

## Disclosure

An inventory of committed receipts and a scratch table of the four reads were run before this file
was written, to establish that per-cell logits are committed for every arm and to size the
instrument. The rule above was fixed before the instrument's own receipt existed; the audit's
receipt is the record, and the scratch table is not evidence.
