# M43 - the per-column step crossing: deciding each column where its own read steps

Status: **predeclared before any scoring run.** The change, its matched control, its bar, its prediction and the
pre-committed reading of every outcome branch are below. Numbers in the "measured" section are added after the run
and are not part of the declaration.

## Why this arm, and not one of the others

M41 and M42 closed geometry on both sides: sampling-side replacement lost `-0.126522` AUC, decision-side support
crossing lost `-0.116317` AUC and `-0.183426` F1 at `0.0`. The M42 diagnosis then located the largest remaining
error mass, and it is not a ranking problem: the record-best read ranks at `0.954529` AUC while scoring `0.372881`
F1 at the fixed `0.0` boundary, and its false positives sit at *high* margin (`1812` at median margin `0.835996`
in the M42 arm; `473` of them in rows 10-14 and `475` in rows 15-19 against M35's `11` and `144`). In other
words the read knows the shape and misplaces the level, and the misplaced level costs it up to `0.316894` F1 -
the gap between `0.372881` and the swept `0.689775`. That is an order of magnitude more than any other measured
residual, so the arm has to attack it.

Ranked by expected headroom against cost, with each alternative's own falsifier:

| rank | candidate | expected headroom | cost | falsifier |
|---|---|---|---|---|
| 1 | **per-column step crossing** (chosen): decide each column at its own sharpest row-to-row descent, one constant per column | high - the diagnosed mass *is* cells above a column's step, so the rule removes it without touching a single ranking inside a column; ceiling is the swept `0.689775` | ~40 lines, no parameter, a 20-value descent per column | F1 at `0.0` fails to beat the control by `0.010`, or AUC falls more than `0.005`, or the late-row mass does not fall |
| 2 | per-board crossing (M16's Otsu rule) applied to the record-best read | low-moderate: M16 measured `+0.014634` over the same read's fixed-`0.0` ceiling, so the rule captures a small part of the gap | tiny - one call, already implemented | gain under `0.010`; also a measured rule re-applied, so a pass would not be new evidence |
| 3 | per-column Otsu crossing | moderate but unstable: Otsu splits *every* column, including empty ones, where the split is arbitrary and inflates positives | small | false positives rise, precision falls |
| 4 | average the six blocks' occupancy logits (stage-ensembled read) | small: averaging shrinks variance, and the level error is systematic rather than noisy | small | no calibration change, AUC flat |
| 5 | learned per-row prior bias | small: the read already carries row embeddings, so the missing quantity is not row identity | small | bias stays near zero at the 64-step budget |
| 6 | objective-side level terms (M9-M13 family: hinge, anchor, quantile, midpoint, dense) | negative, already measured five times: every supervised level term raised the read instead of pinning it | - | not re-run; re-running a falsified family is tuning |

The choice is the only candidate whose headroom is the measured mass itself, and it is a decision rule rather
than a scalar field: no configuration value, threshold, weight, margin or prevalence enters it.

## The change

`--occupancy-column-step-crossing 1` calls `column_step_crossing(logits, visible_rows)` inside the head, after the
declared standardise/level chain and before the monotone envelope:

1. per board, take the visible rows only (spawn rows pass through untouched);
2. `descent[h] = read[h] - read[h+1]` over the bottom-first rows - the row-to-row change;
3. `h* = argmax(descent)`, the column's own sharpest step (ties resolve to the lowest row, deterministically);
4. subtract `(read[h*-1] + read[h*]) / 2` from that column, so `0.0` downstream is exactly the column's step;
5. a column whose largest difference is not a descent (flat or rising) has no step and is returned exactly as it
   was - the declared identity, not a guess.

Properties that make the isolation checkable: exactly one constant per column, so no order inside a column can
change and the within-column AUC is invariant; no sampled point moves, so the geometry and the gradient path to it
are untouched; no parameter is added.

Two label-free statistics were available for a per-column level and both are named here so the choice is on the
record. **Mass balance** - the split equalising the read's sum on both sides, which is M17's emission maximised
per column - is attracted *upward* by the very upper blob this arm exists to remove, because a blob contributes
mass; that is also why M17's joint skyline failed. **The step** is the alternative: a locked stack is a step
function, so the column's sharpest descent is its own statement about where it ends, and a blob whose own end is
softer does not displace it. The declared weakness of the step statistic is its own falsifier: a column whose read
has no genuine step, or whose spurious descent is sharper than the true one, is cut in the wrong place, and that
shows up as lost recall in the late rows.

## Isolation: the matched control

The control is `m43-control-winner-config`: the winner's recorded configuration (`m30-level-calibrate-r5`,
`occupancy_pixel_query 3`, `occupancy_read_standardise 1`), the new flag at `0`, trained under the current code
in the same pass, same seed, same 64-step protocol, same catalogue. The arm differs from it in exactly one
recorded config field, `occupancy_column_step_crossing`.

The reference is `m30-level-calibrate-r5` itself, whose checkpoint is `eddec1fd…`. The M42 pass already proved a
control with this configuration reproduces that checkpoint bit-exactly (`m42-control-winner-config`), and this
pass retrains it with the new field present so that "absent is zero" is measured, not normalised. **The arm is
only evidence if the control reproduces `eddec1fd…` bit-exactly; if it does not, the pass is a defect and is
recorded as one, not read as an arm.**

Isolation assertions to be recorded in the evidence directory: identical parameter counts, one differing config
field, distinct checkpoint hashes, identical sampled points.

## Bar, gates and prediction

Bar (all three, against the winner re-scored in the same pass at `0.372881` / `0.954529`):

- **F1 at `0.0`**: `>= 0.392881` - a `+0.010` gain, which is the smallest gain that is larger than the measurement
  noise the campaign has recorded for identical configurations (`0.000000` on reproduction, `0.001-0.004` across
  same-config reruns);
- **AUC**: `>= 0.949529` - no more than `0.005` down; a per-column shift reorders boards against each other, so a
  small ranking cost is expected and this is the largest cost the declaration accepts;
- **the diagnosed mass**: false positives in rows 10-19 must fall below the control's, per row, not only in
  aggregate.

The standing per-cell/ranking/attribution gates are reported unchanged (`0.70` occupied F1 at `0.0`, `0.90` AUC,
`0.05` attribution over the control's best swept), and the closure decomposition (closure / placement /
representation, summing to the `0.443205` span) is reported against the pixel ceiling as every arm of this
campaign reports it.

**Prediction, declared before the run:** the rule is a level mechanism on the strongest read yet, so the
prediction is *a real gain that does not clear the gate*: F1 at `0.0` in `0.42-0.55`, AUC in `0.92-0.95`
(a small cost, inside the declared tolerance), best swept roughly unchanged, and the late-row false-positive
mass down by at least a third. The reading is "level-moves-without-clearing" unless every bar element is met.

## Pre-committed reading of every branch

- **CLEARS** - all three bar elements and all three gates: the per-column decision rule is the first mechanism to
  turn ranking into fixed-boundary precision, and the next arm composes it upward rather than replacing the read.
- **level-moves-without-clearing** (the predicted branch) - F1 at `0.0` gains at least `0.010` but AUC falls past
  `0.005`, or a gate stays false: the level is movable per column, the ranking cost is the price, and the next arm
  must keep the ranking while moving the level (the measured trade is the evidence).
- **harmful** - F1 at `0.0` loses `0.010` or more, or AUC loses more than `0.005`: the per-column step is a worse
  level statistic than the board's own standardised read, the family is closed, and no width, tie-break or
  smoothing variant is offered afterwards - that would be per-arm tuning.
- **defect** (not an arm reading) - the control fails to reproduce `eddec1fd…`, or the arm's checkpoint equals the
  control's, or its logits are bit-identical to the control's on the fixed population: the rule is inert, the
  cause is found and recorded in `docs/BUG_LOG.md`, and the arm is re-run under a fresh name only after the cause
  is repaired.

No branch is re-run to look for a better number, and a negative is reported as the result.

## Receipts

To be filled by the run: `runs/m43-control-winner-config{,-profile}`, `runs/m43-column-step{,-profile,
-registration}`, `docs/evidence/m43-column-step-r1/{scope,diagnosis}.json`.
