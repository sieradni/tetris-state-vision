# M47 - the pixel-sole occupancy read: does the route or the evidence bind?

Status: **predeclared before any scoring run.** What feeds the head, the absence of a latent route, the expected
parameter delta, the protocol, the falsifier (derived from already-committed receipts) and every branch are below.
The measured section is added after the run and is not part of the declaration.

## What M46 left open, and why this is a routing test

M46 closed the additive raster family: three members cost `-0.021` to `-0.197` AUC for `-0.068` to `-0.232` F1 at
`0.0`, and the two steps that made the statistic more like the diagnosis asked for lost more, not less. The same
pass recorded the fact that makes this arm worth running: the **committed pixel-ceiling read-out scores `0.974763`
F1 at `0.0` and `0.999195` AUC on these same `2,986` cells from the same thirty statistics**, and the winner's own
read receives those statistics too - through a projection added to the decoder latent. So the question is not
whether the raster carries the evidence. It is whether the *route* lets the read use it.

This arm is therefore a restriction, not a feature. **The occupancy head's input becomes the raster projection
itself. No decoder-latent path reaches the decision.** If the pixel-sole read reproduces the ceiling read-out's
quality, the latent route was the problem. If it collapses to M46-like levels, the raster-to-decision path itself is
not the binding constraint and the successor has to be structural somewhere else - a different decision surface, a
different supervision, or a different geometry, not another raster statistic.

## The exact read that feeds the head

In the declared pixel-read block (`occupancy_pixel_query`, the winner's own form `3`), the thirty statistics are the
nine declared in-cell samples per channel at `+/-0.3` cell units plus three per-channel spreads, sampled at the
model's own predicted lattice for the stage - exactly the numbers the ceiling read-out measured. They pass through
the existing projection `occupancy_pixel_query` (30 -> 64 -> `dimension`), and with `--occupancy-pixel-sole 1` that
projection's output **is** the head's input:

```
occupancy_features = projected          # the sole read: assignment, not addition
occupancy_logits   = self.occupancy(occupancy_features)
```

- **No latent path:** the decoded cell features, the detail tap, the neighbourhood block, the column context and
  every additive raster term are excluded by construction - the assignment replaces the accumulated latent features
  rather than joining them. The arm's own test captures the occupancy head's input tensor and the projection's
  output in one forward pass and requires them to be `torch.equal`, and the same probe on the control shows the
  latent path present, so the test is not vacuous.
- **No new tensor:** the arm creates, initialises, zeroes and removes nothing. Its state dict has the same keys and
  the same shapes as the control's, so **the expected parameter delta is exactly `0`**.
- **Not the published function at step zero, by design.** The projection's last layer is zero-initialised in a fresh
  build, but the arm starts from the project's published source checkpoint (`runs/m3-percell-balance-r1`), which
  already carries a trained projection (30 -> 64 weights at norm `4.67`). The arm's first step is therefore the
  trained-projection read, not the winner's read; this is declared here so that the difference is expected rather
  than discovered, and the arm's tests pin non-inertness with a declared non-zero projection.

## Protocol, control and isolation

- **Control**: `m47-control-winner-config` - the winner's recorded configuration (`m30-level-calibrate-r5`:
  `--model accuracy`, `--component-capacity 64`, `--component-identity-readout positional`, `--component-endpoint-term
  length`, `--occupancy-factor 4`, `--occupancy-positive-weight 4`, `--occupancy-pixel-query 3`,
  `--occupancy-read-standardise 1`, `--curriculum stream-v1`, `--capture-max-side 512`, dimension `192`, blocks `6`),
  same seed, same `64` steps, same evaluator, same splits, same `2,986` fixed cells, same catalogue, under this code.
- **Reference**: the control must reproduce `m30-level-calibrate-r5`'s checkpoint `eddec1fd…` bit-exactly; five
  earlier controls of this configuration did (`m42`-`m46`). A miss makes the pass a defect, not an arm.
- **Arm**: `m47-pixel-sole`, differing from the control in exactly one recorded field and **no** parameter, with a
  distinct checkpoint and a test-pinned non-inertness.
- **One run.** If host memory refuses admission, the driver retries with fresh receipt names inside the pass, each
  refusal preserved; admission pressure never ends the pass.

## Falsifier, derived from already-committed receipts

The numbers this bar uses were committed before this arm existed: the winner's arm (`0.372881` F1 at `0.0`,
`0.954529` AUC, best swept `0.689775`, precision at `0.0` `0.230430`) and the ceiling read-out on the same cells
(`0.974763` F1 at `0.0`, `0.999195` AUC, best swept `0.987179`). No seed spread is committed for the winner, so this
pass treats deltas larger than `0.02` AUC as resolvable and claims nothing smaller.

- **HELPFUL** - F1 at `0.0` `>= 0.382881` (control `+ 0.010`) **and** AUC `>= 0.954529`: the route was the problem,
  the raster's evidence reaches the decision when nothing else dilutes it.
  - **DECISIVE CEILING** (the strongest form): F1 at `0.0` `>= 0.70`, the campaign's standing per-cell gate, cleared
    by a learned read for the first time.
- **NEUTRAL** - F1 at `0.0` in `[0.362881, 0.382881)` **and** AUC `>= 0.949529`: the restriction neither helps nor
  hurts at the declared resolution; the raster-to-decision path is viable but not sufficient.
- **HARMFUL** - F1 at `0.0` `<= 0.362881` **or** AUC `< 0.949529`: reading the raster alone is not enough at this
  budget, so the raster-to-decision path is not the binding constraint and no further raster routing is open.
- **DEFECT** (not an arm reading) - the control misses `eddec1fd…`, or the arm's checkpoint equals the control's, or
  the arm's logits are bit-identical to the control's on the fixed population.

Also reported, as in every arm: the complete TP/FN/FP/TN table and precision/recall at the fixed `0.0` crossing, the
closure decomposition against the `0.443205` span, the per-cell / ranking / attribution gates, and the row-wise
false-positive profile for rows 10-19 and 0-9 separately, reconciled to the control's own `140`.

**Prediction, declared before the run:** the thirty statistics carry the ordering (the ceiling read-out proves it on
these cells), but a 30 -> 64 -> 192 projection trained for `64` steps at the `4x` objective is not the ceiling
read-out: F1 at `0.0` `0.45-0.75`, AUC `0.96-0.99`, precision at `0.0` `0.35-0.65`, late rows 10-19 `20-90`. The
predicted reading is **HELPFUL, possibly decisive**.

## Pre-committed reading of every branch

- **HELPFUL** - the latent route is the binding constraint on the raster's evidence; the next arm keeps this routing
  and replaces the weak part (the 30 -> 64 -> 192 projection and its 64-step budget) with the ceiling read-out's own
  committed read as the decision surface, then asks how much of the ceiling survives the joint objective.
- **NEUTRAL** - routing moves the read materially but not enough: the successor must change the decision surface
  (a per-cell logistic read on the same statistics, committed rather than learned) rather than the route again.
- **HARMFUL** - recorded as the campaign's answer that no routing of these thirty statistics, trained under this
  protocol, reaches the winner's read: the next step is structural elsewhere (supervision over the decision surface,
  or the registered decision surface itself), and no further raster routing or raster term is declared.
- **DEFECT** - the cause is found, recorded in `docs/BUG_LOG.md` with its prevention test, and any re-run happens
  under a fresh name only after the repair.

No branch is re-run for a better number.

## Receipts

`runs/m47-control-winner-config{,-profile}`, `runs/m47-pixel-sole{,-profile,-registration}`,
`runs/m47-pixel-sole-r1-job.json`, `docs/evidence/m47-pixel-sole-r1/{scope,diagnosis}.json`.
