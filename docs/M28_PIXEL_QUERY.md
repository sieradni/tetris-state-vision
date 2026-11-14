# M28: the per-cell pixel-statistics read-out

Status: **predeclared before this pass has any receipt of its own.** The lever, the matched
control, the budget, the gates, the attribution and the failure criterion below are fixed; the
measured section is appended only after the canonical receipts exist.

## Where this comes from

M27 measured the occupancy deficit on one population
([M27_THREE_FACTOR.md](M27_THREE_FACTOR.md), `docs/evidence/m27-three-factor-r3/scope.json`):

| read | AUC | F1 at `0.0` | best swept F1 |
|---|---|---|---|
| model, own lattice (2,986 matched cells) | 0.933138 | 0.439024 | 0.5440 |
| model, registration granted | 0.970233 | 0.574181 | 0.7626 |
| 30-feature pixel read-out, same cells | 0.999195 | 0.974763 | 0.9872 |

Registration and representation are equal first-order terms (`+0.218616` and `+0.224590` best
swept F1), and the boundary term is `+0.188409`. The read-side family is therefore bounded: no
read-side lever can close more than the representation term, and this pass tests the largest
measured one.

The question is deliberately narrow. The pixel ceiling was **granted the target lattice**. The
model must read at its **own predicted** lattice, whose mean absolute placement offset is `2.575`
target cells. So the open quantity is not "are the colours informative" - that is measured - but
**how much of the `0.2246` a learned head recovers when the same feature class is computed at the
model's own placement**.

## The single declared change

`ModelConfig.occupancy_pixel_query` (default `0`, published). When `1`, the occupancy read-out is
given 30 deterministic per-cell statistics computed from the prepared raster at the lattice the
stage already decodes:

- nine sub-cell samples on a 3x3 grid over the declared interior extent (`0.3` of a cell in each
  lattice axis), giving twenty-seven channel values, plus three per-channel spreads;
- `occupancy_features = features + W * statistics + b`, with `W`, `b` of a `Linear(30, 192)`
  **zero-initialized**, so at step zero the block adds exactly zero and the arm is strictly
  additive against its matched control;
- the sampler reads `pixels` and the model's own `lattice` only: **no label, no target lattice,
  no decoded state, no threshold and no prevalence**.

The lever is refused beside every other occupancy read-out selector (`--occupancy-context`,
`--occupancy-detail`, stencil, resolution, deform, column, lattice-free, offset rows, and the
five level/supervision transforms), so it is one declared statement about what the read sees.

## Matched pair, one budget

Both arms are **continuations** of the published checkpoint `runs/m3-percell-balance-r1`, at the
campaign's standing 64-step protocol: `--model accuracy --component-capacity 64
--component-identity-readout positional --component-endpoint-term length --occupancy-factor 4
--occupancy-positive-weight 4 --curriculum stream-v1 --capture-max-side 512 --seed 20261026
--steps 64 --learning-rate 0.001 --evaluate-every 64 --development-groups 6`, 1,152 MiB aggregate
commit, 1,200 s runner wall, one heavy job at a time.

| arm | run | `occupancy_pixel_query` |
|---|---|---|
| control | `runs/m28-pixelread-base-r1` | `0` |
| lever | `runs/m28-pixelread-lever-r1` | `1` |

Nothing else differs. The control is a continuation too, so the comparison isolates the block, not
"more training".

## Scoring, on the unchanged population and metric

Each arm is scored read-only by `scripts/sweep_stream_occupancy.py` on the deciding population
(36 training images, 28 nonempty boards, 667 occupied and 4,933 empty known cells, 12 ancestry
groups), standing 81-point grid `-6.0 .. 6.0` step `0.15`, occupied iff `logit > threshold`, pooled
AUC with ties counted one half, best F1 with the sweep's tie-break. The arm's own registration
mapping is measured by `scripts/probe_registration_attribution.py` (36 forwards), and the same
three reads M27 defined - own lattice, oracle registration, pixel read-out on the same cells - are
then recomputed by `scripts/pixel_query_read.py` from the arm's own receipts.

**The lever CLEARS iff, at its own lattice on the full deciding population:**

1. **per-cell gate** - occupied-cell F1 at `0.0` is `>= 0.70` (`percell_gate.CELL_F1_THRESHOLD`);
2. **ranking gate** - `occupancy_sweep.sweep_verdict` is `informative` (pooled AUC `>= 0.90` and
   some swept threshold reaches F1 `>= 0.70`);
3. **attribution floor** - its best swept F1 exceeds the matched control's by `>= 0.05`.

**Otherwise the lever is NOT CLEARED.** If either arm cannot finish inside the declared envelope
the outcome is **NOT ADMISSIBLE AS DECLARED**, recorded with its measured peak and wall; no ceiling
is raised and no run is repeated to pass.

## Declared attribution, fixed before the run

The pass reports, for each arm, three reads on the arm's own co-scoreable cells:

- `registration_gain` = oracle best swept F1 - own-lattice best swept F1 (the same arm, one
  population);
- `representation_gap` = pixel best swept F1 - oracle best swept F1 (the same cells);
- the fraction of the M27 `0.5440 -> 0.9872` span the lever closes, computed as
  `(lever own-lattice best - 0.5440) / (0.9872 - 0.5440)` on the M27 matched population.

**If the lever does not improve**, the pass must record which of two causes holds, from the
committed numbers rather than from narrative:

| cause | condition |
|---|---|
| **`features-unavailable-at-own-lattice`** | the pixel read-out on this arm's co-scoreable cells keeps its M27-level F1 (>= `0.90`), while the arm's own read does not move: the information exists at the placement the arm used, and the learned head still failed to exploit it |
| **`head-cannot-exploit`** | the arm's own read moves by less than the `0.05` floor while its oracle read also stays within `0.05` of the control's oracle read |
| **`placement-only`** | the arm's own read does not improve but its oracle read beats the control's oracle read by `>= 0.05`: the added features help only once registration is granted |

A missing or failed receipt yields no attribution.

## Receipts this pass must produce

1. `runs/m28-pixelread-{base,lever}-r1-profile-r1` (2 steps, 1,152 MiB / 300 s / 2 work items).
2. `runs/m28-pixelread-{base,lever}-r1` (64 steps, 1,152 MiB / 1,200 s / 64 work items), each
   admitted only by its own profile.
3. `runs/m28-pixelread-*-r1-sweep-r1{-profile-r1,}` (2 then 54 forwards, 640 MiB).
4. `runs/m28-pixelread-*-r1-registration-r1` (36 forwards, 640 MiB).
5. `runs/m28-pixelread-r1-read-r1-job.json` (the reader, 128 MiB) writing
   `docs/evidence/m28-pixel-query-r1/scope.json` after its joins hold.
6. The owned family `m28-pixelread-*` declared in `record_stream_run.OWNED_RUN_PATTERN` and bound
   by a test.

## Limits, stated before the measurement

- Synthetic, single seed, training-fit boards; no real-image, calibration, generalization or
  complete-state claim, and the fit gate is not re-opened.
- The matched population is a subset of the deciding cells; every matched number is reported beside
  its full-population counterpart.
- The pixel read-out is a ceiling with the target lattice granted; it is not a candidate model.
- A 64-step continuation can only show that the block is used at this budget, not that the
  representation is sufficient at a longer one.

## Measured

Receipts: `runs/m28-pixelread-{base,lever}-r1` (64 steps each, admitted by their own 2-step
profiles; measured peaks 1,058,054,144 and 1,054,588,928 bytes against the 1,152 MiB ceiling),
`...-sweep-r1` (54 forwards),
`...-registration-r1` (36 forwards), and the reader `runs/m28-pixel-query-r2-job.json` at 128 MiB.
Joins passed: the reader's recomputed full-population F1 at `0.0`, AUC and best swept F1 match each
sweep's `deciding` block exactly; the arm's weights are the ones its sweep and probe measured; the
population is the declared 36 boards / 28 nonempty / 667 occupied / 4,933 empty.

### Own lattice, full deciding population (the standing metric)

| arm | AUC | F1 at `0.0` | best swept F1 | at |
|---|---|---|---|---|
| control `m28-pixelread-base-r1` | 0.742741 | 0.284833 | 0.433768 | 3.90 |
| lever `m28-pixelread-lever-r1` | **0.860163** | **0.387588** | **0.461959** | 1.95 |
| delta | **+0.117422** | **+0.102755** | **+0.028191** | |

The lever's `0.387588` at `0.0` lands on the published control's `0.381640`, but its ranking is
better than that arm's `0.822643` AUC, with a best swept F1 between the published `0.392496` and the
M27 matched `0.5440`.

### The same arms on their own co-scoreable cells, registration granted

| arm | read | AUC | F1 at `0.0` | best swept F1 |
|---|---|---|---|---|
| control | own pairing | 0.6237 | 0.1613 | 0.2801 |
| control | oracle registration | 0.3445 | 0.1316 | 0.2194 |
| control | pixel read-out, same cells | 0.9991 | 0.9743 | 0.9889 |
| lever | own pairing | 0.8922 | 0.3430 | **0.5079** |
| lever | oracle registration | 0.8562 | 0.3153 | **0.5345** |
| lever | pixel read-out, same cells | 0.9982 | 0.9565 | **0.9771** |

### Verdict: **NOT CLEARED**

All three declared conditions fail on the full deciding population: per-cell F1 at `0.0` is
`0.387588` against the `0.70` bar; the ranking verdict is `uninformative` (`0.860163` against
`0.90`); the best-swept delta is `+0.028191` against the `0.05` floor. The block is nonetheless
used and is not a null: `+0.117` AUC and `+0.103` F1 at `0.0` is the largest read-side movement
this campaign has measured at 64 steps, and on the matched cells the lever's own read is `+0.228`
best swept F1 above its control's.

### Residual attribution, and why the declared rule misfires here

The predeclared rule assigns `placement-only`, because the lever's oracle read beats the control's
by `+0.315070` (>= `0.05`). **That assignment is recorded together with its disqualifying
measurement, not instead of it:** the control's oracle read (`0.219413`) is *below* its own-lattice
read (`0.280088`), so on that arm granting registration cannot be an upper bound and the cross-arm
oracle delta says nothing about placement. The within-arm reads, which need no such assumption:

- lever registration gain `0.534483 - 0.461959 = +0.072524`;
- lever representation gap `0.977120 - 0.534483 = +0.442637`.

So `residual_within_arm = representation`: with the block present, the reference registration
moves the read by about `0.07` while `0.44` of span remains between the registration-granted read
and the pixel read on the same cells. The M27 span closure is `-0.081311` and is **not** evidence
against the block: it compares the lever's matched read (`0.5079`) with the *different* M27 arm's
matched read (`0.5440`) on a per-arm cell set, while this arm's own control sits at `0.2801` on the
same cells.

### What this establishes, and what it does not

- The added feature class is *available* at the model's own lattice: the pixel read keeps `0.9565`
  at `0.0` on the lever's cells, so the features are not lost to placement.
- A 64-step continuation with a zero-initialized `30 -> 192` projection recovers roughly a third of
  the own-lattice F1-at-`0.0` deficit but only `0.028` of best swept F1, far short of the `0.32`
  the standing gate would need: feeding the read-out better features is **not sufficient by
  itself** at this budget, which is the falsifiable outcome this pass was built to produce.
- Not shown: a longer or wider block, a different budget, real images, or that the representation
  is sufficient at any budget. One seed, training-fit boards, no generalization claim.
