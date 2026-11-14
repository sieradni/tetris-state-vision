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
