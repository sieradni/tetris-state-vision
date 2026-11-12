# Occupancy placed targets — supervise where the tap samples

Status: predeclared (2026-11-12) and committed before any profile, study, sweep, field, position or
evidence directory for this instrument exists. The change, its matched control, the envelope, the
population, the gates, the three readings and the position statistic below are fixed; the measured
section is appended only after the canonical receipts exist.

## Where this comes from

[M3_OCCUPANCY_REGISTRATION.md](M3_OCCUPANCY_REGISTRATION.md) closed the reach question and then the
constant-mode question:

| quantity | control | one cell | wide | de-meaned |
|---|---|---|---|---|
| deciding AUC | 0.822643 | 0.866566 | **0.882262** | 0.820429 |
| best swept F1 | 0.412034 | 0.490752 | **0.514821** | 0.427911 |
| field constant-share | — | 0.890 | 0.978 | **0.000752** |
| mean `|offset|` cells | — | 0.909 | 3.760 | **0.541** |

The de-meaned arm proved the field can be per-cell and still buy nothing: with the constant mode
removed by construction, the field moves occupied cells by 1.153 cells on average and the ranking is
indistinguishable from control. So per-cell displacement is **expressible** and the occupancy objective
carries no information about **which** displacement is correct. The previous pass named the next
structural change: supervise the displacement instead of asking the occupancy term for it.

## The single declared change

`--occupancy-placed 1`, available only with `--occupancy-deform 2` and `--occupancy-detail 1`, adds one
term to the objective:

```
wanted(c) = clip( (target_lattice(c) - predicted_lattice(c)) in cell units , -DEFORM_REACH, +DEFORM_REACH )
placed    = masked_mean( smooth_l1( applied_field(c) , wanted(c) ) , known(c) )
total    += 0.25 * placed
```

- `applied_field` is the offset the detail tap actually applied (`occupancy_deform_offsets`), so the
  term supervises the same tensor the forward uses, not a copy.
- Both lattices are **detached**: the box is the geometry term's business, and this term must only ever
  move the field. The test asserts the target carries no gradient and that the field's tensors do.
- The target is clipped to the arm's own reach (4.0 cells), so a lattice further away than the field can
  move never asks for an impossible sample.
- The weight is `0.25`, the scale the semantic, visibility and pose heads already use. Declared before
  any receipt: this term names where a sample belongs and must not be able to dominate the objective.
- The arm **adds no tensors**. The field is the published widened field (fine + coarse, zero-initialised,
  clamped at four cells), so the comparison isolates the supervision and not the capacity, step zero is
  the published function bit for bit, and the allowlist constant is unchanged.

Why this adds information: the occupancy label at cell `c` says what that cell contains; it cannot say
that the tap's sample for `c` physically landed somewhere else. `wanted` does say that, per cell, from
the target lattice - so it is the first term in this family that can prefer the correct displacement to
a re-centred one. It is also the only honest way to tell the two live explanations of the free field's
`5.128`-cell translation apart: if the translation is a decode correction of about 2.6 cells, a
supervised field should adopt roughly that and keep the wide arm's gain; if it is a shifted-window
exploit, a supervised field will not.

## Envelope and declared work

| stage | work items | ceiling | wall cap |
|---|---|---|---|
| profile (per arm) | 2 | 1,792 MiB | 300 s |
| study (per arm, 64 steps) | 64 | 1,792 MiB | 1,200 s |
| sweep profile (per arm) | 2 | 640 MiB | 300 s |
| sweep (per arm, 54 forwards) | 54 | 640 MiB | 600 s |
| field instrument (per deformation arm) | 36 | 640 MiB | 600 s |
| position instrument (per arm) | 36 | 640 MiB | 600 s |
| curation | 9 | 1,900 MiB | 600 s |

Four arms, one seed (`20261026`), `stream-v1`, capture side 512, learning rate `0.001`, initialized from
the published checkpoint `runs/m3-percell-balance-r1`: `control` (`deform 0`), `wide` (`deform 2`),
`demean` (`deform 3`) and `placed` (`deform 2` + placed), one catalogue. No ceiling, gate or population
is changed; the study ceiling stays at the published 1,792 MiB and the suite runs at 2,048 MiB.

## Scoring population and gates

The widened deciding population as published: 36 training images, 28 nonempty boards, 667 occupied and
4,933 empty cells, twelve ancestry clusters, scored read-only on the sweep's 81-point grid. The gates are
the standing ones - per-cell F1 at `0.0` `>= 0.70`, ranking AUC `>= 0.90` and swept F1 `>= 0.70`, and
attribution `>= 0.05` best swept F1 over the named control. **Reproduction is a condition of publication:**
`control` must return `0.822643...` and `wide` `0.882262...` exactly, and `demean` must return
`0.820429...`; a discrepancy is the finding and is chased before anything is published.

## The three declared readings (first match wins)

1. **`supervision-is-the-lever`** - `placed` beats `control` by `>= 0.05` best swept F1 *and* beats
   `wide` by `>= 0.05`, with the `placed - wide` AUC interval clear of zero. Then displacement
   supervision is the lever this architecture carries and the next pass spends it on a larger budget.
2. **`supervision-moves-without-ranking`** - `placed` clears the attribution floor against `wide` but the
   AUC delta is not resolvable at twelve clusters. Then registration is learnable and still not the
   ranking signal; the next step is a larger budget on this pair, declared separately.
3. **`objective-class-fails`** - `placed` does not clear the `0.05` floor against `wide`. Then the
   objective class, not its reach, bound or field, is what fails: occupancy-only supervision and
   placed-target supervision of this field both cannot make per-cell registration pay. The next step is
   named from the evidence (a different read-out, or placed targets in a different space), never another
   reach or bound change.

A clean negative publishes as such. The fit gate is unaffected unless 3 of 28 exact boards appear.

## The position instrument (declared read-only)

`scripts/measure_deform_position.py` re-projects every arm's applied tap offsets into image space and
reports, over the same 36 images:

| statistic | definition |
|---|---|
| `in_frame_fraction` | samples whose re-projected point lies inside the source frame in both axes |
| `in_board_fraction` | samples landing inside the target lattice's own `0..columns x 0..rows` frame |
| `mean_landing_cells` | mean landing point in target-cell coordinates, per axis |
| `mean_shift_cells` | mean `\\|applied offset\\|` per sample, in cells (the field instrument's quantity) |
| `mean_square_shift_cells` | mean squared applied offset, summed over axes |
| `shift_coherence` | `\\|mean applied vector\\|^2 / mean_square_shift_cells` |

`shift_coherence` is re-derived by the curator from the stored mean vector and mean square, exactly as
`constant_share` is in the field instrument, so a statistic that reads a constant regardless of the data
cannot ship (V-106 class). Both instruments are read-only: they reload the frozen archive, forward, and
refuse if any weight changed.

Its job is to say whether the wide arm's `5.128`-cell translation is a decode offset or a shifted-window
exploit at the frame edge, and whether the placed arm's displacement lands on real structure rather than
outside the board.

## Measured (2026-11-12) - placed supervision loses to the objective it competes with

Four arms were admitted inside 1,792 MiB and completed 64 steps under one catalogue from the published
checkpoint. **Reproduction held exactly**: control `0.8226433610683003`, wide `0.8822623454135491` and
de-meaned `0.8204288287642111` AUC, with per-cell F1 at `0.0` and best swept F1 identical to
`docs/evidence/m3-occupancy-deformation-r3/scope.json`. Evidence:
`docs/evidence/m4-placed-supervision-r1/scope.json`, curated by `runs/m4-occplaced-curation-r3-job.json`
(passed, 9 work items).

| quantity | control | wide | de-meaned | **placed** |
|---|---|---|---|---|
| deciding AUC | 0.822643 | **0.882262** | 0.820429 | 0.762448 |
| F1 at `0.0` | 0.381640 | **0.385902** | 0.376663 | 0.377663 |
| best swept F1 | 0.412034 @ `0.90` | **0.514821** @ `1.95` | 0.427911 @ `0.60` | 0.378316 @ `0.15` |
| precision at `0.0` | 0.236325 | **0.240955** | 0.235895 | 0.233701 |
| out-of-fold F1 | 0.400399 | **0.498413** | 0.427911 | 0.378316 |
| exact nonempty boards | 0 of 28 | 0 of 28 | 0 of 28 | 0 of 28 |
| study peak / wall | 1,047,506,944 B / 322.7 s | 1,059,835,904 B / 317.0 s | 1,083,195,392 B / 191.4 s | 1,038,139,392 B / 382.9 s |

**Verdict `not-cleared` on all three conditions**, and the declared reading is **`objective-class-fails`**:
placed is `-0.136505` best F1 against wide, nowhere near the `+0.05` floor.

| contrast | best swept F1 | AUC | F1 at `0.0` | out-of-fold F1 | attribution |
|---|---|---|---|---|---|
| placed vs control | -0.033718 | **-0.060196** (-0.116936 .. -0.015133, MDE 0.076290) | -0.003977 | -0.022083 | not met |
| placed vs wide | -0.136505 | **-0.119815** (-0.208163 .. -0.052118, MDE 0.115599) | -0.008239 | -0.120097 | not met |
| placed vs de-meaned | -0.049595 | -0.057981 (-0.140698 .. +0.015318, MDE 0.110477) | +0.001000 | -0.049595 | not met |
| wide vs control | +0.102787 | +0.059619 (+0.029411 .. +0.094463, MDE 0.049478) | +0.004262 | +0.098014 | met |

Both of placed's AUC losses - against control and against wide - are **resolvable** (intervals clear of
zero): supervision of the displacement did not merely fail to help, it made the occupancy decision
significantly worse while the same field without supervision still reproduced its published gain.

### What the supervision actually did

| | wide | de-meaned | **placed** |
|---|---|---|---|
| mean `\|offset\|` cells | 3.760 | 0.541 | **0.462** |
| constant-share | 0.978 | 0.000752 | **0.892** |
| board mean of the applied field | (-3.488, -3.760) | (-0.001, +0.024) | **(-0.388, -0.384)** |
| at or near the bound | 83.8% | 0.0% | 0.0% |
| samples outside their own cell | 99.8% | 36.5% | 39.4% |

The placed term engaged - the field became **89.2% one constant vector**, the only constant-shaped field
in the family besides the free arm's - but it moved only **0.462** cells against a target displacement of
about **2.6** cells, i.e. the field reached roughly a sixth of what it was told. Two things follow, and
both matter for what comes next:

1. **The supervision is weaker than the objective.** The same zero-initialised convolution, over the same
   64 Adam steps at the same rate, was moved `3.760` cells by the occupancy gradient and `0.462` cells by
   the placed term at its declared `0.25` weight. The auxiliary scale that keeps the term from dominating
   is, in this budget, also too small to express the displacement it declares.
2. **The displacement it did adopt hurt.** A `0.46`-cell constant push is neither the published sampling
   nor the correction, and it costs `-0.060196` AUC against control with an interval clear of zero. Being
   nudged a sixth of the way toward the right place is worse than not being nudged.

### Where the samples land (the new instrument)

| arm | in frame | in the target board | outside both | mean landing (column, row) cells | mean shift |
|---|---|---|---|---|---|
| control | 1.000 | 0.710 | 0.290 | (3.891, 9.518) | 0.000 |
| **wide** | **0.950** | **0.359** | **0.641** | (-0.261, 5.496) | 3.760 |
| de-meaned | 1.000 | 0.777 | 0.223 | (6.578, 8.081) | 0.541 |
| placed | 1.000 | 0.612 | 0.388 | (3.377, 8.663) | 0.462 |

This is the answer to the question the last pass left open. The wide arm's `5.128`-cell translation is
**not a decode correction**: it moves **5.0% of samples out of the image frame** and leaves **64.1%** of
them outside the target board's own cell frame, against 29.0% for control. The gain that made the wide arm
the family's best is bought by reading off the board - a shifted window in which the occupancy label
correlates with stack height rather than with the cell. The de-meaned arm, whose displacement is per-cell
by construction, is the only arm that *improves* board coverage (0.710 -> 0.777) and it gains nothing,
which is the same conclusion from the other side.

### The verdict, stated plainly

**Per-cell registration is not a lever this architecture carries through this occupancy read-out, and the
reason is now measured rather than inferred.** Occupancy-only supervision finds a field that is per-cell
and useless (de-meaned, `0.820429`). Supervising the displacement toward the correct cell finds a field
that is constant and too weak, and pays `-0.060196` AUC for the part it does apply (placed, `0.762448`).
The objective's own preference, worth `+0.059619` AUC with an interval clear of zero, puts 64.1% of the
samples off the board. A single translated sample per cell cannot be both correctly placed and
predictive, because the label it is scored against is dominated by how much material sits below it.

**The structural change the evidence points to is the read-out, not the field.** The occupancy head needs
cell-relative vertical context - a per-column stack profile, or a tap that sees the column below the cell
rather than one point - so that the only displacement the objective can reward is not a shift off the
board. That is a different read-out, and it is the next declared instrument. Two consequences are recorded
for whoever declares it: a placed-target term at a weight that can actually move the field (the same
`0.25` could not, by two orders of magnitude against the occupancy gradient) is worth measuring only after
the read-out changes, because the target displacement it enforces is exactly the one the current objective
pays to avoid; and no further reach or bound change can help, since the de-meaned arm already shows bound
saturation is not the binding constraint.

**Fit gate: FALSE, unchanged** - 0 of 28 exact boards on all four arms. No gate, threshold, population or
memory ceiling was altered, and the `0.25` placed weight was published before the receipts and not moved
after them.

## Limits, stated before the measurement

- 64 further steps from a published checkpoint at a 36-image deciding population cannot establish a
  real-image or complete-state claim, and the fit gate is expected to stay closed.
- Twelve ancestry clusters give a minimum detectable best-F1 effect near `0.05`; a smaller movement is
  reported as unresolved, never as a null.
- The placed target is derived from the target lattice, which the synthetic corpus supplies. It is
  supervision the corpus can give and a real screenshot cannot, so a positive result would be a
  statement about what the architecture can learn, not about deployable labels.
