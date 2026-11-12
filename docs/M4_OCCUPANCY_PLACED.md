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

## Limits, stated before the measurement

- 64 further steps from a published checkpoint at a 36-image deciding population cannot establish a
  real-image or complete-state claim, and the fit gate is expected to stay closed.
- Twelve ancestry clusters give a minimum detectable best-F1 effect near `0.05`; a smaller movement is
  reported as unresolved, never as a null.
- The placed target is derived from the target lattice, which the synthetic corpus supplies. It is
  supervision the corpus can give and a real screenshot cannot, so a positive result would be a
  statement about what the architecture can learn, not about deployable labels.
