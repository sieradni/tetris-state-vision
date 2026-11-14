# M40: the predicted lattice, corrected by the support

Status: **predeclared before any M40 scoring run.**

## What this pass is for, and a defect it caught before running

The campaign's recorded next step (M39, `docs/M39_DUAL_PLACEMENT.md`) was to keep the model's predicted
lattice as the sampled geometry and apply the support only as a declared alignment composed with it, so the
read stays a function of what the network predicts while gaining the raster's correction.

**The literal formulation of that is a duplicate, and this document records why before spending a run on
it.** An affine lattice mapped affinely onto the support box *is* the support lattice: the predicted box
becomes the support box, the column and row vectors scale to it, and the result equals
`pixel_support_lattice` for every board, whatever the predicted lattice was. The composition as first
declared therefore collapses onto M34's arm, and measuring it would have produced a second copy of
`0.274869` / `0.883260` / `0.385925` while claiming to test something new. V-165 records the defect.

The form below is the same idea made non-degenerate: a declared **convex combination**, so the sampled
geometry is half the predicted lattice and half the aligned one.

## The mechanism, exactly (form 14)

`--occupancy-pixel-query 14`, one head block, no second selector, no new parameter:

- **Sampled geometry**: `pixel_composed_lattice(lattice, valid, rows, columns, device, dtype)`, which
  returns, per board and per lattice vector, `(1-w) * predicted + w * aligned`, with
  `PIXEL_COMPOSED_WEIGHT = 0.5` declared as a fixed constant (not learned, not selected, not tuned).
  `aligned` is the affine lattice that carries the predicted board box onto the bounding box of the valid
  pixels, computed exactly as M34's helper does.
- **Properties the tests pin**: it is the identity when the support already fills the capture (so step zero
  and any full-frame board are untouched), and on a padded capture it is neither the predicted lattice nor
  the support lattice - the two endpoints M39 measured.
- **Features and projection**: unchanged - the `30` statistics, `Linear(30, 64) -> SiLU -> Linear(64,
  dimension)` with a zero-initialised output. Against the winner the form adds **no parameter at all**
  (`653,243` either way): `0.5*(predicted+aligned)` is arithmetic, and the gradient still reaches the
  decoder's lattice through the predicted half.
- **Boundary**: the consumer's fixed global `0.0`. Nothing fitted, no label, no prevalence, no scored cell.

## Protocol

One profile, one 64-step study, one 36-board probe under the `m40-composed` owned family (1,152 MiB study,
640 MiB probe), scored by `scripts/m37_stencil_read.py --inputs 30 --instrument m40-composed-r1
--predeclaration docs/M40_COMPOSED_ALIGNMENT.md` on the same 2,986 cells and both fixed label vectors, the
winner re-scored in the same pass.

## Gate thresholds, unchanged

per-cell `0.70` on both fixed label vectors; ranking (AUC `>= 0.90` and best swept `>= 0.70`); attribution
over the unchanged control `>= 0.05`.

## The deciding bar, fixed before scoring

New operating point only if, against the winner's re-scored numbers:

- F1 at `0.0` `>= 0.372881`, and
- best swept F1 `>= 0.689775`.

**A refusal means** a half-strength exogenous correction is not learnable at this depth either, and the
campaign can then state - across the two endpoints already measured and this arm between them - whether the
placement degradation is a property of *any* exogenous component in the sampling geometry, rather than of
how strongly it is applied.

## Branch interpretations, pre-committed

- **PASS**: the arm becomes the operating point; re-run its probe and reproduce weights and logits (not an
  earlier delta by assumption), and record the next structural step.
- **FAIL**: report the gate flags, deltas and split, and interpret the result **between the measured
  endpoints** - M34/M7 support lattice (`0.274869` / `0.883260` / `0.385925`) and the winner's predicted
  lattice (`0.372881` / `0.954529` / `0.689775`) - to say whether the exogenous correction is
  present-but-insufficient, absent, or swamped, using false-positive counts, margins and the same row bands
  as M37-M39. Capacity is not available as an explanation and will not be offered.

## Receipts

`runs/m40-composed-r2{,-profile,-registration}` and
`docs/evidence/m40-composed-alignment-r6/{scope,diagnosis}.json`. Every earlier attempt keeps its receipt.

## Measured (form 14, `m40-composed-r2`, scored `m40-composed-alignment-r6`)

| read | F1 at `0.0` | AUC | best swept |
|---|---|---|---|
| `m30-level-calibrate-r5` (winner, re-scored in the same pass) | `0.372881` | `0.954529` | `0.689775` (at `3.6`) |
| **M40 composed alignment** | **`0.260599`** | **`0.897918`** | **`0.439535`** (at `1.65`) |
| delta | `-0.112283` | `-0.056611` | `-0.250240` |
| anchored variant | `0.326074` | `0.836073` | `0.429792` (at `2.1`) |

Gate: per-cell **false**, ranking **false**, attribution **true**; the same three flags on the anchored
variant. Bar not met. Peak commit `1,032,208,384` (study, against `1,152` MiB), probe `564,199,424`, scoring
reader `44` MB; checkpoint `f888685a2d04…`.

Split (closure / placement / representation), sum asserted: `-0.104439` / `-0.009742` / `+0.557387`, i.e.
`-0.235645` / `-0.021982` / `+1.257627` of the `0.443205` span; `short_stage: closure`.

### What the refusal says, in the terms predeclared for it

The arm does not sit where a learned correction would put it. Its error profile is a *partial* copy of the
support-lattice endpoint's, not a blend toward the winner: false positives `1181` at median margin
`1.233860` (M35 `1088` at `1.352717`), weakest true positive `0.168971`, and `910` of those false positives
in rows 0-9 with `136` more in rows 15-19; `textured_empty_separated: false`. Contrast AUC `0.897918` - it
recovers `+0.014658` of the support endpoint's `-0.071268` gap and `0.260599` recovers nothing at `0.0`.

Measured against *both* endpoints, which is what the predeclaration asked for:

| exogenous share of the sampled geometry | arm | F1 at `0.0` | AUC | best swept |
|---|---|---|---|---|
| `0.0` | `m30-level-calibrate-r5` | `0.372881` | `0.954529` | `0.689775` |
| `0.5` | M40 `m40-composed-r2` | `0.260599` | `0.897918` | `0.439535` |
| `1.0` | M34 `m34-support` | `0.274869` | `0.883260` | `0.385925` |

Both ranking measures fall monotonically in the exogenous share, and the half-strength arm sits nearer the
full-strength endpoint than the winner on both (`0.897918` against `0.954529` and `0.883260`; `0.439535`
against `0.689775` and `0.385925`). A correction that was being *learned* would have to beat the endpoint it
corrects from; a substitution shows exactly this - a graded loss, in both metrics, in proportion to how much
of the endogenous geometry it replaces. At `0.0` the ordering is not monotone (`0.260599` is below the
endpoint's `0.274869`): half a correction buys neither the endpoint's coarse placement nor the winner's
fine one.

So: **the placement-consistency signal is present-but-swamped at any strength, and its cost is graded.** The
signal is present (the arm is not the winner, and its split moves the placement term by `-0.009742` while the
representation term rises `+0.557387`); it is swamped because the confident false positives stay in the dense
upper bands at margins above the weakest true positive (`910` in rows 0-9, `136` in rows 15-19, median
`1.233860` against a weakest true positive of `0.168971`), exactly the signature of M34-M39. Capacity is not
offered as an explanation and does not apply: the form adds **no parameter at all** against the winner
(`653,243` either way).

## Defects this pass found

- **V-165** - the literal composition is a duplicate arm (recorded in the predeclaration above): an affine
  lattice mapped onto the support box is the support lattice, so the declared form became a blend instead.
  Caught before any run.
- **V-166** - the first wiring assigned the composed placement *after* the statistics had been sampled, so
  form 14 trained as form 3. Caught by the re-scored checkpoint hash being byte-identical to the anchor's
  (`eddec1fd…` in both), which no distinct arm can produce. Repaired by selecting the placement in the
  lattice chain before sampling; pinned by `tests/test_m40_composed.py::FormTest::test_the_sampled_geometry_actually_follows_the_support`,
  which spies on the declared helper and requires form 14 to call it and form 3 not to. The inert run is kept
  as `runs/m40-composed-superseded-inert` with its receipt, and its scoring as
  `docs/evidence/m40-composed-alignment-r5-superseded-inert`.

## What this pass settles

**The question, answered as far as the measurements go:** an exogenous correction to placement was not
learned at this depth by any composition measured. All seven arms that contain one sit below the winner on
both ranking measures, none clears the gate, and the half-strength arm recovers only `+0.014658` of AUC and
`+0.053610` of best swept against the support endpoint it corrects from while landing `0.250240` of best
swept below the winner. The cleanest evidence is the **strength axis**, where the read is held fixed (the
`30` statistics, the same one-block head, the same protocol) and only the share of the sampled geometry taken
from the support changes:

| support share of the sampled geometry | arm | AUC | best swept |
|---|---|---|---|
| `0.0` | `m30-level-calibrate-r5` | `0.954529` | `0.689775` |
| `0.5` | M40 `m40-composed-r2` | `0.897918` | `0.439535` |
| `1.0` | M34 `m34-support` | `0.883260` | `0.385925` |

Ranking falls monotonically in both metrics in proportion to the replaced share - the signature of a
substitution, not of a learned correction.

**What the evidence does not support, and an earlier draft of this section wrongly claimed:** a clean
ordering between arms that keep the predicted lattice and arms that move it. The two ranges overlap, on both
measures - AUC in `0.886350`-`0.897918`, best swept in `0.420935`-`0.439535` - and inside that overlap the
ordering is mixed:

| arm | what it does to the sampled geometry | AUC | best swept |
|---|---|---|---|
| M40 `m40-composed-r2` | moves it (half the support box) | `0.897918` | `0.439535` |
| M38 `m38-capacity` | keeps it (width `128`) | `0.895041` | `0.463551` |
| M36 `m36-monotone` | keeps it (one crossing per column) | `0.886350` | `0.420935` |
| M34 `m34-support` | moves it (all of the support box) | `0.883260` | `0.385925` |

M40, which moves the geometry, sits *above* M36, which keeps it, on both measures (`+0.011568` AUC,
`+0.018600` best swept) and above M38 on AUC (`+0.002877`) while `0.024016` below it on best swept; M34,
which also moves it, sits below M36 on both (`-0.003090` / `-0.035010`). Geometry provenance is therefore not
sufficient to order these arms - they differ in features, width and protocol as well.

**Why the overlap cannot be pushed further with these receipts:** the recorded configuration makes the
overlapping pairs confounded. The winner (form `3`), M40 (form `14`) and M34 (form `7`) each record
`8,023,562` parameters and differ only in where the sampled geometry comes from; M36 (form `10`) records
`8,025,482` - its extra `30` inputs, and it also carries a one-crossing decision rule - and M38 (form `12`)
records `8,039,114`, its wider head plus `40` inputs. Every other recorded setting (`dimension` `192`,
`blocks` `6`, all lever flags `0`, same curriculum, initial weights, `64` steps, same population) is
identical. So no overlapping pair varies provenance alone, and the overlap is consistent with feature count,
head width or the decision rule rather than with geometry provenance. That is an open question for the next
step, not a result: see the same table in `ROADMAP.md`.

**The genuine gap this leaves:** what the campaign measures is a *graded cost of replacing the endogenous
geometry within one read*, and a refusal of every composition tried, not a family boundary between two kinds
of arm. The mechanism statement - that the endogenous sampling path is what the winning arm uses to correct
its own placement - remains the best-supported account of the strength gradient and of M39 (which sampled the
support *and* carried the predicted-lattice statistics beside it, and still scored lowest of the seven at
`0.836907` / `0.351779`, so the missing piece was not knowledge of the alignment). It is not established as
an ordering principle, because the overlap contains a moving arm above two keeping arms. The closure is
unchanged in what it refuses - no composition learned the correction, none came near the bar - and narrowed in
what it explains.
