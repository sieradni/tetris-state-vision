# M41: the same features, sampled at the support - provenance isolated

Status: **predeclared before any M41 scoring run.**

## The confound this arm removes

M40's pass closed the campaign's question only in part. The closure now says two things, and the second is
the one that left work: (a) no composition measured learned the exogenous placement correction - all seven
exogenous arms sit below the winner on both ranking measures and none clears the bar; (b) **provenance does
not order the arms**, because the "keeps the predicted lattice sampled" and "moves it off" groups overlap on
AUC (`0.886350`-`0.897918`: M36, M38, M40) and best swept (`0.420935`-`0.439535`: M36, M40), with M40
moving the geometry and outranking M36 on both measures (`+0.011568` AUC, `+0.018600` best swept) and M38 on
AUC (`+0.002877`).

The receipts say why that overlap cannot be read as a provenance result: the overlapping pairs differ in more
than provenance. Recorded configs (`runs/<arm>/checkpoint/manifest.json`):

| arm | recorded form | recorded parameters | features | head width | decision rule | sampled geometry |
|---|---|---|---|---|---|---|
| `m30-level-calibrate-r5` (winner) | `3` | `8,023,562` | `30` | `64` | none | predicted lattice |
| M40 `m40-composed-r2` | `14` | `8,023,562` | `30` | `64` | none | predicted, half-corrected |
| M34 `m34-support` | `7` | `8,023,562` | `30` | `64` | none | support lattice |
| M35 `m35-contrast` | `8` | `8,025,482` | `60` | `64` | none | predicted lattice |
| M36 `m36-monotone` | `10` | `8,025,482` | `60` | `64` | one crossing per column | predicted lattice |
| M38 `m38-capacity-r2` | `12` | `8,039,114` | `40` | `128` | none | predicted lattice |

Two of those facts co-vary with provenance in every overlapping pair, so the missing measurement is a cell
of the 2x2 that the recordings define - geometry source (predicted / support) against feature set
(`30` plain statistics / `60` statistics plus their sample-mean deviations) - with head width, decision rule
and every other recorded setting held fixed:

| | predicted lattice | support lattice |
|---|---|---|
| `30` statistics | winner (form `3`) | M34 (form `7`) |
| `60` contrast statistics | M35 (form `8`) | **M41 (form `15`) - this arm** |

**M41 is the missing cell**: M35's declared features (each statistic beside its deviation from the sample's
own mean over the lattice) sampled at the **support lattice** instead of the predicted one. Against **M35** it
changes exactly one recorded fact - where the geometry comes from - at an identical parameter count.

## The mechanism, exactly (form 15)

`--occupancy-pixel-query 15`, one head block, no second selector, no new parameter:

- **Sampled geometry**: `pixel_support_lattice(valid, rows, columns, ...)` - M34's declared exogenous
  lattice, the bounding box of the valid pixels mapped to the board's rows and columns. No fitted, learned or
  board-derived quantity beyond the raster mask.
- **Features**: exactly M35's - the `30` cell statistics concatenated with their deviation from the sample's
  own mean over the lattice, `60` inputs. The deviation is computed from the sample itself, never from a
  population or a scored cell. No stencil fusion, no neighbour mean, no dual placement half.
- **Projection**: `Linear(60, 64) -> SiLU -> Linear(64, dimension)`, zero-initialised output, so step zero is
  bit-identical to the published read.
- **No decision rule**: `occupancy_stack_monotone` stays `0` and form 15 does not apply
  `stack_contiguous`, which form 10 does. That is what keeps M36's monotone rule out of the contrast.
- **Held fixed against M35**: `dimension` `192`, `blocks` `6`, `occupancy_read_standardise` `1`, every other
  lever flag `0`, the same curriculum, the same initial weights, the same `64` steps, the same evaluation
  population and the same fixed global `0.0` boundary. Recorded parameter count is **`8,025,482`**, M35's
  exactly.

## Protocol

One profile, one `64`-step study, one `36`-board probe under the new `m41-support-contrast` owned family
(`1,152` MiB study, `640` MiB probe), scored by `scripts/m37_stencil_read.py --inputs 60 --features 30
--instrument m41-support-contrast-r1 --predeclaration docs/M41_SUPPORT_CONTRAST.md` on the same 2,986 cells
and both fixed label vectors, with the winner re-scored in the same pass. No existing arm is re-scored and no
prior evidence directory is touched.

## Gate thresholds, unchanged

per-cell `0.70` on both fixed label vectors; ranking (AUC `>= 0.90` and best swept `>= 0.70`); attribution
over the unchanged control `>= 0.05`.

## The bar, fixed before scoring

New operating point only if, against the winner's re-scored numbers:

- F1 at `0.0` `>= 0.372881`, and
- best swept F1 `>= 0.689775`.

The verdict is reported as **cleared** / **not cleared** on those two numbers, with the standing gate flags
reported beside it.

## Prediction and discriminating outcome, committed in advance

**Prediction: the geometry, not the features, carries the loss at this depth.** The axis trio is monotone in
both measures (`0.0` -> `0.954529` / `0.689775`, `0.5` -> `0.897918` / `0.439535`, `1.0` -> `0.883260` /
`0.385925`), and M39 shows that having the aligned statistics available did not help. So M41 is predicted to
land with the exogenous group, well below its own comparison partner M35 (`0.919512` / `0.589286`).

The three outcomes below are fixed now, so the reading cannot be chosen after the numbers arrive:

- **REPRODUCES the provenance cost (provenance isolated):** M41 AUC `<= 0.910000` **and** best swept
  `<= 0.480000`. Then geometry provenance alone reproduces the loss when features, width and decision rule
  are held fixed, and the overlap is attributable to the recorded confound - M40's feature set
  (`30`) against M36's (`60`) plus M36's monotone rule, and its width (`64`) against M38's (`128`) - rather
  than to provenance failing to order anything. The closure's negative stands; its demotion must then be
  narrowed to say that provenance *does* order at matched recorded config, and that the overlap is a
  cross-configuration effect.
- **DOES NOT reproduce (provenance insufficient at `60` inputs):** M41 AUC `>= 0.919512` **or** best swept
  `>= 0.589286` - at or above M35 on either measure. Then holding the features fixed removes the geometry
  cost, recorded config no longer explains the overlap, and the closure's demotion stands with the confound
  unresolved; the next arm must isolate the interaction instead.
- **INTERMEDIATE (present but smaller):** anything else. Then report the fraction of the `30`-input cost the
  `60`-input arm retains - `(M41 - M35) / (M34 - winner)` on each measure - and state that the cost is real
  but feature-dependent, which keeps the closure's negative and leaves the overlap's ordering partly
  unexplained.

## Branch interpretation for clearance

If the bar is **cleared**, re-run the probe and reproduce weights and logits (not an earlier delta by
assumption) before calling it an operating point, and record the next structural step against it. If the bar
is **missed** - the predicted outcome - report the gate flags, the deltas against the winner and against M35,
the split closure / placement / representation with the sum asserted, and the diagnosis in the same terms the
last three passes used (false-positive counts, margins, the same row bands). Capacity is not available as an
explanation and will not be offered.

## Receipts

`runs/m41-support-contrast{,-profile,-registration}` and
`docs/evidence/m41-support-contrast-r1/{scope,diagnosis}.json`. Refused attempts keep their receipts; only
completed runs are quoted.
