# M22: the free channel's coarse local stencil, and the in-loop versus post-hoc crossing ablation

Declared before the block ran. Numbers are added in *What the six arms measured* once the catalogue is curated; nothing
in this section may be edited afterwards to fit them.

## What M21 left open

M21 measured the two-path read under the monotone envelope and the crossing rule, and its verdict was plain: the
`level-passes` reading cleared the standing `+0.05` on the primary (`+0.020547` F1@0, `+0.000957` AUC), and M16's
envelope-with-centring-and-crossing arm is still ahead on every measure. Its stage decomposition is the useful residue:
the `monotone` stage raises the positive share by `+0.220416`, and the `crossing` stage takes `−0.406250` of that back
while lifting F1@0 by `+0.187614` - only when the rule is *trained* in the loop.

That last qualifier is a caveat, not a finding: the block never ran the comparison it rests on. And the free channel
itself is a weak reader - envelope `0.607085` AUC alone, crossing `0.497469` - because it reads image-level pooled
context plus fixed cell coordinates and nothing else. It cannot express "the coarse box above this cell is empty and
this one is not", which is what an occupancy read-out is for.

## The lever: a coarse local stencil on the fixed lattice

A declared field, `occupancy_free_stencil`, valid only beside `--occupancy-lattice-free 2`. When it is on, the
dimension-wide coarse feature map is divided into `rows+2` by `columns+2` equal boxes *of the capture* - the fixed
lattice, never the decoded one - and each cell reads five taps: its own box, the box above, below, left and right. Those
`5*dimension` values are concatenated with the cell's own row-plus-column embedding and passed through
`Linear(6*dimension → dimension), SiLU, Linear(dimension → 1)`. Only the output projection is zeroed (V-143), and the
head is constructed last, so a run without the stencil draws identical random values.

Three thing this formulation deliberately avoids:

- **registration error.** Nothing in the stencil reads the decoded board. The boxes are equal divisions of the capture,
  so a misplaced lattice cannot misplace the stencil; a shift moves every box by the same small amount and a box average
  is insensitive to it. That is why the pixel-local features that failed earlier are not the same instrument.
- **the training fixed point.** Both MLP layers zero-initialised is a stationary point (V-143); only the output is, so
  the hidden layer moves from the first step.
- **an unused parameter.** The head is only constructed when the field is declared, so a run without it carries no
  resident Adam state for it.

The local read is kept in `occupancy_parts["local"]` beside the sum, so a measurement can score the stencil alone, and
score the free channel with the stencil removed *exactly* (`free − local`), rather than by ablation.

## Arms

One catalogue, one seed, one population, 64 steps, 54 forwards:

| arm | `lattice_free` | `stack_monotone` | `crossing_threshold` | `free_stencil` | reads |
| --- | --- | --- | --- | --- | --- |
| `control` | 0 | 0 | 0 | 0 | the published lattice-sampled read |
| `envelope` | 2 | 1 | 0 | 0 | the M20 two-path read under the envelope |
| `crossing` | 2 | 1 | 1 | 0 | the M21 composition |
| `bare` | 2 | 0 | 0 | 0 | the sum without the envelope |
| `stencil` | 2 | 1 | 0 | 1 | the M20 read with the local stencil |
| `stencil-crossing` | 2 | 1 | 1 | 1 | the M21 composition with the local stencil |

`envelope` must reproduce the M20 checkpoint bit-for-bit and `crossing` the M21 one; if either does not, the block is
broken and no number in it counts.

## Bar and readings

Standing bar, unchanged: `+0.05` F1@0 over the control **and** at least one strict exact board at the 0.0 boundary. A
stencil arm records `stencil-passes` if it clears both, `stencil-moves-without-clearing` if it moves the primary but does
not, `stencil-fails` if it does not move it. If an arm clears the bar it is ranked against M16's recorded `0.405439` /
`0.835241`; if it does not, the stage and the channel where the gain or loss appears are named from the decomposition and
the channel split, not summarised as "movement".

Every arm is decomposed as raw sum → `monotone` → `crossing`, at the standing 0.0 boundary, on the standing 81-point grid,
and every free-channel arm is split into `summed`, `lattice_only`, `free_only`, `free_without_local` and `local_only`
(scored through the arm's own chain), with a declared channel verdict (`complementary` if the sum adds `+0.005` AUC over
its lattice part, `one-channel-noise-dominates` / `no-complementary-signal` otherwise).

## The ablation this block exists to settle

The claim M21 rests on is that the crossing rule works because it is *trained in the loop*. The controlled comparison:
`crossing` (trained with the rule) against the same rule applied post-hoc to `envelope`'s own weights, same seed, same
population, same chain up to the crossing stage, then `stack_contiguous` → `otsu_board` on the envelope arm's own cases.
Recorded as `ablation.verdict`: `in-loop-training-is-what-makes-the-rule-work` if the trained arm's F1@0 is higher,
`post-hoc-application-matches-the-rule` otherwise. Whichever answer appears, `docs/M21_LEVEL_COMPOSITION.md`, the handoff
and `docs/ROADMAP.md` are corrected to it.

## What would make this a failure

The stencil adds `~6*dimension²` parameters and one more read of the coarse map; it is not free. If it does not clear the
bar, the failure is localised to the stage that takes the gain back and the channel that does not separate, and the
conclusion is that pooled context plus fixed coordinates is not the free channel's limiter - which would move the
question to where the registration error enters, not to another round of tuning.

## What the eight arms measured

One catalogue, 36 training images / 28 boards / 667 occupied of 5600 known cells, 64 steps, 54 forwards, evaluated
read-only at the standing `0.0` boundary and over the standing 81-point grid. `control` reproduces
`0.8226433610683003` / `0.381640` exactly, `envelope` reproduces the M20 checkpoint bit-for-bit and `crossing` the M21
one, so the two arms this block did not change are the arms it measured before.

| arm | AUC | F1@0 | best F1 @ level | positives at 0.0 | exact boards | row error | reading |
|---|---|---|---|---|---|---|---|
| control | `0.822643` | `0.381640` | `0.4120 @ 0.90` | `49.95%` | 0 / 28 | `+7.819` | — |
| envelope | `0.898066` | `0.350636` | `0.6001 @ 3.30` | `55.52%` | 0 / 28 | `+3.322` | level-fails |
| crossing | `0.823601` | `0.402187` | `0.4473 @ 0.90` | `46.88%` | 0 / 28 | `+1.289` | level-passes |
| bare | `0.883359` | `0.365342` | `0.4790 @ 2.40` | `52.80%` | 0 / 28 | `+7.128` | level-fails |
| stencil | `0.917020` | `0.480776` | `0.6053 @ 0.90` | `35.93%` | 0 / 28 | `+0.404` | stencil-moves-without-clearing |
| stencil-crossing | `0.782265` | `0.311695` | `0.3761 @ 1.20` | `61.54%` | 0 / 28 | `+3.855` | stencil-fails |
| centred | `0.859823` | `0.414810` | `0.4765 @ 1.80` | `45.00%` | 0 / 28 | `+1.007` | stencil-moves-without-clearing |
| centred-crossing | `0.736989` | `0.261500` | `0.4243 @ 2.10` | `66.89%` | 0 / 28 | `+3.278` | stencil-fails |

### The verdict, stated plainly

**The stencil does not clear the standing bar.** `stencil` moves the primary by `+0.099136` - twice the `+0.05` floor -
and its AUC by `+0.094377`, and it produces **no strict exact board** (0 of 28, against a bar of at least one). That is
`stencil-moves-without-clearing`, not a pass. The de-meaned form moves it `+0.033170` and also produces no exact board:
`stencil-moves-without-clearing`. Both composed arms are `stencil-fails`: `stencil-crossing` at `-0.069945` F1@0 and
`-0.040379` AUC, `centred-crossing` at `-0.120140` / `-0.085654`.

Ranked against the record it is still the strongest read this engine has produced on the primary: `stencil` is
`0.917020` / `0.480776` against M16's recorded arm `0.835241` / `0.405439`, with row error `+0.404` against that arm's
`+1.189`. So the lever is real and the bar is not met - the two statements are both true and neither substitutes for the
other.

### Where the gain and the loss live

The stage decomposition names the stage. In `stencil` the read's raw sum already reaches `0.864536` AUC at `10.06%`
positive, and the `monotone` stage raises the positive share by `+0.178888` to `27.94%`, taking F1@0 to `0.480776` and
AUC to `0.917020`. There is no crossing stage to take it back. Compare `envelope`, whose raw sum sits at `14.51%` and
whose `monotone` stage raises the share `+0.286667` to `43.18%` with F1@0 `0.350636`: the stencil's raw sum is both
better ranked and better placed *before* any transform, and the envelope then overshoots where the stencil does not.

The channel split names the channel, and it is not the one the predeclaration hoped for:

| arm | summed | lattice part | free channel | free without local | local alone | sum − lattice |
|---|---|---|---|---|---|---|
| stencil | `0.917020` | `0.921538` | `0.772972` | `0.781555` | `0.634832` | `-0.004518` |
| stencil-crossing | `0.782265` | `0.782597` | `0.596182` | `0.631941` | `0.686111` | `-0.000332` |
| centred | `0.859823` | `0.886644` | `0.761345` | `0.790640` | `0.689375` | `-0.026821` |
| centred-crossing | `0.736989` | `0.736946` | `0.708503` | `0.721457` | `0.641425` | `+0.000043` |

Three facts, all against the hoped-for story:

1. **The local stencil does not improve the free channel's ranking.** `local alone` ranks `0.634832` against the pooled
   channel's `0.781555`, and adding it *lowers* the free channel from `0.781555` to `0.772972`. The stencil's
   `+0.094377` over the control comes from the composition - the free channel's level and the envelope's monotone
   stage - not from local detail out-ranking the pooled context.
2. **What the stencil mostly contributes is a board-wide level.** `largest_level_ratio` for the plain form is `0.82`:
   most of the local read's own magnitude is a constant offset across the board. Applying the constant-mode bar drops
   that to `1e-06` and drops F1@0 from `0.480776` to `0.414810` and AUC from `0.917020` to `0.859823`. The de-meaned
   arms are the controlled test of exactly this: **the primary gain was the level, not the locality.**
3. **The crossing rule cannot use the stencil.** `stencil-crossing` loses `0.069945` F1@0 and `0.040379` AUC against
   the control and `0.169081` against `stencil`; `centred-crossing` loses more. Whatever the rule derives from a board
   is contradicted by the local read, in both formulations, and de-meaning does not rescue it.

### The formulation change this block made after the first number

The first six arms measured the plain stencil. Its diagnosis - a level, not local detail (`local alone` below the pooled
channel, `largest_level_ratio` `0.82`, no exact board, collapse under the crossing rule) - was testable, so the same
stencil was given the constant-mode bar (`occupancy_free_stencil 2`): the local read is de-meaned **over the visible
board**, the population every measurement scores, before it reaches the sum. The first attempt de-meaned over the whole
tensor including the four spawn rows above the visible board, which left up to `40%` of the local read's magnitude as a
level inside the window that is actually read; the curator's `constant_mode_removed` guard caught it, the mean was moved
to the scored window, and both arms were re-measured from scratch. `centred` is the result: the level is gone
(`1e-06`), the primary gain falls to `+0.033170`, and the ranking gain falls to `+0.037180`. So the honest reading of
the whole block is that this formulation of local detail is worth roughly a third of what the plain form appeared to be
worth, and the rest was the level it was shifting.

### The ablation: in-loop versus post-hoc

Same seed, same population, same catalogue; `crossing` trains with the rule, and the same rule is applied afterwards to
`envelope`'s own weights (`stack_contiguous` then `otsu_board`, on that arm's own cases).

| reading | AUC | F1@0 | positives at 0.0 |
|---|---|---|---|
| `envelope`, trained without the rule | `0.898066` | `0.350636` | `55.52%` |
| `envelope` + the rule applied post-hoc | `0.868019` | `0.341542` | `42.63%` |
| `crossing`, trained with the rule | `0.823601` | `0.402187` | `46.88%` |

In-loop minus post-hoc: **`+0.060645` F1@0** and **`-0.044418` AUC**. Verdict:
`in-loop-training-is-what-makes-the-rule-work` on the primary, and only on the primary - the post-hoc application
actually *lowers* its own arm's F1@0 by `0.009094` while raising its AUC by `0.044418` over the arm it was applied to.
So the rule's operating-point benefit is not a post-processing effect: it has to be part of what the read learns. Its
ranking benefit is the opposite: post-hoc application ranks better than in-loop training does, and in-loop training
*buys* its F1@0 by giving up `0.074465` AUC against the arm it starts from. `docs/M21_LEVEL_COMPOSITION.md` said the rule
"re-decides the same place" when applied afterwards; the measurement says it decides a *different and worse* place on the
primary while ranking better, and its wording is corrected to those numbers.

### Limits of this reading

- The eight arms run under two source fingerprints: the six plain-stencil arms predate the de-mean branch, the two
  de-meaned arms follow it. `docs/evidence/m22-stencil-r1-inertness.json` loads one checkpoint into the archived and the
  current source and finds the read and every recorded part bit-identical for modes 0 and 1, so the branch is inert for
  every arm that does not declare mode 2; no earlier arm does.
- One catalogue, one seed, 64 steps, 36 deciding images: the `+0.099136` is one draw of the training stream and the
  exact-board gate it fails is the strict (V-139) one, not a relocated boundary.
- The stencil reads a 16x16 coarse map divided into 22 by 12 boxes of the capture, so a box is roughly 0.7 of a
  coarse token. Finer detail than that is not in this channel, and nothing here says a finer fixed-lattice stencil would
  behave the same way.

