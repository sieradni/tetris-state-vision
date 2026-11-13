# M23: a scale-invariant local-contrast read for the occupancy model

Declared before the arms ran; the measured section below is added from the curated catalogue and is not edited
afterwards.

## Why this mechanism, and not another tuning pass

M22 measured a coarse five-tap stencil on the free channel and localised its failure precisely: `local alone` ranked
`0.634832` against the pooled channel's `0.781555`, adding it *lowered* the free channel, and `82%` of its own magnitude
was a board-wide level. Removing that level after the fact (mode 2, de-meaned over the visible board) dropped the primary
from `+0.099136` to `+0.033170`. The diagnosis is therefore not "the stencil needs a scale" but "the read was mostly a
level" - and a level removed by subtraction is still a read that spends its capacity producing one.

This pass replaces the operation rather than its output: each coarse box is expressed against the **mean and spread of its
own 5x5 neighbourhood**, so an offset cancels in the difference and a rescaling cancels in the ratio. The read carries
only relative structure *by construction*, before the head sees anything.

## The formulation, and why this one

For a coarse map `grid` (channels x rows+2 x cols+2, obtained by dividing the capture into equal boxes - the fixed
lattice, never the decoded one):

```
shifted   = grid - grid.mean()                       # conditioning only; cancels exactly everywhere below
mean      = avg_pool(shifted, 5x5, replicate pad)    # each box's own neighbourhood mean
square    = avg_pool(shifted^2, 5x5, replicate pad)
spread    = sqrt(max(square - mean^2, 0))
delta     = shifted - mean
contrast  = clamp(delta / where(spread > 0, spread, 1), -20, +20)
```

Three choices are load-bearing and each was forced by a measurement or by an algebra requirement:

1. **Divide by the neighbourhood spread, not by a constant.** A mean subtraction alone is *centred*, not *scale free*: a
   board twice as bright would read twice as strongly. Dividing by the local spread makes the statistic invariant to both.
2. **No additive epsilon in the denominator.** The obvious form `delta/(spread + eps)` breaks exact invariance under
   rescaling, because the floor does not scale with the numerator. Measured on a random coarse map: `1.3e-2` of drift at
   `grid*40` with an epsilon floor, against `1.2e-6` without one. A perfectly flat neighbourhood has zero contrast by
   definition, so its denominator is replaced by one where `delta` is exactly zero anyway.
3. **A clamp at `±20`, declared as a numerical guard.** It is applied after the ratio, is itself scale and offset
   invariant, and is inactive on this population (largest observed contrast `2.403`). If it bit, it would silently
   reintroduce the dependence the read exists to remove, so its inactivity is a test.

The statistic is also conditioned by the map's own global mean before the pooled pass. That shift cancels in every
difference below, so it changes nothing mathematically; it keeps the two-pass variance away from the catastrophic
cancellation an unshifted board with a large offset suffers (measured: `1.6e-3` of read drift at an offset of `12.5`
without it, `2.7e-6` with it).

The five taps are then taken **on the contrast field** - the cell's own box and its four neighbours - and concatenated
with the cell's row+column embedding, through the same two-layer head the M22 modes use, with only the output projection
zeroed (V-143) and the head constructed last so every other arm draws identical values. The read is de-meaned over the
**visible board** (the M22 lesson: never include the spawn rows in the window that gets scored).

## What is proved mechanically, not asserted

`tests/test_stencil_read.py::ContrastReadTest` holds each of these on the real function:

| property | measurement |
|---|---|
| identity at step zero | the read is exactly `0.0` and the arm's output is bit-identical to the same arm without the stencil |
| leaves the zero fixed point | output projection non-zero after one optimizer step, hidden layer carries gradient on the next |
| no board level in the scored window | `abs(mean)/mean(abs(read))` over the visible rows is `0.0` |
| invariant to a board-wide offset | `<= 3.9e-6` drift for offsets `+0.4`, `-3`, `+12.5` |
| invariant to rescaling of all coarse values | `<= 1.2e-6` drift at `x2.5`, `x0.01`, `x40` |
| a genuine local change moves it | `>= 1e-2` drift for a 3x3 box increment and for a 2-column roll |
| a uniform board has no contrast | exactly `0.0` at every level tried |
| the read through the head is invariant too | `<= 1e-6` between `grid`, `grid+0.7`, `grid*3.0`, `grid*3.0+0.7` |

The invariance is a statement about the **coarse map**, not about pixels: the encoder is non-linear, so a brightness
change in the capture is not an affine change in the map, and claiming read-level invariance under pixel brightness would
be false. What the channel declares is that the *detail it carries* has no level, and that is what is proved.

## Arms

Same protocol as M22, unchanged: one catalogue, one seed, 64 steps, 54 forwards, read-only at the standing `0.0` boundary
over the standing 81-point grid. The eight M22 arms are re-used from their own receipts (the new mode is inert for them,
proved below), and two new arms are trained:

| arm | `lattice_free` | `stack_monotone` | `crossing` | `free_stencil` |
|---|---|---|---|---|
| `stencil` (record best, M22) | 2 | 1 | 0 | 1 |
| `centred` (M22 de-meaned) | 2 | 1 | 0 | 2 |
| `norm` | 2 | 1 | 0 | 3 |
| `norm-cross` | 2 | 1 | 1 | 3 |

Bar unchanged: `+0.05` F1@0 over the reproduced control **and** at least one strict exact board.

## What the two contrast arms measured

Same catalogue, same seed, same populations, same 64 steps and 54 forwards as M22. The control reproduces
`0.8226433610683003` / `0.381640` exactly and the `envelope` and `crossing` arms reproduce the M20 and M21 checkpoints
bit-for-bit inside this catalogue, so the harness is the one that measured the record.

| arm | AUC | F1@0 | best F1 @ level | positives at 0.0 | exact boards | row error | out-of-fold | reading |
|---|---|---|---|---|---|---|---|---|
| control | `0.822643` | `0.381640` | `0.4120 @ 0.90` | `49.95%` | 0 / 28 | `+7.819` | `0.400399` | — |
| stencil (M22 best) | `0.917020` | `0.480776` | `0.6053 @ 0.90` | `35.93%` | 0 / 28 | `+0.404` | `0.605314` | stencil-moves-without-clearing |
| centred (M22 de-meaned) | `0.859823` | `0.414810` | `0.4765 @ 1.80` | `45.00%` | 0 / 28 | `+1.007` | `0.467641` | stencil-moves-without-clearing |
| **norm** | `0.856281` | `0.414810` | `0.4679 @ 2.25` | `45.00%` | 0 / 28 | `+1.007` | `0.440809` | **stencil-moves-without-clearing** |
| norm-cross | `0.835742` | `0.377106` | `0.4435 @ 1.20` | `49.55%` | 0 / 28 | `+1.925` | `0.387452` | stencil-fails |

### The verdict, stated plainly

**The scale-invariant contrast read does not clear the standing bar.** `norm` moves the primary by **`+0.033170`** - below
the `+0.05` floor - and its AUC by `+0.033637`, and it produces **no strict exact board** (0 of 28, against a bar of at
least one). That is `stencil-moves-without-clearing`, not a pass, and it is the same reading the de-meaned M22 arm earns.
`norm-cross` moves the primary by **`-0.004534`** (AUC `+0.013099`) and is `stencil-fails`.

Against the record's best arm the gap is unambiguous: `norm` is `0.856281` / `0.414810` against `stencil` `0.917020` /
`0.480776`. Removing the level *before* the head instead of after it changes AUC by `-0.003542` and the primary by
exactly `0.000000` relative to the de-meaned M22 form (`centred` `0.859823` / `0.414810`). On this evidence the M22
level was not capacity that a better operation could reclaim: the M22 gain *was* the level, and no formulation of local
detail tested here reproduces it without one.

### Where this differs from the M22 reading, and where it does not

Two measured facts separate the contrast read from the M22 stencil, and neither is visible in the primary:

1. **Its detail is stronger than the pooled channel where it matters.** `local_only` is `0.696297` against
   `free_without_local` `0.792135` in `norm` (`-0.095838`), but in `norm-cross` the local part **beats** the pooled
   channel - `0.742764` against `0.654209`, `+0.088555` - which no M22 formulation achieved (`stencil` `-0.146723`,
   `centred` `-0.101265`).
2. **It composes with the crossing rule where the M22 stencil destroys it.** `norm-cross` is the only arm in either
   catalogue with a `complementary` channel verdict: the summed read beats its own lattice part by `+0.027048` AUC. And it
   is far better than the other local forms under the rule (`stencil-crossing` `0.782265` / `0.311695`,
   `centred-crossing` `0.736989` / `0.261500`).

Neither is enough to clear the bar: `norm-cross` still lands `0.377106` on the primary, `0.004534` *below* the control,
because the crossing stage takes the level back (`+1.192513` mean level per board) and the arm's ranking advantage
(`0.835742` against the control's `0.822643`) cannot pay for it at `0.0`.

The stage decomposition says the same thing as M22's, more sharply. `norm`'s raw sum ranks `0.759847` at `9.93%`
positive; the `monotone` stage raises the share by `+0.250694` to `35.00%`, taking F1@0 to `0.414810`. The band the arm
occupies between its raw sum and its swept best is smaller than the M22 stencil's (`0.4679` against `0.6053`), the
signature of a read that has detail but no level to spend.

### The proof that the harness and the earlier arms are unchanged

`docs/evidence/m23-contrast-r1-inertness.json` loads one M22 checkpoint into the **archived pre-change source** and into
the current source and compares the read and every recorded part for modes 0, 1 and 2: all identical, in both `occupancy`
and `parts`. The mode-3 branch is therefore inert for every arm that does not declare it, and the eight M22 arms in this
catalogue are the arms that measured M22. `envelope` and `crossing` reproducing the M20 and M21 checkpoints bit-for-bit is
the second, independent check.

### Limits of this reading

- One catalogue, one seed, 36 deciding images. The `+0.033170` is one draw, and the exact-board gate it fails is the
  strict (V-139) one.
- The contrast read is a five-tap statistic at `22` by `12` boxes of a `16x16` coarse map; whether a wider neighbourhood,
  a second scale, or contrast computed on the detail map rather than the coarse map behaves differently is not measured
  here.
- The invariance proved is invariance of the *detail field* under affine changes of the coarse map. It is not invariance
  of the whole read under a change of lighting in the capture, because the encoder between them is non-linear; that was
  never claimed and is not claimed now.
