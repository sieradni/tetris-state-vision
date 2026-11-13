# M21 - the level repair, composed and decomposed

Declared before the arm's receipt. Status at declaration: the probe below exists and is committed evidence from the M20
block's own cases; the M21 arms have not run.

## What is already measured, and what it says

The M20 two-path read (AUC `0.898066`, swept `0.600117`, row error `3.322`) beats the published read on ranking and
placement and *loses* at the fixed boundary (`0.350636` against `0.381640`), with `55.52%` of cells positive against
`49.95%`. The declared reading of that failure was "the read's level moved". A read-only probe of the M20 cases
(`runs/probe_level.py`, receipt `runs/m21-probe-r1-job.json`) reconstructs that arm's read stage by stage from the
recorded per-channel parts - raw sum, then the monotone envelope, then the crossing rule - on the arm's own weights:

| stage | AUC | F1 at `0.0` | positive share | swept optimum |
|---|---|---|---|---|
| raw sum (no envelope) | 0.827538 | **0.429907** | 14.5% | 0.458138 @ `1.20` |
| + monotone envelope | 0.898066 | 0.350636 | 43.2% | 0.600117 @ `3.30` |
| + crossing rule | 0.868019 | 0.341542 | 42.6% | 0.546674 @ `3.15` |

(The probe's share uses all 36 training boards; the curator's uses the 28 non-empty ones, so shares are comparable
inside one table and not across tables.)

Two things follow, and they are the reason this pass is not simply "add the crossing rule":

1. **The stage that re-inflates the level is the monotone envelope, and it does so by construction.**
   `stack_contiguous` is the per-column suffix maximum, which can only raise a cell. Raising cells moves them across a
   fixed boundary, so the number of positives rises whether or not the read got better - here from `14.5%` to `43.2%`
   while the ranking improved. The bookkeeping is exact: the envelope does not add information to the read, it adds mass
   above `0.0`.
2. **The crossing rule does not recover it.** `otsu_board` subtracts each board's own between-group level, so it moves the
   boundary to the split of the read's *mass*. When the mass has already been inflated by a max envelope, that split sits
   near the same place: share `43.2%` -> `42.6%`, F1 `0.350636` -> `0.341542`. A level rule that divides a distribution at
   its own split cannot undo a transform that changed how much of the distribution is on one side.

That makes the composition's outcome predictable rather than open, so the honest declaration has to be about *both*
questions at once: does the composed arm repair the level (the requested arm), and if it does not, is the coupling
itself - the M20 rule that the sum is only declared on the envelope - the defect?

## The block (four arms, one catalogue)

| arm | `occupancy_lattice_free` | `occupancy_stack_monotone` | `occupancy_crossing_threshold` |
|---|---|---|---|
| `control` (published read) | 0 | 0 | 0 |
| `envelope` (the M20 two-path arm) | 2 | 1 | 0 |
| `crossing` (the requested composition) | 2 | 1 | 1 |
| `bare` (the sum without the envelope) | 2 | 0 | 0 |

Everything else is `0` in all four arms, including every supervision, anchor and read lever. Each arm is trained for the
declared 64 steps from the published checkpoint, swept over the standing 81-point grid on the standing population, and
scored at the fixed `0.0` boundary. The `bare` arm requires relaxing the M20 coupling (see "Declaration of change"
below); nothing about the other arms changes.

**Numeric bar.** The requested composition `passes` if its plain-binary F1 at `0.0` **exceeds the reproduced control's
`0.381640`** and its AUC is **at least `0.822643` (the control's) minus the standing `0.02` ranking tolerance**. Below the
control's F1 it is `level-fails`, and then the diagnosis below must name the stage. The same bar, applied to `bare`,
decides whether decoupling is the repair.

**Readings:** `level-passes` / `level-without-ranking` (F1 above the control, AUC outside the tolerance) / `level-fails`.
For `bare`: `decoupling-passes` / `decoupling-with-ranking-price` / `decoupling-fails`.

## The decomposition every arm is scored by

For each arm, on its own fold and its own weights, the read is rebuilt from its recorded parts and each declared
transform is applied in the order the model applies it:

- `raw sum` - the sum of the two channels, no transform (for arms without the free channel, the published read).
- `monotone` - `stack_contiguous`.
- `crossing` - `otsu_board`, and for the composition also the level each board was moved by.

Each stage reports AUC, plain F1 at `0.0`, positive share, swept optimum and its threshold. The record then names the
**stage that raises the positive share** (largest share increase in the chain) and whether the crossing stage moved the
share back. This is the structural diagnosis the pass is required to produce, and it is computed, not written by hand.

## Channel split

For the three arms that carry the free channel, the same three readings as M20 - the summed read, the lattice part alone
under that arm's own transforms, and the free channel alone - so the lattice path, the free part and the monotone level
repair are each reported separately on one fold.

## Declaration of change to the M20 coupling

M20 refused `occupancy_lattice_free 2` without `--occupancy-stack-monotone 1` in both config validation and the parser,
with the reasoning that the two channels are "declared together or not at all". That was a declaration made before any
measurement of the envelope's effect on the level. The probe above is the measurement: the envelope is the stage that
moves the level, so a rule that forbids measuring the sum without it forbids the only comparison that can tell the two
mechanisms apart. The coupling is therefore relaxed to an option, the M20 arm stays measured exactly as it was (its
document, evidence and readings are unchanged), and the refusal that remains is the one with evidence behind it - the
single free channel beside the envelope is still a different declared arm and is still refused.

If `bare` beats the composition, the engine's `2` mode has been carrying a transform that costs it the fixed boundary,
and that is a structural improvement, not a threshold choice. If the composition beats `bare`, the crossing rule is doing
work the probe could not see (it saw the M20 arm's weights, not the arm trained with the rule in the loop) and the
composition stands.

## Realisation and Measured

**Realisation.** Three declared fields composed in one block: `occupancy_lattice_free 2` (the summed two-path read),
`occupancy_stack_monotone 1` (the M15 envelope) and `occupancy_crossing_threshold 1` (the M16 per-board rule). The M20
coupling that refused the sum without the envelope was relaxed to an option - the decomposition below is the measurement
that justified it - while every single-channel form stays its own arm. Four arms ran in one catalogue: `control` (the
published read), `envelope` (M20's arm, unchanged), `crossing` (the requested composition) and `bare` (the sum without the
envelope). Family `m21-level-`, instrument `m21-level-composition-r1`, curator `scripts/record_level_composition.py`,
launcher `runs/launch_level_composition.sh`.

`tests/test_level_composition.py` (22 tests) holds the mechanism: the order of the transforms (envelope then crossing, and
that the reverse order is a different read), one crossing level per board, shift-equivariance, that neither transform's
executable body names a label, step-zero identity for the sum alone, all four arms reaching the real parser while the
single free channel stays refused beside the envelope, and the record bindings.
`tests/test_declaration_propagation.py` is the standing instrument for the class that produced V-141, V-142 and V-145.

**Field-off identity.** Control reproduced AUC `0.8226433610683003` and plain F1 at `0.0` `0.381640` exactly. The four
arms' checkpoints are all distinct and each sweep carries its own study's checkpoint; the `envelope` arm reproduced the
M20 checkpoint bit-for-bit, which is this block's determinism check on the same code path.

| arm | lattice-free / envelope / crossing | AUC | plain F1 at `0.0` | best swept | out-of-fold | TP / FN | positive | strict exact | row error | occ. loss |
|---|---|---|---|---|---|---|---|---|---|---|
| control | 0 / 0 / 0 | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 0.400399 | 661 / 6 | 49.95% | 0 / 28 | +7.819 | 0.7684 |
| envelope (M20) | 2 / 1 / 0 | 0.898066 | **0.350636** | 0.600117 @ `3.30` | 0.459725 | 662 / 5 | 55.52% | 0 / 28 | +3.322 | 0.8314 |
| crossing (requested) | 2 / 1 / 1 | 0.823601 | **0.402187** | 0.447280 @ `0.90` | 0.447280 | 662 / 5 | 46.88% | 0 / 28 | +1.289 | 0.6330 |
| bare (no envelope) | 2 / 0 / 0 | 0.883359 | **0.365342** | 0.479007 @ `2.40` | 0.397204 | 662 / 5 | 52.80% | 0 / 28 | +7.128 | 0.9959 |

### Reading: `level-passes`, and decoupling `decoupling-fails`

The requested composition **clears the declared bar**: plain F1 at `0.0` **0.402187** against the
control's `0.381640` (`+0.020547`) with AUC **0.823601**
(`+0.000957`, inside the `0.02` ranking tolerance), TP/FN
`662/5`, positive share
`46.88%` against the control's `49.95%`,
swept 0.447280 @ `0.90`, out-of-fold
0.447280, row error +1.289 against the control's
`+7.819`, occupancy loss 0.6330 against
`0.7684`. Strict exact boards remain
0 of 28 (boundary-exact 0).

The sum without the envelope is **`decoupling-fails`**: primary `0.365342`
(`-0.016298`) despite the best ranking of the free-channel arms at AUC `0.883359`
(`+0.060716`) - so the envelope is not the cause of the boundary loss, it is what carries the
boundary, and the predeclaration's probe (scored on M20's weights) was wrong about that. The two mechanisms are only
separable when both are measured with their own rules in the loop, which is why the coupling had to go.

### What this is *not*

The composition is not the best primary on record. M16 already measured envelope + centring + own crossing at
**0.405439** with AUC `0.835241` and an out-of-fold
`0.423254`, so the free channel added to that composition *costs*
`-0.003252` primary and
`-0.011641` AUC, and the M20 envelope arm still holds the best ranking
(`0.898066`) and swept optimum (`0.600117`) in this block. The pass's gain is the
boundary: `+0.020547` in plain F1 at `0.0` over the control where the read alone gave back
`-0.031004`. Its row error was also lower (`+1.189` against this arm's `+1.289`), so on every recorded measure the
M16 composition is ahead of this one and the free channel is the difference.

### The stage decomposition: which stage moves the level

Every arm's read, rebuilt from its recorded parts and scored after each declared transform on its own fold and weights:

| arm | stage | AUC | plain F1 at `0.0` | positive share | swept optimum |
|---|---|---|---|---|---|
| control | `raw_sum` | 0.822643 | 0.381640 | +38.85% | 0.412034 @ `0.90` |
| envelope | `raw_sum` | 0.827538 | 0.429907 | +14.51% | 0.458138 @ `1.20` |
| envelope | `monotone` | 0.898066 | 0.350636 | +43.18% | 0.600117 @ `3.30` |
| crossing | `raw_sum` | 0.642201 | 0.255292 | +55.04% | 0.259947 @ `0.90` |
| crossing | `monotone` | 0.860251 | 0.214573 | +77.08% | 0.500413 @ `3.75` |
| crossing | `crossing` | 0.823601 | 0.402187 | +36.46% | 0.447280 @ `0.90` |
| bare | `raw_sum` | 0.883359 | 0.365342 | +41.07% | 0.479007 @ `2.40` |

**The stage that raises the positive share is `monotone` (`+0.220416` share
points) in the requested arm.** The crossing stage then moves the share
`-0.406250` and plain F1 at `0.0`
`+0.187614` relative to the monotone stage, subtracting a mean
level of `+2.7022` per board (range `+2.2813` .. `+3.1522` over `36`
boards). The envelope arm shows the same raising stage (`monotone`) and the published read has
no transform to raise it, so the movement belongs to the transform and not to the added channel.

Note the direction the probe could not see: trained *with* the rule in the loop, the read's raw sum starts far higher
(`77.08%` positive after the envelope, AUC `0.860251`) and the crossing rule pulls it all the way back to
`36.46%`.

**Corrected by M22's controlled ablation** (`docs/M22_STENCIL_READ.md`, same seed, same population, one block): applied
post-hoc to the `envelope` arm's own weights, the rule reaches F1@0 `0.341542` and AUC `0.868019` at `42.63%` positive,
against the same arm trained without it at `0.350636` / `0.898066` and the arm trained with it in the loop at `0.402187` /
`0.823601`. So "it re-decides the same place" was wrong in both directions: post-hoc application decides a *different and
worse* place on the primary (`-0.009094` against its own arm) while ranking *better* (`-0.030047` AUC for the arm it was
applied to, and `+0.044418` over the in-loop arm). In-loop training is what makes the rule work on the primary - it is
worth `+0.060645` F1@0 over post-hoc application, and it pays `0.074465` AUC against the arm it starts from to buy it.
The structural answer to "why does the level repair not survive composition" is therefore narrower than this document
first stated: the rule survives composition only when it is trained as part of the composition, and it buys that survival
with ranking.

### The channel split, per arm

The summed read against its own parts on each arm's own fold, the lattice part under that arm's own transforms:

| arm | summed AUC | lattice part | free channel | sum − lattice | verdict |
|---|---|---|---|---|---|
| envelope | 0.898066 | 0.896211 | 0.607085 | +0.001855 | one-channel-noise-dominates |
| crossing | 0.823601 | 0.829205 | 0.497469 | -0.005604 | one-channel-noise-dominates |
| bare | 0.883359 | 0.885828 | 0.720219 | -0.002469 | one-channel-noise-dominates |

The free channel is the limiter in every arm: alone it reads `0.497469` in the
requested composition and `0.607085` in the M20 arm, while the lattice part carries
the ranking. Adding it to the level composition costs primary and AUC (above), and the sum never beats its own lattice
part by the declared `0.005`.

### References, each measured in its own catalogue

| arm | AUC | plain F1 at `0.0` |
|---|---|---|
| *two path* | *0.898066* | *0.350636* |
| *envelope* | *0.845948* | *0.409161* |
| *crossing rule* | *0.835241* | *0.405439* |
| *span under level* | *0.789794* | *0.350613* |
| *combined read* | *0.888912* | *0.390805* |
| *centring* | *0.815395* | *0.374142* |
| *midpoint anchor* | *0.830500* | *0.310966* |

### Limits

Four arms, one catalogue, one seed, one launch, 28 non-empty boards over 12 images, synthetic label-integrity populations -
not photographed boards, not complete states. `bare` differs from `envelope` in one declared field *and* in what that
field does during its 64 training steps, so it is a composition effect, not an ablation of a fixed read. The strict exact
gate stays at 0 of 28 in every arm, so this pass improves the boundary and the level but does not produce an exact board.
, plain F1 at `0.0`, TP/FN, positive share, precision, recall, strict exact boards, boundary-exact boards,
row error, occupancy loss, swept optimum with its threshold, out-of-fold F1, and the occupancy loss paid. Plus: the
stage decomposition per arm, the crossing level per board for the arms that use it, the per-channel split for the three
arms that carry the free channel, and the contrasts against the reproduced control.

## Limits

Four arms, one catalogue, one seed, one launch, the same 28 non-empty boards over 12 images, synthetic label-integrity
populations - not photographed boards, not complete states. The `bare` arm differs from `envelope` in one declared field
*and* in what that field does during training, so it is a composition effect, not an ablation of a fixed read. The probe
table above is scored on the M20 arm's weights; the block re-measures it with the rule in the training loop.
