# Contributor handoff — 2026-10-30

## The occupancy deficit is two first-order factors, and the gate was unreachable one lever at a time (M27, 2026-11-14)

[M27_THREE_FACTOR.md](M27_THREE_FACTOR.md) reads two committed receipts together - the registration-attribution
receipt (model logits plus the model-cell -> target-cell mapping) and the pixel-ceiling receipt - and scores three
reads on the same `2,986` co-scoreable cells. No forward pass, no training, no weight, threshold, gate or label
change. Receipt `runs/m27-three-factor-r3-job.json` (passed, 3 work items, 0.39 s, 38,686,720 B of the declared
128 MiB); evidence `docs/evidence/m27-three-factor-r3/scope.json`.

| read | population | AUC | F1@0 | best swept F1 (threshold) |
|---|---|---|---|---|
| model, as-is | 5,600 deciding cells | 0.800604 | 0.379147 | 0.392496 (`0.60`) |
| model, as-is | 2,986 matched cells | 0.933138 | 0.439024 | 0.543974 (`2.10`) |
| model, oracle registration | 2,986 matched cells | 0.970233 | 0.574181 | **0.762590** (`2.25`) |
| pixel read-out, same cells | 2,986 matched cells | 0.999195 | 0.974763 | **0.987179** (`1.05`) |
| pixel read-out, full | 5,600 deciding cells | 0.998765 | 0.978006 | 0.985936 (`1.05`) |

Verdict `two-factor-first-order`: **registration `+0.218616`**, **representation `+0.224590`**, **boundary
`+0.188409`** (reported, inert - the level family stays closed). All four joins reproduce the published
`0.439024 / 0.933138`, `0.574181 / 0.970233`, `0.379147 / 0.800604` and `0.978006 / 0.998765` or the pass refuses to
write.

The first consequence is about the campaign, not the model: the standing gate is occupied-cell F1 `>= 0.70` at
`0.0` from `0.379147`, and the representation term alone is `0.224590`, so **no read-side lever could have cleared
that gate** even by perfectly closing its own term. That is the structural explanation for twenty-three arms that
all landed between `0.40` and `0.48` - each was moving one of three first-order terms while the gate charged the sum.
The second is that the pixel ceiling is no longer only a label result: on identical cells it is `0.224590` ahead
of the correctly registered learned read, so the missing information is per-cell raster content, not label policy.
The next lever is therefore a read-out that sees what the 30-feature colour read sees at the model's own lattice
(additive, zero-initialized, measurable at the standing 64-step protocol), with a registration-robust read as its
separately declared sibling - not a composition, and not another level, pair or threshold term.

Limits: the matched population is `53.3%` of the deciding cells and is biased away from the board edge the offset
spills over; the pixel read is arm-independent while the model side is the `4x`-budget arm; the pixel read is
granted the target lattice; one seed, synthetic, fit gate still `FALSE` at `0 / 28`. Three refused launches are
preserved (`runs/m27-three-factor-r1..r2-job.json`, V-149) and are not evidence.

## The symmetric pair margin works on the bare read, fails on the record-best read, and closes the axis (M26)

[M26_GAP_MARGIN.md](M26_GAP_MARGIN.md): `pair_gap_term` charges each cross-class pair by the closer of its two directional
violations, so the two modes cannot be collapsed into each other to lower it. Margin `1.0`, weight `.5`, both fixed before
the receipt; the curator and launch script are shared with M25 (`--family m26-gap`), receipt `runs/m26-curate-r1-job.json`.

| arm | AUC | F1@0 | exact | row error | stage share | within-board AUC | inversion rate |
|---|---|---|---|---|---|---|---|
| repro (published control) | 0.822643 | 0.381640 | 0 / 28 | +7.819 | 49.95% | 0.775698 | 0.224302 |
| gap (free + symmetric margin) | 0.866499 | 0.332749 | 0 / 28 | +9.040 | 59.36% | 0.797969 | 0.202031 |
| gap-stencil (free + envelope + stencil + symmetric margin) | 0.881989 | 0.403049 | 0 / 28 | +1.403 | 46.66% | 0.818553 | 0.190075 |

Reading: on the bare read the symmetric charge is the strongest single-arm gain since the stencil (**+0.043856** AUC over
control, inversion rate **-0.022271**, no collapse - 59.36% share). On the record-best read it is a regression
(**-0.035031** AUC, F1@0 `0.403049` against `0.480776`), and exactness does not move (inversion rate `0.190075` against
stencil's `0.189525`).

The axis is closed. The symmetric charge is orientation-blind by construction - an inverted-but-separated board is free -
so it cannot teach the ordering exactness needs; `tests/test_pair_gap.py` pins both consequences (the exact tie is a
symmetric saddle with zero slope; the two charges coincide on boards with no inversion). Further variants of this family
would be tuning a lever whose definition excludes the quantity. Record-best remains `stencil`
(`0.9170200020605955` / `0.480776`), and the residual stays where M24 put it: within-board discrimination.

## The pair-order lever fails, and its failure mode is mechanical (2026-11-13, M25)

[M25_PAIR_ORDER.md](M25_PAIR_ORDER.md) builds the lever M24 left standing: `pair_rank_term`, the complete cross-class
`mean(relu(1 - (occupied - empty)))` per board over known visible cells. Constants (`PAIR_RANK_MARGIN = 1.0`,
`PAIR_RANK_WEIGHT = .5`) were declared before any receipt; the lever adds no tensor, so parameter identity and construction
order are uninfluenced by it, and it is refused without the free channel (`occupancy_lattice_free 2`) and beside the five
boundary/margin terms and structural centring, which supervise a level the term is invariant to.

Measured on the unchanged protocol (same seed, 64 steps, 54 forwards, standing grid and `0.0` boundary, curator at
1900 MiB; `runs/m25-curate-r1-job.json`):

| arm | AUC | F1@0 | exact | row error | within-board AUC | inversion rate |
|---|---|---|---|---|---|---|
| repro (published control) | 0.822643 | 0.381640 | 0 / 28 | +7.819 | 0.775698 | 0.224302 |
| rank (free + pair) | 0.786891 | 0.212997 | 0 / 28 | +12.007 | 0.753922 | 0.246080 |
| rank-stencil (free + envelope + stencil + pair) | 0.918588 | 0.430619 | 0 / 28 | +1.443 | 0.828472 | 0.183009 |

Verdict `pair-order-fails`. The composition's AUC gain is inside the 0.02 tolerance and its inversion fall is under the
declared 0.01 bar, while F1@0 loses 0.050157 and row error triples-ish; the bare arm is worse than the control on both AUC
and inversions. The mechanism is right in principle and too weak in practice at this budget, and the reason is structural:
a shift-invariant one-sided hinge is paid most cheaply by lifting empties, which is exactly the precision collapse observed
(positive share 49.95% -> 42.91% on the composition, 99.93% on the bare arm). Record-best remains `stencil`.
Mechanical proofs live in `tests/test_pair_order.py` (20 tests): exact zero on a margin-separated board, exact charge on an
inverted pair, complete/permutation-free pairs, shift invariance, unknown-and-spawn blindness, non-zero gradient at the
zero read with one step moving it, no parameter and a bitwise-identical forward, and gradient reaching the real head.

## The exactness barrier is ordering, and no level or projection reaches it (2026-11-13, M24)

[M24_EXACTNESS_BARRIER.md](M24_EXACTNESS_BARRIER.md) decomposes why the record's best arms produce no strict exact board,
per board and per row, from the frozen sweeps (`docs/evidence/m24-exactness-r1/scope.json`). Six arms: `stencil`,
`centred`, `norm`, `norm-cross`, `envelope`, `control`.

- **No level works.** Searching every distinct level the read can be cut at - the complete set of cut points, not the
  81-point grid - gives `0 of 28` boards exact in every arm. The fewest wrong cells achievable at any level is `1` on the
  best board in the record and the median board needs about `16`.
- **No projection works.** The record's own consistency projections (`stack_contiguous`, then `otsu_board`) each leave
  `0 of 28` boards exact at their best level: a projection can move a cell across a level but cannot invent separation.
- **It is ordering.** Within-board AUC is `0.818993` for `stencil` and `0.753676`-`0.806938` for the rest, i.e. an
  inversion rate of `18.95%` and `21.2%`-`26.3%` of all occupied/empty pairs, against the `0.0` exactness requires. The
  median board carries `151.5` inverted pairs (`stencil`) up to `252.5` (`envelope`).
- **The direction explains every earlier result.** At `0.0` the read over-declares massively - `48.86` false positives
  against `0.82` false negatives on the best arm - so the boundary always looked like a level problem, and every level
  mechanism M14-M23 built moved it without ever touching the inversions underneath.

The instrument is controlled in `tests/test_exactness_barrier.py`: a separable synthetic board is reported exact, the same
board with one inverted pair is not exact at any level, a constant offset changes no verdict, and the search length equals
the number of distinct values plus one (so it is complete). The record claim is then held to those numbers.

Not built, and named as unmeasured: an explicit within-board pairwise ranking objective, which is the only lever the
decomposition leaves standing because it is the only one that acts on the quantity that is wrong.

## The scale-invariant contrast read: invariant by construction, still short of the bar (2026-11-13, M23)

[M23_CONTRAST_READ.md](M23_CONTRAST_READ.md) declared the mechanism and its two arms before they ran.
`--occupancy-free-stencil 3` reads the five taps on a field where each coarse box is divided by the spread of its own
5x5 neighbourhood (conditioned by the map's global mean first; no additive epsilon, which would break exact rescaling
invariance - measured `1.3e-2` drift with a floor against `1.2e-6` without; a `+/-20` clamp that is inactive at a
measured maximum of `2.403`). Proved mechanically: zero read at step zero, leaves the zero fixed point inside the real
training path, `abs(mean)/mean(abs(read)) = 0.0` over the visible rows, `<= 3.9e-6` drift under board-wide offsets and
`<= 1.2e-6` under rescaling of coarse values, `>= 1e-2` under a genuine local change, exactly `0.0` on a uniform board,
and the same invariance through the head.

Ten arms, same catalogue and protocol as M22, control `0.8226433610683003` / `0.381640` exactly, M20/M21 checkpoints
reproduced. `norm` `0.856281` AUC / `0.414810` F1@0 (`+0.033170`, `0 of 28` exact) is `stencil-moves-without-clearing`;
`norm-cross` `0.835742` / `0.377106` is `stencil-fails`. The primary is identical to the M22 de-meaned arm and the AUC
`0.003542` lower, so the level the M22 stencil produced was not reclaimable capacity - it was the gain. What the contrast
read does buy: the first `complementary` channel verdict in the record (`+0.027048` AUC over its own lattice part), a
`local_only` that beats the pooled channel (`0.742764` vs `0.654209`), and a local read the crossing rule survives
(`0.835742` / `0.377106` against `0.782265` / `0.311695` for the M22 stencil and `0.736989` / `0.261500` for the
de-meaned one).

Still open: no strict exact board in any arm of either catalogue. The named next question is no longer "what local
detail" - three formulations agree the local detail is worth `~0.03` primary - but where the remaining `0.05` could come
from, since the best primary in the record (`0.480776`) is bought with a board-wide level that no composition keeps.

## The free channel's local stencil: best read, no exact board, and the gain is a level (2026-11-13, M22)

[M22_STENCIL_READ.md](M22_STENCIL_READ.md) declared eight arms before any receipt and a controlled ablation: the
published read, the M20 two-path read, the M21 composition, the sum without the envelope, and the same two compositions
with a coarse local stencil on the fixed lattice (plain, and de-meaned over the visible board). Control reproduced
`0.8226433610683003` / `0.381640` exactly; `envelope` reproduced the M20 checkpoint bit-for-bit and `crossing` the M21
one.

| arm | AUC | F1@0 | share at 0.0 | exact | row error | reading |
|---|---|---|---|---|---|---|
| control | 0.822643 | 0.381640 | 49.95% | 0/28 | +7.819 | — |
| envelope | 0.898066 | 0.350636 | 55.52% | 0/28 | +3.322 | level-fails |
| crossing | 0.823601 | 0.402187 | 46.88% | 0/28 | +1.289 | level-passes |
| bare | 0.883359 | 0.365342 | 52.80% | 0/28 | +7.128 | level-fails |
| stencil | 0.917020 | 0.480776 | 35.93% | 0/28 | +0.404 | stencil-moves-without-clearing |
| stencil-crossing | 0.782265 | 0.311695 | 61.54% | 0/28 | +3.855 | stencil-fails |
| centred | 0.859823 | 0.414810 | 45.00% | 0/28 | +1.007 | stencil-moves-without-clearing |
| centred-crossing | 0.736989 | 0.261500 | 66.89% | 0/28 | +3.278 | stencil-fails |

The verdict is plain: `+0.099136` primary clears the `+0.05` floor, no strict exact board means the bar is **not met**.
Where it lives: the `monotone` stage (share `+0.178888`), and the channel split shows the local read is weaker than the
pooled one (`0.634832` vs `0.781555`), lowers the free channel, and carries `82%` of its magnitude as a board-wide level.
The de-meaned re-measurement is the controlled test - `+0.033170` primary, level ratio `1e-06` - so the appearance of a
large gain was the level. Composed with the crossing rule the stencil loses against the control in both formulations.

Ablation, same seed, same population: `crossing` `0.402187` / `0.823601` against the rule applied post-hoc to
`envelope`'s own weights `0.341542` / `0.868019`. In-loop minus post-hoc `+0.060645` F1@0, `-0.044418` AUC. The rule's
operating-point benefit needs in-loop training; its ranking benefit does not. M21's wording is corrected in place.

Also closed: V-147 (a declared lever missing from the hand-written recovery contract; the contract is now completed from
the config and the standing instrument covers it). Next lever named: the stencil's detail is compensating a level, so the
honest next question is whether *normalised* local contrast - a stencil that reads the box against its neighbourhood
rather than raw feature magnitude - adds ranking where this one added level.

## The level repair composed and decomposed: `level-passes`, decoupling `decoupling-fails` (2026-11-13, M21)

[M21_LEVEL_COMPOSITION.md](M21_LEVEL_COMPOSITION.md) declared four arms and each one's bar before any receipt. Control
reproduced `0.8226433610683003` / `0.381640` exactly; the `envelope` arm reproduced the M20 checkpoint bit-for-bit; all
four checkpoints are distinct.

| arm | AUC | plain F1 at `0.0` | positive | row error | out-of-fold |
|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 49.95% | +7.819 | 0.400399 |
| envelope (M20) | 0.898066 | 0.350636 | 55.52% | +3.322 | 0.459725 |
| crossing (requested) | 0.823601 | **0.402187** | 46.88% | +1.289 | 0.447280 |
| bare (no envelope) | 0.883359 | 0.365342 | 52.80% | +7.128 | 0.397204 |

The composition clears the bar: `+0.020547` primary over the control with AUC
`+0.000957`, positive share `46.88%` (the control's
`49.95%`) and the second-lowest row error on record (`+1.289`, against the
control's `+7.819`; M16's envelope + centring + crossing arm holds the lowest at `+1.189`).
Strict exact stays 0 of 28. The sum alone (`bare`) fails the primary (`-0.016298`) with a better
ranking (`+0.060716`), so the envelope carries the boundary and the M20 coupling had hidden it.

The stage that raises the positive share in the requested arm is `monotone` (`+0.220416` share points); the crossing stage moves the share by `-0.406250` against the monotone stage and its F1 at `0.0` by `+0.187614`. The envelope arm reproduces the same raising stage (`monotone`), and the published read carries no transform, so it has no stage to raise it; the level movement therefore belongs to the transform rather than to the added channel.

Two things a future arm must not redo: the crossing rule has to be trained in the loop (applied afterwards to a read
trained without it, it moves the share `-0.406` but leaves F1 at `0.0` worse than the M20 arm), and M16's envelope +
centring + crossing arm still holds the best primary (`0.405439`).

**Next candidate:** the coarse local stencil for the free channel (it reads
`0.497469` alone here), then re-test the composition with that channel.

## The two-path read: the best ranking and placement measured, failing only at the boundary (2026-11-13, M20)

[M20_TWO_PATH_READ.md](M20_TWO_PATH_READ.md) declared the sum of the two reads that each held a best result: the
lattice-sampled envelope (best ranking `0.888912`, best placement `1.189` rows) and the lattice-free channel (the only
2-wrong-cell board - a claim V-143 retracts). `--occupancy-lattice-free 2` adds them. The sum is ungated and its head
zeroes only its output projection: a zero scale and a fully zeroed two-layer head are both training fixed points, so the
first two attempts measured a channel that never moved (V-143, archived void).

| arm | AUC | plain F1 at `0.0` | best swept | out-of-fold | TP/FN | positive | strict exact | boundary | row error | occ. loss |
|---|---|---|---|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 | 0.400399 | 661/6 | 49.95% | 0/28 | 0/28 | +7.819 | 0.7684 |
| two-path read | 0.898066 | **0.350636** | 0.600117 | 0.459725 | 662/5 | 55.52% | 0/28 | 0/28 | +3.322 | 0.8314 |

Reading **`two-path-fails`**, secondary `ranking-preserved`, delta `-0.031004`;
gate `0 of 28` under both strict definitions. But the same fold gives the programme's
best ranking (`+0.075423` AUC over the control, above the M15 combined `0.888912`), best swept placement
(`+0.188083`, out-of-fold `+0.059326` above the control) and less than
half the control's row error (`-4.497`). The failure is the read's *level*: positive share
`55.52%` against `49.95%`, best threshold
`3.30` against `0.90`.

Channel split on the arm's own fold: sum `0.898066`, lattice part alone `0.896211`
(`+0.001855`, under the `0.005` bar), free channel alone
`0.607085` -> **`one-channel-noise-dominates`**. So the declared mechanism is falsified while the useful finding is
another one: the lattice channel inside this arm's weights reaches `0.896211`, against
`0.845948` in the envelope arm - the added channel acted on training, not on cell evidence.

**Next candidate:** `--occupancy-lattice-free 2 --occupancy-stack-monotone 1 --occupancy-crossing-threshold 1` - the
across-the-board measured level repair (M19: `80.55%` -> `52.16%` positive, `level_repaired = true`) applied to the read
that now ranks best. Bar unchanged: `+0.05` over the reproduced control and at least one strict exact board.

## The span under the level transforms: the level was repaired and the signal was not there (2026-11-12, M19)

[M19_SPAN_LEVEL.md](M19_SPAN_LEVEL.md) declared the composition the M18 record named: the row-span tap with M14 centring
and M16's crossing rule, admitted together and only together, measured against the reproduced control in one catalogue.

| arm | AUC | plain F1 at `0.0` | best swept | out-of-fold | TP/FN | positive at `0.0` | strict exact | boundary | row error | occ. loss |
|---|---|---|---|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 | 0.400399 | 661/6 | 49.95% | 0/28 | 0/28 | +7.819 | 0.7684 |
| span + centring + rule | 0.789794 | **0.350613** | 0.364139 | 0.328718 | 629/38 | 52.16% | 0/28 | 0/28 | **+6.993** | 0.6925 |

Reading **`span-level-fails`**, `level_repaired = true`. The transforms did exactly what they were declared to do - the
positive share fell from `80.55%` to `52.16%`, the row error from `9.409` to `6.993` (better than the control's `7.819`) -
and the primary still fell by `0.031027`. **This reverses the M18 reading**: under a repaired level the span's swept
optimum (`0.364139`) and out-of-fold F1 (`0.328718`) are both *below* the control's, so M18's apparent separation gain was a
property of its displaced level. The row-neighbour tap carries no recoverable per-cell information on this population and
that line is closed by measurement. The M18 document carries the correction.

**Where the programme stands.** Nineteen read-side arms, strict gate still `0 of 28` under both definitions. The best
ranking (`0.888912`) and the best placement (`1.189` rows) both belong to the M15 combined read (monotone envelope +
centring); the only board ever within 2 cells came from the lattice-free read (`0.840147`), which does not consume the
predicted lattice. Those two bests are disjoint and no arm has combined them.

**Read or population?** Not population: 28 boards would show two or three exact boards at a tenth-per-board exactness and
nineteen arms show none, so per-cell read precision binds first. Judge future arms on exact-board counts and near misses,
not on clearing 28 in one block.

**Named next candidate.** A **two-path read**: sum the lattice-free channel's evidence (the only 2-cell board) with the
lattice-sampled envelope's (best ranking and placement) in the read-out, so the read gets placement-independent evidence
and the best-separating evidence at once. Structural, label-free, and never measured.

## The row-span tap: separation without a level (2026-11-12, M18)

> Superseded in part by M19: the span's swept and out-of-fold gains did not survive a repaired level, so "separation
> without a level" is not what M18 measured. The numbers below remain accurate.

[M18_TAP_SPAN.md](M18_TAP_SPAN.md) was predeclared with measured justification: over the 84 retained sweeps the wrong
cells are 2.2x denser in the four rows at the top of each stack (0.419 / 0.407 / 0.428 at distances 0, 1, 2-3 against
0.188 far below), and every cell logit was a function of one interior sample. `--occupancy-tap-span 1` gives the detail
tap three row-aligned samples (the cell, one row above, one row below in cell units) through one zero-initialized
projection. Family `m18-span-`, control reproduced `0.8226433610683003` / `0.381640` exactly. One whole-block rerun was
needed (V-141), so both arms are one catalogue and every stage is `passed 0`.

| arm | AUC | plain F1 at `0.0` | best swept | out-of-fold | TP/FN | positive at `0.0` | strict exact | boundary-exact | row error | occ. loss |
|---|---|---|---|---|---|---|---|---|---|---|
| control (point tap) | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 0.400399 | 661/6 | 49.95% | 0/28 | 0/28 | +7.819 | 0.7684 |
| row-span tap | 0.795536 | **0.244496** | **0.443255** @ `2.55` | **0.429607** | 633/34 | 80.55% | 0/28 | 0/28 | +9.409 | 0.9289 |

Reading **`span-fails`**, secondary `ranking-degraded`. **Both declared falsifications fired**: the primary is `-0.137144`
and the gate stays `0 of 28` under both strict definitions, so nothing moved and there is no V-139-class artefact to audit.

**The failure is a level, not an absence of signal.** The arm's best separation and its out-of-fold F1 both beat the point
tap's (`+0.031221` and `+0.029208`), while it decides `80.55%` of cells positive at `0.0` and places stacks `1.59` rows
too high. Three samples spanning a whole cell give the projection two neighbours nearly anti-correlated with the cell, and
the arm spends them pushing the level rather than separating rows. No span width, sample count or projection size is
offered - each would be a change to the statistic being read.

**Read or population?** Settled by the block: 28 boards would show two or three exact boards at a tenth-per-board
exactness and nineteen arms show none, so per-cell read precision is the binding constraint; the population becomes the
constraint only after a read produces exact boards at all. **Named next candidate**: the row-span tap *composed* with the
already-validated structural level transforms (M14 centring, M16's crossing rule), as its own declared arm with its own
falsification - the span supplies separation, the level transforms supply where zero sits.

## The joint skyline is falsified; the gate is a per-cell read limit (2026-11-12, M17)

[M17_SKYLINE_RECOVERY.md](M17_SKYLINE_RECOVERY.md) was predeclared with its falsification before any receipt:
`--occupancy-skyline-recovery 1` recovers the board's whole skyline with a dynamic programme over per-column heights
(emission = signed mass on each side of the split, penalty `1.0` logit, deterministic, label-free by signature and by
test), applied last after the envelope, centring and the crossing rule. Family `m17-skyline-`, catalogue `5c02208e`, all
receipts `passed 0`.

| arm | AUC | plain F1 at `0.0` | TP/FN | positive at `0.0` | strict exact | boundary-exact | row error | occupancy loss |
|---|---|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 661/6 | 49.95% | 0/28 | 0/28 | +7.819 | 0.7684 |
| envelope + centring + rule + skyline | 0.620664 | **0.308333** | 592/75 | 56.66% | **0/28** | **0/28** | +3.993 | 0.5988 |

Reading **`skyline-fails`**, secondary `ranking-degraded` (AUC `-0.201979`). **Both declared falsifications fired**: the
recovery does not beat the per-column rule (`0.308333` vs `0.405439`) and the gate stays `0 of 28` under both strict
definitions - nothing moved, so there is no artefact of the V-139 kind to audit.

**Why it failed, as mechanism.** The emission is a *volume* score, so it prefers heights that accumulate positive read
rather than the level at which the column steps: `56.66%` of cells positive against control's `49.95%`, recall `0.8876`
against `0.9910`, and a row error *worse* than the read it was applied to (`3.993` against `1.189`). Putting every cell
on its side of the recovered height then destroyed the cross-board ordering, hence AUC `0.62`. A softer penalty or a
different emission is a variant of this arm, not a new one; none was run, because that is the per-arm tuning the
declaration forbids.

**What seventeen arms now say about the gate.** Decision-side arms (`M16` rule, `M17` skyline) cannot get below the
read's own row error and the skyline prior hurts; read-side arms hold every best placement (`M15` envelope, `1.189` rows)
and ranking (`M15` combined, `0.888912`); the closest board ever produced is **2 wrong cells**, from a cell-read lever
(`m3-occlatfree-lever`, AUC `0.840147`). A 200-cell board at AUC `0.89` implies ten to twenty expected cell errors and the
gate asks for zero on all 28, so the remaining distance is a **per-cell read limit**, not a decision or structure limit.
That is the structural diagnosis to carry forward: work on the read's per-cell precision (what the tap samples, at what
resolution, with what registration), not on more decision layers.

## The fit gate was mis-measured: `1 of 28` is `0 of 28` (2026-11-12, V-139)

Before building anything for the gate, the metric was audited end to end, and the audit found the defect.

**The defect.** Every curator's `exact_boards_at_zero` compared *column boundaries* with `if got is not None and want is
not None`, dropping any column whose decision or label was empty. A board that filled genuinely empty columns had those
columns removed from the comparison and could be scored exact while its decided state was wrong. The sweep's own
cell-exact flag (`_counts(...)[4]`, `right`) never had this hole; only the curators did.

**The corrected history.** Recomputed from the retained per-case logits of every arm sweep in `runs/` (79 arms; no block
run, no receipt rewritten):

| definition | arms at 1/28 | arms at 2/28 | arms at 0/28 |
|---|---|---|---|
| strict cell-exact (sweep's own flag) | 0 | 0 | **79** |
| strict boundary-exact (empty column compared as a value) | 0 | 0 | **79** |
| the previous lenient rule | 26 | 5 | 48 |

So the fit gate has been **0 of 28 for every arm in the program's history**. The `1 of 28` credited to M12's dense hinge
and M13's midpoint anchor - and to 24 further M3/M4/M5/M6/M7/M8 levers - is withdrawn here; those scopes are immutable
evidence and are not rewritten.

**How close anything got.** The best board any arm ever produced has **2 wrong cells** (`m3-occlatfree-lever`, AUC
`0.840147`), with three of its 28 boards within ten wrong cells. No structural arm is below eight wrong cells
(`M15` envelope `1.19` rows of row error, `M16` ruled arm `1.19`). The gate requires 0 on all 28, so it is not one board
from moving; it is four to twenty cells per board away on the best arms.

**What was changed.** `exact_boards_at_zero` in all six curators now requires every visible cell to agree *and* compares
every column with an empty column as a value, returning `boundary_exact` beside `exact`. `tests/test_exact_state_metric.py`
pins the behaviour on synthetic boards (perfect board, one wrong cell, the V-139 stray cell in a truly empty column, a
correctly empty column, agreement with the sweep's flag, refusal when no board is present) and walks every retained
receipt asserting `0 of 28` under both strict definitions.

**What was deliberately not built.** The sequence-level joint recovery was conditional on the metric being sound. That
premise did not hold, and its target is now known to be an unapproached `0 of 28`. It remains the right next structural
candidate if anyone wants the gate anyway - a board-level decision over the monotone skyline is available without labels -
but it must be declared and measured against the corrected metric, not against a phantom.

## The label-free decision rule works and is not enough (2026-11-12)

[M16_CROSSING_THRESHOLD.md](M16_CROSSING_THRESHOLD.md) was predeclared before any receipt, including its falsification.
`--occupancy-crossing-threshold 1` subtracts each board's own exact Otsu level (over every gap between consecutive sorted
visible logits) inside the head, after the envelope and after centring, so downstream `0.0` is that board's crossing. No
label, prevalence, threshold grid or parameter; family `m16-crossing-`, catalogue `ad204845`, all receipts `passed 0`.

| arm | AUC | plain F1 at `0.0` | best swept F1 | positive at `0.0` | exact at `0.0` | out-of-fold | row error | occupancy loss |
|---|---|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 49.95% | 0/28 | 0.400399 | +7.819 | 0.7684 |
| envelope + centring + own crossing | 0.835241 | **0.405439** | 0.428088 @ `1.35` | 45.87% | 0/28 | 0.423254 | +1.189 | 0.6550 |

Reading `crossing-moves-without-clearing` (`+0.023799` vs control, `+0.014634` **over the same read's fixed-`0.0`
ceiling**), secondary `ranking-preserved` (`+0.012598`), verdict `not-cleared`, fit gate FALSE (0/28). Aliasing: the arms
differ in exactly the three declared fields, `bf1ca813…` vs `7cd49078…`, and M15's combined checkpoint is excluded by
name. Leakage guard green: the rule's output is bit-identical under five different label arrays and its body contains no
label reference.

**Result.** The decision-rule hypothesis is **not falsified and not sufficient**. Breaking the ceiling by `+0.0146` proves
the fixed `0.0` boundary was costing F1; landing `0.0227` below the read's own swept optimum, and raising false negatives
from `4` to `11`, proves a per-board rule that cannot see prevalence cannot recover the population optimum. The bar
(`0.431640`) is `0.0262` away and the gap is a recall/precision trade, not a threshold problem.

**Next, named by the evidence.** The fit gate: `0` of `28` exact boards across sixteen arms, at a row error of `1.19`
rows. Everything else is at or near its best; exact-state is the quantity no arm has moved, and the next arm should
attack it directly rather than another decision level. Any further threshold search is tuning and must not be presented
as a structural arm.

## Structure plus a centred level: best ranking, still not `0.0` (2026-11-12)

The follow-on declared in [M15_STACK_CONTIGUITY.md](M15_STACK_CONTIGUITY.md) before its own receipt ran as
`m15-stack-combined-r1` (`--occupancy-stack-monotone 1 --occupancy-centre 1`, envelope first then centring), curated
against the M15 block's own control in the same catalogue `d9d867a6`. All receipts `passed 0`.

| arm | AUC | plain F1 at `0.0` | best swept F1 | positive at `0.0` | exact at `0.0` | out-of-fold | row error | occupancy loss |
|---|---|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 49.95% | 0/28 | 0.400399 | +7.819 | 0.7684 |
| envelope only | 0.845948 | **0.409161** | 0.442636 @ `2.85` | 45.79% | 0/28 | 0.409639 | +1.255 | 0.8494 |
| envelope + centring | 0.888912 | **0.390805** | **0.502686** @ `1.20` | 48.68% | 0/28 | 0.392122 | +1.846 | 0.5987 |

Reading `stack-moves-without-clearing` (`+0.009165`), secondary `ranking-preserved` (`+0.066269`), verdict
`not-cleared`, fit gate FALSE (0/28). Aliasing guard: the arms differ in exactly `occupancy_centre` and
`occupancy_stack_monotone`, `ba369e92…` vs `bf1ca813…`, and the envelope arm's checkpoint is excluded by name.

**Where the program stands after fifteen arms.** Five supervised terms, a level statistic, a monotone structural
envelope and now the envelope plus centring have each been measured: none clears the standing `+0.05` bar at a fixed
`0.0`. The same read, however, reaches `0.502686` at its own optimum with the best AUC (`0.888912`) and the fewest missed
occupied cells (`4` false negatives) yet measured. The blocker is therefore the fixed-`0.0` decision rule, not the read's
quality.

**Next, declared by the numbers rather than by preference.** A label-free decision rule that follows the crossing the
monotone structure already exposes - a column's decided height under the envelope - so the operating point is the
structure's own crossing rather than `0.0` by fiat. It must be predeclared with its constants (there are none to tune),
its readings against the standing bars, and its failure mode (the crossing is a *per-column* object, so a board-level
scorer may still be wrong where columns disagree with the labels), and compared against the same M15 control in one
catalogue. That is the only direction the measurements now support; any further threshold or level search is tuning and
must not be presented as a structural arm.

## Structure places the crossing; only its level is still free (2026-11-12)

[M15_STACK_CONTIGUITY.md](M15_STACK_CONTIGUITY.md) was predeclared before any receipt: `--occupancy-stack-monotone 1`
replaces each column of the occupancy logits with its monotone envelope (suffix maximum in bottom-first order) inside the
head, so a column crosses `0.0` at most once and that crossing is its height. No label, threshold, fitted parameter or
prevalence; spawn rows pass through untouched; `0` is the published function; family `m15-stack-`, catalogue `d9d867a6`.

| arm | AUC | plain F1 at `0.0` | best swept F1 | positive at `0.0` | exact at `0.0` | out-of-fold | row error | occupancy loss |
|---|---|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 49.95% | 0/28 | 0.400399 | +7.819 | 0.7684 |
| monotone envelope | 0.845948 | **0.409161** | 0.442636 @ `2.85` | 45.79% | 0/28 | 0.409639 | **+1.255** | 0.8494 |
| *M14 centred (reference)* | *0.815395* | *0.374142* | *0.393230* @ *`1.95`* | *50.52%* | *0/28* | *0.378651* | *+8.007* | *0.6166* |
| *M13 midpoint (reference)* | *0.830500* | *0.310966* | *0.401408* @ *`3.00`* | *64.5%* | *1/28* | *0.348472* | *+9.779* | — |
| *M12 dense hinge (reference)* | *0.862838* | *0.354770* | *0.465582* @ *`3.15`* | *54.7%* | *1/28* | *0.465582* | *+8.544* | — |
| *M11 quantile (reference)* | *0.815340* | *0.264308* | *0.404026* | *78.0%* | — | *0.379323* | *+10.020* | — |
| *M10 extremal (reference)* | *0.885462* | *0.310966* | *0.540811* | *64.5%* | — | *0.540811* | *+9.779* | — |
| *M9 hinge (reference)* | *0.864667* | *0.251701* | *0.474210* | — | — | *0.429457* | *+10.691* | — |

Reading **`stack-moves-without-clearing`** (primary `+0.027521`), secondary `ranking-preserved` (AUC `+0.023305`),
verdict `not-cleared`, fit gate FALSE (0 of 28). Control reproduced `0.8226433610683003` / `0.381640` exactly; aliasing
guard passed (configs differ in exactly `occupancy_stack_monotone`, `5976d8fa…` vs `bf1ca813…`, M14's checkpoint
excluded by name).

**The result.** This is the first arm to move the primary at all, and the first whose swept maximum beats the standing bar
(`0.442636` vs `0.431640`). Structural placement is real: row error fell from `7.819` rows to `1.255` with identical
recall (`661` TP / `6` FN) and `233` fewer false positives at `0.0`, and AUC improved, which a monotone projection can
only do by suppressing isolated high reads that were never stack.

**What it did not do, and why that is the next question.** The swept optimum moved to `2.85` rather than to `0.0`, so the
envelope places the crossing's shape but not its level - exactly what its equivariance requires, since
`envelope(L + c) = envelope(L) + c`. The level of a monotone read is the single remaining degree of freedom.

**Declared follow-on (predeclared in the M15 document before its own receipt):** `--occupancy-stack-monotone 1
--occupancy-centre 1` - envelope first, centring after it, so the crossing is placed by structure and its level is the
board's mean instead of a free parameter. Declared readings `stack-clears-the-crossing` / `stack-moves-without-clearing`;
declared failure mode is that the mean is not the operating point on a `12%`-occupied board (M14 measured that as
`+1.95`), so either direction is possible and both are reported.

## The level is now pinned, and the pin is at the wrong place (2026-11-12)

[M14_READ_CENTRING.md](M14_READ_CENTRING.md) was predeclared before any receipt: `--occupancy-centre 1` subtracts each
board's own mean over its visible window from that board's occupancy logits inside the head, before the objective or any
scorer sees them. No added term, no weight, no margin, no parameter; the spread is untouched, so alignment invariance is
exact. Family `m14-centre-`, new catalogue `c9544334`.

| arm | AUC | plain F1 at `0.0` | best swept F1 | positive at `0.0` | exact at `0.0` | out-of-fold | row error | occupancy loss |
|---|---|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 49.95% | 0/28 | 0.400399 | +7.819 | 0.7684 |
| centred in the head | 0.815395 | **0.374142** | 0.393230 @ `1.95` | 50.52% | 0/28 | 0.378651 | +8.007 | 0.6166 |
| *M13 midpoint (reference)* | *0.830500* | *0.310966* | *0.401408* @ *`3.00`* | *64.5%* | *1/28* | *0.348472* | *+9.779* | — |
| *M12 dense hinge (reference)* | *0.862838* | *0.354770* | *0.465582* @ *`3.15`* | *54.7%* | *1/28* | *0.465582* | *+8.544* | — |
| *M11 quantile (reference)* | *0.815340* | *0.264308* | *0.404026* | *78.0%* | — | *0.379323* | *+10.020* | — |
| *M10 extremal (reference)* | *0.885462* | *0.310966* | *0.540811* | *64.5%* | — | *0.540811* | *+9.779* | — |
| *M9 hinge (reference)* | *0.864667* | *0.251701* | *0.474210* | — | — | *0.429457* | *+10.691* | — |

Control reproduced `0.8226433610683003` and `0.381640` exactly. Reading **`centre-fails`** (primary `-0.007498`),
secondary `ranking-preserved` (`-0.007248`), verdict `not-cleared`, fit gate **FALSE** (0 of 28), overshoot flag false.
The block proves its own identity: configs differ in exactly `occupancy_centre`, each sweep carries its own checkpoint,
the two weights differ, and M13's checkpoint is excluded by name.

**What the pass established.** The transform does exactly what it claims: the level drift collapsed from `+14.5` points
to `+0.57` (49.95% -> 50.52% positive at `0.0`), the read fits its labels better (occupancy loss `-0.152`), and the
primary delta `-0.0075` is seven times smaller than the next-best arm. But the pinned level is the board's *mean*, and on
a `12%`-occupied population the swept optimum sits `1.95` logits above `0.0` - the pin is in the wrong place, so `0.0`
still cuts through the middle of the population. The declaration named this mechanism and expected it in the other
direction (an overshoot below zero); the observed failure is a pin at the mean, and the record says which was observed.

**What it rules out.** Adding supervision cannot place the boundary (M9-M13) and removing the level cannot either (M14):
the only label-free level the read owns is its own mean, and the mean is not the operating point. What is left is where
the read's *structure* puts the crossing.

**Next arm, declared.** *Per-column stack-contiguity*: `--occupancy-stack-monotone 1` parameterises the occupancy read
as a per-column cumulative maximum from the floor, so each column crosses `0.0` at most once and that crossing is the
column's height - no prevalence, no threshold, no margin, no weight, no parameter. Declared failure mode: holes and
overhangs cannot be expressed by a monotone column, so it may trade per-cell accuracy for placement; standing bars
`+0.05` primary and `-0.02` AUC, restated in its own document before its own receipt.

Defects: V-135 (the initial-weights config comparison rejected a source checkpoint that predates a default-0 structural
field; fixed through the declared `allow_absent` mechanism) and V-136 (a path constant used as a dict in the curator,
caught by its own failing receipt and repaired before any record existed).

## Supervising the location fails; the shift itself has to go (2026-11-12)

[M13_MIDPOINT_ANCHOR.md](M13_MIDPOINT_ANCHOR.md) was predeclared before any receipt: `--occupancy-midpoint 1`, i.e. per
qualifying board `|0.5*mean_occupied + 0.5*mean_empty - 0.0|` over known visible cells, pooled at `ANCHOR_WEIGHT = 0.5`,
no margin, equal class weight. Family `m13-midpoint-`, new catalogue `19077be4`.

| arm | AUC | plain F1 at `0.0` | best swept F1 | positive at `0.0` | exact at `0.0` | out-of-fold | row error |
|---|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 49.9% | 0/28 | 0.400399 | +7.819 |
| midpoint anchor | 0.830500 | **0.310966** | 0.401408 @ `3.00` | 64.5% | 1/28 | 0.348472 | +9.779 |
| *M12 dense hinge (reference)* | *0.862838* | *0.354770* | *0.465582* | *54.7%* | *1/28* | *0.465582* | *+8.544* |
| *M10 extremal anchor (reference)* | *0.885462* | *0.310966* | *0.540811* | *64.5%* | — | *0.540811* | *+9.779* |
| *M11 quantile anchor (reference)* | *0.815340* | *0.264308* | *0.404026* | *78.0%* | — | *0.379323* | *+10.020* |
| *M9 hinge (reference)* | *0.864667* | *0.251701* | *0.474210* | — | — | *0.429457* | *+10.691* |

Control reproduced `0.8226433610683003` and `0.381640` exactly. Reading `midpoint-fails` (primary `-0.070674`),
secondary `ranking-preserved` (`+0.007857`, inside tolerance, so the declared collapse override did not fire), verdict
`not-cleared`, fit gate **FALSE** (1 of 28). Paid charge mean `0.923`, at `0.0` on only 16 of 64 steps.

**Note for the next reader:** this arm's `0.0` counts (`665/2945/2`), F1 (`0.310966`) and row diagnostic (`9.779`) are
identical to M10's extremal anchor. That is a coincidence of where the two reads sit, not a duplicated measurement -
different catalogue, different checkpoint, different AUC (`0.830500` against `0.885462`), and the curator refuses a
sweep whose weights differ from its own run's checkpoint.

**What the pass established.** A per-board *location* is no easier to buy than a per-board *shape*: the midpoint can be
driven to zero from either side, so the likelihood's own pull (which raises the occupied class) wins the trade and the
equilibrium still sits above the boundary. The positive share rose another `14.5` points to `64.5%`, the swept optimum
stayed at `3.00` and the primary fell `0.071` - the fifth negation out of five supervision arms. The two arms that came
closest (M12: best swept `+0.054`, out-of-fold `+0.065`) did so on shape, not placement. What no arm did is remove the
shift.

**Next arm, declared.** *Structural centring of the read*: `--occupancy-centre 1` subtracts each board's own mean over
its visible cell window from the occupancy logits inside the head, before the objective or any scorer sees them - no
labels, no thresholds, no parameters, no prevalence. It differs from M8's `normalise` by removing only the additive
mode, so it cannot be paid off by a level shift because the level is gone. Its failure mode is declared with it: the
mean of all cells is not the F1-optimal level on a `12%`-occupied population, so it can overshoot below zero, and if the
swept optimum moves below zero the honest reading is `centre-fails`. Standing bars `+0.05` primary and `-0.02` AUC, to
be restated in its own document before its own receipt.

Defect: none new in the engine or the instrument this pass; the two failures in this pass's test file were expectation
errors caught before launch and are covered by the standing independent-reference instrument added with V-133.

## Charging every cell fixes the shape and leaves only the placement (2026-11-12)

[M12_DENSE_HINGE.md](M12_DENSE_HINGE.md) was predeclared before any receipt: `--occupancy-dense-hinge 1`, i.e. per
qualifying board `0.5 mean_occ relu(margin - logit) + 0.5 mean_empty relu(margin + logit)` over known visible cells, at
`ANCHOR_MARGIN = 1.0`, `ANCHOR_WEIGHT = 0.5`, balance `.5/.5` (equal class weight, never a count). Family
`m12-dense-hinge-`, new catalogue `24c76620`.

| arm | AUC | plain F1 at `0.0` | best swept F1 | positive at `0.0` | exact at `0.0` | out-of-fold | row error |
|---|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 49.9% | 0/28 | 0.400399 | +7.819 |
| dense class-balanced hinge | 0.862838 | **0.354770** | **0.465582** @ `3.15` | 54.7% | **1/28** | **0.465582** | **+8.544** |
| *M10 extremal anchor (reference)* | *0.885462* | *0.310966* | *0.540811* | *64.5%* | — | *0.540811* | *+9.779* |
| *M11 quantile anchor (reference)* | *0.815340* | *0.264308* | *0.404026* | *78.0%* | — | *0.379323* | *+10.020* |
| *M9 hinge (reference)* | *0.864667* | *0.251701* | *0.474210* | — | — | *0.429457* | *+10.691* |

Control reproduced `0.8226433610683003` and `0.381640` exactly. Reading `dense-fails` (primary `-0.026870`), secondary
`ranking-preserved` (`+0.040195`, a gain), verdict `not-cleared`, fit gate **FALSE** (1 of 28 - the first exact board
at `0.0` any arm has produced). Paid charge mean `0.670`, `3.6x` below the extremal anchor's `3.004`.

**What the pass established.** Density was the right lever for everything except the placement. This is the first arm
to beat control on the swept best F1 (`+0.053548`) and on out-of-fold (`+0.065183`), the first to improve the row
diagnostic, and it cut the level drift from `+28.0` to `+4.8` points of the population while quadrupling nothing. Its
primary delta (`-0.026870`) is the smallest any supervised arm has had, an order below M11's `-0.117332`. What it
could not do is put `0.0` where the decisions are: the swept optimum moved **up** to `3.15`, so the read separates well
and sits about three logits high. A per-cell band is indifferent to a translation it can slide inside, which is exactly
the free additive level M11 diagnosed.

**Next arm, declared by the data.** *Midpoint anchoring*: `mean over qualifying boards [ |(m_occupied + m_empty)/2| ]`
over known visible cells, pooled at `ANCHOR_WEIGHT = 0.5`, no margin (target location is exactly `0.0`), equal class
weight, no parameters and nothing read about prevalence. It is a location instrument rather than a shape: a shift
`+delta` moves both class means by `delta`, so the term grows linearly and unconditionally on drift and cannot be paid
off by raising the occupied class - the trick every earlier term allowed. Its failure mode is collapse of the two
means onto each other, which the published BCE punishes, so its declared reading keeps the `-0.02` AUC tolerance; its
constants, metric and readings belong in their own document before their own receipt.

Defect: V-133 (two hand-derived expectations in this pass's test file written as sums where the declared term is a
mean; the third defect in the test-instrument class, so per AGENTS.md the class now carries a standing instrument - a
plain-Python reimplementation of the declared formula compared against the term on random boards).

## Bounding the mass fails too, and the drifting quantity is the read's level (2026-11-12)

[M11_QUANTILE_ANCHOR.md](M11_QUANTILE_ANCHOR.md) was predeclared before any receipt: the class-quantile anchor
`--occupancy-quantile-anchor 1`, i.e. per qualifying board
`max(0, margin - q10(occupied)) + max(0, margin + q90(empty))` over known visible cells, at `ANCHOR_MARGIN = 1.0`,
`ANCHOR_WEIGHT = 0.5`, quantiles `.10`/`.90`, `linear` interpolation - the same margin and weight as M10, so the only
change is the statistic. Family `m11-quantile-anchor-`, new catalogue `cb8775b5`.

| arm | AUC | plain F1 at `0.0` | best swept F1 | prec/rec at `0.0` | positive at `0.0` | out-of-fold | row error |
|---|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 0.2363 / 0.9910 | 49.9% | 0.400399 | +7.819 |
| class-quantile anchor | 0.815340 | **0.264308** | 0.404026 @ `2.25` | 0.1523 / 0.9970 | 78.0% | 0.379323 | +10.020 |
| *M10 extremal anchor (reference)* | *0.885462* | *0.310966* | *0.540811* | — | *64.5%* | *0.540811* | *+9.779* |
| *M9 hinge (reference)* | *0.864667* | *0.251701* | *0.474210* | — | — | *0.429457* | *+10.691* |

Control reproduced `0.8226433610683003` and `0.381640` exactly. Reading `bulk-fails` (primary `-0.117332`), secondary
`ranking-preserved` (`-0.007303`, inside the `-0.02` tolerance), verdict `not-cleared`, fit gate **FALSE** (0 of 28).

**What the pass established.** The declaration predicted this term would be pointwise *no larger* than the extremal
anchor (it is: paid mean `2.417` against `3.004`; the ordering is `q10 >= min`, `q90 <= max`, so the converse cannot
occur) and hypothesised that pressure landing on the body would beat pressure landing on two cells. **That hypothesis
is refuted.** Every supervised arm so far has *raised* the read instead of pinning it: `49.9%` of cells positive at
control, `64.5%` after M10's anchor, `78.0%` here, and the swept optimum moved from `+0.90` to `+2.25`. A per-board
statistic buys level: raising the occupied statistic is charged once per board while the cost of raising the whole
read is spread across `4,933` cells the term never counts. So the problem was never which statistic - it is that **too
few cells are charged**, and the read's free additive level absorbs the pressure.

**Next arm, declared by the data rather than by the earlier fallback.** The predeclared mean anchor is *not* the next
step and the record says why (same shape of instrument, and a class mean is dominated by confident cells, so the
band that decides `0.0` barely registers). Instead: a class-balanced **dense** margin hinge,
`mean over boards [0.5 mean_occ relu(margin - logit) + 0.5 mean_empty relu(margin + logit)]`, margin `1.0`, weight
`0.5`, over known visible cells only - every cell charged, both classes equally weighted (never by frequency), and
only the cells inside the declared band carry gradient, which is exactly where the operating point is decided. Its
satisfaction implies a perfect `0.0` boundary on the published population, so the informative reading is how much of
that is reached; margin, weight and bindings to be declared in its own document before its own receipt.

Defect: V-132 (a refusal test that would have passed on an unrelated `SystemExit`; now asserts the refusal's own
wording, so a broken guard cannot masquerade as a working one).

## The anchored margin is the best read yet and still not an operating point (2026-11-12)

[M10_SUPERVISED_ANCHOR.md](M10_SUPERVISED_ANCHOR.md) was predeclared before any receipt. It adds the two-sided anchor
the M9 record named - `--occupancy-anchor 1`: for every board carrying both occupied and empty *known* visible
cells, `max(0, margin - lowest_occupied) + max(0, margin + highest_empty)`, averaged over qualifying boards, pooled
at `ANCHOR_MARGIN = 1.0` / `ANCHOR_WEIGHT = 0.5` beside the occupancy term. Family `m10-anchor-`, its own catalogue.

| arm | AUC | plain F1 at `0.0` | best swept F1 | precision / recall at `0.0` | out-of-fold F1 | row error |
|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 0.2363 / 0.9910 | 0.400399 | +7.819 |
| two-sided anchor | **0.885462** | **0.310966** | **0.540811** @ `2.25` | 0.1842 / 0.9970 | **0.540811** | +9.779 |
| *M9 hinge (reference, own catalogue)* | *0.864667* | *0.251701* | *0.474210* | — | *0.429457* | *+10.691* |

Control reproduced `0.8226433610683003` and `0.381640` exactly in catalogue `6e207218`. Reading `anchor-fails`
(primary `-0.070674`), secondary `ranking-preserved` (`+0.062819`), verdict `not-cleared`, fit gate **FALSE**.

**What this pass actually bought.** `0.885462` AUC and `0.540811` best F1 are the strongest occupancy numbers in the
project - above the widened-field arm (`0.882262` / `0.514820`) with no deformation field, no sampling change, no
parameter and no tensor - and the gain transfers: all twelve leave-one-group-out folds chose the *same* boundary
`2.25`, so out-of-fold F1 equals the swept best to six decimals. The anchor also beats the one-sided hinge on both
axes (`0.885462` / `0.310966` vs `0.864667` / `0.251701`). What it did not do is make `0.0` an operating point:
`64.5%` of cells sit above zero (`3,610 / 5,600`), so precision at `0.0` is `0.1842` even at recall `0.9970`.

**Diagnosis, and the next arm.** Both supervised terms are *extremal* - two cells of two hundred - so the empty
**mass** is unconstrained and rides upward; the term reached `0.0` on the final step while `99%` of cells were free
to sit anywhere. Next: the **class-quantile anchor**,
`mean [max(0, margin - q10(occupied)) + max(0, margin + q90(empty))]`, same family, same wiring pattern, margin /
weight / quantiles declared before its receipt. Its ceiling is declarable in advance (if `90%` of empty cells sit
below `-1` and `90%` of occupied above `+1`, precision at `0.0` is at least `0.55` at recall `0.90`, F1 near `0.68`),
and its likely cost to the `0.540811` ranking must be watched - mass constraints are strong, which is what M8 showed
with a far blunter instrument. The predeclared learned-bias fallback is argued down in the record: a single bias
shifts both modes and cannot move `90%` of empty cells without dragging the occupied mode with it.

Defect: V-131 (two defects in the previous turn's test instrument - a zero floor that made "clean" arithmetic cases
fail, and a refusal assertion aimed at `main` instead of `arguments` - both repaired before launch; the engine and
read path needed nothing, all eight stages passed first time and the family was registered before training).

## Supervising the margin buys ranking and still not a boundary (2026-11-12)

[M9_SUPERVISED_MARGIN.md](M9_SUPERVISED_MARGIN.md) was predeclared before any receipt. It is the first pass that
*supplies* the boundary in the objective instead of choosing one afterwards: a declared within-board hinge over the
known visible cells of every board that carries both classes,
`mean max(0, margin - (lowest_occupied - highest_empty))`, with margin `1.0` and weight `0.5` fixed in the doc
before any receipt, no parameter, no threshold, no prevalence term, and `0` exactly the published objective.

| arm | AUC | plain F1 at `0.0` | best swept F1 | precision / recall at `0.0` | out-of-fold F1 | row error |
|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 0.2363 / 0.9910 | 0.400399 | +7.819 |
| margin hinge | 0.864667 | **0.251701** | 0.474210 @ `2.70` | 0.1440 / 0.9985 | 0.429457 | +10.691 |

Control reproduced `0.8226433610683003` and `0.381640` exactly in catalogue `eae71837`. Reading `margin-fails`
(primary `-0.129939`), secondary `ranking-preserved` (`+0.042024`), verdict `not-cleared`, fit gate **FALSE**.

**The split is the result.** The hinge is a real *ranking* lever - `+0.042024` AUC, `+0.062176` best swept F1
(clears the attribution floor on its own), `+0.029058` out-of-fold, all with no sampling change and no tensor - and
a negative *operating point*: the boundary drifted up to `2.70`, `4,625` of `5,600` cells are called occupied at
`0.0`, and the row error worsens to `+10.691`. The paid hinge reached `0.0` on the final step, so separation was
achieved; location was never in the objective, which is exactly what the predeclaration warned.

**Next arm, specified and supported by the data.** Two-sided anchor:
`mean [max(0, margin - lowest_occupied) + max(0, margin + highest_empty)]`. It keeps what the hinge found and pins
the empty mode at least `margin` below zero, so `0.0` is a supervised boundary. Declare its margin and weight
before its receipt, measure control plus anchor in one catalogue, same envelope, same primary metric and reading
vocabulary. Do not re-tune the one-sided margin/weight; it stays at `0` in every other arm as a measured ranking
result.

**Consequence for the row offsets.** The `-1` offset's `+0.048086` AUC was measured against control `0.822643`;
the hinge now sits at `0.864667` and reaches `0.474210` best F1, so that gain is no longer additive-by-default -
re-measure it on top of a supervised boundary, and only beside it (offsets are never scored at a boundary chosen
for them after the fact).

Defect: V-130 (the first scope omitted the paid hinge the predeclaration promised to report; repaired and the
superseded evidence directory kept under `runs/attempts/m9-supervision-evidence-v1/`).

## Normalising the read makes zero a median, not an operating point (2026-11-12)

[M8_NORMALISED_BOUNDARY.md](M8_NORMALISED_BOUNDARY.md) was predeclared before any receipt. It follows M7's
conclusion that a post-hoc rule cannot convert the `-1` offset's ranking gain into F1 under a `11.9%` base rate,
by moving the boundary inside training: the engine now has a declared output parameterisation
(`--occupancy-normalise 1`) that z-scores each board's occupancy logits over its own visible cells **inside the
forward pass**, so the tensor the loss differentiates is the tensor every scorer thresholds, and `0.0` is the
operating point by construction. No parameter, no tensor, no state key, no label, no prevalence term; `0` is
exactly the published function (tested bit-for-bit, plus shift- and scale-equivariance tests).

| arm | AUC | plain F1 at `0.0` | best swept F1 | precision / recall at `0.0` | out-of-fold F1 | row error |
|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 0.2363 / 0.9910 | 0.400399 | +7.819 |
| normalised | 0.788463 | **0.384661** | 0.393300 @ `0.30` | 0.2399 / 0.9700 | 0.393367 | +5.134 |

Control reproduced `0.8226433610683003` and `0.381640` exactly in the new catalogue (`efa02b84`). Reading
`normalisation-moves-without-clearing` (`+0.003021`, bar `+0.05`), secondary `ranking-degraded` (`-0.034181`,
tolerance `-0.02`), verdict `not-cleared`, fit gate **FALSE**.

**The important part is why it could not have worked.** A per-board z-score fixes zero at the board's centre of
mass, so `2,697 / 5,600 = 48.2%` of cells sit above it regardless of what the head learns. At `11.9%` prevalence
a cut admitting `48%` of cells has precision `<= 0.119/0.482 = 0.247` and F1 `<= ~0.396`; the arm measured
`0.2399` and `0.384661`. It reached the ceiling of its own parameterisation - so `+0.003021` is not a lever to
build on, and no amount of training moves it. Removing the board's global scale also deleted information the
ranking used (`-0.034181` AUC, `-0.007032` out-of-fold), while the row error improved `+7.819 -> +5.134` for the
same reason.

**Next arm, specified.** Supervise the boundary directly on the unnormalised read: a declared within-board hinge
on the margin between each board's lowest occupied and highest empty logit, `max(0, margin - (lowest_occupied -
highest_empty))`, with `margin` and its weight fixed before the receipt, so the objective is told what separation
is rather than where zero should sit. Declared as a supervision change; no threshold, no sweep. Note the
parameterisation lever stays in the engine as a measured negative result - do not add it to a positive arm
without a new declaration, because it costs AUC.

Defect: V-129 (curator recorded a Wilson interval where a point recall was declared; caught before the receipt).

## The shifted read has no usable boundary it can find itself (2026-11-12)

[M7_CALIBRATION.md](M7_CALIBRATION.md) was predeclared before any receipt of the block; the mechanism is a
per-board threshold chosen from each board's own 200 occupancy logits by Otsu's criterion - no labels, no swept
F1, no held-out boundary. It is translation-equivariant, so it absorbs an additive misregistration shift, which
is what the `0.90 -> 2.70` boundary drift of the `-1` row offset looked like. Three arms were re-captured in one
catalogue (`771d7fc8...`) and every read was taken read-only over the frozen sweeps.

| arm | AUC | F1 at `0.0` | best swept F1 | F1 at its own label-free thresholds | threshold mean (sd) |
|---|---|---|---|---|---|
| control | 0.822643 | 0.381640 | 0.412034 @ `0.90` | **0.381117** | `-0.188` (0.140) |
| wide | 0.882262 | 0.385902 | 0.514821 @ `1.95` | **0.358234** | `-0.362` (0.360) |
| down (`-1`) | 0.870729 | 0.345873 | 0.469259 @ `2.70` | **0.380898** | `1.145` (0.252) |

Control and wide reproduced `0.8226433610683003` and `0.8822623454135491` exactly, and `down` reproduced its M6
record (`0.8707292410960544`, best `0.46925858951175414`).

**Reading `calibration-fails`**: the label-free rule lifts the shifted read `+0.035025` off its own `0.0`
boundary - most of the `0.087386` the label-chosen boundary found - but it lands `0.000742` **below** control and
is `-0.000219` against control at control's own label-free threshold, so the secondary reading is
`offset-advantage-was-a-threshold-effect`. At any boundary the read can supply itself, the offset is not ahead of
control; the `+0.048086` AUC gain is real and threshold-free but does not become an operating point.

**Why, and it matters for what comes next.** Cells are `11.9%` occupied and Otsu maximises *between-class
variance*, which under that base rate cuts inside the empty mass: all three arms land at recall `0.99` and
precision `0.24`, a saturated detector, and the rule costs the best arm `-0.027668` F1. So the finding is
narrower than "no label-free statistic works" - it is that an imbalance-blind rule does not work, and the
boundary has to come from somewhere else. Two candidates, in order of preference:

1. **A learned operating point** - train the occupancy logit in a per-board normalised parameterisation (subtract
the board's mean logit, divide by its spread, both label-free), so `0.0` *is* the operating point and training
places it. This changes the objective's geometry, adds no per-cell parameter, and a 64-step study costs about
four minutes, so it can be measured in the same harness as a single declared arm against control.
2. **A prevalence-aware label-free rule** - a two-component Gaussian mixture crossing per board, which estimates
the mixing weight instead of assuming a balanced split. This is re-scoring only, so it costs seconds, but it is
still a post-hoc boundary and should be read as a diagnostic of candidate 1, not as the fix.

Either way the `-2` offset arm stays unmeasured until a boundary exists that makes the `-1` arm operational;
measuring a second sign at an operating point that saturates tells nothing.

Fit gate stays **FALSE** (0 of 28 exact boards). Verdict `not-cleared` on per-cell F1, ranking AUC and
attribution. Defects V-126 (the run family was not declared, so the read stages were refused after the studies
had trained), V-127 (curator invocation) and V-128 (the weight-blob pruning below).

**Disk policy, so this does not refill.** `runs/` had reached 69 GB, 66.8 GB of it `.pt` weight blobs, on a
volume that was 96% full. After a sweep has been curated, its periodic `training-snapshots/*.pt` and resumable
`checkpoint/*.pt` are dead weight: nothing in the measurement path reads them, because the sweep captures the
logits into `result.json`. Keep the pinned init run (`runs/m3-percell-balance-r1`), the few runs a document
names, and every JSON/log/receipt; prune the rest. 1,165 blobs (56.1 GiB) were pruned this pass, leaving 101 GB
free. The cost is recorded in V-128: the M2/M3 replay analyses can no longer be re-run for pruned families,
because their inputs are gone while their published receipts remain.

## The row bias is real, and it was pointing the other way (2026-11-12)

[M6_ROW_OFFSET.md](M6_ROW_OFFSET.md) was predeclared before any receipt of the block. It starts with a
read-only diagnostic of the four frozen M5 sweeps (`docs/evidence/m5-row-error-r1/`): for every column of
every deciding board, the top-most decided row against the top-most occupied row, in rows, bottom-first.

| arm | mean row error at `0.0` | at own best boundary | exact on the label row |
|---|---|---|---|
| control | +7.819 | +2.886 | 0.013 |
| wide | +5.872 | +0.716 | 0.094 |
| de-meaned | +5.651 | +1.973 | 0.094 |
| column | +6.832 | +1.081 | 0.067 |

**Reading `systematic-row-offset`, positive**: every arm decides the stack above its label row, so the
residual is sampling geometry, not a mis-specified label. The arm follows from that: a declared constant
`--occupancy-offset-rows` (a non-persistent buffer, so no tensor and no state key; `0` is exactly the
published function) measured in both signs, because a one-sign claim cannot distinguish a mechanism from a
lucky direction.

| quantity | control | wide | up (+1) | **down (-1)** |
|---|---|---|---|---|
| deciding AUC | 0.822643 | **0.882262** | 0.814109 | **0.870729** |
| best swept F1 | 0.412034 | **0.514821** | 0.393958 | **0.469259** |
| F1 at `0.0` | 0.381640 | **0.385902** | 0.378286 | 0.345873 |
| out-of-fold F1 | 0.400399 | **0.498413** | 0.326910 | 0.368808 |
| row error at own best | +2.886 | **+0.716** | +4.624 | **+1.342** |

Control and wide reproduced `0.8226433610683003` and `0.8822623454135491` exactly. The **predicted** sign
fails (`-0.018076` F1, `-0.008535` AUC, boundary `0.282` rows further from the labels); the **opposite**
sign clears the attribution floor (`+0.057225` best F1) with a resolvable AUC gain (`+0.048086`, interval
`+0.013560 .. +0.071269`, MDE `0.043683`) and removes more than half the row bias. So most of wide's
`+0.059619` AUC is a one-row registration error, not a shifted-window exploit.

**But it is not a better operating point yet**: F1 at `0.0` falls `0.035767`, the best boundary moves from
`0.90` to `2.70`, and out-of-fold F1 falls `0.031591`; wide is still `+0.045562` F1 ahead. The next
declared instrument is therefore the **operating point** - a label-free calibration of the shifted read on
the same population - and only then a `-2` offset arm, because a shift that keeps moving its threshold is
not yet a usable read. Fit gate stays **FALSE** (0 of 28 exact boards; per-cell F1 still far under `0.70`).
V-123 and V-124 record this pass's defects; the refused receipts are under `runs/attempts/`.

## The column read-out: used heavily, ranks no better (2026-11-12)

[M5_COLUMN_READOUT.md](M5_COLUMN_READOUT.md) was predeclared and committed before any receipt, amended
once before any admissible receipt (V-119: the first version fixed `--occupancy-detail 0` for the whole
family, which the engine refuses for a deformation arm), and then run as four matched 64-step
continuations of `runs/m3-percell-balance-r1` differing only in the deformation lever and the declared
read-out. Control, wide and de-meaned reproduced the previous record exactly, so the block is one
implementation.

| quantity | control | wide | de-meaned | **column** |
|---|---|---|---|---|
| deciding AUC | 0.822643 | **0.882262** | 0.820429 | 0.832796 |
| F1 at `0.0` | 0.381640 | **0.385902** | 0.376663 | 0.384302 |
| best swept F1 | 0.412034 | **0.514821** | 0.427911 | 0.432650 |
| out-of-fold F1 | 0.400399 | **0.498413** | 0.427911 | 0.381790 |
| study peak / wall | 1,019,527,168 B / 246.3 s | 1,093,607,424 B / 249.2 s | 1,043,214,336 B / 246.2 s | 1,024,061,440 B / 215.7 s |

**Verdict `not-cleared`; reading `readout-moves-without-ranking`.** The arm gains `+0.020616` best F1 and
`+0.010152` AUC over control, with an interval `-0.030013 .. +0.062077` that straddles zero at twelve
clusters, and its out-of-fold number (the only one that says a gain transfers) falls by `0.018609`. It
remains `0.082171` behind the wide arm.

The instrument that makes this a result rather than a null: the new read-only **ablation** reloads the
frozen checkpoint, runs the deciding population with the projection as trained and with it zeroed, and
scores both sides on the same cells. Zeroing costs `0.017048` AUC and moves `0.933284` logit on average -
`42.3%` of the mean absolute logit - so the read-out is not idle. What it does is raise true positives by
`89` while raising false positives by `695`: a summary of a column can say *this column has material* and
cannot say *this row is filled*.

**Answer: the vertical information was never the missing piece.** Moving the sample to it (wide, M3),
supervising that move (placed, M4), and putting it beside the cell (column, M5) have now all been measured;
only the one that takes the sample off the board gains, and it does so by being *more precise* at the price
of reading the wrong place. The next declared instrument is cheap and read-only: measure the **signed row
error** of each arm's decision boundary on the frozen sweeps. If the boundary sits about one row from the
labels, the fix is a declared sampling offset - geometry, not tuning - and if it does not, the
label-objective class is what fails and should be named as such rather than tuned around. Fit gate stays
**FALSE** (0 of 28 exact boards on all four arms); V-119 to V-122 record this pass's defects and the
refused receipts are under `runs/attempts/`.

## Placed-target supervision loses, and the samples were reading off the board all along (2026-11-12)

[M4_OCCUPANCY_PLACED.md](M4_OCCUPANCY_PLACED.md) was predeclared and committed before any receipt, then
run as four matched 64-step continuations of `runs/m3-percell-balance-r1` differing only in the
registration lever: control, the widened field, the de-meaned field, and the widened field under a
**placed-target term** that supervises the displacement against the cell each sample should have read
(`smooth-L1` between the applied offsets and the target lattice's per-cell displacement, both latices
detached, targets clipped to the arm's 4-cell reach, weight `0.25`, **no new tensors**). Control, wide and
de-meaned reproduced the previous record exactly, so the block is one implementation.

| quantity | control | wide | de-meaned | **placed** |
|---|---|---|---|---|
| deciding AUC | 0.822643 | **0.882262** | 0.820429 | 0.762448 |
| F1 at `0.0` | 0.381640 | **0.385902** | 0.376663 | 0.377663 |
| best swept F1 | 0.412034 | **0.514821** | 0.427911 | 0.378316 |
| out-of-fold F1 | 0.400399 | **0.498413** | 0.427911 | 0.378316 |
| study peak / wall | 1,047,506,944 B / 322.7 s | 1,059,835,904 B / 317.0 s | 1,083,195,392 B / 191.4 s | 1,038,139,392 B / 382.9 s |

The placed arm is **`not-cleared` and worse than control**, with resolvable AUC losses against control
(`-0.060196`, `-0.116936 .. -0.015133`) and wide (`-0.119815`, `-0.208163 .. -0.052118`). Its field tells
the story: constant-share **0.892** (it did become a translation toward the right place) but mean
`|offset|` only **0.462** cells against a target of about `2.6`. The same zero-initialised convolution,
over the same 64 Adam steps at the same rate, was moved `3.760` cells by the occupancy gradient and
`0.462` by the supervision. The auxiliary weight that keeps the term honest is also, in this budget, too
small to express it - and the sixth of the way it did travel costs resolvable AUC.

The new read-only **position instrument** settles the previous pass's open question. The wide arm's
`5.128`-cell translation moves **5.0%** of samples out of the image frame and leaves **64.1%** outside the
target board's own cell frame (control 29.0%). The de-meaned arm is the only arm that *improves* board
coverage (**0.777** against control's 0.710) and it gains nothing. So the objective's preferred
displacement is a **shifted-window read** in which the occupancy label tracks the stack below the cell,
not the cell itself.

**Answer: registration is not a lever this architecture carries through this read-out, and neither is its
supervision.** Occupancy-only supervision (de-meaned) is per-cell and useless; displacement supervision
(placed) is too weak to enforce itself and harmful where it lands. The next declared instrument is a
**different read-out**: cell-relative vertical context - a per-column stack profile, or a tap that sees
the column below the cell - so that the only displacement the objective rewards is not a shift off the
board. A stronger placed term only becomes worth measuring after that, because the target displacement it
enforces is the one the current objective pays to avoid. No reach or bound change can help. Fit gate stays
**FALSE** (0 of 28 exact boards on all four arms); V-109 through V-116 record this pass's defects and the
failed receipts are under `runs/attempts/`.

## The constant mode was the whole signal: per-cell registration is learnable and buys nothing (2026-11-12)

[M3_OCCUPANCY_REGISTRATION.md](M3_OCCUPANCY_REGISTRATION.md) was extended with a fourth arm
(`occupancy_deform=3`) whose applied field is **de-meaned across the board**, so the constant mode is
exactly zero by construction, not by penalty: the same coarse-plus-fine field as the wide arm, then
`shift = shift - shift.mean(dim=1, keepdim=True)`. It has an identical parameter set to the wide arm and
is bounded at `2 x DEFORM_REACH = 8.0` cells, because clamping after de-meaning would reintroduce a
mean; it never comes near that bound.

Four matched 64-step continuations of `runs/m3-percell-balance-r1` differ only in that lever, and **all
three shared arms reproduced the previous block exactly**: control `0.8226433610683003`, one cell
`0.8665658048737642`, wide `0.8822623454135491` AUC, with per-cell F1 and best swept F1 identical to
`docs/evidence/m3-occupancy-deformation-r2/scope.json`. The comparison is admitted.

| quantity | control | one cell | wide | **demeaned** |
|---|---|---|---|---|
| deciding AUC | 0.822643 | 0.866566 | **0.882262** | 0.820429 |
| F1 at `0.0` | 0.381640 | 0.369420 | **0.385902** | 0.376663 |
| best swept F1 | 0.412034 @ `0.90` | 0.490752 @ `3.15` | **0.514821** @ `1.95` | 0.427911 @ `0.60` |
| out-of-fold F1 | 0.400399 | 0.456054 | **0.498413** | 0.427911 |
| exact nonempty boards | 0 of 28 | 0 of 28 | 0 of 28 | 0 of 28 |
| study peak / wall | 1,053,507,584 B / 240.6 s | 1,064,845,312 B / 154.8 s | 1,042,235,392 B / 302.7 s | 1,041,952,768 B / 367.5 s |

The de-meaned arm is **`not-cleared` on all three conditions**: +0.015877 best F1 against control
(below the 0.05 floor), AUC `-0.002215` with an interval of `-0.037595 .. +0.033223` and MDE
`0.050764`, so it is **indistinguishable from control**, not "slightly worse". Against the wide arm it
loses **0.061834 AUC** with an interval clear of zero — resolvably worse.

The field settles what happened. The de-meaned arm's constant share is **0.000752** (wide: 0.978), its
board mean is `(-0.001, +0.024)` (wide: `-3.488, -3.760`), and it is genuinely per-cell rather than
inert: mean `|offset|` **0.541** cells, **1.153** on occupied cells, p50 0.283, p90 1.358, max 3.387,
**nothing at its bound**, 36.5% of samples outside their nominal cell. Declared reading:
**`registration-without-signal`** — the displacement is expressible and the objective cannot use it.

**Answer: per-cell registration is not a lever this architecture carries through the occupancy
objective, and the family's entire gain was the translation.** A free field that can only shift the
read-out gains `+0.061834` AUC over one that is forced to move per-cell; nothing in the loss prefers the
correct cell. The translation it chose is **5.128** cells Euclidean — twice the 2.57-cell residual the
oracle pairing measured — so the next pass should either **supervise the displacement** with a
placed-target term (the only change that adds information) or **re-project the field into image space
with an in-frame fraction** to tell a five-cell decode offset from a shifted-window exploit. Not
another reach or bound change. Fix gate stays **FALSE** (0 of 28 exact boards on all four arms);
V-107 and V-108 record this pass's own curation defects, preserved under `runs/attempts/`.

## The widened field answers the registration question: the bound capped it, and what it learned is a translation (2026-11-12)

[M3_OCCUPANCY_REGISTRATION.md](M3_OCCUPANCY_REGISTRATION.md) was extended, predeclared before its
receipts, with a **coarse-to-fine field that reaches four cells** (`occupancy_deform=2`: the existing
one-cell fine level plus a coarse level predicted from an eight-times pooled map at three cells),
because the measured placement residual is 2.57 cells and a one-cell bound cannot express it. Three
matched 64-step continuations of `runs/m3-percell-balance-r1` differ only in that lever, and the third
arm keeps the one-cell field selectable. The one-cell arm **reproduced the previous pass exactly**
(AUC 0.866566, best F1 0.490752 at 3.15), so the block is one implementation.

| quantity | control | one cell | wide (4 cells) |
|---|---|---|---|
| deciding AUC | 0.822643 | 0.866566 | **0.882262** |
| F1 at `0.0` | 0.381640 | 0.369420 | **0.385902** |
| best swept F1 | 0.412034 @ `0.90` | 0.490752 @ `3.15` | **0.514821** @ `1.95` |
| out-of-fold F1 | 0.400399 | 0.456054 | **0.498413** |
| exact nonempty boards | 0 of 28 | 0 of 28 | 0 of 28 |
| study peak / wall | 1,044,045,824 B / 150.6 s | 1,065,558,016 B / 196.1 s | 1,045,843,968 B / 232.6 s |

Wide vs control is the family's largest attribution (**+0.102787** best F1, **+0.059619** AUC with
interval +0.029411 .. +0.094463, MDE 0.049478) and it recovers the fixed-boundary F1 the fine field
cost (**+0.016482** over the one-cell arm, resolvable). The wide-vs-one-cell AUC increment (+0.015697)
and every out-of-fold delta here are **not resolvable** at twelve clusters.

What settles the question is the field itself. The one-cell field sat at `>= 0.95` of its bound on
**70.4%** of cells (mean 0.909, mean vector 1.23 cells) - it wanted to move further. The wide field
saturates harder (**83.8%** at four cells) and applies a mean **3.760** cells (**5.128** Euclidean),
which is twice the 2.57-cell residual, with **97.8% of its displaced energy in one constant vector**
(-3.488, -3.760) and 99.8% of samples leaving their nominal cell. A near-uniform offset field is a
lattice shift, so the trained mechanism is a read-out **translation**, not per-cell registration. Per
the predeclared rule the reading is **`reach-exceeded`**.

**Answer: per-cell registration is not a lever this design can resolve.** Releasing the bound did raise
ranking and restore the fixed boundary, but it let the field express a constant translation; nothing in
the occupancy objective distinguishes the correct cell from a re-centred read-out. The next lever is
supervision that can tell them apart - a zero-mean constraint on the field, or a placed-target term
that scores the displacement - **not** more reach, since the wide field already saturates. The fit gate
stays **FALSE** (0 of 28 exact boards on all three arms).

V-104 (the widened arm's module missed the initial-weight allowlist), V-105 (the curator passed tuple
counts to a mapping helper) and V-106 (a vacuous constant-share statistic) are logged with their
records preserved under `runs/attempts/`; two admission refusals from host commit pressure are
preserved there too.

## Registration is a real mechanism now: it closes half the oracle gap in ranking and the replay point can no longer read it (2026-11-12)

[M3_OCCUPANCY_REGISTRATION.md](M3_OCCUPANCY_REGISTRATION.md) was predeclared before its first receipt and
is bound by `tests/test_occupancy_deform.py`. The campaign's largest attributed residual is **placement**:
per-cell F1 `0.4390 -> 0.5742` when model cells are paired with the cells they cover, 48.3% of cells on the
wrong target cell. Every earlier geometry lever corrected the **lattice** globally (origin, extent, stage
residual, board-conditioned four-vector, boundary anchors); this pass corrects **each cell locally**: a
zero-initialized `Conv2d(48 -> 2, 3x3)` over the tap's own feature predicts a per-cell shift, clamped to one
cell, added to that cell's sample offset before bilinear sampling. 866 added weights, trained by the
existing occupancy term, no new supervision. Identity at step zero by construction, so both arms compute
the published function before training and every shared tensor is byte-identical.

| quantity | lever (field) | control (detail) |
|---|---|---|
| deciding AUC | **0.866566** | 0.822643 |
| F1 at `0.0` | 0.369420 | **0.381640** |
| best swept F1 | **0.490752** at `3.15` | 0.412034 at `0.90` |
| best-boundary precision | **0.416754** | 0.263488 |
| out-of-fold F1 | **0.456054** | 0.400399 |
| zero-boundary cost | -0.121332 | -0.030394 |
| exact nonempty boards | 0 of 28 | 0 of 28 |
| study peak / wall | 1,090,600,960 B / 142.5 s | 1,058,295,808 B / 246.7 s |

**`not-cleared`** on the per-cell gate (`0.369420` < `0.70`) and the ranking gate (AUC `0.033434` short of
`0.90`, best F1 `0.209248` short of `0.70`), with the attribution floor **met**: best F1 **+0.078718**,
AUC **+0.043922**. That is **48.5%** of the control's `0.162166` distance to the oracle-registration level
at the same 64-step budget, from 866 weights - the largest attribution measured in this read-out family.

What resolves and what does not, before any interpretation: the AUC delta **+0.043922** (interval
+0.017963 .. +0.063903, MDE `0.032257`) and the fixed-boundary deltas (F1 **-0.012220**, precision
**-0.009380**, both intervals clear of zero) are **resolvable**; the out-of-fold F1 delta **+0.044328**
is **not** (interval -0.105165 .. +0.290645, MDE `0.315147`). Best swept F1 and best-boundary precision
carry no interval because their boundaries are fitted on the cells they score.

The mechanism works and the decision boundary no longer reads it: the field's own logit scale moved the
best threshold `0.90 -> 3.15` (and the leave-one-group-out thresholds sit at `3.15` in ten of twelve
groups), so the fixed `0.0` replay point now leaves `0.121332` F1 on the table against the control's
`0.030394`. **Next strongest lever: read this arm at a calibrated boundary, then repeat this exact
matched pair at a matched multiple of budget** - not a wider trunk, which nothing here indicates. The
fit gate stays **FALSE** (0 of 28 exact boards on both arms).

One defect in this pass's own curation, preserved rather than overwritten: **V-103** (the first
curation read mechanism precision at the fixed `0.0` boundary the lever moves, the V-091 pattern),
with its scope and receipt under `runs/attempts/m3-occdeform-curation-r1-defective/`. The control
study's first launch was also killed with its shell at the host's 295 s command cap; that `running`
receipt and partial output are preserved under `runs/attempts/m3-occdeform-control-killed-r1/` and the
arm was re-run under the declared name.

## The resolution lever is measured at last: it clears the attribution floor and still misses both gates (2026-11-12)

[M3_OCCUPANCY_RESOLUTION_R2.md](M3_OCCUPANCY_RESOLUTION_R2.md) was predeclared before its first receipt
and is bound by `tests/test_resolution_r2.py`. The first declaration of this lever was **refused**, not
measured: its stride-1 branch cost 188.4 MiB against a 1,152 MiB ceiling it missed by 22.8 MiB, so the
one information-changing read-out question in the campaign had no number. This pass re-declares the
same single change at an admitted **1,792 MiB** envelope - the host allows `min(RAM/3, 8 GiB)` and
the branch's own two-step profile reached 1,485,766,656 B, so the new ceiling is measured, not
liberal - and nothing else: same checkpoint `9ae1fef9f649...`, same seed, curriculum, capture side,
occupancy terms, 64 steps, gates, floor, population and grid.

Both arms passed admission and completed 64 steps under one catalogue; both sweeps scored the
widened deciding population (36 images, 28 nonempty boards, 667 occupied, 4,933 empty, 12 groups).

| quantity | lever (stride-1) | control (stride-2) |
|---|---|---|
| deciding AUC | **0.868612** | 0.822643 |
| F1 at `0.0` | **0.413858** | 0.381640 |
| best swept F1 | **0.472012** at `1.65` | 0.412034 at `0.90` |
| out-of-fold F1 | **0.460754** | 0.400399 |
| exact nonempty boards | 0 of 28 | 0 of 28 |
| study peak / wall | 1,500,233,728 B / 217.6 s | 1,061,023,744 B / 122.7 s |

**Verdict `not-cleared`** on the per-cell gate (`0.413858` < `0.70`) and the ranking gate (AUC
`0.031388` short of `0.90`, best F1 `0.227988` short of `0.70`), with the attribution floor **met**:
best F1 **+0.059978**, AUC **+0.045968**, fixed-boundary F1 **+0.032218**, precision **+0.025637**
(below its `0.05` floor, so `precision_improved = false`). It is the first matched continuation in the
read-out family to clear that floor.

The calibration-neutral report is what keeps that honest, and this is its first use. The components
separate: fixed-boundary **+0.032218** (whole-cluster interval **+0.016476 .. +0.039566**, MDE
`0.016226`) is resolvable, out-of-fold **+0.060355** (interval -0.138621 .. +0.179634, MDE `0.222851`)
is **not**. The lever also shifts its own zero-boundary cost (-0.058154 against -0.030394), so part of
the fixed-boundary gain is offset, not separation. At twelve clusters this pass can say finer sampling
helps and cannot say how much of it transfers - and neither reading is above the bar the fit gate
needs. **Sampling is now closed with a number; registration remains the attributed residual**
(+0.1352 F1 when model cells are paired with the cells they cover, at 4x budget).

Two defects were found in this pass's own curation before anything was published, both preserved
under `runs/attempts/` rather than deleted: V-097 (components recomputed over all 36 images while the
verdict reads the 28 nonempty boards) and V-098 (a bootstrap interval printed beside a different
delta). Both are now bound - one `deciding_rows` helper, an assertion that each arm's component
matches the sweep-derived row, and a guard that a reported interval belongs to the statistic it names.

## The standing occupancy attribution is a boundary, not a model difference, and two compute refusals reverse (2026-11-11)

[M3_OPERATING_POINT.md](M3_OPERATING_POINT.md) was predeclared before its receipt; it is read-only
(`docs/evidence/m3-operating-point-r2/result.json`, `runs/m3-opaudit-r5-job.json` passed, 88.7 s, peak
46,661,632 B of 128 MiB) and it moves no verdict. Every occupancy verdict in the campaign was read at
a fixed threshold `0.0`, while the levers move that boundary: the fit-budget arm's best swept
threshold is `4.35` against its control's `0.15`, the width lever's `+0.75` against `-0.75`, the
schedule's `+1.95` against `0.0`. On the committed per-cell logits of 27 arms over the widened
deciding population (36 cases, 12 ancestry groups, 7,200 known cells, 667 occupied, 28 nonempty
boards) the audit recomputes four reads per arm and chooses the threshold **leave-one-group-out** -
the sweep's own grid, criterion and tie-break, with the boundary never seeing the cell it scores.
Every receipt reproduced its own published `deciding.auc` and `threshold 0.0` F1 first; the shared
control's `0.414649` is the verdict table's own number.

**Verdict `standing_read_confounded`:** of 14 declared pairs, **6** are `refusal_reversed` (standing
attribution below +0.05, out-of-fold at or above it), **1** `gain_reversed`, **5** `shifted` by at
least a floor, **2** `aligned`. Paired whole-group intervals, reported and never read by a state,
leave **2** reversals standing:

- the **1.5x fit budget** at **+0.149** out-of-fold F1 (interval **+0.006 .. +0.277**), AUC +0.074 and
  its own published placement better (1.9812 -> 1.8419 cells), refused by the standing read at
  **-0.134**;
- the **cosine schedule** at **+0.084** (**+0.010 .. +0.168**), AUC +0.041, placement 39.9% better,
  refused at **+0.005**.

Five further reversals are unresolved at twelve groups (stencil +0.067, translation +0.051, extent
+0.072, board refinement +0.053, and the precision arm's credited +0.097 that becomes -0.018 out of
fold). Four refusals are supported (stacking -0.062, lattice-free -0.043, 4x scale -0.024, support
anchors -0.018), so the read-out family's negative is not uniformly an artifact.

**No gate moves:** 0 of 28 exact nonempty boards at `0.0` in every arm, 0 of 28 under the out-of-fold
boundary in every arm but the cosine arm's 1, and no arm's `cv_f1` exceeds 0.533, so the level claim
stands and the fit gate stays FALSE. The width `capacity_null` is unaffected - it read best swept F1
and AUC. What the audit retires is the *compute* closure: "budget is not the limiter" and "more
optimisation is not a substitute" were read at a boundary the levers moved. Ranked next is a
declared calibration-neutral re-read of those two arms, with replicate arms and a group-level
interval, before any further lever is read at `0.0`. V-091 records the defect and its standing
instrument; V-092 records the join defect this pass exposed in its own first two attempts, both
preserved under `runs/attempts/`.

## Every power figure is generated, suite-checked, and the guard is itself exercised by mutation (2026-11-11)

[M3_MINIMUM_DETECTABLE_EFFECT.md](M3_MINIMUM_DETECTABLE_EFFECT.md) no longer contains a typed number:
its figure block is emitted by `scripts/power_analysis.py --render` from
`docs/evidence/m3-power-mde-r3/result.json`, and `tests/test_power_evidence.py` recomputes the whole
analysis from the nine committed sweep receipts, requires the committed evidence to equal that
recomputation field by field, requires the document's block to equal `render`, and requires the
headline figures in this file and ROADMAP to equal the recomputed strings. The check is demonstrated
rather than assumed: negative tests mutate a figure inside the block, a row of the `20261103`
decomposition table and a number in the committed evidence, and require the check to raise with a
message naming the figure, so a guard that accepted a mutation would itself fail (V-096).

The figures: MDE `0.083151` for the budget lever and `0.126475` for the
schedule at 80% power, both observed magnitudes inside them, power at a true `+0.05` of
`0.3917` and `0.1981`; the audit's fit-budget pair
at MDE `0.196212` with power `0.5665` against its own claim; cluster design
effects against cells of `16.8` and
`29.2`.

**The between-population variance is a band, not a point.** On 2 df the scaled chi-square interval is
`0.003959` to `0.231199` around `0.011859`, and
propagating it through the same seeds-needed arithmetic gives `3 / 5 / 83 seeds and 1.20 / 1.99 / 33.11 arm-hours` (low / point / high) for the
`+0.149` claim, against `44 seeds, 528 clusters, 17.55 arm-hours` for the campaign's own `+0.05` floor. The range is
`3 to 83` seeds and `1.20 to 33.11` arm-hours, not one number: the point
estimate clears `+0.149` by roughly 0.003, so the band's weak end is what a claim that must actually be
settled should be planned against.

The conclusion is unchanged: the lever reads were underpowered by construction, the compute closure is
retired because it could not be resolved at the design that was run, and no verdict is re-read here.

## The compute levers do not replicate, so the compute closure is retired but unconfirmed (2026-11-11)

[M3_COMPUTE_REPLICATION.md](M3_COMPUTE_REPLICATION.md) was predeclared before its first receipt and is
read-only at `docs/evidence/m3-compute-replication-r1/result.json` (`runs/m3-replread-r1-job.json`
passed, 29.4 s, peak 40,108,032 B of 256 MiB). Seeding decided the design: `sample_case` derives each
scene from `scene_seed = seed + group_index`, so a seed is a different synthetic population, and
`train_m2_stream` refuses a continuation whose source seed differs, so the audit's exact arms cannot
be pseudo-replicated. The block is three fresh-baseline seeds x three arms, paired within seed:

| seed | control AUC | base128 | budget192 | cosine128 |
|---|---|---|---|---|
| `20261101` | 0.800414 | 0.400447 | 0.370578 | 0.463932 |
| `20261102` | 0.826166 | 0.312561 | 0.321326 | 0.417742 |
| `20261103` | 0.917477 | 0.673421 | 0.416493 | 0.552129 |
| pooled, 36 clusters | - | 0.443717 | 0.364118 | 0.464992 |

All nine arms completed their declared steps with matched starts inside each seed and different
initial weights and populations across seeds, and all nine sweeps reproduced their own published
`deciding.auc` and `threshold 0.0` F1 before contributing, so no seed is unreadable.

**Both levers are `unconfirmed` under the predeclared rule.** The 1.5x fit budget is `-0.0796`
(interval `-0.1380 .. -0.0250`; per seed `-0.0299`, `+0.0088`, `-0.2569`); the cosine schedule is
`+0.0213` (`-0.0721 .. +0.1136`; `+0.0635`, `+0.1052`, `-0.1213`), against the `+0.05` floor with a
paired whole-group bootstrap over 36 clusters. Each fails its own conditions - point below the floor,
lower bound not above zero, one seed reversing - and no condition was relaxed afterwards. Controls
clear the 0.70 AUC floor everywhere, so `floor_limited` never applied.

Two findings outrank the verdict. Control quality moves 0.12 AUC across populations while either
lever moves pooled F1 by under 0.08, so a single-trajectory delta is not a measurement of a lever -
the audit's one-trajectory reversals had no way to separate lever from population. And the budget
interval excludes zero *below* control at 128 -> 192 steps, the opposite direction to the audit's
`+0.149` at 256 -> 384 on a curated continuation; the block cannot arbitrate that contrast (fresh
starts only), so the correct object is that the lever has no replicated support at any scale tested
and points down at the scale that could be replicated.

Consequence: the compute closure stays retired but is now **unconfirmed** rather than a live lever,
no further compute arm is ranked on the audit's reversal, and the width `capacity_null` stands as the
remaining actionable verdict. Arbitrating `+0.149` needs its own 256/384 block at three or more seeds
under this same rule. V-093 records the restart defect this pass exposed.

## The readout cleanup is proven at every consumer, and a receipt reproduces (2026-11-11)

`tests/test_consumer_pins.py` drives each converted non-loss call site - `visible_units`, `sweep_logits`,
`reading_map`, `pilot_maps`, `pack`, `fit_correct` - with a 26-visible-row declaration and with a
shape-mismatched map, so every consumer reads the board its declaration names and refuses one it cannot cover.
All three converted replay instruments reproduce their historical receipts end to end under the published
config - headroom (`docs/evidence/m3-deciding-headroom-rerun-r4`), the occupancy sweep
(`runs/m3-sweep-obcref-rerun-r4`, verdict `uninformative`, 54 cases, grid and sweeps identical) and the stream
error probe (`docs/evidence/m3-fit-occupancy-r1-errors-r5`, identical apart from two default-valued fields its
receipt predates) - with only digests and timing differing, so no published attribution, F1 or threshold moved.
Identity no longer depends on import order: `checkpoint_architecture` now loads the present
`src/tetris_vision` by path under a private name (`working_tree_model()`), and
`tests/test_identity_import_order.py` proves it survives an archive-era `model.py` shadowing `sys.path` (V-090).
All three instruments were re-run after that fix and still reproduce their receipts, now compared field by field by `scripts/replay_equivalence.py`.The `uninformative` verdict now carries the deciding population's power (`docs/evidence/m3-sweep-deciding-power-r6/power.json`): 36 images in 12 ancestry groups and 7,200 known cells (667 occupied, 9.3%), which resolves a pooled AUC to +/-0.020 at 95%; the 0.90 bar sits 7.4 standard errors above the observed 0.8255 with the upper bound 0.8452 below it, the best swept F1 0.4237 is 0.276 short of its 0.70 bar, and the optimum threshold -0.30 is interior with an F1 range of 0.4237 across the grid. `uninformative` is a measured miss of the predeclared bars, not an artefact of an underpowered subset or an unresolvable grid. The frozen `tests/fixtures/m2_original/losses.py` keeps its literal deliberately
and a test pins that, so the scan exclusion is documented. Full suite green at 874 tests
(`runs/suite-falsification-r2-job.json`, passed, 60.0 s, peak 603,385,856 B).

## The width null is read-validated: the decision path already consumed the whole visible board (2026-11-11)

`output["final"]["occupancy"]` is `(1, 26, 10)` - one logit channel per cell over 20 visible rows plus the
6-row spawn margin - so the probe's `[:, :20]` was a row slice, not a channel truncation, and every labelled
visible cell was already read; the spawn rows are unscored, so no trunk capacity was lost. The literal was
still replaced by `visible_board_logits(occupancy, config)` in `scripts/width_lever_read.py`: the window is
sized from the verified config, the map shape is asserted, and every receipt now records the read-out shape
and per-block logit statistics (control mean absolute 3.958 / std 2.791 on the visible block, lever 1.217 /
1.352). The declared pass was re-run under the corrected read (`docs/evidence/m3-trunkwidth-r2`, admission
`runs/m3-trunkwidth-r2-profile-r1`, 131.3 s, peak 679,268,352 B) and reproduced r1 exactly - `capacity_null`,
attribution 0.0322, 2,701 of 7,200 flips, 0/28 exact - so the width null stands and is not a read-out
artifact. The wider arm's ~3.3x smaller logit scale is the same effect behind its swept threshold moving from
-0.75 to +0.75. V-089 records the latent literal and its now-cleared coupling: the loss, metrics, trainer fit read,
headroom and sweep reads and all three probe/pilot readers size the window from the config or the target
through `board_window`/`visible_window`, a standing test fails if the literal returns, and the base arm's
published development evaluation reproduces to the digit under both the working tree and the frozen
pre-change archive (`docs/evidence/visibleboard-proof-{current,archive}-r1`: loss 18.55253728230794,
cell accuracy 0.8986111150847541, geometry 2.192851414283117). The full suite is green at **842 tests**
(`runs/suite-falsification-r2-job.json`, passed, 60.0 s, peak 603,385,856 B).

## The decision-path width lever is a null, and the wider arm fits worse (2026-11-11)

[M3_TRUNK_WIDTH_LEVER.md](M3_TRUNK_WIDTH_LEVER.md) is spent; the predeclaration was on disk before any
receipt, so nothing here moves a rule. Two fresh-baseline arms differing only in `dimension` - control
8,009,097 parameters at 192, lever 10,565,241 at 240 - trained to the declared 256 steps (peak 991,731,712 B
and 1,031,798,784 B under the 2048 MiB ceiling, 984.8 s and 846.5 s) and were read under their own admission
profile (4 forwards, 10.6 s) and then as the declared pass (`docs/evidence/m3-trunkwidth-r1/scope.json`, 108
forwards, 139.1 s, peak 679,141,376 B of 1024 MiB, weights unchanged).

**Verdict `capacity_null`:** attribution +0.0322 against the +0.05 floor, best swept F1 0.4178 -> 0.4500,
F1 at threshold 0 0.4075 -> 0.3875, deciding AUC 0.7815 -> 0.8561, 0 of 28 exact nonempty boards in either
arm. The activation floor was met (2,701 of 7,200 known cells flipped, 37.5%) and the control cleared the
0.70 AUC floor, so the pass speaks rather than voiding. The lever also reads **worse on the trainer's own
development evaluation** (AUC 0.9268 -> 0.8468, geometry 2.19 -> 5.97 cells) because at 256 fixed steps it
spends ~1.35x the FLOPs per step: this is capacity at **step parity, not compute parity**.

Consequences recorded: widening the decision path is not the lever, stacking more width is not the next step,
and any future capacity pass must be compute-matched with a diagnostic that separates optimisation cost from a
representational ceiling. The published configuration keeps `dimension` 192; `TetrisStateNet-M3` is untouched.
One instrument limitation is recorded with the outcome: the first read attempt
(`runs/m3-trunkwidth-r1-interrupted-attempt-job.json`) reports `wall_timeout` at 4,822.5 s against a 300 s cap
because the host was suspended mid-pass - a suspended host cannot run the supervisor - so no wall-cap claim is
made across host suspension; the kernel memory ceiling held and the re-run passed from scratch.

The verdict rule is pinned by `tests/test_width_lever.py` (all five branches, the activation void and the
guard refusals), so it cannot be chosen after seeing numbers. The full suite is green at **826 tests**
(`runs/suite-trunkwidth-r2`: job passed, 60.0 s, peak 603,385,856 B), and V-088 records the legacy-manifest
case the new identity guard had to narrow.

## Structural and wider changes are authorized: the decision-path width lever is predeclared (2026-11-11)

The owner authorized **all structural and wider changes**, which closes the A/B fork in
[M3_TRUNK_WIDTH_PRICING.md](M3_TRUNK_WIDTH_PRICING.md) as *authorized* and replaces it with an executable lever,
predeclared before any receipt in [M3_TRUNK_WIDTH_LEVER.md](M3_TRUNK_WIDTH_LEVER.md). The predeclaration records two
findings that changed the plan rather than ratifying it. First, the priced wide A widens the **encoder** only: it
adds 826,272 parameters while `dimension`, the six decoder blocks, the attention and the `Linear(dimension, 1)`
occupancy head stay at 192, so it adds no decision capacity - and the `bias_limited` headroom verdict localises the
residual exactly there. The lever is therefore `dimension` 192 -> 240 on the published widths, one knob on the
decision path, heads staying at 6. Second, the contractual exact matched start that every previous lever met is
**not achievable** for a width change: `GroupNorm(8, c)` statistics shift under partial channel duplication and no
fixed affine correction restores them, while uniform 2x duplication exceeds the 256-channel cap. The pass is a
fresh-baseline pair (both arms seeded random, extra ledger note: incomparable to the current table) and its
`floor_limited` branch names what a collapsed pair means. The identity is derived from the verified configuration -
`TetrisStateNet-M3` keeps its name, a wider shape becomes `TetrisStateNet-M3-W48x96x192x256-D240` - the knob defaults
to published, and a bounded 2048 MiB pilot must clear the envelope before either 256-step arm is admitted. No arm is
launched, the fit gate stays FALSE, and no prior verdict moves.

## The deciding error is bias, not trajectory variance: the width fork has no cheap alternative (2026-11-11)

[M3_DECIDING_HEADROOM.md](M3_DECIDING_HEADROOM.md) (predeclared before any receipt,
`docs/evidence/m3-deciding-headroom-r1/scope.json`) read the one decomposition the campaign was missing.
Two committed arms of the identical architecture - same initial tensor file, seed, catalogue and data
order, differing only in schedule - were reloaded and swept in one read-only execution, and their
unweighted mean logit was read by the predeclared three-branch rule on the declared deciding population
(12 groups, 36 cases, 28 nonempty boards, 667 occupied cells, 108 forwards, peak 635,166,720 B
of 640 MiB). The members reproduce their sweeps to the digit (AUC 0.8457 / 0.8864, F1@0 0.4307 /
0.4357).

**Verdict `bias_limited`:** ensemble AUC 0.8872 and best swept F1 0.5074 (-0.0099 against the
better member), standing-point F1 +0.0259, 0 exact nonempty boards. The two trajectories disagree
on 169 of 667 occupied cells, so variance exists and averaging still buys no separation:
the residual is a representational ceiling shared by both solutions. **Ensembling and more optimisation
are not a substitute for the priced trunk-width fork** ([M3_TRUNK_WIDTH_PRICING.md](M3_TRUNK_WIDTH_PRICING.md)),
which stays the user's decision; development numbers (AUC 0.9672, best F1 0.7716) are
descriptive only. V-087 records the narrow-population default this pass exposed and the required-input
guard that now binds it.


## The schedule lever is measured and it is a null; the regime is audited (2026-11-11)

The first ranked step of [M3_OPTIMISER_REGIME.md](M3_OPTIMISER_REGIME.md) is spent
([M3_OPTIMISER_SCHEDULE.md](M3_OPTIMISER_SCHEDULE.md), predeclared before any receipt,
`docs/evidence/m3-optimiser-schedule-r1/scope.json`). Two matched 256-step arms from `9ae1fef9f649...`
differed only in the schedule - constant 1e-3 against cosine 1e-3 -> 1e-4 - with **zero added
parameters**, both complete under 1,152 MiB / 1,800 s (peaks 1,016,332,288 / 1,071,558,656 B).

The schedule is proved from the runs, not the code: every curve record carries its own
`learning_rate`, the control shows **1 distinct rate** and the lever
**256**, strictly decaying 0.001 -> 0.0001. Matched
start holds on identical initial weight bytes, seed and step-1 `loss`/`gradient_norm`.

**Verdict `null`:** per-cell F1 at `0.0` 0.4307 -> 0.4357
(+0.005041 against a +0.05 floor), placement 39.9% better, AUC
0.8457 -> 0.8864 and the fit gate unchanged at **0 of 28** in both arms. The best
swept F1 did rise ( 0.4307 -> 0.5173 at `1.95` ), which is
recorded and not read: like the read-out levers, the schedule moves the operating point, not the
standing precision. So the audited clip regime is **not** the missing occupancy separation at this
budget; V-086 is closed, no prior verdict moves, and the trunk-width fork stays the user's decision.

## The optimiser regime was never audited: all 40,796 committed steps are clipped (2026-11-11)

A census instrument over every `*.jsonl` curve under `docs/evidence`
([M3_OPTIMISER_REGIME.md](M3_OPTIMISER_REGIME.md), `scripts/stability_regime.py`) shows the published
trainer never once escaped its gradient clip: **106 files / 90 runs / 40,796 steps, 100.00% above the
1.0 clip** (median 23.89, minimum **1.657**, maximum 1,785.62), **90 of 90 runs fully clipped**, **0
files with an unclipped step**. There is no learning-rate schedule anywhere in the tree and all three
clip sites are `clip_grad_norm_(..., 1.)` immediately before `optimizer.step()`. Since the clip
*rescales*, every update in the record is `lr x unit direction` at a constant `lr 1e-3` - the regime
V-085 diagnosed on the pilot, where the gate peaked at 1,000 steps (0.8044), drifted, and recovered
under cosine decay (0.8337 at step 1,150, 0.5983-cell geometry against 0.933).

**This overturns no verdict.** Every lever compared two arms at the same `lr`, clip, population, seed,
catalogue and step count, so a shared regime does not bias a pairwise attribution; it does mean the
campaign's *absolute* claims were measured under one schedule and no other, and it does **not** predict
the fit gate (the pilot's decayed arm still produced 0 exact nonempty boards).

Ranked next: (1) a declared **schedule lever** on the published architecture - 256 matched steps from
`9ae1fef9f649...`, cosine `1e-3 -> 1e-4`, **zero added parameters**, 1,152 MiB / 1,800 s, read by the
standing fit gate (3 of 28), the +0.05 attribution floor and placement non-regression; (2) the
trunk-width fork, which remains the user's decision and is priced; (3) corpus and real-image admission,
parked. Order matters: (1) is one bounded job pair at the existing envelope and retires the last
untested non-structural suspect before any new architecture identity is created.

## Wider-trunk fork priced with measured steps, not estimates (2026-11-10)

The open fork from the capacity branch now has numbers ([M3_TRUNK_WIDTH_PRICING.md](M3_TRUNK_WIDTH_PRICING.md),
pricing only, `docs/evidence/m3-trunk-width-pricing/pricing.json` with its job receipt). Measured through the bounded runner at the real
512-pixel streaming step: the published trunk costs **1.3425 s** of model time per
step (8,018,505 parameters, 271 tensors), wide A
`[64, 128, 224, 256]`/192 costs **1.7082 s**
(8,844,777 parameters, ratio 1.272) and wide B `[96, 192, 256, 256]`/240
costs **2.5867 s** (12,870,089, ratio 1.927).

Scaled to the trainer's measured 4.0867 s whole step at capacity
64, that is **440 / 346 / 228**
steps inside the 1,800 s wall. Against the 256-step baseline and 384-step budget lever, wide A
is comparable at the baseline (90% of the lever) and wide B is not
(89% of the baseline). **published, wide A fits the settled 1,152 MiB profile while wide B does not** in isolated per-trunk runs
at the settled ceiling (peaks and receipts in the brief); the combined run had died at 1,229,508,608 B, 21,549,056 B
over, and only the wider trunks passed under a declared 2560 MiB profile (peak 1,442,033,664 B).

**The decision is the user's**: authorize a new architecture identity with a width knob, a fresh baseline and its
own resource profile (contract spelled out in the brief), or adopt the frozen-architecture limit and end the
occupancy campaign with the gate FALSE. Nothing was changed in the published model, checkpoint or lever table to
produce these numbers.
## Capacity branch blocked: 64 is a slot bound and the trunk has no width knob (2026-11-10)

The step the fit-budget record ranked is not runnable and was not run
([M3_OCCUPANCY_TRUNK_WIDTH.md](M3_OCCUPANCY_TRUNK_WIDTH.md), committed before any launch; no receipt exists).
The reason is code-level, not budgetary. `component_capacity` is the number of component slots the decoder may
emit: the declared configuration at capacity **32** and at capacity **64** both build
8,018,505 parameters over 271 tensors, so capacity is not a width and "wider
than 64" has no smaller sibling to extrapolate from. `ModelConfig` refuses anything above 64
(`component capacity must be0 (M2) or1..64 (M3)`), the driver refuses it first with the renderer's ten-slot
reason, and the encoder widths `(48, 96, 192, 256)` with attention dimension `192` are fixed in the config
with no training flag.

A wider trunk would change the shape of every trunk tensor, so the published checkpoint `9ae1fef9f649...`
cannot be continued into it and the matched-start contract cannot be satisfied by declaring added tensors. That
makes it a new architecture identity with a fresh baseline, not a continuation.

**Outcome.** No arm, sweep, probe or curation ran for this instrument; no occupancy, placement or gate number is
claimed, and the fit gate remains FALSE at 0 of 28 exact nonempty boards. The rule now reports the block and the
requirement instead of naming an impossible arm (V-082). The campaign's next step is a user decision: authorize
a declared trunk-width lever (new identity, fresh baseline, per-step cost measured before the budget is fixed,
explicitly not comparable to the capacity-64 control through the existing table), or accept the standing limit
that under the frozen architecture every ranked suspect is spent or inexpressible.

**Receipts.** None. This pass wrote no measurement, and the two runner limits that bind any future architecture
arm remain as settled: `run_bounded.py` at 3,600 s and the training driver at 1,800 s (V-080/V-081).
## M2 fit gate measured at 1.5x budget: it stands, and the budget is not the limiter (2026-11-10)

The step the per-board refinement pass ranked is spent and settled
([M3_OCCUPANCY_FIT_BUDGET.md](M3_OCCUPANCY_FIT_BUDGET.md), predeclared at `875543d`,
`docs/evidence/m3-occupancy-fit-budget-r1/scope.json`, curation `runs/m3-occfit-curation-r1-job.json`).
The declared change was **training budget only**: the lever arm continues the published checkpoint for
**384 steps** (1.5x the settled 256) and the baseline arm for **256**, with no new parameter, objective,
label, read-out or stage policy in either. Both arms ran from `9ae1fef9f649...` under one catalogue (`f0c833eb9221...`),
lever at 1,033,187,328 B / 1522.4 s and baseline at
1,060,024,320 B / 1046.2 s.

**Activation condition, answered.** The budget **engaged**: **0.5481** of known deciding cells (3946/7200) changed decision between the arms, against the declared floor **0.05**. The relative L2 weight displacement between the endpoints
is **0.0724** over
266 of 271 tensors.

**Fit gate.** The fit gate **STANDS (FALSE)**: **0 of 28** exact nonempty training boards at threshold `0.0`, below the declared threshold **3**. Mechanism `per-cell-error`; the baseline reports
0 of 28. Under the predeclared rule the boolean ends exactly here:
**FALSE**, on the measured count and nowhere else.

**Verdict: `not-cleared`, failing per_cell_f1, ranking, placement, fit_gate.** Per-cell F1 **0.2808** against
**0.4146**, AUC **0.8998** against **0.8255**, best swept F1
**0.5459** at `4.35` against **0.4237** at
`-0.3`, attribution **+0.1221** with twelve-cluster interval
**-0.2589 to -0.0326**, placement
1.9812 -> 1.8419 cells
(+7.0% improvement), spread 4.6043 ->
2.6965.

**Campaign reconciliation on the single control.**

| lever | attribution | placement | activation | verdict |
|---|---|---|---|---|
| geometry stage | +0.0781 | 19.9% cost | n/a | not-cleared |
| box origin | +0.0102 | 11.0% cost | n/a | not-cleared |
| box extent | +0.0318 | 67.5% cost | n/a | not-cleared |
| geometry capacity | -0.0241 | 28.2% cost | n/a | not-cleared |
| per-stage residual | -0.0115 | 19.2% cost | n/a | not-cleared |
| lattice-free read-out | +0.0147 | 22.3% gain | gate 0.0742 | not-cleared |
| per-board refinement | +0.0539 | 19.2% worse | 0.0014 cells (off) | not-cleared |
| fit budget (1.5x) | +0.1221 | +7.0% | 0.5481 flips | not-cleared |

**Ranked next.** The non-circular rule takes its **`capacity_limit`** branch and names **a wider shared trunk than capacity 64**.
**Ranked next: predeclare a wider shared trunk than capacity 64**, read with the same conditions and activation discipline; the other
branch stays behind it, and the seven spent levers cannot be returned by the rule.

**Receipts.** Profiles `runs/m3-occfit-{lever,baseline}-r1-profile-r1`, studies
`runs/m3-occfit-{lever,baseline}-r1` (384 and 256 steps), sweeps `...-sweep-r1` with their own profiles
(54 forwards, 640 MiB), probes `...-probe-r1` (36 forwards, 640 MiB / 300 s), curation
`runs/m3-occfit-curation-r1-job.json`. Each probe reproduces its own sweep's per-cell F1 and TP/FP/FN counts,
the baseline reproduces the curated control to 1e-6, and the matched start is proven by the bitwise
identity of the two saved initial state dictionaries plus the step-0 probes. No prior evidence or frozen
manifest was touched.
## Per-board refinement capacity lever refused: the channel never engaged and the arm regressed (2026-11-10)

The lever the read-out pass ranked is spent and settled
([M3_OCCUPANCY_BOARD_REFINEMENT.md](M3_OCCUPANCY_BOARD_REFINEMENT.md), predeclared at `72f95bc`,
`docs/evidence/m3-occupancy-board-refinement-r1/scope.json`, curation
`runs/m3-occbref-curation-r1-job.json`). The declared change was **capacity per board**: the published
decode is one shared affine function, and two earlier levers corrected it globally (origin) or per stage
(residual). This one added a two-layer projection of the board token producing one four-vector per board,
applied through the published `0.25*tanh` cell-space map - origin offset in cells plus log-scale per axis
- identically at every stage, for **12,612 declared parameters** in four zero-initialized tensors. Both
arms ran 256 steps from `9ae1fef9f649...` under one catalogue (`f0c833eb9221...`) at 1,152 MiB / 1,800 s (peaks
1,070,809,088 B / 1,027,080,192 B,
1059.9 s / 1019.5 s).

**Activation condition, answered - and it fails.** The probe measured the applied correction at
**0.001428** cells against the declared 0.10 floor and
**0.006048** relative axis scale against the declared 0.02 floor (maxima
0.002142 / 0.009243). The head started at zero and never
left it: the correction is about a thousandth of a cell, so the lever was measured switched off and **no
claim about per-board refinement capacity follows**. This is the activation condition working as declared,
not an excuse: the number is published, and it is the reason the verdict cannot be spun as a channel test.

**Verdict: `not-cleared`, failing per_cell_f1, ranking, placement, activation.** Per-cell F1 **0.4232** (570/1457/97)
against **0.4146** (351/675/316);
AUC **0.8596** against **0.8255**; best swept F1 **0.4776** at
`0.9` against **0.4237** at `-0.3`, attribution
**+0.0539** with twelve-cluster interval **-0.1341 to +0.0740**
(median -0.0247); placement **19.2% worse**
(1.9812 -> 2.3614); spread
4.6043 -> 5.4599 cells, column extent ratio
1.2257 -> 1.4471. Since the correction itself carried
0.0014 cells, the regression is a trajectory effect of an inert 12,612-parameter addition at this budget,
which is itself a finding about how tight the 256-step continuation is.

**Campaign reconciliation on the single control.**

| lever | attribution | placement | activation | verdict |
|---|---|---|---|---|
| geometry stage | +0.0781 | 19.9% cost | n/a | not-cleared |
| box origin | +0.0102 | 11.0% cost | n/a | not-cleared |
| box extent | +0.0318 | 67.5% cost | n/a | not-cleared |
| geometry capacity | -0.0241 | 28.2% cost | n/a | not-cleared |
| per-stage residual | -0.0115 | 19.2% cost | n/a | not-cleared |
| lattice-free read-out | +0.0147 | 22.3% gain | gate 0.0742 | not-cleared |
| per-board refinement | +0.0539 | -19.2% (worse) | 0.0014 cells (off) | not-cleared |

**Ranked next.** The non-circular rule takes its **fit_gate** branch and names **the M2 fit gate at capacity 64**.
**Ranked next: predeclare the M2 fit gate at capacity 64**, read with the same four conditions and the same activation
discipline; `scaling` stays behind it, and the six spent levers cannot be returned by the rule.

**Receipts and defects.** Profiles `runs/m3-occbref-{lever,baseline}-r1-profile-r1`, studies
`runs/m3-occbref-{lever,baseline}-r1`, sweeps `...-sweep-r1` with their own profiles (54 forwards,
640 MiB), probes `...-probe-r1` (36 forwards, 640 MiB / 300 s) carrying the activation diagnostic,
curation `runs/m3-occbref-curation-r1-job.json`. Each probe reproduces its own sweep's per-cell F1 and
TP/FP/FN counts, the baseline reproduces the curated control to 1e-6, and the matched start is proven by
the parameter-addition branch (four declared zero tensors, shared tensors bitwise identical, step-0
probes equal). No launch was refused or killed. New defect V-079 is logged; no prior evidence or frozen
manifest was touched.
## Lattice-agnostic read-out lever refused: the gate barely opened and bought placement, not occupancy (2026-11-10)

The lever the residual pass ranked is spent and settled
([M3_OCCUPANCY_LATTICE_FREE.md](M3_OCCUPANCY_LATTICE_FREE.md), predeclared at `7b3ca9a`,
`docs/evidence/m3-occupancy-lattice-free-readout-r1/scope.json`, curation
`runs/m3-occlatfree-curation-r1-job.json`). The declared change was the **read-out**, not supervision,
capacity or geometry: the published head reads each stage's decoder latent at that stage's own predicted
lattice, so the lever added a channel built only from the encoder's image-level context (valid-token mean)
plus fixed row and column embeddings - identical at every stage - and blended it with the published logits
through one scalar gate, `logits = (1-gate)*published + gate*free`, `gate = tanh(raw)`. The two-layer head
and the gate are zero-initialized, for **4,226 declared parameters** in five tensors, so step zero is
bitwise the published function and `|gate| -> 1` replaces the lattice-sampled path and its detail tap. Both
arms ran 256 steps from `9ae1fef9f649...` under one catalogue (`93742d5fb23f...`) at 1,152 MiB / 1,800 s (peaks
1,082,392,576 B / 1,025,159,168 B, 1,057.0 s / 1,025.0 s).

**Verdict: `not-cleared`, failing all four declared conditions.** Per-cell F1 **0.4284**
(371/694/296)
against **0.4146** (351/675/316);
AUC **0.8401** against **0.8255**; best swept F1 **0.4384** at
`1.65` against **0.4237** at `-0.3`, attribution
**+0.0147** with twelve-cluster paired-bootstrap median **+0.0150** and
interval **-0.0836 to +0.1284**. Placement improved **22.3%** (1.9812 ->
1.5393), short of the **25%** floor.

**The declared channel was barely activated, and that is the finding.** The trained gate reached raw
0.0744, `|tanh| = 0.0742` - above the 0.05 shut falsifier, but only
~7% of the logits came from the lattice-agnostic path. What the arm did buy is the
first movement in this campaign's other direction: placement improved rather than being spent, both signed
offsets moved toward zero (+1.3830 -> +0.4926,
-2.1914 -> -1.0181), and the per-board error spread
fell 4.6043 -> **3.3544** cells. AUC
0.8401 is the highest of the six levers, so the ranking signal moved without the decision
boundary. The declared limit applies: a null result at a near-zero gate cannot prove the read-out is not
the bottleneck.

**Campaign reconciliation on the single control.**

| lever | attribution | placement cost | verdict |
|---|---|---|---|
| geometry stage | +0.0781 | 19.9% | not-cleared |
| box origin | +0.0102 | 11.0% | not-cleared |
| box extent | +0.0318 | 67.5% | not-cleared |
| geometry capacity | -0.0241 | 28.2% | not-cleared |
| per-stage residual | -0.0115 | 19.2% | not-cleared |
| lattice-free read-out | +0.0147 | 22.3% gain | not-cleared |

**Ranked next.** The non-circular rule takes its **first** branch - the lever's own spread
**3.3544 >= 1.5** cells - and names **per-board lattice refinement
capacity**. **Ranked next: predeclare a per-board lattice refinement capacity lever**, read with the same
four conditions; `scaling` stays behind it, and `translation`, `scale`, `geometry_capacity`,
`placement_capacity` and `lattice_residual` cannot be returned by the rule.

**Receipts and defects.** Profiles `runs/m3-occlatfree-{lever,baseline}-r1-profile-r1`, studies
`runs/m3-occlatfree-{lever,baseline}-r1`, sweeps `...-sweep-r1` with their own profiles (54 forwards,
640 MiB), probes `...-probe-r1` (36 forwards, 640 MiB / 300 s), curation
`runs/m3-occlatfree-curation-r1-job.json`. Each probe reproduces its own sweep's per-cell F1 and TP/FP/FN
counts, the baseline reproduces the curated control to 1e-6, and the matched start is proven by the
parameter-addition branch (five declared zero tensors, shared tensors bitwise identical, step-0 probes
equal). No launch was refused or killed, so `runs/attempts/` is unchanged. New defects V-077 and V-078 are
logged; no prior evidence or frozen manifest was touched.
## Per-stage lattice-residual lever refused: more box capacity moved the lattice and lost both scores (2026-11-10)

The lever the capacity pass ranked is spent and settled
([M3_OCCUPANCY_LATTICE_RESIDUAL.md](M3_OCCUPANCY_LATTICE_RESIDUAL.md), predeclared at `473440c`,
`docs/evidence/m3-occupancy-lattice-residual-r1/scope.json`, curation
`runs/m3-occlatres-curation-r1-job.json`). The declared change was **capacity**, not supervision: the
published model decodes one shared box through a single sigmoid that ties origin to extent and refines
it before that sigmoid for only two of six stages, so each stage gained a zero-initialized 4x4
projection applied to its own decoded lattice - origin offset in cells plus a log-scale per axis -
for **120 declared parameters**, supervised by the existing geometry term. Both arms ran 256 steps
from `9ae1fef9f649...` under one catalogue (`242e8c767c9f...`) at 1,152 MiB / 1,800 s (peaks
1,059,573,760 B / 1,056,604,160 B, 826.1 s / 1,017.6 s).

**Verdict: `not-cleared`, failing all four declared conditions.** Per-cell F1 **0.3520** (661/2,428/6)
against **0.4146** (351/675/316); AUC **0.8209** against **0.8255**; best swept F1 **0.4122** at `2.25`
against **0.4237** at `-0.30`, attribution **-0.0115** with twelve-cluster paired-bootstrap median
**-0.0954** and interval **-0.2795 to +0.0300**. Placement failed in the wrong direction: mean error
1.9812 -> **2.3615** (**-19.2%** improvement), column offset +1.3830 -> **+1.5856**, row -2.1914 ->
**-2.9092**.

**The capacity acted, and the predeclared placement-regressed falsifier holds.** The column extent
ratio **shrank** 1.2257 -> **0.9084**, rows 1.1804 -> 1.0781, spread 4.6043 -> **4.8346**, shifted
share 0.5375 -> **0.7093**, mapped share 0.6507 -> 0.7243. The occupancy head answered with an almost
all-positive regime: recall **0.991** at `0.0` with 2,428 false positives, which is why the per-cell
and ranking gates fail while AUC stays near baseline. Per-stage box capacity is not the binding
constraint; it is a channel that moves the lattice without the score following.

**Campaign reconciliation on the single control.**

| lever | attribution | placement cost | verdict |
|---|---|---|---|
| geometry stage | +0.0781 | 19.9% | not-cleared |
| box origin | +0.0102 | 11.0% | not-cleared |
| box extent | +0.0318 | 67.5% | not-cleared |
| geometry capacity | -0.0241 | 28.2% | not-cleared |
| per-stage residual | -0.0115 | 19.2% | not-cleared |

Two supervision-side gains were bought with placement; the two capacity-side levers lost both.

**Ranked next.** The non-circular rule takes its first branch - improvement **-0.192 < 0.05** - and
names **a lattice-agnostic occupancy read-out**. **Ranked next: predeclare a lattice-agnostic
read-out lever**, read with the same four conditions; `dispersion` (spread 4.8346, below the 1.5-cell
floor) and `scaling` stay behind it, and `translation`, `scale`, `geometry_capacity` and
`placement_capacity` cannot be returned by the rule.

**Receipts and defects.** Profiles `runs/m3-occlatres-{lever,baseline}-r1-profile-r1`, studies
`runs/m3-occlatres-{lever,baseline}-r1`, sweeps `...-sweep-r1` with their own profiles (54 forwards,
640 MiB), probes `...-probe-r1` (36 forwards, 640 MiB / 300 s), curation
`runs/m3-occlatres-curation-r1-job.json`. Each probe reproduces its own sweep's per-cell F1 and
TP/FP/FN counts, the baseline reproduces the curated control to 1e-6, and the matched start is proven
by the parameter-addition branch (twelve declared zero tensors, shared tensors bitwise identical). The
baseline study the host killed at step 247 of 256 is preserved as
`runs/attempts/m3-occlatres-baseline-r1.attempt1` with a note and re-recorded at the same declared
name. New defects V-075 and V-076 are logged; no prior evidence or frozen manifest was touched.

## Geometry-capacity lever refused: boundary supervision lost occupancy and placement together (2026-11-10)

The lever the extent pass ranked is spent and settled ([M3_OCCUPANCY_CAPACITY.md](M3_OCCUPANCY_CAPACITY.md),
predeclared at `8211a5f`, `docs/evidence/m3-occupancy-geometry-capacity-r1/scope.json`, curation
`runs/m3-occgeocap-curation-r3-job.json`). The declared change was one objective term and no new
parameter: eight boundary support anchors - four corners and four edge midpoints in cell units -
mapped through the predicted and target lattices, differenced, normalized by the target spacing and
averaged as smooth-L1 on the **final** decoder lattice at `support_anchor_weight 1.0`, the geometry
term's own weight. The published geometry mean gives those anchors 4/204 = 2% of one term; this made
them a full-weight term of their own, so the same supervision was spent where the box is read. Both
arms ran 256 steps from `9ae1fef9f649...` under one catalogue (`7675899b8186...`) at 1,152 MiB / 1,800 s
and stayed inside it (peaks 1,033,744,384 B / 1,055,576,064 B, 946.6 s / 701.7 s).

**Verdict: `not-cleared`, failing all four declared conditions.** Per-cell F1 **0.2017** (106/278/561)
against **0.4146** (351/675/316); AUC **0.8116** against **0.8255**; best swept F1 **0.3996** at `-1.95`
against **0.4237** at `-0.30`, attribution **-0.0241** with twelve-cluster paired-bootstrap median
**-0.2007** and interval **-0.2795 to -0.0696**. The placement condition fails in the wrong direction:
mean error 1.9812 -> **2.5401** (**-28.2%** improvement), signed column +1.3830 -> **+2.3077**, row
-2.1914 -> **-2.9483**.

**The anchors did act, and the predeclared regress falsifier holds.** Column extent ratio 1.2257 ->
**1.3338** (+8.8%, past the 5% line), rows 1.1804 -> 1.0975, spread 4.6043 -> **5.6679**, mapped share
0.6507 -> 0.6175. The box moved further right and down rather than onto its target boundary, so this
term may not be described as boundary pinning and no occupancy gain can be credited to it.

**Reconciliation.** Geometry **+0.0781** at 19.9% placement cost, origin **+0.0102** at 11.0%, extent
**+0.0318** at 67.5%, capacity **-0.0241** at 28.2%. The first three bought occupancy at a placement
cost; capacity is the first to lose both, so boundary supervision does not repay any earlier gain and
is not a cheaper route to one.

**Ranked next.** The non-circular rule takes its first branch - placement improvement **-0.282 < 0.05**
- and names **per-stage lattice residual capacity**. **Ranked next: predeclare a per-stage lattice
residual lever**, read with the same four conditions plus this pass's placement condition;
`read_out_inheritance`, dispersion (spread 5.6679) and scaling stay behind it, and `translation`,
`scale` and `geometry_capacity` cannot be returned by the rule.

**Receipts and defects.** Profiles `runs/m3-occgeocap-lever-r1-profile-r1` and
`...-baseline-r1-profile-r2` (the r1 baseline profile was killed by a runner restart and is preserved
under `runs/attempts/` with a note), studies `runs/m3-occgeocap-{lever,baseline}-r1`, sweeps
`...-sweep-r1` with their own profiles (54 forwards, 640 MiB), probes `...-probe-r1` (36 forwards, 640
MiB / 300 s), curation `runs/m3-occgeocap-curation-r3-job.json` (r1 and r2 refused on curator path
wiring and are preserved with notes). Each probe reproduces its own sweep's per-cell F1 and TP/FP/FN
counts, and the baseline reproduces the curated control to 1e-6. New defects V-073 (patch generator
dropped `extent_error`'s return) and V-074 (curator receipt-path wiring, two instances) are logged.
Source checkpoint, catalogues and weights identities are in the scope file; no prior evidence or
frozen manifest was touched.

## Box-extent lever refused: the extent term inflated the box and cost both scores (2026-11-09)

The lever the origin pass ranked is spent and settled
([M3_OCCUPANCY_EXTENT.md](M3_OCCUPANCY_EXTENT.md), predeclared at `e5e9668`,
`docs/evidence/m3-occupancy-extent-r1/scope.json`, curation
`runs/m3-occextent-curation-r2-job.json`). The declared change was one objective term and no new
parameter: the mean absolute log ratio of the predicted to the target basis norms on the **final**
decoder lattice, at `box_extent_weight 1.0`, so the box's scale is supervised directly while the origin
and the occupancy read-out stay untouched. Both arms ran 256 steps from `9ae1fef9f649...` under one
catalogue, 1,152 MiB / 1,800 s, and stayed inside it (peaks 1,066,991,616 B / 1,067,384,832 B,
814.2 s / 1,120.7 s).

**Verdict: `not-cleared`, failing all four declared conditions.** Per-cell F1 **0.3315**
(361/1,150/306) against **0.4146** (351/675/316); AUC **0.7526** against **0.8255**; best swept F1
**0.4556** at `2.10` against **0.4237** at `-0.30`, so attribution is **+0.0318** - below the +0.05
floor and an operating-point move, since the twelve-cluster paired bootstrap has median **-0.0848**
and interval **-0.1345 to -0.0313**. The declared extent condition fails **in the opposite direction**:
columns 1.2257 -> **1.5255** (log magnitude 0.2035 -> 0.4225, reduction **-107.5%**) while rows only
fell 1.1804 -> 1.1466 (**+17.5%**).

**Placement did not follow, so both predeclared falsifiers hold.** Mean placement error rose **67.5%**
(1.9812 -> 3.3182), signed offsets +1.3830 -> **+3.6865** / -2.1914 -> **-2.9483**, spread 4.6043 ->
**5.9710**, mapped share 0.6507 -> 0.5145. Neither "extent is the binding placement constraint" nor
"the placement residual was an extent residual" can be credited.

**Reconciliation.** Geometry **+0.0781** at 19.9% placement cost; origin **+0.0102** at 11.0%; extent
**+0.0318** at **67.5%**. The costs add and the third is the largest, so no earlier gain was a hidden
extent correction and extent supervision is not a cheaper route to either. The origin pass had already
left the box wider than the baseline (1.3242 / 1.1709), so the box had been drifting outward before
this lever.

**Ranked next.** The non-circular rule takes its first branch - smallest extent reduction **-1.075 <
0.25** - and names **geometric supervision capacity (support anchors)**. **Ranked next: predeclare a
support-anchor capacity lever on the geometry supervision**, read with the same four conditions;
`read_out_inheritance`, dispersion (spread 5.9710) and scaling stay behind it.

**Receipts and defects.** Profiles `runs/m3-occextent-{lever,baseline}-r1-profile-r1`, studies
`runs/m3-occextent-{lever,baseline}-r1`, sweeps `...-sweep-r1` with their profiles (54 forwards, 640
MiB; the baseline profile was killed by a caller timeout and re-recorded as `-r2`, and each sweep now
binds its profile through its own `admission.run` record), placement probes `...-probe-r1` (36
forwards, 640 MiB / 300 s), curation `runs/m3-occextent-curation-r2-job.json`. Each probe reproduces
its own sweep's per-cell F1 and TP/FP/FN counts at 1e-6 before the verdict was written. **V-072**
records the first curation refusal: the shared matched-start guard compared the step-0 probe record
including its loss, which an objective lever changes by construction; it now strips the objective's own
fields recursively. The refused curation and the killed profile launch are preserved under
`runs/attempts/` with notes and are cited nowhere.

## Box-origin (translation) lever refused: the origin did not move, and the residual is extent (2026-11-09)

The lever the geometry pass ranked is spent and settled
([M3_OCCUPANCY_TRANSLATION.md](M3_OCCUPANCY_TRANSLATION.md), predeclared at `24ae9b9`,
`docs/evidence/m3-occupancy-translation-r1/scope.json`, curation
`runs/m3-occorigin-curation-r2-job.json`). The declared change was one structural addition: a
zero-initialized `Linear(5,4)` over the coarse feature map's spatial moments (energy centroid x/y, spread
x/y, log energy) added to `raw_box`, so the lattice origin can depend on where the board is in the image.
Both arms ran 256 steps from `9ae1fef9f649...` under one catalogue, 1,152 MiB / 1,800 s, and stayed inside
it (peaks 1,085,526,016 B / 1,083,441,152 B, 380.8 s / 380.7 s).

**Verdict: `not-cleared`, failing all four declared conditions.** Per-cell F1 **0.4165** (622/1,698/45)
against **0.4146** (351/675/316); AUC **0.8446** against **0.8255**; best swept F1 **0.4340** at `0.45`
against **0.4237** at `-0.30`, so attribution is **+0.0102** against the +0.05 floor, with the
twelve-cluster paired bootstrap at median **-0.0378** and interval **-0.1700 to +0.0872**. The declared
signed condition fails on both axes: columns **+1.3998** against **+1.3830** (reduction **-0.0122**) and
rows **-2.6181** against **-2.1914** (**-0.1947**) - the origin moved outward, not toward zero. Mean
placement error rose **11.0%** (1.9812 -> 2.1988), mapped share fell 0.6507 -> 0.6334, spread rose
4.6043 -> 4.7870. The occupancy movement is an operating-point shift: +271 true positives bought with
+1,023 false positives.

**Reconciliation.** The geometry lever's **+0.0781** occupancy gain came with **19.9%** worse placement;
this lever buys **+0.0102** (an eighth of it) and adds another **11.0%** placement cost, so the two costs
add rather than compensate, and the geometry gain was not a hidden origin correction.

**Ranked next.** The declared non-circular rule takes its first branch - smallest signed reduction
**-0.19 < 0.25** - and names **box extent**, which the same receipts measure at ratios **1.3242 / 1.1709**
(both past the 0.10 log-ratio floor). **Ranked next: predeclare a box-extent supervision lever on the
lattice box**, read with the same four conditions; dispersion (spread 4.7870) and read-out inheritance
stay behind it.

**Receipts and defects.** Profiles `runs/m3-occorigin-{lever,baseline}-r1-profile-r1`, studies
`runs/m3-occorigin-{lever,baseline}-r1`, sweeps `...-sweep-r1` with their profiles (54 forwards, 640 MiB),
placement probes `...-probe-r1` (36 forwards, 640 MiB / 300 s), curation
`runs/m3-occorigin-curation-r2-job.json`. Each probe reproduces its own sweep's per-cell F1 and TP/FP/FN
counts over the full deciding population before the verdict was written. **V-071** records the first
curation refusal: the inherited matched-start guard asserted one identical `initial-weights.pt` digest,
which a lever that *adds* parameters cannot satisfy; it now asserts shared-tensor identity, a zero
projection and identical step-0 probe loss. The refused launch is preserved under `runs/attempts/` and
cited nowhere.

## Geometry-stage lever refused: occupancy moved, placement did not, and the residual is translation (2026-11-09)

The step the registration pass ranked is spent and settled
([M3_OCCUPANCY_GEOMETRY.md](M3_OCCUPANCY_GEOMETRY.md), predeclared at `35ae175`,
`docs/evidence/m3-occupancy-geometry-r1/scope.json`, curation
`runs/m3-occgeom-curation-r2-job.json`). The declared change was a single objective policy: the
geometry term is the **final** decoder stage's lattice error (`geometry_final_only = 1`) instead of
the mean over all six stages (`0`), because only the final lattice feeds the occupancy read-out. Both
arms ran 256 steps from `9ae1fef9f649...` under one catalogue and one identical initial tensor file,
1,152 MiB / 1,800 s, and stayed inside it (peaks 1,087,176,704 B / 1,058,406,400 B, 396.3 s / 399.7 s).

**Verdict: `not-cleared`, failing `per_cell_f1`, `ranking` and `placement`.** The lever does improve
the occupancy decision - best swept F1 **0.5018** at `0.45` against the baseline's **0.4237** at
`-0.30`, F1@0 **0.4791** (459/790/208) against **0.4146**, AUC **0.8316** against **0.8255**, so
attribution is **+0.0781** (cluster interval -0.0152 to +0.1243) and that condition is met - but
per-cell F1 is still far below 0.70, AUC below 0.90, and **placement error rose 19.9%**, from 1.9812
to 2.3757 cells (shift share 0.5375 -> 0.5816), so the declared 25% placement condition fails. The
occupancy gain without any placement gain is the informative part: the read-out absorbed the offset
rather than the lattice correcting it.

**The refusal names the next lever by the predeclared rule.** Signed offsets are **+2.3243 cells
right** and **-2.3848 cells down** (baseline +1.3830 / -2.1914), both past the 1.0-cell translation
floor, with extent ratios 1.2231 / 1.1386 and a 5.4583-cell per-board spread. Rule: `translation`.
**Ranked next: predeclare a box-origin (translation) supervision lever on the lattice box**, read
with the same four conditions, with the signed offsets as the declared quantities to move. The
remaining branches are already ranked behind it: extent (ratios 1.22 / 1.14), dispersion (spread
5.46 cells), then read-out inheritance.

**Receipts and defects.** Profiles `runs/m3-occgeom-{lever,baseline}-r1-profile-r1`, studies
`runs/m3-occgeom-{lever,baseline}-r1`, sweeps `...-sweep-r1` (54 forwards, 640 MiB), placement probes
`...-probe-r1` (36 forwards, 640 MiB, peak 565,149,696 B / 599,879,680 B), curation
`runs/m3-occgeom-curation-r2-job.json`. Each probe reproduces its own sweep's published per-cell F1
and operating counts at 1e-6 before the verdict is written (the V-069 join). V-070 records the
curation guard inherited from the read-out passes that asserted the arms must start from *different*
initial tensors; it refuses this pair, which starts from one identical file. The refused curation is
preserved under `runs/attempts/` with a note and is not evidence. Full suite: 690 tests green.

## 4x budget refused, registration implicated: the occupancy residual is cell placement (2026-11-09)

The step the pixel ceiling ranked is spent and settled
([M3_OCCUPANCY_SCALED.md](M3_OCCUPANCY_SCALED.md), predeclared at `1ce41e6`,
`docs/evidence/m3-occupancy-scaled-r1/scope.json`, registration fallback in
`docs/evidence/m3-occupancy-registration-r1/scope.json`). The ranked hypothesis was that the 2x2
stencil was information-rich and starved of steps, so both arms ran **256** steps (4x the 64-step
stencil pass) from the published `9ae1fef9f649...` checkpoint under a single captured catalogue, with
only the stencil form differing, at 1,152 MiB / 1,800 s.

**Verdict: `not-cleared` - scaling is refused at the declared envelope.** The lever's best swept F1
is **0.3925** (`0.60`) against the matched control's **0.4237** (`-0.30`), AUC **0.8006** against
**0.8255**, per-cell F1@0 **0.3791** (320/701/347) against **0.4146**, so **attribution is -0.0312**
(cluster interval -0.1082 to +0.0411) and per-cell F1 and ranking miss as well. The extra budget did
not merely fail to help: it lowered the stencil's own 64-step ordering (AUC 0.8804 -> 0.8006, best F1
0.4913 -> 0.3925) while the center-sample control held (0.8226 -> 0.8255), i.e. the ordering that
made the stencil the campaign's best arm inverted. Both studies stayed inside the envelope
(peaks 1,065,533,440 B / 1,067,073,536 B, 396.1 s / 395.1 s, 256 work items each).

**Fallback: `registration_implicated`.** The declared conditional instrument replayed the scaled
lever's 36 deciding boards read-only (36 forwards, 640 MiB, 20.5 s, peak 565,141,504 B) and located
each model cell's centre in target cell coordinates. Its as-is reading over the 28 nonempty boards
reproduces the scaled sweep exactly (F1 0.379147, counts 320/701/347 - the curator's cross-receipt
join). On the 2,986 of 5,600 cells both pairings can score, the oracle registration pairing gives F1
**0.5742** / AUC **0.9702** against as-is **0.4390** / **0.9331**, a delta of **+0.1352** (interval
+0.0460 to +0.2833, 400 cluster draws). The geometry error is large, not marginal: mean |offset|
**2.57 cells**, 48.3% of cells on the wrong target cell, 53.3% still on the board. Because the
occupancy head samples at the model's own predicted lattice, its cell indices inherit that placement
error directly.

**Ranked next: predicted-lattice placement.** Predeclare a geometry-side lever on the
registration/lattice objective (the existing `registration` term is the natural place), read it with
the same three gates plus a declared placement-error measurement, and keep the attribution target
this pass measured as the quantity that must move. Budget scaling on this form is closed at 4x; the
read-out and label families remain closed. Not a repair: the oracle pairing is a diagnostic, the
excluded 46.7% of cells cannot be scored, and nothing here is a real-image or generalization claim.

**Receipts and defects.** Profiles `runs/m3-occscale-{lever,control}-r1-profile-r1`, studies
`runs/m3-occscale-{lever,control}-r1`, sweeps `...-sweep-r1` (54 forwards, 640 MiB), curation
`runs/m3-occscale-curation-r1-job.json`, probe `runs/m3-occscale-registration-r2-job.json`,
curation `runs/m3-occscale-registration-curation-r5-job.json`. V-069 records the registration
summariser's population drift (it scored all 36 boards, not the declared 28 nonempty ones) and the
cross-receipt join that now catches it; the superseded probe and three failed curations are
preserved under `runs/attempts/` with a note and are not evidence. Full suite: 651 tests green.

## Pixel ceiling measured: the labels are separable, so the label lever is refused before admission (2026-11-09)

The step the stacking pass ranked now has a measured premise and a verdict
([M3_OCCUPANCY_PIXEL_CEILING.md](M3_OCCUPANCY_PIXEL_CEILING.md), predeclared at `6e40206`,
`docs/evidence/m3-occupancy-pixel-ceiling-r1/scope.json`, curation
`runs/m3-occpix-ceiling-curation-r3-job.json`). The ranked step was an explicit indeterminate /
occluded cell target, justified by a label-overlap signature. Instead of spending a matched budget
on that inference, this pass asked the premise's own question read-only: **can the captured pixels
recover the committed determinate labels at all?**

**Instrument.** Thirty per-cell pixel features (3x3 interior sRGB means plus per-channel interior
standard deviation), an L2 logistic read-out with one ancestry group held out, its own `0.0`
boundary, the standing 81-point grid beside it, and the identical pipeline on within-group shuffled
labels as the control. No model, no forward pass, no training, no label or threshold change.
Governing statistics are pooled count-based F1 and the **mean per-fold AUC**, because a
leave-one-group-out score can be near-constant within a fold and still rank fold frequency pooled
(measured design of this defect: V-066).

**Measured, on the committed 28 nonempty deciding boards** (667 occupied / 4,933 empty cells,
twelve ancestry groups, 36 replayed training images): F1 **0.9780** (667 true / 30 false / 0 missed),
precision 0.9570, recall 1.0000, mean per-fold AUC **0.9976** (every fold 0.9799-1.0000), best swept
F1 0.9859 at `1.05`. The model's own committed numbers on exactly these cells are F1 **0.4019** and
AUC **0.7960**. The shuffled control collapses to F1 **0.0029** and fold-mean AUC **0.4885**, so the
pipeline is measuring label recovery, not board density.

**Verdict: `separable`** - both declared bars cleared (F1 +0.278, AUC +0.098), gaps to the model
+0.5761 operating F1, +0.5763 best F1, +0.2016 on the ranking statistic. **Consequence: the
label-overlap signature is refuted, `indeterminate_target_licensed = false`, and the
indeterminate/occluded cell target is not admitted.** No further label-policy variant is ranked.
This is not an accuracy claim: the ceiling is a linear read-out at oracle registration on the
stream's own synthetic drawings.

**Ranked next: model-side.** Scale the matched budget on the 2x2 stencil configuration - the only
read-out arm that met the +0.05 attribution floor (AUC 0.8804 / best F1 0.4913) - and carry
**registration** as the named alternative suspect, because this ceiling was granted the target
lattice while the model's geometry error is about 3.3 cells. Stacking stays closed, the refused
resolution lever reopens only as its own declared configuration, and the real-image chain stays
parked pending authorization.

**Receipts and defects.** Canonical: profile `runs/m3-occpix-ceiling-r1-profile-r2` (2 work items,
266,375,168 B / 2.0 s) and study `runs/m3-occpix-ceiling-r1-r2` (36 work items, 266,182,656 B /
3.3 s), both inside the declared 640 MiB, plus curation `-r3`. V-066 records the pooled-AUC and
unregularized-score defect the exploratory read exposed; V-067 the refused launch that consumed a
declared name; V-068 the curator's assumed label order. Each refused receipt and its note is
preserved under `runs/attempts/` and none is evidence. The instrument is `scripts/pixel_ceiling.py`
with `scripts/probe_pixel_ceiling.py`, `scripts/record_pixel_ceiling.py` and
`tests/test_pixel_ceiling.py` (20 tests).

## Context+detail stacking measured: not cleared, and the stack orders worse than either half (2026-11-09)

The lever the resolution pass ranked now has its own verdict ([M3_OCCUPANCY_STACKING.md](M3_OCCUPANCY_STACKING.md),
predeclared at `4135a2b`, `docs/evidence/m3-occupancy-stacking-r1/scope.json`, curation
`runs/m3-occstack-curation-r2-job.json`). The declared change was the composition: the context residual
used to overwrite the detail tap, so the two-flag model silently ignored one of them. The pass sums the
two existing zero-initialized residuals - no new parameter, no new knob - and runs that against the
detail tap alone, 64 matched steps from `m3-percell-balance-r1` under one catalogue (`c8fba7ab`), both
inside 1,152 MiB (lever 1,033,072,640 B / 190.9 s, control 1,039,544,320 B / 182.1 s), scored on the
same population (36 images, 28 nonempty boards, 667/4,933 cells, 81 points, 54 forwards per sweep
under 640 MiB, twelve clusters).

**Verdict: NOT CLEARED**, and all three conditions fail: per-cell F1 0.4019 against 0.70; ranking
`uninformative` (AUC **0.7960**, CI 0.7751-0.8168, **0.104 short** of 0.90; best swept F1 0.4097,
**0.290 short** of 0.70); 0/28 exact boards; attribution **-0.0024** against the +0.05 floor.
**Mechanism: `precision_improved = false`** - precision at `0.0` rose 0.2363 -> 0.2521 (+0.0158,
cluster 0.0038-0.0228) but below the declared 0.05 floor.

**The substantive result is that stacking is worse than either half.** The stack is **-0.0844 AUC**
and **-0.0816** best F1 below the stencil arm (0.8804 / 0.4913) and **-0.0284 / -0.0103** below the
context arm alone (0.8244 / 0.4200); those are unmatched cross-pass references and do not enter the
verdict. Two residuals summed into one occupancy projection compete rather than complement. The
control reproduced the stencil pass's control bit-for-bit under a different catalogue (AUC
`0.8226433610683003`, counts 661/2136/6, best F1 `0.41203400915631133` at `0.90`), so the negative is
not drift.

**Ranked next: leave the read-out and target the labels** - an explicit indeterminate/occluded cell
target that distinguishes cells the pixels cannot decide from cells that are empty, scored on
determinate cells. Justification: every read-out lever moved ordering (0.7427 -> 0.8244 -> 0.8226 ->
0.8804) while the swept-F1 deficit stayed 0.21-0.29, the standing-point operating point is degenerate
(precision 0.25 at recall 0.99) and 0/28 boards are exact at any threshold, which is a label-overlap
signature rather than a capacity one. Stacking is closed; the refused resolution lever reopens only
as a separately declared cheaper configuration; the real-image chain remains parked pending
authorization.

The pass also found and closed one lever-level defect (V-063: the context residual overwrote the
detail tap) before any measurement, and two harness defects in my own launch and curation plumbing
(V-064, V-065), each preserved under `runs/attempts/`. No ceiling, gate, population or floor moved.

## Encoder resolution refused at the declared envelope: the stride-1 source costs 188 MiB, not 48 (2026-11-09)

The step the stencil pass ranked was predeclared at `eee9362`
([M3_OCCUPANCY_RESOLUTION.md](M3_OCCUPANCY_RESOLUTION.md)) as a single declared change: a stride-1
convolution of the captured image whose cell-center sample joins the same zero-initialized detail
projection, against the same single-sample control, 64 matched steps from `m3-percell-balance-r1`,
one catalogue, 1,152 MiB, 54 sweep forwards and the standing 0.70 / 0.90 / 0.05 gates and floor.

**Outcome: REFUSED, not scored.** The 2-step admission profile - run before any study, as V-057
requires - breached the declared ceiling on its first launch: lever peak **1,231,880,192 B** against
1,207,959,552 B (exit 3221225477, 46.0 s), while the same-session stride-2 control peaked at
1,034,313,728 B. The branch therefore costs **197,566,464 B (188.4 MiB)**, 3.93x the 50,331,648 B
one-map estimate the predeclaration carried, and 23,920,640 B (22.8 MiB) more than the envelope
allows.

**Disposition:** the predeclaration's own refusal rule applied - no ceiling raised, no smaller branch
substituted, no capture side reduced, no variant tried, no study/sweep/curation launched and no re-run
under a fresh number. `precision_improved = false` because precision was **never measured**, not
because it failed to improve; every gate and the population stay untouched and unanswered. The attempt
is preserved at `runs/attempts/m3-occres-lever-refused-r1/` with a written note, and
`docs/evidence/m3-occupancy-resolution-r1/refusal.json` binds the numbers to both receipts (recorder
`runs/m3-occres-refusal-r1-job.json`). The estimate error is ledger entry **V-062**
(resource-profile), guarded by a census instrument, so a lever's declared cost is now four retained
maps rather than one.

**Audit:** 30 numeric claims on the lines this pass added, across the five touched documents, were re-derived from the two profile receipts and the refusal record, with 16 receipt joins re-derived - **0 drift**. A whole-file sweep flags 130 prior-pass history figures in these same files, so the instrument is scoped to added lines (`docs/evidence/m3-occupancy-resolution-r1/audit.json`) rather than weakening the documents.

**Next-ranked: stack `occupancy_context` with `occupancy_detail`.** It is the last lever that changes
the information the read-out sees rather than the sampling of it, and it fits the declared envelope
(the 8x8 context tap retains about 0.02 MiB against the refused branch's 188.4 MiB). Resolution stays
open only as a separately declared configuration - smaller branch, reduced capture side, or a
computed-at-the-cell formulation - never as a substitution inside this pass. Scale waits on whichever
lever carries the gain; the real-image chain remains parked pending authorization.

## 2x2 in-cell detail stencil measured: not cleared, attribution floor met, separation now binds (2026-11-09)

The step the precision pass ranked now has its own measured verdict
([M3_OCCUPANCY_STENCIL.md](M3_OCCUPANCY_STENCIL.md), predeclared at `3ecbf2a`,
`docs/evidence/m3-occupancy-stencil-r1/scope.json`, curation `runs/m3-occdstencil-curation-r1-job.json`).
Both arms carried the declared zero-initialized detail tap and differed only in its form: the lever
read four quarter-cell samples of a 2x2 in-cell grid, the control the single center sample. Each arm
was continued **64 matched steps** from `m3-percell-balance-r1` under the same captured catalogue
(`6ff287a1`) inside 1,152 MiB (lever 202.8 s / 1,032,060,928 B, control 218.5 s / 1,031,536,640 B) and
scored on the widened deciding population (36 images, 28 nonempty boards, 667/4,933 cells, 81-point
grid, 54 forwards per sweep under 640 MiB, twelve ancestry clusters).

**Verdict: NOT CLEARED** - per-cell F1 0.3952 against 0.70; ranking `uninformative` (AUC **0.8804**,
CI 0.8633-0.8975, **0.020 short** of 0.90; best swept F1 0.4913, **0.209 short** of 0.70); 0/28 exact
boards. The **attribution condition is met** (+0.0793 against the +0.05 floor) - the first arm of the
campaign to clear it. **Mechanism: `precision_improved = false`** - precision at `0.0` moved 0.2363 ->
0.2474 (+0.0111, cluster 0.0047-0.0210, interval excluding zero but below the declared 0.05 floor),
so the extra in-cell capacity did not carry the demonstrated precision gain further; it raised
ordering (AUC 0.8226 -> 0.8804, cluster delta +0.0385 to +0.0943) and the swept optimum (0.4120 ->
0.4913) instead.

**Implication: separation, not ordering, is the binding constraint.** The ordering deficit has shrunk
from 0.076 (neighborhood) and 0.077 (precision tap) to 0.020, while the swept-F1 deficit has stayed at
0.209-0.288 and no board is exact. Next-ranked: **encoder resolution** (a finer native tap than
stride-2), then scale; threshold, conditioned-threshold and class-budget families stay closed, and
stacking `occupancy_context` with `occupancy_detail` is still untested. The single-sample control
reproduced the prior pass's lever bit-for-bit (AUC 0.8226433610683003, counts 661/2136/6, best F1
0.41203400915631133), so the continuation is deterministic and the contrast is against exactly the
demonstrated tap. No escaped run, no ceiling, gate, population or floor change, and no new defect.

## Cell-precision detail tap measured: not cleared, but precision improved as declared (2026-11-09)

The next-ranked lever after the neighborhood result now has its own measured verdict
([M3_OCCUPANCY_PRECISION.md](M3_OCCUPANCY_PRECISION.md), predeclared at `684d775`,
`docs/evidence/m3-occupancy-precision-r1/scope.json`, curation `runs/m3-occdetail-curation-r1-job.json`).
One declared change — the stride-2 first-stage feature sampled at each cell center through a
zero-initialized projection, exactly the published function at step zero — was continued **64 matched
steps** from `m3-percell-balance-r1` against a fresh control under the same captured catalogue; both
arms ran inside 1,152 MiB and 64 steps (lever 191.2 s / 1,053,134,848 B, control 229.5 s /
1,055,244,288 B), and each was scored on the widened deciding population (36 images, 28 nonempty
boards, 667/4,933 cells, 81-point grid, 54 forwards per sweep under 640 MiB, twelve ancestry
clusters).

**Verdict: NOT CLEARED**, all three gate conditions failing: per-cell F1 **0.3816** against 0.70;
ranking `uninformative` (AUC **0.8226**, CI 0.8027-0.8426, **0.077 short** of 0.90; best swept F1
0.4120, **0.288 short** of 0.70); attribution -0.0217 against the +0.05 floor. **Mechanism:
`precision_improved = true`** — precision at `0.0` rose **0.1752 -> 0.2363** with the twelve-cluster
interval excluding zero, and it did not cost coverage (recall 0.7616 -> 0.9910), so standing-point
F1 gained **+0.0968**, the largest read-out gain so far.

Implication: precision is addressable from the read-out, but read-out information at this budget is
not separation — the swept optimum still sits below the control's (0.4338 at `3.90`). **Next-ranked
step, in priority order:** (1) a 2x2 in-cell detail stencil per cell through the same
zero-initialized projection (more capacity for a mechanism that measurably moved precision); (2)
encoder-side resolution if that saturates; (3) scale the matched budget on whichever carries the
gain. Threshold, conditioned-threshold and class-budget families stay closed. The first control
study was killed by the platform's 295 s cap at step 26 and is preserved under `runs/attempts/`,
re-recorded canonically (V-059 corrects a hand-computed AUC gap in four documents).

## Occupancy representation lever measured: not cleared — ordering improved, separation did not (2026-11-09)

The last ranked occupancy question now has a measured verdict rather than a ranking
([M3_OCCUPANCY_REPRESENTATION.md](M3_OCCUPANCY_REPRESENTATION.md), predeclared at `a07f9d2`,
`docs/evidence/m3-occupancy-representation-r1/scope.json`, curation receipt
`runs/m3-occctx-curation-r3-job.json`). One declared read-out change — a zero-initialized 3x3
neighborhood residual before the occupancy head, exactly the published function at step zero —
was continued **64 matched steps** from the published checkpoint `m3-percell-balance-r1` against a
control continuation from the same bytes; both arms ran inside the declared 1,152 MiB / 64-step
envelope (lever 206.4 s / 1,059,500,032 B, control 193.8 s / 1,043,173,376 B) and were scored
read-only on the widened deciding population (36 images, 28 nonempty boards, 667/4,933 cells, 81
point grid, 54 forwards each under 640 MiB).

**Verdict: NOT CLEARED**, all three declared conditions failing: per-cell F1 0.3600 against 0.70;
ranking `uninformative` (AUC **0.8244** CI 0.8045-0.8442, cluster 0.8199 [0.7941, 0.8390], still
0.076 short of 0.90; best swept F1 0.4200 against 0.70); attribution **-0.0138** against the +0.05
floor. What the block did do is improve **ordering** (AUC 0.7427 -> 0.8244 versus its control, and
standing-point F1 0.2848 -> 0.3600, cluster delta 0.0142-0.1106) and make the read-out permissive
(recall at `0.0` 0.73-0.79 -> 0.985-0.998) rather than better-separating: the control's swept
optimum is slightly higher (0.4338 at `3.90`), and exact boards stay 0/28 in both arms.

So separation remains the binding constraint at this budget, and read-out local context is not the
representation that supplies it. **Next-ranked step:** a representation aimed at cell-level
*precision* (contrastive or boundary-aware detail at the lattice, or higher-resolution sampling at
the cell points), with the same matched-control discipline; threshold, conditioned-threshold and
class-budget families stay closed. Attempts preserved and re-recorded canonically: a runner-restart
kill at step 46, a 36-item work cap on 54-forward sweeps, and an all-split bootstrap scope — each
under `runs/attempts/`, none used as evidence (V-057, V-058).

## Occupancy conditioning measured: the failure is per-board, so the boundary is closed when conditioned (2026-11-09)

The **unranked** representation question is now ranked by measurement rather than inference
([M3_OCCUPANCY_CONDITIONING.md](M3_OCCUPANCY_CONDITIONING.md), predeclared at `deaba0f` before the
canonical receipt, `docs/evidence/m3-occupancy-conditioning-r1/scope.json`). Read-only, on the
committed widened sweep cases, with the standing 81-point grid and the 0.70 bar: **0 of 28**
nonempty deciding boards reach the bar even at its own oracle grid point (floor was 4), so the
verdict is `representation-limited` and the oracle threshold spread is 0.0. Even with one
threshold per board chosen with the labels in hand, the oracle-pooled F1 is **0.4305** (551 true /
1,342 false positive / 116 missed), 0.27 short of the bar; the per-board F1 at the standing `-0.90`
point has median **0.0857**, while per-board AUC runs min **0.291**, median **0.829**, max
**0.970** — the logits order occupied above empty on most boards but never separate them enough
for a board to clear the bar. Density joins (dense 12 boards at median oracle threshold -1.35,
sparse 12 at -0.225, empty 4 at -0.45) but cannot license a conditioned rule when no board reaches.

Consequence: the decision-boundary family is closed **when conditioned as well as globally**, so
no density-conditioned or per-board threshold is ranked, and the next lever is the representation —
now a measured claim instead of an inference. The ranking verdict itself is untouched (`occupancy_sweep`
still returns `uninformative`), and no model, loss, threshold, gate or prior evidence byte moved.
Receipt `runs/m3-occupancy-conditioning-r1-job.json` (passed, 384 MiB ceiling, 37,457,920 B peak);
the first receipt predated the predeclaration commit and is preserved under `runs/attempts/` with
its evidence directory, with the canonical one re-run afterwards. `tests/test_board_conditioning_artifacts.py`
re-derives every published number from the committed bytes.

## Deciding population widened: both verdicts re-derive unchanged and tighter (2026-11-09)

The instrument change is measured ([M3_WIDENED_DECIDING_POPULATION.md](M3_WIDENED_DECIDING_POPULATION.md),
predeclared at `111aa9e` before any widened-population receipt existed,
`docs/evidence/m3-widened-deciding-population-r1/scope.json`). Every M3 verdict had been decided
on the three-group training probe population while development already admitted twelve; the
deciding split is now the canonical twelve groups, and **no threshold, gate or bar changed**.

| population | narrow (superseded) | widened |
|---|---|---|
| training groups / images | 3 / 9 | **12 / 36** |
| nonempty boards | 7 | **28** |
| occupied known cells | 168 | **667** |
| empty known cells on nonempty boards | 1,232 | **4,933** |
| development images (unchanged) | 18 | 18 |

Neither verdict flips, and neither is now explained away by a small population:

| gate number | narrow (superseded) | widened, with 95% interval | verdict |
|---|---|---|---|
| ranking AUC (bar 0.90) | 0.7491, CI 0.7049-0.7934 | **0.7852, CI 0.7640-0.8064** | UNINFORMATIVE |
| best swept F1 (bar 0.70) | 0.3470 | **0.3704**, precision 0.2522 [0.2329, 0.2725], recall 0.6972 [0.6612, 0.7308] | UNINFORMATIVE |
| operating-point F1 (bar 0.70) | 0.3109 | **0.3455**, gap 0.3545, precision 0.2810 [0.2548, 0.3088], recall 0.4483 [0.4109, 0.4862] | STANDS |
| exact nonempty training boards (bar 3) | 0/7 | **0/28** | FALSE |
| identity distinct classes (bar 3) | `EOS`x54, `J`x9 | **`EOS`x216, `J`x33** | STANDS |
| identity exact piece rate | 0/54 | **0/216**, CI [0.0, 0.0175] | cost |
| hold exact rate | 3/12 | **15/33**, CI [0.2984, 0.6201] | cost (J's frequency) |

The intervals are the finding as much as the points: the AUC interval narrowed from +-0.044 to
+-0.021 and its **upper bound is still 0.094 below the bar**; the widened best F1 sits 0.33 below
0.70. At the sweep's best recall (465 of 667 recovered, 202 still missed), reaching F1 0.70
would need false positives to fall from 1,379 to **at most 196** - a 7.0x precision gain at fixed
recall. The fit gate is also a *harder* gate at the same threshold - 3 boards now means 3 of 28. The identity heads emit one class each at four times the population and 216
supervised positions, so the constant-predictor finding is confirmed rather than an artefact
of `n`; hold exactness (15/33) is exactly `J`'s frequency.

Superseded without invalidation: the recorded nine-case subset of each widened replay reproduces
the earlier published bytes exactly - 67 true / 196 false positive / 101 missed occupied cells,
0 exact nonempty boards, and the queue `EOS`x54 / hold `J`x9 classes. The earlier receipts,
evidence directories and the frozen manifest are untouched; the narrow numbers in
[M3_OCCUPANCY_RANKING.md](M3_OCCUPANCY_RANKING.md), [M3_PERCELL_DECISION.md](M3_PERCELL_DECISION.md)
and [M3_FIT_GATE.md](M3_FIT_GATE.md) remain their runs' own scope and are marked there.

Receipts: admission `runs/m3-sweep-widened-occupancy-r1-profile-r1-job.json` (passed),
occupancy replay `runs/m3-percell-widened-r1-job.json` (passed, 54 forwards, 45.0 s, 559,869,952 B),
identity replay `runs/m3-identity-factor-widened-r1-job.json` (passed, 54 forwards, 45.7 s,
555,425,792 B, weights `8042c288...` unchanged), sweep
`runs/m3-sweep-widened-occupancy-r1-job.json` (passed, 54 forwards, 44.3 s, 585,424,896 B), and
curation `runs/m3-widened-population-curation-r1-job.json` (passed) into the evidence directory.
The rule is code in `scripts/probe_population.py` (population), `scripts/gate_power.py`
(intervals, which no verdict reads), `scripts/identity_gate.py` (the discrimination rule) and
`scripts/record_widened_gates.py` (curator), re-derived by `tests/test_widened_gates.py`.
V-055 records the objective-binding defect this pass exposed in the replay.

### Clustering addendum: the intervals above are cell-count intervals (2026-11-09)

The 95% intervals in that table treat 667 occupied and 4,933 empty cells as independent. They
are 28 boards of 200 cells inside 12 ancestry groups, and measuring that nesting changes the
widths (V-056, [M3_WIDENED_DECIDING_POPULATION.md](M3_WIDENED_DECIDING_POPULATION.md) addendum,
`docs/evidence/m3-widened-deciding-population-clustering-r1/scope.json`):

| gate number | point | cell-level 95% (published) | ancestry-cluster 95% (corrected) |
|---|---|---|---|
| ranking AUC (bar 0.90) | 0.7852 | 0.7640-0.8064 | **0.7474-0.8186** |
| precision at `0.0` | 0.2810 | 0.2548-0.3088 | **0.0908-0.4538** |
| recall at `0.0` | 0.4483 | 0.4109-0.4862 | **0.4076-0.4695** |
| F1 at `0.0` (empirical) | 0.3455 | - | **0.1488-0.4538** |
| precision at `-0.90` | 0.2522 | 0.2329-0.2725 | **0.0802-0.4237** |
| recall at `-0.90` | 0.6972 | 0.6612-0.7308 | **0.6473-0.7400** |
| hold exactness (15/33) | 0.4545 | 0.2984-0.6201 | **0.1818-0.7273** (design effect 3.3) |

Measured design effect: 1.19 (board) / 2.87 (ancestry) on the per-cell error at the operating
point, 2.18 / 5.30 at the swept best, ICC 0.000941 / 0.004024 and 0.005906 / 0.009233 - so the
instrument has **232-562 effective occupied cells, not 667**. The clustering is almost entirely
in *precision* (design effect 14.4-76.0) and not in recall (0.25-1.99): how many of a board's
predicted cells are wrong is a board property. The queue's 0/216 census has design effect 1.0
and its interval cannot widen. **No verdict changes**: the corrected AUC upper bound 0.8186 is
still 0.081 below 0.90, every corrected precision upper bound is far below what F1 0.70 needs,
and the identity findings rest on counts and class frequencies. The only claim qualified is that
the cell-level precision intervals were over-confident by **3.80x-8.72x** - the square root of
the 14.4x-76.0x precision design effects, confirmed by the measured interval-width ratios of
6.73x at `0.0` and 8.67x at `-0.90` (the first draft printed 3.4x-6.7x, which was wrong at both
ends). Superseded-without-invalidation still holds, and a like-for-like recomputation of the
narrow sweep (0.7049-0.7934 cell, 0.7039-0.7794 board) shows the narrow numbers were superseded
for population, not width.

Stopping point: occupancy is closed from all three sides **on a population 3.8x larger**, and
identity is closed at four times the supervised positions. The next model question is unchanged
and still unranked: a representation that separates occupied from empty cells at this budget.

## Occupancy ranking measured: UNINFORMATIVE, operating-point family exhausted (2026-11-09)

The read-only threshold sweep named by the per-cell pass is measured and returns the second
predeclared verdict: the locked occupancy **ranking is UNINFORMATIVE**
([M3_OCCUPANCY_RANKING.md](M3_OCCUPANCY_RANKING.md), rule committed at `9ad3598` before any
sweep receipt, `docs/evidence/m3-sweep-occupancy-r1-ranking-r1/scope.json`). On the committed
`m3-percell-balance-r1` checkpoint, swept over the predeclared 81-point grid, training AUC is
**0.7491** against chance 0.5 (bar 0.90) and no threshold on the whole grid reaches the
required occupied-cell F1 0.70 - the best point is **-0.90** with precision 0.2393, recall
0.6310, F1 **0.3470**, 0/7 exact boards and 122/400 empty-population false positives.
Development AUC is 0.8712, best F1 0.5456 at -0.45, 0/14 exact boards. The grid's zero row
reproduces the replay's 67/196/101 counts exactly, so the sweep is a join as well as a
measurement.

Consequence, stated as the rule requires: the occupancy head of this representation is
**closed** and the operating-point family is **exhausted** - nobody re-sweeps it. The logits
rank better than chance but do not separate the classes well enough for any threshold to
approach fit; the class-budget run's 0.3109 sits below the swept ceiling of 0.3470, so the
entire threshold family buys a few percent and nothing more. Nothing here re-opens the fit
gate, the class weight or the scale lever, and no real-image, complete-state or quality claim
follows.

Receipts: admission `runs/m3-sweep-occupancy-r1-profile-r1-job.json` (passed), sweep
`runs/m3-sweep-occupancy-r1-ranking-r1-job.json` (passed, 27 forwards, 25.6 s, under the
640 MiB ceiling), curation `runs/m3-sweep-occupancy-r1-ranking-r1-curation-job.json`
(passed), tests `runs/m3-sweep-occupancy-tests-r4-job.json`. The rule is code in
`scripts/occupancy_sweep.py` (`threshold_grid`, `pooled_auc`, `evaluate`, `sweep_verdict`),
curated by `scripts/record_stream_sweep.py`, and re-derived by `tests/test_sweep_rule.py`.
V-054 records the counting defect the predeclared zero-row join caught in the first attempt.

Stopping point: the occupancy question is closed from all three sides (weight scale, class
split, decision boundary). The next model question is a representation that separates
occupied from empty cells at this budget, which is not a threshold choice and not ranked here.

## Per-cell decision lever measured: class budget STANDS, error reallocation (2026-11-09)

The predeclared per-cell rule ([M3_PERCELL_DECISION.md](M3_PERCELL_DECISION.md), rule
committed at `0d77cef` with no receipts in the tree) is measured and **STANDS**: the deciding
quantity, occupied-cell F1 on the seven nonempty training boards, is **0.3109** against a
predeclared threshold of 0.70 (development 0.5350). Evidence:
`docs/evidence/m3-percell-balance-r1-errors-r1/scope.json`. The lever was
`--occupancy-positive-weight 4`, a class budget inside the balanced occupancy term with the
term's total weight held at the already-measured `4.`, matched to `m3-fit-occupancy-r1` on
capacity 64, identity factor 4, capture 512, seed `20261026`, LR .001, 128 updates,
eval/checkpoint 64 and six development groups.

The mechanism is a reallocation, not discrimination: recall rose 34/168 -> 67/168 while
precision fell 34/93 -> 67/263, misses 134 -> 101 and inventions 59 -> 196, so F1 moved only
0.2605 -> 0.3109; empty-population false positives tripled 23 -> 72, which is a decision
boundary shift rather than a representation change. Exact boards stayed 0/7 and 0/14, empty
boards 0/2 and 0/4, roles 0/9 and 0/18, ghost 0/9 and 0/18, readable components 0/9 and
0/18. Family instability continued and is inert: queue EOS returned to exact 9/9 and 15/15
(0/9 and 0/15 in the fit-gate run) while piece tokens collapsed 15/54 -> 0/54 and 9/90 ->
0/90; presence 54/63, 90/105 and slot status 54/54, 90/90 were unchanged.

Receipts: admission `runs/m3-percell-balance-r1-profile-r1-job.json`, study
`runs/m3-percell-balance-r1-job.json` (128/128, 459.2 s, 963,743,744 B),
replay/curation `runs/m3-percell-balance-r1-errors-r1-job.json` and `...-curation.json`
(weights `9ae1fef9f649...`, 27 forwards, all saved metrics reproduced), tests
`runs/m3-percell-balance-tests-r2.json`. The rule and this verdict are executable in
`scripts/percell_gate.py` with `tests/test_percell_gate.py`; the objective join is
`arguments.occupancy_positive_weight` + `recovery_contract`, replayed and curated.

Ranked next instrument: a read-only threshold sweep of the locked occupancy logits on this
committed checkpoint, to separate a wrong operating point from logits that do not rank the
classes. If the ranking is informative, the operating point is an owner precision/recall
budget choice, not something this evidence can settle.

## Fit gate measured: it stands, and the occupancy term is not starved - 2026-11-08

`docs/M3_FIT_GATE.md` predeclared the rule at `5065811`, before any fit receipt: the gate
opens iff `docs/evidence/m3-fit-occupancy-r1-errors-r1/result.json` reports >= 3 exact
nonempty **training** boards of seven; empty boards, development, roles, geometry and
component counts are costs that can never upgrade a negative. The lever was
`--occupancy-factor 4`, scaling only the locked-occupancy weight from `1.` to `4.`, to test
the length run's constant-empty read-out (`true_positive 0, false_negative 168`), with
everything else matched: capacity 64, factor 4, capture 512, seed `20261026`, LR .001, 128
updates, eval/checkpoint 64, six development groups, positional read-out, length endpoint.

Measured: **the gate STANDS at 0/7** exact nonempty training boards (development 0/14, both
training empty boards also wrong). The mechanism is not starvation. At 4x the head stops
emitting constant-empty (34 of 1,400 training occupied cells recovered, 230 of 2,800
development) and still reproduces no board, because the errors are per-cell and
bidirectional (59 false positives, 134 misses on the training nonempty cells) and the scale
is paid elsewhere: queue EOS fell 9/9 -> 0/9 and 15/15 -> 0/15, presence 63/63 -> 54/63 and
105/105 -> 90/105, position spread 0.0877/0.0838 -> 0.0641/0.0379. Queue piece tokens did
rise to 15/54 and 9/90 (from 0/54 and 0/90) and slot status stayed exact (54/54, 90/90);
roles 0/9 and 0/18, ghost 0/9 and 0/18 (9/9 in the prototype run, 3/9 in the endpoint run,
so the family's instability continues), exact readable components 0/9 and 3/18
(empty-queue cases), geometry 3.3397 and 2.6864.

Ranked next: the **locked read-out's per-cell decision** (its class imbalance and cell-wise
scoring) - the recorded errors are recall (34 of 168 occupied cells) plus 59 false
positives, measured by this same rule. Not ranked: another occupancy weight scale, a budget
extension of this lever, or any further identity/endpoint variant (that disjunction is
closed). The owner-decision real-image chain is unchanged.

Receipts: admission `runs/m3-fit-occupancy-r1-profile-r1-job.json` (passed), study
`runs/m3-fit-occupancy-r1-job.json` (passed, 128/128, 250.6 s, 990,515,200 B, weights
`d0bb45ce...`), canonical replay `runs/m3-fit-occupancy-r1-errors-r1-job.json` (passed, 27
forwards, weights unchanged, every saved metric reproduced, `occupancy_factor: 4`),
curation `runs/m3-fit-occupancy-r1-errors-r1-curation.json` (passed) into
`docs/evidence/m3-fit-occupancy-r1-errors-r1/scope.json`, tests
`runs/m3-fit-occupancy-tests-r4.json` (7 modules, 101 tests, `OK`). The rule is code:
`scripts/fit_gate.py` with `tests/test_fit_gate.py`, which also re-derives this verdict from
the committed `result.json`.

V-053: the first curation died on `runs/m3-fit-occupancy-r1-errors-r1.log` because the
bounded launch writes the probe's stdout as `<name>-job.log` while the curator copies
`<name>-job.json`, `<name>.log`, `<name>.stderr.log`. The sidecars are now declared once
(`record_stream_errors.NATIVE_SIDECARS`) and bound by a test, the failed attempt and its
partial evidence directory are preserved under `runs/attempts/`, and the re-run replay and
curation occupy the canonical names.

## Queue length: the identity disjunction is closed — 2026-11-08

`docs/M3_IDENTITY_QUEUE_LENGTH.md` predeclared at `d2895de`, before any receipt, the last
lever of the disjunction: `--component-endpoint-term length`, one distribution over queue
lengths `0..capacity` derived from a per-position hazard, with the 7-way piece softmax, the
positional read-out and the four-query hold bank unchanged and everything else matched
(capacity 64, factor 4, capture 512, seed `20261026`, LR .001, 128 updates, six groups).

Measured: queue `EOS`x54 training (distinct **1**, constant true) and `EOS`x90 development;
hold `J`x9 (distinct **1**, 3/9 exact) and `J`x12 (3/12). Position spread collapsed to
0.0877 training and 0.0838 development (from 1.2338 and 1.2304), and presence — a
per-position term sharing the read-out — fell from 54/63 and 90/105 to 9/63 and 15/105
while EOS and slot stayed exact. Verdict **NEGATIVE** under the predeclared rule.

Mechanism: with one global length answer, the objective stops asking any position to
differ, and the shared read-out collapses — the emitted class is `EOS` everywhere, and the
only terms that stay exact are the two that do not need position. Read-out ruled out (three
variants; the spread tracks the objective, not the architecture), data ruled out (presence
was 54/63 when asked per position), loss masking remains: every objective tried admits a
single class everywhere as a minimiser. **Disjunction CLOSED on evidence** — all three
sides measured at the fixed budget, each predeclared and receipted, each negative.

Ranked next: the **fit gate** (locked occupancy and role read-out), FALSE at 0/9 and 0/18
boards and 0/18 roles, and unstable across runs (ghost 9/9 one run, 0/9 here). No further
identity/endpoint variant and no budget extension of the same objective is ranked. This is
ranking, not a measured claim; the fit gate has to be measured before anything is asserted
about it.

Receipts: admission `runs/m3-identity-factor-length-r1-profile-r1-job.json` (passed, 34.2 s,
981,618,688 B), study `runs/m3-identity-factor-length-r1-job.json` (passed, 128/128,
228.0 s, 1,011,888,128 B, checkpoint `8042c288...`), canonical replay
`...-errors-r1-job.json` (passed, 27 forwards, 16.2 s, 602,841,088 B,
`component_endpoint_term: length`, weights unchanged, saved metrics reproduced), curation
`...-curation.json` (passed) into
`docs/evidence/m3-identity-factor-length-r1-errors-r1/scope.json`, tests
`runs/m3-identity-length-tests-r3.json` (6 modules, `OK`, 92 tests).

V-052: the first study was refused because the admission gate's allowed set was a
hand-copied tuple that never learned the new mode. The endpoint objectives are now declared
once (`components.ENDPOINT_TERMS`) with one admission join (`require_endpoint_term`), bound
to the driver by a test; the second refusal (superseded profile catalogue after the fix)
was the gate working as declared. Both attempts and the superseded profile are preserved
under `runs/attempts/`.

## Endpoint loss: the constant moves, the disjunction does not close — 2026-11-08

`docs/M3_IDENTITY_ENDPOINT_LOSS.md` predeclared at `008be3d`, before any receipt, the
third lever: `--component-endpoint-term separate`, which takes the endpoint out of the 7-way
piece softmax and gives it its own binary `queue_endpoint` logit, emitting 8 columns that
are log-probabilities of the factorization so the ordinary argmax metrics still apply.
Everything else matched the positional run (capacity 64, factor 4, capture 512, seed
`20261026`, LR .001, 128 updates, six development groups, positional read-out and
four-query hold bank).

Measured: queue `predicted_classes` `EOS`x54 training (distinct **1**, constant true) and
`EOS`x90 development; hold `S`x7/`T`x2 (distinct **2**, constant false, 1/9 exact) and
`S`x11/`T`x1 (4/12). Position spread rose to 1.2338 training / 1.2304 development while
every argmax stayed one class. Verdict **NEGATIVE** on the predeclared rule.

Mechanism: the endpoint term is charged at every known position (63 training, 105
development; 9 and 15 positives — exactly 1/7), so a constant "endpoint here" answer costs
0.410 nats in both splits and preserves EOS 9/9, slot 54/54, presence 54/63; the product
factorization then suppresses every piece class, so the argmax is `EOS` while piece tokens
are 0/54 and 0/90. The shortcut migrated; it did not disappear. Held: read-out ruled out
(spread larger than under either prior lever), data ruled out (54 positions, 7 classes, 6
slots), so the remaining constraint is the loss's per-position scoring of a
per-sequence count. The class-side/position-side/loss-side disjunction is **not closed**;
what is closed is where the constant lives.

Hold: four identity runs have now produced 2, 1, 1 and 2 distinct classes with three
different identities (`L`/`O`, `T`, `S`/`T`) — initialization-dependent two-class splits,
never the three-class bar, so no robust hold mechanism exists yet.

Structure cost/gain: ghost 9/9 and active 3/9 training exact, roles 0/18 development,
boards 0/9 and 0/18 (the control's two empty-board successes are lost), geometry 4.0672
and 2.3110, exact readable 0/9 and 0/18. Fit gate **FALSE**.

Receipts: admission `runs/m3-identity-factor-endpoint-r1-profile-r1-job.json` (passed,
35.1 s, 981,151,744 B under 1,152 MiB), study `runs/m3-identity-factor-endpoint-r1-job.json`
(passed, 128/128, 256.6 s, 1,024,016,384 B, checkpoint `84faf263...`), canonical replay
`runs/m3-identity-factor-endpoint-r1-errors-r1-job.json` (passed, 27 forwards, 16.2 s,
565,972,992 B, `component_endpoint_term: separate`, weights unchanged, saved metrics
reproduced), curation `...-curation.json` (passed) into
`docs/evidence/m3-identity-factor-endpoint-r1-errors-r1/scope.json`, tests
`runs/m3-identity-endpoint-tests-r2.json` (6 modules, `OK`, 88 tests). V-051 records that
the run ledger cannot itself show which component objective a run trained under, now
covered by probe/curator provenance and the mismatched-join tests rather than a new
ledger field (`compare_layer_studies.py` iterates the last curve row's keys).

Ranked next: **queue length as one global decision** — a single distribution over
`0..capacity` (or one position classification) instead of a per-position binary endpoint
question, so "endpoint everywhere" is not in the hypothesis space; same discrimination
rule, same read-outs, position spread reported. Not ranked: further read-out conditioning,
budget, or re-weighting the per-position endpoint term. Fit gate and the owner-decision
real-image chain unchanged.

## Per-position queries: position information reaches the decision, which ignores it — 2026-11-08

`docs/M3_IDENTITY_POSITIONAL.md` predeclared the lever at `aa5bcc0` before any receipt:
the validated prototype read-out (eight image-derived class queries plus the four-query
hold bank) with one learned query per queue position attending the same memory, so a
position's own content can move its class logits. Everything else matched the prototype
run — capacity 64, factor 4, capture 512, seed `20261026`, 128 updates, six development
groups. The rule is discrimination-only: queue needs >= 3 distinct training predicted
classes with `constant_predictor` false, hold needs >= 3, 2x2 total, structure recorded as
a cost that cannot upgrade a negative.

Measured by the independent 27-forward replay: training queue `predicted_classes` is
`EOS` x54 (distinct **1**, constant true) and development `EOS` x90; training hold is
`T` x9 (distinct **1**) and development `T` x12. Verdict **NEGATIVE** on both conditions,
with the position demonstration recorded in the same evidence: per-position logits differ
within a frame (max spread 0.3627 training, 0.5758 development; means 0.3594, 0.3347;
`position_axis_reaches_logits` true) while every argmax is one class. Two tests hold the
halves — every position reading the frame through its own query, and the diagnostic
carrying a positive spread with a constant decision.

The three candidate constraints are separated by the artifacts. The **read-out** is ruled
out: two independent conditioning axes are wired in and the position axis demonstrably
reaches the logits. The **data's position independence** is ruled out: 54 supervised
positions carry seven classes across six slots. What remains is the **loss**: the
endpoint is a class inside the same 8-way identity softmax and masked mean as the pieces,
and it is the one always-present, always-exactly-predictable label (9/9 and 15/15 here,
and in every previous run), so emitting `EOS` everywhere keeps endpoint, slot and
presence terms at or above control (presence 54/63 and 90/105) and pays only the piece
term. Also recorded: the prototype run's two-class hold split did not reproduce here
(same structure, different RNG draws, constant `T`, 0/9) — it was init-dependent. The
three development exact-readable cases are empty-queue frames whose hold target equals the
emitted constant, not reading.

Fit gate still FALSE: 2/9 and 0/18 boards (both training successes empty), 0/18 and 0/36
roles, 0/9 and 0/18 exact readable components; geometry 3.5398 and 2.4024.

Receipts: admission `runs/m3-identity-factor-positional-r1-profile-r1-job.json` (passed,
37.5 s, 980,447,232 B under 1,152 MiB), study
`runs/m3-identity-factor-positional-r1-job.json` (passed, 238.5 s, 978,120,704 B,
checkpoint `060ee296...`), canonical replay
`runs/m3-identity-factor-positional-r1-errors-r1-job.json` (passed, 27 forwards, 15.8 s,
566,697,984 B, weights unchanged), curation
`runs/m3-identity-factor-positional-r1-errors-r1-curation.json` (passed) into
`docs/evidence/m3-identity-factor-positional-r1-errors-r1/scope.json`, and tests
`runs/m3-identity-positional-tests-r1.json` (5 modules, `OK`, 461,545,472 B). V-050
records the killed-supervisor receipt the shell cap left behind; the killed receipt and
partial output directory are retained under `runs/attempts/`.

Ranked next: the **loss formulation** under the same rule — give the endpoint its own
binary term or drop it from the piece softmax so the always-predictable label stops being
the cheapest constant, reporting the position-axis spread unchanged. Not ranked: more
read-out conditioning (class and position sides both closed here) or another budget
extension. The owner-decision real-image chain and the FALSE fit gate are unchanged.

## Per-piece image queries: a measured negative, with the position side named — 2026-11-08

`docs/M3_IDENTITY_PROTOTYPES.md` predeclared the lever at `bdae6e1` before any receipt:
a third identity read-out (`--component-identity-readout prototype`) in which eight
learned class queries attend the observed memory and board, positions are classified
against those image-derived class vectors, and the hold piece reads a four-query bank
instead of one role vector. Capacity 64, factor 4, capture 512, seed `20261026`, 128
updates, six development groups — everything else matched to `m3-identity-factor-r1`.

The predeclared rule is about class-level discrimination only: queue needs >= 3 distinct
training predicted classes and hold needs >= 3, because V-048 showed exact counts are
class frequencies when a head is constant. Measured by the independent 27-forward
replay: training queue `predicted_classes` is `EOS` x54 (distinct **1**, constant
true) and development is `EOS` x90; training hold is `L`x3/`O`x6 (distinct **2**),
development `L`x2/`O`x9/`S`x1. The verdict is **NEGATIVE** on both conditions.

The mechanism is measured. The class side did what it was designed to do — identical
queries produce different class vectors on different frames — but the emitted class is
invariant across 54 supervised positions from nine frames, and it is `EOS`, whose own
term was already satisfied (9/9 training and 15/15 development exact, against the
control's 8/9 and 3/15). Predicting it everywhere keeps endpoint and slot status perfect
and pays only the masked piece term, so the head takes the constant with free structural
terms. The only position-varying input left is the index encoding plus the single shared
role vector, and it never separated. The four-query hold bank is the only mechanism that
moved off a constant (2 training and 3 development classes, 3/9 exact vs 0/9).

Structure is recorded, not used as a verdict branch: presence 54/63 -> 39/63 training and
90/105 -> 48/105 development, roles 12/18 -> 0/18, EOS improved, geometry 3.83 -> 3.56
training and 2.03 -> 2.32 development. Fit gate still FALSE: 2/9 and 0/18 boards (both
training successes empty), 0/18 and 0/36 roles, 0/9 and 0/18 exact readable components.

Receipts: admission `runs/m3-identity-factor-prototype-profile-r1-job.json` (passed,
39.1 s, 967,761,920 B under 1,152 MiB), study
`runs/m3-identity-factor-prototype-r1-job.json` (passed, 223.8 s, 1,013,481,472 B,
checkpoint `f818567d...`), canonical replay
`runs/m3-identity-factor-prototype-r1-errors-r1-job.json` (passed, 27 forwards, 15.9 s,
561,905,664 B, weights unchanged), curation
`runs/m3-identity-factor-prototype-r1-errors-r1-curation-r2.json` (passed) into
`docs/evidence/m3-identity-factor-prototype-r1-errors-r1/scope.json`, and instrument
tests `runs/m3-identity-prototype-tests-r3.json` (19 tests, `OK`). V-049 records the
tally crash the replay exposed (an `EOS` prediction at a piece position) and its fix.
Retained, not re-scored: the crashed first replay output
(`runs/attempts/...-errors-r1-attempt1`, receipt `-job-attempt1`), the first curation
attempt's partial evidence directory (`runs/attempts/evidence-...-attempt1`), and the
pre-commit test receipts r1/r2.

Ranked next: the **position side** of the identity read-out — a per-position
image-conditioned query, or a spatial anchor for the panel icon, so the read-out can
differ between positions — keeping the four-query hold bank, which is the one mechanism
that separated classes. Re-running this class-side lever at a larger budget is not
ranked: the emitted class is the cheapest constant and budget does not change that
economics. The owner-decision real-image chain and the FALSE fit gate are unchanged.

## M3 identity heads are constant predictors — 2026-11-08

The identity totals every M3 lever was judged on are class frequencies, not reading.
`docs/M3_IDENTITY_DISCRIMINATION.md` records the measurement: read from the four
published `result.json` files plus the frozen exposure audit's labels, model-free, with no
forward pass. The hold-piece head emits `T` for every supervised case in
`m3-identity-factor-r1`, its capture-640 re-run and the tied read-out run, and its exact
counts are exactly that class's target frequency — 0/9 training, 3/12 development. The
training queue-token head is provably one constant on three of four runs (`S` at 3/54,
`S` at 3/54, `I` at 12/54, each equal to that class's frequency in the 54 training
tokens); only the tied read-out run escapes constancy on three of nine training images,
and no development split is explainable by a single constant.

So the coefficient lever's identity "positive", the budget's `3/54 -> 12/54` rise and
the tied read-out's hold movement are the head switching which single class it emits.
The verdicts themselves are unchanged — each was judged on its predeclared rule, and no
frozen receipt was touched — but no M3 identity number so far is evidence of identity
discrimination, and the "gradient-starved at the published scale" reading is replaced.
The fit gate stays FALSE: 0/9 and 0/18 boards, 9/18 and 0/36 roles, 0/9 and 0/18 exact
readable components.

Receipts: audit `runs/m3-identity-discrimination-r2.json` (passed, 4 work items, 0.23 s,
35.7 MB under 128 MiB), curation
`runs/m3-identity-discrimination-r2-curation.json` (passed) into
`docs/evidence/m3-identity-discrimination-r2/scope.json`, and instrument tests
`runs/m3-identity-discrimination-tests-r1-job-r3.json` (19 tests, `OK`). The instrument
is `scripts/m3_error_analysis.py`: per-case `identity.*` and per-split
`predicted_classes`/`distinct_predicted_classes`/`constant_predictor` for both heads,
close on V-048. Two attempts that captured no stdout/stderr logs are retained
(`runs/identity-discrimination-r1`, `runs/m3-identity-discrimination-r1.json`); nothing
was deleted or re-scored.

Re-ranked, the open identity question is the read-out's conditioning: the separate hold
head reads one role vector and has never left a constant, and a lever that only changes
which constant is emitted is not a lever. Any next identity lever must be able to
produce class discrimination by construction and must report the discrimination fields.
The owner-decision real-image chain and the FALSE fit gate are otherwise unchanged; the
identity levers already closed stay closed.

## M3 input representation (capture side 640): a declared negative — 2026-11-08

`docs/M3_IDENTITY_REPRESENTATION.md` predeclared the lever at `b229972`, before any
receipt: one value, the model capture side, 512 -> 640, threaded through
`model_capture_style` and the trainer's input budget, with every target, mask, order,
loss, architecture and curriculum term unchanged. The control is
`m3-identity-factor-r1` (128 updates, capacity 64, factor 4, separate read-out, seed
`20261026`, LR .001, `stream-v1`, eval/checkpoint 64, six development groups).

The rule is total over the rise x structure square and the observed state lands in its
**NEGATIVE** cell: the primary did not rise (exact queue piece tokens 3/54 training and
9/90 development, equal to control on both) while structure held on all eight declared
terms (training EOS 9/9, slot 54/54, presence 54/63, hold 0/9; development EOS 3/15,
slot 90/90, presence 90/105, hold 3/12). The replay also measures the control's exact
54 and 90 queue tokens and 9/12 hold cases, so the finer capture did not change the
supervised component population on these probes. The fit gate stays FALSE (0/9 and
0/18 boards, 9/18 and 0/36 roles, 0/9 and 0/18 exact readable components, geometry 3.82
and 2.28 cells).

Receipts: admission `runs/m3-identity-representation-profile-r3-job.json` (passed,
87.4 s, 944,017,408 B under 2,048 MiB), study
`runs/m3-identity-factor-representation-r1-job.json` (passed, 128/128 steps, 542.6 s,
986,275,840 B, checkpoint `acb6a8a9...`, catalogue `2d3623a1...`), independent replay
`runs/m3-identity-factor-representation-r1-errors-r1-job.json` (passed, 27 forwards,
39.0 s, 631,767,040 B, every saved metric reproduced, weights unchanged) curated in
`docs/evidence/m3-identity-factor-representation-r1-errors-r1/scope.json` (`passed`),
and the capture-side contract tests inside the whole suite (410 tests, 31.2 s,
540,966,912 B, `docs/evidence/m3-capture-side-tests-r2/`, base `73f0452`), which is
the suite re-run on the final code after the admission-guard repair; the earlier
run at `a892dfb` is preserved as `m3-capture-side-tests-r1`.

Three instrument defects surfaced and are recorded: V-045 (a predeclared ceiling above
the engine's 2 GiB bounded-job cap; lowered to 2,048 MiB, never the capture side),
V-046 (the profile admission compared its observed maximum to the literal 512^2,
refusing every other side; now `512**2 <= observed <= max_side**2`, green on the r3
profile), V-047 (retry receipts with suffixed names are invisible to readers that hard
require `runs/<name>-job.json`; the passing receipt now occupies the canonical name and
every refused attempt is preserved beside it).

Re-ranked, the executable identity levers are exhausted: the coefficient moves identity
at a structural cost, the read-out tying is a negative, the budget is a miss, and the
representation is a negative with structure held. What remains is the owner-decision
real-image chain (admitted, permissioned, reviewed labels) and the unchanged FALSE fit
gate, which no identity lever has moved. The next pass should close the fit gate on its
own terms or take the owner decision on real screenshots; another identity lever is not
ranked.

## M3 identity budget lever: identity rises, structure regresses — 2026-11-08

The budget lever is executed and judged against its own predeclaration (`9970e5a`).
Fresh admission `runs/m3-identity-budget-profile-r1-job.json` (passed, 943,157,248 B),
the 512-update study `runs/m3-identity-factor-budget-r1-job.json` (passed, 512/512 steps,
1729.7 s, 985366528 B under 1,152 MiB, checkpoint
`9b79863b`), its independent 27-forward replay
(34.9 s, 563867648 B, every saved metric
reproduced) and curation curate in
`docs/evidence/m3-identity-factor-budget-r1-errors-r1/scope.json` (`passed`).

By the predeclared rule this is a miss. Identity does rise with budget: exact queue
piece tokens 3/54 -> 12/54 training and
9/90 -> 12/90 development. But the
structure-held condition fails on two declared terms — training queue EOS
8/9 -> 5/9 (declared >= 8/9) and development
hold identity 3/12 -> 0/12 (declared
>= 3/12) — while slot status and presence hold and the fit gate stays FALSE
(0/9 and 0/18 boards,
9/18 and 0/36 roles,
0/9 and 0/18 exact readable
components, geometry 3.03 and 2.07).
Training roles also fall 12/18 -> 9/18. The observed state matched none of the three
declared branches, which is its own defect, V-044, now fixed by making the rule total
over the primary-rise/structure-held square.

Ranked next is therefore the input representation (pixels per icon, the model capture
side): budget moves identity but multiplies a structural trade-off, so extending the
budget again is not the next lever and is not ranked. That lever needs its own
predeclaration, admission and criteria; nothing about it is admitted here.

## M3 identity supervision measured, and the guard policy text corrected — 2026-11-08

The ranked open item after both identity levers closed was "the supervision signal",
so this pass measured it instead of declaring another lever. `docs/M3_IDENTITY_EXPOSURE.md`
records it: `docs/experiments/audit_identity_exposure.py` replays the 27 recorded probes
of `m3-identity-factor-r1` from the run's archived code with no model forward, refuses
unless recipe, raster and annotation bytes match, rebuilds the capacity-64 queue/hold
targets, tallies masked identity by class and slot, and joins the published per-case
diagnostics; `docs/experiments/record_identity_exposure.py` re-derives every count from
the per-case rows. Audit `runs/m3-identity-exposure-r4.json` (passed, 27 cases, 3.248 s,
266,608,640 B under 640 MiB) and curation `runs/m3-identity-exposure-r4-curation.json`
(passed, 128 MiB) curate in `docs/evidence/m3-identity-exposure-r4/scope.json`; the first
three attempts are retained beside it as receipts, each refused by the audit tooling or
its log capture rather than by a measured disagreement.

The counts: all seven piece classes are supervised in both splits (54 training tokens
across slots 0-5, 90 development across slots 0-8, 9 and 15 EOS labels), so coverage is
complete; per-class exposure is 3-18 unique labels, and hold identity sees four of seven
classes. Correctness is class-selective, not positional: the only identities ever exact
are queue `S` (3/3 training, 9/18 development) and hold `T` (0/3 training, 3/3
development), while six queue classes and three hold classes are at zero; development's
nine successes sit at slots 1, 3, 5 and 7, and all 15 development slot-0 labels are
wrong. With the coefficient lever having already moved the totals and the stream drawing a
new ancestry every three steps (128 distinct images from 43 ancestries in the
factor-4 run), what remains is optimization budget versus the input representation —
not global starvation and not class coverage. The budget lever is predeclared in
`docs/M3_IDENTITY_BUDGET.md`.

Also fixed: `docs/RESOURCE_GUARD.md` still described V-040's retired physical stop line
as binding (V-043, green on this audit receipt's `stop_commit_bytes`), so the policy record
now quotes the expressions the supervisor evaluates.

## M3 identity read-out tying: a measured negative, with the role head paying — 2026-11-08

`docs/M3_IDENTITY_READOUT.md` predeclared the lever at commit `e83d70e`, before any
receipt: replace the two sparse identity heads with the cell-type classifier plus one
learned EOS logit (`--component-identity-readout shared`), matched on capacity 64, 128
updates, seed `20261026`, learning rate .001, identity factor 1. `b80d82d` fixed the
first admission refusal at its root (V-039: the shared mode kept dead identity heads,
which the resident-Adam guard rejects); admission r2 passed on the exact mode
(`runs/m3-stream-identity-shared-profile-r2-job.json`, 80.218 s, 945,143,808 B,
checkpoint `c9e8f3f8...`, 262 tensors with warm Adam state), and the study completed
(`runs/m3-stream-identity-shared-r1-job.json`, 128 steps, 573.629 s of recorded
updates inside 574.770 s outer, checkpoint `55a35243...`, catalogue `2f69689a...`,
peak 949,555,200 B under the declared 1,152 MiB ceiling).

By the predeclared rule this is a negative. Identity does move: exact queue piece
tokens 0/54 -> 9/54 training and 0/90 -> 6/90 development, hold 0/12 -> 3/12
development. But the rule required structure held, and it is not: training presence
63/63 -> 54/63, development EOS 15/15 -> 13/15, and — the new cost — training roles
collapse 12/18 -> 0/18 where the coefficient lever had held them. Fit gate stays
FALSE (0/9 and 0/18 boards, 0/9 and 0/18 exact readable components), queue EOS 9/9
and slot status 54/54 and 90/90 still exact, geometry 3.32 training (better than 4.04)
and 2.55 development (worse than 2.03). Independent 27-forward replay (35.867 s,
554,905,600 B, weights `55a35243...` unchanged, every saved per-case field reproduced)
curates in `docs/evidence/m3-stream-identity-shared-r1-errors-r1/scope.json` (`passed`).

The verification gap is closed. The complete contract suite is green on this tree in
`docs/evidence/m3-identity-readout-fullsuite-r11/` (407 tests collected, `Ran 407 tests
in 72.290s`, `OK`, exit 0, receipt peak 545,845,248 B, admitted at 38,347,567,104 B
commit against a 45,626,134,118 B stop line). That count is auditable rather than
asserted: `git diff 5c6b41f HEAD -- tests/` adds exactly 9 test methods and removes
none, so 407 = the previous green run's 398 (its log is preserved at
`docs/evidence/audit-398-baseline-derivation/identity-factor-fullsuite-r1.stderr.log`,
which records `Ran 398 tests in 59.453s` and `OK`) + the 4 identity-read-out tests + the 5
pressure-decision tests; the 32 component-identity/component/joint-stream tests
sit inside it (`test_component_identity` 10, `test_components` 15, `test_joint_stream`
7). The collection is preserved as `collection-job.json`/`collection.log` beside the
suite receipt.

The 398 log deliberately sits in `audit-398-baseline-derivation/`, not beside the
frozen run's receipts: an evidence directory is an atomic record keyed by its name
(the recorders refuse an existing evidence path), so a frozen run's directory cannot
gain files (V-041). Closing it needed no test change and no
weakened receipt: the five earlier refusals (`-r1` mid-run, `-r2`–`-r5` at admission)
were a supervisor defect, V-040, because the commit stop line was derived from physical
RAM (`physical − 2 GiB` = 31.4 GB) while the exhaustible budget is the commit limit
(50.7 GB with this page file). The line is now the commit limit's 10% headroom and the
2 GiB available-physical floor still refuses genuine physical pressure (`-r4` was such a
refusal and remains correct). All refusal receipts are preserved; nothing was re-scored.

So both identity-side levers reachable from current data are measured and closed: the
loss coefficient (positive on localization, structural cost) and the read-out
tying (negative — identity moves, structure does not hold and the role head collapses).
The fit gate stays FALSE. Open items narrow to the supervision signal (54 queue tokens
and 9 hold cases against ~1,800 supervised cells per pass), the input representation
(pixels per icon), and the owner decision on admitting real screenshots.

So read-out sharing is not the identity lever either: tying supervision into the cell
classifier moves the sparse identity targets but buys nothing readable and costs the
role head. Per the predeclaration the next lever is the input representation (pixels
per icon) or the component targets themselves; the coefficient and read-out questions
are closed. Real-image items 4–5 remain an owner decision (reviewed, permissioned
labels).

## M3 component-identity lever: the identity scale is a measured positive — 2026-11-08

`docs/M3_COMPONENT_IDENTITY.md` predeclares the lever at commit `65ee2f1`, before any
receipt: 128 matched updates at capacity 64 with the queue-token and hold-piece terms
scaled by a declared factor 4, everything else identical to `m3-stream-joint-fit-r1`.
Fresh admission `runs/m3-identity-factor-profile-r1-job.json` (passed, 62.276 s,
952,553,472 B under 1,152 MiB, checkpoint `e1ba82fc...`, catalogue `b3caba97...`), then
`runs/m3-identity-factor-r1-job.json` (passed, 494.121 s outer, 436.42 s of updates,
982,077,440 B peak, 128 steps, checkpoint `acb6a8a9...`).

By the predeclared rule the lever is a measured positive on its localization: exact
queue piece tokens rise 0/54 -> 3/54 on the training probes and 0/90 -> 9/90 on
development, and development hold identity 0/12 -> 3/12, with slot status still exact
(54/54 and 90/90). The fit gate stays FALSE (0/9 and 0/18 boards, 12/18 and 0/36 roles,
0/9 and 0/18 exact readable components, 3.83 and 2.21 cells). The declared structural
regression also shows: development EOS falls 15/15 -> 3/15 and training presence
63/63 -> 54/63, so the scale buys identity at a measurable structural cost. Losses are
not comparable across factors because the scaled terms enter them.

Independent replay `runs/m3-identity-factor-r1-errors-r3-job.json` (27 forwards,
36.492 s, 551,710,720 B, weights `acb6a8a9...` unchanged) reproduces every saved
per-case field and curates in
`docs/evidence/m3-identity-factor-r1-errors-r3/scope.json` (`passed`).

The chain exposed two instrument defects, both fixed at their root (V-038): the replay
scored the published factor-1 objective and so could not reproduce a factor-4 run's
saved losses, and the owned-run-name guard did not list the predeclared
`m3-identity-factor-*` family. Real-image items 4–5 remain unreachable without
reviewed, permissioned labels.

Next: identity targets or representation at this scale (the coefficient question is
closed). The staged head refinement stays a measured negative, and development stays
comparable across stages on the canonical 18 images at 4/18 boards, 0/36 roles,
.901944 cells and 3.530471 geometry with loss 11.0885 versus 11.8129.

## First M3 joint learning is a measured negative — 2026-11-08

Roadmap item 3's open half is measured for the first time. Items 4–5 were checked
first and skipped on purpose: they need admitted, permissioned, independently
reviewed real labels, and the tree holds only 166 real drafts with zero
training/scoring eligibility. Nothing in the repo converts those pixels into
admitted labels, so the preferred real-image chain is an owner decision (review
plus permission), not an executable pass. `docs/M3_JOINT_FIT.md` records it.

The next ranked item repo data supports is the joint queue/hold heads, which had
never been trained. `docs/M3_JOINT_FIT.md` predeclares the study at commit
`90218b3`, before any receipt existed: 128 updates of the accuracy model with
`--component-capacity 64` on `stream-v1`, seed `20261026`, evaluation and
checkpoint every 64, canonical 6 development groups, criterion = 9/9 exact boards,
18/18 exact roles, 9/9 exact readable components and <= 0.25-cell geometry on the
run's own nine training probes.

Fresh capacity-64 admission `runs/m3-stream-joint-fit-profile-r2-job.json`
(passed, 82.759 s, 943,411,200 B under 1,152 MiB, checkpoint `ad14d69a...`,
catalogue `7074e6a6...`, work-item bound 2), then
`runs/m3-stream-joint-fit-r1-job.json` (passed, 417.353 s outer, 360.29 s of
updates, 984,649,728 B peak, 128 steps, checkpoint `b2e44c9c...`).

Result: the criterion is FALSE. Training probes 0/9 exact boards, 12/18 exact
roles (ghost 9/9, active 3/9), .8494 cell accuracy, 4.0417-cell geometry, loss
7.739; development 0/18 boards, 0/36 roles, .8481 cells, 2.0294 geometry, loss
7.332. Against the two-update profile (roles 3/18, geometry 19.82, loss 29.97
training; geometry 16.18, loss 30.56 development) the optimizer clearly trains:
geometry and loss collapse, ghost identity fits. Components do not: queue EOS
9/9, slot status 54/54 and presence 63/63 are exact while piece tokens are 0/54
and hold identity 0/9, so every readable component fails on identity alone. That
is the M2 class-collapse signature localized to the queue-token and hold-type
heads.

Independent replay `runs/m3-stream-joint-fit-r1-errors-r2-job.json` (27 forwards,
35.184 s, 546,545,664 B, weights `b2e44c9c...` unchanged,
`automatic_metrics_reproduced`) is curated in
`docs/evidence/m3-stream-joint-fit-r1-errors-r2/scope.json` (`passed`). Three
launches were refused for authored rather than derived bounds (wall 2,700 s vs the
trainer's (0, 1800]; admission work items 256 vs the required 2; probe work items
64 vs the required 27 forwards) and all receipts are retained; no code, fixture or
gate defect was found.

The staged head refinement remains a measured negative, and development stays
comparable across stages on the canonical 18 images: stage 1 (ordinary replay of
its artifact) and stage 2 both reach 4/18 exact boards, 0/36 exact roles, .901944
cell accuracy and 3.530471-cell geometry, with development loss 11.0885 versus
11.8129. The M3 study does not change those numbers.

Next: component-identity learning is now the narrowest open M3 lever (targets,
loss weighting or a longer fit at the same criterion), each needing its own
predeclaration and admission; varied-shape long memory and review-only runtime
decoding stay open, and the real-image chain still needs admitted labels.

## Staged frozen-backbone head refinement is a measured negative — 2026-11-08

The roadmap's next joint-learning lever is done and closed as a measured negative.
`docs/STAGED_HEAD_REFINEMENT.md` predeclares it at commit `e601182`, and every
staged receipt is later. Stage 2 loads the ordered artifact
`runs/m2-stream-headrate-control-s3` (`1f20007c...`), freezes everything but the
shared type head and starts a fresh AdamW over the same 1,024 ordered pairs.

The fresh native admission for supervisor `d9382fa2...` passed on the exact
staged mode: `runs/staged-head-refine-profile-r3-job.json`, 442,085,376 B peak
under the 671,088,640 B ceiling, 36.287 s, warm backward over exactly 2
parameter tensors, source catalogue `7074e6a6...`. The study then ran as one
interrupted first attempt (`runs/staged-head-refine-interrupted-s1`, outer command
cap at step 716, retained), a 512-step segment stopped by the OS commit guard at
430,661,632 B, a 404-step resume stopped the same way at 429,887,488 B (both
external pressure, not training failure), and `runs/m2-stream-staged-head-r1-s3`,
which restored the step-384 snapshot and finished 640 updates to step 1,024:
`passed`, peak 433,860,608 B, 268.64 s, checkpoint `e02a02e7...`.

Result: exact locked boards 32/32, semantic cells 128/128 and geometry
.21058594 all stay identical to the digit, while exact training roles rise from
20/64 to 36/64 (observed 12/56 to 28/56, absences 8/8). The predeclared 64/64 role
gate fails, and the run stops 26 roles short of the 62/64 copied-head reference, so
head-only refinement recovers about a quarter of that gap and then stalls. That
retires the "copied head is reachable from pixels" hypothesis without proving the
joint forward lacks class information. Development is now comparable on the contract's canonical six-group set: an ordinary
replay of the stage-1 artifact over the same 18 images
(`m2-stream-headrate-control-s3-dev6-errors-r1`, 50 forwards, weights `1f20007c...`
unchanged, every saved metric reproduced, curator in
`docs/evidence/m2-stream-headrate-control-s3-dev6-errors-r1/`) reaches the same
4/18 exact boards, 0/36 roles, .901944 cells and 3.530471 geometry, with stage 2's
loss higher (11.8129 vs 11.0885). V-036 closes the denominator discrepancy; stage
1's published 9-image row (2/9 boards, 0/18 roles) is a saved subset, not a
different result.
Independent evidence: `runs/m2-stream-staged-head-r1-s3-errors-r1/result.json`
(50 forwards, weights unchanged, every saved per-case metric reproduced, 425,635,840 B
/ 20.032 s) and the curator's
`docs/evidence/m2-stream-staged-head-r1-s3-errors-r1/scope.json`.

The chain exposed and the fix log now closes two defects. V-034: the ownership
test for a staged start was written against `ROOT/runs`, but the driver re-executes
from its verified archive at `runs/<name>/code`, so the stage refused its own
source; ownership is now the written run's exact sibling and is still re-read from
source bytes. V-035: the flat exclusion between initial weights and resume blocked
the only valid resumed form, since a staged run must restate the frozen mode and
start artifact that its recovery contract stores. Carried limit: a segment cannot
be resumed after the repository source changes, because the recovery contract
pins the source catalogue by design.

Next: with accumulation, interleaved ordering, type-head rate and head-only
refinement all closed as measured negatives, the next candidate is a
representation/architecture or data-coverage lever predeclared from scratch, and
every new mode needs its own fresh native admission. The real-image and complete
state measurement chain remains the top open quality item, and the M2/M3 gates are
unchanged.

## Accumulation arms replayed, V-030 closed — 2026-11-08

The two completed accumulation arms now have independent ordinary replays and
curators, so their learning outcome is citable rather than trainer self-report:
`docs/evidence/accumulation-arm-balanced-errors-r1/scope.json` (50 forwards,
424,710,144 B, 19.523 s under the unchanged 640 MiB ceiling, weights
`a5e6bc56...`) and `docs/evidence/accumulation-arm-online-errors-r1/scope.json`
(424,316,928 B, 19.202 s, `93e79820...`), both from base commit `64516ad`. Each
reproduces every saved per-case metric exactly and leaves the model tensors
unchanged. Balanced-eight reaches 16/32 exact training boards, 34/64 roles and
.7507 cells geometry; online interleaved reaches 28/32, 37/64 and .3862. Both
fail the predeclared fit, and development stays 4/18 boards (all four empty) and
0/36 roles for both, with development loss rising from 6.6425 to 11.7978 and
13.3888. Batching eight images per update is therefore a measured negative
result; the ordered-eight arm was never run, so no mechanism claim follows.
`BALANCED_ACCUMULATION.md` holds the full tables and limits.

`scripts/run_bounded.py` now resolves the child executable before the Job is
armed and refuses a relative forward-slash path, closing V-030; receipts record
`child_executable`, and a refusal is marked `launch_refused` with no `child_pid`
so it reads as an instrumentation failure rather than a failed gate. The
supervisor hash is now `d9382fa2...` (was `d2fb1478...`), so fresh native
admission is required before any further long training window, and copy-paste
commands must name the interpreter with an absolute or backslash path. The
shared `require_owned_run` policy also admits the accumulation arms to the error
probes and curators (V-033). One 68-case gate passes at 355,688,448 B under
640 MiB: `docs/evidence/v030-launch-preflight-r1/scope.json`.

Next: with the accumulation direction closed as a negative result, predeclare the
next joint-learning lever and its resource/recovery admission. Controller/curator
routing for the accumulation cursors and an in-group interruption at arm scale
stay open but are no longer on the critical path. Every V-entry in the ledger is
now closed on a named receipt; the M2/M3 quality and real-image chain remains
fully open.

## Accumulation integration — 2026-11-08

`scripts/train_m2_stream.py` now executes the predeclared whole-group arms when
`--accumulation-policy` is given: each group is validated and journaled only
after its optimizer step, the stepped gradients are cleared before any boundary,
evaluation fires only at the declared image cadence, and a boundary snapshot is
published only at a complete-group boundary. Image and optimizer cursors stay
separate in `curve.jsonl`, `training.jsonl`, progress and summary, and the
default streaming path is unchanged. Five scoped driver checks pass under448MiB
(`docs/evidence/accumulation-driver-integration-r1/`): accounting/cadence,
per-group-size admission, the profile CLI coupling, a forced in-group
interruption replayed to byte-identical model/Adam/RNG/journal bytes, and the
native profile characterization. 29 accumulation,43 stream and5 type-head
regression tests also pass. The pilot model's
`--profile-square --accumulation-policy` path ran natively for both predeclared
group sizes at640MiB: size1 519,929,856B /33.30s /0.060 updates/s, size8
561,786,880B /29.35s /0.068 updates/s, each with a byte-exact boundary reload
(0.212s and0.086s) and a zero reload delta because the warm peak was already
higher; evidence in `docs/evidence/accumulation-profile-r1/`. Admission now
requires the matching policy, group size, `2*group_size` microsteps and a
positive measured update rate. Both arms then ran natively to their full1,024-image budgets across two bounded
segments each, every segment admitted from the measured two-update profile:
balanced-eight 64+64 updates (175.3s/179.8s in-child, receipts175.944s/180.456s,
peaks565,817,344B and567,971,840B, peak difference+2,154,496B), online-interleaved
512+512 updates (216.1s/289.6s, receipts216.863s/290.393s, peaks565,100,544B and
569,487,360B, peak difference+4,386,816B). The first journal is an exact byte
prefix of the resumed one and the resume pointer matches the published boundary;
measured reload cost0.219s/0.200s. Every published number is re-derived from those
bytes by `docs/evidence/accumulation-arm-r1/audit/audit_arm_evidence.py` (no
mismatch, result in that folder's `audit/audit.json`), which also
records the wall-clock and rate-denominator definitions and the
receipt-admitted headroom (tightest101,601,280B under the671,088,640B ceiling)
(`docs/evidence/accumulation-arm-r1/`). Resume now goes through
`restore_boundary`. A bounded partial segment exits2 with `status: step_partial`,
so its supervisor receipt says `failed` while the summary carries the partial
status. Still open: automated controller/curator routing, an independent final
pixel replay, and an interruption inside a group at arm scale.

## State recovery — 2026-11-08

Recovered an unrecorded change set on local `main` 86f7362 (8 unpushed commits;
three untracked files, no tracked modifications). `scripts/accumulation_checkpoint.py`
is a durable group-boundary bridge: fresh exclusive journal creation, append only
of validated whole groups after a successful optimizer step, boundary-only
model/Adam/RNG publication with no pending gradients, refusal of a mismatched or
wrongly advanced optimizer, and restore into a new directory that excludes every
uncommitted suffix record. Eight no-Torch contract checks pass under128MiB:
`runs/accumulation-checkpoint-pure-r6-job.json`,0.522s/37,761,024B, and one Toy
Torch group-interruption check passes at640MiB/332,423,168B; receipts and logs
are in `docs/evidence/accumulation-checkpoint-contracts-r1/`.
The first Torch attempt was refused with `os_commit_reservation` before any child
launch (`runs/accumulation-checkpoint-torch-r1-job.json`, memory_pressure,125,no
child) and is retained as an admission refusal, not a test result; r2 passed once
pressure cleared. A launch trap is recorded as V-030: a relative
forward-slash child interpreter fails in `CreateProcess`, so bounded children
must use an absolute or backslash path form; the supervisor is unchanged.
The bridge is used by the trainer's accumulation arm only (see the new
checkpoint above): the controller and independent curator still do not route
these cursors, ordinary fixed-fit recipes only; two-update full-square group profiles and native
segment-boundary resume now exist for group sizes1 and8; both1,024-image arms are
complete, and no interruption has been forced inside a group at arm scale. At
that recovery point no training had been admitted and no research job was running.
Next: route the image/update cursors through the controller and independent
curator, wire accumulated resume to the boundary restore, then obtain fresh
native full-square warm/reload profiles per group size and an actual in-group
interruption recovery before any arm of the predeclared comparison. Commit pressure was31.2GB against the31.368GB stop
at review and briefly cleared to1.12GB of headroom, so long training windows stay
blocked. Local commits remain unpushed;
no push without explicit authorization.

## Accumulation journal contracts — 2026-11-07

System commit was 30,986,448,896 B against 31,368,351,744 B stop when a
512 MiB reservation was checked, so no heavy work was relaunched. Added
`scripts/accumulation_journal.py`: validates whole public-recipe groups,
separate image/update cursors, complete-group snapshot positions and exact
optimizer ownership; its execution boundary preflights before calling the
existing group primitive and only returns complete records. Eight no-Torch
checks pass in1.565 s / 41,299,968 B under128 MiB / 30 s / work8; raw scope
`docs/evidence/accumulation-journal-contracts-r1/`. Execution is mocked in
these tests and ordinary fit recipes only are supported; full-square profiles
need explicit handling. The actual trainer, durable journal publisher,
controller and independent curator remain unintegrated; no neural workload or
training admission follows. Continue those changes and fresh native gates when
resource pressure safely permits; never disable the existing guard.

## Balanced accumulation preparation — 2026-11-07

`docs/BALANCED_ACCUMULATION.md` predeclares online-interleaved, balanced-eight
and ordered-eight arms at1,024 images each, explicitly distinguishing their
1,024 versus128 optimizer updates. `scripts/accumulation_contract.py` contains
finite group mapping, boundary validation and a streamed mean-gradient primitive;
seven bounded checks pass, including actual public-label mapping, full-batch
gradient parity, rejected partial groups and toy interruption/recovery parity.
Evidence `docs/evidence/accumulation-contracts-r1/`: 5.806 s / 332,820,480 B,
640 MiB / 30 s / work7. This is not integrated or native admission. Next
integrate image/update accounting, group-boundary checkpoint/replay and
controller/curator routing, including an optimizer-owns-all-parameters check,
then obtain fresh full-square resource and actual
in-group interruption/recovery gates before any study. No job remains active.
An additional ownership-check test run was declined by the OS reservation guard
(r4), then its 512 MiB retry was stopped by OS commit pressure after two tests
(r5). Both raw receipts/logs are preserved in the named evidence. The unverified
ownership edit was reverted byte-for-byte to the seven-test r3 source; do not
claim the extra check passed or relaunch under unchanged pressure.

## Checkpoint forgetting diagnostic completed — 2026-11-07

`docs/TYPE_FORGETTING_DIAGNOSTIC.md` and the source-bound scopes in
`docs/evidence/type-forgetting-{control,candidate,comparison}-r1/` record 17
checkpoints/544 ordinary pixel forwards per arm, independently recomputed.
Control type correctness ranges8..20/64; fourfold candidate stays8/64 at
every sampled checkpoint despite80 correct-to-wrong and80 wrong-to-correct
role transitions across 16 intervals. The control records86/98 transitions.
These are repeated dependent roles, not independent events, and intervals may
hide intermediate reversals. Local final-head class-gradient cosines are not
full joint/Adam gradient evidence. No new real-image or generalization claim.
Both observers, curators and comparator passed; no diagnostic process remains.
Next predeclare and safely admit a matched class-balanced gradient accumulation
test with equivalent example exposure and explicit optimizer budgets; do not
train or alter frozen studies until its design and admission gates pass.

## Type forgetting diagnostic in progress — 2026-11-07

The completed rate comparison remains frozen. A new read-only diagnostic
replays all32 supported training images at initialization and each64-update
checkpoint through1024 (17 checkpoints, 544 forwards per model). It captures
actual final type-head features/logits and supported public targets, verifies
each retained snapshot's tensor and ledger-prefix identity, replays exact
fixture pixels, and checks final outputs against the independent ordinary
report. The separate pure analyzer computes per-class mean final-head CE
gradients, cosine and checkpoint-to-checkpoint type changes; this is not the
complete joint gradient, Adam update, feature-gradient or real-image evidence.
Two mathematical/unknown/refusal tests passed under128 MiB / 30 s / work2.

Control observer planned: `docs/experiments/probe_type_forgetting.py --arm
control --output runs/type-forgetting-control-r1`, 640 MiB / 600 s / 544
forwards, expected below600 MiB, CPU1/cache16 and existing pressure guard.
Receipt/log prefix `runs/type-forgetting-control-r1`; preflight output, log,
receipt names and capture process identity. One heavy job at a time. Candidate
follows under the same bounds only after control terminates. An independent
bounded collector is required before interpreting the outputs.
Control observer passed 166.366 s / 427,859,968 B, 544 forwards and final
independent-report agreement. After verifying process death, candidate observer
uses `docs/experiments/probe_type_forgetting.py --arm candidate --output
runs/type-forgetting-candidate-r1`, same 640 MiB / 600 s / 544 forwards,
expected below600 MiB, CPU1/cache16 and pressure guard. Receipt/log prefix
`runs/type-forgetting-candidate-r1`; preflight all paths.
Candidate passed 170.071 s / 423,342,080 B, 544 forwards and final independent
report agreement; all observer processes gone. Stronger analyzer tests r2
passed. Sequential independent curators planned: `docs/experiments/
record_type_forgetting.py --arm {control,candidate}`, 128 MiB / 30 s /
544 role-image forward records per arm, expected below100 MiB, CPU1 and
pressure guard. Receipt/log prefixes `runs/type-forgetting-{arm}-curation-r1`,
new evidence directories `docs/evidence/type-forgetting-{arm}-r1/`.
Both curators passed; control 42,598,400 B, candidate 42,672,128 B under
128 MiB, with actual checkpoint/logit reconstruction and public-target checks.
Matched class-distribution comparator planned `docs/experiments/
compare_type_forgetting.py`, 128 MiB / 30 s / 1088 image-checkpoint cases,
expected below100 MiB, CPU1/pressure guard, receipt/log prefix
`runs/type-forgetting-comparison-r1`; preflight new evidence path.

## Joint type-head fit completed — 2026-11-07

Both fresh same-source fixed1024 studies and independent ordinary41-image
replays/collectors are complete. `docs/TYPE_HEAD_JOINT_OPTIMIZATION.md` and
`docs/evidence/type-head-joint-comparison-r1/scope.json` supersede the earlier
in-progress entries below. Both joint-fit gates FAIL: control32/32 boards,
20/64 roles,128/128 semantics, geometry.211; candidate32/32 boards,8/64 roles,
128/128 semantics, geometry.108. Development both2/9 boards (both empty),
0/18 roles. Source18ce798e..., actual initial tensors and full1024-row data
ledgers match; control weights1f20007c..., candidate8ec2cfab.... The fourfold
head rate is not selected. One seed, 16 training/3 dev ancestries; no real or
generalization claim. Do not extend either finished study. Next investigate
per-class gradient direction and forgetting on supported fixtures and declare
any new intervention before training. No research job remains active.

## Initial tensor admission — 2026-11-07

The trainer now saves and reads back actual pre-evaluation initial state tensors;
their file identity is checked before completion and by independent curators.
Focused tests passed. Candidate gate r9 is planned with explicit output/receipt/
log preflight and free-disk check: parent 1024 MiB / 600 s / 14 updates,
child 640 MiB / 90 s, CPU1/cache16 and existing pressure guards, expected
under 800 MiB. Command `scripts/probe_stream_recovery.py --output
runs/m2-stream-recovery-gate-r9 --curriculum all-class-fit-v1
--type-head-rate-factor 4`; receipt/log prefix `runs/m2-stream-recovery-gate-r9`.
It performs profile2 then admitted continuous6 and partial3+resume3, not the
1,024-update study. Control r10 requires a separate same-source gate.
r9 passed 147.947 s / 599,404,544 bytes; terminal receipt and all matching
processes must be checked before control. Control r10 uses the same limits,
ordered curriculum, factor1 and namespace/disk preflight, expected under
800 MiB. Receipt, output and logs use `runs/m2-stream-recovery-gate-r10`.
r10 passed 142.762 s / 600,817,664 bytes, pending process-death checks.
Curate r9 and r10 sequentially at 128 MiB / 30 s / 181 items, expected under
100 MiB with CPU1 and existing guards; use new `-curation` receipt/log prefixes
and preflight them. Then compare actual initial tensors with a separate 640 MiB /
30 s / two-snapshot collector, expected under 400 MiB. No full fit yet.
r9/r10 curators passed, each replaying 17 records and 164 saved probes. The
matched r2 collector verified actual initial tensor bytes identical, initial
probe metrics, first loss/gradient norm, six data recipes and optimizer groups;
2.259 s / 257,298,432 bytes under 640 MiB. Source catalogue 18ce798e...;
evidence `docs/evidence/type-head-initial-admission-r2/` and gate r9/r10 scopes.
The finite recovery and start-comparison admissions pass; full-study initial
artifacts still need comparison to these starts before final claims.

Control study first segment planned: `scripts/run_stream_segment.py --name
m2-stream-headrate-control-s1 --target-steps 1024 --updates 128 --admission
runs/m2-stream-recovery-gate-r10/profile`. Plan-only receipt/log
`runs/type-head-control-plan-r1*` passes and binds catalogue18ce798e.... Parent
cap 1280 MiB / 2100 s / work297 (128 training updates plus 169 curation
items), child 640 MiB / 1860 s / 128 updates, serialized curator 128 MiB /
120 s / 169 items; expected parent under 800 MiB, CPU1/cache16 and existing
pressure guard. Parent receipt/log prefix `runs/type-head-control-s1-controller`;
child receipt/logs use the run name, and controller progress is in its pipeline
directory. Verify all output/receipt/log paths absent, >40 GB free and exact
startup/death process command lines. This segment is partial128/1024, not a
fit verdict. Afterward compare its actual initial tensors to the admitted
control, and conduct the matched candidate segment with its own profile.
Control s1 completed128/1024, explicit `verified_partial`, 140.772 s /
600,555,520 bytes, exact saved initial file SHA256
51c58ee18ca9df6de3f03ba5430251ed831b662da360e0446b2eeced7e1e2aa8 matches
the admitted control. No matching processes remain. Candidate s1 plan-only
receipt/log `runs/type-head-candidate-plan-r1*` confirms factor4 and same
source/config/seed/recipe. Planned command `scripts/run_stream_segment.py
--name m2-stream-headrate-candidate-s1 --target-steps 1024 --updates 128
--admission runs/m2-stream-recovery-gate-r9/profile`, parent 1280 MiB /
2100 s / work297, child 640 MiB / 1860 s / 128 updates, curator 128 MiB /
120 s / 169 items. Expected parent under 800 MiB; CPU1/cache16, kernel and
pressure guards unchanged. Preflight every run/receipt/log/evidence path and
>40 GB free. Parent receipt/log prefix `runs/type-head-candidate-s1-controller`.
Candidate s1 verified partial128/1024, parent 136.545 s / 602,480,640 B;
no matching processes remain. Both actual initial files have SHA256 51c58ee1...
and their complete 128-row data ledgers have SHA256 9e4d4248...; independent
segment scopes are `docs/evidence/m2-stream-headrate-{control,candidate}-s1/`.
No accuracy verdict follows. Next control s2 plan-only passes: resume s1 at
128, cap448 new updates, target1024, factor1 and identical source identity.
Command `scripts/run_stream_segment.py --name m2-stream-headrate-control-s2
--target-steps 1024 --updates 448 --admission runs/m2-stream-recovery-gate-r10/profile
--resume-from runs/m2-stream-headrate-control-s1`. Parent 1280 MiB / 2100 s /
work1065 (448 updates plus 617 curation items); training child 640 MiB / 1860 s /
448 updates; curator 128 MiB / 120 s / 617 items. Expected parent under 800 MiB,
CPU1/cache16 and pressure guard unchanged. Preflight all paths and >40 GB free;
parent receipt/log prefix `runs/type-head-control-s2-controller`. One heavy job.
Control s2 verified partial576/1024, 414.573 s / 601,214,976 bytes; process
tree gone and checkpoint at576. Candidate s2 plan-only passed; resume its s1
at128 for448 updates to576 with factor4 and same frozen source/target. Planned
`scripts/run_stream_segment.py --name m2-stream-headrate-candidate-s2
--target-steps 1024 --updates 448 --admission runs/m2-stream-recovery-gate-r9/profile
--resume-from runs/m2-stream-headrate-candidate-s1`. Parent 1280 MiB / 2100 s /
work1065; child 640 MiB / 1860 s / 448 updates; curator 128 MiB / 120 s /
617 items; expected parent under800 MiB, CPU1/cache16 and pressure guard.
Preflight all paths and >40 GB free. Parent receipt/log prefix
`runs/type-head-candidate-s2-controller`. No fit verdict at576.
Candidate s2 verified partial576/1024, 394.045 s / 601,661,440 bytes;
process tree gone. Control/candidate training ledgers at576 and initial state
files hash identically. Control s3 plan-only passes: resume576, 448 updates
to fixed1024. Planned `scripts/run_stream_segment.py --name
m2-stream-headrate-control-s3 --target-steps 1024 --updates 448 --admission
runs/m2-stream-recovery-gate-r10/profile --resume-from
runs/m2-stream-headrate-control-s2`. Parent 1280 MiB / 2100 s / work1513
(448 updates + 1065 replay items); child 640 MiB / 1860 s / 448 updates,
curator 128 MiB / 120 s / 1065 items. Expected parent under800 MiB, CPU1/
cache16, unchanged pressure guard. Preflight all paths and >40 GB free;
parent receipt/log prefix `runs/type-head-control-s3-controller`. Final
ordinary pixel replay remains separate and mandatory.
Control s3 completed1024/1024 and native curator passed, 401.226 s /
605,925,376 bytes. Final weights 1f20007c... match the historical ordered
baseline, but new source/evidence scope remains distinct. All matching
processes gone; independent ordinary pixel replay remains pending.
Candidate s3 plan-only passes: resume576, 448 updates to fixed1024. Planned
`scripts/run_stream_segment.py --name m2-stream-headrate-candidate-s3
--target-steps 1024 --updates 448 --admission runs/m2-stream-recovery-gate-r9/profile
--resume-from runs/m2-stream-headrate-candidate-s2`. Parent 1280 MiB / 2100 s /
work1513; child 640 MiB / 1860 s / 448 updates; curator 128 MiB / 120 s /
1065 items; expected parent under800 MiB, CPU1/cache16 and pressure guard.
Preflight all paths and >40 GB free; parent receipt/log prefix
`runs/type-head-candidate-s3-controller`. Independent final pixel replay and
matched comparison remain required for both studies.
Candidate s3 completed1024/1024, curator passed, 398.877 s / 606,023,680 B;
all matching processes gone. Both complete training ledgers SHA256
91c5bb06c1b209477c2c0342ba4c27fd69cbb93ef49a62b9bf8bdda01509ae5c.
Independent control pixel replay planned: `scripts/probe_stream_errors.py
--run m2-stream-headrate-control-s3 --output runs/m2-stream-headrate-control-errors-r1`,
640 MiB / 120 s / 41 forwards, expected below600 MiB, CPU1/cache16 and
pressure guard. Receipt/logs prefix `runs/m2-stream-headrate-control-errors-r1`;
check output, receipts and logs absent first. Candidate follows only after
control process death, with its own analogous output. Separate 128 MiB /
30 s / 41-case collectors and final matched comparison remain mandatory.
Control replay passed 15.742 s / 425,357,312 B under640 MiB. After confirming
its process tree gone, candidate replay uses `scripts/probe_stream_errors.py
--run m2-stream-headrate-candidate-s3 --output
runs/m2-stream-headrate-candidate-errors-r1`, same640 MiB / 120 s / 41 forwards,
expected under600 MiB, CPU1/cache16 and pressure guard; receipt/log prefix
`runs/m2-stream-headrate-candidate-errors-r1`. Preflight all paths.
Candidate pixel replay passed 15.071 s / 424,677,376 B; both process trees
gone. Curators run sequentially `scripts/record_stream_errors.py --name
m2-stream-headrate-{control,candidate}-errors-r1`, 128 MiB / 30 s / 41 cases,
expected under100 MiB, CPU1/pressure guard; receipt/log prefixes
`runs/type-head-{control,candidate}-errors-curation-r1`. Preflight names.
The final comparison remains pending their independent scopes.
Both ordinary replay curators passed (control 78,360,576 B; candidate
77,758,464 B under128 MiB). Final matched comparator planned:
`docs/experiments/verify_type_head_finish.py`, 128 MiB / 30 s / 2130 items
(two1024-row ledgers + 82 pixel cases), expected below100 MiB, CPU1/pressure
guard. Receipt/log prefix `runs/type-head-joint-comparison-r1`; output
`docs/evidence/type-head-joint-comparison-r1/`, preflight all paths.

## Joint type-head optimization preparation — 2026-11-07

The predeclared comparison is `docs/TYPE_HEAD_JOINT_OPTIMIZATION.md`. An opt-in
fourfold type-head optimizer factor is implemented with default single-group
behavior preserved, profile/recovery identity bound, and segment routing guarded.
Focused optimizer/admission/routing checks passed; these do not admit training.

Candidate recovery admission r6 planned: `scripts/probe_stream_recovery.py
--output runs/m2-stream-recovery-gate-r6 --curriculum all-class-fit-v1
--type-head-rate-factor 4`, parent Windows Job 1024 MiB / 600 s / 14 updates,
serialized children each 640 MiB / 90 s, CPU1/cache16 and unchanged OS
pressure guard. Expected aggregate under 800 MiB from prior pilot gate; first
two updates are the full-square warm profile, followed only if admitted by
continuous6 and partial3+resume3. Receipt `runs/m2-stream-recovery-gate-r6-job.json`,
logs same prefix `.log`/`.stderr.log`, progress in the output directory. More
than 40 GB free was checked. This gate does not start the 1,024-update study.
Correction: r6 already existed. The runner refused its existing receipt before
work; redirect opened historical local log names. Failed-attempt copies are
`runs/type-head-recovery-name-collision{,.stderr}.log`; historical local logs
were restored byte-for-byte from `docs/evidence/m2-stream-recovery-gate-r6/`.
See V-029. Use r7 only after checking all output, receipt and log names absent;
same command, limits, output/log/receipt prefix, and work cap as above.
r7 passed 57.769 s / 602,124,288 bytes, exact continuous6 == partial3+resume3,
profile child peak 558,534,656 bytes under 640 MiB; all matching processes
gone. Independent curator planned under 128 MiB / 30 s / 181 replay items,
expected under 100 MiB, same CPU/pressure guards. Receipt/log prefix
`runs/m2-stream-recovery-gate-r7-curation`.
r7 curator passed 2.059 s / 87,764,992 bytes; replayed 17 records and 164
saved probes, evidence `docs/evidence/m2-stream-recovery-gate-r7/`. Control r8
planned using the same gate, ordered curriculum and factor1, parent 1024 MiB /
600 s / 14 updates, serialized child 640 MiB / 90 s, expected under 800 MiB,
CPU1/cache16 and unchanged pressure guard. Preflight output, receipt and both
log paths; more than 40 GB free must remain. This is not the 1,024-update fit.
r8 passed 118.063 s / 600,768,512 bytes; matching processes terminated.
Control curator planned under 128 MiB / 30 s / 181 items, expected under
100 MiB with the same guard, receipt/log prefix
`runs/m2-stream-recovery-gate-r8-curation`. Match comparator planned at
640 MiB / 30 s / two snapshots, expected under 400 MiB, after curation.
r8 curator passed 2.414 s / 88,158,208 bytes; 17 records/164 images replayed.
Matched first-step comparator passed 2.219 s / 256,782,336 bytes; initial
probe metrics and first loss/gradient norm match, non-head first-step weights
match and group rates are .001 versus .004 for the candidate head. Exact
initial tensor bytes were not recorded. The predeclared identical-initial-
weights criterion remains open; add source-bound capture/verification and
repeat affected admissions before launching any 1,024-update study. Receipts:
`docs/evidence/type-head-admission-contracts-r1/`. No learning claim follows.

## Pixel replay work in progress — 2026-11-07

The conditional pixel replay and independent collectors are complete; see
`docs/CONDITIONAL_PIXEL_REPLAY.md`. Both joint-fit gates fail and both development
type scores remain 0/18. The next research action is to predeclare a controlled
joint-optimization comparison, including resource and recovery admission; no
new training is admitted yet. `scripts/pixel_replay_contract.py`
adds a strict byte comparison of every non-type output at each stage and final
output, with only the designated stage type logits exempted. Three focused
checks pass under the bounded CPU runner; receipt and log are in
`docs/evidence/pixel-replay-contract-r2/` (peak 246,312,960 bytes, 640 MiB cap,
30 seconds, work 3). This is a comparison instrument, not a pixel replay or
model-quality result on its own. The original runs remain frozen.

Ordered replay attempt r1 planned: `docs/experiments/conditional_pixel_replay.py
--study ordered --output runs/conditional-pixel-ordered-r1`, supervised by
`scripts/run_bounded.py` at 640 MiB / 120 s / 82 forwards; expected under
600 MiB, CPU1/cache16, Windows Job and existing pressure guard. Receipt
`runs/conditional-pixel-ordered-r1-job.json`, stdout/stderr same prefix with
`.log`/`.stderr.log`. This is an experimental attempt, not an audited result.
Ordered r1 terminated passed at 12.606 s / 426,795,008 bytes; all 82 forwards,
saved feature and original-report checks pass; independent audit remains open.
All startup processes 8856/23544/8412/17196 are gone. Interleaved r1 planned
with the same guard, 640 MiB / 120 s / 82 forwards, expected under 600 MiB:
`docs/experiments/conditional_pixel_replay.py --study interleaved --output
runs/conditional-pixel-interleaved-r1`, receipt and logs under that prefix.
Interleaved r1 passed 11.734 s / 426,926,080 bytes; startup processes
2808/22148/12384/4840 gone. Independent collectors planned sequentially:
`docs/experiments/record_conditional_pixel_replay.py --study STUDY`, 41 cases,
256 MiB / 30 s, expected under 150 MiB, CPU1 and unchanged pressure guard.
Receipts and logs use `runs/conditional-pixel-STUDY-curation-r1` prefix.
Both collectors passed, with raw evidence under `docs/evidence/conditional-pixel-
{ordered,interleaved}-r1/`. All known replay and collector processes exited.

## Owner request and delivery

The owner asked for a clean project, a repository under `sieradni`, a ZIP to send
and a document of the core goals. This is the active delivery task. No further
research training is selected during handoff. No research jobs are running.

- Repository: https://github.com/sieradni/tetris-state-vision (PRIVATE; owner/account verified).
- Core brief: docs/CORE_GOALS.md. Contributor workflow/setup: CONTRIBUTING.md.
- The incoming contributor's next research task is conditional pixel replay below.
- GitHub holds source/history; the ZIP is a snapshot of the same source commit
  plus current ignored model/run artifacts. ZIP extraction alone creates no remote.
- No collaborator was invited because their GitHub username was not supplied.
  The owner must grant access before that person can clone/push a branch.
- Original screenshots, private contact sheets, `.venv`, `.git`, the neighboring
  engine and unselected historical runs are excluded. Original local data remains.
- Bundle contains current image runs, required warm admissions, fitted heads,
  saved synthetic probes, exact frozen sources and one latest optimizer generation
  per selected run. The archive manifest is the exact file/commit inventory.

Local research results were committed at64939f2, then the contributor tooling/docs
at e3312ef. Six transport tests passed:7.743s(native8.231s/120,905,728B under128MiB).
They verify integrity,latest snapshot selection,non-destructive/idempotent restore,
source preservation,conflict preflight and unsafe/duplicate members. Source-bound
raw evidence:docs/evidence/handoff-contracts-r1/. This is not a whole-suite gate.
Actual artifact preflight passed .736s/74,731,520B:925 artifact files/313,244,976B,
19 explicit roots. Counts are preliminary;final archive manifest is authoritative.
Private repository creation passed6.059s/138,231,808B under256MiB/90s. Raw receipts,
logs,startup/death and verified owner/visibility:docs/evidence/handoff-preflight/.

Export verification is intentionally outside the source commit to avoid a
self-referential archive hash. Durable owner-side jobs/results are in ignored
runs/handoff*/ and runs/handoff/. Final ZIP,SHA256 and adjacent verification JSON
are delivered in Downloads. They name the exact exported Git commit. The final
acceptance is local main==origin/main==GitHub main==ZIP source commit,clean source,
verified every archive file,restoration into a fresh GitHub clone,latest optimizer
payload hashes,relocated original-model pixel smoke for all3models and copied-head
vector replay. The pinned existing runtime is reused;no fresh install or full
conditional pixel intervention is claimed by that transport check.

Before each export job: state expected RAM/cap/guard,verify actual processes,
record receipt/log/command and inspect terminal status. No force pushes or
history rewrites. Local main is advanced only by a normal fast-forward merge.
Current one-off publisher:runs/handoff/export_remote.py (no credential output).
Push:expected<300MiB,512MiB/180s/work1,unchangedpressureguard:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 512 --wall-seconds 180
--work-items 1 --receipt runs/handoff-push-r1-job.json --
.venv/Scripts/python.exe runs/handoff/export_remote.py push
Logs sameprefix.log/.stderr.log;result runs/handoff/push-result.json.
Build:expected<200MiB,256MiB/180s/work12000:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 256 --wall-seconds 180
--work-items 12000 --receipt runs/handoff-build-r1-job.json --
.venv/Scripts/python.exe scripts/handoff_bundle.py build --zip ZIP_IN_DOWNLOADS
Logs sameprefix.log/.stderr.log;ZIP plus SHA256 and native receipt are sentinels.
Relocation:expected<800MiB,1024MiB/240s/work12000:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 1024 --wall-seconds 240
--work-items 12000 --receipt runs/handoff-verify-r1-job.json --
.venv/Scripts/python.exe scripts/verify_handoff_checkout.py --zip ZIP_IN_DOWNLOADS
--checkout runs/handoff-check-COMMIT_PREFIX
Logs sameprefix.log/.stderr.log;adjacent *.verification.json is result sentinel.
Do not launch duplicate export operations because a read/poll timed out.
Exact post-commit export paths/command/PID/log/limits/results are recorded before
launch in ignored runs/handoff/export-state.json. This fixed source checkpoint
points there so the delivered commit need not contain its own later ZIP checksum.
Before final source commit:local verification-script check,expected<800MiB,
1024MiB/60s/work167(3ordinarypixels+2x82headvectors),no training/sourcewrites.
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 1024 --wall-seconds 60
--work-items 167 --receipt runs/handoff-local-smoke-r1-job.json --
.venv/Scripts/python.exe runs/handoff/local_smoke.py
Hiddenlogs sameprefix.log/.stderr.log;native receipt terminal sentinel.
Local verifier smoke PASS12.893s/525,479,936B under1GiB/60s/work167:all3frozen
model origins and original pixel logits exact,both82-row fitted-head replays
exact. Actualsource/rawreceipt/log/startup/death in docs/evidence/handoff-local-smoke-r1/.
This verifies the program locally;full fresh-clone ZIP verification is separate.

## Incoming contributor: selected next experiment

Read AGENTS.md,docs/CORE_GOALS.md,docs/ROADMAP.md,docs/TYPE_HEAD_FIT.md and BUG_LOG.md.
There is no validated complete screenshot parser. The original4096baseline and
both1024controlled image studies remain complete/frozen and fail adequate/joint
learning. API remains needs_review/unsupported. The full release goal stays open.

Implement conditional pixel replay for the ordered and interleaved copied raw
float32 heads. No additional image-model training or vector-fit extension is
selected. Acceptance:

1. Verify curated model/feature/headfit source,config,weights,payload and hashes.
2. Replay all41saved rasters/public targets per model;original ordinary metrics
   and per-image type-head inputs/logits must match independent saved reports.
3. Replace only layer_type weights/bias in a MEMORY COPY. Pixel/validity only
   to model.forward. Exactly2forwards/image=82/model;no oracle geometry/labels.
4. Require every non-type tensor at EVERY stage to remain bitwise equal and
   final head features to remain exact. Record honest float32 matrix-product
   roundoff versus vector-batch logits;actual class counts must match headfit.
5. Restore original head in finally/percase,verify full model state/source/file
   identity. Never overwrite original model/checkpoint/optimizer/data.
6. Independently derive/audit type,pose,absence,locked,geometry and joint-fit
   metrics from the actual changed pixel outputs. Keep the intervention separate
   from original ordinary reports and historical study tags.

Expected inference<600MiB based on measured425MBobservers;enforce640MiB/120s/
82forwards,CPU1/cache16,existing pressure guard. Declare/bound an independent
collector separately. Implementation and those conditional runs are still pending.
Transport smoke performs just1ORIGINAL pixel forward/model;it does not meet this gate.

## Latest verified findings and artifact map

TYPE_HEAD_FIT.md: both finite copied-head fits complete at1024updates with per-bank
native warm/reload admission and independent collectors. Original image models,
source and optimizer files unchanged. New diagnostic optimizer is explicitly reset;
fullbatch64known vectors,AdamWLR.01/decay0,training-onlymean/popstd floor1e-6,
float64 conditioning,converted shared8x64+bias rawfloat32 head. Initial per-image
float32 logits exact;first2fit records match warm prefix;head/Adam reload exact.

- Ordered types20/64->62/64(active32,ghost30):64/64 criterionFAILS. Remaining two
  errors:T->S,both views of all-class-04(one ancestry). Dev0/18;CE3.279->58.590.
- Interleaved types37/64->64/64(active32,ghost32):vector criterionPASSES.
  Dev0/18;CE4.933->25.486. Rawfloat64/float32 counts agree.
- This proves representation on these saved training vectors,not image reading
  or generalization,and does not identify the cause of joint-training failure.

Evidence/runtime names:type-head-{ordered,interleaved}-{profile,fit}-r1.
Profiles:2warm+continuousnext1+reloadednext1(work4),640MiB/60s.
Fits:1024updates,640MiB/120s. Peak333MBboth;native~3s;independent collectors
~.35-.4s/78-80MB under128MiB. All known PIDs/commands gone;raw source/payload/log/
receipt/startup/death scopes in docs/evidence/<name>/. Six pure contracts in
handoff source:docs/evidence/type-head-contracts-r1/. No new defects;V001..V028
closed on named receipts. Do not attach those green scopes to changed sources.

Ordered model:runs/m2-stream-allclass1024-s3;interleaved:runs/m2-stream-interleave1024-s3.
Fitted payloads:runs/type-head-{ordered,interleaved}-fit-r1/head-state.pt.
Raw32values are in result.json/head.weight_raw32,bias_raw32;rederive from the
normalized payload+mean/scale and verify. Curated inputs_sha256 carries model,
feature and source identities;do not hand-rewrite hashes/absolute historical paths.
Payload SHA ordered6749fa5c64c5f6e5dce9fd820c9f1b94854ed2bc14abea931b10fb79545fc696;
interleaved58b417a8c50e434462c5d728c2828b564b33d74c3bdf98391cb7d178e66f86e9.

INTERLEAVED_FIT.md original image studies:ordered32/32boards,12/56visible roles,
8/8absences,geometry.21059;interleaved28/32boards,30/56visible roles,7/8absences,
geometry.38618. BothFAIL joint32boards/64roles/128semantics/geometry<.25.
Devboth2/9boards(the2empty),0/18roles,only3ancestries. One seed,16trainingancestries,
lockedonlyempty/fixed8cellrow. Do not count views as independent accuracy samples.

ACCURACY4096_BASELINE.md original larger7.69Mmodel complete and frozen. Final
ordinary27-image replay/curator pass;dev0/18boards,0/36roles,369/401positivecells,
32misses,813inventedcells,geometry2.01544. All9correct training-probe ghosts are
absence. Run/source/optimizer integrity is not adequate learning. Current model
runs/m2-stream-accuracy4096-s10,original source121fd05a... and weights06751638....

LAYER_FEATURE_DIAGNOSTIC.md actual inputs/logits and unchanged ordinary outputs
verified for all3models. Centroid failure did not prove absent class information;
subsequent copied-head fits above are more informative. Original raw references,
source catalogs,summary/probe files and current model files are in the handoff.
Older unselected tensors/run folders remain local;Git preserves curated evidence.

## Safety and full scope

Windows kernel Job/kill-on-close,CPU1/cache16,GetPerformanceInfo/.2spressure guard
remain armed. Never disable/raise machine ceilings;before detached work record
RAM/wall/work caps and expectedpeak,verify actual startup/termination and keep logs.
Oneheavy job at a time;>40GBfree beforelongtraining. Reference machine31.2GBRAM,
stopcommit31,368,351,744B. NewhostcrashrequiresSystem41/1001anddump audit first.
Do not delete user run data,closeapps,change power or touch neighboring engine/stage.
Historical pressure stops/restarts retain their evidence;no crash was observed
in this handoff. Read RESOURCE_GUARD.md and TRAINING_RECOVERY.md.

The166real screenshots are unreviewed and excluded. Broad rendering/occlusion/
association,reviewedreal/independentsplits,adequateM2/M3jointlearning,scratch/
pretrained/directdecoder comparison,calibration/export/CPU/adapters and>=95%
readable-real exactcomplete-state remainOPEN. See CORE_GOALS.md for acceptance.
No claim that the research goal or release gate is complete.

## GPU fit test (admitted, in flight)

Training moved off the CPU wall: `.venv-cuda` carries `torch 2.14.1+cu126` and runs on the
RTX 4060 Laptop (capability 8.9, 8,585,216,000 B). The shipped and tested path stays on
`.venv` (`2.14.1+cpu`, `cuda_available False`); a CPU forward on the published checkpoint and
the full 784-test suite both pass there, so the CPU contract is unchanged.

Instrument: `runs/gpu_fit_test.py` (working tree, uncommitted) replays the campaign recipe with a
fixed pool, batch 4 and `joint_loss(endpoint_term="length", occupancy_factor=4,
positive_weight=4)`, and initialises the occupancy bias to logit(0.1354) so logit 0.0 is the
measured positive-cell prevalence's own 0.5 point instead of an arbitrary shift.

Measured on this machine: 0.383 s/step logged (async launch), ~1.1 s/step effective wall,
1.84 GB allocated / 2.35 GB reserved. `--memory-mib 6144` host ceiling, 6 GiB declared
accelerator envelope asserted from the measured peak.

Validated arm (`docs/evidence/m3-gpu-fit-r1/smoke-400.json`, 400 steps, pool 256, batch 4):
train exact 0/28 nonempty + 0/8 empty at 0.0, F1 0.5722, best 2.7 -> F1 0.6634 with 0/28 exact,
geometry 0.7791 cells (spread 1.4264); development exact 0/14 + 0/4, F1 0.6338, best 3.0 ->
F1 0.7349 with 0/14 exact, geometry 1.0572 cells (spread 1.3084). Against the 384-step budget arm
(F1 0.2808, best 0.5459, placement 1.8419, 0 exact) the fixed pool and corrected prior move
probability and geometry a long way and exactness not at all.

The 20,000-step chain (`runs/drive_gpu_fit.sh`) reached 12,800 steps in 17 resumable segments and was
then stopped on its own measurement, not on a resource or plumbing failure: the frozen gate peaks at
1,000 steps and degrades afterwards (`docs/evidence/m3-gpu-fit-r1/chain-12800.md`). Development F1
0.8044 at best threshold with 1.17-cell geometry at 1,000 steps; 0.6373 with 4.52-cell geometry at
12,600; loss rises from ~7 to ~15 over the same span and zero nonempty boards are exact at every
checkpoint measured (400, 1,000, 12,600, 12,800). The binding constraint is therefore not step budget.
The partial chain is recorded as `chain-12800.json`; no 20,000-step arm is assembled from these
checkpoints because the later ones are measurably worse.

Fixed on the way: `components.py` built its length-distribution reference tensor on CPU, which
fails on any accelerator run (now on `distribution.device`); `cudnn.benchmark` was disabled
because padded batch shapes retriggered autotuning; the per-process CUDA fraction was removed
in favour of a measured-peak assertion.

## Stability arms: the collapse is the optimiser stepping at a fixed size

Two 1,000-step-new arms came off the 1,000-step checkpoint on the cached 512-frame pool
(`--pool-cache`, `pool_origin` cache, `skipped_batches` all zero, 0.44 s/step on an idle GPU).

The pool is not a held-out set: the 36 training gate boards are pixel-identical to pool frames
0..35 (max absolute pixel difference 0.0), so the only unseen boards are the 18 development ones,
and they score better than the seen ones (0.8044 vs 0.6611 at best threshold, geometry 1.0572 vs
1.0985). Memorisation is excluded.

Every logged step is clip-saturated at 1.0 (plain arm gradient-norm median 35.86, max 109.17,
100% clipped), so each update has magnitude lr and cannot shrink near an optimum: loss stays flat
(4.9-8.8) while the swept best threshold wanders 4.5 logits and development F1 at logit 0.0 falls
0.7362 -> 0.5636 over 300 steps. Cosine decay 1e-3 -> 1e-4 reaches 0.8337 at step 1,150 (above the
0.8044 peak) and ends at 0.7919 with 0.5983-cell geometry, against 0.6962 / 0.933 cells for the
plain arm. Exactness remains zero nonempty boards in both arms, so no board is exact yet.

Evidence: `docs/evidence/m3-gpu-fit-r1/stability-1600.md` and `stability-1600.json`; defect record
V-085. CPU suite after the change: OK: 784 tests in 45.727 s, exit 0, run under scripts/run_bounded.py (receipt runs/stability-suite-r2-job.json, peak job commit 570,347,520 B)

## Independent judgement on the guard chain (2026-11-11)

A falsification pass built scratch copies of every published receipt and ran the real checks on them. All six
corruption classes are rejected with a named reason: the three instrument receipts through
`scripts/replay_equivalence.py` (changed carrier, dropped case, duplicated case, retyped scalar, nested edit,
injected section), the deciding-power record through `sweep_deciding_power.record_problems`, and the green suite
receipt through `record_stream_run.job_receipt_problems`. Irrelevant edits pass: prose whitespace (the figure
match is whitespace-insensitive), reordered map keys, an added docstring, a changed log line and a renamed local.
Two real gaps were closed - the suite receipt had no validator at all, and the power record accepted injected
sections - and two pins were loosened because they were brittle rather than protective: the documented figure no
longer requires exact whitespace, and the record no longer pins its own script digest, which a docstring or a
renamed local would otherwise fail even though recomputation already catches behavioural change.

Worth the cost? On the corruption side, yes. The guards are one strict comparer, one record validator and one
receipt validator over receipts that already exist; they add seconds to a suite that already runs in about a
minute, and the alternative is a wrong attribution claim that no test can distinguish from a right one. The
maintenance cost is the carrier tables: every new receipt section must get a carrier or an explicit metadata
entry. The likeliest breakage a year from now is someone adding a section and silencing `unaccounted:<key>` by
adding it to the metadata allowlist, which quietly shrinks protection rather than failing. The second is path
coupling: the pins name evidence directories, so archiving or renaming evidence breaks tests although nothing is
wrong.

The remaining weakness of the deciding-power claim is not its point estimate but its independence assumption.
AUC 0.8255 against the 0.90 bar and best F1 0.4237 against 0.70 are far from the bars, and the cell-level interval
is +/-0.0197. But that interval treats 7,200 cells from 36 boards in 12 ancestry groups as independent. At the chosen
threshold the measured ICC(1) is 0.049 at ancestry-group level (600 cells per group), a design effect of 30.44,
and 0.047 at board level (design effect 10.33); the cluster-respecting half-width is +/-0.1086 at group level.
Once clustering is respected the 0.90 bar is no longer excluded, so the claim holds only under the independent-cell assumption.

Closing the gap at the current design would need four times the cells for +/-0.010: 144 images in 48 groups under independence, or about 4384 images in 1461 groups at the measured design effect (roughly 122x the present sweep). That is not feasible at 12-group granularity: the instrument needs a group-level evaluation (group-mean AUC with a bootstrap over groups) over about 1461 independent groups, which this pass does not build.

M29 closes the question M28 opened. On the same fixed 2,986-cell population the pixel-query
read-out's collapse is **capacity**, not placement, not scale and not amplitude: the committed form-1
projection reaches the logit at `0.703541` of the head's gain (starved is false), centring
the inputs moves `-0.012545` and adding neighbourhood context
`+0.036428`, while the same statistics through a 64-wide hidden
layer reach `0.610101` best swept F1 and `0.946604` AUC -
`+0.184398` over form 1 and above the `0.543974` reference
(`docs/evidence/m29-pixel-form-r2/scope.json`, M29_PIXEL_QUERY_FORM.md, ROADMAP M29). Span closure is
`+0.149202`, placement `-0.202732`,
representation `+1.053530`, summing to the `0.443205`
span by construction. The predeclared gates still fail on the operating point only (F1 at `0.0`
`0.291464` against `0.70`; best `0.610101` against
`0.70`; attribution passes at `+0.390475`), so the lever is not
cleared and the next lever targets the threshold the read lands on, not its inputs.
M28's own record stands as measured: form 1 at `0.425703` best swept F1, per-arm 3,316-cell read kept
as the labelled secondary (V-150), reference provenance pinned in
`docs/evidence/m28-reference-provenance-r1/scope.json` (V-151).

M30 attacked that operating point with three predeclared label-free level mechanisms, each identity at
step zero, on the same fixed 2,986 cells. A learned scale and bias is excluded
(`-0.005750` F1 at `0.0`, learned scale `0.995428`),
a data-derived class budget is excluded and negative (`-0.015693`), and a
gated board standardisation is confirmed at `+0.081417`: F1 at `0.0`
`0.372881`, AUC `0.954529`, best `0.689775`
(`docs/evidence/m30-operating-point-r1/scope.json`, M30_OPERATING_POINT.md, ROADMAP M30). Span closure
`+0.145801`, placement `-0.114151`, representation
`+0.411555`, summing to `0.443205` by construction; attribution over the
control `+0.470149`. Gates: per-cell `0.372881` and best
`0.689775` fail, attribution passes, so `cleared` is false and the
cause is `level-inconclusive:calibrate`. The measured limit is the gate's single global `0.0` boundary on a read whose
correct level is per-board - a metric question, not a capacity or ordering question.

M31 settles what M30 left open. On the same frozen arm and fixed 2,986 cells the unchanged global `0.0`
gate scores `0.372881` F1 at `0.0` and `0.689775` best swept F1,
the campaign's declared **label-free** per-board `otsu_board` rule scores
`0.278699` / `0.581359`, and a label-fitted per-board
boundary - oracle only, excluded from every verdict - reaches `0.763916`
(`docs/evidence/m31-decision-rule-r2/scope.json`, M31_DECISION_RULE.md, ROADMAP M31). The label-free rule
loses, so the branch is representation; the oracle says the logits carry almost enough, so the limit is the
shape of the read's score distribution rather than its information. The declared structural follow-up,
a context-carrying hidden layer on the winning arm, measured `0.518657` best against
`0.689775` (`-0.171118`, span closure
`-0.057123`, `docs/evidence/m31-followup-r1/scope.json`) and is refused. Gates on the
unchanged boundary remain per-cell `false`, ranking `false`, attribution `true`.

M32 tested M31's diagnosis on the winner's arm with one declared supervision-side term: each board's
predicted positive mass charged against its occupancy, weight `1.0`, no parameter, inference boundary
untouched (`docs/evidence/m32-mass-match-r1/scope.json`, M32_MASS_MATCH.md, ROADMAP M32). It is
**negative** - F1 at `0.0` `0.277778` (-0.095104), best
`0.602198` (-0.087577), AUC -0.012205, attribution
+0.382572 - so the score shape is not fixable by mass matching. The span split now
names **placement** as the short stage (`-0.103647`, `-0.233858` of
span, the only negative term) with representation at `1.102488`; the recorded
next step is a fixed-lattice placement for the pixel-statistics read, not another supervision term.
