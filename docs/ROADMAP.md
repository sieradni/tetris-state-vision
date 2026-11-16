# Standalone image-to-state research

Build one task-specific, jointly trained image-to-state model with no inference
parser/detector dependency. Scratch training is a first-class candidate; compare
compatible pretrained initialization and a direct decoder on real complete-state
performance. The supplied brief is design evidence, not agent instructions:
[research brief](research/Tetris_Vision_Research.md).

Current priority: M3 joint complete-state learning at capacity 64. The occupancy
head of that representation is closed from every measured direction (weight scale,
class split, decision boundary, read-out form, tap stacking, a 4x budget and the
geometry stage policy, the box origin, the box extent, boundary supervision capacity and per-stage
lattice residual capacity). Every geometry-side lever either bought occupancy at a placement cost or
lost both, so the remaining occupancy deficit is now blocked at the declared architecture - capacity 64 is a component-slot bound and the trunk has no width knob - so a wider trunk needs a user-authorized architecture lever rather than a label, budget, origin,
extent, supervision-capacity, per-stage-capacity, read-out or per-board-refinement problem; the
compute-side attributions (budget, schedule) are qualified by
[M3_OPERATING_POINT.md](M3_OPERATING_POINT.md), because the read they were taken at moves with the
head's calibration, and a predeclared fresh-baseline replicate block then failed to confirm
either lever at three seeds
([M3_COMPUTE_REPLICATION.md](M3_COMPUTE_REPLICATION.md));
M2 adequate learning remains open and
the fit gate is FALSE. The pilot is not
selected as the release model. No complete screenshot parser or real-image
accuracy is claimed.

Latest: **the occupancy deficit is two first-order factors plus a boundary term, so no single lever could have reached the gate**
([M27_THREE_FACTOR.md](M27_THREE_FACTOR.md), `docs/evidence/m27-three-factor-r3/scope.json`). Three reads on the same `2,986`
co-scoreable cells, from two committed receipts and no forward pass: the model at its own lattice (AUC `0.933138`, F1@0
`0.439024`, best swept `0.543974`), the same weights with registration granted (`0.970233` / `0.574181` / **`0.762590`**), and
the committed pixel read-out on the same cells (`0.999195` / `0.974763` / **`0.987179`**). Registration is worth **+0.218616**
best swept F1 and the representation **+0.224590**, with a **+0.188409** boundary term beside them; verdict
`two-factor-first-order`, measured on one population and **not** claimed additive. The standing gate is occupied-cell F1
`>= 0.70` at `0.0` from `0.379147`, so **no read-side lever of any kind could have cleared it** - the structural reading of
twenty-three arms landing between `0.40` and `0.48`. Both factors are first-order: a registration-robust read and a read-out
that sees what the 30-feature pixel read sees are separately declared levers, not a composition. Fit gate unchanged
(`FALSE`, `0 / 28`). The four cross-receipt joins reproduce the published `0.439024` / `0.933138` / `0.574181` / `0.970233` /
`0.978006` / `0.998765` before any number is written. Previous: **the symmetric pair margin avoids the collapse, improves the bare read, regresses the record-best read, and
closes the pair-supervision axis** ([M26_GAP_MARGIN.md](M26_GAP_MARGIN.md), `docs/evidence/m26-gap-r1/scope.json`). The
M25 charge was replaced by one that charges a pair by the closer of its two directional violations
(`min(relu(1 - gap), relu(1 + gap))`, margin `1.0`, weight `.5`, both declared before the receipt). On the bare free
channel it is a real gain: `0.866499` AUC against control's `0.8226433610683003` (**+0.043856**) and an inversion rate of
`0.202031` against `0.224302` (**-0.022271**), with the boundary's positive share at 59.36% instead of M25's collapse to
99.93%. On the record-best read the composition **regresses**: `0.881989` AUC against stencil's `0.9170200020605955`
(**-0.035031**) and F1@0 `0.403049` against `0.480776`, with the boundary pushed from 27.94% to 46.66%. Strict exact is
**0 / 28 in every arm**, and `gap-stencil`'s inversion rate (`0.190075`) is stencil's (`0.189525`) to within a thousandth.
The symmetric charge is orientation-blind by construction - proven in `tests/test_pair_gap.py`, and its exact tie is a
symmetric saddle with zero slope - so it cannot supply the ordering exactness needs. The pair-supervision axis is closed:
both readings have been measured, neither clears, and further variants would tune a definition that excludes the quantity.
Record-best remains `stencil`.

Previous: **the pair-order lever fails: exact stays 0, the ordering barely moves and the boundary pays for it**
([M25_PAIR_ORDER.md](M25_PAIR_ORDER.md), `docs/evidence/m25-rank-r1/scope.json`). The one lever M24's decomposition left
standing - a declared within-board pairwise ordering term, `mean(relu(1 - (occupied - empty)))` over the complete
cross-class product, weight `.5`, both constants fixed before the receipt - was built, proved and measured under the
unchanged protocol. Verdict `pair-order-fails`: strict exact is **0 / 28 in every arm**. On the record-best read the
composition scores `0.918588` AUC against stencil's `0.9170200020605955` (+0.001568, inside the 0.02 attribution
tolerance) and cuts the inversion rate only from `0.189525` to `0.183009` (-0.006516, under the declared 0.01 bar for a
move), while F1@0 falls `0.480776` -> `0.430619` and row error rises +0.404 -> +1.443. On the bare read it is worse than
the control (`0.786891` against `0.8226433610683003`) and the inversion rate *rises* to `0.246080`. The failure mode is
mechanical and named: the term is invariant to a board's level and its hinge is one-sided, so the cheapest way to lower its
charge is to lift empty cells toward the occupied mode - which is why the positive share at the boundary jumped to 42.91%
and 99.93%. The record-best arm stays `stencil` (`0.9170200020605955` / `0.480776`).

Previous: **the exactness barrier is ordering, not level, and it is now measured rather than guessed**
([M24_EXACTNESS_BARRIER.md](M24_EXACTNESS_BARRIER.md), `docs/evidence/m24-exactness-r1/scope.json`). Taking the
record's six strongest arms apart board by board and row by row: **no level makes any board exact** - searching every
distinct level the read can be cut at, not the 81-point grid, gives `0 of 28` in every arm, and the best board in the
record still has exactly `1` wrong cell at its best level - and **no projection does either** (`monotone` `0 of 28`,
`monotone` then the crossing rule `0 of 28`). The miss is ordering: the best arm's within-board AUC is `0.818993`, an
inversion rate of `18.95%` of all occupied/empty pairs, and exactness needs `0.0`. At the `0.0` boundary the read
over-declares (`48.86` false positives against `0.82` false negatives on `stencil`), which is why every level mechanism in
M14-M23 could move the number without ever reaching a single exact board. No mechanism was invented to move it: the
decomposition rules out the level family, and the only lever it leaves standing - an explicit within-board pairwise
ranking objective - is named as unmeasured rather than claimed.

Previous: **a scale-invariant local-contrast read is invariant by construction and still does not meet the bar**
([M23_CONTRAST_READ.md](M23_CONTRAST_READ.md), `docs/evidence/m23-contrast-r1/scope.json`). Mode 3 of
`--occupancy-free-stencil` reads the same five taps on a field where every coarse box is divided by the spread of its own
5x5 neighbourhood, so an offset or a rescaling of the coarse values cancels before the head sees anything - proved
bit-stable (`<= 3.9e-6`) under offsets `+0.4/-3/+12.5` and rescaling `x0.01/x2.5/x40`, with a local change moving it
(`>= 1e-2`) and a uniform board reading exactly zero. Ten arms in one catalogue, control reproducing
`0.8226433610683003` / `0.381640` exactly and the M20/M21 arms reproducing their checkpoints. `norm` measures **0.856281**
AUC / **0.414810** F1@0 (`+0.033170` primary, below the `+0.05` floor) with **0 of 28** strict exact boards, so it is
`stencil-moves-without-clearing`; `norm-cross` is `stencil-fails` at `-0.004534`. The primary is unchanged from the M22
de-meaned arm (`0.414810`) and AUC `0.003542` lower, so **the M22 gain was the level and not capacity a better operation
could reclaim**. What the contrast read does buy is the first `complementary` channel verdict in the record
(`norm-cross` sums `+0.027048` AUC over its own lattice part, and its `local_only` `0.742764` beats the pooled `0.654209`)
and a stencil that survives the crossing rule instead of collapsing under it.

Previous: **the free channel's local stencil is the strongest read measured, and it does not meet the bar**
([M22_STENCIL_READ.md](M22_STENCIL_READ.md), `docs/evidence/m22-stencil-r1/scope.json`). Eight arms in one catalogue -
the published read, the M20 two-path read, the M21 composition, the sum without the envelope, and the same two
compositions with `--occupancy-free-stencil` on - control reproducing `0.8226433610683003` / `0.381640` exactly, the
envelope reproducing M20 bit-for-bit and the crossing arm reproducing M21. The stencil arm reaches **0.917020** AUC and
**0.480776** F1 at `0.0` (`+0.099136` primary, `+0.094377` AUC, row error **+0.404**, best F1 `0.605314`), the best primary
and the best ranking in the record - and **0 of 28** strict exact, so the standing bar is **not cleared**:
`stencil-moves-without-clearing`. Composed with the crossing rule it collapses (`stencil-crossing` `-0.069945`).
The channel split says why: the local stencil alone ranks `0.634832` against the pooled channel's `0.781555`, adding it
*lowers* the free channel, and `82%` of its own magnitude is a board-wide level. The de-meaned form
(`--occupancy-free-stencil 2`) removes the level exactly and the gain falls to `+0.033170`: **the primary gain was the
level, not the locality**. The in-loop versus post-hoc ablation is measured: in-loop `0.402187` against post-hoc
`0.341542` (`+0.060645` primary, `-0.044418` AUC), so in-loop training is what makes the rule work on the primary while
post-hoc application ranks better - M21's "re-decides the same place" is corrected to those numbers.

Previous: **the level repair works, and the decomposition says where the level comes from**
([M21_LEVEL_COMPOSITION.md](M21_LEVEL_COMPOSITION.md), `docs/evidence/m21-level-r1/scope.json`). Four arms in one
catalogue, control reproducing `0.8226433610683003` / `0.381640` exactly, and the M20 arm reproducing the M20 checkpoint
bit-for-bit. The requested composition
`--occupancy-lattice-free 2 --occupancy-stack-monotone 1 --occupancy-crossing-threshold 1` is **`level-passes`**: plain F1 at
`0.0` **0.402187** (`+0.020547`), AUC **0.823601**
(`+0.000957`), positive share `46.88%` against the
control's `49.95%`, row error **+1.289**
against `+7.819`, out-of-fold `0.447280`, swept
`0.447280 @ 0.90`,
**0 of 28** strict exact. The sum without the envelope is
**`decoupling-fails`** (`-0.016298` primary, `+0.060716` AUC), so the envelope
carries the boundary and the M20 coupling was the thing that hid it; the coupling is now an option. The stage
decomposition names `monotone` as the stage that raises the positive share
(`+0.220416`) and shows the crossing stage taking it back by
`-0.406250` share points once the rule is trained in the loop. M16's recorded
envelope + centring + crossing arm remains the best primary at `0.405439` /
`0.835241`, so the free channel still costs this composition
`-0.003252` primary.

**Next candidate, named:** give the free channel a coarse local stencil on the fixed lattice. It is the limiter in every
arm - alone `0.497469` in the composition, `0.607085`
in the M20 arm - because it consumes pooled context plus fixed coordinates and nothing local, so a coarse stencil adds
detail without inheriting the decoded lattice's registration error. Bar unchanged: `+0.05` over the reproduced control and
at least one strict exact board.

Previous: **the summed two-path read is the best-ranking and best-placing read this programme has measured, and it
still fails the fixed boundary - with the channel split naming why**
([M20_TWO_PATH_READ.md](M20_TWO_PATH_READ.md), `docs/evidence/m20-two-path-r1/scope.json`). `--occupancy-lattice-free 2`
adds the lattice-free channel to the lattice-sampled read under the M15 envelope; the sum is ungated, its head zeroes only
its output projection, and it is refused without the envelope in both config validation and the parser. Family
`m20-two-path-`, control reproduced `0.8226433610683003` / `0.381640` exactly, every stage `passed 0` on the fourth attempt
(three void attempts archived; V-142, V-143). AUC **0.898066** (`+0.075423`, the best ever here,
above the M15 combined `0.888912`), swept optimum **0.600117 @ 3.30**
(`+0.188083`, best ever), out-of-fold **0.459725** (best ever), row error
**+3.322** against the control's `+7.819` - and plain
F1 at `0.0` **0.350636** (`-0.031004`) with
`55.52%` of cells positive against `49.95%`, so the
failure is the read's *level*, not its signal. TP/FN `662/5`,
**0 of 28** strict exact boards. Channel split: sum `0.898066`,
lattice part alone `0.896211` (`+0.001855`), free channel alone
`0.607085` AUC -> **`one-channel-noise-dominates`**: the declared mechanism (added evidence) is falsified, while the
lattice channel in *this* arm's weights reaches `0.896211` against the envelope arm's
`0.845948`, so the gain is a training-time effect of the added channel, not added cell evidence. Against
the M15 envelope arm the primary is `-0.058525`. V-143 also retracts the M3 lattice-free
claim: alive, that channel alone reads `0.607085` AUC.

**Next candidate, named:** the same read with the measured structural level repair -
`occupancy_lattice_free 2` + `occupancy_stack_monotone 1` + `occupancy_crossing_threshold 1`. This arm's failure is
exactly the level failure the crossing rule repaired in M19 (positive share `80.55%` -> `52.16%`, row error `9.409` ->
`6.993`, `level_repaired = true`), and here the level is off in the same direction (`55.52%` positive, best threshold
`3.30`), so the composition is the smallest declared change that can put the boundary where the read's own groups meet. The
bar stays the declared one: primary at least `0.05` above the reproduced control (`0.381640`) and at least one strict exact
board.

Previous: **the span under the level transforms repairs the level and still loses - so M18's separation gain was a level
artefact** ([M19_SPAN_LEVEL.md](M19_SPAN_LEVEL.md), `docs/evidence/m19-span-level-r1/scope.json`). The declared
composition (`--occupancy-tap-span 1` **with** `--occupancy-centre 1 --occupancy-crossing-threshold 1`, admitted together
and only together) ran against the reproduced control in one catalogue; control reproduced `0.8226433610683003` /
`0.381640` exactly and every stage is `passed 0`. Reading **`span-level-fails`**, secondary `ranking-degraded`, with
`level_repaired = true`: plain F1 at `0.0` **0.350613** (`-0.031027`), AUC **0.789794** (`-0.032849`), TP/FN `629/38`,
positive `52.16%` (M18's `80.55%`, control's `49.95%`), row error **`+6.993`** - better than the control's `+7.819` - but
best swept F1 `0.364139` and out-of-fold `0.328718`, both **below** the control's `0.412034` / `0.400399`. The transforms
did exactly what they were declared to do and the primary still fell: M18's swept and out-of-fold gains were properties of
its displaced level, not of separable cell evidence, and the row-neighbour tap carries no recoverable per-cell
information. The M18 record now carries that correction. Nineteen read-side arms have left the strict gate at **0 of 28**
under both definitions; the standing bests are disjoint - M15 combined holds the best ranking (`0.888912`) and placement
(`1.189` rows), and the only 2-wrong-cell board ever produced came from the lattice-free read (`0.840147`). Named next
candidate: a **two-path read** summing the lattice-free channel's evidence with the lattice-sampled envelope's - the one
combination the receipts point at that has never been measured.

Previous: **the row-span tap is falsified at the boundary but is the first read-side lever to improve both the swept
optimum and the out-of-fold figure** ([M18_TAP_SPAN.md](M18_TAP_SPAN.md),
`docs/evidence/m18-tap-span-r1/scope.json`). `--occupancy-tap-span 1` makes the detail tap read three row-aligned samples
- the cell and the rows immediately above and below it in cell units - through one zero-initialized projection, chosen
because the measured error density is 2.2x higher in the four rows at the top of each stack than below it and no arm had
ever given a cell its row neighbour. Family `m18-span-`, control reproduced `0.8226433610683003` / `0.381640` exactly,
every stage `passed 0` after one whole-block rerun (V-141). Reading **`span-fails`**, secondary **`ranking-degraded`**:
plain F1 at `0.0` **0.244496** (`-0.137144`), AUC **0.795536** (`-0.027107`), TP/FN `633/34`, `80.55%` of cells positive
(`49.95%`), row error `+9.409` (`+7.819`) - but best swept F1 **0.443255** at `2.55` against the control's `0.412034` at
`0.90`, and out-of-fold **0.429607** against `0.400399`. So the geometry carries separation and loses the level: the
declared falsification fired, no span width or projection size is offered, and the gate stays **0 of 28 under both strict
definitions**. Not a record - four swept arms are higher (`0.540811`, `0.502686`, `0.474210`, `0.465582`). The receipts
also settle read-versus-population: 28 boards would show two or three exact boards at a tenth-per-board exactness and
nineteen arms show none, so the constraint is per-cell read precision, not population size. Named next candidate: the
span beneath the validated structural level transforms (M14 centring, M16 rule) as its own declared arm.

Previous: **the joint skyline decision is falsified, and with it the idea that the columns' independence was the
problem** ([M17_SKYLINE_RECOVERY.md](M17_SKYLINE_RECOVERY.md), `docs/evidence/m17-skyline-recovery-r1/scope.json`).
`--occupancy-skyline-recovery 1` recovers each board's whole skyline with a dynamic programme over per-column heights
(emission = signed mass on each side of a split, penalty `1.0` logit, deterministic), applied last after the envelope,
centring and the crossing rule. Family `m17-skyline-`, catalogue `5c02208e`; control reproduced `0.8226433610683003` /
`0.381640` exactly and every stage receipt `passed 0`. Reading **`skyline-fails`**, secondary **`ranking-degraded`**:
plain F1 at `0.0` **0.308333** (`-0.073307` against control, `-0.097106` against the per-column rule), AUC **0.620664**
(`-0.201979`, the worst in the program), TP/FN `592/75` against control's `661/6`, cells positive `56.66%` (control
`49.95%`), row error `+3.993` - worse than the read it was applied to (`1.189`) - occupancy loss `0.5988`, and the gate
**0 of 28 under both strict definitions**, so nothing newly fits and there is no V-139 class of artefact to audit. Both
declared falsifications fired. The mechanism of the failure is on the record: a *volume* emission prefers heights that
accumulate positive read rather than the level the column steps at, and replacing cell magnitudes with their side of the
recovered height destroyed the cross-board ranking. No penalty sweep or alternate emission is offered - that would be the
per-arm tuning the declaration forbids. Seventeen arms now separate by where they act: decision-side arms cannot beat the
read's row error, read-side arms hold every best placement and ranking, and the closest board ever produced is 2 wrong
cells from a cell-read lever - so the remaining distance is a per-cell read limit, not a decision or structure limit.

Previous: **the fit gate's `1 of 28` was a metric artefact - the true figure is `0 of 28` for all 79 arms** (V-139).
The obstacle was audited before anything was built: every curator's exactness test compared column boundaries and
silently dropped any column whose decision or label was empty, so a board that filled genuinely empty columns could be
scored exact. Recomputed from the retained per-case logits of **every arm sweep in `runs/`** - no block run, no receipt
rewritten - the strict definitions give **0 of 28 for all 79 arms**: cell-exact (every visible cell right at `0.0`, the
sweep's own flag) and boundary-exact (every column's decided top equals its occupied top, an empty column compared as a
value). The lenient rule showed `1 of 28` for 26 arms and `2 of 28` for 5 - all phantom, including the boards credited to
M12 and M13, whose figures are withdrawn in the record rather than rewritten. The gate is therefore explained rather than
approached: the closest any arm has come is **2 wrong cells** on its best board (`m3-occlatfree-lever`, AUC `0.840147`),
three boards within ten wrong cells, while no structural arm (`M15`/`M16`) is below eight. Metric corrected in all six
curators (cell-exact and boundary-exact both required, empty columns compared as values), pinned by
`tests/test_exact_state_metric.py` (8 tests, including a walk over every retained receipt asserting `0 of 28` under both
definitions). The sequence-level recovery that was conditional on the metric being sound was therefore not built: its
premise did not hold, and its target - exact boards - is now known to be an unapproached `0 of 28` rather than a gate one
board from moving.

Previous: **deciding at each board's own crossing breaks the fixed-threshold ceiling, and not the bar**
([M16_CROSSING_THRESHOLD.md](M16_CROSSING_THRESHOLD.md), `docs/evidence/m16-crossing-threshold-r1/scope.json`).
`--occupancy-crossing-threshold 1` subtracts, per board, the exact Otsu level of that board's own visible occupancy
logits inside the head (after the envelope and centring), so every consumer still thresholds `0.0` and `0.0` *is* that
board's crossing. It is label-free by assertion, not by claim: the tests permute, invert and zero the labels and require
bit-identical output, and scan the executable body for any label reference. Family `m16-crossing-`, catalogue
`ad204845`; control reproduced `0.8226433610683003` / `0.381640` exactly. Reading
**`crossing-moves-without-clearing`**: plain F1 at the rule's crossing **0.405439** (`+0.023799` against control, and
**`+0.014634` against the same read's fixed-`0.0` ceiling of `0.390805`** - the declared falsification did not fire),
AUC `0.835241` (`+0.012598`), best swept `0.428088 @ 1.35`, out-of-fold `+0.022855`, row error `+1.189`, occupancy loss
`-0.113`, cells positive `-4.07` points, fit gate still `0` of `28`. Two declared expectations were falsified: the rule
lands `0.0227` below its own swept maximum, so per-board selection is only partly aligned with each board's optimum, and
it bought precision with recall (`4` -> `11` false negatives). Verdict: the decision-rule hypothesis is **not falsified
and not sufficient** - the fixed `0.0` was costing real F1, the bar was never reachable this way, and the next candidate
the evidence names is the fit gate (`0` of `28` exact boards across sixteen arms).

Previous: **stack structure plus a centred level buys the best ranking in the program, but not `0.0`**
([M15_STACK_CONTIGUITY.md](M15_STACK_CONTIGUITY.md), `docs/evidence/m15-stack-combined-r1/scope.json`). The declared
follow-on block ran: `--occupancy-stack-monotone 1 --occupancy-centre 1` (envelope first, centring after it) against the
M15 block's own control in the same catalogue `d9d867a6`. Reading **`stack-moves-without-clearing`**: plain F1 at `0.0`
**0.390805** (`+0.009165`), AUC **0.888912** (`+0.066269`, the highest of any arm, above M10's `0.885462`), best swept
**0.502686** at `1.20` (`+0.091`, the first arm to clear the `0.431640` bar by a wide margin), out-of-fold `0.392122`,
`4` false negatives against control's `6` and `73` fewer false positives, row error `+1.846`, occupancy loss `0.5987`
(lowest measured), fit gate still `0` of `28`. Two declared expectations held (`0.0` sits at the mean, `48.68%`
positive; row error near the envelope's) and one was falsified on its face: the mean is still not the operating point
(the optimum sits at `1.20`). State of the program after fifteen arms: no mechanism clears the bar at a fixed `0.0`,
while the same read reaches `0.502686` at its own optimum - so the remaining obstacle is the fixed-`0.0` decision rule,
not the read, and the direction the numbers point at is a declared, label-free rule that follows the crossing the
monotone structure already shows.

Previous: **the stack's own shape places the crossing, and it is the first real gain** ([M15_STACK_CONTIGUITY.md](M15_STACK_CONTIGUITY.md),
`docs/evidence/m15-stack-contiguity-r1/scope.json`). `--occupancy-stack-monotone 1` replaces each column of the
occupancy logits with its monotone envelope from the floor up (suffix maximum in bottom-first order) inside the head, so
a column crosses `0.0` at most once and that crossing is its height - no statistic, no label, no threshold, no parameter.
Family `m15-stack-`, catalogue `d9d867a6`; control reproduced `0.8226433610683003` and `0.381640` exactly. Reading
**`stack-moves-without-clearing`**: plain F1 at `0.0` **0.409161** (`+0.027521`, the first arm ever to move the primary
in the right direction), AUC **0.845948** (`+0.023305`, a ranking gain), best swept **0.442636** at `2.85` - the first
treated swept maximum above the `0.431640` bar - out-of-fold `0.409639`, cells positive `-4.16` points, and row error
**`+1.255` rows from `+7.819`**, a factor of six, with identical recall (`661` TP, `6` FN) and `233` fewer false
positives. Fit gate still 0 of `28`. Both declared falsifications were checked: the optimum did *not* move to `0.0` (it
moved to `2.85`, so the envelope places the crossing's shape but not its level, exactly as its equivariance requires),
while row error fell rather than rose. Seventh arm: structure *can* place the crossing, the placement ceiling is not
absolute, and the one remaining degree of freedom is the level of a monotone read - the declared follow-on is contiguity
plus centring.

Previous: **removing the level pins it at the mean, which is the wrong level** ([M14_READ_CENTRING.md](M14_READ_CENTRING.md),
`docs/evidence/m14-read-centring-r1/scope.json`). The declared structural option `--occupancy-centre 1` subtracts each
board's own visible-window mean from its occupancy logits inside the head, before any loss or scorer sees them - no
label, no threshold, no parameter, spread untouched. Family `m14-centre-`, new catalogue `c9544334`; control reproduced
`0.8226433610683003` and `0.381640` exactly, and the block asserts its own identity (configs differ in exactly
`occupancy_centre`, both sweeps carry their own checkpoints, the previous block's checkpoint excluded by name). Reading
**`centre-fails`** at the narrowest margin yet: plain F1 at `0.0` **0.374142** (`-0.007498`), AUC `0.815395`
(`-0.007248`), best swept `0.393230` at `1.95` (`-0.019`), out-of-fold `0.378651`, row error `+0.188`, cells positive
`+0.57` points (49.95% -> 50.52%), occupancy loss `-0.152` better, fit gate FALSE at `0` of `28`, no overshoot below
zero. The transform does exactly what it claims - the drift collapsed from `+14.5` to `+0.6` points - but the pinned
level is the board mean, and on a `12%`-occupied board the F1-optimal boundary sits `1.95` above it. Sixth negation:
adding supervision cannot place the boundary and removing the level cannot either. Declared next: **per-column
stack-contiguity of the read** (cumulative max from the floor, so a column crosses `0.0` at most once and the crossing
is its height) - label-free, no constants, with holes/overhangs as its declared failure mode.

Previous: **supervising the location also fails, and the shift itself is what must go** ([M13_MIDPOINT_ANCHOR.md](M13_MIDPOINT_ANCHOR.md),
`docs/evidence/m13-midpoint-r1/scope.json`). The declared midpoint anchor -
`mean over qualifying boards of |(m_occupied + m_empty)/2|`, target `0.0`, no margin, weight `0.5`, equal class
weight - is a pure location instrument: a shift of the read moves both means, so unlike every earlier term it cannot
be paid off by raising the occupied class. Family `m13-midpoint-`, new catalogue `19077be4`; control reproduced
`0.8226433610683003` and `0.381640` exactly. It failed: AUC `0.830500` (`+0.007857`), plain F1 at `0.0` **0.310966**
(`-0.070674`), best swept `0.401408`, out-of-fold `0.348472`, cells positive `+14.52` points to `64.5%`, swept optimum
`3.00`, fit gate **FALSE** at `1` of `28`; paid charge mean `0.923`, never held. Reading `midpoint-fails`, secondary
`ranking-preserved`, override rule not triggered (AUC inside tolerance, so not the collapse mode). The level moved the
wrong way a fifth time, which closes off **adding supervision a shift can partially satisfy**. Declared next:
**structural centring of the read** - subtract each board's own window mean from the occupancy logits inside the head,
no labels, no parameters, removing only the additive mode that M8's `normalise` also removed spread from - with its
failure mode (overshoot below zero on a `12%`-occupied population) and the standing `+0.05` / `-0.02` bars to be
restated in its own document before its own receipt.

Previous: **charging every cell fixes everything but the placement** ([M12_DENSE_HINGE.md](M12_DENSE_HINGE.md),
`docs/evidence/m12-dense-hinge-r1/scope.json`). The declared dense class-balanced hinge -
`mean per qualifying board [0.5 mean_occ relu(margin - logit) + 0.5 mean_empty relu(margin + logit)]`, margin `1.0`,
weight `0.5`, balance `.5/.5` over known cells - supersedes the three per-board summaries. Family `m12-dense-hinge-`,
new catalogue `24c76620`; control reproduced `0.8226433610683003` and `0.381640` exactly. The arm is the closest yet:
AUC **0.862838** (`+0.040195`, a gain), best swept F1 **0.465582** (`+0.053548`), out-of-fold **0.465582**
(`+0.065183`) - **the first arm to beat control on either** - row error better by `+0.725`, the paid charge `3.6x`
lower at `0.670`, the level drift cut from `+28.0` to `+4.8` points of the population, and the **first exact board at
`0.0`** (`1` of `28`). Plain F1 at `0.0` still fell to **0.354770** (`-0.026870`), reading `dense-fails`, secondary
`ranking-preserved`, fit gate **FALSE**. So the shape is solved and the placement is not: the swept optimum moved
*up* to `3.15`, i.e. the read separates well and sits about three logits high, which is the free additive level.
Declared next: **midpoint anchoring**, `mean over boards [ |(m_occupied + m_empty) / 2| ]` at weight `0.5`, no margin
- a location instrument that grows on any drift and cannot be satisfied by raising the occupied class - with its
constants to be declared before its own receipt.

Previous: **bounding the mass also fails, and the level is the thing that drifts** ([M11_QUANTILE_ANCHOR.md](M11_QUANTILE_ANCHOR.md),
`docs/evidence/m11-quantile-anchor-r1/scope.json`). The declared class-quantile anchor -
`mean [max(0, margin - q10(occupied)) + max(0, margin + q90(empty))]`, margin `1.0`, weight `0.5`, quantiles `.10`/`.90`,
linear interpolation over known cells - replaces the extrema with the classes' own percentiles. Family
`m11-quantile-anchor-`, new catalogue `cb8775b5`; control reproduced `0.8226433610683003` and `0.381640` exactly. The
arm came out **negative on every axis**: plain F1 at `0.0` **0.264308** (`-0.117332`), AUC `0.815340` (`-0.007303`),
best swept `0.404026`, out-of-fold `0.379323`. Reading `bulk-fails`, secondary `ranking-preserved` (inside tolerance),
fit gate **FALSE**. The cause is now measured, not guessed: control calls `49.9%` of cells positive and its best
boundary is `+0.90`, the quantile arm calls `78.0%` and its best boundary `+2.25`, so supervising a **per-board
statistic** buys a free upward shift of the read's level rather than pinning it - the charge falls from `3.004` to
`2.417` while the level rises. Three summary statistics in a row (two cells, ten percent, one number per class) have
now failed, so the declared next arm charges **every** cell instead: a class-balanced dense margin hinge
`mean [0.5 mean_occ relu(margin - logit) + 0.5 mean_empty relu(margin + logit)]`, margin `1.0`, weight `0.5`, whose
satisfaction implies a perfect `0.0` boundary and whose gradient falls exactly on the band that decides the operating
point. The predeclared mean fallback is argued down in the record rather than taken.

Previous: **the strongest occupancy result yet, and still no operating point** ([M10_SUPERVISED_ANCHOR.md](M10_SUPERVISED_ANCHOR.md),
`docs/evidence/m10-supervised-anchor-r1/scope.json`). A declared **two-sided anchor** -
`mean [max(0, margin - lowest_occupied) + max(0, margin + highest_empty)]`, margin `1.0`, weight `0.5`, known cells
only, on the unnormalised read - names `0.0` in the objective on both sides instead of leaving the gap free. Family
`m10-anchor-`, new catalogue `6e207218`; control reproduced `0.8226433610683003` and `0.381640` exactly. The anchor
arm reached AUC **0.885462** (`+0.062819`), best swept F1 **0.540811** (`+0.128777`), out-of-fold **0.540811**
(`+0.140412`, all twelve folds selecting the same boundary) - **above the widened-field arm on both axes with no
deformation field, no sampling change, no parameter and no tensor** - while plain F1 at `0.0` fell to **0.310966**
(`-0.070674`). Declared reading `anchor-fails`, secondary `ranking-preserved`; fit gate **FALSE** (0 of 28). Both
supervision terms so far are *extremal* (2 cells of 200), so the empty mass is unconstrained and rides up: `64.5%` of
cells sit above zero. Next: the **class-quantile anchor** -
`mean [max(0, margin - q10(occupied)) + max(0, margin + q90(empty))]` - which speaks about mass, with a declarable
ceiling (precision `>= 0.55`, F1 near `0.68`) and a declared cost to ranking; the learned-bias fallback is argued
down in the record because the failure is mass, not a scalar offset.

Latest: **supervising the margin buys ranking and still not a boundary** ([M9_SUPERVISED_MARGIN.md](M9_SUPERVISED_MARGIN.md),
`docs/evidence/m9-supervised-margin-r1/scope.json`). The occupancy objective now carries a declared within-board
hinge - `mean over boards max(0, margin - (lowest occupied logit - highest empty logit))`, margin `1.0`, weight
`0.5`, over known cells only, fixed in the predeclaration - so the read is finally *told* that its modes must be
separated instead of being thresholded afterwards. Control reproduced `0.8226433610683003` and `0.381640` exactly
in the new catalogue (`eae71837`). The hinge arm: AUC **0.864667** (`+0.042024`), best swept F1 **0.474210**
(`+0.062176`, clearing the `0.05` attribution floor), out-of-fold **`+0.029058`** - the strongest occupancy
ranking result in the project with **no sampling change** - but plain F1 at `0.0` **0.251701** (`-0.129939`), so
the declared reading is `margin-fails` and the boundary drifted *up* (`0.90 -> 2.70`; `4,625` of `5,600` cells
called occupied; row error `+10.691`). The paid hinge reached `0.0` on the final training step: separation was
achieved and location was not controlled, exactly as declared. Next: the **two-sided anchor** -
`max(0, margin - lowest_occupied) + max(0, margin + highest_empty)` - which keeps the `+0.042024` separation and
pins the empty mode below zero, so `0.0` becomes a supervised boundary. The `-1` offset's `+0.048086` must now be
re-measured against the hinge baseline (`0.864667`), not against control. Fit gate **FALSE** (0 of 28).

Latest: **training the read in per-board normalised space cannot place the boundary either - the ceiling is
geometric** ([M8_NORMALISED_BOUNDARY.md](M8_NORMALISED_BOUNDARY.md),
`docs/evidence/m8-normalised-boundary-r1/scope.json`). A declared output parameterisation
(`--occupancy-normalise 1`) z-scored each board's occupancy logits over its own visible cells *inside* the model,
so the tensor the loss differentiates is the tensor scored and `0.0` is the operating point by construction - no
parameter, no tensor, no label, no prevalence, and `0` exactly the published function. Control reproduced
`0.8226433610683003` and `0.381640` exactly in the new catalogue (`efa02b84`). The normalised arm reached plain
F1 **0.384661** (`+0.003021`, declared reading `normalisation-moves-without-clearing`) and AUC **0.788463**
(`-0.034181`, secondary `ranking-degraded`), out-of-fold `-0.007032`. The reason is arithmetic, not training: a
per-board z-score puts zero at the board's centre of mass, so `48.2%` of cells sit above it and at `11.9%`
prevalence precision cannot exceed `0.247` - the arm measured `0.2399` and F1 `0.384661`, i.e. **it hit the
ceiling of its own parameterisation**. Standardising the output makes zero a median, and a median is not an
operating point. Row error did improve (`+7.819 -> +5.134`), which is the same fact: the read can no longer sit
high as a whole. Next: put the **margin into the loss** - a declared within-board hinge on the gap between each
board's lowest occupied and highest empty logit, on the unnormalised read so ranking is not damaged. Fit gate
stays **FALSE** (0 of 28). V-129 records this pass's defect.

Latest: **the shifted read's operating point is not recoverable from its own logit distribution by an
imbalance-blind rule, so the offset's advantage is a threshold effect**
([M7_CALIBRATION.md](M7_CALIBRATION.md), `docs/evidence/m7-calibration-r1/scope.json`). Three arms were
re-captured in one catalogue and scored read-only at two rules; control and wide reproduced
`0.8226433610683003` and `0.8822623454135491` exactly. A per-board threshold chosen from each board's own 200
occupancy logits by Otsu's criterion - no labels, no swept F1, translation-equivariant by construction, so it
absorbs exactly the `0.90 -> 2.70` boundary drift - puts the `-1` offset arm at F1 **0.380898**: `+0.035025`
over its own `0.0` boundary, but `0.000742` **below** control and `-0.000219` against control at *its* label-free
threshold. Declared reading `calibration-fails`, secondary `offset-advantage-was-a-threshold-effect`. The
mechanism of the failure is diagnosable and is not the read's fault alone: cells are `11.9%` occupied and Otsu
splits by between-class *variance*, so all three arms land at recall `0.99` / precision `0.24` - a saturated
detector - and the rule hurts the best arm too (`-0.027668` F1 for wide). What survives is the ranking finding:
`-1` buys `+0.048086` AUC (interval `+0.013560 .. +0.071269`) with no tensor added. Next: make the boundary a
**learned** quantity rather than a post-hoc one - an occupancy logit trained in a per-board normalised
parameterisation, so `0.0` is a supervised operating point and the drift cannot hide in an unconstrained scale -
or, failing that, a label-free rule that models the occupied rate. Fit gate stays **FALSE** (0 of 28 exact
boards). V-126 to V-128 record this pass's defects.

Latest: **a one-row sampling offset recovers most of the best arm's gain, in the opposite direction to the
one predicted** ([M6_ROW_OFFSET.md](M6_ROW_OFFSET.md), `docs/evidence/m6-row-offset-r1/scope.json`,
diagnostic in `docs/evidence/m5-row-error-r1/`). A read-only diagnostic of the four frozen sweeps shows
every arm deciding the stack **above** its label row - +7.819 rows for control at the standing boundary,
+5.872 for wide, still +0.716 to +2.886 at each arm's own best boundary - so the residual error is a
sampling geometry, not a mis-specified label. A declared constant `--occupancy-offset-rows` (a
non-persistent buffer: **no tensor, no state key**, step zero exactly the published function) was measured
in both signs. The predicted sign (`+1`) **fails** (`-0.018076` best F1, boundary moves `0.282` rows away
from the labels); the opposite sign **works**: `-1` reaches AUC **0.870729** (`+0.048086` over control,
interval `+0.013560 .. +0.071269`, MDE `0.043683`) and best swept F1 **0.469259** (`+0.057225`, clearing
the `0.05` attribution floor), with its boundary at `+1.342` rows against control's `+2.886`. So four
fifths of wide's `+0.059619` AUC - previously read as a shifted-window exploit - is a **one-row
registration error**. It is not calibration though: F1 at `0.0` is `-0.035767`, the best boundary moves
from `0.90` to `2.70`, and out-of-fold F1 falls `0.031591`, leaving it `0.045562` best F1 behind wide.
Declared reading `offset-class-fails` (on the predicted arm). Fit gate stays **FALSE** (0 of 28 exact
boards). Next: the **operating point** - a declared, label-free calibration of the shifted read - not
another offset. V-123 and V-124 record this pass's defects.

Latest: **the column read-out is used heavily and still does not rank**
([M5_COLUMN_READOUT.md](M5_COLUMN_READOUT.md), `docs/evidence/m5-column-readout-r1/scope.json`). A
fourth arm gives each cell its own column - the mean decoder feature strictly below and strictly above it,
plus the spans, through a zero-initialised `2*dimension+2 -> dimension` projection - so the vertical
structure the shifted window was buying is available where the cell is and no sample moves. Control, wide
and de-meaned reproduced the previous record exactly (`0.822643...`, `0.882262...`, `0.820429...`). The
column arm moves every deciding number and none of it resolves: AUC **0.832796** (`+0.010152` over
control, interval `-0.030013 .. +0.062077`, MDE `0.061504`), best F1 **0.432650** (`+0.020616`, two fifths
of the `0.05` floor), out-of-fold F1 **`-0.018609`**, and it is still `-0.082171` behind wide. The new
read-only **ablation** settles the mechanism: zeroing the projection on the same weights costs `0.017048`
AUC and moves **42.3%** of each cell's logit magnitude, with true positives up `89` and false positives up
`695` - the read-out is genuinely used, it reports *that a column has material*, not *which row is filled*.
Declared reading **`readout-moves-without-ranking`**. So the vertical information was never the missing
piece; the next declared instrument is a **signed row error of the decision boundary** on the existing
frozen sweeps - if the boundary sits about a row off the labels, the correction is a declared sampling
offset (geometry, not tuning), and if it does not, the label-objective class is what fails. Fit gate stays
**FALSE** (0 of 28 exact boards on all four arms). V-119 through V-122 record this pass's defects.

Latest: **placed-target supervision loses to the objective, and the position instrument says why**
([M4_OCCUPANCY_PLACED.md](M4_OCCUPANCY_PLACED.md),
`docs/evidence/m4-placed-supervision-r1/scope.json`). A fourth arm adds a term that supervises the
*displacement* - smooth-L1 between the offsets the detail tap applied and the per-cell displacement to the
target lattice's own cell - on the unchanged widened field, so it adds no tensors and step zero stays the
published function. Control, wide and de-meaned reproduced the previous record exactly (`0.822643...`,
`0.882262...`, `0.820429...`). The placed arm is **worse**: AUC **0.762448**, best F1 **0.378316**, with
**resolvable** AUC losses against both control (`-0.060196`, interval `-0.116936 .. -0.015133`) and wide
(`-0.119815`, interval `-0.208163 .. -0.052118`). Its field engaged (constant-share **0.892**) but moved
only **0.462** cells of an ~`2.6`-cell target: the `0.25` auxiliary weight that keeps the term from
dominating is, in 64 Adam steps, also too weak to express what it declares. The new read-only position
instrument answers the open question: the wide arm's `5.128`-cell translation leaves **5.0%** of samples
outside the image frame and **64.1%** outside the target board (control 29.0%, de-meaned 22.3%), so it is
a shifted-window read, not a decode correction. Declared reading: **`objective-class-fails`**. Per-cell
registration is not a lever this architecture carries through this read-out, and the next structural
change is the **read-out** - cell-relative vertical context, a per-column stack profile - not another
reach, bound or supervision weight. Fit gate stays **FALSE** (0 of 28 exact boards on all four arms).
V-109 through V-116 record this pass's own defects, preserved under `runs/attempts/`.

Latest: **per-cell registration is learnable and buys nothing; the family's gain was the translation**
([M3_OCCUPANCY_REGISTRATION.md](M3_OCCUPANCY_REGISTRATION.md),
`docs/evidence/m3-occupancy-deformation-r3/scope.json`). A fourth arm de-means the applied field across
the board, so its constant mode is exactly zero by construction rather than by penalty; the control,
one-cell and wide arms reproduced the previous block's numbers exactly (`0.822643...`, `0.866566...`,
`0.882262...`), so the comparison is admitted. The de-meaned field is genuinely per-cell: constant-share
**0.000752** against the wide arm's **0.978**, mean `|offset|` **0.541** cells (**1.153** on occupied
cells, p90 1.358, max 3.387, nothing at its bound, 36.5% of samples outside their own cell). It still
buys nothing: AUC **0.820429** against control `0.822643` (interval -0.037595 .. +0.033223, MDE
`0.050764`), best F1 `0.427911` (+0.015877, below the floor), and it is **0.061834 AUC behind** the free
field with an interval clear of zero. Declared reading: **`registration-without-signal`**. So per-cell
registration is expressible and useless to this objective, and the free field's entire gain came from a
`5.128`-cell translation - twice the 2.57-cell residual the oracle pairing measured. The next step is
supervision of the displacement (a placed-target term) or image-space re-projection of the field with an
in-frame fraction to separate a decode offset from a shifted-window exploit - not another reach change.
Fit gate stays **FALSE** (0 of 28 exact boards on all four arms). V-107 and V-108 record the defects this
pass found in its own curation, preserved under `runs/attempts/`.

Latest: **the widened registration field settles the question: the bound capped it, and what it applies
is a translation** ([M3_OCCUPANCY_REGISTRATION.md](M3_OCCUPANCY_REGISTRATION.md),
`docs/evidence/m3-occupancy-deformation-r2/scope.json`). Three matched 64-step continuations differing
only in the registration lever - detail control, the one-cell field, and a coarse-to-fine field reaching
four cells - were scored on the same deciding population. The wide arm is the largest attribution in
this family (**+0.102787** best F1, AUC **0.882262** against 0.822643, **+0.059619** with interval
+0.029411 .. +0.094463) and it restores the fixed-boundary F1 the fine field cost (**+0.016482** over
the one-cell arm, resolvable). But the measured field explains why that is not registration: the
one-cell field sat at 70.4% of its bound, the wide field saturates at **83.8%** of four cells, its mean
displacement is **3.760** cells (**5.128** Euclidean, twice the 2.57-cell residual it aimed at), and
**97.8% of its displaced energy is one constant vector** - a read-out translation, with 99.8% of samples
leaving their nominal cell. Per the predeclared rule the reading is **`reach-exceeded`**; per-cell
registration is **not a lever this design can resolve**, because nothing in the loss distinguishes the
right cell from a re-centred read-out. Both gates still fail, 0 of 28 boards are exact and the fit gate
stays **FALSE**. The next lever is supervision that can tell the correct displacement from a constant
one - a zero-mean constraint or a placed-target term - not more reach. V-104, V-105 and V-106 record the
defects this pass found in its own work, all preserved under `runs/attempts/`.

Latest: **the registration residual is attacked with a real mechanism, and it is the largest attribution
this family has measured** ([M3_OCCUPANCY_REGISTRATION.md](M3_OCCUPANCY_REGISTRATION.md), predeclared
before its receipts, `docs/evidence/m3-occupancy-deformation-r1/scope.json`). A bounded per-cell
deformation field - a zero-initialized convolution over the tap's own feature, sampled at each decoded
cell centre and added to that cell's sample offset, clamped to one cell - was trained as a matched
64-step continuation against a control differing only in that field. **Attribution floor met: best
swept F1 0.490752 against 0.412034 (+0.078718), AUC 0.866566 against 0.822643 (+0.043922, interval
+0.017963 .. +0.063903, MDE 0.032257)**; the field costs 32.3 MB and was admitted at 1,090,600,960 B of
1,792 MiB. It closes **48.5%** of the control's distance to the oracle-registration level (0.5742) at
the same budget. The calibration-neutral decomposition keeps that honest: the ranking gain is
resolvable, the out-of-fold F1 gain (+0.044328) is **not** at twelve clusters (interval -0.105165 ..
+0.290645, MDE 0.315147), and the lever's own logit scale moved the best boundary from 0.90 to 3.15 so
its **zero-boundary cost quadrupled** (-0.030394 -> -0.121332) and F1 at `0.0` fell -0.012220
(resolvable). Both absolute gates still fail, 0 of 28 boards are exact and the fit gate is **FALSE**.
The next lever is a calibrated boundary for the arms that move the scale, then this same pair at a
matched multiple of budget. V-103 records a defect this pass's own curation exposed, preserved under
`runs/attempts/` and bound by tests.

Latest: **the resolution lever is measured at last, and finer sampling buys ranking rather than fit**
([M3_OCCUPANCY_RESOLUTION_R2.md](M3_OCCUPANCY_RESOLUTION_R2.md), predeclared before its receipts,
`docs/evidence/m3-occupancy-resolution-r2/scope.json`). The first declaration was refused 22.8 MiB
short of a self-imposed 1,152 MiB ceiling, so the stride-1 detail tap had never been measured; it is
now re-declared at an admitted 1,792 MiB and run as a matched 64-step continuation of the published
checkpoint. **It is the first read-out lever to clear the +0.05 attribution floor**: best swept F1
**0.472012** against **0.412034** (**+0.059978**), AUC **0.868612** against **0.822643**, F1 at `0.0`
**0.413858** against **0.381640**, both arms 0 of 28 exact boards, lever peak 1,500,233,728 B of
1,879,048,192 B. The calibration-neutral decomposition says which part of that is real: the
fixed-boundary delta **+0.032218** clears its own floor (interval **+0.016476 .. +0.039566**, MDE
`0.016226`) while the out-of-fold delta **+0.060355** is **not resolvable** at twelve clusters
(interval -0.138621 .. +0.179634, MDE `0.222851`), so part of the gain is an offset the lever shifts.
Both absolute gates still fail - AUC `0.031388` short of 0.90, best F1 `0.227988` short of 0.70 - and
the fit gate stays FALSE. The sampling family is now closed with a number instead of a refusal, and
the residual the campaign has already attributed is registration, worth `+0.1352` F1 when corrected
at 4x budget: three times this lever. V-097 and V-098 record two join defects this pass exposed in
its own curation, both preserved under `runs/attempts/` and bound by tests.

Latest: **every power figure is generated, suite-checked, and the guard is itself exercised by mutation**
([M3_MINIMUM_DETECTABLE_EFFECT.md](M3_MINIMUM_DETECTABLE_EFFECT.md),
`docs/evidence/m3-power-mde-r3/result.json`). The detectable floor is `0.083151` for the
budget lever and `0.126475` for the schedule at 80% power, with both observed magnitudes
inside it and power at a true `+0.05` of `0.3917` and
`0.1981`; the audit's `+0.149` stays inside its own design's floor
(`0.196212`, power `0.5665`), and the 36 clusters carry a design effect
against cells of `16.8` and `29.2`.
The method document carries no typed number, and `tests/test_power_evidence.py` recomputes the whole
analysis from the nine committed receipts, compares the committed evidence field by field and the
document's block with `render`, and mutates a block figure, a `20261103` decomposition row and an
evidence number to prove it rejects them. The between-population variance rests on 2 df, so the
`+0.149` recommendation is a band - `3 / 5 / 83 seeds and 1.20 / 1.99 / 33.11 arm-hours` (low / point / high), `3 to 83` seeds and
`1.20 to 33.11` arm-hours - against `44 seeds, 528 clusters, 17.55 arm-hours` for the campaign's own floor.
The point estimate clears `+0.149` by about 0.003, so the weak end is what to plan against. Nothing
re-reads either verdict.

Latest: **the two compute levers do not replicate, so the compute closure is retired rather than
confirmed** ([M3_COMPUTE_REPLICATION.md](M3_COMPUTE_REPLICATION.md), predeclared before its receipts,
`docs/evidence/m3-compute-replication-r1/result.json`). Seeding is the data here: `sample_case`
derives each scene from `seed + group_index`, so a seed is a different synthetic population, and a
continuation refuses a foreign seed, so the audit's exact arms cannot be pseudo-replicated. A
fresh-baseline block of three seeds x three arms (128-step control, 192-step budget, cosine 128),
with matched starts inside each seed and nine sweeps that each reproduced their own published read,
reads **1.5x fit budget -0.0796 (-0.1380 .. -0.0250)** and **cosine +0.0213 (-0.0721 .. +0.1136)**
against a predeclared +0.05 floor on 36 paired clusters. **Both are `unconfirmed`** - point below the
floor, lower bound not above zero, one seed reversing in each - and the budget interval excludes zero
*below* control, the opposite direction to the audit's +0.149 continuation. Controls clear the 0.70
AUC floor (0.800/0.826/0.917) while spreading 0.12 across populations, more than either lever moves,
so a one-trajectory delta cannot separate lever from population. Under the predeclared rule the
compute closure stays retired but unconfirmed, and the width `capacity_null` stands as the remaining
actionable verdict; arbitrating the 256-vs-384 contrast needs its own block, not a re-read here.

Latest: **the standing occupancy attribution is a boundary, not a model difference, and two compute refusals reverse**
([M3_OPERATING_POINT.md](M3_OPERATING_POINT.md), predeclared before its receipt, `docs/evidence/m3-operating-point-r2/result.json`).
Every occupancy lever was read at a fixed threshold `0.0`, and the levers move that boundary: the
fit-budget arm's best swept threshold is `4.35` against its control's `0.15`. With the boundary chosen
on the other eleven ancestry groups, **7 of 14** committed pairs reverse the standing attribution - 6
refused arms clear the +0.05 floor and 1 credited gain does not - and **2** survive their own paired
whole-group interval: the 1.5x fit budget (**+0.149** out-of-fold F1, interval +0.006..+0.277, AUC
+0.074, placement better) and the cosine schedule (**+0.084**, +0.010..+0.168, AUC +0.041, placement
39.9% better), refused by the standing read at **-0.134** and **+0.005**. **No gate moves:** 0 of 28
exact nonempty boards at `0.0`, 0 of 28 under the out-of-fold boundary in every arm but the cosine
arm's 1, and no arm's out-of-fold F1 exceeds 0.533, so the model is still far from fit - but "budget is
not the limiter" and "more optimisation is not a substitute" were artifacts of the read. The width
`capacity_null` is unaffected (it read best swept F1 and AUC). Ranked next: a declared,
calibration-neutral re-read of the two compute-side arms with replicate arms, not another lever at
`0.0`.

Latest: **the decision-path width lever is a null, and the wider arm is worse at equal steps**
([M3_TRUNK_WIDTH_LEVER.md](M3_TRUNK_WIDTH_LEVER.md), `docs/evidence/m3-trunkwidth-r1/scope.json`). Two
fresh-baseline arms differing only in `dimension` (192 vs 240; 8,009,097 vs 10,565,241 parameters) trained to
the declared 256 steps and were read on the declared deciding population (12 groups, 36 cases, 28 nonempty
boards, 7,200 known cells, 108 forwards, peak 679,141,376 B of 1024 MiB). **Verdict `capacity_null`:**
+0.0322 best-F1 attribution against the +0.05 floor, F1 at threshold 0 down 0.020, 0 of 28 exact boards in
either arm, and a development read that is *worse* for the wider arm (AUC 0.9268 -> 0.8468, geometry 2.19 ->
5.97 cells). The activation floor held (37.5% of known cells flipped) and the control cleared the AUC floor,
so the pass speaks: 31.9% more decision-path parameters do not move the deciding read, and at step parity they
cost fit. Capacity is not the missing lever here; any future capacity pass must be compute-matched and aimed at
what the representation carries. The fit gate stays FALSE and no prior verdict moves.

The decision path itself was audited before this verdict was trusted: `final.occupancy` is one logit
channel per cell over 26 rows (20 visible plus a 6-row spawn margin), so the probe's row slice already
read the whole visible board, and the literal `20` is now a config-derived slice whose corrected re-run
reproduces this verdict to the digit (`docs/evidence/m3-trunkwidth-r2`).

Previous: **structural and wider changes are authorized, so the decision-path width lever is now executable
([M3_TRUNK_WIDTH_LEVER.md](M3_TRUNK_WIDTH_LEVER.md), predeclared before any receipt). The owner authorized all
structural and wider changes; the old A/B fork is closed as *authorized* and the lever replaces it. Two design
consequences are recorded rather than assumed: the priced wide A widens the encoder and leaves the decision path at
`dimension` 192, which is the wrong end of the model given the `bias_limited` headroom verdict, so the declared
lever is `dimension` 192 -> 240 on the published widths; and the required matched start cannot be exact because
`GroupNorm(8, c)` statistics change under partial channel duplication while uniform 2x duplication exceeds the
256-channel cap, so the pass is a **fresh-baseline pair** whose numbers are explicitly incomparable to the current
table. Nothing is trained yet: the identity is derived from the verified configuration (`TetrisStateNet-M3` keeps
its name, the wider shape becomes `...-W48x96x192x256-D240`), the knob defaults to published, and a bounded pilot
must clear the 2048 MiB envelope before either arm is admitted. The fit gate stays FALSE and no prior verdict
moves.


([M3_DECIDING_HEADROOM.md](M3_DECIDING_HEADROOM.md), predeclared before any receipt,
`docs/evidence/m3-deciding-headroom-r1/scope.json`). Two committed arms of the identical published
architecture - same initial tensor file, seed and data order, differing only in the learning-rate schedule -
were both reloaded and swept in one read-only execution on the declared deciding population (12 groups, 36
cases, 28 nonempty boards, 667 occupied cells), and their **unweighted mean logit** was read by the
predeclared three-branch rule. The members reproduce their own sweeps exactly (AUC 0.8457 / 0.8864;
F1@0 0.4307 / 0.4357). The ensemble reaches AUC **0.8872** and best swept F1 **0.5074**,
**-0.0099** against the better member and short of both bars, so the verdict is **`bias_limited`**: the
two trajectories disagree on 169 of 667 occupied cells, yet combining them buys no
separation, so the residual is a representational ceiling shared by both solutions rather than sampling
variance. Nothing is trained, no weight or parameter changes, the fit gate stays FALSE, and no prior
verdict moves. Consequence: **ensembling and more optimisation are not a substitute for the priced
trunk-width fork** ([M3_TRUNK_WIDTH_PRICING.md](M3_TRUNK_WIDTH_PRICING.md)), which remains the user's
decision; the variance measured here is a lower bound because the pair shares its initialisation.

Latest: **the capacity branch is inexpressible, not untested: capacity 64 is a slot bound, and the trunk has no width knob**
([M3_OCCUPANCY_TRUNK_WIDTH.md](M3_OCCUPANCY_TRUNK_WIDTH.md), committed before any launch, with no receipt in
existence). The fit-budget rule ranked *"a wider shared trunk than capacity 64"*; building the declared
configuration at capacity 32 and 64 returns **identical** censuses (8,018,505 parameters over
271 tensors), so capacity gates the component decoder rather than any width, `ModelConfig`
refuses anything above 64, the driver refuses it earlier with the renderer's ten-slot reason attached, and the
encoder widths `(48, 96, 192, 256)` with dimension `192` have no training flag at all. Widening the trunk
would change the shape of every trunk tensor, so the published checkpoint cannot be continued into it and the
matched-start contract cannot be met by added tensors: it is a new architecture identity, and therefore a user
decision rather than an arm. Nothing was trained, swept or probed, no occupancy number is claimed, and the
fit gate stays FALSE. The next step is a declared width lever with a fresh baseline and a measured per-step
cost, or the standing limit.
Previous: **the M2 fit gate is measured at 1.5x budget and stands (FALSE): the extra compute moved decisions without reproducing a single board**
([M3_OCCUPANCY_FIT_BUDGET.md](M3_OCCUPANCY_FIT_BUDGET.md), predeclared at `875543d`,
`docs/evidence/m3-occupancy-fit-budget-r1/scope.json`). The declared change was **budget only**: the same
architecture for **384 continuity steps against the settled 256**, with **no new tensor in either arm**
(the lever's declared added-parameter count is exactly zero), so this is the first budget lever on the
reconciled read-out and the first time the standing FIT-GATE-FALSE sentence was measured as a gate. The
criterion is the repository's own: `fit_gate.fit_gate_opens` needs **3 of 28** exact nonempty training
boards at threshold `0.0`, and the lever reports **0 of 28**
(mechanism `per-cell-error`). The pass also declared a behavioural activation floor, answered
at **0.5481** of known deciding cells flipping against **0.05**
(engaged). Both arms ran from `9ae1fef9f649...` under one catalogue (peaks
0.962 GB / 0.987 GB,
1522.4 s / 1046.2 s). F1@0 **0.2808** against
**0.4146**, AUC **0.8998** against **0.8255**, best swept F1
**0.5459** at `4.35` against **0.4237**, attribution
**+0.1221**, placement 1.9812 -> 1.8419 cells. The
rule takes its **`capacity_limit`** branch and names **a wider shared trunk than capacity 64**.
Previous: **the per-board refinement capacity lever is refused: its channel never engaged, and the arm
regressed anyway** ([M3_OCCUPANCY_BOARD_REFINEMENT.md](M3_OCCUPANCY_BOARD_REFINEMENT.md), predeclared at
`72f95bc`, `docs/evidence/m3-occupancy-board-refinement-r1/scope.json`). The declared single change was
capacity per board: one board-conditioned four-vector, predicted from the board token, correcting the
decoded lattice in cell space (origin offset plus log-scale per axis) and applied identically at every
stage, for **12,612 declared parameters** in four zero-initialized tensors. The pass also declared an
**activation floor** after the read-out null, and that floor is what the measurement answers: the applied
correction reached only **0.001428 cells** against the declared **0.10** and
**0.006048 axis scale** against **0.02**, so the channel is **not engaged** and
this pass cannot speak for per-board refinement capacity. Both arms ran 256 steps from `9ae1fef9f649...` under
one catalogue at 1,152 MiB / 1,800 s (peaks 0.997 GB /
0.957 GB, 1059.9 s /
1019.5 s). F1@0 **0.4232** against **0.4146** (570/1457/97 against
351/675/316), AUC
**0.8596** against **0.8255**, best swept F1 **0.4776** at
`0.9` against **0.4237** at `-0.3` (attribution
**+0.0539**, twelve-cluster interval **-0.1341 to +0.0740**),
and placement **19.2% worse** with the error spread
4.6043 -> 5.4599. Because the correction was
a thousandth of a cell, the regression is a trajectory effect of the inert parameter addition, not the
refinement. The rule takes its **fit_gate** branch and names **the M2 fit gate at capacity 64**.
Previous: **the lattice-agnostic read-out lever is refused: the gate barely opened, and the free channel
bought placement rather than occupancy** ([M3_OCCUPANCY_LATTICE_FREE.md](M3_OCCUPANCY_LATTICE_FREE.md),
predeclared at `7b3ca9a`, `docs/evidence/m3-occupancy-lattice-free-readout-r1/scope.json`). The declared
single change was the occupancy read-out: a channel built from the encoder's image-level context and
fixed cell coordinates, never from the predicted lattice, blended with the published logits through
`logits = (1-gate)*published + gate*free` with `gate = tanh(raw)` - **4,226 declared parameters** in five
zero-initialized tensors, so step zero was bitwise the published function and `|gate| -> 1` suppresses
the lattice-sampled path and its detail tap. Both arms ran 256 steps from `9ae1fef9f649...` under one catalogue
at 1,152 MiB / 1,800 s (peaks 1.082 GB / 1.025 GB, 1,057.0 s / 1,025.0 s). F1@0 **0.4284**
against **0.4146** (371/694/296
against 351/675/316),
AUC **0.8401** against **0.8255**, best swept F1 **0.4384** at
`1.65` against **0.4237** at `-0.3` (attribution
**+0.0147**, twelve-cluster interval **-0.0836 to +0.1284**,
median **+0.0150**), and placement **22.3% better** (1.9812 -> 1.5393)
with the error spread falling 4.6043 -> **3.3544**. The trained gate reached
`|tanh| = 0.0742`, so the channel carried only ~7% of the
logits: the change was barely activated, all four conditions fail, and the rule's spread branch names
**per-board lattice refinement capacity**.
Previous: **the per-stage lattice-residual lever is refused: independently correctable stage boxes
moved the lattice and lost both scores** ([M3_OCCUPANCY_LATTICE_RESIDUAL.md](M3_OCCUPANCY_LATTICE_RESIDUAL.md),
predeclared at `473440c`, `docs/evidence/m3-occupancy-lattice-residual-r1/scope.json`). The declared
single change was capacity, not supervision: each of the six decoder stages gained a zero-initialized
4x4 projection applied to its decoded lattice - origin offset in cells along its own axes plus a
log-scale per axis - for **120 declared parameters**, supervised by the existing geometry term, so step
zero was bitwise the published function. Both arms ran 256 steps from `9ae1fef9f649...` under one
catalogue at 1,152 MiB / 1,800 s (peaks 1.060 GB / 1.057 GB, 826.1 s / 1,017.6 s). F1@0 **0.3520**
against **0.4146** (661/2,428/6 against 351/675/316), AUC **0.8209** against **0.8255**, best swept F1
**0.4122** at `2.25` against **0.4237** at `-0.30` (attribution **-0.0115**, twelve-cluster interval
**-0.2795 to +0.0300**), and placement **19.2% worse** (1.9812 -> 2.3615) while the column extent ratio
shrank to **0.9084** and 71% of cells were shifted. All four conditions fail; the predeclared
placement-regressed falsifier holds, and the rule names **a lattice-agnostic occupancy read-out**.

Previous: **the geometry-capacity lever is refused: spending a full-weight term on the boundary
support anchors lost occupancy and placement together** ([M3_OCCUPANCY_CAPACITY.md](M3_OCCUPANCY_CAPACITY.md),
predeclared at `8211a5f`, `docs/evidence/m3-occupancy-geometry-capacity-r1/scope.json`). The declared
single change was a boundary support-anchor geometry term on the final lattice - eight declared corner
and edge-midpoint anchors, smooth-L1 in normalized cell units, at weight `1.0` equal to the geometry
term's own weight - with no new parameter, input, stage policy or read-out, replacing the 2% share the
boundary had inside the published geometry mean. Both arms ran 256 steps from `9ae1fef9f649...` under
one catalogue at 1,152 MiB / 1,800 s (peaks 1.034 GB / 1.056 GB, 946.6 s / 701.7 s). F1@0 **0.2017**
against **0.4146**, AUC **0.8116** against **0.8255**, best swept F1 **0.3996** at `-1.95` against
**0.4237** at `-0.30` (attribution **-0.0241**, twelve-cluster interval **-0.2795 to -0.0696**), and
placement **28.2% worse** (1.9812 -> 2.5401) with column extent ratio **+8.8%**. All four declared
conditions fail in the wrong direction and the predeclared extent-or-origin regress falsifier holds. The
non-circular rule names **per-stage lattice residual capacity**.

Previous: **the box-extent lever is refused: the extent term inflated the box, and the residual is now
the geometry supervision capacity** ([M3_OCCUPANCY_EXTENT.md](M3_OCCUPANCY_EXTENT.md), predeclared at
`e5e9668`, `docs/evidence/m3-occupancy-extent-r1/scope.json`). The declared single change was a relative
log-scale term on the final decoder lattice - the mean absolute log ratio of the predicted to the target
basis norms over registered boards, at weight `1.0` equal to the geometry term's own weight - with no
new parameter, input or read-out. Both arms ran 256 steps from `9ae1fef9f649...` under one catalogue at
1,152 MiB / 1,800 s (peaks 1.067 GB / 1.067 GB, 814.2 s / 1,120.7 s). It made every score worse: F1@0
**0.3315** against **0.4146**, AUC **0.7526** against **0.8255**, best swept F1 **0.4556** at `2.10`
against 0.4237 at `-0.30` (attribution **+0.0318**, twelve-cluster interval **-0.1345 to -0.0313**,
median -0.0848), the column extent ratio **rose** 1.2257 -> 1.5255 while only the row axis fell 1.1804
-> 1.1466, and mean placement error rose **67.5%** (1.9812 -> 3.3182). All four declared conditions
fail, so the verdict is **`not-cleared`**. Reconciled against the geometry lever (+0.0781 occupancy at
19.9% placement cost) and the origin lever (+0.0102 at 11.0%), this is a third and larger placement
cost carrying only an operating-point gain. The declared non-circular rule takes its first branch
(smallest extent reduction -1.075 < 0.25) and names **geometric supervision capacity (support
anchors)**. Ranked next: **predeclare a support-anchor capacity lever**, read with the same four
conditions.

Latest: **the box-origin (translation) lever is refused: the origin did not move, and the residual is
box extent** ([M3_OCCUPANCY_TRANSLATION.md](M3_OCCUPANCY_TRANSLATION.md), predeclared at `24ae9b9`,
`docs/evidence/m3-occupancy-translation-r1/scope.json`). The declared single change was a
zero-initialized `Linear(5,4)` over the coarse feature map's spatial moments (energy centroid x/y, spread
x/y, log energy) added to `raw_box`, so the lattice origin could depend on where the board is. Both arms
ran 256 steps from `9ae1fef9f649...` under one catalogue at 1,152 MiB / 1,800 s (peaks 1.086 GB / 1.083 GB,
380.8 s / 380.7 s). It moved the operating point, not the ranking: F1@0 **0.4165** against **0.4146**
(trading 271 misses for 1,023 false positives), AUC **0.8446** against **0.8255**, best swept F1 **0.4340**
against **0.4237** - attribution **+0.0102**, with the twelve-cluster interval **-0.1700 to +0.0872**
(median -0.0378). The signed offsets went the wrong way - columns **+1.3830 -> +1.3998** (-1.2%), rows
**-2.1914 -> -2.6181** (-19.5%) - and mean placement error rose **11.0%** (1.9812 -> 2.1988), so all four
declared conditions fail and the verdict is **`not-cleared`**. Reconciled against the geometry lever: its
+0.0781 occupancy gain and 19.9% placement cost were not repaid by moving the origin - this lever buys an
eighth of that gain and adds another 11.0% placement cost. The declared non-circular rule now names **box
extent** (smallest signed reduction -0.19 < 0.25; measured ratios 1.3242 / 1.1709, both past the 0.10
log-ratio floor). Ranked next: **predeclare a box-extent supervision lever**, read with the same four
conditions.

Latest: **the geometry-stage lever is refused: it moves occupancy, not placement, and the measured
residual is a box translation** ([M3_OCCUPANCY_GEOMETRY.md](M3_OCCUPANCY_GEOMETRY.md), predeclared at
`35ae175`, `docs/evidence/m3-occupancy-geometry-r1/scope.json`). The declared single change spent the
lattice geometry term on the final decoder stage - the only lattice the occupancy read-out samples -
where the published objective averages six stages. Both arms ran 256 steps from `9ae1fef9f649...`
under one catalogue and one identical initial tensor file at 1,152 MiB / 1,800 s (peaks 1.087 GB /
1.058 GB, 396 s / 400 s). The lever improves the occupancy decision - best swept F1 **0.5018** at
`0.45` against the matched baseline's **0.4237** at `-0.30`, F1@0 **0.4791** against **0.4146**, AUC
**0.8316** against **0.8255**, attribution **+0.0781** (interval -0.0152 to +0.1243) - but placement
got **19.9% worse** (1.9812 -> 2.3757 cells, shift share 0.5375 -> 0.5816), so `per_cell_f1`,
`ranking` and `placement` all fail and the verdict is **`not-cleared`**. The declared decomposition
then hits its first branch: signed offsets **+2.32 cells right and -2.38 cells down**, with extent
ratios 1.22 / 1.14 and a 5.46-cell per-board spread. Ranked next: **predeclare a box-origin
(translation) lever**, read with the same four conditions, targeting the signed offsets toward zero.

Latest: **the 4x budget lever is refused, and the occupancy residual is registration**
([M3_OCCUPANCY_SCALED.md](M3_OCCUPANCY_SCALED.md), predeclared at `1ce41e6`,
`docs/evidence/m3-occupancy-scaled-r1/scope.json` and
`docs/evidence/m3-occupancy-registration-r1/scope.json`). Both arms ran 256 continuation steps from
the published `9ae1fef9f649...` checkpoint under one captured catalogue, 1,152 MiB and 1,800 s, and
both stayed inside the envelope (peaks 1.066 GB / 1.067 GB, 396 s / 395 s). The stencil arm did not
carry its advantage: best swept F1 **0.3925** at `0.60` against the matched control's **0.4237** at
`-0.30`, AUC **0.8006** against **0.8255**, F1@0 **0.3791** against **0.4146** - attribution
**-0.0312**, so **all three standing conditions fail** and scaling is refused at the declared
envelope (against the unmatched 64-step stencil record it is -0.0988 F1, reported but never read by
the verdict). The 4x budget actually inverted the 64-step ordering the lever was ranked on, to
0.8006 AUC from 0.8804. The declared fallback then attributed the residual: on the 2,986 cells both
pairings can score, pairing each model cell with the label of the target cell it physically covers
raises F1 **0.4390 -> 0.5742** (+0.1352, cluster interval +0.0460 to +0.2833, AUC 0.9331 -> 0.9702),
with a mean placement offset of **2.57 cells** and 48.3% of cells landing on the wrong target cell,
so **`registration_implicated`**. Ranked next: **attack predicted-lattice placement** - the
occupancy read-out samples at the model's own lattice, so its indices inherit that error - read with
the same three gates plus a declared placement-error measurement. Budget scaling on this form is
closed at 4x; the label and read-out families stay closed.

Latest: the **label lever is refused before admission, because the committed labels are pixel-separable**
([M3_OCCUPANCY_PIXEL_CEILING.md](M3_OCCUPANCY_PIXEL_CEILING.md), predeclared at `6e40206`,
`docs/evidence/m3-occupancy-pixel-ceiling-r1/scope.json`). The stacking pass ranked an explicit
indeterminate cell target next on a label-overlap signature; this read-only pass measured the premise
instead of assuming it. On the committed 28 nonempty deciding boards (667 occupied / 4,933 empty cells,
twelve ancestry groups), a logistic read-out over **30 per-cell pixel features** that never sees the
model, with one ancestry group held out, scores occupied-cell F1 **0.9780** (667/30/0) and mean
per-fold AUC **0.9976** (per-fold 0.9799-1.0000) against the standing 0.70 / 0.90 bars, while the
label-shuffled control in the identical pipeline collapses to F1 **0.0029** and fold-mean AUC
**0.4885**. The model's own committed numbers on the same cells are F1 **0.4019** and AUC **0.7960**.
So the targets are already recoverable from this capture under oracle registration: an explicit
indeterminate target cannot supply the missing separation, `indeterminate_target_licensed = false`,
and no label-policy variant is admitted. Nothing moved - no model, loss, threshold, label, gate or
prior evidence byte. Ranked next is therefore **model-side**: scale the matched budget on the 2x2
stencil configuration (the only arm to meet the +0.05 attribution floor, AUC 0.8804 / best F1 0.4913),
with **registration** named as the untested alternative suspect, because the ceiling was granted the
target lattice while the model's geometry error is about 3.3 cells. The refused resolution lever still
reopens only as its own declared configuration, and the real-image chain stays parked pending
authorization.

Latest: the **context+detail tap stack is measured: not cleared, and it orders worse than either
half** ([M3_OCCUPANCY_STACKING.md](M3_OCCUPANCY_STACKING.md), predeclared at `4135a2b`,
`docs/evidence/m3-occupancy-stacking-r1/scope.json`). The declared change was the composition itself -
the context residual used to overwrite the detail tap, so the two-flag model ignored one tap; the pass
summed the two existing zero-initialized residuals and added no parameter. Against the detail tap
alone at the identical 64-step budget, gates, 1,152 MiB ceiling, population and catalogue, the stack
scores pooled AUC **0.7960** against **0.8226** (cluster delta **-0.0463 to -0.0189**), best swept F1
**0.4097** against **0.4120**, 0/28 exact boards, and precision 0.2521 against 0.2363 (+0.0158, below
its 0.05 floor, so `precision_improved = false`). It is **-0.0844 AUC** and **-0.0816** best F1 short
of the stencil arm and **-0.0284 / -0.0103** short of the context arm, so stacking is closed. The
control reproduced the stencil pass's control bit-for-bit, so the negative is not drift. Next-ranked:
**leave the read-out and target the labels** - an explicit indeterminate/occluded cell target scored
on determinate cells - because every read-out lever moved ordering (0.7427 -> 0.8244 -> 0.8226 ->
0.8804) while the swept-F1 deficit stayed 0.21-0.29 and no board is exact at any threshold. The
real-image chain stays parked pending authorization.

Latest: the **encoder-resolution lever is refused at the declared envelope, not scored**
([M3_OCCUPANCY_RESOLUTION.md](M3_OCCUPANCY_RESOLUTION.md), predeclared at `eee9362`,
`docs/evidence/m3-occupancy-resolution-r1/refusal.json`). Its single declared change - a stride-1
convolution of the captured image feeding the same zero-initialized detail projection - breached the
1,152 MiB ceiling on its 2-step admission profile: peak **1,231,880,192 B** against 1,207,959,552 B,
while the same-session stride-2 control peaked at 1,034,313,728 B, so the branch costs
**197,566,464 B (188.4 MiB)** and the envelope is short by 23,920,640 B (22.8 MiB). No ceiling was
raised, no smaller variant substituted, no capture side reduced and nothing re-run;
`precision_improved = false` because precision was never measured, and the gates stand unanswered.
The estimate error (one map against four retained) is ledger entry **V-062**. Next-ranked: **stack
`occupancy_context` with `occupancy_detail`** - the last information-changing lever, at about
0.02 MiB of retained activation - then scale; resolution reopens only as a separately declared
configuration, and the real-image chain stays parked pending authorization.

Latest: the **2x2 in-cell detail stencil is measured: not cleared, but it met the attribution floor
and closed most of the ordering deficit** ([M3_OCCUPANCY_STENCIL.md](M3_OCCUPANCY_STENCIL.md),
predeclared at `3ecbf2a`, `docs/evidence/m3-occupancy-stencil-r1/scope.json`). Both arms carried the
declared zero-initialized detail tap and differed only in its form - four quarter-cell samples instead
of the single center sample - continued 64 matched steps from the published checkpoint under one
catalogue, both inside 1,152 MiB. The stencil is the best-ordering arm of the campaign: pooled AUC
**0.8804** (cluster delta +0.0385 to +0.0943), **0.020 short** of 0.90, and its best swept F1
**0.4913** beats the single-sample control's 0.4120 by **+0.0793**, clearing the +0.05 attribution
floor for the first time. Standing-point precision moved only 0.2363 -> 0.2474 (below its 0.05
mechanism floor, so `precision_improved = false`), so the extra capacity bought ordering, not
precision; the gate still fails (best F1 **0.209 short** of 0.70, 0/28 exact boards) because
**separation, not ordering, now binds**. Encoder resolution was the next-ranked step and has since
been **refused at the declared envelope** (above), so next is stacking the two read-out taps, then
scale.

Latest: the **cell-precision representation lever is measured: not cleared, but its mechanism worked**
([M3_OCCUPANCY_PRECISION.md](M3_OCCUPANCY_PRECISION.md), predeclared at `684d775`,
`docs/evidence/m3-occupancy-precision-r1/scope.json`). A declared zero-initialized stride-2 detail
tap at the cell centers — identity at step zero — continued 64 matched steps from the published
checkpoint against a fresh control under one catalogue, both inside 1,152 MiB. Precision at the
standing point rose **0.1752 -> 0.2363** (cluster delta 0.0104-0.0939, excluding zero) *and* recall
rose 0.7616 -> 0.9910, so standing-point F1 gained **+0.0968** (0.2848 -> 0.3816) — the largest of
any read-out lever so far. No gate cleared: AUC 0.8226 is **0.077 short** of 0.90, best swept F1
0.4120 is **0.288 short** of 0.70, 0/28 boards are exact, and the swept optimum stays below the
control's (0.4338 at `3.90`). So precision is addressable from the read-out, but read-out
information at this budget is not separation. Next-ranked: a 2x2 in-cell detail stencil through the
same projection, then encoder resolution, then scale.

Latest: the **occupancy representation question has a measured verdict** — not cleared, with a named
mechanism ([M3_OCCUPANCY_REPRESENTATION.md](M3_OCCUPANCY_REPRESENTATION.md), predeclared at
`a07f9d2`, `docs/evidence/m3-occupancy-representation-r1/scope.json`). A declared zero-initialized
read-out neighborhood (eight lattice neighbors before the occupancy head, identity at step zero)
was continued 64 matched steps from the published checkpoint against a control continuation from
the same bytes, both inside 1,152 MiB. The lever raised pooled AUC **0.7427 -> 0.8244** (cluster
0.7491 -> 0.8199) and the standing-point F1 **0.2848 -> 0.3600** (cluster delta 0.0142-0.1106), but
its recall at `0.0` went 0.73-0.79 -> 0.985-0.998, its **best swept F1 was slightly lower**
(0.4200 against 0.4338 at `3.90`) and **no gate cleared**: 0/28 exact boards, AUC 0.076 short of
0.90, best F1 0.28 short of 0.70, attribution -0.0138 against the declared +0.05 floor. So the
block moves ordering and coverage, not separation — the mechanism the conditioning pass named
still binds. Next-ranked step: a representation targeting cell-level **precision** (contrastive or
boundary-aware detail at the lattice); threshold, conditioning and class-budget families stay
closed.

Latest: the occupancy failure is measured to be **per-board, not per-boundary**, so the last open
model question is now ranked by evidence ([M3_OCCUPANCY_CONDITIONING.md](M3_OCCUPANCY_CONDITIONING.md),
predeclared at `deaba0f`, `docs/evidence/m3-occupancy-conditioning-r1/scope.json`). Read-only on
the committed widened cases with the standing 81-point grid and 0.70 bar: **0 of 28** nonempty
deciding boards reach the bar even at their own oracle grid point (floor 4), so the verdict is
`representation-limited` and no conditioned threshold can exist. Even with one threshold per board
chosen using the labels, the oracle-pooled F1 is **0.4305** (551/1,342/116), 0.27 short; median
per-board F1 at the standing `-0.90` point is **0.0857**, while per-board AUC runs 0.291 /**0.829** /
0.970 — the ordering is real, the separation is not. Density offsets are recorded descriptively but
license nothing. This closes the decision-boundary family **globally and conditioned**, leaves the
ranking verdict untouched, and moves no model, loss, threshold, gate or evidence byte; the ranked
next lever is the representation.

Latest: the **deciding population is widened and both verdicts re-derive on it unchanged and
tighter** ([M3_WIDENED_DECIDING_POPULATION.md](M3_WIDENED_DECIDING_POPULATION.md), predeclared
at `111aa9e` before any receipt, `docs/evidence/m3-widened-deciding-population-r1/scope.json`).
The training deciding population moves from 3 groups / 9 images / 7 nonempty boards / 168
occupied cells to **12 groups / 36 images / 28 nonempty boards / 667 occupied cells**, with the
fit-gate threshold 3, the F1 bar 0.70, the AUC bar 0.90, the 81-point grid and the identity
class bar 3 all unchanged. Occupancy stays **UNINFORMATIVE**: AUC **0.7852** with 95% CI
0.7640-0.8064 against 0.90 (narrow 0.7491, CI 0.7049-0.7934), best swept F1 **0.3704** against
0.70 (narrow 0.3470), the replay's own operating point at F1 **0.3455**, gap 0.3545 (narrow
0.3109), and the fit gate **0/28** exact nonempty boards (narrow 0/7). Identity still
**stands**: the queue head emits `EOS` at all **216** supervised training positions (narrow 54)
and the hold head `J` at all 33 cases (narrow 9), so the constant-predictor finding was not
small-`n`. The narrow numbers are **superseded without invalidation** - the recorded nine-case
subset reproduces 67/196/101 and the published identity classes exactly.

Addendum (2026-11-09): those intervals are **cell-count** intervals, and the deciding cells are
nested (28 boards of 200 cells in 12 ancestry groups). Measured clustering is small on the pooled
per-cell error (ICC 0.0009-0.0092, design effect 1.19-5.30, **232-562 effective occupied cells**
rather than 667) but large on *precision* alone (design effect 14.4x-76.0x, so its standard error
was inflated **3.80x-8.72x**; recall 0.25-1.99).
Correlation-respecting 95% intervals: AUC **0.7474-0.8186** (cell 0.7640-0.8064) still 0.081
below the 0.90 bar, precision at `0.0` **0.0908-0.4538** (cell 0.2548-0.3088), recall at `0.0`
**0.4076-0.4695**, empirical F1 at `0.0` 0.1488-0.4538, hold exactness **0.1818-0.7273** (design
effect 3.3); the queue's 0/216 census has design effect 1.0 and cannot widen. **No verdict
changes** - occupancy stays UNINFORMATIVE, the per-cell gate stands and is strengthened, identity
stands - and nothing is re-swept, re-litigated or moved off its threshold (V-056).

Latest: the **occupancy ranking sweep is measured and returns UNINFORMATIVE**, closing the
occupancy head and exhausting the operating-point family
([M3_OCCUPANCY_RANKING.md](M3_OCCUPANCY_RANKING.md), rule predeclared at `9ad3598` before any
receipt, `docs/evidence/m3-sweep-occupancy-r1-ranking-r1/scope.json`). Read-only, on the
committed `m3-percell-balance-r1` checkpoint over the fixed 81-point grid: training AUC
**0.7491** against chance 0.5 (bar 0.90) and no threshold on the grid reaches the required
occupied-cell F1 0.70 - the best point is -0.90 with precision 0.2393, recall 0.6310, F1
**0.3470**, 0/7 exact boards and 122/400 empty-population false positives. Development AUC
0.8712, best F1 0.5456 at -0.45, 0/14 exact boards. The grid's zero row reproduces the
replay's 67/196/101 counts, so the sweep joins the earlier receipt instead of restating it.
Consequence: the occupancy head of this representation is **closed** and the
operating-point family is **exhausted**; the logits order occupied above empty better than
chance but not enough for any threshold to approach fit, and the class-budget run's 0.3109
sits below the swept ceiling of 0.3470. Nothing re-opens the fit gate, class weight or scale
lever, and no real-image, complete-state or quality claim follows. Occupancy is now closed
from all three sides (weight scale, class split, decision boundary); the next model question
is a representation that separates occupied from empty cells at this budget, which is not a
threshold choice and is not ranked as a lever here.
Latest: the **per-cell decision lever is measured and STANDS**, with the mechanism labelled
as error reallocation rather than discrimination ([M3_PERCELL_DECISION.md](M3_PERCELL_DECISION.md),
rule predeclared at `0d77cef` before any receipt,
`docs/evidence/m3-percell-balance-r1-errors-r1/scope.json`). The predeclared rule is an
occupied-cell F1 >= 0.70 on the seven nonempty training boards; the independent replay reports
**0.3109** (development 0.5350). The lever was `--occupancy-positive-weight 4`, a class budget
inside the balanced occupancy term with the term's total weight matched at the already-measured
`4.`. It moved the bound as asked and stopped there: recall 34/168 -> 67/168, precision
34/93 -> 67/263, misses 134 -> 101, inventions 59 -> 196, F1 0.2605 -> 0.3109, and
empty-population false positives 23 -> 72 - a decision-boundary shift, labelled
`specificity-limited` from the receipt's own counts. Exact boards stayed 0/7 and 0/14; roles
0/9 and 0/18, ghost 0/9 and 0/18, readable components 0/9 and 0/18 are costs. The family's
queue instability ran the other way this time (EOS 0/9 and 0/15 -> 9/9 and 15/15 while piece
tokens 15/54 and 9/90 -> 0/54 and 0/90) and is inert. Neither the term's scale nor its class
split opens fit, so the ranked next instrument is a read-only threshold sweep of the locked
occupancy logits on the committed checkpoint: it decides whether the logits rank occupied above
empty cells and only the operating point is wrong, in which case the precision/recall budget is
an owner choice, or whether the ranking itself is near chance.
Latest: the **fit gate is measured for the first time and STANDS**, with a mechanism that is
not the one it was ranked on ([M3_FIT_GATE.md](M3_FIT_GATE.md), rule predeclared at
`5065811` before any receipt, `docs/evidence/m3-fit-occupancy-r1-errors-r1/scope.json`).
The predeclared rule is >= 3 exact nonempty training boards; the independent replay reports
**0 of 7** (development 0/14). The lever was `--occupancy-factor 4`, scaling only the
locked-occupancy weight, to test the constant-empty read-out the length run recorded
(`true_positive 0, false_negative 168`). It is not gradient-starved: at 4x the head recovers
34 of 1,400 training occupied cells (230 of 2,800 development), so the term moves, and still
reproduces no board - the errors are per-cell and bidirectional (59 false positives, 134
misses) - while the scale is paid elsewhere: queue EOS 9/9 -> 0/9 and 15/15 -> 0/15,
presence 63/63 -> 54/63 and 105/105 -> 90/105, position spread 0.0877/0.0838 ->
0.0641/0.0379. Queue piece tokens rose to 15/54 and 9/90, slot status stayed exact (54/54,
90/90), roles 0/9 and 0/18, ghost 0/9 and 0/18, geometry 3.3397 and 2.6864, exact readable
components 0/9 and 3/18 (empty-queue cases). Ranked next is the **locked read-out's per-cell
decision** (its class imbalance and cell-wise scoring), measured by the same rule; another
occupancy weight scale, a budget extension of this lever, and any further identity/endpoint
variant are not ranked. `scripts/fit_gate.py` with `tests/test_fit_gate.py` reproduce the
rule and the verdict from the committed bytes. V-053 records the curator log-name defect
this pass exposed.

Latest: the queue-length lever is a measured **negative**, and it **closes** the
class-side / position-side / loss-side disjunction at the fixed matched budget
([M3_IDENTITY_QUEUE_LENGTH.md](M3_IDENTITY_QUEUE_LENGTH.md), predeclared at `d2895de`,
`docs/evidence/m3-identity-factor-length-r1-errors-r1/scope.json`): one length decision per
case still leaves the queue head emitting `EOS` at all 54 supervised training positions
(distinct 1, constant true; 90/90 development) and the hold head on one class (`J`x9 and
x12, 3/9 and 3/12 exact). The measured mechanism is a representation collapse: with no
per-position question left, the per-position logit spread fell from 1.2338 and 1.2304 to
0.0877 and 0.0838, and presence — a per-position term sharing the same read-out — fell from
54/63 and 90/105 to 9/63 and 15/105, while endpoint and slot stayed exact (9/9, 15/15,
54/54, 90/90). Read-out is ruled out (three variants, and the spread follows the objective,
not the architecture), data is ruled out (54 positions, 7 classes, 6 slots, and presence was
54/63 when asked per position), and the loss masking remains: every objective tried admits
a single class everywhere as a minimiser. Structure: geometry 3.6709 and 3.1173, roles 0/9
and 0/18, boards 0/9 and 1/14 nonempty development, exact readable 0/9 and 0/18; the fit
gate is **FALSE**. Disjunction closed on evidence, so the ranked next lever is the **fit
gate itself** (locked occupancy and role read-out), with no further identity or endpoint
variant ranked. V-052 records the admission gate that refused the newly declared objective
and the single declaration that now binds driver, model and gate.

Latest: the endpoint-loss lever is a measured **negative** that moves the constant rather
than removing it
([M3_IDENTITY_ENDPOINT_LOSS.md](M3_IDENTITY_ENDPOINT_LOSS.md), predeclared at `008be3d`,
`docs/evidence/m3-identity-factor-endpoint-r1-errors-r1/scope.json`): with the endpoint as
its own binary decision the queue head still emits `EOS` at all 54 supervised training
positions (distinct 1, constant true; 90/90 development) and the hold head emits `S`x7 and
`T`x2 (distinct 2, still below the predeclared three-class bar; 4/12 exact). Position
information reaches the decision more strongly than ever — per-position logit spread rose
to 1.2338 training and 1.2304 development, `position_axis_reaches_logits` true — and is
still collapsed. The mechanism is measured: the endpoint term is charged at every known
position, 63 training and 105 development of which only 9 and 15 are endpoints (a 1/7
rate), so a constant "endpoint here" answer costs 0.410 nats and preserves EOS 9/9, slot
54/54 and presence 54/63 while the multiplicative factorization suppresses every piece
class. Structure: ghost 9/9 and active 3/9 training exact, boards 0/9 and 0/18, geometry
4.0672 and 2.3110, fit gate **FALSE**. The class-side / position-side / loss-side
disjunction is **not closed**: what this lever closes is where the constant lives — not in
the class read-out, the position read-out, or the endpoint's membership of the piece
softmax, but in the endpoint being scored per position while it is a per-sequence count.
Ranked next is queue length as one global decision (a single distribution over
`0..capacity` instead of a per-position binary question); more read-out conditioning,
budget and re-weighting of the same term are not ranked.

Latest: the per-position identity read-out is a measured **negative** that pins the
remaining constraint to the loss
([M3_IDENTITY_POSITIONAL.md](M3_IDENTITY_POSITIONAL.md), predeclared at `aa5bcc0`,
`docs/evidence/m3-identity-factor-positional-r1-errors-r1/scope.json`): the queue head
still emits `EOS` at all 54 supervised training positions (distinct 1, constant true; 90/90
development) and the hold head emits `T` at all 9 and 12 cases, so both conditions of the
predeclared rule fail. The lever's own diagnostic shows the position axis reaches the
decision: per-position logits differ from each other with max spread 0.3627 training and
0.5758 development and `position_axis_reaches_logits` true, while the argmax stays one
class. With the read-out ruled out (two independent conditioning axes now wired in) and
the data ruled out (54 supervised positions carry seven classes across six slots), the
remaining constraint is the loss: the endpoint is a class in the same 8-way identity
softmax and masked mean as the pieces and is exactly predictable (9/9 and 15/15), so
one class everywhere is the cheapest descent. Structure is at or above control (presence
54/63 and 90/105), and the fit gate stays FALSE (2/9 and 0/18 boards, 0/18 and 0/36
roles, 0/9 and 0/18 components). Ranked next is the loss formulation under the same
discrimination rule; more read-out conditioning or budget is not ranked.

Latest: the per-piece image-query identity read-out is a measured **negative** by its
predeclared discrimination rule ([M3_IDENTITY_PROTOTYPES.md](M3_IDENTITY_PROTOTYPES.md),
predeclared at `bdae6e1`, `docs/evidence/m3-identity-factor-prototype-r1-errors-r1/scope.json`):
on the run's nine training probes the queue-token head still emits **one** class
(`EOS` at all 54 positions, distinct 1, constant true; development 90/90 the same) and
the hold head emits **two** (`L`x3, `O`x6), short of the declared >= 3 on both, so the
verdict is NEGATIVE. The mechanism is measured, not inferred: class vectors are now
image-derived, yet the emitted class is invariant, and it is `EOS` — the one class whose
own term was already exact (9/9 training, 15/15 development, better than the control's
8/9 and 3/15) — so the head takes the constant whose structural terms are free and pays
only the masked piece term. The position read-out (index encoding plus one shared role
vector) is the only position-varying signal left and never separated. The four-query
hold bank is the one part that moved off a constant (2 training and 3 development
classes, 3/9 exact against 0/9), so the ranked next lever is the **position side** of the
read-out, not another class-side or budget change. Structure is recorded as a cost
(presence 54/63 -> 39/63 and 90/105 -> 48/105, roles 12/18 -> 0/18, EOS improved), and
the fit gate stays FALSE (2/9 and 0/18 boards, the two training boards both empty,
0/18 and 0/36 roles, 0/9 and 0/18 components).

Latest: the M3 identity heads are measured to be **constant predictors**, so no identity
count on any closed run is evidence of identity reading
([M3_IDENTITY_DISCRIMINATION.md](M3_IDENTITY_DISCRIMINATION.md),
`docs/evidence/m3-identity-discrimination-r2/scope.json`): on the frozen runs the
hold-piece head emits `T` for all 9 training and all 12 development cases and its exact
count equals that class's frequency in the targets exactly (0/9 and 3/12), while the
training queue-token head is provably a single constant on three of four runs (`S`, `S`,
`I`, matching the 3/54, 3/54 and 12/54 totals). The coefficient lever's declared
identity positive, the budget's rise and the tied read-out's hold movement are therefore
class-frequency changes, not learning; every recorded verdict and receipt stands, and the
fit gate stays FALSE (0/9 and 0/18 boards, 9/18 and 0/36 roles, 0/9 and 0/18 components).
Discrimination is now reported beside correctness wherever identity is measured
(V-048), and the open identity question is the read-out's conditioning rather than its
budget, coefficient, resolution or tying.

Latest: the input representation (model capture side 512 -> 640) is a declared negative
([M3_IDENTITY_REPRESENTATION.md](M3_IDENTITY_REPRESENTATION.md), predeclared at
`b229972`, `docs/evidence/m3-identity-factor-representation-r1-errors-r1/scope.json`):
the primary does not rise (exact queue piece tokens 3/54 training and 9/90 development,
equal to `m3-identity-factor-r1`) while every structure term holds, and the independent
replay measures the same 54/90 tokens and 9/12 hold cases, so the finer capture did not
change the supervised population. The fit gate is unchanged and FALSE (0/9 and 0/18
boards, 9/18 and 0/36 roles, 0/9 and 0/18 components, 3.82 and 2.28 cells). With the
coefficient, read-out, budget and representation all measured, no executable identity
lever remains: what is left is the owner-decision real-image chain and the FALSE fit
gate.

Latest: the 512-update identity budget is a miss on its own criterion
([M3_IDENTITY_BUDGET.md](M3_IDENTITY_BUDGET.md),
`docs/evidence/m3-identity-factor-budget-r1-errors-r1/scope.json`): exact queue piece
tokens rise 3/54 -> 12/54 training and
9/90 -> 12/90 development at the same
capacity, factor, data and order, but training queue EOS falls
8/9 -> 5/9 and development hold identity
3/12 -> 0/12, training roles fall to
9/18, and the fit gate stays FALSE. Identity is budget-sensitive but
the budget multiplies a structural trade-off, so the next ranked lever is the input
representation (pixels per icon), not another budget extension. V-044 fixes the
predeclaration rule that had no branch for this state.

Latest: identity supervision is class-complete but label-scarce
([M3_IDENTITY_EXPOSURE.md](M3_IDENTITY_EXPOSURE.md),
`docs/evidence/m3-identity-exposure-r4/scope.json`): all seven piece classes are
supervised in both splits (54 training and 90 development queue tokens), yet the only
identities ever exact are queue `S` (3/3 and 9/18) and hold `T` (0/3 and 3/3), with six
queue classes and three hold classes at zero and the development successes spread over
slots 1, 3, 5 and 7. The scored probes are not the training stream, though: the factor-4 run's 128
updates drew 128 distinct images from 43 ancestries, so what remains is optimization
budget versus the input representation. The budget lever is predeclared in
[M3_IDENTITY_BUDGET.md](M3_IDENTITY_BUDGET.md) and is running next.

Latest: tying the identity read-out into the cell-type classifier is a measured
negative ([M3_IDENTITY_READOUT.md](M3_IDENTITY_READOUT.md), predeclared at `e83d70e`,
shared mode admitted at`runs/m3-stream-identity-shared-profile-r2-job.json`):matched128
updates atcapacity64 move exact queue pieces0/54->9/54 training and0/90->6/90
development and development hold0/12->3/12, but the predeclared structure-held
condition fails (training presence63/63->54/63, development EOS15/15->13/15) and
training roles fall12/18->0/18, so the fit gate stays FALSE(0/9 and0/18 boards,0/9 and0/18
components,3.32 and2.55 cells). Independent27-forward replay(weights`55a35243...`
unchanged) and curator: `docs/evidence/m3-stream-identity-shared-r1-errors-r1/`. Both identity-side levers(coefficient and read-out)are now measured and closed: the
coefficient moved localization at a measured structural cost, and the read-out tying is
a negative. The fit gate stays FALSE, and the open items narrow to the supervision
signal, the input representation(pixels per icon)and the owner decision on admitting
real screenshots. The whole contract suite is green on this tree(407 tests,
`docs/evidence/m3-identity-readout-fullsuite-r11/`)after the supervisor's commit stop line
was corrected from physical RAM to the commit limit(V-040).

Latest: a declared4x identity-loss scale is a measured positive on the M3 identity
localization ([M3_COMPONENT_IDENTITY.md](M3_COMPONENT_IDENTITY.md), predeclared at
`65ee2f1`): matched128 updates atcapacity64 raise exact queue piece tokens0/54->3/54
training and0/90->9/90 development, hold0/12->3/12, slot status still exact, while the
fit gate stays FALSE(0/9 boards,12/18 roles,0/9 components,3.83 cells) and development
EOS falls15/15->3/15. Independent27-forward replay and curator:
`docs/evidence/m3-identity-factor-r1-errors-r3/`.

Latest: the first joint M3 fit (capacity64,128 updates) is a measured negative
([M3_JOINT_FIT.md](M3_JOINT_FIT.md), predeclared at `90218b3`): training probes
0/9 exact boards,12/18 roles (ghost9/9, active3/9) and0/9 readable components with
queue tokens0/54 andhold0/9, while EOS9/9, slot status54/54 andpresence63/63 are
exact; geometry19.82->4.04 cells and loss29.97->7.74 show the optimizer trains,
but development stays0/18 boards and0/36 roles. Independent27-forward replay
(weights`b2e44c9c...` unchanged) and its curator are in
`docs/evidence/m3-stream-joint-fit-r1-errors-r2/scope.json`. Items4-5 need
reviewed, permissioned real labels and stay unreachable from repo data.

The staged frozen-backbone type-head refinement is also a measured negative
([STAGED_HEAD_REFINEMENT.md](STAGED_HEAD_REFINEMENT.md), predeclared at `e601182`):
head-only training on the ordered artifact raises exact training roles 20/64 to
36/64 with boards32/32, cells128/128 and geometry.21059 bit-identical, still short
of the64/64 gate and26 short of the62/64 copied-head reference, and an independent
50-forward replay (unchanged weights`e02a02e7...`) plus curator reproduce every
saved metric; development on the canonical18 images is the same4/18 boards and0/36
roles for both stages, with stage2 loss higher (11.8129 vs11.0885, replay
`m2-stream-headrate-control-s3-dev6-errors-r1`); do not re-tune this stage. Conditional copied-head pixel
replay and independent audits are complete for both controlled models. Ordered training roles reach 62/64; interleaved
64/64, but both joint-fit criteria fail and development types remain 0/18.
See [CONDITIONAL_PIXEL_REPLAY.md](CONDITIONAL_PIXEL_REPLAY.md). Next predeclare
a controlled joint-optimization lever and its resource/recovery admission.
[TYPE_HEAD_JOINT_OPTIMIZATION.md](TYPE_HEAD_JOINT_OPTIMIZATION.md) now defines
the fixed candidate and records the completed matched comparison: both fail the
joint fit; increasing the type-head rate reduces exact training roles 20/64 to
8/64, despite improving geometry, and development roles remain 0/18 for both.
The checkpoint diagnostic in [TYPE_FORGETTING_DIAGNOSTIC.md](TYPE_FORGETTING_DIAGNOSTIC.md)
finds recurrent type-correctness turnover on the fixed fixtures and local
class-gradient conflicts, with significant limits on causal interpretation.
The matched design and focused grouping/gradient/recovery instrument are in
[BALANCED_ACCUMULATION.md](BALANCED_ACCUMULATION.md). Both1,024-image arms now
have independent ordinary replays and curators
(`docs/evidence/accumulation-arm-{balanced,online}-errors-r1/scope.json`):
balanced-eight reaches16/32 exact training boards,34/64 roles and.7507 geometry,
while online interleaved reaches28/32,37/64 and.3862; both fail the predeclared
fit and both stay at4/18 development boards(all four empty)with0/36 roles and
rising development loss. Batching eight images per update is a measured negative
result,not a mechanism claim:the ordered-eight arm was never run and grouping is
still confounded with update count and clipping frequency. Explicit image and
optimizer cursors are now integrated into the actual trainer behind an explicit
accumulation policy, with image-cadence evaluation/snapshots and separate counts in
every ledger and summary. Measured two-update native profiles now exist for both
predeclared group sizes under the unchanged 640 MiB ceiling and supervisor
(size1: 519,929,856 B peak,33.3 s; size8: 561,786,880 B peak,29.35 s, both with
exact boundary reload), and each arm is admitted only through those measured
numbers. Both 1,024-image arms now complete across two bounded segments each with exact
journal-prefix resume, byte-identical Adam/RNG continuation and measured peaks
under640MiB (`docs/evidence/accumulation-arm-r1/`); automated controller/curator
routing remains open; that independent pixel replay and both curators are now complete for both arms (see the accumulation paragraph above). The pure journal and ownership boundary pass their own scoped
tests, and the whole-group arm also passes scoped driver checks including a forced
in-group interruption replayed to identical bytes (toy network; V-030 records that
bounded children need an absolute interpreter path). Retain the original joint-fit
and generalization gates.

1. The finite768-step diverse memory gate is GREEN at unchanged640MiB: peak
   519,782,400B,2.6MiB second-half growth, exact565 prior record/data prefix,
   final recovery/inference parity and full recipe replay. V-013 CLOSED.
   Development is still0/18 exact boards,0/36 known layers and1.85-cell geometry
   error; M2 learning remains OPEN. See the named analysis below.
2. The actual larger model (7,694,010 parameters) passed its first two-step
   characterization at1GiB:910,176,256B peak, full warm/recovery/output parity,
   docs/evidence/m2-stream-accuracy-probe/scope.json. Its128-update diagnostic
   completed as93+resume35: dev4/18 exact boards are all zero-locked, nonempty
   boards0/14,layers0/36,geometry2.4785cells. The finite segment controller now
   passes an actual pressure-saved step1+resume1/curation gate. The fixed4096
   baseline is COMPLETE4096 through bank3840+256 verified updates: dev0/18 boards,
   0/36 layers,geometry2.01544cells. All4096 recipes and27 probes replayed;
   docs/evidence/m2-stream-accuracy4096-s10/scope.json preserves complete4096/4096
   and native controller/child/curator receipts. The prior host-pressure stop
   logged1172/banked1152; all20 recomputed updates and1172-row prefixes match
   exactly,excluding time. New inference weights06751638... retain original
   catalogue121fd05a; adequate learning remains open. Keep this study complete. Its
   final ordinary27-forward replay and independent curator now pass,with813
   invented development cells,32 misses,0/18boards and0/36roles. The entire9/18
   training-probe layer score is ghost absence,not visible-ghost reading.
   ACCURACY4096_BASELINE.md records source/artifact/native scope and limits.
   Use those errors to track independent active/
   ghost errors and repair the learning failure before choosing architecture or
   initialization. Preserve explicit partial
   status and source/artifact identity. M2_LEARNING_ACCEPTANCE.md keeps the
   earlier fixed-fit criterion; a lower loss is not complete-state success.
   Exact masked-target audit of the investigated1664 prefix confirms all seven
   observed piece classes were supervised,but training ghosts only occur at
   row10 and fixed probes omit classes/statuses. TARGET_EXPOSURE.md records
   counts and limits. A separate optional dense layer-evidence experiment is
   now a completed matched768-update pilot: both studies pass resource/recovery,
   but broader192-image evaluation gives0/148 exact nonempty boards for both,
   only4/165 observed active poses and0/144 ghosts for the candidate, with more
   locked false positives. It remains default off. LAYER_LEARNING_PILOT.md records
   raw source/data/artifact-bound results and coverage limits. Retain the original
   baseline; investigate class collapse with an all-class positive-control fit
   before selecting another learning mechanism. Its32 balanced public-observer
   fixtures now pass5 source-bound checks and exact file/pixel/recipe replay;
   ALL_CLASS_FIT.md predeclares the fixed1,024-update fit criterion. The frozen
   driver, fresh warm admission and exact2+6+3+3 recovery are verified on
   catalogue6147a558. The fixed1,024 updates and independent final replay are
   complete:32/32 boards,12/56 observed role labels,8/8 absences,128/128 semantic
   cells and.211-cell geometry. All observed positions/rotations fit, but types
   collapse to Z/J/L and the fit gate FAILS. The next selected lever is the
   order-only comparison in INTERLEAVED_FIT.md. Its sampler and policy are
   integrated, with88 scoped checks plus the8-case profile-ancestry regression.
   Fresh640MiB warm admission and independent replay pass; the same-input
   comparison has identical actual weights,losses,gradients,recipes and probes.
   New-policy recovery and the full matched1,024-update comparison are now
   complete:visible labels12/56->30/56,but boards32/32->28/32 and geometry
   .211->.386. Both fit gates fail;INTERLEAVED_FIT.md records controls,raw evidence
   and limits. Do not extend either finished study or adopt this ordering as a
   release fix. The original larger4,096 baseline is now complete with its
   original frozen source/optimizer and independently replayed final errors.
   Read-only actual type-head feature observation and independent collectors
   now pass on all three completed models,with ordinary outputs/weights unchanged.
   The ancestry-excluded centroid readouts provide no obvious replacement;they
   do not prove absence of class information. LAYER_FEATURE_DIAGNOSTIC.md records
   actual class coverage/counts/receipts and limits. The subsequent fixed1,024-update
   conditioned full-batch linear-head fits now complete with separate native warm/
   reload admission and independent audits. Ordered training types20/64->62/64,
   interleaved37/64->64/64,development0/18both;TYPE_HEAD_FIT.md records criteria,
   worsened development losses and scope. Original image models/source/optimizers
   stay frozen. Conditional pixel replay now confirms the copied-head training
   counts while leaving other outputs unchanged; both joint-fit gates fail.
   No generalization gain follows from this intervention.
   More than40GB free is required before each long training window.
3. Complete M3 linked queue/hold heads, masked losses and review-only output.
   Optional shared role/index heads and masked losses now pass15 scoped checks,
   with14 M2 numerical,7 runtime and8 observation-target compatibility checks.
   Capacity0 preserves M2; capacity1..64 uses explicit M3 checkpoint identity.
   Joint streaming integration passes68 scoped unit/contract checks onf542096.
   Native pilot recovery6==3+3 is exact, and both pilot and larger capacity64
   configurations pass full512-square warm/reload resource profiles. The finite
   controller now also passes actual continuous3==partial1+resume2 routing,
   curation and byte-identical model/Adam/RNG recovery on the admitted pilot;
   docs/evidence/m3-controller-recovery-r1/scope.json. The larger
   has7,997,016 parameters and942,903,296B peak under1GiB. Joint learning now has its first measured datum: a128-update joint fit atcapacity64
   on the run's nine training probes is complete and fails its predeclared criterion
   (0/9 boards,12/18 roles withghost9/9 andactive3/9,0/9 readable components with
   queue tokens0/54 andhold0/9 against exact EOS9/9, slot status54/54, presence63/63)
   while development stays0/18 boards and0/36 roles. Geometry19.82->4.04 and loss
   29.97->7.74 show the optimizer trains; component identity does not.
   docs/M3_JOINT_FIT.md; curator docs/evidence/m3-stream-joint-fit-r1-errors-r2/.
   A declared4x identity-loss scale (matched, predeclared `65ee2f1`) moves exact queue
   piece tokens0/54->3/54 and development0/90->9/90 with hold0/12->3/12 while the fit
   gate stays FALSE and development EOS falls15/15->3/15: identity is gradient-starved,
   not unreachable.Identity representation at that scale (pixels per icon, or the component targets
themselves) and varied-shape long memory remain pending; no quality admission follows.
   Synthetic final-model-size
   captures now reobserve labels at that exact transform (9 checks). Arbitrary
   post-labeled resize/invalid alpha still refuses; the API keeps queue/
   hold unknown. Current M2 training must resume its original archive.
   Preserve unknown middle positions and unobserved queue ends; never infer
   hidden length or delete locked cells using a guessed active/ghost footprint.
4. Expand synthetic appearances, crops/occlusion, queue arrangements and board
   association. Admit reviewed real labels and compare scratch/pretrained/direct
   decoders on independent session/ancestry/family splits. The166 user images
   are drafts, not a labeled benchmark. Broad render and real coverage are open.
   This chain is not executable from repo data: it needs reviewed, permissioned
   labels, so it is skipped on purpose rather than treated as a blocker.
5. Release only after the full gate: >=95% exact complete state on the readable
   real population, abstentions counted as failures, confidence intervals,
   calibration/risk-coverage, exported-artifact parity, CPU latency/memory gates
   and thin solver/bot adapters. Missing evidence leaves this gate open.

Verified foundations and their limits:

| Area | Named evidence | What remains open |
|---|---|---|
| State1.1, bottom-first coordinates, explicit unknowns, strict solver conversion and renderer | docs/evidence/v006/; canonical-frame repair69 fixtures | Broad real visibility and full renderer/client coverage |
| Scratch encoder/lattice/axial cell reader, independent locked/type/active/ghost heads, safe checkpoints | docs/evidence/m2-numerics/numerical.json;85 checks on named original sources | Accurate converged whole-state model; larger configuration |
| Fixed-data learning16/128/saved511 | docs/evidence/m2-learn16/, m2-learn128/, m2-learn512/ | Saved511/512 fits6 boards but layers10/12 and geometry.4106 fail joint fit; dev0/6 |
| Local single-cell reading | docs/evidence/m2-local-reading/scope.json |36/36 probability direction, only20/36 targets and18/36 changed boards exact; two unrelated changes |
| Streaming diverse memory/learning | docs/evidence/m2-stream-learn768-r3/analysis.json and scope.json | Resource gate green; dev0/18 boards,0/36 layers; no real/whole-state quality |
| Larger model learning128 | docs/evidence/m2-stream-accuracy128-r2/analysis.json and scope.json | Exact93+35 continuation; all four exact dev boards have no locked blocks; nonempty0/14 |
| Oracle-grid diagnostic | docs/evidence/m2-stream-pilot-geometry/scope.json | Deliberate supported-label intervention still0 exact boards; not automatic accuracy |
| CPU resource/recovery contracts | docs/evidence/v015-contracts/scope.json;29 scoped checks on f583792 | Not a whole-suite or model-accuracy claim |
| Exact stream recovery and archived driver execution | docs/evidence/m2-stream-recovery-gate-r2/scope.json | Native2+6+3+3 matches losses/gradients/data/evaluation/model/Adam/RNG; long memory remains separate |
| Actual pressure interruption recovery | docs/evidence/segment-pressure-recovery/scope.json; m2-stream-accuracy-segment-probe-r2/controller-gate.json | Saved larger step1 restored then resumed one update with explicit partial status and evidence replay; global target3 unfinished |
| Review-only one-image API | docs/evidence/m2-runtime/scope.json | Always needs_review/unsupported; queue/hold unresolved |
| Queue/hold joint heads, masked losses and final synthetic captures | docs/M3_COMPONENT_TARGETS.md; joint-stream-contracts/ and joint-recovery-contracts/ | Joint learning, general resized real supervision and runtime decoding open |
| Native M3 resource/recovery | docs/evidence/m3-stream-pilot-profile-r1/, m3-stream-large-profile-r1/, m3-stream-recovery-gate-r1/ | Larger/pilot warm profiles and finite pilot recovery only; long varied-shape memory and quality open |
| Corpus audit/review/split/seal tooling | docs/evidence/corpus-integrity-final/scope.json;26 adversarial checks | Correct reviewed labels and real-image model results |
| Unlabeled real stress set | docs/evidence/real-drafts/scope.json;166 images/seven private sheets | Zero training/scoring eligibility; independent review and permission needed |

Milestone scope remains intact: M0 state/benchmark foundation and reviewed pilot;
M1 label integrity and broad rendering; M2 scratch lattice model and adequate
learning; M3 complete joint state; M4 controlled real comparisons; M5 release.
Historical counts/artifacts retain their original source scope. Current safety
and recovery gates are separately named; none imply a fresh unified suite.

Prior tetrguide failures retained as design constraints:

| Issue | Contract here |
|---|---|
| B-260 active guessing deleted stack; glow confused with blocks | Direct locked labels/heads and separate active/ghost outputs; never erase guessed footprints |
| B-227 reversed rows | One bottom-first frame with asymmetric pixel/solver tests |
| B-259 observed and assumed state confused | Explicit unknown/status and refusal to convert unsupported solver input |
| B-102 unrelated files staged | Isolated repository, explicit task-owned paths and index review |

The neighboring engine roadmap, binaries, package and run state are not owned by
this project. Never modify them to make a vision gate pass. See BUG_LOG.md,
TRAINING_RECOVERY.md, RESOURCE_GUARD.md and CORPUS.md for the current contracts.

## M28 - the per-cell pixel-statistics read-out is measured on the deciding population and refused

`occupancy_pixel_query` is a zero-initialized, strictly additive read-out of 30 deterministic
per-cell colour/contrast statistics computed at the model's own predicted lattice. On the fixed
M27 matched population of 2,986 cells ([M28_PIXEL_QUERY.md](M28_PIXEL_QUERY.md),
`docs/evidence/m28-pixel-query-deciding-r3/scope.json`) it moves its own read from `0.219626` to
`0.425703` best swept F1 and `0.574178` to `0.898191` AUC over a matched control - `0.464970` of
the `0.5440 -> 0.9872` span - but stays `0.118271` below the `0.543974` reference arm and `0.561476`
below the `0.987179` pixel read. The per-cell (`0.282638` against `0.70`) and ranking (`0.898191`
against `0.90`) gates fail while the attribution gate passes, so the lever is **not cleared**.
Placement is measured at zero (`-0.002900` of best swept F1 when the reference registration is
granted) and the pixel read holds `0.987179` on the same placement, so the residual is
representation: the read-out must change what it computes, not where it samples.
The baselines carry their own populations (V-151, `docs/evidence/m28-reference-provenance-r1/`):
`0.5440`, `0.7626` and `0.9872` are matched-2,986 figures of the reference arm and the pixel read,
while `0.3791` is the reference arm's own-lattice read on the full 5,600-cell set and was never the
control's value - the control's own full-population read is `0.284833`. The earlier per-arm
3,316-cell verdict is retained as a labelled secondary read (V-150).

## M29 - the pixel-query collapse is capacity, and a hidden layer fixes most of it

A predeclared four-mechanism instrument ([M29_PIXEL_QUERY_FORM.md](M29_PIXEL_QUERY_FORM.md),
`docs/evidence/m29-pixel-form-r2/scope.json`) separates the causes on one fixed 2,986-cell
population. Amplitude is excluded from the committed checkpoint (`0.703541` of the head's
gain), centring the inputs is excluded (`-0.012545`), neighbourhood
context is excluded (`+0.036428`), and capacity is confirmed:
the same 30 statistics through a 64-wide hidden layer reach `0.610101` best swept
F1 and `0.946604` AUC, `+0.184398` over the
form-1 arm and above the `0.543974` reference arm. Span closure is `+0.149202`
(placement `-0.202732`, representation `+1.053530`).
The gates still fail on the operating point (F1 at `0.0` `0.291464`
against `0.70`; best `0.610101` against `0.70`), so the lever is not cleared, but
the read-out's remaining deficit is no longer representation of the features: it is the threshold the
read lands on.

## M30 - the level is adjustable, the per-cell boundary is not

A predeclared operating-point instrument ([M30_OPERATING_POINT.md](M30_OPERATING_POINT.md),
`docs/evidence/m30-operating-point-r1/scope.json`) declares three single-change mechanisms that can only
move the M29 winner's level, each identity at step zero. A learned scale and bias on the projection is
excluded (`-0.005750` F1 at `0.0`; the learned scale settled at
`0.995428`), a data-derived class budget is excluded and negative
(`-0.015693`), and a label-free gated board standardisation is
confirmed at `+0.081417` - F1 at `0.0` `0.372881`, AUC
`0.954529`, best `0.689775` against the unchanged control `0.219626`
and reference `0.543974`. The gates still fail on two of three (per-cell
`0.372881` against `0.70`; best `0.689775`, `0.010225`
short), so the lever stays uncleared. The measured limit is now the gate's single global `0.0` boundary
applied to a read whose correct level is per-board; closing it would change the metric, which this pass
does not do.

## M31 - the per-cell deficit is the read's score shape, not the boundary

A predeclared three-boundary read ([M31_DECISION_RULE.md](M31_DECISION_RULE.md),
`docs/evidence/m31-decision-rule-r2/scope.json`) scores one frozen arm on the same 2,986 cells and both
fixed label vectors: the unchanged global `0.0` gate `0.372881` F1 at `0.0` /
`0.689775` best, the declared **label-free** per-board `otsu_board` rule
`0.278699` / `0.581359`, and a label-fitted per-board
boundary as an oracle-only diagnostic `0.763916` excluded from every verdict. The
label-free rule loses to the global boundary, so the branch is representation, and the oracle says the
logits still carry enough to clear `0.70` - the failure is where a label-free rule can look, not what the
read knows. The branch's declared structural step, the hidden layer given one cell of neighbourhood context,
measured `0.518657` best swept F1 against `0.689775`
(`-0.171118`, `docs/evidence/m31-followup-r1/scope.json`), so it is refused on
measurement. The standing gate is untouched and still fails on both per-cell and ranking.

## M32 - mass matching is refused, and the split now names placement

A predeclared supervision-side term ([M32_MASS_MATCH.md](M32_MASS_MATCH.md),
`docs/evidence/m32-mass-match-r1/scope.json`) charged each board's predicted positive mass against its
own occupancy, with a fixed unit weight, no new parameter and no change to the inference boundary. It
is **negative**: F1 at `0.0` `0.277778` (-0.095104), best swept
`0.602198` (-0.087577), AUC `0.942324`
(-0.012205), attribution +0.382572 over the unchanged control.
Gate block: per-cell false, ranking false, attribution true. Span split: closure
`s0.058224` (`0.131370`), placement `-0.103647`
(`-0.233858`), representation `+0.488628`
(`1.102488`). Placement is now the only negative term, so the recorded next
structural step is a fixed-lattice placement for the pixel-statistics read rather than another
supervision term on the score map.

## M33 - a constant lattice does not buy placement

The M32 receipt's recorded next step was measured ([M33_FIXED_LATTICE.md](M33_FIXED_LATTICE.md),
`docs/evidence/m33-fixed-lattice-r1/scope.json`): form 6 samples the same pixel statistics at a fixed
lattice over the capture (no parameter, nothing fitted, boundary still `0.0`). It is **negative** -
F1 at `0.0` `0.255630` (-0.117251), AUC `0.945085`
(-0.009444), best swept `0.632327` (-0.057448),
attribution +0.412701 - and the split leaves placement the only negative term at
`-0.118649` (`-0.267707` of span), representation
`+0.473501` (`1.068357`). So the placement question is
not answered by removing the decoder from the sample: the next declared step is a lattice derived from
the capture's valid support (mask centroid and extents), which reads the raster only and fits nothing.

## M34 - placement is reducible, and the support lattice proves it

The M33 receipt's recorded step was measured ([M34_SUPPORT_LATTICE.md](M34_SUPPORT_LATTICE.md),
`docs/evidence/m34-support-lattice-r1/scope.json`): form 7 maps the board onto the bounding box of the
capture's valid pixels. The gate is still **not cleared** - F1 at `0.0` `0.274869`
(-0.098012), AUC `0.883260` (-0.071268), best swept
`0.385925` (-0.303850), attribution +0.166299 - but the
placement question is settled: the placement term drops from `-0.103647` to `-0.004778`
against a predeclared bar of `-0.083647`
(`placement_reducible: true`). Placement was padding-driven all along. The split now reads closure
`-0.158049` (`-0.356605`), placement
`-0.004778` (`-0.010780`), representation
`+0.606031` (`1.367384`), so the short stage is **closure**
and the recorded next step is a contrast read at the support lattice.

## M35-M36 - closure responds to features, not to structure

M35 measured the contrast read ([M35_CONTRAST_READ.md](M35_CONTRAST_READ.md),
`docs/evidence/m35-contrast-lattice-r1/scope.json`): form 8 puts the `30` statistics beside their
sample-mean deviations (`60` inputs) at the support lattice. F1 at `0.0` `0.277778`,
AUC `0.945596`, best swept `0.570048` - the bar (`0.372881` / `0.689775`) is
missed, but closure moves off the floor for the first time (`+0.026074` against M34's
`-0.158049`) and the diagnosis shows the read separating while calling `1,088` empty cells confidently.

M36 took the structural step that diagnosis implies ([M36_MONOTONE_CONTRAST.md](M36_MONOTONE_CONTRAST.md),
`docs/evidence/m36-monotone-contrast-r1/scope.json`): form 10 adds the declaration that a locked stack is
column-contiguous, so each column crosses the unchanged boundary once. F1 at `0.0`
`0.313667`, AUC `0.886350`, best swept `0.420935` - **refused**.
The envelope clears the lower bands (`0` false positives below row `10`) and lifts precision to
`0.186667`, but compressing the margins costs more ranking than it buys. Both branches are
now measured: closure wants better features, and the recorded next step is the coarse lattice-free channel
combined with the pixel statistics.

## M37-M38 - the coarse channel and a doubled width are both refused

M37 fused the coarse `stencil_contrast` summary into the pixel read
([M37_STENCIL_FUSION.md](M37_STENCIL_FUSION.md), `docs/evidence/m37-stencil-fusion-r2/scope.json`):
F1 at `0.0` `0.270096`, AUC `0.919512`, best swept
`0.589286` - bar missed, and the channel does not separate the textured-empty cells
(false positives `1131` against M35's `1088`, median margin higher, upper bands unchanged).

M38 doubled the same read's hidden width to `128` ([M38_CAPACITY.md](M38_CAPACITY.md),
`docs/evidence/m38-capacity-r1/scope.json`): F1 at `0.0` `0.273082`, AUC
`0.895041`, best swept `0.463551` - refused, and worse than the `64`-wide arm at the
same inputs, so at a fixed 64-step budget the width is not the binding constraint. The ladder now reads:
winner `0.372881` / `0.954529` / `0.689775`; every support-lattice arm below it on all three. The recorded
next step is a read that samples each cell at **both** placements (predicted lattice and support lattice,
`60` inputs) so the placement-consistency difference is available to the read.

## M39 - both placements in one read, and the campaign's central question answered

[M39_DUAL_PLACEMENT.md](M39_DUAL_PLACEMENT.md) predeclared and measured form 13: the `30` statistics at the
model's own predicted lattice beside the same `30` at the support lattice, `60` inputs, one head. Measured
`docs/evidence/m39-dual-placement-r1/scope.json`: F1 at `0.0` `0.246228`, AUC
`0.836907`, best swept `0.351779` - the weakest arm yet, gate per-cell/ranking false,
attribution true, bar missed. The placement-consistency signal is **present and swamped**: false positives
rise to `1239` at median margin `2.316557`, rows 15-19 alone
`283` against M35's `144`.
[ladder](M34_SUPPORT_LATTICE.md) - see the maintained regression set below. The evidence supported one
structural reading at this point: the winner's geometry is *endogenous* (it samples the lattice it predicts,
so its features and its placement are trained together), and every support-lattice arm replaces that with an
exogenous prior. The next declared step keeps the geometry endogenous and uses the support only as an
alignment applied to the predicted lattice. **That reading was tested in M40 and narrowed there**: it remains
the best-supported account of the strength gradient, but it is not an ordering principle - see the corrected
closure section below.

## M40 - the predicted lattice, corrected by the support, and the question closed

[M40_COMPOSED_ALIGNMENT.md](M40_COMPOSED_ALIGNMENT.md) predeclared and measured form 14: the `30`
statistics sampled at a declared per-board blend of the model's predicted lattice and the support lattice
(`PIXEL_COMPOSED_WEIGHT = 0.5`, fixed, no parameter), so the geometry stays a function of what the network
predicts while the raster's support enters as a correction. Measured
`docs/evidence/m40-composed-alignment-r6/scope.json`: F1 at `0.0` `0.260599`, AUC `0.897918`, best swept
`0.439535` (at `1.65`) - refused, gate per-cell/ranking false, attribution true, bar missed
(`-0.112283` / `-0.056611` / `-0.250240`). Split closure `-0.104439` / placement `-0.009742` /
representation `+0.557387`, sum asserted; false positives `1181` at median margin `1.233860`, `910` of them in
rows 0-9 and `136` in rows 15-19, `textured_empty_separated: false`.

The predeclaration records the defect caught before the run: the *literal* composition (map the predicted
board box onto the support box) is provably the support lattice itself, so it would have re-measured M34
under a new name; the declared form is a convex combination instead. A second defect (V-166) was caught after
the first run: the composed placement was assigned after the statistics had been sampled, so the arm trained as
the winner - visible because its checkpoint hash was byte-identical to the anchor's. Repaired before the
scored run, with the inert run and its scoring kept and labelled.

## M41 - provenance isolated, and the confound settled

[M41_SUPPORT_CONTRAST.md](M41_SUPPORT_CONTRAST.md) predeclared and measured the arm the M40 pass lacked: form
`15` is M35's `60` contrast inputs sampled at the **support lattice**, completing the geometry-by-features
square whose other three cells are the winner (predicted, `30`), M34 (support, `30`) and M35 (predicted, `60`).
Recorded configuration pins the isolation: `8,025,482` parameters, identical to M35's, with
`occupancy_pixel_query` the only config field that differs (`15` against `8`), so features, head width,
decision rule, dimension, blocks, standardise flag, curriculum, initial weights and step count are all held
fixed.

Measured `docs/evidence/m41-support-contrast-r1/scope.json`: F1 at `0.0` `0.272073`, AUC `0.819074`, best
swept `0.340000` (at `1.20`) - **not cleared** on the predeclared bar, gate per-cell/ranking false,
attribution true, deltas `-0.100808` / `-0.135455` / `-0.349775`; split closure `-0.203974` / placement
`-0.016333` / representation `+0.663512`, sum asserted.

The predeclared branch that fired is **REPRODUCES** (its thresholds were AUC `<= 0.910000` and best swept
`<= 0.480000`, both met): at matched features the geometry swap costs `-0.126522` AUC and `-0.230048` best
swept, against `-0.071268` / `-0.303850` for the same swap at `30` inputs. **The overlap in the section below
is attributable to its recorded confound, not to provenance failing to order anything.**

## M44 - the pixel edge ring: boundary brightness is not the missing per-cell evidence

[M44_PIXEL_EDGE.md](M44_PIXEL_EDGE.md) predeclared the first arm since M31 to add *information* rather than a
level: the raster sampled on each cell's own four edge midpoints, half a cell out and outside the declared `+/-0.3`
in-cell grid, through a zero-initialised projection of exactly `2,496` parameters. `docs/evidence/m44-pixel-edge-r1/
scope.json`: F1 at `0.0` `0.304665`, AUC `0.934008`, best swept `0.550622` (at `3.15`) - **not cleared**, gates
per-cell/ranking false, attribution true, deltas `-0.068217` / `-0.020521` / `-0.139153`; branch **harmful**.

The error mix is the finding: true positives `209` and false negatives `5` are **identical to the control's**, and
the whole delta is `251` extra false positives (`698` -> `949`) at a higher median margin, so precision at the fixed
crossing fell `0.230430` -> `0.180484` (`-0.049946`) and rows 10-19 went **`140` -> `240`**, gaining cells in rows
13, 14, 18 and 19 the control had clean. The mechanism is in the statistic: above a stack the cell's top edge
coincides with the board's own background transition, so boundary brightness fires there too and the `4x`-weighted
objective raises the empty region instead of lowering it. Control `m44-control-winner-config` reproduced
`m30-level-calibrate-r5`'s checkpoint bit-exactly (`eddec1fd…`); the arm carries exactly the declared `2,496`
parameters with checkpoint `d4ae2727…`.

What the negative leaves standing is precise: the missing per-cell evidence must be *contrast against the cell's own
interior* - which cancels the background transition - rather than the boundary value, and any such arm is a
different feature to be declared on its own. No offset, width or sample-count variant of this ring is open.

## M43 - the per-column step crossing, and the level axis closed at every granularity

[M43_COLUMN_STEP_CROSSING.md](M43_COLUMN_STEP_CROSSING.md) predeclared the one decision rule left that acts on the
mass M42's diagnosis found - cells above each column's own step still reading moderately positive - and chose the
step statistic over mass balance for the reason M17 already measured (an upper blob contributes mass and attracts
a volume split). Each column is decided at its own sharpest row-to-row descent, one constant per column, no
parameter, no annotation, no prevalence. `docs/evidence/m43-column-step-r1/scope.json`: F1 at `0.0` `0.270008`,
AUC `0.752899`, best swept `0.272654` (at `-0.15`) - **not cleared**, gates per-cell/ranking false, attribution
true, deltas `-0.102873` / `-0.201629` / `-0.417120`; branch **harmful**.

The mass did not move the way the arm needed: false positives at `0.0` rose `698` -> `856`, and rows 10-19 went
**`140` -> `173`**, with the movement column-shaped - rows 3, 4, 9, 10-14, 16, 17 emptied exactly while rows 0, 1,
6, 7, 8 and 15 filled, because a column whose read has no sharp step still returns an `argmax`, often near its
top, leaving the whole column positive. Ranking fell `0.201629` because a per-column shift reorders cells inside a
board. Control `m43-control-winner-config` reproduced `m30-level-calibrate-r5`'s checkpoint bit-exactly
(`eddec1fd…`), the arm differs in exactly `occupancy_column_step_crossing` at the same `8,023,562` parameters, and
its checkpoint is `c5257239…`.

With this the declared level axis is closed at every granularity proposed: board mean (M14), board crossing (M16),
board standardisation (M31), raster support (M42), column step (M43) - and the two aimed at the diagnosed mass made
it worse. The ranking is already `0.954529`; what is left is per-cell precision at a fixed crossing.

## M42 - the support as a decision rule, and geometry's role on both sides

[M42_SUPPORT_CROSSING.md](M42_SUPPORT_CROSSING.md) predeclared and measured the one untested role for the
raster's support: instead of the winner's calibrated crossing, each board's own support-box offset sets that
board's crossing level, applied inside the head as one constant per board, while every sampled point stays
where the network put it. `docs/evidence/m42-support-crossing-r1/scope.json`: F1 at `0.0` `0.189455`, AUC
`0.838212`, best swept `0.374034` (at `1.65`) - **not cleared**, gate per-cell/ranking false, attribution
true, deltas `-0.183426` / `-0.116317` / `-0.315741`; split closure `-0.169940` / placement `-0.050224` /
representation `+0.663369`, sum asserted.

The predeclared branch that fired is **NOT CLEARS - harmful** (both falsifiers for the neutral branch exceeded:
`dAUC -0.116317`, `dF1@0 -0.183426`). The rule was not inert - it moved every board's crossing and reordered
the boards against each other - and false positives rose to `1812` at median margin `0.835996`, with `473` in
rows 10-14 and `475` in rows 15-19 against M35's `11` and `144`.

**The isolation is literal, not normalised.** The winner's own manifest predates `occupancy_read_level`, so
this pass trained a matched control under current code (`m42-control-winner-config`) whose checkpoint is
**bit-identical to the winner's** (`eddec1fd…`). Against that control the arm differs in exactly one recorded
config field (`occupancy_support_crossing`), has an identical parameter count (`8,023,562`), identical
sampled points and a distinct checkpoint (`fa4a7b60…`), so the contrast cannot be explained by features,
width, the monotone rule, dimension, blocks, standardise, curriculum, initial weights or step count.

## The campaign's remaining empirical question: answered in the negative, measured but not cleanly separated

**Can an exogenous correction to placement be learned at this depth by any composition with the predicted
lattice? Measured answer: not by any composition tried and not at this depth or budget - but the evidence
supports a graded cost of replacement, not a clean separation between the two kinds of arm.** Read across the
whole regression set, not from M40 alone:

| what the arm does to the sampled geometry | arms | AUC range | best swept range |
|---|---|---|---|
| keeps the predicted lattice sampled (adds features or width) | M35, M37, M38, M36 | `0.886350`-`0.945596` | `0.420935`-`0.589286` |
| moves it off the predicted lattice (replaces or blends in the support) | M34, M40, M39 | `0.836907`-`0.897918` | `0.351779`-`0.439535` |

**Those ranges overlap on both measures** - AUC in `0.886350`-`0.897918`, best swept in `0.420935`-`0.439535` -
and the earlier draft of this section wrongly said they did not. Inside the overlap the ordering is mixed:

| arm | sampled geometry | AUC | best swept |
|---|---|---|---|
| M40 `m40-composed-r2` | moves it (half support) | `0.897918` | `0.439535` |
| M38 `m38-capacity` | keeps it (width `128`) | `0.895041` | `0.463551` |
| M36 `m36-monotone` | keeps it | `0.886350` | `0.420935` |
| M34 `m34-support` | moves it (full support) | `0.883260` | `0.385925` |

M40, which moves the geometry, outranks M36, which keeps it, on both measures (`+0.011568` AUC, `+0.018600`
best swept) and outranks M38 on AUC (`+0.002877`) while sitting `0.024016` below it on best swept; M34, which
also moves it, is below M36 on both (`-0.003090` / `-0.035010`). What separates the two tables is therefore
not provenance alone - these arms differ in features, width and protocol too.

### What the overlap is, and is not, attributable to

Recorded facts only - the form id from each arm's saved config, the parameter count from its checkpoint
manifest (`runs/<arm>/checkpoint/manifest.json`), and what the declared form samples. Nothing re-scored, no
new arm:

| arm | recorded form | recorded parameters | sampled geometry (as the form declares it) |
|---|---|---|---|
| `m30-level-calibrate-r5` (winner) | `3` | `8,023,562` | predicted lattice |
| M40 `m40-composed-r2` | `14` | `8,023,562` | predicted lattice, half-corrected onto the support box |
| M34 `m34-support` | `7` | `8,023,562` | support lattice, fully exogenous |
| M36 `m36-monotone` | `10` | `8,025,482` | predicted lattice, one crossing per column |
| M38 `m38-capacity-r2` | `12` | `8,039,114` | predicted lattice, head width `128` |

Every other recorded setting is identical across these five: `dimension` `192`, `blocks` `6`,
`occupancy_read_standardise` `1`, every lever flag `0` (`occupancy_deform`, `occupancy_context`,
`occupancy_detail`, `lattice_support`, `occupancy_normalise`, `occupancy_centre`,
`occupancy_crossing_threshold`, `occupancy_stack_monotone`), the same curriculum, initial weights, `64`
steps and evaluation population.

The parameter offsets are the forms' own arithmetic - `+1,920` for M36's extra `30` inputs, `+15,552` for
M38's wider head - and that is the point: **the three arms with identical parameter counts (winner, M40,
M34) differ only in where the sampled geometry comes from**, which is why the strength axis is the one
attribution these receipts support. No overlapping pair is matched: M40 vs M36 differ in features (`30` vs
`60`) and in M36's monotone decision rule, M40 vs M38 in head width (`64` vs `128`) and features (`30` vs
`40`), M34 vs M36 likewise. The overlap is therefore **consistent with a confound and not attributable to
geometry provenance from recorded configuration** - the form id carries feature count, head width, sampling
geometry and, for form 10, a decision rule at once.

**Answered in M41** (the arm this paragraph asked for). Form `15` holds M35's `60`-input feature set, head
width `64`, `192`-dimension, `6` blocks, no decision rule and every other recorded setting fixed, and samples
the support lattice instead of the predicted one: `8,025,482` parameters, M35's exactly, with
`occupancy_pixel_query` the only config field that differs. Measured `0.272073` / `0.819074` / `0.340000`
against M35's `0.277778` / `0.945596` / `0.570048` - the provenance cost reproduces at matched config
(`-0.126522` AUC, `-0.230048` best swept, `1.78x` and `0.76x` the `30`-input axis cost). The overlap is
therefore a **cross-configuration effect**: M40's position above M36 is attributable to M36's `60`-input
features and its one-crossing rule, and its position above M38 on AUC to M38's wider head, not to
provenance being inert.

What *is* clean, and what carries the negative:

- **the winner is above all seven arms on both measures** (`0.954529` / `0.689775` against a best keeping arm
  at `0.945596` / `0.589286` and a best moving arm at `0.897918` / `0.439535`), and no arm clears the gate;
- **the strength axis is monotone in both metrics**, with the read held fixed and only the support's share of
  the sampled geometry changed: `0.0` → `0.954529` / `0.689775`, `0.5` → `0.897918` / `0.439535`, `1.0` →
  `0.883260` / `0.385925`. That is what a substitution shows; a learned correction would have had to beat the
  endpoint it corrects from, and the half-strength arm recovers only `+0.014658` of AUC and `+0.053610` of
  best swept against it while landing `0.250240` of best swept below the winner;
- **M39 rules out the information account**: that read sampled the support *and* carried the aligned
  statistics beside the predicted-lattice ones, and still scored lowest of the seven (`0.836907` /
  `0.351779`), so the missing piece was not knowledge of the alignment.

The mechanism statement - the winner uses its endogenous sampling path to correct its own placement - thus
remains the best-supported account of the strength gradient, and is **demoted from an ordering principle**:
the overlap contains a moving arm above two keeping arms, so a future arm could outrank some keeping arms
without being close to the winner.

**M41 explains that overlap without changing the sentence above.** With provenance held as the only varied
recorded fact, the geometry swap costs `-0.126522` AUC and `-0.230048` best swept (M35 `0.945596` / `0.570048`
-> M41 `0.819074` / `0.340000`), reproducing the strength axis at a second feature set. So the demotion is
about *cross-configuration* comparisons, which this regression set is full of, and not about geometry being
inert: the overlap's ordering comes from the feature count and decision rule that came with each arm. M41 is
also the worst-ranked arm measured while M35 is the best arm that keeps the predicted lattice, and they differ
in one recorded fact - so the contrast features amplify rather than repair a displaced geometry, and no
feature-side rescue of the support lattice is available to a future arm.

**M42 extends the same reading to the decision side, and the closure sentence still stands.** With the sampled
placement left untouched and the support allowed to set each board's crossing level, the arm lost `-0.116317`
AUC and `-0.183426` F1 at `0.0` (branch: harmful), so geometry measured at matched configuration now has a
negative role on **both** sides - sampling (M41) and decision (M42). Nothing here changes the closure's claim,
which is about this regression set as configured; what it adds is that no correctable placement prior was
found on either side, and that a future decision-side arm has to beat a measured `-0.116317` AUC rather than
merely avoid moving samples.

**After M43 the level axis is closed too.** Deciding each column at its own sharpest step lost `-0.102873` F1 at
`0.0`, `-0.201629` AUC and raised the late-row false-positive mass (`140` -> `173`), so the recorded next step is
not another level at any granularity: it is per-cell precision in the representation at the fixed crossing, which
is the only quantity left with measured headroom.

**After M44 that headroom is specified more sharply.** Boundary brightness added false positives only (true
positives unchanged at `209`, precision `-0.049946`, late-row mass `140` -> `240`), because a cell's top edge
coincides with the board's own background transition. The next arm on this axis therefore reads *interior contrast*
- the cell against its own inside - which cancels that transition, rather than any boundary value or another
level. Scope: one fixed population (28 boards, 2,986 cells), both fixed label
vectors, one seed, `64` steps, a `64`-wide one-block head, the fixed global `0.0` boundary, and three measured
points on the strength axis. Capacity is excluded as an explanation (M38 refused at width `128`; M40 adds no
parameter), and a learned per-board mixing weight was predeclared out of scope.

**After M46 the additive raster family is closed, and the next step is structural.** Three measured members
(M44 boundary value, M45 relative contrast, M46 spread-gated contrast) cost `-0.021` to `-0.197` AUC for `-0.068` to
`-0.232` F1 at `0.0`, while the committed ceiling read-out proves the same thirty statistics carry `0.974763` F1 at
`0.0` on the same cells. The open question is the route, not the evidence: an occupancy head that reads the thirty
statistics directly (no latent route into that head), a frozen raster projection that separates optimisation freedom
from information, and the ceiling read as the decision itself are the ranked candidates, each with its falsifier.
Scope is unchanged: one fixed population (28 boards, 2,986 cells), both fixed label vectors, one seed, `64` steps, a
`64`-wide one-block head, the fixed global `0.0` boundary.

**After M47 the question is the read-out itself, and answering it needs no training run.** Removing every
latent path from the occupancy decision cost `-0.239131` F1 at `0.0` and `-0.094155` AUC, with every empty cell
positive and the ordering itself worse, so neither the raster statistic nor its route is the binding constraint. The
committed ceiling read-out (`0.974763` F1 at `0.0` on these cells from the same thirty statistics) is what any
successor must explain: the decisive measurement is that read-out fitted on the statistics sampled at the model's
predicted lattice, which separates sampling geometry from read-out training at the cost of one script. Scope is
unchanged: one fixed population (28 boards, 2,986 cells), both fixed label vectors, one seed, `64` steps, a `64`-wide
one-block head, the fixed global `0.0` boundary.

**After M48 the binding term is registration, and it is measured.** The committed read-out scores `0.974763` F1
at `0.0` at the target lattice and `0.037152` at the predicted one, with a mean `2.479843`-cell displacement and
`0.764765` standard deviations of feature drift between them. Every feature-side arm of the last four passes was
measured in front of that displacement. The named next test is registration-first: the decision's statistics sampled
at a support- or anchor-derived lattice, falsified if AUC stays under `0.93`. Scope unchanged: one fixed population
(28 boards, 2,986 cells), both fixed label vectors, one seed, `64` steps where a step is needed.

**After M49 the placement axis needs a board box, and the frame no longer counts.** The declared label-free support
registration maps the board onto the capture's valid extent, which is `3.325530`x wider and `1.736611`x taller than the
board, so the same read-out that scores `0.974763` F1 at `0.0` at the target lattice scores `0.000000` there at a mean
`8.256949`-cell displacement. M34's `-0.004778` placement residual was paired, not geometric, and is no longer read as
accuracy. The named next test is board-box registration (label-free detection of the board's own box), falsified at a
mean displacement above `0.25` cells or a registered AUC under `0.90`; if it fails, placement moves inside the model.
Scope unchanged: one fixed population (28 boards, 2,986 cells), both fixed label vectors, one seed, `64` steps where a
step is needed.

**After M50 the geometry axis is closed on the raster side.** A label-free detection of each capture's own well walls
places the thirty statistics well enough that the committed read-out scores `0.979200` F1 at the fixed `0.0` crossing and
`0.999150` AUC, slightly ahead of the target lattice on the crossing and within `0.000045` on the ranking, against
`0.037152` / `0.442341` at the model's predicted lattice. The declared `0.25`-cell population displacement gate is missed
by two named captures (`13.509043` and `7.389043` cells) while the median board sits at `0.061328` cells. The named next
test is the detector repair on those two captures; the arm after it gives the winner's own head this lattice. Scope
unchanged: one fixed population (28 boards, 2,986 cells), both fixed label vectors, one seed, `64` steps where a step is
needed.

**After M51 the detector repair is measured, and it is under-specified rather than wrong.** Replacing M50's wall-run
generator with one declared objective over wall and cell-grid candidates - scored by wall support, grid alignment,
cell uniformity and stack containment, applied identically to all 28 boards - finds a box of the right *size* on every
capture: width error `0.006892` cells and height error `0.004720`, against M50's `0.043312` / `0.017508`, with both
former outliers cut (`training-20261035-v2` `13.509043` -> `1.9285`, `training-20261030-v1` `7.389043` -> `5.2098`).
The falsifier still fires: the population mean displacement is `0.383358` cells against the declared `0.25`, and the
read falls to `0.904376` F1 at `0.0` / `0.933275` AUC - above the winner's learned head on F1 (`0.372881`) and now
`0.021254` AUC *below* it, which M50 was not. Two boards M50 placed (`training-20261031-v1` / `-v2`, `0.0588` ->
`0.9241`) break the declared no-regression rule, so the repair bought the two outliers at a measured price. The
binding constraint is the cell *pitch*, not the presence of a box. The named next test is lattice refinement by
interior fit: keep this arm's box, correct the pitch by fitting the block edges inside it, falsified if the population
mean stays above `0.25` cells. Scope unchanged: one fixed population (28 boards, 2,986 cells), both fixed label vectors,
the fixed `0.0` boundary, one read-only pass.

## The measured regression set (maintained)

The current best arm is **`m30-level-calibrate-r5`** and nothing in M34-M40 has overtaken it. Every row
below is the same fixed 2,986 cells, both fixed label vectors, the same seed, the same 64-step protocol and
the fixed global `0.0` boundary; the numbers are the receipts' own fields.

| arm | what it changes (one thing) | F1 at `0.0` | AUC | best swept | gate |
|---|---|---|---|---|---|
| **`m30-level-calibrate-r5` (winner)** | calibration of the published read | **`0.372881`** | **`0.954529`** | **`0.689775`** | per-cell / ranking / attribution all pass on its own arm |
| M34 `m34-support` | sample lattice = raster's valid support | `0.274869` | `0.883260` | `0.385925` | fail |
| M35 `m35-contrast` | + sample-mean deviations (60 inputs) | `0.277778` | `0.945596` | `0.570048` | fail |
| M36 `m36-monotone` | + one crossing per column | `0.313667` | `0.886350` | `0.420935` | fail |
| M37 `m37-stencil` | + coarse `stencil_contrast` summary (40 inputs) | `0.270096` | `0.919512` | `0.589286` | fail |
| M38 `m38-capacity` | hidden width 64 -> 128 | `0.273082` | `0.895041` | `0.463551` | fail |
| M39 `m39-dual` | both placements in one read (60 inputs) | `0.246228` | `0.836907` | `0.351779` | fail |
| M40 `m40-composed-r2` | predicted lattice blended half onto the support box | `0.260599` | `0.897918` | `0.439535` | fail |
| M41 `m41-support-contrast` | M35's features sampled at the support lattice | `0.272073` | `0.819074` | `0.340000` | fail |
| M42 `m42-support-crossing` | support sets each board's crossing level (decision side) | `0.189455` | `0.838212` | `0.374034` | fail |
| M43 `m43-column-step` | each column decided at its own sharpest step | `0.270008` | `0.752899` | `0.272654` | fail |
| M44 `m44-pixel-edge` | cell edge midpoints join the pixel read | `0.304665` | `0.934008` | `0.550622` | fail |
| M51 `m51-board-box-read` | the committed read at a candidate-search board box (wall + cell-grid, one objective) | `0.904376` | `0.933275` | `0.908497` | fail: mean displacement `0.383358` > `0.25` (`repair-under-specified`) |

Deltas against the winner (F1 at `0.0` / AUC / best swept): M34 `-0.098012` / `-0.071268` / `-0.303850`;
M35 `-0.095104` / `-0.008933` / `-0.119726`; M36 `-0.059214` / `-0.068178` / `-0.268839`;
M37 `-0.102785` / `-0.035016` / `-0.100489`; M38 `-0.099799` / `-0.059488` / `-0.226223`;
M39 `-0.126653` / `-0.117621` / `-0.337996`; M40 `-0.112283` / `-0.056611` / `-0.250240`; M41
`-0.100808` / `-0.135455` / `-0.349775`; M42 `-0.183426` / `-0.116317` / `-0.315741`; M43
`-0.102873` / `-0.201629` / `-0.417120`; M44 `-0.068217` / `-0.020521` / `-0.139153`.
M51 is a read-only transfer of the committed read-out, not a fitted arm of this family: against the winner it moves
`+0.531495` / `-0.021254` / `+0.218722`, and against M50's board box (`0.979200` / `0.999150` / `0.985507`) it moves
`-0.074824` / `-0.065875` / `-0.077010` - the read-side cost of a pitch that is `0.35` cells less tight on the median
board even though the tail is shorter.

## What the evidence settles (narrowed by M40 - superseded as an ordering claim)

**Superseded by the corrected closure section above.** This paragraph, written after M39, stated the ordering
as settled. M40 measured the missing case and the statement below is no longer true as written: the two
groups overlap on both ranking measures, a moving arm outranks two keeping arms inside that overlap, and the
overlapping pairs are confounded in the recorded configs. What survives is the narrower claim recorded in
the closure section - a graded cost of replacement within one read, monotone in the replaced share
(`0.0` / `0.5` / `1.0`), with the winner above all seven arms - not a rule that endogenous geometry outranks
exogenous geometry. The paragraph is kept as the record of what was believed before M40:

**Arms that read a fixed support lattice rank below the arm that samples its own predicted lattice because
in the winner the sampling geometry is endogenous: the lattice the read samples is the lattice the network
predicts, so the gradients that shape the features also shape where they are read, and the read is trained
to correct its own placement. A support lattice is exogenous - it is computed from the raster's valid-pixel
bounding box before the forward pass and is not a function of anything the read predicts. Replacing the
endogenous geometry with an exogenous one removes the only channel through which this arm class corrects
placement, and the measured signature of that loss is the same in every one of the six arms: confident false
positives in the dense upper bands (`1,088`-`1,239` at median margins `1.35`-`2.32`, always above the
weakest true positive), with AUC falling from `0.954529` to `0.836907`-`0.945596`. M39 shows that handing
the read *both* geometries does not repair it: an exogenous half does not become a correction just by being
placed beside an endogenous one.**

## The next structural step (recorded after M41)

The confound is settled, so the next arm is chosen by what M41 leaves standing. Geometry replacement costs
ranking at both feature sets, and the best-feature arm measured (`M35`) became the worst-ranked arm measured
(`M41`) by that change alone, so the sampled lattice must stay the predicted one and there is no feature-side
rescue to try. What is untested is the **decision side**: use the raster's support only to set the board's own
crossing level, leaving every sampled point exactly where the network puts it. That is a decision rule rather
than a sampled geometry - the one axis the evidence has not closed - and it must be judged against the winner
and the predeclared bar, not against an arm family, since cross-configuration comparisons are exactly what
this set cannot read as provenance.

**After M42 the decision side is measured too.** The support-as-crossing-level rule lost `-0.116317` AUC and
`-0.183426` F1 at `0.0` at a literal one-field isolation, so the claim that the sampled lattice must stay the
predicted one is now joined by a second, decision-side finding of the same sign. What remains open is not
whether *the* support box offset helps (it does not) but whether some other decision-side use of the support
could: this arm tested the declared box-offset level and nothing else, and one harmful rule does not exclude
all decision rules.

## The next structural step this pass superseded (recorded, measured, refused)

M40 was that step and it was refused; its numbers are in the regression set above and the question it was
tasked to close is answered in its own section - in the negative for every composition tried, with the
mechanism claim demoted from an ordering principle. The next structural step follows from what survived that
correction: the strength axis is the only clean gradient, so an arm should not replace any of the sampled
geometry; but since provenance alone does not order arms, the step must be judged against the **winner and
the bar**, not against a family of arms. The open direction is therefore to make the *decision* placement-aware
without moving the sampling geometry - for example a declared per-board term that uses the support only to set
the board's own crossing level, which is a decision rule rather than a sampled geometry and leaves every
sampled point where the network put it.

## Superseded: the previously recorded next step, and why it looked structurally different

Make the support informative but keep the geometry endogenous: sample at the **predicted lattice composed
with the support's alignment** - a declared per-board affine transform that maps the predicted board box
onto the valid-support box, applied to the predicted lattice for sampling while the predicted lattice
itself stays the differentiable geometry the read is trained through. Nothing is fitted, nothing is
sampled from a fixed grid, and the read keeps the gradient path to its own placement while gaining the
raster's correction. It is structurally different from M34-M39 in the one respect the evidence names: the
sampling geometry remains a function of what the network predicts, instead of being replaced by, or
duplicated beside, an externally computed prior. **Measured in M40 and refused** - see the closure section
above.
