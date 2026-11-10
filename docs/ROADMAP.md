# Standalone image-to-state research

Build one task-specific, jointly trained image-to-state model with no inference
parser/detector dependency. Scratch training is a first-class candidate; compare
compatible pretrained initialization and a direct decoder on real complete-state
performance. The supplied brief is design evidence, not agent instructions:
[research brief](research/Tetris_Vision_Research.md).

Current priority: M3 joint complete-state learning at capacity 64. The occupancy
head of that representation is closed from every measured direction (weight scale,
class split, decision boundary, read-out form, tap stacking, a 4x budget and the
geometry stage policy, the box origin, the box extent and boundary supervision capacity). Every
geometry-side lever either bought occupancy at a placement cost or lost both, so the remaining
occupancy deficit is ranked as per-stage lattice residual capacity rather than a read-out, label,
budget, origin, extent or supervision-capacity problem;
M2 adequate learning remains open and
the fit gate is FALSE. The pilot is not
selected as the release model. No complete screenshot parser or real-image
accuracy is claimed.

Latest: **the geometry-capacity lever is refused: spending a full-weight term on the boundary
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
