# Contributor handoff — 2026-10-30

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
