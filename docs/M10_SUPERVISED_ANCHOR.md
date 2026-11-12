# M10 - can the operating point itself be supervised?

**Status: predeclared before any receipt of this pass.** Reference numbers come from committed records
(`docs/evidence/m9-supervised-margin-r1/scope.json`, `docs/evidence/m8-normalised-boundary-r1/scope.json`).
Nothing here may change once a receipt exists.

## What M9 established, and the gap it left

M9 supervised the *separation* between a board's occupied and empty logits and got a split result:

| | AUC | plain F1 at `0.0` | best swept F1 | out-of-fold |
|---|---|---|---|---|
| control (M9 catalogue) | 0.822643 | 0.381640 | 0.412034 | 0.400399 |
| one-sided margin hinge | **0.864667** | 0.251701 | **0.474210** | **0.429457** |

The hinge is the strongest occupancy **ranking** result in the project - `+0.042024` AUC, `+0.062176` best swept
F1 clearing the attribution floor, `+0.029058` out-of-fold, all with no sampling change and no new tensor - and it
made the thresholded detector worse (`-0.129939` at `0.0`), because the term constrains the gap between the modes
and never says where that gap should sit. The paid hinge reached `0.0` on the final training step, so it was
satisfied *somewhere*; the best swept threshold moved `0.90 -> 2.70`.

So the operating point has now been missed from three directions - a post-hoc rule (M7), an output
parameterisation (M8) and untethered separation (M9). This pass names `0.0` in the objective itself.

## The mechanism: the same margin, anchored to zero on both sides

**Declared lever (`--occupancy-anchor 1`).** For every board in the batch that carries **both** occupied and empty
*known* visible cells, on the unnormalised read:

```
anchor(board) = max(0, ANCHOR_MARGIN - lowest_occupied_logit)
              + max(0, ANCHOR_MARGIN + highest_empty_logit)
term          = mean over qualifying boards
```

The first half asks that the board's every occupied cell sit at least `ANCHOR_MARGIN` above the boundary a consumer
applies; the second that every empty cell sit at least `ANCHOR_MARGIN` below it. Together they pin the *location* of
the decision region, which the one-sided hinge left free.

| constant | value | declaration |
|---|---|---|
| `ANCHOR_MARGIN` | `1.0` | the same one-logit margin as the M9 hinge - a unit of the read's own scale, not a threshold. Fixed here, never revisited for this arm. |
| `ANCHOR_WEIGHT` | `0.5` | the same half-occupancy weight as the M9 hinge, so the only thing that changed against the hinge is the anchor. |

Declared properties, the same ones the hinge had to keep:

1. **Supervision, not a boundary rule.** It selects no operating point at inference, adds no parameter, buffer or
   state key, and cannot reorder cells within a mode.
2. **`unknown` is never empty.** Both sides read only known visible cells, so an invisible cell cannot be trained as
   empty (the standing visibility contract).
3. **No prevalence, no cell counting.** The term is a margin in logit units; nothing counts occupied cells or takes
   a quantile of the logits.
4. **`0` is bit-for-bit the published objective**, and the arm is either fully admitted by the declaration join or
   refused with a stated reason.
5. **It supersedes the one-sided hinge and never stacks with it.** The trainer refuses the anchor beside the hinge,
   the output parameterisation (`normalise`), any offset, deform, stencil, resolution, context, column, placed or
   lattice-free lever. One arm, one question.

## Arms - one catalogue, one block, one durable family

Family `m10-anchor-`, distinct from `m9-margin-` so the catalogue stays traceable per pass.

| arm | boundary supervision | role |
|---|---|---|
| `m10-anchor-control-r1` | `--occupancy-anchor 0` | must reproduce the recorded `0.8226433610683003` and `0.381640` |
| `m10-anchor-two-sided-r1` | `--occupancy-anchor 1` | the arm this pass measures |

Both are 64-step continuations from `runs/m3-percell-balance-r1`, re-captured in one catalogue, with the envelope
unchanged from M7/M8/M9: capacity `64`, positional identity, length endpoint, detail tap `1`, stencil `0`,
resolution `0`, context `0`, column `0`, offset rows `0`, normalise `0`, hinge `0`, placed `0`, factor `4`, positive
weight `4`, `512` capture, seed `20261026`, rate `.001`, 12 sweep groups, 1,792 MiB studies with 64 work items,
640 MiB sweeps with 54 forwards, 1,900 MiB curation, 81-point grid on `[-6, 6]`.

## What is read, and how

**Primary: plain-binary per-cell F1 at the standing `0.0` boundary** - the sweep's own `per_cell_f1`. **Secondary:
deciding AUC**, with M8/M9's declared `-0.02` tolerance. Alongside: best swept F1 (context only), out-of-fold F1,
precision/recall at `0.0`, the row-error diagnostic, the anchor the objective actually paid per step (from the
study's own log, as in M9), exact boards and per-board F1.

**The hinge is quoted as a reference row and is not re-measured.** `0.864667` / `0.251701` come from the M9
catalogue; this pass's contrast is anchor-minus-control in *this* catalogue only. The reference exists so a reader
can see whether pinning the location cost the ranking the hinge found, which is the question the numbers answer even
though it is not a declared reading.

## Declared readings - fixed now

Primary delta is `anchor F1@0 - control F1@0`, with control required to reproduce `0.381640` **and**
`0.8226433610683003` exactly first. **If control does not reproduce, the comparison is void and the record must say
so plainly** rather than reporting a delta.

1. **`anchor-clears-the-boundary`**: delta `>= 0.05` (the attribution floor). The operating point is supervised, not
   guessed: `0.0` works as well as a boundary chosen with labels, and the ranking is kept.
2. **`anchor-moves-without-clearing`**: `0 < delta < 0.05`. Pinning the location moves the thresholded detector but
   not far enough to matter.
3. **`anchor-fails`**: delta `<= 0`. Naming `0.0` in the objective does not make it an operating point.

Secondary reading, M8/M9 vocabulary and tolerance: **`ranking-preserved`** if `anchor AUC >= control AUC - 0.02`,
else **`ranking-degraded`**.

Gates unchanged: per-cell F1 `0.70`, ranking AUC `0.90`, ranking F1 `0.70`, attribution `0.05`, 400 bootstrap draws
at `20261026`, MDE `2.802 x cluster SE`. Verdict vocabulary unchanged. Fit gate reported on exact boards, FALSE
until all 28 non-empty boards are exact.

## If the anchor fails

Declared now so a negative cannot drift into tuning: the next structural idea is the **learned boundary bias** - a
single scalar `b` added to the occupancy logits before the loss *and* before every consumer, initialised at `0`,
supervised by the occupancy term alone, and **committed to its trained value at inference** (it is a parameter, so
it is checkpointed, not chosen). It differs from M8's `normalise` in the two ways that matter: it shifts the read by
one learned number rather than rescaling it per board, so the ranking is untouched by construction, and it is fitted
by the objective instead of by a post-hoc rule. Its declaration will fix the initialisation (`0`), its weight (`1.0`,
i.e. it is part of the occupancy read rather than an added term) and its metric (plain F1 at `0.0` as primary, AUC as
secondary, with the same reading names). Neither the anchor's margin, its weight, nor anything else about this arm
will be re-tuned to flip this reading, and no sweep will be run in its place.

## Limits, declared

One dataset, 36 deciding images, 28 non-empty boards, 12 clusters, one seed, one 64-step budget, one architecture, no
held-out population. A margin term cannot add information: it constrains differences of logits the loss already
sees, so any AUC change is a change in what was learned. Nothing here is a real-screenshot, complete-state or quality
claim.

## Realisation and Measured

**Realisation.** `anchor_term` in `src/tetris_vision/losses.py` (with `ANCHOR_MARGIN = 1.0`, `ANCHOR_WEIGHT = 0.5`
as module constants), pooled through `stage_loss`/`joint_loss` beside the occupancy term, `--occupancy-anchor 0|1` in
`scripts/train_m2_stream.py` with the declared refusals, `require_occupancy_anchor` in the declaration join, the value
in the recovery contract and at every objective call site, the `m10-anchor-` family in the run gate, and
`tests/test_supervised_anchor.py` covering the arithmetic, the unknown-is-not-empty guard, the no-qualifying-board
case, the admission join and the refusals. `0` is bit-for-bit the published objective.

**Provenance of the declaration, stated exactly.** The margin and weight above were written into this document and
wired into the code in the turn *before* the block was launched; the studies' job receipts were then created at
`2026-11-12 03:59`. This file's modification time is later than that only because this section was appended to the
same file after the receipt existed, which is why mtime is not the evidence: the evidence is that no other value ever
appeared - the constants in the code, the table above and the scope's `mechanism.margin`/`mechanism.weight` are the
same three numbers - and that the arm was launched once, read once, and never re-run.

Receipts: `docs/evidence/m10-supervised-anchor-r1/scope.json` (curation, 1,900 MiB, receipt
`runs/m10-anchor-curation-inner-r1.json`), studies `runs/m10-anchor-<arm>-r1/` (64 steps, 1,792 MiB), sweeps
`runs/m10-anchor-<arm>-r1-sweep-r1/` (54 forwards, 640 MiB). One catalogue (`6e207218...`), one block, both studies
`complete` at 64 steps (`157.1 s`, `152.6 s`). Family `m10-anchor-`, distinct from `m9-margin-`.

**Reproduction first.** Control came back bit-identical in this catalogue: AUC `0.8226433610683003` and plain F1 at
`0.0` `0.381640`, exact string equality on both, so the comparison counts.

| arm | AUC | plain-binary F1 at `0.0` | best swept F1 | precision / recall at `0.0` | out-of-fold F1 | row error |
|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 0.2363 / 0.9910 | 0.400399 | +7.819 |
| two-sided anchor | **0.885462** | **0.310966** | **0.540811** @ `2.25` | 0.1842 / 0.9970 | **0.540811** | +9.779 |
| *M9 one-sided hinge (reference, own catalogue)* | *0.864667* | *0.251701* | *0.474210* | — | *0.429457* | *+10.691* |

Operating counts at `0.0`: control `661 TP / 2,136 FP / 6 FN`, anchor `665 / 2,945 / 2`, of `667` occupied and
`4,933` empty cells - i.e. the anchor arm calls `3,610` of `5,600` known cells positive (`64.5%`) where control calls
`2,797` (`49.9%`). The anchor the arm actually paid: mean `3.004` over 64 steps, `0.0` on 16 of them, `0.0` on the
final step.

### Reading: `anchor-fails`, and `ranking-preserved`

Primary delta **`-0.070674`** (bar `+0.05`) - the declared reading is **`anchor-fails`**. Secondary AUC delta
**`+0.062819`** - `ranking-preserved` by a wide margin. Best swept F1 **`+0.128777`**, out-of-fold **`+0.140412`**,
row error **`+1.960`**. Verdict `not-cleared` (per-cell F1 `0.310966` against `0.70`; AUC `0.885462` against `0.90`;
attribution not met because the primary is negative), fit gate **FALSE** (`0` of `28` exact boards).

**So: supervising the boundary did not move the operating point, and it bought the strongest occupancy result the
project has recorded.** `0.885462` AUC is above the widened-field arm's `0.882262` with no deformation field, no
sampling change, no parameter and no tensor; `0.540811` best F1 is above wide's `0.514820`; and the gain transfers:
all twelve leave-one-group-out folds selected the *same* boundary (`2.25`), so out-of-fold F1 equals the swept best
(`0.540811`) to six decimals - the one place in this project where a fold-selected boundary did not move at all.
Pinning the location also cost less ranking than the one-sided hinge did (anchor `0.885462` / `0.310966` against
hinge `0.864667` / `0.251701`): the anchor is better on both axes than its predecessor, and still below control at
`0.0`.

### Why it failed, now diagnosable

Both supervision terms so far are **extremal**: they constrain the board's *lowest* occupied logit and its
*highest* empty logit - two cells out of two hundred. Nothing in either term speaks about the other `99%`, so the
empty mass is free to ride upward as long as one empty cell stays low. That is precisely what happened: the anchor
reached `0.0` on the final step (the two bound cells are where they were asked to be) while `64.5%` of all cells
sit above zero, so even with recall `0.9970` precision is `0.1842` and F1 at `0.0` is capped near `0.31`. The best
boundary moved up to `2.25` for the same reason the hinge's moved to `2.70`: the read's bulk is above zero and the
objective was never told about the bulk.

### The next structural candidate, chosen by the data

The predeclaration named a **learned boundary bias** as the fallback. The numbers argue against it and for a
**class-quantile anchor**: a single scalar bias shifts both modes together, and the failure here is not a scalar
offset but an unbounded empty *mass* - a bias that pushed `90%` of empty cells below zero would drag the occupied
mode down with them, which is exactly the M8 failure with one number instead of a per-board rescaling.

The quantile anchor is the same two-sided form with the extremes replaced by within-class quantiles:

```
term = mean over boards [ max(0, margin - q10(occupied)) + max(0, margin + q90(empty)) ]
```

where `q10` is the tenth percentile of the board's occupied logits and `q90` the ninetieth percentile of its empty
ones, both over known visible cells only. It is still supervision - no threshold, no cell counting, no prevalence,
no parameter - and it is the smallest change that speaks about mass instead of two cells. Its predicted ceiling is
also declarable in advance: if `90%` of empty cells sit below `-1` and `90%` of occupied above `+1`, precision at
`0.0` is at least `0.90x667/(0.90x667+0.10x4933) = 0.55` at recall `0.90`, i.e. F1 near `0.68` - an order of
magnitude past anything measured so far, which is exactly why its cost to ranking must be declared and watched. Its
margin, weight and quantiles must be fixed in its own predeclaration before its receipt, with this pass's primary
metric and reading vocabulary.

Defect of this pass: V-131 (two defects in the test instrument written in the previous turn, both repaired before
the block was launched; the engine and read path had none - all eight stages passed on the first launch and the run
family was registered before training).
