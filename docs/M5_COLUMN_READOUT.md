# M5 - the column read-out: put the stack beside the cell

**Status: predeclared before any receipt.** Nothing below may change after the first receipt exists.
Declaration committed at the commit that introduces `--occupancy-column`; all receipts, sweeps and
instruments follow it.

## The question this pass answers

The placed pass measured where the objective puts its reward. Supervising the displacement lost
(`0.762448` AUC, resolvable losses against both control and wide); the de-meaned field was per-cell and
bought nothing (`0.820429`); the free widened field gained `+0.059619` AUC and its samples land with
**64.1%** outside the target board and **5.0%** outside the image frame, against 29.0% for control. So the
gain that makes this family's best arm best is a **read that has slid off the cell toward the material the
label actually reports**.

That measurement makes a prediction a read-out change can test: if the vertical structure the shifted
window exploits is available *at the cell*, the objective no longer has to buy a displacement to reach
it. This pass builds that read-out and measures whether it pays.

## The mechanism, and why this one

**`--occupancy-column 1`**: each cell's occupancy feature additionally sees its own column, summarised
strictly **below** and strictly **above** the cell - the mean decoder cell feature of those spans plus the
two fractions of the column they cover - passed through a `2*dimension+2 -> dimension` projection that is
**zero-initialised**.

Two sentences of justification, from the code this builds on. `ModelConfig` already carries `cells` as one
feature per board cell (rows bottom-first, so "below" is the strictly smaller row index) and the occupancy
head reads exactly that tensor, so a column summary is a **re-indexing of information the read-out already
has**, costing one intermediate and one small projection - not a new sampling location, which is what every
losing arm of the last two passes changed. And because the projection is zero-initialised like the detail
and deformation taps (`nn.init.zeros_` on weight and bias), step zero is exactly the published function and
the paired control starts from the same tensors, so any difference after 64 steps is the read-out alone.

The read-out is additive around the cell, not a change of sample, so the arm carries the detail tap the
rest of its family shares (`--occupancy-detail 1`) and is refused only beside the levers that change *what*
is read (stencil, resolution, neighborhood) or move it (a deformation field, a placed target).

Rejected without surveying: a second strided tap or a second encoder pass (new sampling and new cost, and
the position instrument says sampling location is exactly what is being exploited), and any change to the
deformation field or its supervision (both already measured and both negative).

## Arms - one catalogue, one budget, one population

Four 64-step continuations of the published `runs/m3-percell-balance-r1` checkpoint, differing only in the
declared lever, all captured under **one** source catalogue and the same detail tap (see the amendment
below):

| arm | `occupancy_deform` | `occupancy_column` | role |
|---|---|---|---|
| `control` | 0 | 0 | published read-out; must reproduce its recorded numbers |
| `wide` | 2 | 0 | the family's best arm; must reproduce its recorded numbers |
| `demean` | 3 | 0 | per-cell field, no gain; must reproduce its recorded numbers |
| `column` | 0 | 1 | **the new read-out** |

The three shared arms are re-captured because the catalogue moved (a run archive is bound to the source
digest it was captured under, V-112/V-113). **They must reproduce `0.8226433610683003`,
`0.8822623454135491` and `0.8204288287642111` AUC exactly, with per-cell F1 at `0.0` and best swept F1
identical** to `docs/evidence/m3-occupancy-deformation-r3/scope.json`; any difference is the finding and
stops the pass.

Shared and unchanged: seed `20261026`, learning rate `.001`, `capture_max_side 512`, curriculum
`stream-v1`, 6 development groups, evaluation and checkpoint every 64/32, `--occupancy-detail 1`,
`--occupancy-detail-stencil 0`, `--occupancy-resolution 0`, `--occupancy-context 0`, `--occupancy-factor 4`,
`--occupancy-positive-weight 4`, `--component-capacity 64`, positional identity read-out, length endpoint
term, 12 training groups for the sweeps.

### Amendment before any admissible receipt

This declaration first said `--occupancy-detail 0`. That is refused by the engine itself - a deformation
field moves the detail tap's sample, so `--occupancy-deform` requires `--occupancy-detail 1` - and the two
deformation arms therefore never ran: their profiles and studies exited 2 in seconds, before any forward.
The setting is amended to `--occupancy-detail 1` for **all four arms**, which is also the setting the three
reproduced arms were published under, so the family can exist and the only differences remain the
deformation lever and the column read-out. The control and column arms did complete under the refused
setting; those receipts are preserved but superseded, because no matched family can exist without the tap.
Full record: `runs/attempts/m5-colread-detail0/`, ledger entry V-119. No gate, threshold, population,
ceiling, weight or floor is touched by this amendment, and it is committed before any receipt that this
record curates.

Declared envelope, unchanged from the last block: study **1,792 MiB** / 64 work items / 900 s wall; sweep
profile **640 MiB** / 2 items; sweep **640 MiB** / 54 forwards; instrument **640 MiB** / 36 forwards;
curation **1,900 MiB** / 9 items. One catalogue digest across all four arms is required before anything is
published.

## Instruments

1. **The widened sweep** - the standing read-only measurement on the 36-image deciding population (28
   non-empty boards, 667 occupied cells, 4,933 empty, 12 ancestry clusters), 81-point threshold grid on
   `[-6, 6]` step `0.15`. Reports AUC, F1 at `0.0`, best swept F1 with its boundary, precision, and the
   out-of-fold F1 that transfers a boundary chosen on the other eleven ancestry groups.
2. **The column ablation** - new, read-only, one receipt per arm that carries the lever. The frozen
   checkpoint is reloaded from its own archive, the deciding population is run twice (the read-out as
   trained, then with the column projection's weight and bias zeroed), and the pass reports each side's AUC
   and F1, the mean absolute change the read-out makes to a cell's occupancy logit, and that change as a
   share of the mean absolute logit. This is what separates "the read-out was available and unused" from
   "the read-out was used and did not help". It changes no tensor: the state dict is compared before and
   after and the run is refused if any value moved.
3. **Field and position**, unchanged, for `wide` and `demean` only. Neither the `control` nor the `column`
   arm has a deformation field, so those instruments do not apply to them and no measurement is invented
   for them.

## Gates, thresholds, verdict - all as before

Unchanged: per-cell F1 gate `0.70`, ranking AUC `0.90`, ranking F1 `0.70`, attribution floor `0.05`,
precision floor `0.05`, bootstrap 400 draws at seed `20261026`, minimum detectable effect
`2.802 x cluster SE`. No gate, threshold, population, ceiling, weight or floor may move after a receipt
exists.

**Attribution is measured against `control`** - the lever is the read-out, so the published read-out is the
baseline - with `column vs wide` reported as the secondary question, because `wide` is the arm the family
currently prefers.

### Declared readings, in this order

1. **`readout-is-the-lever`**: `column - control` best swept F1 `>= 0.05` **and** its AUC interval clear of
   zero at both ends. The read-out pays on its own, without a displacement.
2. **`readout-class-fails`**: best swept F1 gain `< 0.05` and (AUC interval entirely below zero **or** AUC
   delta `<= 0`). The per-column stack context does not pay; the objective class is what fails, not the
   field.
3. **`readout-moves-without-ranking`**: anything else - the read-out clears the attribution floor but its
   ranking interval still spans zero at twelve clusters, or it moves ranking without clearing the floor.

The reading must match the rule applied to the printed numbers; the record is refused otherwise. A negative
result is a result and is recorded with whatever the ablation says about the mechanism.

**Fit gate stays FALSE unless it genuinely clears**: the pass reports the fit gate on its own numbers and
makes no claim it does not earn. 28 exact boards out of 28 would be the first honest clearance; two
consecutive passes have produced 0 of 28 on every arm.

## Declared limits

One dataset, 36 deciding images, 12 clusters, one seed, one 64-step budget, one architecture. Nothing here
is a real-screenshot, complete-state, calibrated or quality claim, and no arm's read-out is declared better
than another's outside this population and budget. The ablation measures the read-out's contribution at the
checkpoint, not why training moved it there.

## Measured (2026-11-12) - the read-out is used, and it buys recall instead of rank

Four arms ran inside 1,792 MiB and completed 64 steps under one catalogue from the published checkpoint.
**All three prior arms reproduced exactly**: control `0.8226433610683003`, wide `0.8822623454135491`,
de-meaned `0.8204288287642111` AUC, with per-cell F1 at `0.0` and best swept F1 identical to the placed
block's record. Evidence: `docs/evidence/m5-column-readout-r1/scope.json`, curated by
`runs/m5-colread-curation-r3-job.json` (passed, 9 work items).

| quantity | control | wide | de-meaned | **column** |
|---|---|---|---|---|
| deciding AUC | 0.822643 | **0.882262** | 0.820429 | 0.832796 |
| F1 at `0.0` | 0.381640 | **0.385902** | 0.376663 | 0.384302 |
| best swept F1 | 0.412034 @ `0.90` | **0.514821** @ `1.95` | 0.427911 @ `0.60` | 0.432650 @ `1.35` |
| precision at `0.0` | 0.236325 | **0.240955** | 0.235895 | 0.238370 |
| out-of-fold F1 | 0.400399 | **0.498413** | 0.427911 | 0.381790 |
| exact nonempty boards | 0 of 28 | 0 of 28 | 0 of 28 | 0 of 28 |
| study peak / wall | 1,019,527,168 B / 246.3 s | 1,093,607,424 B / 249.2 s | 1,043,214,336 B / 246.2 s | 1,024,061,440 B / 215.7 s |

**Verdict `not-cleared` on all three conditions**, and the declared reading is
**`readout-moves-without-ranking`**. It is the third branch for a real reason: the arm gains on every
deciding number, and none of the gains is resolvable at twelve clusters.

| contrast | best swept F1 | AUC | F1 at `0.0` | out-of-fold F1 | attribution |
|---|---|---|---|---|---|
| column vs control | +0.020616 | +0.010152 (-0.030013 .. +0.062077, MDE 0.061504) | +0.002662 | **-0.018609** | not met |
| column vs wide | -0.082171 | -0.049467 (-0.122459 .. +0.015537, MDE 0.096249) | -0.001600 | -0.116623 | not met |
| column vs de-meaned | +0.004739 | +0.012367 (-0.048049 .. +0.078105, MDE 0.089678) | +0.007639 | -0.046121 | not met |
| wide vs control | +0.102787 | +0.059619 (+0.029411 .. +0.094463, MDE 0.049478) | +0.004262 | +0.098014 | met |

The movement is real but small: `+0.020616` best F1 is two fifths of the declared attribution floor, its
interval comfortably straddles zero, and the out-of-fold number - the only one that says the gain
*transfers* to a boundary chosen without the cluster - goes **down** by `0.018609`. The wide arm still
beats it by `0.082171`, so making the column visible at the cell does not recover what the off-board read
was buying.

### The ablation says the read-out was used

Same frozen weights, same cells, the column projection zeroed:

| | AUC | F1 at `0.0` | precision | true positive | false positive | false negative |
|---|---|---|---|---|---|---|
| with the read-out | **0.882162** | 0.312087 | 0.185206 | **661** | 2908 | **6** |
| projection zeroed | 0.865114 | **0.331402** | **0.205386** | 572 | 2213 | 95 |

The read-out changes a cell's occupancy logit by `0.933284` on average - **42.3%** of the mean absolute
logit `2.207107` - so there is no question whether training used it: it accounts for nearly half of the
decision's magnitude. What it does with that influence is the finding. It raises true positives by **89**
and false negatives fall by **89**, while false positives rise by **695**: the column context makes the head
report a stack where one is present and also where one is not. AUC rises by `0.017048`, F1 and precision
fall. That is the signature of a read-out that has learned "this column has material in it" and not "this
cell is this filled", which is exactly what a summary of the column can express.

### What this pass establishes

**The vertical structure was not the missing information.** M4 measured that the objective's only profitable
displacement was a read of the material below the cell; this pass put that material beside the cell, and the
head used it heavily (42% of its logit) without recovering the gain. The remaining gap between the
de-meaned/control band (`0.820429`-`0.822643`) and the wide arm (`0.882262`) is therefore not a lack of
information at the cell. It is the objective's own preference, and the ablation explains its shape: the
shifted read is *more precise*, because moving the sample down-and-left picks a position whose occupancy
label is decided by a stack that is already there, while a column summary is an average and cannot say
*which* row is filled.

The concrete, measurable hypothesis this leaves is a **vertical label offset**: the column read-out raised
recall by 89 cells and cost 695 false positives, which is what a head does when its boundary sits about one
row from where the labels sit. Two passes have now shown that moving samples and adding context each fail;
the next declared instrument should measure the **signed row error** of each arm's decision boundary on the
existing frozen sweeps - read-only, no new training - before any further structural change is declared. If
that offset is there, the correction is a declared half- or full-cell sampling offset, which is geometry
rather than tuning; if it is not, the label-objective class is what fails and should be named as such.

**Fit gate: FALSE** - 0 of 28 exact boards on all four arms. No gate, threshold, population, ceiling,
weight or floor was altered after a receipt existed; the one thing that did change before the receipts - the
shared `--occupancy-detail` setting - is recorded as an amendment above and as V-119, and no reading below
depends on it being anything other than the setting the published arms were measured under.
