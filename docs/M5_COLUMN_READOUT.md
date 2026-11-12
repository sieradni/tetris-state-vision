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

## Measured

*(appended after the receipts exist - see the record under `docs/evidence/m5-column-readout-r1/`)*
