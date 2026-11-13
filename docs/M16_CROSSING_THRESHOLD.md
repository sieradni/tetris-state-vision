# M16 — deciding at the read's own crossing, without labels

Declared before any receipt. Status at declaration: engine, admission path and tests written and green; no block run.

## The wall this arm attacks

Fifteen arms established one thing beyond doubt: the read is not the problem, the *decision level* is. At a fixed `0.0`
the plain-binary F1 sits at `0.381640` for the published read and at `0.390805` for the best structural read (envelope
plus centring), while the same read reaches `0.502686` at its own optimum, with the best AUC the program has measured
(`0.888912`). Every mechanism that tried to move the read until `0.0` became the operating point failed - five
supervised terms, a level statistic, and now two structural options together, all clearing far less than the standing
`+0.05` bar.

So this arm stops moving the read and moves the *decision*: each board is decoded at the level its own logits put between
their two groups. Nothing about the labels is consulted - the rule cannot see them - so it is not a threshold chosen
because it scores well, it is a threshold the read itself states.

## The mechanism, exactly

`--occupancy-crossing-threshold 1` subtracts, per board, the **exact Otsu level** of that board's own visible occupancy
logits, inside the head, after the envelope and after centring when both are on:

    visible  = logits[:, :visible_rows].reshape(batch, -1)          # 20x10 per board
    ordered  = sort(visible)                                        # every candidate split, no binning
    between  = i*(n-i) * (mean_below - mean_above)^2                # proportional to w0*w1*(mu0-mu1)^2
    position = argmax(between)                                      # the split with the most separation
    level    = (ordered[position] + ordered[position+1]) / 2        # the midpoint of that gap
    logits   = logits - level                                       # every consumer decides at 0.0 as before

The scan runs over every distinct gap between consecutive sorted values, so there is no bin count, no grid, no
smoothing width, no learning rate and no weight to choose. "Two groups" is what *occupied or not* means, not a knob: the
rule is the Otsu criterion with `k = 2` fixed by the definition of the label space. A constant board collapses to the
degenerate split and subtracts the constant it already is; a single visible cell is returned unchanged.

**Label-free, structurally.** The function's signature is `(logits, visible_rows)`. It cannot see a label, a target
array, an occupied mask, a prevalence, or any statistic of the training set, because none is in scope, and it is applied
to the emitted read before any loss is evaluated. This is asserted rather than asserted-by-claim: the tests call it with
the same logits and five different label arrays (flipped, inverted, all-zero, all-one) and require bit-identical output,
and they scan the executable body for any reference to labels, targets, occupancy masks, prevalence or a threshold grid.

## Declared constants

| constant | value | why it cannot be tuned |
|---|---|---|
| groups | 2 | occupied vs empty is the label space's definition, not a modelling choice |
| candidate splits | every gap between consecutive sorted visible logits | exact scan; no binning or grid parameter exists |
| level | midpoint of the selected gap | the geometric middle of the two groups the read separated |
| applied to | the whole board tensor, after envelope and centring | keeps the spawn rows on the same scale as the board |
| parameters / terms / weights | 0 / 0 / none | nothing is fitted; the rule is arithmetic on the read |

## Declared expected readings

Standing bars restated unchanged: primary `+0.05` on plain-binary F1 at `0.0` (control `0.381640`, so `0.431640`), and
`-0.02` on deciding AUC.

The comparison that matters is against the **fixed-threshold ceiling for the same read**: the M15 combined arm at `0.0`
scored `0.390805` with AUC `0.888912`, swept maximum `0.502686 @ 1.20`, row error `+1.846`, `4` false negatives. Because
the rule is applied inside the head, `0.0` downstream *is* each board's own crossing, so the treated arm's plain-binary
F1 is the rule's result and nothing else.

| reading | declared expectation |
|---|---|
| primary at the rule's crossing | `> 0.390805` and inside `0.05` of the swept maximum `0.502686` - the rule is worth its complexity only if it captures the curve's own gain |
| AUC | `>= 0.868912` (`-0.02` tolerance). A per-board shift can only *reorder* boards against each other, so AUC may move either way; a large fall would mean the rule is discarding separation the read had |
| false negatives | `<= 4` - the rule must not buy precision by missing cells |
| row error | `<= 2.5` rows, i.e. the structural placement must survive the re-decision |
| exact non-empty boards at `0.0` | `>= 1` would be the first move of the fit gate in the whole program |
| drift | whatever the rule selects; it is the measurement, not a target |

## Declared falsification (before the run)

**The decision-rule hypothesis is falsified if the primary at the rule's crossing does not exceed the fixed-threshold
ceiling of the same read (`0.390805`).** In that case the honest record is that a label-free per-board rule does not
recover the read's own optimum on this population - the swept optimum of the fixed threshold is an oracle over the
population, and a per-board rule that cannot see prevalence is not entitled to it - and no retreat into per-arm tuning,
threshold grids or the swept number as if it were the primary is permitted. The next structural candidate named by the
evidence in that case: the fit gate itself (the exact-state objective, stuck at `0` to `1` of `28` across every arm), not
another decision level.

## Arms

| arm | flags | role |
|---|---|---|
| `m16-crossing-control-r1` | all structural options `0` | must reproduce AUC `0.8226433610683003` and plain F1@`0` `0.381640` exactly, else the comparison is void |
| `m16-crossing-ruled-r1` | `--occupancy-stack-monotone 1 --occupancy-centre 1 --occupancy-crossing-threshold 1` | the rule on the best published read |

Both: 64 steps, seed `20261026`, `--init-weights-from runs/m3-percell-balance-r1`, `--development-groups 6`,
`--evaluate-every 64`, `--checkpoint-every 32`/`64`, then the standing 54-forward sweep. Evidence
`docs/evidence/m16-crossing-threshold-r1/`, instrument `m16-crossing-threshold-r1`, family `m16-crossing-`.

## Refusals, through the real parser with stderr asserted

`0` and `1` admitted, `0` the published decision; `2`, `-1` and non-integers refused by the parser. Beside the rule the
trainer refuses the normalised output, the five supervision terms and every read-moving lever (`context`,
`detail_stencil`, `resolution`, `deform`, `placed`, `column`, `offset_rows`, `lattice_free`), each asserting the declared
wording. The envelope and centring compose with it by declaration, and the published arms are untouched when it is off.

## Aliasing guard

Structural options change the read globally, so the block must prove it measured its own checkpoint, at curator level and
by assertion: each sweep's `weights_sha256` equals *its own* study's `checkpoint_sha256`; the two arms' weights differ;
the arms' configs differ in exactly `occupancy_stack_monotone`, `occupancy_centre` and `occupancy_crossing_threshold`;
and the neighbour checkpoint (M15's combined arm) is excluded **by name**.

## Fit gate

28 non-empty training boards, cleared only at all 28 exact. Expected: not cleared.

## Realisation and Measured

**Realisation.** `otsu_board` in `src/tetris_vision/model.py` (exact Otsu over every gap between consecutive sorted
visible logits, level = the selected gap's midpoint, degrading to the constant board and the single visible cell),
`occupancy_crossing_threshold` as a validated config field refused beside `occupancy_normalise` and admitted beside the
envelope and centring, applied in the head *after* both and before any loss or scorer; `--occupancy-crossing-threshold
0|1` with the declared refusals; `require_occupancy_crossing_threshold` in the admission join over config, arguments and
recovery contract; the field in the config, the recovery contract and the summary; family `m16-crossing-` registered in
the run gate before launch; `tests/test_crossing_threshold.py` covering the selected gap, the offset read decided at its
own crossing, determinism, per-board and per-column behaviour, spawn-row exclusion, the constant board, the single cell,
the declared composition order, config validation, the admission join, every refusal through the real parser, the
leakage guard and the record bindings. `0` is exactly the published decision.

Receipts: `docs/evidence/m16-crossing-threshold-r1/scope.json` (curation at `1,900 MiB`), `runs/m16-crossing-<arm>-r1/`
studies (64 steps, `1,792 MiB`), `runs/m16-crossing-<arm>-r1-sweep-r1/` sweeps (54 forwards, `640 MiB`). One new
catalogue, `ad204845`. Every stage receipt `passed 0`. The first curation attempt was refused by the curator's own
config projection (V-138); the receipts were archived to `runs/attempts/m16-curation/` and the block re-curated clean.

**Reproduction first.** Control came back bit-identical: AUC `0.8226433610683003` and plain F1 at `0.0` `0.381640`.

| arm | AUC | plain-binary F1 at `0.0` | best swept F1 | precision / recall at `0.0` | cells positive at `0.0` | exact boards at `0.0` | out-of-fold F1 | row error | occupancy loss |
|---|---|---|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 0.2363 / 0.9910 | 2,797 (49.95%) | 0 of 28 | 0.400399 | +7.819 | 0.7684 |
| envelope + centring + own crossing | 0.835241 | **0.405439** | 0.428088 @ `1.35` | 0.2554 / 0.9835 | 2,569 (45.87%) | 0 of 28 | 0.423254 | **+1.189** | 0.6550 |

**Aliasing, by assertion.** The arms' configs differ in exactly `occupancy_centre`, `occupancy_crossing_threshold` and
`occupancy_stack_monotone`; each sweep's weights equal its own study's checkpoint (`bf1ca813…`, `7cd49078…`); the two
differ from each other; and the neighbour block's checkpoint (M15's combined arm) is excluded by name. AUC `0.835241`
differs from both M15 arms, so nothing here is a re-labelled neighbour.

**The label-free claim is asserted, not described.** The rule's signature is `(logits, visible_rows)`; the tests call it
with the same board and five label arrays (flipped, inverted, all-zero, all-one) and require bit-identical output, and
they scan the executable body for any reference to labels, targets, occupancy masks, prevalence or a threshold grid. The
model's forward has no label parameter and the rule is applied before any loss exists.

### Reading: `crossing-moves-without-clearing`, secondary `ranking-preserved`

Primary delta against control **`+0.023799`**, against the declared falsification the **fixed-threshold ceiling of the
same read was broken: `0.405439` against `0.390805`, `+0.014634`**. Secondary AUC `+0.012598` (`0.835241`,
`ranking-preserved`). Best swept `0.428088 @ 1.35` (`+0.016054`), out-of-fold `+0.022855`, cells positive `-4.07`
points, row error `+1.189` (from `+7.819`), occupancy loss `-0.113`. Fit gate still FALSE at `0` of `28`.

| reference | AUC | plain F1 at `0.0` |
|---|---|---|
| *M15 combined read (the ceiling)* | *0.888912* | *0.390805* |
| *M15 envelope only* | *0.845948* | *0.409161* |
| *M14 centred read* | *0.815395* | *0.374142* |
| *M13 midpoint anchor* | *0.830500* | *0.310966* |
| *M12 dense hinge* | *0.862838* | *0.354770* |
| *M11 quantile anchor* | *0.815340* | *0.264308* |
| *M10 extremal anchor* | *0.885462* | *0.310966* |
| *M9 hinge* | *0.864667* | *0.251701* |

**What the pass established.** The rule works in the direction it was built for: deciding each board at its own crossing
beats the *same read* at a fixed `0.0` by `+0.0146`, beats the published control by `+0.0238`, keeps the structural
placement (row error `1.189` rows), improves the out-of-fold number (`+0.0229`) and lowers the paid occupancy loss
(`-0.113`). It is the first arm to break a fixed-threshold ceiling from *inside* the read, without a threshold grid and
without ever seeing a label.

**What it did not do.** It does not clear the standing `+0.05` bar (`0.405439` against `0.431640`, `0.0262` short), and
two declared expectations were falsified on their face:

1. **The rule did not capture the curve's gain**: it lands `0.0227` below its own swept maximum (`0.428088 @ 1.35`), so
   per-board selection is only partly aligned with each board's optimal level; and
2. **it bought precision with recall**: false negatives rose from `4` to `11` (recall `0.994` -> `0.984`), against a
   declared ceiling of `4`. An Otsu split on a `12%`-occupied read places the level where the two groups separate most,
   not where misses are cheapest, and this population punishes misses far more than it punishes false positives.

### The honest verdict on the decision-rule hypothesis

**Not falsified, and not sufficient.** Breaking the ceiling by `+0.0146` says the fixed `0.0` boundary was costing real
F1; being `0.0227` below the read's own swept optimum says a per-board rule that cannot see prevalence cannot recover the
population optimum, exactly as the declaration allowed. The bar is not reachable this way: `0.405439` is still `0.0262`
short of `0.431640`, and the remaining gap is not a threshold problem - it is the recall/precision trade the rule makes.

**Next structural candidate, named by the evidence.** The fit gate: `0` of `28` exact boards across sixteen arms, while
the row error is now `1.19` rows and every other aggregate is at or near its best. The read places stacks within roughly
a row; an exact-state objective is what turns "nearly placed" into "right", and it is the one quantity no arm has moved.

### Limits

Twenty-eight boards, twelve images, one catalogue, one block, one seed, one launch; synthetic label-integrity
populations, not photographed boards. The rule is label-free in the decision, but the *comparison* is what the labels are
for, and the numbers above are measurements on that population only.

## Limits

Twenty-eight boards, twelve images, one catalogue, one block, one seed, one launch. Synthetic label-integrity
populations: this is not a photographed-board, complete-state or quality claim, and a per-board rule selected from the
read is still a decision about synthetic labels.
