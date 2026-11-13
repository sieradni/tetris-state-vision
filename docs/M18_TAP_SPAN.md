# M18 — reading each cell with its row neighbourhood

Declared before any receipt. Status at declaration: engine, admission path and tests written and green; no block run.

## Why this lever, from measured evidence rather than preference

V-139 corrected the exact-board metric, and M17 falsified the last decision-side candidate: every structural
*decision* arm sits at or above the read's own row error, while every best placement and every best ranking is held by a
*read*-side arm. So the remaining distance is in what each cell's logit samples.

Two measurements fix the choice.

**Where the wrong cells are.** Over every retained sweep of the program (84 sweeps, 3,522 nonempty boards, 45,360
columns), counting cells that lie *below* a column's truth so only real errors are in the denominator:

| distance from the stack's top row | 0 | 1 | 2–3 | 4 | 5 | 6+ |
|---|---|---|---|---|---|---|
| share of cells decided wrong at `0.0` | 0.4187 | 0.4115 | 0.4300 | 0.3356 | 0.3156 | 0.1671 |
| cells | 7,800 | 14,087 | 28,570 | 10,061 | 8,452 | 32,841 |

The published read alone shows the same profile (`0.419`, `0.407`, `0.428` at distances 0, 1, 2–3 against `0.188` far
below): the errors are **2.2 times denser in the four rows at the top of each stack** than in the body beneath them. The
stack boundary - the one place where a monotone column changes value - is where the read fails.

**What the tap can express.** Every published cell logit is a function of *one* sample at the cell's own centre
(`occupancy_detail`), or of four samples strictly *inside* the same cell (`occupancy_detail_stencil`, measured in M3 and
falsified: precision_improved = false). Nothing in the nineteen-arm program has ever handed a cell the value of the row
*next to* it, so a cell cannot state the only relation that marks a boundary: *this row differs from the row below it*.
The cell's decoder feature does see its column through attention, but the sharp per-cell sample - the one quantity the
whole read is built on - is strictly local.

Hence the lever: give the tap its row neighbours. It is the only untried structural read lever the retained receipts
point at, and the alternatives are exhausted by measurement rather than by argument: the in-cell stencil is falsified,
the stride-1 resolution lever is refused at the declared envelope, the deformation field is falsified, the column and
neighbourhood context reads are measured and not cleared.

## The mechanism, exactly

`--occupancy-tap-span 1` is a declared **form of the detail tap** (`--occupancy-detail 1`), beside the published point
sample and the in-cell stencil:

    offsets = TAP_SPAN = ((0, +1), (0, 0), (0, -1))     # cell units, rows bottom-first
    tap     = Linear(widths[0]*3 -> dimension)          # zero-initialized, same name as the point tap
    logits  = published read-out( cell features + tap( three row samples ) )

The middle slot is the published sample itself, so the span *contains* the published read exactly (a projection that
reads only the middle slot reproduces the point tap's logits - asserted by test), and the two extra slots are the row
above and the row below. **Declared constants, all a priori:** three offsets, no free parameter, no width to choose, no
learnable field, no sweep. `0` is exactly the published function: the projection is zero-initialized and the buffer is
non-persistent, so step zero is bit-for-bit the published read (asserted).

## Declared falsification (before the run)

1. **Lever falsified** if the primary at `0.0` does **not exceed the published read's own primary at the same step
   budget** (the reproduced control's `0.381640`). Then a missing row comparison in the tap was not what the boundary
   errors were made of, and the honest record is falsification - with no span width, sample count or projection size
   offered in its place, because each of those is a change to the statistic being read, i.e. per-arm tuning.
2. **Gate falsified** if the strict exact-board count stays at **0 of 28**. Declared here so that it cannot later be
   softened into the boundary number or reported as a surprise.

If the gate moves, the record says by how much **and** whether the newly exact boards are genuinely exact - every visible
cell right - or the class of artefact V-139 found. Both strict definitions are reported per arm.

## Arms

| arm | flags | role |
|---|---|---|
| `m18-span-control-r1` | `--occupancy-detail 1`, everything else `0` | the published read; must reproduce AUC `0.8226433610683003` and plain F1@`0` `0.381640` exactly, else the comparison is void |
| `m18-span-tap-r1` | `--occupancy-detail 1 --occupancy-tap-span 1` | the same read with a three-row tap geometry |

Both: 64 steps, seed `20261026`, `--init-weights-from runs/m3-percell-balance-r1`, `--development-groups 6`,
`--evaluate-every 64`, `--checkpoint-every 32`/`64`, the standing 54-forward sweep. Evidence
`docs/evidence/m18-tap-span-r1/`, instrument `m18-tap-span-r1`, family `m18-span-`.

## Refusals, through the real parser with stderr asserted

`0` and `1` admitted; `2`, `-1` and non-integers refused. Without `--occupancy-detail 1` the option is refused (it is a
geometry of that tap), and so is the in-cell stencil (one geometry per arm). The span is refused beside every other tap
form, every lever that moves the sample or changes where zero sits, and every supervised read-out - so the arm's only
declared difference is the geometry it is testing.

## Reported per arm, on the corrected strict metric

Plain-binary F1 at the declared boundary (`0.0`), AUC, true positives and false negatives, **strict exact boards** (every
visible cell right) and **strict boundary-exact boards** (every column's decided top right, an empty column compared as a
value), row error, occupancy loss, and the swept optimum for reference.

## Realisation and Measured

**Realisation.** `TAP_SPAN = ((0, +1), (0, 0), (0, -1))` and `occupancy_tap_span` as a validated config field (a declared
form of `occupancy_detail`, refused beside the in-cell stencil, refused without the detail tap), the tap sampling three
row-aligned points through one zero-initialized `Linear(widths[0]*3 -> dimension)` under the published tensor name,
`--occupancy-tap-span 0|1` with the declared refusals, `require_occupancy_tap_span` in the admission join, and family
`m18-span-` registered before launch. `tests/test_tap_span.py` (22 tests) covers the declared offsets, the three sampled
rows on a synthetic board whose cells read their own row number, the span containing the published point read exactly, a
projection placed on a neighbour slot changing the read, step-zero identity, shared tensors, determinism, the leakage
guard, config validation, the admission join, every refusal through the real parser with stderr asserted, and the record
bindings.

Receipts: `docs/evidence/m18-tap-span-r1/scope.json` (curation at `1,900 MiB`), `runs/m18-span-<arm>-r1/` studies (64
steps, `1,792 MiB`), sweeps (54 forwards, `640 MiB`). One new catalogue. Every stage `passed 0`. The first launch's tap
arm died at its study admission (V-141); the block was archived to `runs/attempts/m18-span-stale` and rerun whole, so both
arms below are one catalogue.

**Field-off identity.** Control reproduced AUC `0.8226433610683003` and plain F1 at `0.0` `0.381640` exactly.

| arm | AUC | plain-binary F1 at `0.0` | best swept F1 | TP / FN | cells positive at `0.0` | strict exact | boundary-exact | row error | occupancy loss |
|---|---|---|---|---|---|---|---|---|---|
| control (point tap) | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 661 / 6 | 2,797 (49.95%) | 0 of 28 | 0 of 28 | +7.819 | 0.7684 |
| row-span tap | 0.795536 | **0.244496** | 0.443255 @ `2.55` | 633 / 34 | 4,511 (80.55%) | **0 of 28** | **0 of 28** | +9.409 | 0.9289 |

Both arms incidentally: precision `0.2363` -> `0.1403`, recall `0.9910` -> `0.9490`, out-of-fold F1 `0.400399` ->
`0.429607`.

**Aliasing, by assertion.** The arms' configs differ in exactly `occupancy_tap_span`; each sweep's weights equal its own
study's checkpoint; the two differ from each other; the neighbour block's checkpoint (M15's combined arm) is excluded by
name. Ten reference rows (M17 skyline, M16 rule, M15 combined, M15 envelope, M14, M13-M9) are carried beside the block.

| reference arm | AUC | plain F1 at `0.0` | swept optimum |
|---|---|---|---|
| *skyline recovery* | *0.620664* | *0.308333* | *0.330257* |
| *crossing rule* | *0.835241* | *0.405439* | *0.428088* |
| *combined read* | *0.888912* | *0.390805* | *0.502686* |
| *envelope* | *0.845948* | *0.409161* | *0.442636* |
| *centring* | *0.815395* | *0.374142* | *0.393230* |
| *midpoint anchor* | *0.830500* | *0.310966* | *0.401408* |
| *dense hinge* | *0.862838* | *0.354770* | *0.465582* |
| *quantile anchor* | *0.815340* | *0.264308* | *0.404026* |
| *extremal anchor* | *0.885462* | *0.310966* | *0.540811* |
| *hinge* | *0.864667* | *0.251701* | *0.474210* |

### Reading: `span-fails`, secondary `ranking-degraded`

Primary delta against the reproduced published read **`-0.137144`**; rating AUC **`-0.027107`**; precision `-0.096001`.
**Both declared falsifications fired.**

1. **Lever falsified**: reading each cell's row neighbours at the same step budget does not beat the point read at the
   declared boundary (`0.244496` against `0.381640`), and it ranks worse. No span width, sample count or projection size is
   offered as a remedy.
2. **Gate falsified**: strict exact boards stay at **`0 of 28`**, and so does boundary-exact. Nothing moved, so there is
   no newly fitting board to audit for the V-139 class of artefact.

**What the numbers actually say - and what they do not license.** The failure is a *level* displacement, not an absence
of signal. The arm's best achievable separation is **`0.443255`** at `2.55` against the control's **`0.412034`** at
`0.90`, and its out-of-fold F1 is **`0.429607`** against `0.400399`: trained on three row-aligned samples, the same read
ranks cells - and generalises - better than the point tap does at its own optimum. What it also does is decide **`80.55%`**
of visible cells positive at `0.0` (`49.95%` before) and place stacks **`1.59` rows too high**, so at the graduated
boundary the primary collapses. Three samples spanning a whole cell give the projection two neighbours whose content is
nearly anti-correlated with the cell, and the arm's gradient uses them to push the level up rather than to separate rows.
That is a diagnosis of *why this geometry fails*, not a licence to trade the level back: composing it with a level
transform is a different declared arm, and no such arm is run here.

**Not a record.** The swept optima beside the block are higher still - the M10 extremal anchor `0.540811`, the M15 combined
read `0.502686`, the M9 hinge `0.474210`, the M12 dense hinge `0.465582` - so `0.443255` is mid-pack, above the published
control and the M15 envelope (`0.442636`) and below four swept arms. Nothing here is a new best.

### Read limit or population limit

Both, and the distinction is decidable from the receipts. **The population is not the binding constraint**: with 28
boards, a read whose boards were genuinely exact even a tenth of the time would show two or three exact boards, and
nineteen arms show none; the closest board ever produced is **2 wrong cells**, from a cell-read lever. So the binding
constraint is per-cell read precision - the read's cell errors are ~10-20 per 200-cell board at AUC `0.89` and the gate
asks for zero - and it is a read limit, not a sampling artefact. The population limit shows up only *after* that: an
exactness requirement over 28 boards is a stringent conjunction, so a future arm that produces exact boards should be
judged on the count and its near misses, not on clearing all 28 in one block.

**Named next candidate.** The span arm is the first read-side lever to improve the swept optimum and the out-of-fold
figure at the same time while failing the boundary - so the evidence points at the *composition* the declaration kept
out: the row-span tap beneath the already-validated structural level transforms (M14 centring, M16's crossing rule),
declared as its own arm with its own falsification. That is the pair the receipts ask for: the span supplies separation,
the level transforms supply where zero sits.

### Limits

Twenty-eight boards, twelve images, one catalogue, one block, one seed, one launch; synthetic label-integrity
populations - not photographed boards, not complete states, not image quality. Three samples one cell apart cannot see a
step that falls between two cell centres.

## Limits

Twenty-eight boards, twelve images, one catalogue, one block, one seed, one launch; synthetic label-integrity
populations - not photographed boards, not complete states, not image quality. Three samples one cell apart cannot see a
step that falls between two cell centres, so the arm can buy boundary evidence only where the misalignment is within
about half a cell.
