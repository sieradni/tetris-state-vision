# M15 — per-column stack contiguity of the occupancy read

Declared before any receipt. Status at declaration: engine and admission path written, tests green, no block run.

## The question this arm asks

Fourteen arms have now argued about *where zero sits*: five supervision terms charged a shift (M9–M13, all falsified),
and M14 removed the shift outright, pinning the level to each board's own mean. M14's numbers settle the question more
precisely than its verdict reads: the level drift collapsed to `+0.57` points, the primary delta fell to `-0.0075`, the
swept optimum moved `+1.05` to `1.95` — and the pin, being the mean of a `12%`-occupied population, is still not the
operating point. So the level is not a free parameter that can be placed; it is a statistic, and no statistic of the
read is the boundary.

This arm stops placing a level and lets the read's own **structure** place the crossing. A locked stack rests on the
floor, so a column's occupancy is one contiguous run from row zero upward — a step, not an arbitrary profile — and the
only free quantity a column has left is *where that step falls*. If that is right, `0.0` needs no supervision at all:
the crossing is the column's height, and the plain-binary decision is the stack.

## The mechanism, exactly

`--occupancy-stack-monotone 1` replaces each column of the occupancy logits with its monotone envelope, computed inside
the head before the objective or any scorer sees the tensor:

    visible = logits[:, :visible_rows]                      # 20 rows, bottom-first
    envelope = cummax(flip(visible, rows), rows).values.flip(rows)   # suffix maximum, i.e. cumulative max from the top of the window down to the floor
    logits = cat((envelope, logits[:, visible_rows:]), rows)         # spawn rows pass through untouched

Read in the project's bottom-first row order this is the suffix maximum: each cell takes the highest read at or above
it. The result is non-increasing upward, so a column crosses `0.0` at most once, and that crossing is the column's
height. It is a maximum, so it can raise a read but never lower one (asserted in the transform itself).

**No label, threshold, fitted weight, margin, parameter or prevalence is read.** Monotonicity is the domain's own
constraint, not a statistic of the data: nothing here learns anything, and there is nothing to tune. `0` is exactly the
published function. Centring (`occupancy_centre 1`) is deliberately *admitted* beside it — applied after the envelope —
because the declared follow-on arm is the combination; every other declared option is refused.

## Declared constants

| constant | value | why it cannot be tuned |
|---|---|---|
| visible window | 20 rows, bottom-first | the board's own geometry; spawn rows are outside it and pass through |
| envelope direction | suffix maximum (non-increasing upward) | material rests on the floor; the mirror would assert it hangs from the ceiling |
| parameters / terms / weights / margins | 0 / 0 / none / none | a maximum has no parameters; the option is structural |
| composition order | envelope, then centring if declared | the envelope places the crossing, centring moves the level |

## Declared expected readings

The standing bars are restated unchanged: **primary `+0.05`** on the plain-binary F1 at `0.0` (control `0.381640`, so
the bar is `0.431640`), and **secondary `-0.02`** on deciding AUC.

Expected reading: **`stack-fails`**, and the declaration states why in advance so the result cannot be reinterpreted:

- **The primary is arithmetically out of reach for any placement-only transform on this population.** Control's own
  swept maximum over the standing 81-point grid is `0.412034`, which is `0.0196` *below* the bar. A transform that only
  moves where `0.0` sits cannot exceed the curve it is thresholding; it can at best reach that maximum. So the declared
  expectation is a fail on the primary, and the informative readings are the *placement* ones:

| reading | control | declared expectation for this arm |
|---|---|---|
| cells positive at `0.0` | 49.95% | higher than control: the envelope fills every cell under a high read |
| row error | +7.819 rows | a reduction if the crossing really is the stack height; an increase is the declared falsification |
| exact non-empty boards at `0.0` | 0 of 28 | ≥ control, and ideally the first arm to move this at all |
| best swept threshold | `0.90` | `≈ 0.0` if the crossing is placed by structure; far from `0.0` falsifies it |
| deciding AUC | 0.822643 | ≥ `0.802643` (the `-0.02` tolerance) to claim `ranking-preserved` |

## Declared failure mode (before the block)

A monotone column cannot represent a **floating overhang**: material read above empty space is projected *downward* onto
it, so real boards with holes and overhangs necessarily gain false positives under the overhang, and a strictly monotone
read cannot express a hole at all. Two outcomes falsify the arm and both are recorded as such:

1. **the swept optimum stays far from `0.0`** — then the envelope did not place the crossing and the structure carries no
   information the level could use; or
2. **row error increases while the exact-board count does not move** — then the envelope is trading cell accuracy for a
   placement it does not actually achieve.

Either way the honest verdict is that per-column monotonicity cannot place the crossing, and the negation is recorded
with its deltas. The overshoot variant (an optimum *below* zero) is not expected: a maximum can only raise cells, which
moves the optimum upward if anything.

## Arms

| arm | flag | role |
|---|---|---|
| `m15-stack-control-r1` | `--occupancy-stack-monotone 0` | must reproduce AUC `0.8226433610683003` and plain F1@`0` `0.381640` exactly, else the comparison is void |
| `m15-stack-contiguous-r1` | `--occupancy-stack-monotone 1` | the treated arm |

Both: 64 steps, seed `20261026`, `--init-weights-from runs/m3-percell-balance-r1`, `--development-groups 6`,
`--evaluate-every 64`, `--checkpoint-every 32`/`64`, then the standing 54-forward sweep. Evidence
`docs/evidence/m15-stack-contiguity-r1/`, instrument `m15-stack-contiguity-r1`, run family `m15-stack-`.

## Refusals, through the real parser with stderr asserted

`0` and `1` are admitted, `0` remains the published objective; `--occupancy-stack-monotone 2`, `-1` and a non-integer are
refused by the parser. Beside it, the trainer refuses the normalised output, all five supervision terms
(`margin_hinge`, `anchor`, `quantile_anchor`, `dense_hinge`, `midpoint`) and every read-moving lever (`context`,
`detail_stencil`, `resolution`, `deform`, `placed`, `column`, `offset_rows`, `lattice_free`), each asserting the
declared wording. Centring is admitted by declaration, not by omission.

## Aliasing guard

Structural options change the read globally, so the block must prove it measured its own checkpoint, at curator level and
by assertion: each sweep's `weights_sha256` equals *its own* study's `checkpoint_sha256`; the two arms' weights differ;
each arm's config and recovery contract carry its own declared value; the arms' configs differ in exactly
`occupancy_stack_monotone`; and the neighbour checkpoint (M14's treated arm) is excluded **by name**. Any aggregate that
looks suspiciously identical to a reference row is explained by an assertion, never by inspection.

## Fit gate

28 non-empty training boards; `cleared` requires all 28 exact at `0.0`. Expected: not cleared — no arm has cleared it,
one board is the standing best (M12/M13).

## If it fails

The negation to record is: *per-column stack contiguity cannot place the crossing either*, with the deltas above. What
it would rule out is the last label-free placement story: level statistics, supervision and now structure have each been
tried, and if structure fails too then a placement that needs no label does not exist for this read, and the remaining
work is the read's ranking (AUC `0.82`, whose swept maximum caps F1 at `0.412`) rather than its operating point.

## If it clears

Immediately run the declared follow-on: the combined contiguity-plus-centring arm (`--occupancy-stack-monotone 1
--occupancy-centre 1`), and report its block's numbers in the same turn.

## Realisation and Measured

**Realisation.** `stack_contiguous` in `src/tetris_vision/model.py` (suffix maximum in bottom-first order, asserting it
never lowers a read), `occupancy_stack_monotone` as a validated config field refused beside `occupancy_normalise` and
admitted beside `occupancy_centre`, applied in the head before centring and before any loss or scorer;
`--occupancy-stack-monotone 0|1` with the declared refusals; `require_occupancy_stack_monotone` in the admission join
(config, arguments, recovery contract); the field in the config, the recovery contract and the summary; family
`m15-stack-` registered in the run gate before launch; `tests/test_stack_contiguity.py` covering the single crossing, the
hole raised to the material above it, the overhang projected downward, the already-monotone column (exact identity), the
constant column, the single visible cell, equivariance to a board-wide shift, per-column and per-board application,
spawn-row pass-through, and both settings plus every refusal through the real parser. `0` is the published function.

Receipts: `docs/evidence/m15-stack-contiguity-r1/scope.json` (curation at its declared `1,900 MiB`),
`runs/m15-stack-<arm>-r1/` studies (64 steps, `1,792 MiB`, `342.1 s` and `338.5 s`), `runs/m15-stack-<arm>-r1-sweep-r1/`
sweeps (54 forwards, `640 MiB`). One new catalogue, `d9d867a6`. All eight stage receipts `passed 0`.

**Reproduction first.** Control came back bit-identical: AUC `0.8226433610683003` and plain F1 at `0.0` `0.381640`,
exact string equality on both, so the comparison counts.

| arm | AUC | plain-binary F1 at `0.0` | best swept F1 | precision / recall at `0.0` | cells positive at `0.0` | exact boards at `0.0` | out-of-fold F1 | row error | occupancy loss |
|---|---|---|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 0.2363 / 0.9910 | 2,797 (49.95%) | 0 of 28 | 0.400399 | +7.819 | 0.7684 |
| per-column monotone envelope | 0.845948 | **0.409161** | 0.442636 @ `2.85` | 0.2578 / 0.9910 | 2,564 (45.79%) | 0 of 28 | 0.409639 | **+1.255** | 0.8494 |
| *M14 centred read (reference)* | *0.815395* | *0.374142* | *0.393230* @ *`1.95`* | — | *2,829 (50.52%)* | *0 of 28* | *0.378651* | *+8.007* | *0.6166* |
| *M13 midpoint anchor (reference)* | *0.830500* | *0.310966* | *0.401408* @ *`3.00`* | — | *3,610 (64.5%)* | *1 of 28* | *0.348472* | *+9.779* | — |
| *M12 dense hinge (reference)* | *0.862838* | *0.354770* | *0.465582* @ *`3.15`* | — | *3,065 (54.7%)* | *1 of 28* | *0.465582* | *+8.544* | — |
| *M11 quantile anchor (reference)* | *0.815340* | *0.264308* | *0.404026* | — | *4,365 (78.0%)* | — | *0.379323* | *+10.020* | — |
| *M10 extremal anchor (reference)* | *0.885462* | *0.310966* | *0.540811* | — | *3,610 (64.5%)* | — | *0.540811* | *+9.779* | — |
| *M9 hinge (reference)* | *0.864667* | *0.251701* | *0.474210* | — | — | — | *0.429457* | *+10.691* | — |

**The block is proven to be its own, by assertion rather than by eyeball.** The arms' configs differ in exactly
`occupancy_stack_monotone`; each sweep's weights equal its own study's checkpoint (`bf1ca813…`, `5976d8fa…`); the two
weights differ from each other and from the previous block's checkpoint, which the curator excludes by name. Control's
numbers are the published ones by construction; every treated aggregate differs from every reference row.

### Reading: `stack-moves-without-clearing`, secondary `ranking-preserved`

Primary delta **`+0.027521`** (bar `+0.05`) — the declared reading is **`stack-moves-without-clearing`**, and it is the
first arm in the whole program to move the primary in the right direction at all. Secondary AUC delta **`+0.023305`**, a
ranking *gain* — `ranking-preserved`, and then some. Best swept F1 **`+0.030602`**, out-of-fold **`+0.009240`**, cells
positive **`-4.16` points**, row error **`-6.564`** rows, occupancy loss **`+0.081`** (the arm fits its labels slightly
worse, as a constrained read should). Fit gate **FALSE** at `0` of `28`.

**What this pass establishes.** Placement by structure is real, and it is the only mechanism that has produced it. The
row error — how far each column's decided top sits from its true top — fell from `7.819` rows to **`1.255`**, a factor of
six, and the cell-level gain came with *identical* recall (`661` true positives, `6` false negatives, as control) and
`233` fewer false positives at `0.0`. The envelope is not a threshold in disguise: it removed no ranking and in fact
improved it (`0.822643` -> `0.845948`), which a monotone projection can do only by suppressing isolated high reads that
were never stacks. And for the first time a treated arm's *swept maximum* beats the standing bar: `0.442636` against
`0.431640`, where control's own ceiling is `0.412034`.

**What the declaration said, and what happened.** The predeclaration named the primary as arithmetically out of reach for
a placement-only transform (control's swept ceiling is `0.0196` below the bar) — and it was, at `0.0`: `+0.0275` short.
It also named two falsifications. The first is observed: **the swept optimum did not move to `0.0`** — it moved to
`2.85`, `+1.95` away. So the envelope places the *shape* of the crossing but leaves its level free, exactly as its
equivariance says it must (`envelope(L + c) = envelope(L) + c`). The second is not observed: row error fell sharply
rather than rising, so the overhang failure mode did not dominate on this population.

### What the seventh arm rules out, and what it points at

It rules out the claim that the read's structure cannot place the crossing: the crossing now follows the stack to within
`1.26` rows. It also *disproves* the placement ceiling as an absolute: with structure, the curve's maximum clears the bar
even though `0.0` does not. What remains is precisely one degree of freedom — the level of a monotone read. That is the
one thing M14 pinned (to the mean) and M13 charged (and failed); it is now the only quantity standing between this arm and
the bar, and the declared follow-on is the two together.

**Declared follow-on, predeclared before its own receipt:** the combined arm `--occupancy-stack-monotone 1
--occupancy-centre 1`, envelope first and centring after it, so the crossing is placed by structure and its level is the
board's own mean instead of a free parameter. Its declared reading: **`stack-clears-the-crossing`** if the primary clears
`+0.05`, **`stack-moves-without-clearing`** if it moves again without clearing; its declared failure mode: centring moves
the crossing to the mean, which is not the operating point on a `12%`-occupied board (M14 measured exactly that as
`+1.95`), so the combination can move the primary in either direction and both are reported.
Its declared numbers, fixed before its own receipt: the primary needs `>= 0.431640` (control `0.381640` + `0.05`), which
is `+0.022479` above the envelope arm's `0.409161`, and the envelope arm's own swept maximum `0.442636 @ 2.85` says the
curve already reaches it, so the question is only whether centring moves `0.0` onto that curve. Declared expectations:
`0.0` sits at the board mean (`~50%` positive, like M14's `50.52%`), the row error stays near the envelope's `1.255`
rows (the crossing's *shape* is structural and centring cannot change a within-column difference), AUC stays within the
`-0.02` tolerance of `0.845948`, and the fit gate stays `0` of `28` unless the mean happens to coincide with the stack.


## Follow-on: the two structural options together (the declared combination)

Predeclared above before its own receipt; the treated arm `m15-stack-combined-r1` carries `--occupancy-stack-monotone 1
--occupancy-centre 1`, envelope first and centring after it, and is curated against the M15 block's own control in the
same catalogue `d9d867a6`. Receipts: `docs/evidence/m15-stack-combined-r1/scope.json`,
`runs/m15-stack-combined-r1/` (64 steps, `1,792 MiB`), `runs/m15-stack-combined-r1-sweep-r1/` (54 forwards, `640 MiB`),
curation at `1,900 MiB`; all `passed 0`. Control is the published `0.8226433610683003` / `0.381640` unchanged.

| arm | AUC | plain-binary F1 at `0.0` | best swept F1 | precision / recall at `0.0` | cells positive at `0.0` | exact boards at `0.0` | out-of-fold F1 | row error | occupancy loss |
|---|---|---|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 0.2363 / 0.9910 | 2,797 (49.95%) | 0 of 28 | 0.400399 | +7.819 | 0.7684 |
| envelope only | 0.845948 | **0.409161** | 0.442636 @ `2.85` | 0.2578 / 0.9910 | 2,564 (45.79%) | 0 of 28 | 0.409639 | +1.255 | 0.8494 |
| envelope + centring | 0.888912 | **0.390805** | **0.502686** @ `1.20` | 0.2433 / 0.9940 | 2,726 (48.68%) | 0 of 28 | 0.392122 | +1.846 | 0.5987 |

Reading **`stack-moves-without-clearing`** (primary `+0.009165`), secondary `ranking-preserved` (AUC **`+0.066269`**),
verdict `not-cleared`, fit gate FALSE at `0` of `28`. Against the envelope arm alone: primary `-0.018356`, AUC
`+0.042964`, swept maximum `+0.060050`, row error `+0.591`.

**What the combination established.** Centring does supply the level the monotone read was missing, and the level is
worth a great deal on the parts of the curve that are allowed to move: deciding AUC `0.888912` is the highest any arm in
this program has measured (above M10's `0.885462`), the swept maximum `0.502686 @ 1.20` is `+0.091` over control and the
first that clears the `0.431640` bar by a wide margin, and the read now misses fewer occupied cells than control (`4`
false negatives against `6`) while costing `73` fewer false positives at `0.0`. The paid occupancy loss fell to `0.5987`,
the lowest of any arm.

**What it did not do.** The plain-binary primary at the standing `0.0` moved `+0.009165` - positive, but a third of what
the envelope alone achieved, and `0.018` below it. Two of the declared expectations were met and one was falsified on its
face: `0.0` sits at the mean (`48.68%` positive, as declared) and the row error stayed near the envelope's (`1.846`
against `1.255`, declared "near"), but the mean is *not* the operating point - the swept optimum sits at `1.20`, so
centring moved the crossing `1.65` logits closer to zero without landing on it. The declared failure mode was exactly
this, and it happened: on a `12%`-occupied board the board mean is still the wrong level, just a better one than a free
parameter left to the training.

**The honest state of the program after fifteen arms.** The bar cannot be cleared at a fixed `0.0` by any of the seven
mechanisms tried (five supervised terms, a level statistic, a monotone envelope, and the two together), while the same
read now reaches `0.502686` F1 at its own optimum - above the bar, above every earlier arm, and with the best ranking in
the program. So the remaining obstacle is no longer the read: it is the rule that the operating point is `0.0`
regardless of what the read's distribution looks like. A future arm that keeps the read and lets the *decision rule*
follow the crossing it can see from the structure (a column's decided height under a monotone read) would no longer be a
tuning change - it would be a declared, label-free decision rule - and it is the one direction the numbers now point at.

### Limits of the follow-on

Same population, same catalogue, same seed; the control is re-used (same catalogue, same envelope) rather than
re-measured, and the curator asserts the shared catalogue. No real-image, complete-state or quality claim.

### Limits

Twenty-eight boards, twelve images, one catalogue, one block, one seed, one launch. Two arms. The placement gain is
measured on synthetic label-integrity populations; it is not a real-image, complete-state or quality claim, and the
`1.26`-row row error is a diagnostic of the decided stack height, not a claim about a photographed board.

## Correction (V-139, 2026-11-12)

The fit gate quoted in this document as `0 of 28` is right, but it is right by accident: the curator that produced it
dropped any column whose decision or label was empty. Under the corrected strict definition the figure is `0 of 28` for
every arm in the program (79 of them), and the `1 of 28` credited to M12/M13 elsewhere is withdrawn.
