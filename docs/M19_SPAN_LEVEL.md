# M19 — the row-span tap under the validated level transforms

Declared before any receipt. Status at declaration: trainer refusal block, family gate, curator and tests written; no
block run.

## Hypothesis

M18's row-span tap is the first read-side lever in the programme to improve **both** the swept optimum (`0.443255`
against `0.412034`) and the out-of-fold F1 (`0.429607` against `0.400399`) while failing the declared boundary
(`0.244496` against `0.381640`). Its failure was a **level**, not a signal: it decided `80.55%` of visible cells positive at
`0.0` (`49.95%` before) and placed stacks `+1.59` rows too high, so the cells it separated correctly could not be read at
the graduated boundary.

Two structural transforms already exist, both validated in their own arms and both free of any label:

- **M14 centring** removes the read's additive mode across the board (`centre_board`), so the board's own visible mean
  sits at zero - the cheapest possible statement about *where zero sits*.
- **M16 crossing rule** re-places zero at the read's own Otsu split (`otsu_board`), so the boundary sits where the read
  actually separates.

**Hypothesis:** the span's separation plus these level transforms yields a read that is better than the published read at
the declared boundary - i.e. the span's information is real and its failure was placement all along. The composition is
measured as its own declared arm; M18 is neither re-run nor re-tuned.

## Why this composition and not another lever

Every other candidate in the programme is closed by measurement, not by preference: the in-cell stencil is falsified
(M3), the stride-1 resolution lever is refused at the envelope, the deformation field is falsified (three arms), the column
and neighbourhood context reads are measured and not cleared, and both remaining decision-side candidates (M16's rule as a
decision, M17's joint skyline) sit at or above the read's own row error, with the skyline falsified outright. The one
positive movement in the last five arms is the span's swept and out-of-fold gain, and its failure mode is exactly the
quantity the two transforms are designed to move. Composing them is therefore the single most evidence-directed next arm
available - and it is only legitimate as a **new declared arm**, because neither M18's numbers nor M14's nor M16's may be
reinterpreted to produce it.

## Order of application

The head already fixes the order, and this arm uses it unchanged: **tap geometry (span) → envelope → centring → crossing
rule**. The transforms act on the read the span produces, in that order, and nothing in the arm adds a parameter, a term,
a threshold or a swept constant.

| declared field | control | arm |
|---|---|---|
| `occupancy_detail` | `1` | `1` |
| `occupancy_tap_span` | `0` | `1` |
| `occupancy_centre` | `0` | `1` |
| `occupancy_crossing_threshold` | `0` | `1` |

Everything else is `0` in both arms, so the configs differ in exactly the three declared fields.

## Non-degeneracy (asserted, not described)

The composition must not be inert. A test asserts that, from the same trained tap weights, the logits change when the
transforms are applied (`span` alone vs `span + centre + rule`), that each transform on its own perturbs the read, and
that the composed geometry is not the control's geometry: the control's tap reads `widths[0]` values per cell and the
arm's reads `3*widths[0]`, so no reading this arm produces can be a reading of the control's tap.

## Declared falsification and readings (before the run)

**Readings, in order of precedence:**

1. `span-level-passes` - primary at `0.0` at least **`0.05` above the reproduced control** (the programme's standing
   attribution floor) **and** at least one strict exact board. This is the only reading that counts as a win.
2. `span-level-moves-without-clearing` - primary above control but short of the floor, or the floor reached with the gate
   still at zero.
3. `span-level-fails` - primary not above control.

**Falsifications, all recorded whatever happens:**

1. **Composition falsified** if the primary does not exceed the control's `0.381640`.
2. **Level hypothesis falsified** if the arm's positive share at `0.0` and its row error are *not both* closer to the
   control's than M18's were (`80.55%` and `+9.409` rows against `49.95%` and `+7.819`): then the M18 failure was not only
   a level, and the span's gain was not recoverable by moving zero.
3. **Gate falsified** if the strict exact-board count stays at **`0 of 28`** (boundary-exact reported beside it).

No penalty, no margin, no transform weight and no threshold may be adjusted to produce movement. If the arm fails, the
record says whether the failure is again level/placement or a genuine absence of signal, and names the next lever.

## Reported

Plain-binary F1 at `0.0`, AUC, TP/FN, positive rate at `0.0`, strict exact boards, boundary-exact boards, row error,
occupancy loss, swept optimum, out-of-fold F1 - per arm, on the corrected V-139 strict metric.

## Realisation and Measured

**Realisation.** The span's refusal block admits exactly the two level transforms and nothing else, and admits them
**only together** (`--occupancy-tap-span 1` with `--occupancy-centre 1 --occupancy-crossing-threshold 1`; a half
composition is refused with its own stderr message). Models, admission join and recovery contract are unchanged: the
composition is three already-validated fields, and the family gate `m19-span-level-` was registered before launch.
`tests/test_span_level.py` (17 tests) proves the transforms perturb the span's logits from the same trained tap weights,
that each transform perturbs the read on its own, that the composition is not the control's geometry (the arm's tap reads
`3*widths[0]` values per cell against the control's `widths[0]`, and with a zeroed tap the composed read equals the
level-only read exactly), that the tap field off is the published function, determinism, the leakage guard, the head
order (tap before centring before the rule), the parser refusals for a half composition and for every other lever, and the
record bindings.

Receipts: `docs/evidence/m19-span-level-r1/scope.json` (curation at `1,900 MiB`), `runs/m19-span-level-<arm>-r1/` studies
(64 steps), sweeps (54 forwards). One catalogue. Every stage `passed 0`.

**Field-off identity.** Control reproduced AUC `0.8226433610683003` and plain F1 at `0.0` `0.381640` exactly.

| arm | AUC | plain-binary F1 at `0.0` | best swept F1 | out-of-fold | TP / FN | positive at `0.0` | strict exact | boundary-exact | row error | occupancy loss |
|---|---|---|---|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 0.400399 | 661 / 6 | 49.95% | 0 of 28 | 0 of 28 | +7.819 | 0.7684 |
| span under centring + rule | 0.789794 | **0.350613** | 0.364139 @ `0.75` | 0.328718 | 629 / 38 | 52.16% | **0 of 28** | **0 of 28** | +6.993 | 0.6925 |

Precision `0.2363` -> `0.2153`, recall `0.9910` -> `0.9430`.

**Aliasing, by assertion.** The arms' configs differ in exactly `occupancy_centre`, `occupancy_crossing_threshold` and
`occupancy_tap_span`; each sweep's weights equal its own study's checkpoint; the arms differ from each other; the
neighbour block's checkpoint and **the span-only block's checkpoint are both excluded by name**. Eleven reference rows are
carried beside the block.

| reference arm | AUC | plain F1 at `0.0` | swept optimum |
|---|---|---|---|
| *span only (M18)* | *0.795536* | *0.244496* | *0.443255* |
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

### Reading: `span-level-fails`, secondary `ranking-degraded`, `level_repaired = true`

Primary delta against the reproduced control **`-0.031027`**; AUC `-0.032849`; best swept `-0.047895`; out-of-fold
`-0.071681`; precision `-0.020988`. **All three declared falsifications are answered, and two fired:**

1. **Composition falsified** - the primary does not exceed the control's (`0.350613` against `0.381640`).
2. **Level hypothesis falsified** - *the transforms worked*: positive share fell from M18's `80.55%` to `52.16%`
   (control `49.95%`) and the row error from `9.409` to `6.993`, which is *better* than the control's `7.819`
   (`level_repaired = true`). So the level was repaired exactly as declared, and the primary still fell.
3. **Gate falsified** - strict exact boards stay at **`0 of 28`**, and so does boundary-exact.

**What this means, and it reverses the M18 reading.** M18's record kept one piece of optimism: that its swept optimum
(`0.443255`) and out-of-fold F1 (`0.429607`) beat the point tap's, so the span carried real separation and only lacked a
level. M19 tests that claim directly, and it does not survive: under a repaired level the same geometry's swept optimum
falls to `0.364139` and its out-of-fold F1 to `0.328718`, both **below** the control's. The M18 gain was a property of the
displaced level, not of separable cell evidence - a read that sits high scores best at a high threshold. The honest
conclusion is therefore that the row-neighbour tap carries **no recoverable per-cell information**, and the line opened by
M18 is closed by measurement, not by opinion. That correction is recorded in the M18 document as well.

**Why the transforms did not help the primary even though they repaired the level.** Centring and the Otsu rule both
re-place zero using statistics of the board's own read; on a read whose ordering is worse than the point tap's (AUC
`0.789794` against `0.822643`), re-placing zero can only trade false positives for false negatives. The arm bought a
better row error and a more honest positive rate, and paid for both in AUC and in precision - the level was never the only
thing wrong, and the read underneath it was not better.

### The measured state of the programme, after twenty arms

Nineteen arms have now attacked the occupancy read and none has moved the strict gate off `0 of 28`. What the receipts do
establish, cleanly: the best **ranking** and the best **placement** ever measured come from the same structural read - the
M15 combined arm (`0.888912` AUC, `1.189` rows of row error) - and the closest a board has ever come to exact is **2 wrong
cells**, produced by the one read that does *not* consume the predicted lattice (`occupancy_lattice_free`, AUC `0.840147`).
Those two results are disjoint: no arm has ever combined the lattice-free read with the envelope/centred read.

**Read limit or population limit.** Both, in the order that matters. The population is not the binding constraint: with 28
boards, a read even a tenth as good at the per-board level would show two or three exact boards, and nineteen arms show
none. Per-cell read precision binds first. The population only becomes the constraint after a read produces exact boards
at all - so future arms should be judged on the count of exact boards and on near misses, not on clearing 28 in one block.

**Named next candidate.** A **two-path read**: give the read-out the lattice-free channel's evidence (the only read that
has ever produced a 2-cell board, because it cannot inherit the lattice's placement error) *and* the lattice-sampled
envelope's evidence (the best ranking and placement measured), summed rather than gated - a structure the repository
already declares for stacked residuals, and the one combination the evidence points at that has never been measured. It is
not a tuning of anything: it changes which evidence the read-out consumes.

### Limits

Twenty-eight boards, twelve images, one catalogue, one block, one seed, one launch; synthetic label-integrity
populations - not photographed boards, not complete states, not image quality.

