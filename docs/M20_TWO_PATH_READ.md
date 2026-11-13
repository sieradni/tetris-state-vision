# M20 — the two-path read: one score from two kinds of evidence

Declared before any receipt. Status at declaration: model, config validation, trainer refusal, family gate, curator,
channel-split recording and tests written; no block run.

## Hypothesis, and why summing should raise precision rather than average two failure modes

Two reads in this programme have produced the two best *different* things:

| read | what it consumes | where its error lives | best measured result |
|---|---|---|---|
| lattice-sampled path (`occupancy_detail`, envelope, centring) | features sampled at the **decoded** cell lattice | the decoded lattice: placement error is `1.189` rows even in its best arm, so every sample inherits a registration error | best ranking `0.888912` AUC and best placement `1.189` rows (M15 combined) |
| lattice-free channel (`occupancy_lattice_free`) | image-level pooled context + **fixed** cell coordinates; never touches the predicted lattice | board registration by construction: it is right where the board sits where the model expects it, and it cannot follow a misplaced board | the only board ever within **2 wrong cells** (`0.840147` AUC) |

Their errors are on **different axes**: the lattice path's error is a per-cell *sampling* error caused by a misplaced
lattice (local, axis-aligned, but displaced); the free channel's error is a *global* registration error (correct locally,
wrong for the whole board when the origin or scale is off) plus a loss of local detail. Summing them means each cell's
score is the evidence of a sharp local sample aligned by the decoded lattice **plus** a globally anchored estimate that
cannot inherit that displacement. On the cells where one channel is wrong and the other is right - precisely the
boundary rows where the measured error density is 2.2x higher - the sum should be right where either alone is wrong.
Averaging *variance* is not the mechanism claimed; **disjoint error axes** is.

The arm therefore changes how the read is formed: `logits = lattice_sampled + gated_free_channel`, with the free
channel's gate starting at exactly zero so step zero is the published function, and the existing monotone envelope then
placing the crossing as it does in M15.

## The exact arm

| declared field | control | arm |
|---|---|---|
| `occupancy_lattice_free` | `0` | `2` (declared sum mode) |
| `occupancy_stack_monotone` | `0` | `1` (the lattice-sampled envelope) |

Everything else `0` in both arms. `occupancy_lattice_free 2` is a new **declared value**, refused without
`--occupancy-stack-monotone 1` in both config validation and the parser: the two channels are declared together or not at
all, and the single-channel forms remain their own arms (`1` is the single free read, and stays refused beside the
envelope; the envelope alone stays M15's arm).

**Numeric bar for `passes`:** primary (plain-binary F1 at `0.0`) at least **`0.05` above the reproduced control**
(`0.381640`, the programme's standing attribution floor) **and** at least one strict exact board. Anything else is
`two-path-moves-without-clearing` (primary above control) or `two-path-fails`. Because the arm contains the envelope, the
record also reports the **incremental** figure against the already-measured M15 envelope arm (`0.409161`), so the free
channel's own contribution is separable from the envelope's known effect.

## Why this arm and not another lever

Every alternative is closed by a measured receipt, not by preference: the in-cell stencil is falsified (M3, precision did
not improve); the deformation field is falsified in three arms; the stride-1 resolution lever is refused at the declared
envelope; the row-span tap is falsified twice (M18 at the boundary, M19 under a repaired level, where its apparent
separation gain did not survive); the joint skyline is falsified (M17); the crossing rule and the envelope each improve
their own thing but leave the gate at zero; the column and neighbourhood context reads are measured and not cleared. The
two-path read is the only composition of two *measured winners on disjoint axes* that has never been run.

## Failure vocabulary (declared before the run)

The sweep records each case's two channel logits beside the summed read, so the same fold can be scored three ways -
`summed`, `lattice_only` (the arm's own transforms applied to the lattice part) and `free_only` (the free channel alone) -
and the record must say **which** failure it was:

1. `complementary` - the summed AUC exceeds `lattice_only` by at least **`0.005`**: the channels add information, whatever
   the primary did.
2. `one-channel-noise-dominates` - `free_only` AUC is at least **`0.01` below** `lattice_only` **and** the sum does not beat
   `lattice_only` by `0.005`: the weaker channel's noise is diluting the sum, and the next lever is to condition that
   channel (registration-aware gating) rather than to add more evidence.
3. `no-complementary-signal` - anything else, i.e. both channels are individually informative and the sum is not better:
   the two channels are redundant at the cell level, and the next lever is **not** the read at all - it is image
   resolution/capture, i.e. the population rather than the architecture.

## Falsification

1. **Arm falsified** if the primary does not exceed the control's `0.381640` or no strict exact board appears.
2. **Gate falsified** if the strict exact-board count stays at **`0 of 28`** (boundary-exact reported beside it).
3. **Mechanism falsified** if `channel_verdict` is not `complementary`: then the stated mechanism - disjoint error axes -
   is wrong even if the primary happened to improve.

No gate weight, envelope variant, summing coefficient or threshold may be adjusted to produce movement; the gate is a
learned scalar that starts at zero and is not swept.

## Reported

Plain-binary F1 at `0.0`, AUC, TP/FN, positive rate at `0.0`, strict exact boards, boundary-exact boards, row error,
occupancy loss, swept optimum, out-of-fold F1 - per arm - plus the three channel scores on the same fold and the derived
`channel_verdict`.

## Amendment before the measured run: the sum had to stop being gated (V-143)

The first block ran, was curated, and is **void**: the free channel came out identically zero after 64 steps. Its
checkpoint shows `occupancy_lattice_free` all three tensors at exactly `0.000e+00` and `tanh(gate) = 0.000000`, and the
curator's channel split consequently reported `summed` and `lattice_only` as the same number to the last digit - the sum
had measured the M15 envelope arm, not this arm. Two fixed points, both created by zero initialization:

1. The head's **hidden** layer was zeroed as well as its output projection, so every activation was exactly zero and the
   output projection's gradient (`delta * activation`) was zero; only the biases could move, which can only produce a
   constant. (The same construction sat in `lattice_board_refine`, which was also being constructed twice in a row; that
   is the V-143 class defect and it is fixed for both heads: zero the output projection only.)
2. With the head able to train, the declared **gate** would still have blocked it: `free_read = tanh(gate) * channel` with
   `gate` starting at zero makes the gradient into the channel zero while the gradient into the gate is the channel's
   output, itself zero. So the multiplication by a learned scalar starting at zero is not an identity element, it is a
   fixed point of training.

The arm is therefore declared as the **ungated sum** `logits = lattice_sampled + free_channel`, with identity at step
zero carried by the head's zeroed output projection (which keeps step zero bitwise the published function, as the field-off
identity check below still verifies against the reproduced control) rather than by a zero scale. Nothing else changes:
same hypothesis, same two channels, same bar, same vocabulary, same control. `tests/test_residual_head_liveness.py` is the
standing instrument - it proves each declared head acquires an input-dependent output through its own training, and it
carries a negative control that shows the old construction only ever learns a constant.

A second, smaller point the profile caught: the summed form must not *build* the gate at all. An unused parameter
receives no Adam state, and the resource profile refuses a run whose every trainable parameter is not resident
(`resource profile requires resident Adam state for every trainable parameter`), which is itself the right guard - a
declared parameter nothing trains is a declaration that reads nothing.

This also reopens a reading elsewhere in the programme: any earlier arm whose declared lever was a zero-initialized
two-layer head (the single lattice-free channel, the per-board lattice refinement) measured a head that could not read its
input, so their published numbers describe the published function plus at most a constant. That is recorded in
[BUG_LOG.md](BUG_LOG.md) as V-143 and in the handoff; the affected arms' own documents are not rewritten - the correction
is stated where the defect is fixed.

## Realisation and Measured

**Realisation.** `occupancy_lattice_free` gained a declared value `2` - the free channel **added** to the lattice-sampled
read instead of gating it - behind config validation that refuses it without the monotone envelope, a parser refusal
("requires --occupancy-stack-monotone 1") for the sum without the envelope, and the existing refusals that keep the
single-channel forms (`1`, and the envelope alone) their own arms. The head keeps both parts beside the sum
(`occupancy_parts`), the sweep records each case's two channel logits beside the summed read, and the curator scores all
three on the same fold with the same grid. Family `m20-two-path-`, instrument `m20-two-path-read-r1`, curator
`scripts/record_two_path.py`, launcher `runs/launch_two_path.sh`.

`tests/test_two_path.py` (27 tests) proves the sum keeps its parts, that each channel moves the summed read, that
silencing the channel reduces the sum exactly to the lattice-sampled read, that zeroing the read-out reduces it exactly to
the free channel (so the free channel alone can carry the score), that the summed form builds no gate to get stuck on,
that the arm cannot produce the control's geometry, step-zero identity, determinism, the leakage guard, the config pairing
rule, the join's declared value set, the parser refusals, and the record bindings. `tests/test_residual_head_liveness.py`
(6 tests) is the V-143 instrument.

Receipts: `docs/evidence/m20-two-path-r1/scope.json` (curation at `1,900 MiB`), `runs/m20-two-path-<arm>-r1/` studies (64
steps), sweeps (54 forwards). One catalogue. Every stage `passed 0` on the fourth attempt: three earlier attempts were
void and are archived under `runs/attempts/` (V-142, V-143).

**Field-off identity.** Control reproduced AUC `0.8226433610683003` and plain F1 at `0.0` `0.381640` exactly.

| arm | AUC | plain-binary F1 at `0.0` | best swept F1 | out-of-fold | TP / FN | positive at `0.0` | strict exact | boundary-exact | row error | occupancy loss |
|---|---|---|---|---|---|---|---|---|---|---|
| control | 0.822643 | **0.381640** | 0.412034 @ `0.90` | 0.400399 | 661 / 6 | 49.95% | 0 of 28 | 0 of 28 | +7.819 | 0.7684 |
| two-path read | 0.898066 | **0.350636** | 0.600117 @ `3.30` | 0.459725 | 662 / 5 | 55.52% | 0 of 28 | 0 of 28 | +3.322 | 0.8314 |

**Aliasing, by assertion.** The arms' configs differ in exactly `occupancy_lattice_free` and `occupancy_stack_monotone`;
each sweep's weights equal its own study's checkpoint; the arms differ; the neighbour, span-only and span-under-level
checkpoints are all excluded by name.

| reference arm | AUC | plain F1 at `0.0` | swept optimum | out-of-fold |
|---|---|---|---|---|
| *span under level* | *0.789794* | *0.350613* | *0.364139* | *0.328718* |
| *span only* | *0.795536* | *0.244496* | *0.443255* | *0.429607* |
| *skyline recovery* | *0.620664* | *0.308333* | *0.330257* | *0.307890* |
| *crossing rule* | *0.835241* | *0.405439* | *0.428088* | *0.423254* |
| *combined read* | *0.888912* | *0.390805* | *0.502686* | *0.392122* |
| *envelope* | *0.845948* | *0.409161* | *0.442636* | *0.409639* |
| *centring* | *0.815395* | *0.374142* | *0.393230* | *0.378651* |
| *midpoint anchor* | *0.830500* | *0.310966* | *0.401408* | *0.348472* |
| *dense hinge* | *0.862838* | *0.354770* | *0.465582* | *0.465582* |
| *quantile anchor* | *0.815340* | *0.264308* | *0.404026* | *0.379323* |
| *extremal anchor* | *0.885462* | *0.310966* | *0.540811* | *0.540811* |
| *hinge* | *0.864667* | *0.251701* | *0.474210* | *0.429457* |

### Reading: `two-path-fails`, secondary `ranking-preserved`

The two-path read does not beat the published read at the declared boundary. The sum hypothesis is falsified on this population, and the channel split says which part of it failed; no gate weight, envelope variant or summing coefficient is offered in its place.

Primary delta against the reproduced control **`-0.031004`**, AUC
`+0.075423`, precision `-0.023395`, best swept
`+0.188083` (at `+2.40` in threshold), out-of-fold
`+0.059326`, row error `-4.497`, occupancy loss
`+0.0631`, positive share `+5.57` points. Against the
already-measured M15 envelope arm the primary is **`-0.058525`** (envelope
`0.409161`).

**Gate:** strict exact boards **0 of 28**, boundary-exact
**0 of 28** - recorded whichever way it fell.

### What this run did establish

The declared primary failed, and the run is still the largest structural movement this programme has measured. Every one
of these is read straight off the same fold and the same 81-point grid:

- **Best ranking ever recorded here: AUC `0.898066`, `+0.075423` over the control and above the
  previous best `0.888912`** (the M15 combined read).
- **Best placement ever recorded: swept F1 `0.600117` against the control's `0.412034`
  (`+0.188083`) and the previous best `0.540811`**, and the out-of-fold figure
  `0.459725` is above the control's `0.400399`
  (`+0.059326`) - so this is not a threshold fitted on the fold it is measured on. (The
  highest out-of-fold figure in the programme is the extremal anchor's `0.540811`, which this arm
  does not beat; its advantage is at the swept optimum and in the error.)
- **Half the placement error: mean row error `+3.322` against the control's
  `+7.819`**, better than the M19 level-repaired arm (`6.993`) that was built for
  exactly this quantity.
- **And yet: plain F1 at the fixed boundary `0.350636`, below the control's `0.381640`,
  with `55.52%` of cells positive against the control's
  `49.95%` and the best swept threshold at `3.30` rather
  than `0.90`.** The read did not lose the cells; it moved the level at which its own
  boundary sits. That is a *level* failure with a ranking gain underneath it, not a loss of signal.

### The channel split: did the sum beat its own parts?

Scored on the arm's own fold, its own weights and the standing 81-point grid - the sum, the lattice-sampled channel alone
under the arm's own transforms, and the free channel alone:

| read | AUC | plain F1 at `0.0` | best swept F1 | precision | positives at `0.0` |
|---|---|---|---|---|---|
| summed | 0.898066 | 0.350636 | 0.600117 | 0.212930 | 3109 |
| lattice-sampled channel alone (with the envelope) | 0.896211 | 0.350265 | 0.596947 | 0.212657 | 3113 |
| free channel alone | 0.607085 | 0.249190 | 0.249190 | 0.163981 | 2110 |

Gain of the sum over its lattice part **`+0.001855`** AUC (declared bar
`+0.005`); the lattice part over the free channel
**`+0.289125`** AUC (declared gap `+0.01`).

**Channel verdict: `one-channel-noise-dominates`.** The free channel is the weaker read on its own and adding it did not lift the sum above the lattice-sampled read: the weaker channel's noise is diluting the sum. The next lever is to condition that channel - a registration-aware gate - rather than to add more evidence to it.

The mechanism as *declared* is falsified: the free channel's evidence, added at scoring time, buys
`+0.001855` AUC, under the `0.005` bar, and alone it is a weak
read (`0.607085` AUC, `0.249190` at best). What the arm did produce is a
different and, for the engine, more useful fact: **the lattice-sampled channel inside this arm's weights reaches
`0.896211` AUC, where the same channel measured `0.845948` in its own arm.**
The free channel did not add evidence at scoring time; it changed what the lattice path learned while it was trained
against the sum. The declared hypothesis - disjoint error axes adding up cell by cell - is not what happened; a *training-time*
effect that acts on the other channel is.

This also settles the M3 claim: the single lattice-free channel, now that its head can actually read its input, is a weak
read on its own (`0.607085` AUC). Its published "`0.840147` AUC and the only 2-wrong-cell board"
belongs to the inert-channel era and is retracted by V-143; it is not evidence for anything.

### Limits

Twenty-eight boards, twelve images, one catalogue, one block, one seed, one launch; synthetic label-integrity
populations - not photographed boards, not complete states, not image quality. The channel split compares the summed read
with its own parts on the arm's own fold, not with another arm's weights, so "the free channel changed what the lattice
path learned" is an inference from two numbers measured in one block, not a controlled ablation of the training signal.
, twelve images, one catalogue, one block, one seed, one launch; synthetic label-integrity
populations - not photographed boards, not complete states, not image quality. The channel split is scored on the arm's
own fold and weights: it compares the summed read with its own parts, not with another arm's.
