# Declared trunk-width lever: a wider decision path, fresh-baseline pair

Status: **predeclared before any receipt.** No arm has been launched. The owner authorized all
structural and wider changes on 2026-11-11, which turns the priced fork in
[M3_TRUNK_WIDTH_PRICING.md](M3_TRUNK_WIDTH_PRICING.md) from a user decision into an executable lever.
This document fixes the lever *before* the first receipt, exactly as every previous lever in this campaign.

Nothing here changes the published configuration, the published checkpoint, `TetrisStateNet-M3` or any lever-table
entry. A wider configuration is a **second declared identity** with its own name derived from its own verified
configuration.

## What the authorization does and does not settle

The authorization removes the "is this in scope" question. It does not remove the design question, and the
headroom result ([M3_DECIDING_HEADROOM.md](M3_DECIDING_HEADROOM.md), verdict `bias_limited`) makes that design
question sharper rather than easier:

- The residual deciding error is a **representational ceiling shared by two different trajectories**. So added
  capacity is the live hypothesis, and this lever tests it.
- The capacity that the evidence points at is the one the **decisions are read from**: the occupancy logit is
  `Linear(dimension, 1)` on the decoder latent, and the decoder, attention and head all live at `dimension`.
- The priced wide A (`[64, 128, 224, 256]` at dimension 192) adds **+826,272 parameters, all in the encoder**, and
  leaves `dimension`, the six decoder blocks, the attention and the decision head exactly as they are. It buys more
  image features, not more decision capacity. That is the wrong end of the model to feed first.

So the declared lever is **`dimension` 192 -> 240 with the published encoder widths**, heads staying at 6
(240 = 6 x 40, and `ModelConfig` refuses anything not divisible by the head count). This is one knob on the decision
path, it keeps the encoder's published shapes, and it is the rung between published (192) and the widest priced
trunk (240 with wider widths, which does not fit the settled ceiling).

## Why there is no exact matched start, and what replaces it

The pricing document required "a matched-start policy for *shape-changed* tensors". For this architecture that
policy **cannot be exact**, and the reason is structural rather than an implementation gap:

- A function-preserving widening duplicates channels and halves the consumer's weights, which is exact for
  convolutions.
- Every encoder stage normalizes with `GroupNorm(8, channels)`. Duplicating *some* channels inside a group changes
  that group's mean and variance, and no fixed per-channel affine correction can restore the original normalized
  values, because the statistics depend on the input. Duplicating *every* channel keeps them exactly, but 2x
  duplication takes stage widths to 96/192/384/512 and exceeds the declared 256-channel cap and the memory
  envelope.
- Attention blocks would additionally need query/key rescaling by `1/sqrt(count)` and output-column scaling by
  `1/count`; that is tractable, but it does not rescue `GroupNorm`.

An arm built by partial duplication would therefore start from a *different function* while claiming a matched
start - exactly the kind of unearned comparability the campaign refuses. The lever is consequently a
**fresh-baseline pair**: both arms start from the same seeded random initialization, with no
`--init-weights-from`, and differ only in `dimension`. Their numbers are **not comparable to the current lever
table**, whose control starts from the curated `m3-percell-balance-r1` weights; this document says so in the record
instead of letting a reader assume otherwise.

Consequence recorded honestly: a fresh random start is a weaker starting point than the curated one, so if both
arms sit near the floor the pass is `floor_limited` and cannot speak about capacity.

## The declared arms

| field | control | lever |
|---|---|---|
| run | `runs/m3-trunkwidth-base-r1` | `runs/m3-trunkwidth-d240-r1` |
| identity | `TetrisStateNet-M3` | `TetrisStateNet-M3-W48x96x192x256-D240` |
| `dimension` | 192 | **240** |
| widths | `[48, 96, 192, 256]` | unchanged |
| heads / blocks | 6 / 6 | 6 / 6 |
| initial state | seeded random, no source run | seeded random, no source run |

Both arms: `--model accuracy --component-capacity 64 --component-identity-factor 4
--component-identity-readout positional --component-endpoint-term length --occupancy-factor 4
--occupancy-positive-weight 4 --curriculum stream-v1 --capture-max-side 512 --seed 20261026 --steps 256
--learning-rate 0.001` (constant, no `--lr-final`), `--evaluate-every 64 --checkpoint-every 32
--development-groups 6 --wall-seconds 1800`.

The single declared change is `--trunk-dimension 240` on the lever arm. Every other argument is identical.

## Resource envelope, stated before launch

- Ceiling **2048 MiB** for both arms (`--memory-mib 2048`), the same value on both so the envelope is matched and
  a wider model is not handed a larger budget silently. That is below the standing `min(RAM/3, 8 GiB)` rule
  (33.5 GB physical here) and it is a *new profile*, because the settled 1,152 MiB profile was measured to fail at
  the widest priced trunk.
- Wall: internal `--wall-seconds 1800`, job wall **2400 s**, work cap **256** items (one update per item), one
  heavy job at a time.
- Expected RAM is measured by the pilot below before the arms are admitted; the predeclaration is not a claim that
  2048 MiB is enough. If the pilot exceeds it, the lever is re-priced rather than run over its ceiling.

Measured, after this declaration and before either arm (`runs/*-profile-r1`, `runs/m3-trunkwidth-d240-r1-pilot-r1`):

| run | steps | peak commit | ceiling | parameters | identity |
|---|---:|---:|---:|---:|---|
| base profile | 2 | 982,712,320 B | 2048 MiB | 8,009,097 | `TetrisStateNet-M3` |
| d240 profile | 2 | 1,071,603,712 B | 2048 MiB | 10,565,241 | `...-W48x96x192x256-D240` |
| d240 pilot | 16 | 1,021,874,176 B | 2048 MiB | 10,565,241 | `...-W48x96x192x256-D240` |

The pilot's 16 steps cost **5.304 s/step**, projecting **1,358 s** for 256 steps inside the 1,800 s internal
wall - so the declared budget is affordable and the envelope is measured rather than assumed. The lever adds
**2,556,144 parameters (+31.9%)** and, notably, fits comfortably inside the settled 1,152 MiB study profile that
the priced wide A could not, which is a direct consequence of widening the decision path instead of the encoder.
The 2048 MiB ceiling is retained on both arms so a memory difference cannot be mistaken for a capacity effect.

## The pilot gate (before either arm)

One bounded 8-step run at the lever configuration, ceiling 2048 MiB / 300 s / 8 items, must show: the configuration
is accepted, the identity is derived as declared, the saved checkpoint reloads exactly, the peak stays below the
ceiling, and per-step time projects inside the 1800 s internal wall for 256 steps. A pilot is an envelope check,
never a result.

## Standing criteria and verdict branches

Read by the existing `scripts/sweep_stream_occupancy.py` and the fit-gate contract, against the fresh control arm:

- Fit gate: **3 of 28** exact nonempty deciding boards at threshold `0.0` (`fit_gate.fit_gate_opens`); unchanged.
- Attribution floor: **+0.05** best swept F1 for the lever; AUC bar **0.90**; placement non-regression.
- Activation floor: at least **0.05** of known deciding cells must flip decisions against the control, so a null
  cannot come from a channel that never engaged.

Verdict names, fixed now, and evaluated in this order by `scripts/width_lever_read.py` (pinned by
`tests/test_width_lever.py`, so the read cannot choose a branch after seeing numbers):

- **`inactive`** (void) - fewer than **0.05** of known deciding cells flip decisions at threshold `0.0`, so the two
  arms barely differ in behaviour and no capacity statement is possible.
- **`floor_limited`** - the control arm itself sits below `AUC 0.70`, so the pass cannot speak about capacity; the
  next step is a matched-start-preserving lever or a data/representation question, not a wider model.
- **`capacity_helps`** - attribution clears +0.05 **and** either the fit gate opens (3 of 28) or the lever clears the
  AUC bar 0.90.
- **`capacity_fits_not_decides`** - attribution clears +0.05 but neither the gate nor the AUC bar moves: the added
  capacity absorbs training signal without changing the deciding read, which is a fitting result, not a quality
  claim.
- **`capacity_null`** - attribution below +0.05 with the activation and control floors met. The decision-path
  capacity hypothesis is weakened at this rung.

Placement is reported alongside (development geometry from each arm's own trainer evaluation) and a regression
beyond the standing tolerance downgrades any `capacity_helps` to a capacity trade rather than an improvement.

## Falsifiers

- Both arms collapse to a constant or near-constant decision: `floor_limited`, recorded, and no capacity claim.
- The lever improves training fit but not the development/deciding read: recorded as fitting-only capacity, not a
  quality claim.
- The lever regresses placement beyond the standing tolerance: recorded as a capacity trade, not an improvement.
- The lever's parameter delta is zero or its identity is not distinct: the pass is void (instrument failure).

## What this is not

Not real-image evidence, not a complete-state claim, not a release candidate, not a replacement for the published
identity, and not comparable to the current single-control table. One seed, synthetic population, two rungs.

## Long-term reading of the change (why this is worth the budget)

If a wider decision path helps, the campaign has a measured scaling direction and its next rungs are dimension 256
and a matched-start-preserving depth lever. If it does not help, then at this data scale and label regime the
residual is not "the model is too narrow", and the honest next investment is data/representation and real-image
labels rather than a larger network - which is a more valuable outcome than spending several such passes on width
before learning that.

## Read-out audit: the decision path already reads the whole visible board (2026-11-11)

`final["occupancy"]` is not a channel stack. The occupancy head is `nn.Linear(dimension, 1)` applied per cell
and reshaped to `(batch, visible_rows+spawn_rows, columns)` = `(1, 26, 10)`: **one logit channel per board
cell**, 20 visible rows plus the 6-row spawn margin. The probe's `[:, :20]` was therefore a **row** slice, and
because rows are bottom-first it is exactly the 20 rows the labels cover - `targets_for` builds a `(20, 10)`
target, so the 60 spawn logits are predictions the task never scores. Nothing the trunk computes for the
visible board was being discarded.

Measured over the same 54 cases, from the corrected receipt (`docs/evidence/m3-trunkwidth-r2/result.json`):

| arm | read-out shape | visible mean abs logit | visible std | spawn mean abs logit |
|---|---|---|---|---|
| control 192 | `[1, 26, 10]` | 3.958 | 2.791 | 3.298 |
| lever 240 | `[1, 26, 10]` | 1.217 | 1.352 | 0.541 |

The literal was still a latent defect: a structural change to the board height or the spawn margin would have
silently truncated the read while the receipt claimed a full-board read. The read now calls
`visible_board_logits(occupancy, config)`, which sizes the window from the verified
`config.visible_rows`/`columns`, refuses any map that is not `(1, visible_rows+spawn_rows, columns)`, and
`readout_statistics` records the shape and per-block logit statistics in every receipt. Four tests pin the
slice, the config derivation (a 26-visible-row config the literal would have got wrong) and the refusals.

**Re-run under the corrected read reproduced the r1 receipt exactly.** Fresh admission profile
(`runs/m3-trunkwidth-r2-profile-r1`, 4 forwards) then the declared pass (`docs/evidence/m3-trunkwidth-r2`,
108 forwards, 131.3 s, peak 679,268,352 B of the 1024 MiB ceiling): AUC 0.7815 / 0.8561, F1@0 0.4075 / 0.3875, best F1
0.4178 / 0.4500, 0/28 exact boards in both arms, 2,701 of 7,200 flips, verdict **`capacity_null`** - identical
to r1 in every arm field. The null is not a read-out artifact.

The threshold-scale finding is real, and the audit explains it: the wider arm's visible logits are ~3.3x
smaller (mean absolute 1.217 vs 3.958), which is the same direction its best swept threshold moved (-0.75 ->
+0.75). Ranking improved while the operating point and the fit did not.

The coupling is now cleared everywhere, not just here: `src/tetris_vision/losses.py`
(`stage_loss` and `diagnostic_metrics`), the trainer's fit read, the headroom and occupancy-sweep reads and
the three probe/pilot readers all size the window through `visible_window`/`board_window`, and
`tests/test_visible_board.py` fails if a literal row slice or a wrong-shape map comes back. The base arm's
published development evaluation reproduces to the digit under both the working tree and its frozen
pre-change archive (`docs/evidence/visibleboard-proof-current-r1`, `-archive-r1`): loss 18.55253728230794,
cell accuracy 0.8986111150847541, geometry 2.192851414283117, 0 exact boards - the cleanup moved no number.

## Outcome: `capacity_null` (2026-11-11)

Both arms trained to the declared 256 steps from the same seeded random initialization
(`initial_weights_from: null` in both summaries). The read ran under its own admission profile
(`runs/m3-trunkwidth-r1-profile-r1`, 4 forwards, 10.6 s) and then once as the declared pass
(`docs/evidence/m3-trunkwidth-r1/{result,scope}.json`: 108 forwards, 139.1 s, peak 679,141,376 B of the
1024 MiB ceiling, weights unchanged during the pass).

| | control `dimension` 192 | lever `dimension` 240 |
|---|---|---|
| parameters | 8,009,097 | 10,565,241 (+31.9%) |
| deciding AUC | 0.7815 | **0.8561** |
| F1 at threshold 0 | 0.4075 | 0.3875 |
| best swept F1 | 0.4178 (at -0.75) | 0.4500 (at +0.75) |
| exact nonempty boards | 0 / 28 | 0 / 28 |
| development AUC / best F1 | 0.9268 / 0.6452 | 0.8468 / 0.4953 |
| development geometry (cells) | 2.1929 | 5.9744 |
| training peak / wall | 991,731,712 B / 984.8 s | 1,031,798,784 B / 846.5 s |

**Verdict `capacity_null`.** Attribution is **+0.0322** against the declared +0.05 floor. The activation
floor is met on its own terms - the arms disagree on 2,701 of 7,200 known cells (37.5%, against a 5%
floor) - and the control sits at AUC 0.7815, above the 0.70 control floor, so this is not `inactive`
and not `floor_limited`: the pass can speak about capacity, and what it says is that this width does not
move the deciding read.

Three readings matter for what comes next:

- The lever's **ranking** improves (AUC +0.0736) while its **operating point** does not: F1 at the
  standing threshold falls 0.020 and the best swept threshold jumps from -0.75 to +0.75. Widening moved
  the logit scale, not the read.
- The lever is **worse on its own trainer development read** (AUC 0.9268 -> 0.8468, geometry 2.19 ->
  5.97 cells). At the declared 256 steps the wider arm spends roughly 1.35x the FLOPs per step, so this
  pass measures capacity at **step parity, not compute parity** - which is what was declared before the
  first receipt, and part of why the null is a null rather than a refutation of capacity in general.
- Under `bias_limited` the residual lives in what the representation carries. 31.9% more parameters on
  the decision path bought 0.032 F1, one board short of the fit gate and a worse fit. **Stacking width
  is not the next step.**

Long-term reading: if capacity is revisited it must be **compute-matched** (steps or samples scaled with
parameters, recorded before the run) and paired with a diagnostic that separates optimisation cost from a
representational ceiling. The published configuration keeps the decision path at 192, and nothing in this
pass supports widening it.

Limitation recorded with the outcome: the first attempt at this read
(`runs/m3-trunkwidth-r1-interrupted-attempt-job.json`) reports `wall_timeout` with an elapsed of 4,822.5 s
against a 300 s cap. The cap is enforced from `time.monotonic` in the supervisor loop, so this is the host
being suspended mid-pass rather than a guard that stopped working - a suspended host cannot run its own
supervisor. The kernel job's memory ceiling held, no child was left running, and the pass was re-run from
scratch and passed in 139.1 s. **No claim is made that a wall cap survives host suspension.**

## Consumer-level pins and a reproduced receipt (2026-11-11)

The window conversion is proven at the call sites, not only in the shared helpers. Every non-loss consumer
exposes a module-level entry point - `ensemble_headroom.visible_units`, `sweep_stream_occupancy.sweep_logits`,
`probe_local_reading.reading_map`, `analyze_m2_pilot.pilot_maps`, `probe_stream_errors.pack` and
`train_m2_stream.fit_correct` - and `tests/test_consumer_pins.py` drives each of those with a 26-visible-row
declaration, which the old literal reads as 20 cells-wide rows, and with a map whose shape cannot cover the
declared board, which must raise. Those tests also assert that their own file still routes through
`board_window`, so one reintroduced literal fails that consumer's named test.

All three converted replay instruments now run end to end on their saved populations and reproduce their
historical receipts: `ensemble_headroom.py` on its published arms over the declared 12-group population
(`docs/evidence/m3-deciding-headroom-rerun-r4/scope.json`), `sweep_stream_occupancy.py` on
`m3-occbref-baseline-r1` (`runs/m3-sweep-obcref-rerun-r4/result.json` - grid, deciding and development sweeps,
verdict `uninformative`, population, cases and weights identical) and `probe_stream_errors.py` on
`m3-fit-occupancy-r1` (`docs/evidence/m3-fit-occupancy-r1-errors-r5/result.json` - identical on run, weights,
config, architecture, totals and cases; the only differences are the two fields that receipt predates,
`occupancy_positive_weight` and `training_population`, both default-valued). Only source digests and timing
differ anywhere, so the width verdict, attribution, F1 and threshold numbers in this document are untouched.
The durable fix followed: architecture identity is resolved by `working_tree_model()`, which loads the present
`src/tetris_vision` by path under a private name, so no archive on `sys.path` can change the rule (V-090), and
`tests/test_identity_import_order.py` pins that with an archive-era shadow that lacks the derived-name helper.

The `uninformative` verdict now carries the deciding population's power (`docs/evidence/m3-sweep-deciding-power-r6/power.json`): 36 images in 12 ancestry groups and 7,200 known cells (667 occupied, 9.3%), which resolves a pooled AUC to +/-0.020 at 95%; the 0.90 bar sits 7.4 standard errors above the observed 0.8255 with the upper bound 0.8452 below it, the best swept F1 0.4237 is 0.276 short of its 0.70 bar, and the optimum threshold -0.30 is interior with an F1 range of 0.4237 across the grid. `uninformative` is a measured miss of the predeclared bars, not an artefact of an underpowered subset or an unresolvable grid.

All three replays are compared field by field by `scripts/replay_equivalence.py`, whose every protected carrier is mutation-tested in `tests/test_replay_equivalence.py`, and the suite is green at 874 tests (`runs/suite-falsification-r2-job.json`).

## Independent judgement on the guard chain (2026-11-11)

A falsification pass built scratch copies of every published receipt and ran the real checks on them. All six
corruption classes are rejected with a named reason: the three instrument receipts through
`scripts/replay_equivalence.py` (changed carrier, dropped case, duplicated case, retyped scalar, nested edit,
injected section), the deciding-power record through `sweep_deciding_power.record_problems`, and the green suite
receipt through `record_stream_run.job_receipt_problems`. Irrelevant edits pass: prose whitespace (the figure
match is whitespace-insensitive), reordered map keys, an added docstring, a changed log line and a renamed local.
Two real gaps were closed - the suite receipt had no validator at all, and the power record accepted injected
sections - and two pins were loosened because they were brittle rather than protective: the documented figure no
longer requires exact whitespace, and the record no longer pins its own script digest, which a docstring or a
renamed local would otherwise fail even though recomputation already catches behavioural change.

Worth the cost? On the corruption side, yes. The guards are one strict comparer, one record validator and one
receipt validator over receipts that already exist; they add seconds to a suite that already runs in about a
minute, and the alternative is a wrong attribution claim that no test can distinguish from a right one. The
maintenance cost is the carrier tables: every new receipt section must get a carrier or an explicit metadata
entry. The likeliest breakage a year from now is someone adding a section and silencing `unaccounted:<key>` by
adding it to the metadata allowlist, which quietly shrinks protection rather than failing. The second is path
coupling: the pins name evidence directories, so archiving or renaming evidence breaks tests although nothing is
wrong.

The remaining weakness of the deciding-power claim is not its point estimate but its independence assumption.
AUC 0.8255 against the 0.90 bar and best F1 0.4237 against 0.70 are far from the bars, and the cell-level interval
is +/-0.0197. But that interval treats 7,200 cells from 36 boards in 12 ancestry groups as independent. At the chosen
threshold the measured ICC(1) is 0.049 at ancestry-group level (600 cells per group), a design effect of 30.44,
and 0.047 at board level (design effect 10.33); the cluster-respecting half-width is +/-0.1086 at group level.
Once clustering is respected the 0.90 bar is no longer excluded, so the claim holds only under the independent-cell assumption.

Closing the gap at the current design would need four times the cells for +/-0.010: 144 images in 48 groups under independence, or about 4384 images in 1461 groups at the measured design effect (roughly 122x the present sweep). That is not feasible at 12-group granularity: the instrument needs a group-level evaluation (group-mean AUC with a bootstrap over groups) over about 1461 independent groups, which this pass does not build.
