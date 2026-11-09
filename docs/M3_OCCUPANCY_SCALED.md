# Occupancy budget scaling — the 2x2 stencil at 4x the matched continuation

Status: predeclared (2026-11-09) and committed before any profile, study, sweep or evidence
directory for this instrument exists. The change, the control, the envelope, the gates, the floor,
the population, the work arithmetic and the **conditional registration rule** below are fixed; the
measured section is appended only after the canonical receipts exist.

## Where this comes from

The pixel-ceiling pass ([M3_OCCUPANCY_PIXEL_CEILING.md](M3_OCCUPANCY_PIXEL_CEILING.md), verdict
`separable`: F1 0.9780 / fold-mean AUC 0.9976 against the model's 0.4019 / 0.7960 on the same 28
nonempty boards) refuted the label-overlap hypothesis and closed the label family. It re-ranked
exactly one lever: **scale the matched training budget on the 2x2 stencil configuration**, because
the pixels demonstrably carry the labels and the information deficit is therefore model-side. The
stencil arm is the campaign's best-ordering read-out (AUC 0.8804, best swept F1 0.4913, the only
arm ever to meet the +0.05 attribution floor) and its budget has only ever been 64 continuation
steps. It also named **registration** as the untested alternative suspect, because the ceiling was
granted the target lattice while the model's geometry error is about 3.3 cells.

This pass tests the ranked lever at 4x budget and, if that fails to clear, settles the named
alternative on measurement. It moves no threshold, gate, population, label or prior evidence byte.

## The single declared change

**Budget only.** Both arms run **256** continuation steps from the same committed published
checkpoint `runs/m3-percell-balance-r1` (weights `9ae1fef9f649...`), 4x the stencil pass's 64, with
everything else identical to that pass:

| arm | `occupancy_detail` | `occupancy_detail_stencil` | everything else |
|---|---|---|---|
| lever | `1` | `1` | identical |
| control | `1` | `0` | identical |

Identical means `seed 20261026`, `learning_rate .001`, curriculum `stream-v1`, capture side `512`,
component capacity `64` with the positional identity read-out and length endpoint, occupancy loss
factor `4` and positive weight `4`, `occupancy_context 0`, `--profile-square` off, the same image
order, the same 1,152 MiB ceiling and the same widened deciding population. The pair is matched by
the repo's own source-identity standard: both arms run under this pass's own captured catalogue and
differ only in the stencil form of the declared detail tap.

No new parameter, knob, loss, label or read-out is introduced. The published 64-step stencil arms
are reported as an unmatched cross-pass **reference** only and cannot enter this verdict.

## Envelope and declared work arithmetic

Per arm: **1,152 MiB** aggregate commit, **256 steps**, **1,800 s** runner wall, one heavy job at a
time - all fixed before admission. If an arm cannot finish inside that envelope the outcome is
**NOT ADMISSIBLE AS DECLARED**, recorded with the measured peak and wall; no ceiling is raised, no
budget shortened, no run repeated to pass.

| launch | work items | what is counted |
|---|---|---|
| each profile | 2 | 2 optimizer updates |
| each study | **256** | 256 optimizer updates (`steps`); evaluations are not work items |
| each sweep profile | 2 | 2 forwards |
| each measured sweep | **54** | 54 forwards = 36 deciding + 18 development |
| curation | **4** | four receipts read (two studies, two sweeps) |

## Scoring, on the standing widened gates

Each arm is scored read-only on the widened deciding population (training, twelve ancestry groups,
36 images, 28 nonempty boards, 667 occupied / 4,933 empty known cells) with the standing 81-point
grid on `[-6, 6]` step 0.15 under its own 640 MiB sweep profile.

**The lever CLEARS iff all three hold** (identical to every preceding pass):

1. **Per-cell gate** - occupied-cell F1 at the replay's operating point (`0.0`) `>= 0.70`.
2. **Ranking gate** - `sweep_verdict` is `informative` (pooled AUC `>= 0.90` **and** some swept
   threshold reaches F1 `>= 0.70`).
3. **Attribution floor** - lever best swept F1 exceeds **its own 256-step control's** best swept F1
   by `>= 0.05`.

**Scaling verdict, fixed here:** `scaling_licensed` iff the lever clears all three conditions at
this budget. Otherwise scaling is **refused at the declared envelope**, with the failing condition
named mechanically and the attribution delta reported as measured against the matched 256-step
control - never asserted, and never read off the cross-pass reference.

Additional declared mechanism quantity, which cannot move the verdict: cell-level precision at `0.0`
with the whole-ancestry-cluster paired bootstrap (400 draws, seed `20261026`, twelve clusters) and
Hanley-McNeil / Wilson intervals as in the three preceding passes. No gate reads an interval.

## Conditional registration rule, declared before any receipt

If and only if the scaling verdict is `not-cleared`, this pass runs one additional read-only
instrument that settles the named alternative suspect. Declared here so it cannot be chosen after
seeing the numbers:

- **Question.** Is the occupancy error explained by misregistration - the model's cell positions -
  rather than by its read-out?
- **Pairing.** For each of the 28 nonempty deciding boards, replay the case from the scaled lever
  study's frozen archive and run one forward pass to obtain the model's **predicted lattice** and
  per-cell logits. On the target lattice, locate each model cell's centre in target cell
  coordinates and pair `logit[c, r]` with the label of the target cell that centre physically
  covers (cells mapping outside the board are excluded). This is an **oracle registration** pairing:
  it uses the model's own geometry and the committed labels only, never a new label.
- **Comparison.** Occupied-cell F1 and pooled AUC at threshold `0.0` under the standing gate pairing
  (as-is) versus the oracle registration pairing, on the same cells, with a whole-ancestry-cluster
  paired bootstrap (400 draws, seed `20261026`) on the F1 delta.
- **Rule.** `registration_implicated` iff the F1 delta is `>= 0.05` **and** its cluster interval
  excludes zero; otherwise `registration_exonerated`. The verdict names the share of the deficit
  attributable to alignment and nothing about real images.
- **Envelope.** One read-only CPU job, **640 MiB**, **300 s**, **36 forwards** one at a time
  (`scripts/run_bounded.py`), plus a curator that re-derives every published number from the
  probe's own per-cell decisions. No weights, thresholds, gates or prior evidence change.

## Receipts this pass must produce

1. `runs/m3-occscale-lever-r1-profile-r1-job.json` and `runs/m3-occscale-control-r1-profile-r1-job.json`
   (2 steps, 1,152 MiB / 300 s / 2 work items).
2. `runs/m3-occscale-lever-r1-job.json` and `runs/m3-occscale-control-r1-job.json` (256 steps,
   1,152 MiB / 1,800 s / 256 work items), each admitted only by its own profile.
3. `...-sweep-r1-profile-r1` (2 forwards) and `...-sweep-r1` (54 forwards) per arm, 640 MiB.
4. Curation `runs/m3-occscale-curation-r1-job.json` into
   `docs/evidence/m3-occupancy-scaled-r1/scope.json`, re-deriving every published number with the
   deciding arithmetic asserted before writing.
5. If scaling does not clear: `runs/m3-occscale-registration-r1-job.json` (36 forwards, 640 MiB)
   and `runs/m3-occscale-registration-curation-r1-job.json` into
   `docs/evidence/m3-occupancy-registration-r1/scope.json`.
6. The owned family `m3-occscale-*` declared in `record_stream_run.OWNED_RUN_PATTERN`, bound by a test.

Any launch that dies to a harness limit or leaves its declared arithmetic is preserved under
`runs/attempts/` with a written note and re-recorded under a fresh number; none is evidence.

## What each outcome licenses

- **Cleared at 256** - scaling carries the stencil's information into separation; the occupancy head
  opens for the first time, the fit gate is re-read, and the next step is the release-side chain
  (never a real-image claim from synthetic evidence).
- **Not cleared, attribution met, ranking short** - the stencil still carries information at 4x but
  budget is not the separator; the registration instrument then decides whether alignment or the
  read-out is the binding constraint, and its verdict sets the next ranked step.
- **Not cleared, attribution below floor** - the stencil's advantage does not survive the budget
  scale; the form is closed and the registration instrument runs as the declared fallback.
- **Not admissible** - the constraint itself is the measured result; nothing is relaxed.

## Limits, stated before the measurement

- Synthetic, training-fit population: these are the stream's own drawings, which the model trained
  on. Nothing here is a real-screenshot, generalization, calibration or complete-state claim.
- One budget point: 4x is a single declared scale, not a curve; a null result closes this budget,
  not every budget.
- The registration instrument, if run, is an attribution diagnostic under oracle pairing, not a
  model, an accuracy estimate or a fix.

## Measured: scaling is refused, and registration is the binding constraint

The declaration above is unchanged from commit `1ce41e6`; it was committed before any profile,
study, sweep or evidence directory for this instrument existed. Everything below is re-derived by
the curator from the probe's own cells (`scripts/record_scaled_lever.py`,
`scripts/record_registration_attribution.py`) and published in
`docs/evidence/m3-occupancy-scaled-r1/scope.json` and
`docs/evidence/m3-occupancy-registration-r1/scope.json`.

### The 4x budget did not carry the stencil's advantage

Both arms ran 256 steps from `runs/m3-percell-balance-r1` (`9ae1fef9f649...`) under one captured
catalogue (`c8fba7ab`), 1,152 MiB and 1,800 s, and both finished well inside the envelope:

| arm | best swept F1 (threshold) | F1 @ 0.0 | AUC | peak commit | wall |
|---|---|---|---|---|---|
| lever (2x2 stencil) | 0.3925 (0.60) | 0.3791 | 0.8006 | 1,065,533,440 B | 396.1 s |
| control (center sample) | 0.4237 (-0.30) | 0.4146 | 0.8255 | 1,067,073,536 B | 395.1 s |

At 0.0 the lever's counts are 320 true / 701 false positive / 347 false negative; the control's are
333 / 656 / 334, i.e. 0.4146. **Attribution against the matched 256-step control is -0.0312**
(AUC -0.0249, per-cell F1 -0.0355, precision -0.0287, recall -0.0465; whole-ancestry-cluster
interval on the best-F1 delta -0.1082 to +0.0411, 400 draws). The lever fails all three standing
conditions at this budget - per-cell F1 0.3791 < 0.70, ranking `uninformative` (AUC 0.8006 < 0.90),
attribution -0.0312 < +0.05 - so the declared verdict is **`not-cleared`: scaling is refused at the
declared envelope**, and the refusal is not close on any condition.

The shape of the two arms is the more informative result. Between 64 and 256 steps the stencil arm
*lost* ordering quality (AUC 0.8804 to 0.8006, best F1 0.4913 to 0.3925) while the center-sample
control held or improved (AUC 0.8226 to 0.8255, best F1 0.4120 to 0.4237): the extra budget did not
trade a little accuracy for robustness, it inverted the ordering that made the stencil the best arm
at 64 steps. Against the unmatched 64-step stencil record the scaled lever is -0.0988 best F1; that
cross-pass number is reported beside the verdict and is never read by it.

### Registration accounts for the deficit

The declared fallback ran on the scaled lever's frozen archive: 36 forwards, one at a time, 640 MiB,
300 s, peak 565,141,504 B, 20.5 s (`runs/m3-occscale-registration-r2-job.json`, curation
`...-curation-r5-job.json`). The probe's as-is reading over the 28 nonempty boards is F1 0.379147
with counts 320 / 701 / 347 - an exact join with the scaled sweep's published per-cell F1, which is
what makes the two instruments comparable at all.

Correcting placement - pairing each model cell's logit with the label of the target cell that its
centre physically covers - gives, on the 2,986 cells both pairings can score (53.3% of the 5,600):

| pairing | F1 | AUC | counts (TP / FP / FN) |
|---|---|---|---|
| as-is (own index) | 0.4390 | 0.9331 | 207 / 522 / 7 |
| oracle registration | 0.5742 | 0.9702 | 298 / 431 / 11 |

**The registration delta is +0.1352 F1** (whole-ancestry-cluster interval +0.0460 to +0.2833,
median 0.1324, 400 draws, twelve clusters), so the declared rule returns
**`registration_implicated`**: alignment alone is worth more than the entire five-point attribution
floor, and more than twice the deficit the read-out family was chasing. The geometry is not marginal:
the mean absolute cell-centre offset is **2.57 cells**, and 48.3% of model cells map to a different
target cell than their own index (53.3% still land inside the board at all).

### What this pair of receipts does and does not show

- It shows the label family, the read-out family and now the budget lever are all closed on this
  population: the pixels are separable (prior pass, F1 0.978), the read-out form neither clears nor
  scales, and the residual is dominated by where the model puts its cells.
- It does **not** say the occupancy head would read F1 0.574 if the lattice were fixed. The oracle
  pairing excludes the 46.7% of model cells that fall off the board, so it is an alignment-only
  comparison on identical cells, not a full-board counterfactual; the excluded cells are precisely
  the ones a corrected lattice would have to place.
- The registration verdict is a diagnostic of the named alternative suspect, not a model, a repair,
  a new label policy or a calibration, and nothing here is a real-screenshot or generalization claim.

### Re-ranked next step

**Attack predicted-lattice placement itself.** The occupancy read-out already samples its features at
*the model's own predicted lattice* (`model.py`), so its cell indices inherit the 2.57-cell placement
error directly; the ceiling pass showed the pixels support F1 0.978 at the target lattice, and this
pass shows the residual is placement, not form or budget. The next pass should predeclare a
geometry-side lever on the registration/lattice objective (the existing `registration` term is the
natural place) and read it with the same three gates *plus* a declared placement-error measurement,
so the attribution target this pass measured becomes the quantity the next pass has to move. Budget
scaling on this form is closed at 4x; the read-out and label families stay closed.
