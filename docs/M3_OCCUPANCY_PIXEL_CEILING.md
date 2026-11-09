# Occupancy pixel ceiling — is the label hypothesis testable before the lever?

Status: predeclared (2026-11-09) and committed before any canonical probe, curation or evidence
directory for this instrument exists. The question, population, features, fit, control, rule,
envelope and receipts below are fixed; the measured section is appended only after the canonical
receipts exist.

## Where this comes from

The stacking pass ([M3_OCCUPANCY_STACKING.md](M3_OCCUPANCY_STACKING.md)) closed every read-out tap
and ranked one step next: **leave the read-out and target the labels** — an explicit
indeterminate/occluded cell target, scored on determinate cells. Its stated justification was a
*label-overlap signature*: the read-out levers moved ordering (AUC 0.7427 → 0.8244 → 0.8226 →
0.8804) while the swept-F1 deficit stayed at 0.21–0.29, the standing-point operating point is
degenerate (precision 0.25 at recall 0.99), and 0/28 boards are exact at any threshold. That
signature is consistent with label overlap, but it does not measure it, and every M3 label policy
so far has kept unknown cells out of the loss (`targets_for` marks a cell known only when the
observer supports it), so "unknown is not empty" was an untested explanation rather than a
demonstrated defect.

This pass measures the premise directly and **read-only**, before a matched training pass is spent
on it. It changes no label, weight, threshold, gate, population or prior evidence byte.

## The one question

**Can the captured pixels recover the committed determinate occupancy labels at all?**

If a read-out that never sees the model, given only the raster and the target lattice, recovers the
labels at the standing bars, then the labels are not the binding constraint: an explicit
indeterminate target cannot supply separation the pixels already contain, and the label lever must
not be admitted on this evidence. If such a read-out also fails the bars, the labels are not
recoverable from this capture, the indeterminate-target lever is licensed, and its headroom is the
measured gap.

## Instrument

`scripts/pixel_ceiling.py`, declared here once:

- **Features (30 per cell, pixels only).** The cell interior at 30% of the cell's half-extent is
  sampled as a 3 × 3 grid of sRGB means (27 values) plus the per-channel standard deviation over
  the interior (3 values). Nothing else enters the feature: no model latent, no scene layer, no
  occupancy label. The cell is located with the **target** lattice in prepared-image pixels, which
  is an oracle registration, declared as such below.
- **Fit.** L2 logistic regression (ridge 1.0, forty IRLS iterations) on standardized features,
  **leave-one-ancestry-group-out**: eleven groups train, the held-out group's cells are scored.
  Every training fold must contain both classes.
- **Decision.** The fit's own `0.0` log-odds boundary, not a tuned threshold, mirroring the
  replay's `0.0` operating point. The standing 81-point grid on `[-6, 6]` step 0.15 is reported
  beside it for comparability with the model sweep.
- **Control.** The same pipeline on labels permuted **within each ancestry group** (seed
  `20261026`), so the class budget and every image's own occupied count are preserved while the
  spatial mapping is destroyed.
- **Statistics.** Governing: pooled count-based occupied-cell F1 at `0.0` and the **mean
  per-fold AUC**. Descriptive only: pooled AUC, best swept F1. No verdict reads the pooled AUC —
  a leave-one-group-out score can be near-constant within a fold and still order by fold frequency
  across folds, which is an interval-validity defect this pass records as V-066.

`scripts/probe_pixel_ceiling.py` replays the deciding boards from the swept run's frozen archive
(`replay_case`, so the recipe, raster and label identity are asserted), extracts the features, and
writes the probe receipt. `scripts/record_pixel_ceiling.py` re-derives every published number from
that receipt's own cells and refuses before writing on any disagreement.

## Population

The committed **28 nonempty deciding boards** of the joined sweep — 36 training images, 667
occupied and 4,933 empty known cells, twelve ancestry groups — exactly the population of the
widened gates and of the sweep's threshold rows. The 8 empty boards are excluded from the fit and
the gate arithmetic and are reported separately; their false positives are already published by the
sweep and are not re-measured here.

The pass joins `runs/m3-occstack-lever-r2-sweep-r1/result.json`: the probe requires that sweep to
have passed, that its weights equal the run's checkpoint bytes, that its deciding arithmetic is
28/667/4,933, and that its `0.0` row reproduces the label-derived counts (661 true / 6 missed).
The model's own operating and best numbers are **read from that sweep's rows**, not recomputed, so
this pass joins that receipt instead of restating it.

## Declared rule

**The pixel ceiling is `separable` iff both hold:**

1. pooled occupied-cell F1 at `0.0` is `>= 0.70` (the standing per-cell gate), and
2. the mean per-fold AUC is `>= 0.90` (the standing ranking gate).

Otherwise it is **`not-separable`**, with the failing condition named by the gap. The instrument is
**invalid** — recorded as such, not as a measurement — if the label-shuffled control reaches either
bar, because then the pipeline is not measuring label recovery.

Consequence, fixed here:

- **`separable`** → the label-overlap signature is **refuted** as an explanation of the occupancy
  deficit, `indeterminate_target_licensed = false`, and the label lever is not admitted. No
  further label-policy variant is ranked; the open question becomes model-side.
- **`not-separable`** → the labels are not recoverable from this capture, the indeterminate-target
  lever is **licensed** with the measured gap as its headroom, and the resolution/representation
  family reopens only inside its own declared envelope.
- **`instrument-invalid`** → nothing is concluded about labels; the instrument is repaired and
  re-recorded under a fresh number.

## Envelope and receipts

One read-only CPU job, no Torch model and no forward pass: ceiling **640 MiB**, profile wall
**300 s** / **2 work items** (two feature extractions), study wall **600 s** / **36 work items**
(thirty-six replays and feature extractions), curation **2 work items**. One heavy job at a time;
`scripts/run_bounded.py` supplies the kernel-enforced ceiling and the receipt.

Receipts this pass must produce:

1. `runs/m3-occpix-ceiling-r1-profile-r1` + `-job.json` (2 work items), and
2. `runs/m3-occpix-ceiling-r1` + `-job.json` (36 work items), admitted only by that profile;
3. curation `runs/m3-occpix-ceiling-curation-r1-job.json` into
   `docs/evidence/m3-occupancy-pixel-ceiling-r1/scope.json`;
4. the owned family `m3-occpix-*` declared in `record_stream_run.OWNED_RUN_PATTERN`, bound by
   `tests/test_pixel_ceiling.py`.

Any launch that dies to a harness limit leaves its attempt under `runs/attempts/` with a written
note and is re-recorded under a fresh number. No ceiling, bar, population or feature moves.

## Limits, stated before the measurement

- **Oracle registration.** The read-out is given the target lattice, so it measures separability
  *given correct registration*, not the model's end-to-end problem. The model must predict geometry
  too (its geometry error is about 3.3 cells). A separateable ceiling therefore says the targets are
  recoverable from this capture under known geometry; it does not say the model's registration is
  adequate, and it is not an accuracy claim.
- **Synthetic, training-fit population.** These are the stream's own synthetic drawings, which the
  model trained on. Nothing here is an estimate of real-screenshot accuracy, generalization,
  calibration or complete-state performance.
- **Not a model comparison.** The read-out is a ceiling diagnostic, not a candidate model, not a
  release artifact and not a decoder.
- **Exploratory reads are not evidence.** An unofficial read of the same population motivated this
  instrument; nothing published depends on it, and an exploratory read found the very defect this
  predeclaration guards against (an unregularized least-squares score extrapolating far past the
  compared grid, and a pooled AUC that ranked fold frequency). Only the canonical receipts above
  produce numbers, and V-066 records the defect and its instrument.
