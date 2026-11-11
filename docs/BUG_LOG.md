# Defect ledger

Historical tetrguide issues are design
constraints in ROADMAP.md, not fixes claimed by this repository.

Every new fix entry needs symptom, root cause, detection class, instrument and
a named green run. An unrun gate leaves the entry OPEN.

| Detection class | Entries | Standing instrument |
|---|---|---|
| visibility-contract | V-001, V-002, V-005, V-006, V-016 (5) | Paired-raster equivalence required for every hidden-component policy; AGENTS.md, test_layout.py and M3 prepared-source admission tripwire |
| provenance | V-003, V-014, V-018, V-020, V-021, V-022, V-024, V-026, V-027, V-028, V-034, V-035 (12) | Standing artifact-derived identity rule in AGENTS; computed hashes/byte comparison, frozen execution, immutable controller origin, unwrapped callable origin, actual receipt status and checkpoint architecture/config/aggregate tripwire; saved text identities checked against actual wire bytes and JSON roundtrip; profile ancestry must match actual profile recipes; a staged start must be an exact sibling run re-read from source bytes and a staged resume must restate that contract |
| test-instrument | V-004, V-011, V-012, V-017, V-019, V-023, V-025, V-037, V-049, V-057, V-058, V-061, V-069, V-070, V-072, V-092 (16) | Native admission/raw counters; warm backward; standing pre-import CPU-library policy/live child tripwire; observer-validated fixtures before model gates; named queue anchor flags; raw versus normalized evidence-copy tripwire; a fixture must derive archived supervisor identity from the bytes it copies; a new read-only instrument must reproduce a published number from another receipt before its own verdict is written (cross-receipt join); a matched start compares the function, not the objective's value, so the objective's own fields are stripped before step-0 probes are compared; a convention shared by a read and its provenance join lives in one helper, because a fix applied to one copy leaves the defect standing |
| resource-identity | V-007 (1) | Exact named Job plus kernel membership/readback |
| corpus-admission | V-008, V-009 (2) | Malformed/duplicate records and off-image known-cell fixtures |
| resource-pressure | V-010 (1) | OS commit admission and pressure/failure process-tree tripwires |
| resource-profile | V-013, V-039, V-040, V-043, V-045, V-046, V-062 (7) | Shape-diverse native memory trajectory, bounded native cache and resumable finite runs; a declared read-out mode must not carry trainable heads it never uses; a page-file host's steady-state commit charge is not this job's risk. Standing instrument: `PressureDecisionTests` plus the synthetic-exhaustion cases in `test_pressure_guard.py`. |
| windows-persistence | V-015 (1) | Bounded sharing-error retry, prior-pointer preservation and fail-closed launch |
| objective-provenance | V-051, V-055 (2) | The probe and curator bind the objective from `arguments` and reproduce saved losses; a regression test asserts the objective's branches are charged and refuse a mismatched join |
| launch-ownership | V-029, V-030, V-050, V-093 (4) | Preflight output, receipt and both log paths before detached launch; verify the child executable path resolves before arming the Job; a supervisor killed by its caller leaves a `running` receipt, so the retry preserves it and the partial output before re-running; one block per seed holds a pid lock, so a live holder refuses a second chain against the same outputs and a dead holder is replaced rather than trusted |
| evidence-reporting | V-032, V-033, V-036, V-038, V-041, V-042, V-044, V-047, V-048, V-053, V-054, V-065, V-068 (13) | Evidence-folder audit instrument that re-derives every published number from the committed bytes and fails on any disagreement; a correctness count is reported next to the classes the head actually emits, so a constant predictor cannot read as identity reading; the ordinary replay completes deterministic development evidence to the canonical group set and the curator re-derives that extension; the replay scores the objective the run actually recorded and the curator binds that identity; an evidence directory is an atomic record keyed by name, so a frozen run's directory cannot gain files; a re-derivation reconstructs labels in the observer's own cell order from the committed bytes instead of inferring an order from counts |
| interval-validity | V-056, V-066, V-091, V-094 (4) | Measure the ICC and design effect of the deciding cells and report a whole-cluster resampling interval beside every cell-level interval, so a nested population cannot be reported with an independent-sample width; a leave-one-group-out score's ranking statistic is the mean per-fold AUC, never a pooled AUC that can order fold frequency, and a continuous score is fitted with a bounded link before any fixed threshold grid is read; an attribution is never read at a fixed threshold the lever itself moves, but at a boundary chosen on the other groups and reported with a paired whole-group interval; before a block is declared, its minimum detectable effect is computed from the committed clusters (bootstrap and delta-method agreeing) and stated beside its floor, so an underpowered block cannot report "no effect" where it could only not resolve one |
| tooling-harness | V-060, V-064, V-067 (3) | Long authoring payloads go through a file, not a shell heredoc: the command transport caps a single command's length and a payload past it is cut mid-string, so the patch script is written to disk, run, and removed; launch argv is read from the receipt as data, script path preserved and interpreter quoted, never rebuilt by slicing and `eval`; the supervisor consumes a run name only after argv resolves, and a refused launch is preserved under `runs/attempts/` with a note and re-recorded under a fresh number |
| tap-composition | V-063 (1) | A composition of two declared residual taps is tested both-on, not only one at a time: the stack is their sum, it differs from each half once trained, zeroing one projection reproduces that half's arm exactly, and the single-tap arms keep the published composition |

## V-091 — CLOSED: the standing occupancy attribution was read at a threshold the levers move

- Symptom: fourteen committed occupancy pairs were credited or refused by per-cell F1 at logit
  threshold `0.0`, while the arms' own receipts report best swept thresholds from `-1.35` to `+4.35`,
  and the arm the standing read scored worst at `0.0` (the 1.5x fit budget, `-0.134`) is the best
  arm in the table once the boundary is transferable (`+0.149`).
- Root cause: F1 at a fixed threshold is a function of ranking *and* of the head's logit offset, and
  that offset is unconstrained - a 64-step continuation moves the occupied-minus-empty median offset
  by more than two logit units (`+4.16` on the shared control, `+1.88` on the support-anchor arm).
  The verdicts read the offset as precision.
- Change: no verdict, gate, threshold, population or model byte changes. A read-only audit
  (`scripts/operating_point.py`) recomputes `f1_at_zero`, `auc`, a leave-one-group-out `cv_f1` and the
  logit offset from every committed sweep receipt, and classifies each declared pair by a rule fixed
  before its receipt (`docs/M3_OPERATING_POINT.md`).
- Detection class: interval-validity.
- Instrument: the leave-one-group-out boundary plus the paired whole-group interval, pinned by
  `tests/test_operating_point.py` (shift invariance, no held-out leakage, the state table and the
  population/join guards). Standing under the three-instance rule.
- Named green run: `runs/m3-opaudit-r5-job.json` (passed, 88.7 s, peak 46,661,632 B of 128 MiB)
  writing `docs/evidence/m3-operating-point-r2/result.json`; the full suite is green.

## V-092 — CLOSED: the audit's read and its provenance join held two copies of one convention

- Symptom: two consecutive audit attempts refused with "sweep receipt does not reproduce its own
  published read" although every receipt was intact.
- Root cause: the sweep's F1 curve is over the 28 nonempty boards while the join pooled all 36, and
  the fix that restricted the read was not applied to the join in the same function - so the defect
  survived its own fix.
- Detection class: test-instrument.
- Instrument: one `nonempty` helper feeds the population guard, the read and the join, and a
  regression test asserts that an all-empty board contributes no cells and is counted separately.
- Named green run: `runs/m3-opaudit-r5-job.json`; the refused attempts are preserved with notes under
  `runs/attempts/m3-opaudit-r1-refused/`, `-r2-refused/` and `-r3-refused/`.

## V-094 — CLOSED: every lever read was judged against a floor the design could not resolve

- Symptom: the campaign retired the compute closure on point deltas, but its `+0.05` attribution
  floor is not resolvable at the design it ran. The block's detectable floor is `0.083` (budget) and
  `0.126` (schedule) at 80% power, above both observed magnitudes (`-0.0796`, `+0.0213`), and at a
  true `+0.05` the block had 39% and 20% power. The audit's `+0.149` sat inside its own design's
  floor (`0.196`, power `0.567`), so the reversal that motivated the re-read was never resolvable.
- Root cause: the floor was declared as a threshold on the point estimate, never checked against the
  design's standard error, so "no effect" and "could not resolve" produced the same verdict string.
- Detection class: interval-validity.
- Instrument: `scripts/power_analysis.py` recomputes the paired standard error two ways (whole-cluster
  bootstrap and a delta-method linearisation that agrees to 1.004/0.975), measures the design effect
  against boards and cells, and reports the minimum detectable effect, the power at the declared
  floor and the seed count each target would need, before any block is read.
- Named green run: `runs/m3-power-r2-job.json` writing `docs/evidence/m3-power-mde-r2/result.json`;
  `tests/test_power_analysis.py` and the full suite are green.

## V-093 — CLOSED: two block chains ran against one seed because nothing knew the other was alive

- Symptom: after a host runner restart, the detached seed-`20261102` chain outlived the tool call that
  had launched it and kept training while a replacement chain was started on the same seed. The
  replacement reused the two finished arms, then failed `PermissionError [WinError 5]` renaming the
  still-live arm directory; later the surviving chain's seed-3 step collided with profile directories
  the replacement had created and exited 1.
- Root cause: the driver's only reuse guard was the existence of outputs and receipts, which a
  second process cannot observe being written. Liveness of the other chain was never consulted, so
  "one heavy job at a time" was a human invariant rather than an enforced one.
- Detection class: launch-ownership.
- Instrument: `acquire_lock` in `scripts/run_compute_replication.py` writes a per-seed lock holding
  the holder's pid; a live holder refuses the second block before it touches any output, and a dead
  holder is replaced rather than trusted, so a crashed block stays resumable.
- Named green run: seed `20261103` then ran alone as `runs/repl-parent-s20261103-r2-job.json`
  (passed, 1,070.4 s, peak 1,037,963,264 B of 2,560 MiB), reusing exactly its three verified
  profiles; the refused and collided receipts are preserved with a note under
  `runs/attempts/m3-repl-restart/`.

## V-068 — CLOSED: the curator rebuilt labels in an assumed order instead of the committed order

**Symptom.** The second canonical curation exited 1 with "the probe's stored statistics do not
re-derive from its own cells" and wrote no evidence directory.

**Root cause.** The re-derivation inferred each board's label vector from counts alone as
occupied-first (`[1]*occupied + [0]*rest`), so a logit at a cell index was scored against a different
cell's label: the observer emits interleaved cells. Counts matched, so the guard passed and the
statistics simply disagreed.

**Change.** The re-derivation takes the committed sweep's own label array, keyed by `sample_id`,
and refuses unless its length and occupied count match the probe's stored case.

**Detection class.** evidence-reporting.

**Instrument.** The curator's arithmetic join against the swept bytes, plus a regression fixture
that pins an interleaved board and shows a counts-identical reorder scoring differently (1/1/1
instead of 2/0/0) and a wrong-length vector being refused.

**Named green run.** `runs/m3-occpix-ceiling-curation-r3-job.json` (passed, 2 work items) writing
`docs/evidence/m3-occupancy-pixel-ceiling-r1/scope.json`; the refused receipt and note are preserved
at `runs/attempts/m3-occpix-curation-refused-r2/`.

## V-070 — CLOSED: a curator guard required the matched arms to differ

- Symptom: the first geometry-lever curation refused with "the arms must not share one saved initial
  tensor file", although both arms were declared to start from the same published checkpoint. No
  evidence directory was written.
- Root cause: the guard was copied from the read-out curators, where a lever added parameters and
  therefore necessarily changed the saved initial tensors. This lever adds no parameter, so identity
  of the initial state is the *matched* condition, and the inherited inequality asserted the
  opposite.
- Detection class: test-instrument.
- Instrument: the guard now asserts equality of `initial_weights_sha256` between the two arms (and
  each arm's origin is separately bound to the published checkpoint bytes), so an unmatched start
  still fails closed. `tests/test_geometry_lever.py` binds the admission join and the flag's
  two-value declaration; the curator's join is unchanged by this defect.
- Green gate: `runs/m3-occgeom-curation-r2-job.json`, which publishes the full deciding population
  (28 boards, 5,600 known cells, 667 occupied) and verdict `not-cleared` with attribution +0.078064
  and placement improvement -0.199086. The refused receipt and its log are preserved under
  `runs/attempts/` with a note and are not evidence.

## V-069 — CLOSED: the registration summariser scored the wrong population

- Symptom: the first curated registration attribution read as-is per-cell F1 0.365088 with 4,218
  matched cells, while the scaled sweep's published per-cell F1 for the same weights and the same
  boards is 0.379147. Two read-only instruments disagreed on a number both derive from one
  checkpoint.
- Root cause: `summarize` pooled all 36 probe boards, including the 8 empty boards the standing
  deciding convention excludes (`occupancy_sweep.evaluate` scores nonempty boards), so 1,600
  all-negative cells entered both curves and shrank the matched subset. The declaration scores the 28
  nonempty deciding boards only.
- Detection class: test-instrument.
- Instrument: `summarize` now selects the nonempty deciding boards, and the curator asserts a
  **cross-receipt join** - the full as-is pairing must reproduce the scaled sweep's published per-cell
  F1 (1e-6, the published precision) and its three operating counts. The join then caught three more
  curator defects before any evidence was written: a superseded probe path, a 1e-9 tolerance against
  a 6-dp published value, and a three-key-versus-four-key count comparison. Each failing launch wrote
  no evidence directory and is preserved under `runs/attempts/`.
- Green gate: `tests/test_registration_attribution.py` (13 tests, including empty-board exclusion,
  the common-translation invariance and a pure-alignment recovery) and curation
  `runs/m3-occscale-registration-curation-r5-job.json`, which publishes as-is F1 0.379147 with counts
  320/701/347 - identical to the sweep - and verdict `registration_implicated` (+0.135157, interval
  +0.046025 to +0.283310). The superseded probe and the three failed curations are not evidence.

## V-067 — CLOSED: a refused launch consumed the declared run name

**Symptom.** Relaunching the profile after the interpreter path was corrected failed with
`FileExistsError` on `runs/m3-occpix-ceiling-r1-profile-r1-job.json`. No step, feature extraction or
evidence byte had been produced.

**Root cause.** The supervisor writes its durable receipt before it resolves and validates the child
argv, so a launch refused by the V-030 executable check still occupies the declared name.

**Change.** None in the supervisor this pass; the launch discipline is what the instrument names.

**Detection class.** tooling-harness.

**Instrument.** The V-030 check still refuses loudly before any work; the refused receipt is preserved
under `runs/attempts/` with a written note and the canonical runs use fresh numbers, so a refused
name never enters evidence.

**Named green run.** `runs/m3-occpix-ceiling-r1-profile-r2-job.json` (passed, 2 work items,
266,375,168 B) and `runs/m3-occpix-ceiling-r1-r2-job.json` (passed, 36 work items, 266,182,656 B),
both under the declared 640 MiB.

## V-066 — CLOSED: pooled cross-fold AUC and an unbounded score made a read-only comparison lie

**Symptom.** An exploratory read-out on the same deciding population reported a label-shuffled
control at AUC 0.99 and a swept best F1 of 0.94 for a classifier that in fact recovered nothing
(1 true / 33 false positive / 666 missed at its own boundary).

**Root cause.** Two independent faults. (a) A leave-one-group-out score that is near-constant within
a fold still orders folds by their class frequency, so a pooled AUC ranks fold frequency rather than
cells. (b) A least-squares score fitted to 0/1 targets extrapolates far past `[-6, 6]`, so a global
threshold grid no longer samples operating points meaningfully and its "best" point is noise.

**Change.** The instrument fits L2 logistic regression (bounded log-odds), governs on pooled
count-based F1 at the fit's own `0.0` boundary and on the mean per-fold AUC, reports pooled AUC as
descriptive only, and requires the label-shuffled control to collapse under the identical pipeline.

**Detection class.** interval-validity.

**Instrument.** `tests/test_pixel_ceiling.py` pins a fold-constant score at pooled AUC 0.81 with
per-fold AUC 0.5, asserts no verdict reads the pooled AUC, and asserts a control that clears a bar
invalidates the instrument instead of measuring it.

**Named green run.** `runs/m3-occpix-ceiling-r1-r2-job.json` (passed, 36 work items) and
`runs/m3-occpix-ceiling-curation-r3-job.json` into
`docs/evidence/m3-occupancy-pixel-ceiling-r1/scope.json`; 20 instrument tests green.

## V-065 — CLOSED: the new curator read a reference record's catalogue from the wrong block

**Symptom.** `scripts/record_stacking_lever.py` exited 1 after 2.6 s with
`KeyError: 'catalogue_sha256'` and wrote no evidence.

**Root cause.** The reference reader assumed every committed scope record carries the catalogue inside
`matched`; the stencil record carries it on the arm, not in that block. A recorder that reads another
instrument's evidence must tolerate that record's shape instead of assuming its own.

**Change.** The reader takes the catalogue from `matched` when present and otherwise from the arm, and
it validates status, instrument and an arm rather than comparing a path to itself.

**Detection class.** evidence-reporting.

**Instrument.** The failure is loud and early: the recorder raises before `mkdir`, so no evidence
directory is half-written, and the refused receipt is preserved at
`runs/attempts/m3-occstack-curation-r1-failed/`.

**Named green run.** `runs/m3-occstack-curation-r2-job.json` (passed, 4 work items) writing
`docs/evidence/m3-occupancy-stacking-r1/scope.json`.

## V-064 — CLOSED: reconstructed launch argv dropped the script and the interpreter path

**Symptom.** Three launch shells died before any work: a study refused with "a matching
warm-optimizer square resource admission is required", then two sweep shells with `Unknown option:
--run` and `child executable ... was not found on PATH`. No step and no forward ran.

**Root cause.** The commands were rebuilt by slicing a prior receipt's `command` list at `[2:]` and by
`eval`-ing the interpreter path: the first dropped the script path, the second ate its backslashes.
Both are string surgery on a receipt instead of using the recorded argv as data.

**Change.** Launch argv is built as a bash array from the receipt with the script path preserved, and
the interpreter is passed as a quoted argument rather than inside the eval'd string.

**Detection class.** tooling-harness.

**Instrument.** The trainer's own admission gate and `run_bounded.resolve_child_executable` both refuse
before any allocation or forward, so a bad argv costs seconds and produces no evidence; each refusal is
kept with a written note under `runs/attempts/`.

**Named green run.** `runs/m3-occstack-lever-r2-job.json` and `runs/m3-occstack-control-r2-job.json`
(passed, 64 work items each) with both 54-forward widened sweeps.

## V-063 — CLOSED: the context residual overwrote the detail tap instead of summing with it

**Symptom.** With `occupancy_context=1` and `occupancy_detail=1` the detail tap had no effect at all:
the context branch assigned `occupancy_features = cells + block(...)`, discarding the tap's term, so the
two-flag model was bit-identical to the context-only model and the declared stack could not be
measured.

**Root cause.** The detail tap was added in one pass and the context residual in a later one, each
written as an assignment over `cells`; nothing exercised both together because every pass declared one
lever and held the other at `0`, so the overwrite was never on any tested path.

**Change.** The context residual is now added to the running `occupancy_features`, so the declared
composition is the sum of the two zero-initialized residuals; with either tap alone the function is
unchanged, which is asserted rather than assumed.

**Detection class.** tap-composition.

**Instrument.** `tests/test_occupancy_stack.py` binds the composition: both arms are exactly the
published function at step zero, the stack differs from each tap alone once trained, zeroing one
projection reproduces that tap's arm exactly, and the single-tap arms keep the published composition.

**Named green run.** `runs/m3-occstack-predeclaration-suite-r2-job.json` (609 tests, OK).

**Scope.** No prior receipt is affected: every earlier arm held the other tap at `0`, where the
overwrite does not exist, and this pass's control reproduced the stencil pass's control bit-for-bit
(AUC `0.8226433610683003`, counts 661/2136/6, best F1 `0.41203400915631133` at `0.90`).

## Scope note — the stacking pass found one lever-level defect, and it was closed before measurement

V-063 is a genuine lever-level defect, and it is the reason the stack could not have been measured as
declared: the two-flag model silently ignored the detail tap. It was found while writing the
predeclaration, fixed and bound by tests **before any profile, study or sweep ran**, so no measured
number in this pass or any earlier pass rests on the broken composition. V-064 and V-065 are harness
defects in my own launch and curation plumbing, both loud and preserved with notes. No run died inside
an envelope, no ceiling, gate, population or floor moved, and no run was repeated to pass.

## V-062 — CLOSED: the resolution lever's memory estimate counted one map and understated 3.9x

**Symptom.** The predeclared admission profile for the stride-1 resolution lever died at its declared
1,152 MiB ceiling (peak 1,231,880,192 B against 1,207,959,552 B, exit 3221225477, 46.0 s) even though
`docs/M3_OCCUPANCY_RESOLUTION.md` said the branch adds "about 50 MB of activations at capture side".

**Root cause.** The estimate counted one 48 x 512 x 512 feature map (50,331,648 B) and omitted the
three other full-resolution maps autograd retains for a trained stride-1 convolution (output, grad
output, grad input) plus the masked image, so the census is 204,472,320 B and the measured increment
over the same-session stride-2 control is 197,566,464 B - low by 147,234,816 B, 3.93x.

**Detection class.** resource-profile.

**Instrument.** A lever predeclaration now states its branch cost as a tensor census of retained
full-resolution maps rather than one activation, and
`tests/test_occupancy_resolution.py::test_the_declaration_states_the_measured_increment` fails if the
declared figures, the census or the exact bytes stop matching
`docs/evidence/m3-occupancy-resolution-r1/refusal.json` and the two profile receipts. The two-step
admission profile caught the error before any study launched, so no step was trained against a bad
envelope and the refusal cost 46 s rather than a wasted 64-step run.

**Named green run.** `runs/m3-occres-refusal-r1-job.json` (recorder, 4 work items) with
`runs/m3-occres-predeclaration-suite-r1-job.json` (590 tests, 102.6 s, peak 525,856,768 B).

## V-061 — CLOSED: the sweep work-item arithmetic is declared and asserted, not remembered

- Symptom: V-057's refusal - a receipt declaring 36 work items while the sweep performed 54 forwards -
  was a live risk on every verbatim-continued pass, because the launch arithmetic lived in per-pass
  prose and no mechanical check joined the declaration to the receipt.
- Root cause: the predeclaration named a forward count, but nothing bound the launch, the receipt and
  the curator to that number together, so each new pass could reproduce the same off-by-18.
- Detection class: test-instrument (declaration/launch join) - twelfth entry in the class, on the
  standing instrument V-057 named.
- Instrument: this pass declares **54** forwards in `docs/M3_OCCUPANCY_STENCIL.md` before any launch,
  launches 54, and the curator refuses unless `job.work_items == result.forwards == 54`, the deciding
  population is exactly 36 images / 28 nonempty boards / 12 ancestry clusters, and the sweep evaluated
  12 groups. `tests/test_detail_stencil.py::test_the_population_and_work_arithmetic_are_the_declared_ones`
  re-derives the same arithmetic from the committed receipts, and the curation asserted 36 cases / 28
  boards / 667 occupied / 4,933 empty / 12 clusters before writing its record.
- Green run: `runs/m3-occdstencil-curation-r1-job.json` (passed, 4 work items) and
  `runs/m3-occdstencil-fullsuite-r1-job.json` (588 tests, OK). No ceiling, gate, population or floor
  moved, and no launch was repeated to pass.

## V-060 — CLOSED: a curation patch exceeded the shell command cap and was truncated

- Symptom: the doc/test patch for this pass was submitted as a single ~15 KB bash heredoc; bash
  reported `here-document at line 1 delimited by end-of-file (wanted PYEOF)` and the child Python died
  with `SyntaxError: unterminated triple-quoted string literal`, so no edit landed and the curation
  stopped before writing any document text.
- Root cause: the command transport caps a single command's length and cuts a payload past it
  mid-string, leaving only the shell's truncated-input warning; the heredoc form put the whole payload
  in the command.
- Detection class: tooling-harness (first entry in this class): a harness limit that truncates an
  authoring payload instead of failing at its boundary.
- Instrument: the workaround is structural rather than per-incident - the payload is written to a file
  with the file-writing tool, executed, and removed (`.patch_stencil.py`, deleted after it reported
  every target it patched); the script asserts each marker's uniqueness before rewriting, and its
  effect was checked by the following greps and `tests/test_detail_stencil.py` (18 tests, OK). Long
  authoring payloads go through a file from now on.
- Green run: `runs/m3-occdstencil-fullsuite-r1-job.json` (588 tests, OK) on the commit carrying the
  patched documents.

## Scope note — the stencil pass produced no lever-level defect

Both entries above are a harness limit and the closure of a prior arithmetic defect; neither is a
defect in the occupancy stencil instrument or its measurement. The pass admitted both arms inside the
declared 1,152 MiB and 64 steps, performed the declared 54 forwards per sweep, refused nothing, moved
no ceiling, gate, population or floor, and needed no re-run to pass. That is recorded deliberately:
the absence of a lever-level entry means "nothing broke in the lever", not "nobody looked".

## V-059 — CLOSED: four documents quoted the AUC gap with a hand-computed error

- Symptom: the neighborhood result was published with "AUC 0.176 short of 0.90" in
  `M3_OCCUPANCY_REPRESENTATION.md`, `M3_OCCUPANCY_RANKING.md`, `ROADMAP.md` and
  `CONTINUATION_STATE.md`, while the evidence gives AUC 0.8244 and therefore a gap of 0.0756 ->
  **0.076**. A 0.10 error, in the direction that overstates how far the lever was from the bar.
- Root cause: the gap was subtracted in prose at authoring time. The curators publish `f1_gap` and
  `best_gap` for F1 but no AUC gap, and no test bound the documents' numeric qualifiers to the
  evidence, so the error survived two commits.
- Detection class: published-number (first entry in this class): a prose qualifier derived from a
  measured value without a mechanical check.
- Instrument: `tests/test_precision_lever.py::test_the_published_auc_gap_prose_matches_the_evidence`
  re-computes both AUC gaps and the best-F1 gap from the committed evidence, asserts the precision
  document quotes them at three decimals, and refuses the stale `0.176` string in all four
  documents. The standing rule for this class is that a derived number in prose must have a test
  that recomputes it.
- Green run: `runs/m3-occdetail-fullsuite-r1-job.json` (full contract suite on the corrective tree,
  `OK`).

## V-058 — CLOSED: the paired bootstrap resampled every split instead of the declared deciding one

- Symptom: the first curation of `m3-occupancy-representation-r1` wrote a block labelled
  `"split": "training"` with 36 images but 42 nonempty boards, and a paired bootstrap over
  **18** ancestry clusters, while the predeclaration fixes the deciding population at 28
  nonempty boards and twelve clusters.
- Root cause: the curator took `result["cases"]`, which holds all 54 forwards (36 deciding +
  18 development), so the nonempty-board count and the seed-token clustering mixed
  `development-*` clusters into a training-scope block. The gate numbers were unaffected —
  they come from the sweep's own `deciding` block — but the published population and the
  interval's resampling were not the declared ones.
- Detection class: test-instrument (declaration scope); the numbers looked plausible, which
  is exactly why the scope has to be asserted rather than assumed.
- Instrument: the curator now selects `split == "training"`, refuses anything but 36 deciding
  cases, asserts exactly twelve clusters, and derives the context count from the data;
  `tests/test_representation_lever.py` re-derives the population, both arms, the bootstrap and
  the verdict from the committed receipts.
- Green run: `runs/m3-occctx-curation-r3-job.json` (passed) into
  `docs/evidence/m3-occupancy-representation-r1/scope.json`; the all-split scope is preserved,
  not deleted, at `runs/attempts/evidence-m3-occupancy-representation-r1-allsplit-bootstrap/`.

## V-057 — CLOSED: a sweep receipt declared fewer work items than the forwards it performed

- Symptom: both widened sweeps of the lever/control pair completed and passed with
  `forwards: 54`, while their job receipts declared `work_items: 36`; the curation refused with
  `lever sweep did not pass under the declared 640 MiB`.
- Root cause: the launches declared the deciding image count (36) instead of the total work
  performed (54 = 36 deciding + 18 development), so the receipts could not certify their own
  execution under the repo's declared work-item bound.
- Detection class: test-instrument (declaration/launch join). With V-058 this is the eleventh
  entry in the class; the standing instrument is the curator's declared-bound joins
  (`job.work_items == forwards`, `evaluated_cases == 36`, `clusters == 12`) rather than a
  per-launch reminder.
- Instrument: the curator requires `job.work_items == result.forwards == 54` and
  `training_population.evaluated_cases == 36`; the canonical re-run declared 54.
- Green run: `runs/m3-occctx-{lever,control}-r1-sweep-r1-job.json` (passed, 54 forwards each)
  and `runs/m3-occctx-curation-r3-job.json`; the 36-cap attempts are preserved under
  `runs/attempts/m3-occctx-*-r1-sweep-r1-workcap36*`.

## V-056 — CLOSED: the published 95% intervals treated nested cells as independent

**Symptom.** Every interval printed beside a gate number in the widened-population record is a
cell-count interval: AUC `0.763959-0.806412`, precision `0.254831-0.308774` at the replay's
operating point, precision `0.232878-0.272491` at the swept best, hold exactness
`0.298429-0.620141`. All four are computed as if the 667 occupied and 4,933 empty cells were
independent draws, although they are 28 nonempty boards of 200 cells nested in 12 ancestry
groups. Resampling whole boards and whole ancestry groups widens precision's standard error by
3.80x-8.72x (the square root of design effects 14.44x-76.02x; the measured interval-width
inflation is 6.73x at 0.0 and 8.67x at -0.90) and the AUC interval by 1.68x at the ancestry
level.

**Root cause.** `gate_power` supplies closed-form Hanley-McNeil and Wilson intervals, which
assume independent observations. The deciding population has two nesting levels and matched
views share ancestry - a fact the repo's own scope lines already state - so the widths were an
untested assumption rather than a measured property. The defect is methodological: the point
estimates, thresholds and verdicts were all correct, the *confidence* attached to the precision
figures was overstated.

**Detection class.** interval-validity (first entry). The class is distinct from
evidence-reporting: the published numbers re-derived exactly from the committed bytes, and the
defect was that the derived interval was the wrong kind of interval for the design.

**Instrument.** `scripts/cluster_interval.py` (one-way ICC(1), design effect, effective size,
deterministic whole-cluster percentile bootstrap, design-effect-reduced Wilson) and
`scripts/record_cluster_addendum.py`, which re-derives every published point estimate, asserts
it unchanged, and records the measured clustering in
`docs/evidence/m3-widened-deciding-population-clustering-r1/scope.json`. Measured: ICC 0.000941
(board) / 0.004024 (ancestry) at the operating point, 0.005906 / 0.009233 at the swept best;
design effect 1.19-2.87 and 2.18-5.30, so the instrument has 232-562 effective occupied cells
rather than 667; precision design effect 14.44x-76.02x while recall stays 0.25-1.99. The
over-confidence multiplier is therefore 3.80x-8.72x, derived as the square root of the precision
design effect and checked against the interval-width ratios 6.73x and 8.67x - the first draft's
3.4x-6.7x was wrong at both ends and is corrected in place.

**Named green run.** `runs/m3-widened-clustering-r2-job.json` (passed, 384 MiB declared ceiling,
39,444,480 B peak) with `tests/test_cluster_addendum.py` re-deriving every interval and design
effect from the committed artifacts and `tests/test_cluster_interval.py` covering the estimator
itself. The gate is green: the corrected widths are now quoted beside every cell-level width in the
addendum, the roadmap and this handoff, with every prior number left beside them, and
`tests/test_cluster_addendum.py` re-derives each one from the committed artifacts.

## V-054 — The occupancy sweep pooled the empty population into its deciding curve

**Symptom.** The first sweep attempt's per-threshold rows reported the training zero row as
67 true positive / 268 false positive / 101 false negative (F1 0.2664) although the committed
replay of the same checkpoint reports 67 / 196 / 101 on the deciding nonempty population, and
the pooled F1 depressed every row of the curve.

**Root cause.** `occupancy_sweep.evaluate` accumulated the per-threshold confusion counts over
all saved boards, including the empty population's 400 known cells, while
`exact_empty_boards` and the empty-population false-positive rate were already separated. The
deciding quantity and the reported curve therefore mixed two populations. The curator's
absolute check on the admission directory name was a second, cosmetic join defect: it assumed
`<run>-profile` while the admitted profile run is `<run>-profile-r1`.

**Detection class.** Instrument, caught before curation by a predeclared instrument: the
grid's zero row must reproduce the replay's published locked counts (`reference_join` in
`scripts/record_stream_sweep.py`).

**Instrument.** The reference join plus
`tests/test_sweep_rule.py::test_counts_come_from_the_deciding_population_only` (a mixed
nonempty/empty population must count only the deciding cells and report the empty board's
cells separately), and the curator now derives the admission directory from the sweep
receipt's own `admission.run` and re-validates that profile summary.

**Named green run.** `runs/m3-sweep-occupancy-r1-ranking-r1-job.json` (passed, zero row
67 / 196 / 101 on training and 275 / 352 / 126 on development) with
`runs/m3-sweep-occupancy-tests-r3-job.json` (11 tests, `OK`). The failed attempt and its
receipt are preserved under `runs/attempts/`.

## V-055 — CLOSED: the replay bound the objective and target policy by hand, not from the archive

**Symptom.** The widened-population identity replay of `m3-identity-factor-length-r1` died on
its first saved case with `TypeError: joint_loss() got an unexpected keyword argument
'occupancy_factor'`; an earlier attempt of the same pass died at the first extension row with
`ValueError: joint component outputs/targets must match`.

**Root cause.** One cause, two bindings. `probe_stream_errors.py` passed every objective term
to its frozen archive's `joint_loss` unconditionally, so any run captured before the per-cell
lever (which added `occupancy_factor`/`positive_weight`) could not be replayed at all. And both
extension paths built their target keywords by hand (`{"layer_evidence": ...}`), omitting the
component target policy the saved rows take from `replay_keywords`, so the first extension row
a capacity-64 run ever evaluated produced targets without components while the model emitted
them. The development extension carried the same latent gap; no earlier run had exercised it
at capacity 64.

**Detection class.** objective-provenance (V-051 was the first entry). The replay must bind the
objective a run trained under from that run's own artifacts, never from a fixed call shape.

**Instrument.** `probe_stream_errors.objective_kwargs` binds the recorded terms to the frozen
archive's own `joint_loss` signature and refuses a non-default term the archive cannot execute,
and both extension paths now take their target keywords from `replay_keywords`.
`tests/test_objective_binding.py` holds the binding (defaults omitted, supported terms bound,
unsupported non-default terms refused); the target-keyword fix is exercised by the two capacity-64
extension replays below.

**Named green run.** `runs/m3-percell-widened-r1-job.json` and
`runs/m3-identity-factor-widened-r1-job.json` (each passed, 54 forwards, weights unchanged at
`9ae1fef9...` and `8042c288...`, every saved metric reproduced, extension rows scored), with
`runs/m3-widened-population-tests-r4-job.json`. Both failed attempts and their partial output
directories are preserved under `runs/attempts/` as `-attempt1` and `-attempt2`.

## V-053 - CLOSED: the curator read a log name the launch never wrote

- Symptom: the first curation of `m3-fit-occupancy-r1-errors-r1` died 0.6 s in with
  `FileNotFoundError: runs/m3-fit-occupancy-r1-errors-r1.log`, after the replay itself had
  passed (27 forwards, weights unchanged, every saved metric reproduced). No evidence
  directory survived past the copy loop.
- Root cause: the curator copies three native sidecars out of `runs/` -
  `<name>-job.json`, `<name>.log`, `<name>.stderr.log` - but a bounded launch redirects the
  child's stdout beside the receipt as `<name>-job.log`, the shape the study receipts use.
  The three names were literals inside the copy loop, so nothing bound the reader's names to
  the launch convention and the mismatch could only appear after a replay had already
  passed.
- Detection class: evidence-reporting (V-047's class: canonical receipt and log naming).
- Instrument: `record_stream_errors.NATIVE_SIDECARS` declares the three suffix shapes once
  and the copy loop iterates it;
  `tests/test_component_identity.py::test_the_curator_reads_the_canonical_native_sidecars`
  binds the tuple, and `docs/M3_FIT_GATE.md` now names the launch paths literally so a
  predeclaration states which log names the curator will read.
- Green run: replay `runs/m3-fit-occupancy-r1-errors-r1-job.json` (passed, 27 forwards),
  curation `runs/m3-fit-occupancy-r1-errors-r1-curation.json` (passed) into
  `docs/evidence/m3-fit-occupancy-r1-errors-r1/scope.json`, tests
  `runs/m3-fit-occupancy-tests-r4.json` (101 tests, `OK`). Attempt 1 and its partial
  evidence directory are preserved under `runs/attempts/`.

## V-052 — CLOSED: the admission gate refused a newly declared objective

- Symptom: the first study of `m3-identity-factor-length-r1` was refused 3.1 s in with
  `ValueError: component endpoint term differs from resource admission`, although its
  admission profile `runs/m3-identity-factor-length-r1-profile-r1/summary.json` records
  `component_endpoint_term: length` in `arguments`, `recovery_contract` and `config`, and
  the CLI accepted `--component-endpoint-term length`. The run was gone before the first
  optimizer step; the failed receipt and the partial output directory are retained under
  `runs/attempts/`.
- Root cause: the declared objective existed in four hand-copied places and one of them
  was missed. `validate_admission` compared against the literal `("joint", "separate")`,
  so the newly declared `length` mode could never be admitted by the gate even though the
  driver, the model configuration and the loss all accepted it. A lever cannot be measured
  if its own gate refuses it, and the refusal looked like a mismatch of the run's records
  rather than of the gate.
- Detection class: objective-provenance (V-051 was the first entry): a declared objective
  was not consistently known to the components that accept or replay it.
- Instrument: the endpoint objectives are now declared once, as
  `tetris_vision.components.ENDPOINT_TERMS`, and the admission join is one function,
  `require_endpoint_term(endpoint_term, summary)`. The driver mirrors the tuple without
  importing torch at parser build time (its own archived-import rule), and a test binds the
  two (`test_every_declared_endpoint_objective_is_accepted_by_the_driver_and_the_gate`)
  and exercises every declared mode through both the CLI and the gate, plus a mismatched
  summary and an undeclared mode. This is the third defect in the objective-provenance
  class (V-051, V-052 plus the loss join), so the standing instrument is the shared
  declaration rather than per-site literals.
- Green run: the preserved retry `runs/m3-identity-factor-length-r1-job.json` (passed,
  128/128, 228.0 s, 1,011,888,128 B) and its canonical replay
  `runs/m3-identity-factor-length-r1-errors-r1-job.json` (passed, saved metrics reproduced
  under `component_endpoint_term: length`). The second refusal on the same lever was the
  gate working as declared, not a defect: after the fix the study's source catalogue no
  longer matched the superseded profile, so the profile was re-run against the fixed tree
  and both attempts are preserved under `runs/attempts/`.

## V-051 — OPEN INSTRUMENT: the run ledger cannot show which component objective a run trained under

- Symptom: the endpoint-loss lever needed to show that the run actually trained under the
  factorized objective, and no artifact in the run says so directly. `curve.jsonl` records
  only the `joint_loss` term keys — geometry, occupancy, semantic, visibility,
  registration, layers, layer_visibility, pose, components, layer_cells — so the
  queue-token, endpoint and hold-piece terms are folded into one `components` number and
  the evidence cannot distinguish a joint objective from a factorized one by its shape.
- Root cause: `joint_loss` keeps only `component_loss(...)[0]`, discarding the per-term
  dict it already computes, and the training loop logs the outer terms only. Nothing is
  wrong with the numbers; the record simply has no slot for the inner terms.
- Detection class: objective-provenance. It was detected only because a predeclared lever
  promised a loss-side demonstration and the natural place for it (the run's own ledger)
  was empty.
- Instrument now in place: the probe and curator read `arguments.component_endpoint_term`
  and replay/report the objective the run trained with, the probe reproduces every saved
  per-case loss exactly (joint and factorized objectives give different scalars, so a
  mismatched join fails loudly), and
  `tests/test_component_identity.py::test_the_separate_endpoint_term_charges_a_constant_piece_class`
  asserts the separate branch is charged and refuses both mismatched joins. The ledger
  format itself is deliberately unchanged: `docs/experiments/compare_layer_studies.py`
  iterates the keys of the last curve row, so nesting the component terms would change
  that analysis. The gap is therefore closed by provenance, not by a new ledger field.
- Green run: replay `runs/m3-identity-factor-endpoint-r1-errors-r1-job.json` (passed, 27
  forwards, saved metrics reproduced, `component_endpoint_term: separate`) and curation
  `runs/m3-identity-factor-endpoint-r1-errors-r1-curation.json` (passed) into
  `docs/evidence/m3-identity-factor-endpoint-r1-errors-r1/scope.json`.

## V-050 — CLOSED: a killed supervisor leaves a `running` receipt that blocks its retry

- Symptom: the replay of `m3-identity-factor-positional-r1` was launched in the same
  shell command as its study; the caller's 295 s cap killed the process tree while the
  replay was running. The receipt stayed at `status: running` with no `exit_code`, the
  partial `runs/m3-identity-factor-positional-r1-errors-r1` directory existed with no
  `result.json`, and the natural retry could not use either canonical path.
- Root cause: `run_bounded.py` writes `admitting` then `running` and only finalises the
  receipt on its own exit, and the probe creates its output directory exclusively. A
  kill from outside the supervisor (here the shell cap, not the Job) therefore leaves an
  unresolvable receipt and an unusable output path, with no message saying so.
- Detection class: launch-ownership; the supervisor did not fail a gate, its caller did,
  and the record showed a run that never terminated.
- Instrument: the retry procedure is now explicit and was used here — preserve the
  `running` receipt beside the canonical name (`-job-killed.json` / `-job-attempt1`),
  move the partial output directory under `runs/attempts/`, then re-run under the
  canonical names. As with V-047, no automatic check resolves a non-terminal receipt
  yet; the guard is the documented procedure, not a code gate.
- Green run: canonical replay receipt
  `runs/m3-identity-factor-positional-r1-errors-r1-job.json` (passed, 27 forwards,
  15.8 s, weights unchanged, saved metrics reproduced) and curation
  `runs/m3-identity-factor-positional-r1-errors-r1-curation.json` (passed) into
  `docs/evidence/m3-identity-factor-positional-r1-errors-r1/scope.json`. The killed
  receipt and partial directory are retained, not deleted or re-scored.

## V-049 — CLOSED: the discrimination tally rejected an endpoint prediction

- Symptom: the independent replay of `m3-identity-factor-prototype-r1` exited 1 with
  `IndexError: tuple index out of range` inside `_predicted_classes`, after the study
  itself had completed 128/128 steps and its receipt had passed.
- Root cause: V-048's tally was written with `PIECES` (seven names) as the whole
  vocabulary, but the queue-token head has eight classes — the seven pieces plus the
  observed `EOS` token. A supervised piece position predicted as `EOS` therefore
  indexed past the tuple. None of the four frozen runs ever emitted `EOS` at a piece
  position, so the instrument was never exercised on that path; the prototype run
  emits `EOS` at every piece position and hit it immediately.
- Detection class: test-instrument; the diagnostic crashed on a legal head output and
  would have been reported as a broken run rather than as the measurement it is.
- Instrument: `QUEUE_CLASSES = PIECES + ("EOS",)` is the declared queue identity
  vocabulary, the tally refuses any class outside it explicitly, and
  `test_a_piece_position_predicted_as_eos_is_tallied_not_rejected` covers the case.
- Green run: `runs/m3-identity-prototype-tests-r3.json` (19 tests, `OK`, 512 MiB) and
  the canonical replay receipt
  `runs/m3-identity-factor-prototype-r1-errors-r1-job.json` (passed, 27 forwards,
  weights unchanged, saved metrics reproduced). The corrected reading is the NEGATIVE
  result in `docs/M3_IDENTITY_PROTOTYPES.md`; no frozen receipt or gate changed.

## V-048 — CLOSED: identity exactness was recorded without a discrimination term

- Symptom: every M3 identity report quoted exact counts only (queue tokens 0/54 ->
  3/54 -> 9/54 -> 12/54, hold 0/12 -> 3/12) and every pass read them as learning
  progress, while the fit gate never moved and no report asked which classes the heads
  emit. On the frozen runs the hold-piece head emits one class for every supervised
  case, and its exact count is exactly that class's frequency in the targets (0/9
  training, 3/12 development for `T`); the training queue-token head is provably a
  single constant on three of four runs (`S`, `S`, `I`, matching 3/54, 3/54, 12/54).
- Root cause: the diagnostic schema saved `known`/`correct`/`error_indices` and a
  single predicted hold class, but never a predicted-class distribution, so a
  degenerate head and a reading head produced the same report. The reasoning in the
  lever records then attributed a class-frequency change to identity learning.
- Detection class: evidence-reporting; extends V-032/V-044 because the published
  number was arithmetically right and still did not mean what the record said.
- Instrument: `scripts/m3_error_analysis.py` reports
  `queue_predicted_classes`, `queue_distinct_predicted_classes`,
  `queue_constant_predictor`, `hold_predicted_piece` per case and the aggregated
  `queue`/`hold` `predicted_classes`, `distinct_predicted_classes`,
  `constant_predictor` per split, so a constant head is visible beside its exact
  count; `docs/experiments/audit_identity_discrimination.py` proves the property on
  the frozen runs from saved error indices and audited labels.
- Green run: audit `runs/m3-identity-discrimination-r2.json` (passed, 4 work items,
  0.23 s, 35.7 MB) and curation `runs/m3-identity-discrimination-r2-curation.json`
  (passed) into `docs/evidence/m3-identity-discrimination-r2/scope.json`; instrument
  tests `runs/m3-identity-discrimination-tests-r1-job-r3.json` (19 tests, `OK`).
  The corrected reading is `docs/M3_IDENTITY_DISCRIMINATION.md`; no frozen receipt,
  verdict or gate was rewritten, and the fit gate stays FALSE.

## V-047 — CLOSED: retry receipts were invisible to the canonical-receipt readers

- Symptom: the first independent replay of the representation study exited 1 with
  `complete native run required` although that run's summary was `complete` with
  128/128 steps and its supervisor receipt said `passed`; the following curation then
  failed on a missing `result.json`.
- Root cause: `probe_stream_errors.py` and `record_stream_errors.py` read the job
  receipt from the fixed path `runs/<name>-job.json`, but a refused attempt had already
  written that canonical name while the passing attempt was written to a suffixed
  receipt (`-job-r3`). The readers validated the failed attempt and no message named the
  file they had chosen.
- Detection class: evidence-reporting; extends V-032/V-033/V-042 because the artifact a
  reader resolves by convention disagreed with the receipt that describes the run.
- Instrument: the convention is now explicit — the passing receipt of a run or replay
  occupies the canonical `<name>-job.json`, and every refused attempt is preserved
  beside it as `-attempt<k>` or `-r<k>` with its own log. No automatic check resolves
  receipts yet, so a reader that is pointed at a non-canonical receipt still gets the
  old error rather than a diagnostic naming the alternatives.
- Green run: the passing replay receipt `runs/m3-identity-factor-representation-r1-errors-r1-job-r2.json`
  is copied byte-for-byte to the canonical name, and curation
  `runs/m3-identity-factor-representation-r1-errors-r1-curation-r3.json` (passed) plus
  `docs/evidence/m3-identity-factor-representation-r1-errors-r1/scope.json` postdate it.

## V-046 — CLOSED: the profile admission identity was hard-coded to 512 pixels

- Symptom: fresh 640 admissions were refused in 3.6 s with `complete two-step square
  warm-optimizer resource admission required` although the profile itself completed both
  steps and reloaded exactly.
- Root cause: `validate_admission` compared the profile's observed maximum input pixels
  to the literal `512**2`. That literal is only right for the square probe canvas: the
  probe is a fixed 512x512 layout, so the warm backward's 262,144-pixel input is correct
  at any capture side, but the second profile case is a real 544x640 canvas that arrives
  at 348,160 pixels when the side is 640, so no correct profile could satisfy the check
  at another side.
- Detection class: resource-profile; the guard, not the representation, refused.
- Instrument: the identity is now `512**2 <= observed <= max_side**2`, which is exactly
  the old check when the side is 512, still requires the square probe to have been
  exercised, and still refuses any profile that exceeded the declared budget.
- Green run: `runs/m3-identity-representation-profile-r3-job.json` (passed, 2 steps,
  87.4 s, 944,017,408 B, checkpoint `e1ba82fc...` identical to the pre-fix profile) and
  the admitted study `runs/m3-identity-factor-representation-r1-job.json` (passed,
  128/128 steps) both postdate commit `73f0452`.

## V-045 — CLOSED: the predeclaration declared a ceiling above the engine's cap

- Symptom: the 640 admission was refused in 0.22 s with `use run_bounded.py with <=2 GiB
  aggregate commit and a wall cap` although the predeclaration had declared 2,304 MiB
  for both the admission and the study.
- Root cause: the declared ceiling was derived from expected model growth without
  checking `probe_model.bounded_job_snapshot`'s bound. A bounded job ceiling above 2 GiB
  is refused by the engine whatever the host's commit headroom is, so the predeclared
  number was unadmittable on any host.
- Detection class: resource-profile.
- Instrument: both receipts use 2,048 MiB (the engine maximum, still under
  min(physical/3, 8 GiB)), stated before the admitted profile ran. The capture side was
  not reduced: the fix was to the declaration, not the lever.
- Green run: `runs/m3-identity-representation-profile-r3-job.json` (passed under
  2,048 MiB, peak 944,017,408 B) and the study under the same ceiling (passed, peak
  986,275,840 B).

## V-044 — CLOSED: the predeclared decision rule was not total

- Symptom: `docs/M3_IDENTITY_BUDGET.md` declared three branches (positive if both
  primary counts rise with structure held, negative if neither rises, partial if one
  rises with structure held). The study produced both counts rising while two declared
  structure terms regressed (training queue EOS 8/9 -> 5/9, development hold 3/12 ->
  0/12), a state no branch covered, so the record had no declared mapping from the
  observed outcome to a verdict or a next lever.
- Root cause: the rule was written as three illustrative cases rather than as a map
  over the two independent conditions it actually depends on (primary counts rising,
  structure held).
- Detection class: evidence-reporting.
- Instrument: the rule is now total over that square. A rise with a structural
  regression is declared a miss on the criterion with its own finding, and it does not
  pick the next lever from the trend alone; the budget study is recorded under that
  rule, and the same totality requirement now applies to every new predeclaration.
- Green run: `docs/evidence/m3-identity-factor-budget-r1-errors-r1/scope.json`;
  replay `runs/m3-identity-factor-budget-r1-errors-r1-job.json` (passed, 27 forwards,
  every saved metric reproduced) and curation
  `runs/m3-identity-factor-budget-r1-curation-r1-job.json` (passed) both postdate the
  predeclaration and record the state this rule now maps.

## V-043 — CLOSED: the guard policy document described a retired stop line

- Symptom: `docs/RESOURCE_GUARD.md` still presented "physical memory minus
  min(2 GiB, physical / 8)" as one of the two binding stop points after V-040 removed it
  from admission. Read literally it says a 31.4 GB physical line still parks jobs on this
  page-file host, while the enforced line is 90% of the commit limit (45626134118 B of 50695704576 B).
- Root cause: V-040 changed `pressure_status` and recorded the change in this ledger and
  the continuation state, but the policy document stating the rule was not updated, so
  the record and the enforced policy diverged.
- Detection class: resource-profile.
- Instrument: the document now quotes the expressions the supervisor evaluates
  (`commit_bytes >= commit_limit * 9 // 10` and
  `commit_bytes + reserved_job_bytes >= commit_limit * 9 // 10`, plus the availability
  floor and the min(physical / 3, 8 GiB) ceiling bound), names V-040 as their origin and
  keeps earlier stop points as history. Every receipt records `stop_commit_bytes`, so a
  document that drifts from the code is visible against the numbers it quotes.
- Green run: `docs/evidence/m3-identity-exposure-r4/scope.json`; audit receipt
  `runs/m3-identity-exposure-r4.json` (passed) was admitted at `commit_bytes` 27165560832 against
  `stop_commit_bytes` 45626134118 with `commit_limit_bytes` 50695704576, which is the corrected policy.

## V-042 — CLOSED: the frozen-evidence rule was prose only

- Symptom: V-041 repaired the planted-file instance by relocating the log, but nothing in the
  tree prevented a repeat: the rule that an evidence directory's file set belongs to the
  record existed only in documentation, so the next planted file would pass review and no
  gate would notice.
- Root cause: no machine-readable record of any evidence directory's file set or digests;
  detection depended on an ad-hoc audit command run by hand exactly once.
- Detection class: evidence-reporting.
- Instrument: `docs/frozen-evidence-manifest.json` (176 directories, 3368 files; every directory
  keyed to the commit that recorded it, plus the evidence-root loose files) and
  `tools/check_frozen_evidence.py`, which names the directory and file for an added, removed
  or modified file, a missing directory, an unregistered directory and an unregistered root
  file. Digests are content with CRLF normalized to LF so checkout line endings are not
  drift. Pure `hashlib`; no network, no suite, no receipt scoring, no manifest writing.
- Green run: clean tree exits 0; then an added file, one appended byte in a frozen receipt,
  a moved-away directory and a re-created `m3-identity-factor-fullsuite-r1/` each exit 1
  naming the offending path; every injection reverted and `git status` clean.

## V-041 — CLOSED: the audit filed evidence under a frozen run's receipt name

- Symptom: commit `59b3f01` asserted "no frozen baseline receipt was rewritten" while
  creating `docs/evidence/m3-identity-factor-fullsuite-r1/` — a directory named for the
  frozen `m3-identity-factor-r1` full-suite run — and adding three files to it.
- Root cause: the audit read the freeze rule as "no existing file byte changed". The
  repository keys records by evidence directory name: `record_stream_run.py` refuses an
  existing enclosing evidence directory (line 215) and an existing evidence
  `summary.json` (line 224). An evidence directory is therefore an atomic immutable
  record, its file set is part of that record, and a new directory under a frozen run's
  name reads as a second receipt for that run rather than an extension of the frozen one.
- Detection class: evidence-reporting. The audit's own claim contradicted its own commit;
  the check is `git diff --stat` path scope plus `git ls-tree`/sha256 comparison against
  the commit that recorded each directory.
- Instrument: the preserved log now lives in `docs/evidence/audit-398-baseline-derivation/`
  — a named audit directory, not a frozen receipt — with `audit.json` recording the source
  paths, per-file sha256 digests, the 407 = 398 + 9 derivation and the frozen-directory
  comparison against `5c6b41f`. The frozen-named directory is removed, so its file set
  equals the pre-audit state (it did not exist before `59b3f01`).
- Green run: this turn's comparison of every committed evidence directory against the
  commit that recorded it, recorded in that `audit.json` (all file sets and sha256 digests
  equal), with the suite receipt `docs/evidence/m3-identity-readout-fullsuite-r11/`
  untouched and still HEAD-derived through `supervisor_script_sha256`.

## V-040 — CLOSED: the commit stop line was derived from physical RAM

- Symptom: five bounded full-contract-suite runs were refused `memory_pressure` on a
  host with 6–8 GiB of free physical memory, while the suite job itself peaked at
  343 MiB (and later at 16 MiB). Four of them (`-r1`, `-r2`, `-r3`, `-r5`) were
  refused by the commit line; `-r4` was a genuine `physical_pressure` refusal with
  1.1 GB free and remains a correct refusal. It looked as if the suite could not be
  run at all.
- Root cause: `pressure_status` used
  `stop_commit = min(physical - min(2 GiB, physical//8), commit_limit*9//10)`. With a
  page file — commit limit 50,695,704,576 B against 33,515,835,392 B RAM — the first
  term wins at 31,368,351,744 B, and total host commit charge sits above that in normal
  steady state (30–46 GB observed). Unrelated processes' commit was therefore charged
  against this job's admission, even though the job's own ceiling and physical
  residency were both satisfied (`system_at_admission.commit_bytes` 38.6 GB with
  `available_physical_bytes` 8.9 GB).
- Detection class: resource-profile. The guard's own receipts carried the
  contradiction; the deciding evidence is the pair of counters in each refusal, not
  the job's size.
- Instrument: the commit line is now the exhaustible budget's headroom,
  `commit_limit*9//10`, while the unchanged 2 GiB available-physical floor keeps the
  residency protection that refused genuinely pressured runs (full-suite `-r4` was
  `physical_pressure` with 1.1 GB free and is still a correct refusal). This is the
  class's third entry, so it now carries a standing instrument:
  `tests/test_resources.PressureDecisionTests` pins the admissible/refused matrix over
  OS counters, and `tests/test_pressure_guard.py` moves its synthetic refusals to real
  commit exhaustion (44 GiB against a 48 GiB limit) instead of commit above RAM.
- Green run: `docs/evidence/m3-identity-readout-fullsuite-r11/summary.json` — 407
  tests, exit 0, 72.290 s, peak 545,845,248 B, admitted at 38,347,567,104 B commit
  against a 45,626,134,118 B stop line.

## V-039 — CLOSED: the shared read-out carried dead identity heads

- Symptom: the first capacity-64 admission in the declared shared read-out mode
  refused with `RuntimeError: resource profile requires resident Adam state for every
  trainable parameter`, so the lever had no admission.
- Root cause: the shared mode wired piece identity through the cell-type head but left
  `components.queue_tokens` and `components.hold_piece` registered. They receive no
  gradient, so Adam never creates state for them while they remain in the trainable
  parameter list.
- Detection class: resource-profile; the profile guard, not the model output, caught it.
- Instrument: `share_type_readout` removes the unused heads instead of carrying them,
  and `tests/test_component_identity.py` pins that no unused identity parameter survives
  the switch (no `queue_tokens`/`hold_piece` key in the shared state dict).
- Green run: `runs/m3-stream-identity-shared-profile-r2-job.json` (passed, 262
  trainable tensors with resident warm Adam state) and
  `docs/evidence/m3-stream-identity-shared-r1-errors-r1/scope.json`.

## V-038 — CLOSED: the replay scored the published objective, not the run's

- Symptom: replaying the factor-4 study failed with
  `ValueError: ordinary predictions fail to reproduce saved metrics`, so a completed
  admitted run could not produce evidence at all.
- Root cause: `probe_stream_errors.py` always called `joint_loss` at the component
  layer's default identity factor 1, while the run saved per-case losses under the
  recorded factor 4. The probe never derived the training objective from the run it
  was replaying, so any lever that changes a loss scale was unreplayable by design.
- Detection class: evidence-reporting; the audit instrument, not the model, was wrong.
- Instrument: the probe reads `arguments.component_identity_factor` (default 1),
  replays that objective, records the factor in its result, and the curator refuses a
  report whose factor differs from the run's. The owned-run-name guard also lists the
  predeclared `m3-identity-factor-*` family while still refusing traversal, so a
  declared lever family can be replayed and curated.
- Green run: `runs/m3-identity-factor-r1-errors-r3-job.json` and
  `docs/evidence/m3-identity-factor-r1-errors-r3/scope.json` — passed, weights
  unchanged, every saved per-case field reproduced.

## V-037 — CLOSED: the segment fixture pinned a stale archived supervisor hash

- Symptom: `tests/test_stream_segment.py` produced 11 errors with
  `ValueError: archived supervisor differs from live guard`, so the whole segment
  contract suite was silently red while narrower test selections stayed green.
- Root cause: the fixture fabricates a source archive by copying the live
  `scripts/run_bounded.py`, but took `files["scripts/run_bounded.py"]` from a saved
  evidence manifest that still recorded the pre-V-030 supervisor (`d2fb1478...`).
  `run_stream_segment.segment_plan` compares that archived entry against the live
  guard bytes, so the fixture could only pass while the supervisor never changed.
- Detection class: test-instrument; the V-004/V-011/V-012 pattern of an instrument,
  not the code under test, being wrong.
- Instrument: the fixture derives the archived supervisor entry and the admission
  profile/receipt supervisor identity from the bytes it actually copies, and a
  regression asserts all three equal those bytes and that a substituted archive
  entry is refused with the guard error.
- Green run: `runs/dev-extension-fullsuite-r3-job.json` — the full
  `unittest discover` suite under the bounded runner, all tests passing.

## V-036 — CLOSED: staged development evidence sat on two different denominators

- Symptom: `STAGED_HEAD_REFINEMENT.md` could only publish "no comparison claimed"
  for development, because stage 1 recorded 3 ancestries and 9 images while stage 2
  used the canonical 6 and 18, and no evidence path could put stage 1 on the same
  population.
- Root cause: the development group count is a run argument and the ordinary error
  probe replays exactly the saved population, so an older run could never be
  measured on the contract's canonical set without retraining; nothing required a
  recorded development population to match the canonical one.
- Detection class: evidence-reporting; extends V-032/V-033, because two runs on
  different denominators do not form a comparison.
- Instrument: `probe_stream_errors.py --development-groups` completes the
  deterministic development drawings to the canonical 6 groups as a prefix
  extension, keeps the saved cases as the unchanged-weight reproduction anchor,
  marks extension cases `saved: false` and records their identities;
  `record_stream_errors.verify_extension` re-derives that prefix from the saved base
  seed and refuses truncation, a non-canonical target, a substituted or short
  extension and a saved-set mismatch, and the curator re-derives every extension
  row's decomposition and totals. Regression:
  `tests/test_development_extension.py`.
- Green run: `runs/m2-stream-headrate-control-s3-dev6-errors-r1/result.json` — 50
  forwards, weights `1f20007c...` unchanged, every saved per-case metric
  reproduced, 420,257,792 B / 18.941 s — curated into
  `docs/evidence/m2-stream-headrate-control-s3-dev6-errors-r1/scope.json` (passed,
  base `87488bc`).
- Outcome: on the canonical 18 images both stages reach 4/18 exact boards, 0/36
  exact roles, .901944 cell accuracy and 3.530471-cell geometry, and stage 2's
  development loss is higher (11.8129 vs 11.0885). Stage 1's published 9-image row
  (2/9 boards, 0/18 roles, .908333, 4.168932, 11.3678) is a saved subset of that
  same population, not a competing result.

## V-034 — CLOSED: the staged stage refused its own owned source artifact

- Symptom: the predeclared staged profile aborted before its first step with
  `ValueError: initial weights must come from a different owned run`, although the
  source was `runs/m2-stream-headrate-control-s3` and the output was a different
  run in the same `runs/` directory. The traceback came from the verified archive
  copy at `runs/<name>/code/scripts/train_m2_stream.py`, not from the repository
  file it was edited in.
- Root cause: the ownership test was written against `ROOT/"runs"`, but the
  trainer re-executes from its byte-verified source archive under
  `runs/<name>/code`, so `ROOT` inside the executing driver is that archive copy
  and the project's runs directory can never be named that way. The first form of
  the comparison also mixed a resolved source path with the still-relative output
  path.
- Detection class: provenance.
- Instrument: `owned_initial_source` must see the written run's exact sibling
  (`source.parent == output.parent`) and never the written run itself, with both
  sides resolved;
  `tests/test_staged_refinement.py::test_initial_weight_source_is_anchored_on_the_written_run`
  covers the archive layout, a valid sibling, the written run itself and a tree
  outside the runs directory. The curator still re-reads the origin bytes from the
  source run directory, so the recorded start stays evidence rather than a
  printed digest.
- Green run: `runs/staged-head-refine-profile-r3-job.json` (fresh native
  admission, `passed`, 442,085,376 B) and
  `runs/m2-stream-staged-head-r1-s3-job.json` (`passed`, 433,860,608 B), whose
  origin and weight bytes are verified in
  `docs/evidence/m2-stream-staged-head-r1-s3-errors-r1/scope.json`.

## V-035 — CLOSED: an interrupted staged segment could not restate its origin

- Symptom: resuming the interrupted staged segment was refused at parse time with
  "initial weights define a fresh start; resume an existing run instead", even
  though the run's recovery contract stores `freeze_backbone` and
  `initial_weights_from` and a resumed segment has to restate both to match it.
- Root cause: the flat exclusion between `--init-weights-from` and
  `--resume-from` was written for the non-staged path, where an initial-weights
  start and a resume really are alternatives. Staged refinement legitimately needs
  both, because the frozen mode and the start artifact are part of the recovery
  contract that resume compares.
- Detection class: provenance.
- Instrument: the exclusion now applies only when the stage is not frozen, and
  `tests/test_staged_refinement.py::test_initial_weights_and_resume_are_mutually_exclusive`
  asserts both the refused plain form and the accepted staged form. The contract
  comparison itself is unchanged: a resume whose catalogue, freeze flag or origin
  differs is still refused before any step.
- Green run: `runs/m2-stream-staged-head-r1-s2-job.json` (admitted, restored the
  step-384 snapshot and trained to step 404 before the OS commit guard stopped it
  at 429,887,488 B) and `runs/m2-stream-staged-head-r1-s3-job.json` (same resume
  path, 640 updates to step 1,024, `passed`). Carried limit: a segment cannot be
  resumed after the repository source changes, because the recovery contract pins
  the source catalogue; that is the intended freeze, not a defect.

## V-032 — CLOSED: published arm numbers were not re-derivable from their evidence

- Symptom: `docs/evidence/accumulation-arm-r1` published a `resume delta` that is
  a between-run peak difference, a `Wall` column that disagreed with the
  supervisor receipt quoted in an earlier poll (175.3 s vs 175.9 s, 289.6 s vs
  290.4 s), and reload costs whose bounding receipts were never copied into the
  folder, so none of them could be checked against the committed bytes.
- Root cause: the evidence builder copied summary fields without recording which
  clock and denominator each field used. The trainer's `elapsed_seconds` excludes
  process start and teardown that the receipt includes, and its
  `optimizer_updates_per_second` is cumulative over the arm while the published
  rate was the segment's marginal rate; only the reload result JSON was copied,
  not the receipt that bounded the measurement.
- Detection class: evidence-reporting.
- Instrument: `docs/evidence/accumulation-arm-r1/audit/audit_arm_evidence.py`
  re-derives every per-segment and per-arm number from the summaries, supervisor
  receipts, journals, boundary manifests, profile artifacts and reload receipts
  with plain arithmetic, and fails if any scope field, table cell or prose figure
  disagrees. The scope now records the wall-clock, rate-denominator and peak-delta
  definitions, and the reload receipts are copied in beside their results.
- Green run: plain audit exit 0 with no mismatch
  (`docs/evidence/accumulation-arm-r1/audit/audit.json`, `result: reproduced`,
  empty `checks_failed`), after sealing the corrected numbers; the six bounded
  gates for this revision also passed (`runs/gates-reaudit.out`, 384 MiB and
  192 MiB ceilings). The prose cites no check count, because a count here drifts
  from the instrument every time a check is added.

## V-033 — CLOSED: accumulation arms had no independent replay path

- Symptom: the two completed accumulation arms published only trainer-side probe
  metrics. The independent ordinary replay refused both names before any forward:
  the probe and curator matched `m[23]-stream-[a-z0-9-]{1,40}` only, so
  `accumulation-arm-*-s2` was outside the owned-run policy and no curator could
  accept the artifacts.
- Root cause: the run-name allowlist was written for stream studies and was never
  extended when the accumulation arms were named after their policy.
- Detection class: evidence-reporting; extends the V-032 pattern that a number no
  instrument can re-derive is not evidence.
- Instrument: one shared `require_owned_run` policy in `scripts/record_stream_run.py`
  used by the probe, the curator and the run collector, with a name-policy
  regression covering accepted and refused forms (leading/trailing separators,
  path traversal, uppercase and underscore spellings, over-long names).
- Green run: `docs/evidence/accumulation-arm-balanced-errors-r1/scope.json` and
  `docs/evidence/accumulation-arm-online-errors-r1/scope.json` — 50 ordinary
  forwards per arm reproducing every saved per-case metric exactly with weights
  unchanged.

## V-031 — CLOSED: frozen archive kept repository modules from the parent

- Symptom: the first native accumulation profile launch (640 MiB, work 2) exited
  1 after 2.4 s with `RuntimeError: interleaved order implementation is not
  frozen`; the receipt recorded `failed` and no training step ran.
- Root cause: the trainer's new module-level imports of `accumulation_checkpoint`
  and `accumulation_contract` imported `m2_interleaved_order` from the repository
  in the parent process. `main()` re-executes the driver from the verified archive
  but reset only four modules, so `sys.modules` kept the repository-path functions
  and the frozen-source identity check refused them.
- Detection class: frozen-source identity.
- Instrument: reset every module the driver imports from the archive
  (`accumulation_contract`, `accumulation_journal`, `accumulation_checkpoint`,
  `training_snapshot`, `m2_interleaved_order`) before re-execution, and archive the
  three accumulation modules in `source_snapshot.SCRIPTS` so the frozen driver can
  import them at all. The archive previously omitted them entirely.
- Failed launch: `docs/evidence/accumulation-profile-r1/online-r1-failed-job.json`
  and its stderr, preserved with the green runs.
- Green run: `runs/accumulation-profile-online-r2-job.json` (33.3 s,
  519,929,856 B under 640 MiB, exit 0) and
  `runs/accumulation-profile-balanced-r1-job.json` (29.35 s, 561,786,880 B,
  exit 0), both admitted only through their measured profile numbers.

## V-030 — CLOSED: relative forward-slash child interpreter did not resolve

- Symptom: an admitted bounded launch with every output name preflighted exited
  1 after 0.01 s with `FileNotFoundError(2, ...)`; no test body ran, yet the
  receipt recorded `status: failed`, which reads like a test failure.
- Root cause: `scripts/run_bounded.py` passes the child argv unchanged to
  `subprocess.Popen`, so `CreateProcess` receives the relative forward-slash
  interpreter `.venv/Scripts/python.exe` and cannot resolve it. The supervisor's
  own interpreter is resolved by the calling shell, so the trap appears only in
  the child, and only for that spelling.
- Detection class: launch-ownership.
- Instrument: `resolve_child_executable` in `scripts/run_bounded.py` resolves the
  child executable before the Job is armed and refuses a relative path spelled
  with forward slashes, a missing path and an unresolvable bare name; the receipt
  records `child_executable`, and a refusal is marked `launch_refused` with no
  `child_pid` so it reads as an instrumentation failure rather than a failed gate.
  Absolute paths (including forward-slash absolute) and backslash-relative paths
  still launch.
- Failed gate: `runs/accumulation-checkpoint-pure-r5-job.json`, preserved with its
  logs in `docs/evidence/launch-path-diagnostic-r1/`, alongside three
  one-command controls (relative forward slash fails; relative backslash and
  absolute both pass).
- Green workaround run: `runs/accumulation-checkpoint-pure-r6-job.json` with the
  absolute interpreter — 8 tests, 0.522 s, 37,761,024 B under 128 MiB.
- Green gate: `docs/evidence/v030-launch-preflight-r1/scope.json` pins 68 scoped
  cases (resources, pressure, error evidence, accumulation contract/journal/
  checkpoint/driver, stream driver, type-head optimizer, atomic write) under the
  new supervisor `d9382fa2...` — 15.992 s, 355,688,448 B under 640 MiB, exit 0.
  The one-command controls live in `tests/test_resources.py`: relative forward
  slash refused before any child, backslash-relative and absolute both launch and
  record their resolved `child_executable`. Historical receipts keep
  `d2fb1478...`; the new hash needs its own fresh native admission before further
  long training, and documented copy-paste commands must use an absolute or
  backslash interpreter path.
- Confirmation that the form was the only blocker: the 640 MiB Torch gate that
  r5 was meant to run passed once pressure cleared
  (`runs/accumulation-checkpoint-torch-r2-job.json`, 332,423,168 B, exit 0).

## V-029 — CLOSED: reused launcher logs on an existing run name

- Symptom: an r6 recovery launch failed on its existing exclusive receipt before
  the job ran, but shell redirection had already truncated the old local logs.
- Root cause: the output/receipt/log namespace was not checked before starting
  the hidden launcher; exclusive receipt protection did not cover redirection.
- Detection class: launch-ownership.
- Instrument: require all output, receipt, stdout and stderr paths to be absent
  before launch. Preserve failed-attempt logs and compare restored old logs to
  curated evidence byte-for-byte; never overwrite prior evidence.
- Failed attempt: `docs/evidence/type-head-admission-contracts-r1/type-head-recovery-name-collision.stderr.log`. Historical
  r6 logs restored from `docs/evidence/m2-stream-recovery-gate-r6/`, whose receipt,
  logs and scope were not modified. No candidate process was launched.
- Green run: `runs/m2-stream-recovery-gate-r7-job.json`, preflight of all four
  names, distinct launch and terminal passed receipt, 57.769 seconds, peak
  602,124,288 bytes under 1 GiB. Old logs match curated copies by SHA-256.

## V-028 — CLOSED: feature class counts changed key type during JSON serialization

- Symptom: collector review found its direct recomputation comparison would
  reject a saved feature analysis. Native sixcase r2 reproduces the mismatch:
  five pass, JSON roundtrip equality fails on integer1 versus string"1" keys.
- Root cause: new pure analyzer returned integer keys in target_counts, while
  JSON stores object keys as strings. The new collector correctly compares the
  complete saved and recomputed analysis, exposing this representation mismatch.
  No neural feature observer/collector has run or published an affected result.
- Detection class: provenance; extends the existing actual-wire-identity pattern.
- Instrument: canonical string keys in the analyzer and exact JSON roundtrip
  equality in the grouped/unknown-invariance regression; sibling exclusion,
  role/split separation, missing support and ties remain checked separately.
- Failed gate: docs/evidence/v028-contracts/before-scope.json binds native r2,
  sixcases/onefailure,.238s/32,243,712B under128MiB/30s/work6, original analyzer,
  test and collector bytes, raw receipts/logs/startup;all known PIDs/commands gone.
- Green gate: docs/evidence/v028-contracts/scope.json binds fresh native r3,
  sixcases/sixpasses in.017s(native.235s/32,563,200B),same128MiB/30s/work6/guard,
  actual fixed analyzer/test bytes and raw receipts/logs/startup;all known
  PIDs/commands gone. This is reporting integrity, not a native feature-observer,
  collector, feature-separation or model-quality gate.

## V-027 — CLOSED: interleaved profile counted the ordinary schedule's ancestry

- Symptom: native interleave-profile-r1 completed its two warm square updates,
  but runs/interleave-profile-curation-r1.stderr.log refused the claimed two
  training ancestries; its actual ledger repeats index0/view0 and has one.
- Root cause: distinct_training_groups used the interleaved normal schedule's
  first two positions even when profile_square overrides both positions to0/0.
  The original ordered schedule happened to have one group in its first two
  positions, masking this mismatch before the new policy.
- Detection class: provenance. The existing independent recipe/aggregate
  collector caught it before publishing evidence or admitting study training.
- Instrument: explicit profile flag in ancestry counting, bounded0..2 profile
  cursor, and a regression comparing the reported count with actual repeated
  public-observer recipes for every policy, including zero updates. Extends the
  standing actual-recipe aggregate instrument; unknown is never counted as truth.
- Failed gate: runs/interleave-profile-curation-r1.json, rc1 in1.087s,
  40,968,192B/128MiB; failed profile/collector sources and receipts remain intact.
  No user data or prior run was overwritten. Original native profile itself
  passed65.797s/552,042,496B/640MiB; it is not a curated training admission.
- Green gates: docs/evidence/v027-adapter-contracts/scope.json pins the8-case
  regression on158a534 (2.598s,native3.108s/48,918,528B under128MiB), including
  actual public recipes for all three policies. Separately,
  docs/evidence/m2-stream-interleave-profile-r2/scope.json verifies fresh r2:
  two warm updates25.915s/561,860,608B under640MiB; actual ancestry1 and all43
  training/probe recipes pass independent interleave-profile-curation-r3 in
  .951s/82,956,288B under128MiB. Actual source catalogue68bb40e3..., snapshot
  and weights are bound; all known PIDs/exact commands gone. The prior r2 audit
  attempt was parked before child spawn by OS pressure, not relabeled green;
  its receipt is preserved in interleave-profile-pressure-r1. This closes the
  accounting defect, not the separate recovery/learning/quality gates.

## V-026 — CLOSED: new fixture writer could change hashed newlines on Windows

- Symptom: pre-run review found the new positive-control writer hashed UTF-8
  strings but used text-mode writes for states and the manifest. No affected
  generated set has been published or used for learning.
- Root cause: string identity was treated as file-byte identity despite native
  newline translation. This is specific to the new writer, not an allegation
  about historical archived streaming drivers.
- Detection class: provenance.
- Instrument: explicit UTF-8 byte writes and a full32-image file round-trip
  fixture: actual state/PNG/decoded-pixel/manifest hashes must match records,
  and output reuse must refuse without changing prior bytes. Extends the
  standing computed-byte-identity instrument.
- Green gates: docs/evidence/positive-control-fixture-contracts/scope.json pins
  positive-control-tests-r1.stderr.log,5/5 in1.335s (native1.690s,52,592,640B/
  128MiB), including actual32-image state/PNG/pixel/manifest byte checks and
  reuse refusal. docs/evidence/positive-control-fixtures-r1/scope.json separately
  replays and byte-verifies all32 actual generated images/public states and
  eight-class counts per role. Generation.652s/46,616,576B, replay.879s/
  80,752,640B, both128MiB/30s. Native receipts and executed sources are retained;
  every known process and exact Python command was verified gone. No training
  admission or model-accuracy claim follows.

## V-025 — CLOSED: absence fixture hid part of the active spawn domain

- Symptom: layer-evidence-model-tests-r1 has14/15 passes;the new known-absence
  fixture expected both dense masks known although its default canvas omits the
  top of the six-row spawn area. The existing observer correctly returns unknown.
- Root cause: latent active=None was mistaken for an image-supported absence.
  This is a test-instrument error;the model/observer masks did not need changing.
- Detection class: test-instrument.
- Instrument: assert observed public statuses before target expectations;use a
  validated full-spawn layout for positive absence and retain the cropped-spawn
  unknown as a negative check. This extends the standing validated-fixture rule.
- Failed gate: runs/layer-evidence-model-tests-r1.stderr.log,15 cases,1failure,
  native3.725s,448,126,976B/640MiB;all recorded processes gone.
- Green gate: docs/evidence/layer-evidence-core-contracts/scope.json pins
  corrected layer-evidence-model-tests-r2,15/15 in1.267s,446,099,456B/640MiB.
  It includes actual observer-validated full-spawn absence and cropped-spawn
  unknown. The same scope has45 existing model/component/runtime/capture checks
  (60 selected total on32 hashed sources);failed r1 remains preserved separately.

## V-024 — CLOSED: geometry collector trusted aggregates and incompletely bound native command

- Symptom: review of the historical geometry collector found altered oracle
  aggregate counts would not be recomputed, and a report's native output path
  and full command were not bound. No historical forged report is alleged.
- Root cause: it verified ordinary per-case metrics and source weights but
  treated conditional totals and partial native command parsing as evidence.
- Detection class: provenance.
- Instrument: pure format2 contract recomputes both conditional populations,
  validates unique cases/ancestry, exact command/output/work/memory and actual
  checkpoint identity. Explicit partial admission preserves original status;
  conditional labels never become automatic accuracy. Old scopes remain old.
- Green gates: docs/evidence/geometry-evidence-contracts/scope.json,10 metadata
  cases on eaf5bee (.113s,35,180,544B/128MiB),and actual native
  docs/evidence/m2-stream-accuracy1664-geometry-r1/scope.json:54 paired forwards,
  23.894s,585,072,640B/1GiB;curator.288s,64,712,704B/128MiB. All ordinary
  metrics reproduce saved1664,weights unchanged,actual partial status and both
  conditional populations/aggregates verified. No automatic-quality claim.
- Candidate gate geometry-evidence-tests-r1 rejected the preserved valid768MiB
  historical envelope before positive cases reached aggregate checks. The new
  collector now includes that measured envelope alongside640MiB/1GiB; no live
  cap changed. Raw failing gate remains retained beside the green r2 scope.

## V-023 — CLOSED: new proof checker mistook normalized JSON for a raw copy

- Symptom: runs/m3-controller-recovery-proof-r1.stderr.log rejects actual p1
  evidence before publishing proof; all three native controller runs passed.
- Root cause: the new checker did not read the existing curator's serialization
  contract: its current snapshot manifest is reformatted JSON, while parent
  manifests are raw copies. Resumed-parent evidence mappings were also missing.
- Detection class: test-instrument.
- Instrument: explicit current-manifest value comparison, raw byte comparison
  for every other selected artifact, complete parent mappings; four regression
  cases reject changed state and reformatted raw parent files.
- Green gates: docs/evidence/m3-controller-recovery-r1/test-scope.json pins four
  controller-evidence-tests-r1 cases (.076s,35,454,976B/128MiB). Actual native
  proof-r2 PASS.413s,72,044,544B/128MiB; scope.json verifies all three cases,
  complete parent mappings and unchanged copied content with explicit normalized
  current manifests. Failed proof-r1 remains preserved in the same folder.

## V-022 — CLOSED: controller result could identify edited source instead of its launch source

- Symptom: review found controller_sha256 was computed only after both children
  finished. Editing the controller while it waited could label old executed
  instructions with the hash of new file bytes. No such historical edit is claimed.
- Root cause: completion-time file identity was mistaken for launch identity,
  the same provenance boundary as V-014.
- Detection class: provenance.
- Instrument: capture controller source hash before planning/spawn; require it
  unchanged before writing a verified result and record that launch hash. New
  source-change fixture must refuse; do not edit an active controller file.
- Green gates: docs/evidence/segment-controller-contracts/scope.json,16 cases
  on b8eb48a/controller11a0a81a,including changed-source refusal. Actual native
  docs/evidence/m3-controller-recovery-r1/scope.json verifies three unchanged
  controller executions (13.512/11.658/11.088s),partial1+resume2==continuous3,
  exact model/optimizer/RNG payload, data/loss/gradient/evaluation and curation
  identities. This is finite integration evidence, not a long-memory/quality gate.

## V-021 — CLOSED: stop collector text assumed the child receipt was stale

- Symptom: the s3 ancestor and child both recorded memory_pressure/125, but
  pre-curation review found fixed scope wording calling the child receipt stale.
- Root cause: prose inherited the first ancestor-only stop even though structured
  child_receipt_status/terminal fields already support both cases.
- Detection class: provenance.
- Instrument: neutral verbatim-receipt wording plus actual structured terminal
  fields. Historical s1 scope remains unchanged; its running/null child was stale.
- Green gate: docs/evidence/m2-stream-accuracy4096-s3-stop/scope.json retains
  both terminal125 receipts, reports child_receipt_terminal=true, verifies1152
  tensor/manifest/ledger hashes and five saved PIDs dead, and has neutral scope
  wording. Native collection.441s,36,564,992B/128MiB/30s. This is file integrity,
  not semantic optimizer restoration or completed training/quality evidence.

## V-020 — CLOSED: joint diagnostic would reject a decorated frozen metric

- Symptom: pre-native review found the new component origin check inspected
  component_metrics.__code__, which belongs to Torch's no_grad decorator.
  The valid frozen metric would be rejected before any model forward.
- Root cause: callable wrapper location was mistaken for implementation origin.
- Detection class: provenance.
- Instrument: inspect.unwrap before checking the implementation's resolved
  source path. Actual decorated component metric must pass for its source tree
  and fail for the scripts tree. The source archive is verified separately.
- Green gate: docs/evidence/joint-error-diagnostics-full/scope.json on577c7ec,
  runs/joint-error-diagnostics-r3,27/27 in1.745s,246,988,800B/512MiB. The actual
  Torch-decorated component metric passes its implementation tree and rejects
  the wrong tree. No native M3 model forward replay is claimed by this unit gate.
  Later actual frozen M3 replay also passes27 forwards plus evidence collection:
  docs/evidence/m3-stream-large-profile-errors-r1/scope.json,25.603s,
  563,679,232B/640MiB. This separately verifies the decorated implementation
  origin in a real archived-model run; it does not establish model accuracy.

## V-019 — CLOSED: component diagnostic fixture swapped queue anchor flags

- Symptom: the first27-case diagnostic gate had26 passes and one fixture error
  before component analysis; Queue rejected a cropped panel with end_observed=True.
- Root cause: new fixture supplied positional start/end booleans in the wrong
  dataclass order. A pure unseen-start fixture also had an impossible cropped/
  observed-end combination, corrected to both anchors unseen.
- Detection class: test-instrument.
- Instrument: existing validated Queue constructor; use named end_observed and
  start_observed arguments in new diagnostic fixtures. Cross-check decomposed
  metrics against actual component_tensors/component_metrics on valid states.
- Failing gate: runs/joint-error-diagnostics-r1.stderr.log,27 cases, one error;
  native512MiB/60s,245,854,208B peak. No model forward/update was attempted.
- Green gate: docs/evidence/joint-error-diagnostics-full/scope.json,27/27
  on577c7ec in1.745s,246,988,800B/512MiB. It executes component tensors/metrics
  on the corrected cropped queue plus absent, empty and unknown-middle states.
  Red r1 and intermediate r2 raw receipts/logs are retained beside this scope;
  only the final r3 contributes the27-case claim.

## V-018 — CLOSED: new joint scopes would inherit a fixed M2 architecture label

- Symptom: pre-curation review found record_stream_run's scope still hardcoded
  Unchanged M2 despite the new joint checkpoint architecture being M3. No wrong
  joint scope has been published; prior M2 scopes keep their original identity.
- Root cause: architecture text was copied from the original collector instead
  of being derived from the verified checkpoint/configuration.
- Detection class: provenance (third; standing instrument promoted in AGENTS).
- Instrument: checkpoint architecture/config/capacity must match the run contract;
  report the actual identity. Legacy M2 and joint M3 acceptance/mismatch fixtures.
  V-003 requires computed identity; V-014 requires actual frozen execution source.
- Green gates: docs/evidence/v018-architecture/scope.json,4 receipt/architecture
  cases on371bfd8; docs/evidence/m3-stream-pilot-profile-r1/scope.json verifies
  the actual native two-step joint artifact as TetrisStateNet-M3/capacity64,
  matching checkpoint/config/source/weight identity,2 recipes and27 saved probes.
  Native558,497,792B/640MiB; old M2 scopes retain their historical identities.

## V-017 — CLOSED: M3 joint-gradient fixture had an undersized canvas

- Symptom: first M3 gate passed14/15; joint-gradient case failed before model
  invocation with source panel outside canvas (m3-component-tests.stderr.log).
- Root cause: fixture queue panel ended at195px on a192px-wide canvas.
- Detection class: test-instrument.
- Instrument: existing layout.validate_panels refuses unsupported fixtures before
  rendering/model work; widen fixture to208px, preserve full source resolution.
- Green gate: runs/m3-component-tests-r2.stderr.log,15/15 in1.774s under512MiB/
  60s,357,052,416B raw peak. Corrected fixture executes finite joint backward and
  isolated component-to-encoder gradients. Earlier red established no joint grad.
  Committed-source scope: docs/evidence/m3-joint-contracts/scope.json onfdb0c81.

## V-016 — CLOSED: M3 source component labels lacked prepared-pixel admission

- Symptom: review of the new optional M3 integration found source Icon labels
  would survive input resizing or invalid alpha without component pixel geometry
  to establish whether the prepared image still supports their identities.
- Root cause: reusing a verified source-observation projection after a separate
  image transform. M2 cell geometry/coverage masking does not cover icons.
- Detection class: visibility-contract.
- Instrument: M3 target preparation temporarily refuses resized or invalid source
  pixels. Pure component labels stay unchanged; no latent identities/counts are
  guessed. Resize/alpha refusal and paired-hidden-raster tensor fixtures required.
  A future resized component observer needs its own paired-raster policy gate.
- Green gate: runs/m3-component-tests-r2.stderr.log,15/15, includes resize/alpha
  refusal and exact paired-hidden-raster tensor targets under512MiB/60s.
  This closes the unsafe admission; implementing resized-component observation
  remains a feature gap. No M3 learning/runtime quality claimed.
  Committed-source scope: docs/evidence/m3-joint-contracts/scope.json onfdb0c81.

## V-015 — CLOSED: transient receipt replacement denied before child launch

- Symptom: recovery-gate profile2/full6 passed, then partial Job failed before
  child on WinError5 replacing partial-job.tmp -> partial-job.json. Error-path
  receipt publication succeeded immediately afterward. Root gate failed1 at
  37.676s. All owned processes terminal; no host crash.
- Root cause: atomic rename can encounter a short-lived Windows sharing/access
  denial. The locking owner is unknown; no antivirus cause is claimed. No retry
  existed, so a transient denial aborted an otherwise valid launch.
- Detection class: windows-persistence.
- Instrument: shared atomic replace helper, at most6 attempts/150ms total sleep
  for WinError5/32/33 only. Persistent/other IO errors still fail closed; no child
  before durable admission receipt. Prior complete snapshot pointer survives.
  Source archive includes helper; diagnostics inspect failed receipt before
  looking for a nonexistent worker summary.
- Green gates: docs/evidence/v015-contracts/scope.json, f583792's atomic3,
  recovery7/resources6/pressure5/source4/admission4 =29 scoped checks. Native
  runs/m2-stream-recovery-gate-r2/proof.json PASSED2+6+3+3, exact model/Adam/RNG,
  losses/gradients/data/evaluation and all child deaths; outer peak590,663,680B
  under1GiB, child max554,541,056B under640MiB. Earlier partial gate stays red.

## V-014 — CLOSED: traceback read edited parent trainer lines

- Symptom: failed learn768-r2 trace names curve.write/final_step for line numbers
  whose original frozen trainer performed loss.backward. Parent source was
  edited during the run; all executed module bytecode remained the old version.
- Root cause: model/sampler imports were frozen, but trainer main was compiled
  with the parent file path. Traceback linecache read the later file text.
- Detection class: provenance.
- Instrument: bootstrap archives source then loads/executes trainer main
  from its verified archive; saved driver path and native integration gate pin
  archive identity. Preserve original archived lines when reading old trace.
- Green gate: runs/m2-stream-recovery-gate-r2/proof.json, all four child driver
  source paths lie in verified archives; identical catalogue121fd05a... and
  exact2+6+3+3 state/optimizer/RNG/evaluation trajectory. Parent script only
  bootstraps; traceback of actual model work is compiled from the archive.

## V-013 — CLOSED: two-step square admission missed long diverse-shape memory

- Symptom: runs/m2-stream-learn768-r2-job.json failed1 at324.600s,565/768
  updates saved in finite ledgers; next backward denied599,040B allocation.
  Raw Job peak704,634,880B under readback640MiB (V-004 accounting caveat), OS
  commit21.09GB, no pressure stop or host crash. No final weights were saved.
- Root cause: one maximum-shape warm step did not characterize long-running
  variable-shape/native-cache behavior. Native oneDNN cache is a hypothesis,
  not established cause. Torch model/data/loss code retains no tensor cache.
- Detection class: resource-profile.
- Instrument: pre-import16-entry native cache request with controlled eviction
  evidence (default repeated shape hit vs bounded miss); raw Job memory every16
  updates, periodic recovery and exact global-step replay. Native capacity query
  is not exported; exact attribution of every allocation remains unmeasured.
  A square-only profile is plumbing evidence, not a general long-memory claim.
- Failure evidence remains in runs/m2-stream-learn768-r2/{code,curve.jsonl,
  training.jsonl,evaluations.jsonl}; original stderr lines map to archived trainer,
  not later parent text (V-014). No automatic restart or cap increase.
- Green gate: docs/evidence/m2-stream-learn768-r3/analysis.json and scope.json,
  complete768 at unchanged640MiB,458.494s inner/459.718s outer. Peak519,782,400B,
  second-half growth2,748,416B <=predeclared32MiB. All565 old recorded loss/terms/
  gradient norms and data/pixel/label recipes match exactly; package/sampler/lock
  unchanged. Final recovery768 and every-final-tensor reload exact. Curator
  replayed768 samples+27 saved probes. Inference weights d183a22a... retained.
  This closes the observed finite resource failure; no baseline model tensors
  survived and no untested longer-run or model-accuracy claim is made.

## V-012 — CLOSED: recovery harness imported unbounded library thread pools

- Symptom: runs/training-snapshot-tests-r2.stderr.log reports OpenBLAS allocation
  failure during imports, before any of seven fixtures ran. Native512MiB cap
  remained armed; raw peak564,621,312B is retained, not clamped (V-004).
- Root cause: test called torch.set_num_threads(1) in setUp after import Torch;
  no OMP/MKL/OpenBLAS environment was set. NumPy's OpenBLAS DLL initializes at
  import. Same512MiB rerun passes with pre-import policy; exact allocation sizes
  inside OpenBLAS remain unmeasured.
- Detection class: test-instrument (third member; standing rule promoted).
- Instrument: runner sets OMP/MKL/OpenBLAS/legacy thread counts to1 in child
  environment before spawn, overriding host values; receipt records policy.
  Live tripwire supplies host values8 and checks child before heavy imports.
  Recovery fixtures refuse missing pre-import limits and assert actual Torch
  thread count1. No memory ceiling raised; no host apps altered.
- Reference: https://www.openmathlib.org/OpenBLAS/docs/runtime_variables/
- Green gates: runs/training-snapshot-tests-r3.stderr.log,7/7 in4.103s,
  332,562,432B raw peak under the unchanged512MiB cap; exact resumed stochastic
  AdamW/model/optimizer/RNG plus interrupted publication and corrupt prefixes.
  runs/v012-resource-tests.stderr.log6/6 in3.116s (includes host8->child1),
  v012-pressure-tests.stderr.log5/5 in2.147s and v012-admission-tests.stderr.log
  4/4 in.275s. All22 checks are separate scoped gates, not a whole-suite claim.
  Committed-source scope: docs/evidence/v012/scope.json (7cbc68c).
  Runner SHA/catalogue changes invalidate
  old live admission; fresh two-step warm characterization before training.

## V-011 — CLOSED: resource profile covered only a cold optimizer backward

- Symptom: source inspection found the one-step square admission performed its
  only backward before AdamW allocated moment tensors. No resource failure or
  host crash is claimed. Installed Torch2.14.1 adam.py initializes exp_avg and
  exp_avg_sq inside _init_group during the first optimizer.step().
- Root cause: input shape coverage was mistaken for training memory coverage;
  the cold backward and final evaluation never combined saved backward tensors
  with resident optimizer state.
- Detection class: test-instrument.
- Instrument: version2 requires two optimizer steps on the same512-square frame;
  before the second forward, every trainable parameter must have both full-size
  moment tensors and step1. Record their bytes and completed warm backward.
  Reject cold/old/partial profiles, missing state and smaller warm input.
  Native Job cap and V-010 pressure monitoring remain unconditional.
- Green gates: docs/evidence/v011-protocol/v011-stream-tests.stderr.log,
  4/4 in .232s, source322e055; docs/evidence/m2-stream-warm-probe/summary.json,
  two512-square steps in9.553s (outer10.394s), raw Job peak594,829,312B under
  640MiB. All142 trainable parameter tensors had moment state (5,177,352B)
  before the second forward; warm backward completed, every final tensor reload
  exact. Curator replayed2 training samples and27 saved probes and revalidated
  version2 admission against the frozen runner. Old cold receipts stay historical.

## V-010 — CLOSED: bounded job ignored system-wide commit pressure

- Symptom: runs/v010-before.stderr.log, synthetic critical OS counter still
  launched the child and exited0; expected refusal125. No host-scale allocation
  or actual crash occurred. A real host snapshot was near30.9 GB commit.
- Root cause: per-Job cap/wall timeout existed, but the supervisor never read
  system commit/available RAM and blocked in one long wait until child exit.
- Detection class: resource-pressure.
- Instrument: PSAPI GetPerformanceInfo page-to-byte counters, independent of
  nested Job identity; unconditional .2s polling, full ceiling reservation at
  launch, early commit/available-RAM thresholds, durable receipt then complete
  owned-Job termination. Counter failures fail closed; no opt-out. Synthetic
  tripwires use tiny real Jobs and verify actual child/grandchild PIDs died.
- References: https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-getperformanceinfo
  and https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-performance_information.
- Partial evidence: docs/evidence/v010-admission-only/scope.json records the red
  before case and a real OS-pressure refusal with the native cap armed. The
  pressure harness parked before executing tests; this is not a green suite.
- Green gates: docs/evidence/v010/v010-pressure-tests-r2.stderr.log, 5/5 in
  2.006 s (admission, pressure/error full-tree death, global counters and policy);
  v010-resource-tests.stderr.log, 5/5 in 2.475 s (native allocation denial,
  wall/tree death, nested/forged identity and clamp); strengthened admission
  v010-stream-admission-tests.stderr.log, 3/3 in .232 s. scope.json pins source
  commit64e582c and supervisor SHA635e764a...; tiny native Jobs, no host-scale
  allocation. The earlier parked harness remains historical partial evidence.

## V-009 — CLOSED: ambiguous corpus labels admitted off-image known cells

- Symptom: final review found full-board bounds were checked only for readable
  records; an ambiguous partial board could claim certainty outside the raster.
- Root cause: geometry existence was mistaken for evidence that each known cell
  lies inside the image. This repeats V-001/V-006's visibility boundary in the
  separate real-corpus admission path, without changing renderer/model policy.
- Detection class: corpus-admission.
- Instrument: each known cell needs all four corners in the source; same-raster
  empty/occupied off-image declarations both fail, unknown cropped cells pass.
- Green gate: docs/evidence/corpus-integrity-final/corpus-fixtures-r3.stderr.log,
  26/26 in 3.620 s, including the same-raster off-image certainty fixture.
  scope.json pins the final code/test hashes and native 256 MiB / 60 s receipt.

## V-008 — CLOSED: malformed corpus rows broke the audit report

- Symptom: pre-gate review found report() called .get on arbitrary JSON rows;
  an invalid duplicate ID could also inflate eligible split counts.
- Root cause: reporting trusted the input shapes rejected by validation and
  counted row occurrences instead of successfully admitted identities.
- Detection class: corpus-admission.
- Instrument: malformed declaration/report fixtures, unique admitted identity
  counts, strict JSON duplicate/nonfinite rejection. The corpus cannot seal
  while any validation error remains.
- Green gate: docs/evidence/corpus-integrity/corpus-fixtures-r2.stderr.log,
  25/25 in 4.499 s, including malformed declarations/reporting and duplicate ID
  count. The actual guarded audit/seal/verify CLI also passed. Source identity and
  the 256 MiB / 60 s / 25-case receipt are in the same directory's scope.json.

## V-001 — CLOSED: latent geometry survived ambiguous crops

- Symptom: first fixtures (runs/fixtures-r1.stderr.log) failed
  `test_partial_crop_is_unknown_not_latent_truth` and
  `test_accepted_cannot_contain_unknown` (37/39 passed).
- Root cause: renderer transformed latent origin even with missing row/column
  anchors; state accepted a board with no supported lattice.
- Detection class: visibility-contract.
- Instrument: conservative missing-anchor lattice masking; acceptance requires
  geometry; the two named fixtures pin both boundaries.
- Green gate: docs/evidence/fixtures-r3.stderr.log, 44/44; repeated on
  docs/evidence/fixtures-final.stderr.log, 46/46. Both named geometry fixtures green.

## V-002 — CLOSED: indistinguishable layers could receive certain labels

- Symptom: code review found active/locked used the same skin while labels
  trusted latent membership; arbitrary remapped locked types had the same risk.
- Root cause: final-frame identifiability had not been separated from generator
  layer/type knowledge.
- Detection class: visibility-contract.
- Instrument: distinct style conventions for identifiable pilot families;
  matched-pixel active/locked counterexample with equal unknown outputs;
  unsupported remapped types masked independently of occupancy.
- Green gate: docs/evidence/fixtures-r3.stderr.log, 44/44; repeated on
  docs/evidence/fixtures-final.stderr.log, 46/46. The paired-image and unseen-
  palette tests are green; real-frame recoverability remains unmeasured.

## V-003 — CLOSED: copied brief had an incorrect handwritten checksum

- Symptom: checkpoint SHA256 disagreed with Get-FileHash of the byte-exact copy.
- Root cause: checksum was entered manually without comparison to source.
- Detection class: provenance.
- Instrument: evidence script compares copied and supplied bytes and computes
  all recorded checksums, rather than trusting the checkpoint string.
- Green gate: docs/evidence/evidence.json (passed, rc=0), foundation.json records
  the checksum computed after byte-for-byte comparison to the supplied brief.

## V-004 — CLOSED: fixture confused peak accounting with admission

- Symptom: fixtures-r2 had 43/44 green. The 80 MiB Python allocation was denied
  within the 64 MiB Job, yet raw PeakJobMemoryUsed read 108,417,024 B, reding an
  assertion that the historical peak could never exceed the configured limit.
- Root cause: the fixture inferred allocation success from a peak counter.
  The precise Windows accounting cause is not established. The limit readback
  and actual native allocation admission are the contracts we can test directly.
- Detection class: test-instrument.
- Instrument: native VirtualAlloc 4 MiB succeeds / 80 MiB fails under the readback-
  verified 64 MiB ceiling. Preserve raw peak values without clamping them.
  Reference: https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_limit_information
- Green gate: docs/evidence/fixtures-r3.stderr.log, 44/44; repeated on
  docs/evidence/fixtures-final.stderr.log, 46/46, including native admission and
  child/grandchild death. Raw accounting behavior remains documented above.

## V-005 — CLOSED: cropped components leaked latent queue length and labels

- Symptom: pre-change review found that a queue wholly outside the crop still
  emitted one unknown token per latent preview. Absent/cropped hold, unseen spawn
  pieces and remapped preview support had related observation-boundary gaps.
- Root cause: v1 constructed labels from latent component presence/length before
  deriving which evidence survived capture; global row indices survived loss of
  their coordinate anchors.
- Detection class: visibility-contract (third entry, standing instrument promoted).
- Instrument: centralized observe.py; queue visible-span trimming and explicit
  start_observed; schema 1.1 reads historical 1.0 conservatively; independent
  preview palettes cannot provide board type support. Crops losing registration
  mask canonical rows. Paired-raster equivalence fixtures are mandatory in AGENTS.
- Green gate: docs/evidence/m1/m1-fixtures-r2.stderr.log, 67/67 in 3.497 s,
  including all twelve ObservationEquivalenceTests; receipt
  docs/evidence/m1/layout.json names source hashes. Thirty demo records and three
  explicit visible-flow images verified by m1-evidence.json (rc=0).

## V-006 — CLOSED: unregistered crop retained canonical active pose

- Symptom: M2 target review found that board rows/geometry were masked after loss
  of coordinate anchors but the separately observed active/ghost footprints were
  still indexed in the latent original board frame.
- Root cause: layer visibility was decided before registration; canonical pose
  observability was never gated on a supported coordinate system.
- Detection class: visibility-contract.
- Instrument: identical crops from boards shifted by one cell, with compensating
  active columns, must yield equal unknown poses. Registration now gates every
  canonical layer target, not only locked rows.
- Reproduction: docs/evidence/v006/v006-before.stderr.log, 1 failed / 21 tests,
  equal pixels but active columns 2/3 differed in the emitted state.
- Green gate: docs/evidence/v006/v006-after.stderr.log, 69/69 in 3.685 s.
  Public constructors now reject observed canonical layers without geometry;
  the historical 1.0 cropped example migrates to masked cells/layers. Scope and
  source hashes are in docs/evidence/v006/scope.json. M1 receipts remain history
  on 3cd063b's pre-repair sources; old affected 1.1 annotations are rejected.

## V-007 — CLOSED: worker inspected an incidental nested Job

- Symptom: runs/m2-probe-job.json failed before Torch import. The supervisor's
  explicit 1 GiB cap read back correctly, but the worker's NULL-handle query
  returned flags 0x3000 and local memory limit 0 (m2-job-diagnostic.log).
- Root cause: an implicit current-Job query does not identify the supervisor's
  object when an additional nested Job exists. Inherited parent enforcement
  remains active; the worker incorrectly treated the incidental local limits
  as the cap's identity. Native allocation-denial gates remain independent proof.
- Detection class: resource-identity.
- Instrument: unique named supervisor Job; worker opens it query-only, verifies
  actual process membership, then reads its limits. Added nested-Job and forged-
  identity fixtures; no memory ceiling was raised or disabled for this repair.
- References: https://learn.microsoft.com/en-us/windows/win32/api/jobapi/nf-jobapi-isprocessinjob
  and https://learn.microsoft.com/en-us/windows/win32/procthread/nested-jobs
- Green gate: docs/evidence/m2-numerics/m2-fixtures-r1.stderr.log, 85/85 in
  6.992 s, including nested/forged identity, native allocation denial and
  child/grandchild death. Numerical worker admitted the exact 1 GiB Job and
  completed one forward/backward step (m2-probe-r2-result.json). Source/lock
  identity: numerical.json. Raw historical failure/diagnostic logs retained.

## V-071 — CLOSED: a no-parameter matched-start guard refused a parameter-adding lever

- Symptom: `runs/m3-occorigin-curation-r1` (`scripts/record_lattice_lever.py`) failed in 0.24 s with
  `ValueError: the arms must start from one identical saved initial tensor file`, before writing a scope,
  on data whose arms did start from the published checkpoint.
- Root cause: the guard was copied from the geometry lever together with its comment "this lever adds no
  parameter". The box-origin lever adds `support_box.{weight,bias}`, so the two saved `initial-weights.pt`
  files differ in digest by four zero tensors. The predeclaration sentence "both arms start from identical
  saved tensors" repeated the same false premise; what it meant - and what matching requires - is
  functional identity at step zero.
- Detection class: test-instrument.
- Instrument: `initial_state_matches` compares the two saved state dictionaries directly - every published
  tensor bitwise equal, the added keys exactly `support_box.{weight,bias}` and all-zero, nothing else added
  or removed - and the curator additionally requires identical step-0 probe metrics
  (`initial_training_probe`, `initial_development`). Bound by `MatchedStartTests` in
  `tests/test_lattice_lever.py`, which rejects a moved shared tensor, a nonzero projection, and any other
  added or missing tensor. The file-digest guard was removed rather than relaxed; it remains the correct
  check for levers that add no parameter.
- Green gate: focused `tests.test_lattice_lever tests.test_placement_error tests.test_geometry_lever
  tests.test_detail_stencil`, 55/55 in 2.810 s, peak 293,011,456 B (`runs/m3-occorigin-focused-r1-job.json`);
  curation `runs/m3-occorigin-curation-r2-job.json` passed and published
  `docs/evidence/m3-occupancy-translation-r1/scope.json`.

## V-072 — CLOSED: a matched-start guard compared the objective's value across the arms

- Symptom: `runs/m3-occextent-curation-r1` (`scripts/record_extent_lever.py`) failed in 2.2 s with
  `ValueError: step zero is not the same measured function in both arms (initial_training_probe)` and
  wrote no scope, on two arms whose saved initial weights were bitwise identical
  (`initial_weights_sha256` equal) and whose step-0 predictions matched on every field.
- Root cause: `matched_start.step_zero_identical` compared the whole step-0 probe record, which carries
  the objective's value both at the top level (`loss`) and per case (`cases[*].loss`). A lever that
  changes the objective changes those numbers at step zero by construction, although the model is the
  same function, so a correct matched start was refused. Same premise error as V-071 one layer lower:
  V-071 assumed the lever adds no parameter, this assumed the lever changes no objective value.
- Detection class: test-instrument.
- Instrument: `step_zero_identical` strips the objective's own fields recursively (top level and every
  case) before comparing, and names the differing field on failure; the shared-tensor/all-zero
  comparison in `matched_start` is unchanged. Bound by `MatchedStartTests` in
  `tests/test_extent_lever.py`, which accepts a re-lossed probe and refuses a moved `cell_accuracy` and
  a missing probe.
- Green gate: `runs/m3-occextent-tests-r4-job.json`, 50/50 in 2.188 s, peak 248,434,688 B; curation
  `runs/m3-occextent-curation-r2-job.json` passed and published
  `docs/evidence/m3-occupancy-extent-r1/scope.json`. The refused launch is preserved as
  `runs/attempts/m3-occextent-curation-r1-job.json.attempt1` with a note.

## V-073 — CLOSED: a patch generator deleted a published function's return line

- Symptom: after applying the capacity lever's `losses.py` patch, `extent_error` returned `None` and
  `tests.test_extent_lever` failed with `TypeError: equal(): argument 'input' ... must be Tensor, not
  NoneType`.
- Root cause: the anchor block was substituted for a span that began with `extent_error`'s
  `return torch.log(predicted/reference)` line, and the replacement text started at the new comment,
  so the published return was dropped. The patch's own count guard checked the anchor span, not the
  preserved neighbour.
- Detection class: generated-patch integrity. This is the first instance in this class; the new
  tests could not have caught it, because they exercise only the added term.
- Instrument: the pass re-ran the *existing* lever suites (`test_extent_lever`, `test_geometry_lever`,
  `test_lattice_lever`, `test_placement_error`) alongside the new ones, so a published function's
  regression is caught by its own instrument rather than by inspection.
- Green gate: `runs/m3-occgeocap-focused-r3-job.json`, 67/67 in 1.881 s, peak 249,286,656 B. A third
  instance in this class would require a standing patch-integrity instrument.

## V-074 — CLOSED: two mis-wired receipt paths refused the capacity curation

- Symptom: `runs/m3-occgeocap-curation-r1` refused in 2.2 s with `declared receipt ...-sweep-r1-
  profile-r1/summary-job.json is missing`; the r2 retry refused in 2.1 s with `TypeError: argument
  should be a str or an os.PathLike object ..., not 'dict'`, both before writing a scope.
- Root cause: the curator derived the sweep profile's job path from `profile_summary.stem` (which is
  `summary`) instead of rebuilding `runs/<profile>-job.json` from the profile directory, and the first
  fix then passed the loaded job *dict* where a path was required. Both are the same wiring error: the
  curator resolves some receipts as paths and others as parsed documents.
- Detection class: curator refusal-to-run wiring. Two instances in this pass; a third requires a
  standing instrument (a dry-run that resolves every declared receipt before any measurement is read).
- Instrument: `receipts_after` names the missing path in its refusal, and the pass's own tests keep
  `receipts_after` bound to real files; the definitive check is that curation re-derives every number
  from the same receipts.
- Green gate: `runs/m3-occgeocap-curation-r3-job.json` passed in 5.153 s, peak 321,175,552 B and
  published `docs/evidence/m3-occupancy-geometry-capacity-r1/scope.json`. Refused launches are
  preserved as `runs/attempts/m3-occgeocap-curation-r{1,2}-job.json.attempt1` with notes.

## V-075 — CLOSED: adding the new owned family silently removed the previous one

- Symptom: `tests.test_capacity_lever.CuratorJoinTests.test_the_family_is_owned` failed with
  `AssertionError: None is not true` for `m3-occgeocap-lever-r1` after the residual lever added its
  own family.
- Root cause: the owned-family update replaced the existing alternation
  `m3-occgeocap-[a-z0-9-]{1,40}` with `m3-occlatres-[a-z0-9-]{1,40}` instead of adding a branch, so a
  previously owned family stopped being recognised by `record_stream_run`.
- Detection class: ownership/wiring regression. First instance in this pass; caught by the *previous*
  lever's own test, which is why the focused run includes every earlier lever suite.
- Instrument: `tests/test_capacity_lever.py::test_the_family_is_owned` asserts the old family, and
  `tests/test_lattice_residual_lever.py` asserts the new one, so neither can regress silently.
- Green gate: `runs/m3-occlatres-focused-r3-job.json`, 85/85 in 0.557 s, peak 296,853,504 B for the
  failing run and 322,686,976 B for the curation.

## V-076 — CLOSED: the new rule ignored the declared placement reduction

- Symptom: `tests.test_lattice_residual_lever.LatticeResidualRuleTests.test_the_decomposition_is_non_circular`
  returned `scaling` where the declared first branch (`improvement < 0.05`) required
  `read_out_inheritance`.
- Root cause: `lattice_residual_decomposition` recomputed `placement_reduction` from the two placement
  summaries instead of reading `decision["placement_improvement"]`, the value the verdict itself
  scored. The two agree on real receipts but not on a constructed decision, which is exactly the state
  a rule test builds; a divergence would have made the rule and the verdict disagree silently.
- Detection class: rule/verdict disagreement. First instance.
- Instrument: the rule is now a pure function of the verdict's own decision record, and its test
  drives improvement and spread independently, including that none of the four spent levers can be
  returned.
- Green gate: `runs/m3-occlatres-focused-r3-job.json` (85/85) and curation
  `runs/m3-occlatres-curation-r1-job.json`, which passed in 6.346 s, peak 322,686,976 B, and
  published `docs/evidence/m3-occupancy-lattice-residual-r1/scope.json`.

## V-077 — CLOSED: the rule-insertion anchor matched two functions

- Symptom: none in a run. `runs/patch_lattice_free.py` refused to apply the
  `lattice_free_decomposition` insertion until its anchor was narrowed, because the trailing two lines of
  `capacity_decomposition` and `lattice_residual_decomposition` are identical.
- Root cause: the anchor was written from the tail of the function it followed, and the earlier capacity
  rule ends with the same `floors` lines apart from one preceding line. A blind replace would have
  inserted the new rule inside `capacity_decomposition`.
- Detection class: text-anchor ambiguity on a shared suffix. First instance in this campaign.
- Instrument: the patch counts occurrences and aborts unless exactly one matches, and the new rule's test
  drives all four branches; the anchor now includes the distinguishing `PLACEMENT_IMPROVEMENT` line.
- Green gate: the patch applied cleanly to five files, `ast.parse` passed for each, and
  `runs/m3-occlatfree-focused-r2-job.json` ran 98/98.

## V-078 — CLOSED: a test compared the declared constant with a tiny-config count

- Symptom: `tests.test_lattice_free_lever.LatticeFreeContractTests.test_the_declared_addition_is_exactly_the_model_difference`
  failed with `4226 != 290` in the first focused run.
- Root cause: the assertion reused the full-model declared parameter constant next to a tiny-config
  expectation, so it checked two different configurations in one line. The model-difference check above it
  was correct; only the arithmetic expectation was wrong.
- Detection class: test expectation conflated two configurations. First instance.
- Instrument: the tiny-config assertion now sums the tiny model's own added tensors, and the declared
  constant is checked separately against the full architecture.
- Green gate: `runs/m3-occlatfree-focused-r2-job.json`, 98/98 in 0.757 s; the predeclaration commit
  `7b3ca9a` was made only after that green run.

## V-079 — CLOSED: an activation test assumed both axis-scale entries shared a sign

- Symptom: `tests.test_board_refine_lever.BoardRefineContractTests.test_the_activation_summary_reports_applied_magnitudes`
  failed with `0.243347 != 0.272530` in the first focused run of this pass.
- Root cause: the test fed `[2, -2, 2, -2]` and expected `|exp(x)-1|` for both scale entries, but the
  declared diagnostic averages the applied magnitudes of a signed pair, `(|exp(x)-1| + |exp(-x)-1|)/2`,
  which is the meaningful quantity for a log-scale. The instrument was right and the expectation was
  wrong; the test now checks both a same-signed pair and a signed pair.
- Detection class: test expectation conflated two inputs. Second instance after V-078.
- Instrument: the diagnostic is one declared function (`activation_summary`) used by the probe, with the
  floor applied by one function (`activation_engaged`) that the rule reads, so a disagreement between
  measurement and verdict has one place to show up.
- Green gate: `runs/m3-occbref-focused-r2-job.json`, 93/93 in 0.958 s; the predeclaration commit
  `72f95bc` was made only after that green run, and the full suite is run again before the record commit.

## V-080 — CLOSED: the fit-budget predeclaration declared a wall envelope the runner refuses

- Symptom: `runs/m3-occfit-lever-r1.log` recorded `run_bounded.py: error: command and wall cap in (0, 3600]
  required`; the lever study never started, although its 2-step admission profile had already passed.
- Root cause: the predeclaration sized the 1,024-step lever at 5,400 s from an arithmetic estimate without
  checking the bounded runner's own hard cap. The cap is a guard and is not raised to fit a declaration.
- Detection class: declared envelope exceeds the execution tool's own limit. First instance.
- Instrument: the curator's admission join now refuses a declared wall above `MAX_WALL_SECONDS = 3600`
  (bound by a test), so an over-cap envelope fails at curation as well as at launch.
- Resolution: the predeclaration is amended in its own commit, before any study receipt, to a 768-step (3x)
  lever at 3,600 s; criterion, activation floor, matched start and the zero-parameter declaration are
  unchanged. The rejected launch is preserved as `runs/attempts/m3-occfit-lever-r1.attempt1.log` with a note
  and is not evidence.
- Green gate: `runs/m3-occfit-focused-r2-job.json`.

## V-081 — CLOSED: the amended envelope still exceeded a tool cap, this time the driver's

- Symptom: after the runner cap was repaired, the 768-step study was refused by the driver itself:
  `train_m2_stream.py: error: wall budget (0,1800] and learning rate (0,.01] required`. No optimizer step ran.
- Root cause: the amendment sized the lever from `run_bounded.py`'s 3,600 s cap while the training driver
  enforces a second, tighter 1,800 s cap of its own, and `--resume-from` completes only an unfinished stream,
  so the finished 256-step arm could not be extended in segments either.
- Detection class: declared envelope exceeds an execution tool's own limit. Second instance (V-080 is the
  first), so the guard now covers both caps and the predeclaration's own envelope table.
- Instrument: `MAX_WALL_SECONDS = 1800` in the curator's admission join, bound by tests for an over-cap wall
  (refused) and an in-cap one (accepted), so neither cap can be declared past again without failing curation.
- Resolution: second amendment, in its own commit and still before any study receipt: lever 384 steps
  (1.5x the settled 256) at 1,800 s, with 4.09 s/step measured on the baseline arm leaving headroom. Criterion,
  activation floor, matched start and zero added parameters are unchanged. The rejected launch is preserved as
  `runs/attempts/m3-occfit-lever-r1.attempt2.log` with a note and is not evidence.
- Green gate: `runs/m3-occfit-focused-r3-job.json`.

## V-082 - CLOSED: the non-circular rule ranked an intervention this architecture cannot express

- Symptom: the fit-budget record's `capacity_limit` branch named *"a wider shared trunk than capacity 64"*.
  Building the declared configuration at capacity 32 and 64 returns identical censuses (8,018,505 parameters
  over 271 tensors), `ModelConfig` refuses 65 and above, the driver refuses it earlier with the renderer's
  ten-slot reason, and the encoder widths `(48, 96, 192, 256)` with dimension 192 have no training flag, so
  the ranked arm had no experiment to run.
- Root cause: the rule was authored from the lever's name (`capacity`) instead of from what that parameter
  changes: `component_capacity` bounds the component decoder's slots, so it is parameter-identical across
  10..64 and cannot be widened at all.
- Detection class: a decision rule naming an intervention the model cannot express. First instance.
- Instrument: `placement_error.capacity_width_admissibility` takes the two censuses, the trunk description and
  the construction bound, and refuses to report "expressible" unless capacity demonstrably changes the parameter
  count; three tests bind the equal-census case, the changed-census case (which refuses rather than repeating a
  stale verdict) and the bound-is-not-a-dial case.
- Resolution: the `capacity_limit` branch now names *"a declared trunk-width lever, which capacity 64 cannot
  express"*, and `docs/M3_OCCUPANCY_TRUNK_WIDTH.md` records the block with the exact numbers and the user
  decision that would unblock a real width lever. No arm was launched and no receipt was manufactured.
- Green gate: `runs/m3-occtrunk-suite-r1-job.json`.

## V-083 - CLOSED: one non-finite forward batch aborted a 1,000-step segment with no checkpoint

- Symptom: segment 17 of the fixed-pool GPU chain exited 1 after about 80 steps with
  `ValueError: finite board-conditioned component query required`, raised from `model.forward` at
  `src/tetris_vision/components.py:71`. `state.pt` and `result.json` were never written, and the driver
  stopped the chain at 12,800/20,000 steps with "produced no resumable state".
- Root cause: one batch produced non-finite board values inside the forward pass, so the engine's own
  finite-state guard raised. The guard was right; the trainer had no per-batch failure path, so a single
  pathological batch was escalated into a whole-segment abort. No backward ran, so no saved weight was
  poisoned (verified: 269 tensors, zero non-finite, in all 20 saved chain states).
- Detection class: a per-item failure escalated into a whole-run abort. First instance.
- Instrument: `runs/gpu_fit_test.py` counts and skips a non-finite batch at forward, loss and gradient-norm,
  saves rolling state across the skip, and writes `skipped_batches` into every `result.json`; a 90 s
  faulthandler watchdog dumps all stacks on genuine silence instead of leaving an unexplained stop.
- Resolution: trainer hardened. The named green run is `runs/m3-gpu-fit-r1-eval-12800/result.json`, which
  records `nonfinite_forward 0 / nonfinite_loss 0 / nonfinite_grad 0` and completes evaluation from the
  12,800-step state. The chain itself was stopped deliberately on its own gate measurement
  (`docs/evidence/m3-gpu-fit-r1/chain-12800.md`), not by this crash.

## V-084 - CLOSED: pool rebuild time was read as a step-220 hang, and the bounded runner was blamed for it

- Symptom: three consecutive chain launches appeared to stop advancing at about step 220 with one host core
  pinned. The first two were attributed to `scripts/run_bounded.py`, which was replaced with a direct
  `timeout` invocation; the apparent stall then recurred anyway.
- Root cause: the logged step counter was read as a liveness signal for a process whose real per-step wall
  time was much larger than the 0.39 s/step it reported early on. Measured afterwards: building the 512-frame
  pool costs 10.6 s (17 ms/frame), so pool construction cannot account for multi-hour segments, and the same
  driver advanced to 12,800 steps across 13 successful segments with no hang. The gap between logged and
  effective step time is consistent with the accelerator being shared, which remains a hypothesis, not a
  measurement. (This entry's first version blamed pool construction; the 10.6 s measurement retracts that.)
- Detection class: progress inferred from a log counter without accounting for setup cost. First instance.
- Instrument: `--pool-cache` persists the built pool once and every log line and result now reports
  `pool_origin` ("built"/"cache") and `pool_size`, so setup cost is visible instead of inferred; the
  faulthandler watchdog separates silence from work.
- Resolution: pool cache added; the earlier "bounded supervisor spins at step 220" explanation is withdrawn in
  favour of this measurement. The bounded runner was not the cause.

## V-085 - CLOSED: the post-1,000-step collapse was a fixed-step random walk, not pool memorisation

- Symptom: on the frozen gate the fixed-pool arm peaked at 1,000 steps (development F1 0.8044 at best
  threshold) and then degraded (0.6373 at 12,600 with 4.52-cell geometry), training loss rising ~7 -> ~15,
  while exactness stayed at zero nonempty boards everywhere.
- Diagnosis: the training pool and the 36 training gate boards are the same frames (pixel-identical, max
  absolute difference 0.0), so the only held-out set is the 18 development boards, and those score *higher*
  than the seen ones (0.8044 vs 0.6611 at best threshold, geometry 1.0572 vs 1.0985). Memorisation is
  therefore excluded. The mechanism is `constant learning rate with permanently saturated gradient clipping`: every logged step exceeds the 1.0 clip (plain arm
  median norm 35.86, max 109.17; 100% clipped), so each update is `lr x unit direction` with a magnitude that
  cannot shrink near the optimum — loss stays flat while the swept best threshold wanders 4.5 logits.
- Detection class: a training curve diagnosed from end-point gate readings alone, without a trajectory or a
  seen-versus-held-out probe. First instance.
- Instrument: `--eval-every` writes a gate curve (both splits, and a seen-pool probe through the same
  `swept`/`map_cells` code) into every `result.json`, and each step log now carries the pre-clip
  `grad_norm`, so a flat loss with saturated gradients is visible rather than inferred.
- Resolution: cosine decay 1e-3 -> 1e-4. The arm exceeds the previous peak inside the run
  (0.8337 at step 1150) and ends at
  0.7919 with 0.5983-cell geometry, against
  0.7886 at 0.933 cells for the same
  starting point without decay. Evidence: `docs/evidence/m3-gpu-fit-r1/stability-1600.md`.
- Green gate: `runs/m3-gpu-fit-r1-armB-decay/result.json` (exits 0, `skipped_batches` all zero,
  `pool_origin` cache) plus the CPU suite: OK: 784 tests in 45.727 s, exit 0, run under scripts/run_bounded.py (receipt runs/stability-suite-r2-job.json, peak job commit 570,347,520 B)

## V-086 - CLOSED: the optimiser regime was never audited, and every committed step is clipped

- Symptom: all 40,796 recorded training steps across 90 committed runs exceed the 1.0 gradient clip
  (minimum pre-clip norm 1.657, median 23.89, maximum 1,785.62), so each update is `lr x unit direction`
  at a constant `lr 1e-3` and no run in the record contains an unclipped step.
- Root cause: `clip_grad_norm_(..., 1.)` runs on every step (`scripts/train_m2_pilot.py:171`,
  `scripts/train_m2_stream.py:842`, `scripts/accumulation_contract.py:81`) and the tree contains no
  learning-rate schedule, so the clip can never release as the model approaches an optimum.
- Detection class: a training regime read from endpoints rather than from its own per-step record, the
  class V-085 opened. This is the second instance, so the standing instrument is added before a third.
- Instrument: `scripts/stability_regime.py` with `tests/test_stability_regime.py` - a stdlib-only census
  of the `gradient_norm` field every curve already writes; `docs/M3_OPTIMISER_REGIME.md` carries the
  reading, what it does and does not license, and the ranked next step.
- Resolution: CLOSED as a measured **null**. The declared schedule lever
  (`docs/M3_OPTIMISER_SCHEDULE.md`, predeclared before any receipt) ran two matched 256-step arms from
  `9ae1fef9f649...` under 1,152 MiB / 1,800 s with zero added parameters. Per-cell F1 at the standing
  `0.0` point moved +0.005041 against the +0.05 floor, placement improved 39.9%, AUC
  0.8457 -> 0.8864, and the fit gate stayed 0 of 28 in both arms. The regime is now
  audited and is not the missing separation; the per-step `learning_rate` is recorded in every curve, so
  the census instrument can read any later arm's schedule. Evidence:
  `docs/evidence/m3-optimiser-schedule-r1/scope.json`.

## V-087 - CLOSED: a read-only headroom pass defaulted to the narrow deciding population

- Symptom: the first execution of `scripts/ensemble_headroom.py` produced a complete receipt on 3 groups /
  9 cases / 7 nonempty boards while its predeclaration names the canonical 12 groups / 36 cases / 28
  nonempty boards.
- Root cause: the script inherited the sweep helper's convention that `--training-groups` is optional, so
  the decisive population was supplied by protocol memory rather than by the declaration; the widened
  population was never requested and the self-check compared the pass against itself.
- Detection class: predeclaration-versus-receipt population mismatch (declared scope silently narrowed).
- Instrument: `--training-groups` is now **required** and cannot default; the admission profile and the
  scope both record the evaluated group count; the off-declaration narrow run and the declared-population
  pre-guard run are both preserved with their receipts (`docs/evidence/m3-deciding-headroom-narrow-r1/`,
  `docs/evidence/m3-deciding-headroom-preguard-r1/`) rather than overwritten.
- Named green run: `docs/evidence/m3-deciding-headroom-r1/scope.json` (12 groups / 36 cases / 28 nonempty
  boards, 108 forwards, peak 635,166,720 B, verdict `bias_limited`).

## V-088 - CLOSED: the derived checkpoint identity rejected the legacy configs it must keep naming

- Symptom: the first full-suite run after the trunk-width plumbing failed 7 tests across
  `tests/test_stream_evidence.py` and `tests/test_target_exposure.py` with `KeyError: 'widths'`.
- Root cause: `checkpoint_architecture` derived the identity with `ModelConfig(**summary["config"])` and
  required a `widths` field unconditionally, while the evidence contracts accept manifests whose recorded
  config names only `dimension`. Derivation now runs only when the trunk shape is actually recorded: a
  widened configuration still cannot be reported under a published name, and a legacy snippet keeps the
  name it was written under.
- Detection class: a new identity guard rejecting the older manifests it was meant to preserve
  (guard-too-wide), caught by the existing evidence suite before any receipt was written.
- Instrument: the existing `tests/test_stream_evidence.py` and `tests/test_target_exposure.py` fixtures,
  plus `tests/test_width_lever.py` pinning the five predeclared verdict branches so the read cannot choose
  a branch after seeing numbers.
- Named green run: `runs/suite-falsification-r2-job.json` - 826 tests OK, job passed, 93.9 s, peak
  569,962,496 B, and `runs/identity-fix-tests-r1-job.json` (33 focused tests) as the targeted rerun.

## V-089 - CLOSED: probe read-outs sliced a single-channel cell map with a hardcoded `:20`

- Symptom: none visible in the current arms, but `output["final"]["occupancy"]` is
  `(batch, visible_rows+spawn_rows, columns)` and three read instruments sliced it with a literal `[:, :20]`,
  so a structural change to the board height or the spawn margin would have truncated the read silently while
  the receipt still claimed a full-board read.
- Root cause: the literal encoded the published visible-row count instead of the config field that defines it;
  because rows are bottom-first, the published shape made the literal correct and hid the coupling.
- Detection class: a hardcoded assumption shadowing a declared/config field - the third instance in this
  campaign after V-087 (narrow-population default) and V-088 (identity guard wider than its contract), so the
  standing instrument is an assertion derived from the declaration, not another literal.
- Instrument: `visible_board_logits(occupancy, config)` sizes the window from the verified config and refuses
  any map that is not `(1, visible_rows+spawn_rows, columns)`; `readout_statistics` records the shape and
  per-block logits in every receipt; `tests/test_width_lever.py` covers the slice, a 26-visible-row config the
  literal would have got wrong, and the refusals.
- Named green run: `docs/evidence/m3-trunkwidth-r2/result.json` reproduces r1 exactly;
  `runs/readout-tests-r1-job.json` (37 focused tests) and `runs/suite-falsification-r2-job.json`
  (830 tests, passed, no orphan child).
- Resolution: the coupling is cleared across every consumer - `src/tetris_vision/losses.py`
  (`stage_loss` and `diagnostic_metrics`), `scripts/train_m2_stream.py` (the fit-policy read),
  `scripts/ensemble_headroom.py`, `scripts/sweep_stream_occupancy.py`, `scripts/probe_local_reading.py`,
  `scripts/probe_stream_errors.py` and `scripts/analyze_m2_pilot.py` all size the window through
  `visible_window`/`board_window`, which refuse a map that cannot cover the declared or labelled board.
  `tests/test_visible_board.py` fails if a literal row slice or a wrong-shape map returns. Proof that no
  reported number moved: the base arm's published development evaluation reproduces to the digit under both
  the working tree and its frozen pre-change archive (`docs/evidence/visibleboard-proof-current-r1`,
  `docs/evidence/visibleboard-proof-archive-r1`) - loss 18.55253728230794, cell accuracy 0.8986111150847541,
  geometry 2.192851414283117, 0 exact boards out of 18 images and 6 ancestry groups.
- Consumer-level proof: each non-loss consumer now exposes its own module-level entry point -
  `ensemble_headroom.visible_units`, `sweep_stream_occupancy.sweep_logits`, `probe_local_reading.reading_map`,
  `analyze_m2_pilot.pilot_maps`, `probe_stream_errors.pack` and `train_m2_stream.fit_correct` - and
  `tests/test_consumer_pins.py` drives every one of those call sites with a 26-visible-row declaration and with
  a shape-mismatched map, so a literal reintroduced in any single file fails that file's own named test.
  `scripts/ensemble_headroom.py` was then re-run on its published arms over the declared 12-group population
  and reproduces its receipt exactly: `docs/evidence/m3-deciding-headroom-rerun-r1/scope.json` equals
  `docs/evidence/m3-deciding-headroom-r1/scope.json` on arms, members, ensemble, verdict, population, grid,
  agreement and weights, with only source digests and timing differing (verdict `bias_limited`, ensemble AUC
  0.8871656812988195, best F1 0.5074050225370251).
- Frozen-fixture decision: `tests/fixtures/m2_original/losses.py` keeps its original `[:, :20]` on purpose. It
  is frozen evidence for byte-compatibility with the published M2 model, not a consumer, and
  `test_the_frozen_m2_reference_fixture_keeps_its_original_literal` fails if anyone edits it - so the scan
  exclusion is deliberate and pinned rather than a silent hole.

- Instrument replays (2026-11-11): all three converted non-loss replay instruments now run end to end under
  the published config on their saved populations and reproduce their historical receipts field by field.
  `ensemble_headroom.py` on its published arms (`docs/evidence/m3-deciding-headroom-rerun-r1`);
  `sweep_stream_occupancy.py` on `m3-occbref-baseline-r1` (`runs/m3-sweep-obcref-rerun-r1/result.json` - grid,
  deciding and development sweeps, verdict `uninformative`, training population, cases and weights identical);
  `probe_stream_errors.py` on `m3-fit-occupancy-r1` (`docs/evidence/m3-fit-occupancy-r1-errors-r2/result.json` -
  identical on run, weights, config, architecture, totals and cases). The only field-level differences are two
  fields that `m3-fit-occupancy-r1-errors-r1` predates (`occupancy_positive_weight` and `training_population`,
  both added later with default values), plus source digests and timing everywhere. Published attribution, F1
  and threshold numbers are byte-identical.

## V-090 - CLOSED: the derived-identity import broke every replay instrument's provenance guard

- Symptom: the first consumer-level rerun of `scripts/ensemble_headroom.py` failed with
  `headroom model/label implementation is not frozen` before reading a single case.
- Root cause: `checkpoint_architecture` imports `tetris_vision.model` to derive the architecture name, and the
  replay instruments call it before inserting their frozen archive on `sys.path`. That cached the working
  tree's `tetris_vision`, so the frozen-source check no longer saw the archive and refused. No instrument had
  been re-run since that change, so nothing had exercised the guard.
- Detection class: a provenance guard defeated by import order - the same class as the width read's first run.
- Instrument: `ensemble_headroom.py`, `sweep_stream_occupancy.py` and `probe_stream_errors.py` purge
  working-tree modules before the path insert, with the reason recorded at the purge site.
- Durable fix (2026-11-11): architecture identity no longer depends on `sys.path` order. `checkpoint_architecture`
  resolves the rule through `working_tree_model()`, which loads the present `src/tetris_vision` package by path
  under a private name (`tetris_vision_identity`), so an archive-era `model.py` shadowing the path can neither
  supply nor withhold the derived-name helper. That replaces the per-consumer ordering patch: the errors probe's
  single `checkpoint_architecture(summary, manifest)` call after the archive insert is restored and now works.
  `tests/test_identity_import_order.py` shadows the model module with an archive-era one that has no
  `architecture_name`, asserts a direct import fails there, and then requires the identity to stay correct - a
  check the old call ordering could not pass. All three replay instruments were re-run afterwards and still
  reproduce their receipts exactly (`docs/evidence/m3-deciding-headroom-rerun-r2`,
  `runs/m3-sweep-obcref-rerun-r2`, `docs/evidence/m3-fit-occupancy-r1-errors-r3`).
- Named green run: `docs/evidence/m3-deciding-headroom-rerun-r1/scope.json`, identical to the historical
  receipt, under `runs/m3-deciding-headroom-rerun-r1-job.json`.

## V-091 - CLOSED: an in-process module purge in a new test corrupted VisionState identity downstream

- Symptom: the first full suite containing `tests/test_identity_import_order.py` failed three tests in
  `test_layer_evidence` and `test_model_capture` with `TypeError: validated VisionState required`, while each of
  those tests passed in isolation.
- Root cause: to simulate the archive shadow, that test deleted every `tetris_vision.*` entry from
  `sys.modules`; modules imported before the purge kept the old `VisionState` class object while the reimported
  `component_targets` compared against a new one, so a valid state failed the `isinstance` check.
- Detection class: test instrument (module-identity pollution between tests), not product code.
- Instrument: the shadow scenario now runs in a fresh interpreter with the archive-era package first on
  `sys.path`, which reproduces the real path race without touching the suite's module identity.
- Named green run: `tests/test_identity_import_order.py` (3 tests OK, `runs/identity-order-tests-r2-job.json`)
  and the full suite at 852 tests OK (`runs/suite-falsification-r2-job.json`).

## V-092 - CLOSED: the replay "empty difference" compared projections and could not fail per field

- Symptom: every replay was declared identical because an ad-hoc comparison printed `differing keys []`. That
  output was not evidence each protected field had been compared: the headroom comparison read only
  `scope.json`, which has no `cases`, so "cases equal" was never tested there; the error comparison excluded
  `occupancy_positive_weight` and `training_population` by name, so a drift in either would have been invisible;
  and the sweep's missing `attribution` was never stated.
- Root cause: comparisons were written per replay over a union of keys with `.get()` on both sides, so a field
  present on one side only compared as `None`, an excluded field was unguarded, and a projection of the receipt
  was treated as the receipt. No mutation ever showed the comparison could report a difference.
- Detection class: evidence instrument (an equivalence test with no failing control), not model or training code.
- Instrument: `scripts/replay_equivalence.py` declares a carrier for each of the seven protected fields per
  instrument, treats an absent carrier as a difference, and reports declared-absent fields with their reason
  instead of skipping them. `tests/test_replay_equivalence.py` perturbs every carried field in a copy of a real
  receipt and requires the comparison to name exactly that field, requires the unmodified pair to compare clean,
  and requires every protected field to be carried by at least one instrument. The two later-added fields are
  compared against the run's recorded defaults, verified against `runs/m3-fit-occupancy-r1/summary.json` and its
  nine training rows rather than assumed.
- Named green run: 7 tests OK, full suite at 859 tests (`runs/suite-falsification-r2-job.json`), and all three
  replays re-run under the strict differ compare identical on every carried field
  (`docs/evidence/m3-deciding-headroom-rerun-r3`, `runs/m3-sweep-obcref-rerun-r3`,
  `docs/evidence/m3-fit-occupancy-r1-errors-r4`).

## V-093 - CLOSED: the first differ was strict about values but tolerant about types, sections and absence

- Symptom: the differ rejected one large perturbation per field, so an empty difference looked decisive. Near
  misses were not: `1 == 1.0` and `0 == False` compared equal, so a type drift in any numeric or boolean carrier
  was invisible; a NaN injected into a numeric field raised while serialising instead of naming that field; the
  error F1 carrier divided its counts, so an int-to-float drift in the counts cancelled out; undeclared top-level
  sections were never inspected, so a newly added receipt section shipped unguarded; and the "absent" fields had
  no failing guard, so an instrument that started recording `attribution` or `threshold` would still pass.
- Root cause: the comparison used Python `==` on extracted values (type-coercing), serialised cases to one JSON
  string (no localisation, no NaN handling), only walked the seven carriers, and described absences in prose
  rather than asserting them. The deciding-power figure was hand-written prose with nothing tying it to the
  receipt.
- Detection class: evidence instrument (an equivalence test whose control was too coarse and whose scope was
  narrower than its claim), not model or training code.
- Instrument: `scripts/replay_equivalence.py` now compares with `first_difference`, which requires identical
  types (bool/int/float distinct), identical key sets, identical lengths and order, NaN only equal to NaN and
  infinities by sign. The error F1 carrier includes the counts it is computed from. Every top-level section must
  be a carrier root, a supporting section, an identity key, a documented default or declared metadata, else the
  comparison fails as `unaccounted:<key>`. Declared-absent fields name the keys that would prove the instrument
  started recording them, and their appearance is a difference. `tests/test_replay_equivalence.py` attacks every
  carrier with ULP changes, 1e-12 nudges, type substitutions, NaN, infinity, dropped/duplicated cases,
  one-sided keys and nested edits, and pins case order as sensitive and map order as insensitive.
  `tests/test_deciding_power_pin.py` recomputes the power record from the receipt it names and requires the
  documented sentence to be the record's own.
- Named green run: 10 replay-equivalence tests and 4 power-pin tests OK; full suite green (see
  `runs/suite-falsification-r2-job.json`); all three instruments re-run after the change
  (`docs/evidence/m3-deciding-headroom-rerun-r4`, `runs/m3-sweep-obcref-rerun-r4`,
  `docs/evidence/m3-fit-occupancy-r1-errors-r5`) and compare identical on every carried field with published
  attribution/F1/threshold bytes unchanged.

## V-094 - CLOSED: two receipts had no guard at all, two pins were brittle instead of protective

- Symptom: the falsification pass built scratch copies of every published receipt. The three instrument receipts
  and the deciding-power record rejected the six corruption classes, but the green suite receipt had no validator
  whatsoever - any passing-looking JSON was accepted - and the power record only checked the keys it knew, so an
  injected section passed silently. In the other direction, two irrelevant edits failed: reflowing the documented
  figure's whitespace broke the pin, and adding a docstring to the power instrument broke its script-digest pin.
- Root cause: the suite receipt was treated as an artifact rather than a claim, and the power record's validator
  was a fixed key list; the prose pin compared raw substrings and the record pinned its own source digest, which
  is configuration, not evidence.
- Detection class: evidence instrument (guards with no adversarial control), not model or training code.
- Instrument: `record_stream_run.job_receipt_problems` validates status, exit code, elapsed, ceiling and peak,
  command shape, thread policy and pressure reason, and rejects unknown or missing keys;
  `sweep_deciding_power.record_problems` recomputes the record from the receipt it names, checks the evidence
  digest and rejects injected sections; `sweep_deciding_power.figure_appears` matches the documented figure with
  whitespace collapsed; the script-digest pin is gone because recomputation already catches behaviour.
  `tests/test_evidence_falsification.py` builds the corrupted and irrelevant copies and runs these checks.
- Named green run: full suite green (`runs/suite-falsification-r2-job.json`), all three instrument receipts re-replayed after the last code
  change with published bytes unchanged.
