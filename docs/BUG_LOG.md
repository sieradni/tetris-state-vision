# Defect ledger

Historical tetrguide issues are design
constraints in ROADMAP.md, not fixes claimed by this repository.

Every new fix entry needs symptom, root cause, detection class, instrument and
a named green run. An unrun gate leaves the entry OPEN.

| Detection class | Entries | Standing instrument |
|---|---|---|
| visibility-contract | V-001, V-002, V-005, V-006, V-016 (5) | Paired-raster equivalence required for every hidden-component policy; AGENTS.md, test_layout.py and M3 prepared-source admission tripwire |
| provenance | V-003, V-014, V-018, V-020, V-021, V-022, V-024, V-026, V-027, V-028, V-034, V-035 (12) | Standing artifact-derived identity rule in AGENTS; computed hashes/byte comparison, frozen execution, immutable controller origin, unwrapped callable origin, actual receipt status and checkpoint architecture/config/aggregate tripwire; saved text identities checked against actual wire bytes and JSON roundtrip; profile ancestry must match actual profile recipes; a staged start must be an exact sibling run re-read from source bytes and a staged resume must restate that contract |
| test-instrument | V-004, V-011, V-012, V-017, V-019, V-023, V-025, V-037, V-049, V-057, V-058 (11) | Native admission/raw counters; warm backward; standing pre-import CPU-library policy/live child tripwire; observer-validated fixtures before model gates; named queue anchor flags; raw versus normalized evidence-copy tripwire; a fixture must derive archived supervisor identity from the bytes it copies |
| resource-identity | V-007 (1) | Exact named Job plus kernel membership/readback |
| corpus-admission | V-008, V-009 (2) | Malformed/duplicate records and off-image known-cell fixtures |
| resource-pressure | V-010 (1) | OS commit admission and pressure/failure process-tree tripwires |
| resource-profile | V-013, V-039, V-040, V-043, V-045, V-046 (6) | Shape-diverse native memory trajectory, bounded native cache and resumable finite runs; a declared read-out mode must not carry trainable heads it never uses; a page-file host's steady-state commit charge is not this job's risk. Standing instrument: `PressureDecisionTests` plus the synthetic-exhaustion cases in `test_pressure_guard.py`. |
| windows-persistence | V-015 (1) | Bounded sharing-error retry, prior-pointer preservation and fail-closed launch |
| objective-provenance | V-051, V-055 (2) | The probe and curator bind the objective from `arguments` and reproduce saved losses; a regression test asserts the objective's branches are charged and refuse a mismatched join |
| launch-ownership | V-029, V-030, V-050 (3) | Preflight output, receipt and both log paths before detached launch; verify the child executable path resolves before arming the Job; a supervisor killed by its caller leaves a `running` receipt, so the retry preserves it and the partial output before re-running |
| evidence-reporting | V-032, V-033, V-036, V-038, V-041, V-042, V-044, V-047, V-048, V-053, V-054 (11) | Evidence-folder audit instrument that re-derives every published number from the committed bytes and fails on any disagreement; a correctness count is reported next to the classes the head actually emits, so a constant predictor cannot read as identity reading; the ordinary replay completes deterministic development evidence to the canonical group set and the curator re-derives that extension; the replay scores the objective the run actually recorded and the curator binds that identity; an evidence directory is an atomic record keyed by name, so a frozen run's directory cannot gain files |
| interval-validity | V-056 (1) | Measure the ICC and design effect of the deciding cells and report a whole-cluster resampling interval beside every cell-level interval, so a nested population cannot be reported with an independent-sample width |

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
