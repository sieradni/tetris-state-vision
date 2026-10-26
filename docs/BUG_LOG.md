# Defect ledger

Historical tetrguide issues are design
constraints in ROADMAP.md, not fixes claimed by this repository.

Every new fix entry needs symptom, root cause, detection class, instrument and
a named green run. An unrun gate leaves the entry OPEN.

| Detection class | Entries | Standing instrument |
|---|---|---|
| visibility-contract | V-001, V-002, V-005, V-006 (4) | Paired-raster equivalence required for every hidden-component policy; AGENTS.md and test_layout.py |
| provenance | V-003 (1) | Computed hashes and byte comparison |
| test-instrument | V-004, V-011 (2) | Native admission/raw counters; resident optimizer-state backward characterization |
| resource-identity | V-007 (1) | Exact named Job plus kernel membership/readback |
| corpus-admission | V-008, V-009 (2) | Malformed/duplicate records and off-image known-cell fixtures |
| resource-pressure | V-010 (1) | OS commit admission and pressure/failure process-tree tripwires |

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
