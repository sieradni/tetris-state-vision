# Defect ledger

Historical tetrguide issues are design
constraints in ROADMAP.md, not fixes claimed by this repository.

Every new fix entry needs symptom, root cause, detection class, instrument and
a named green run. An unrun gate leaves the entry OPEN.

| Detection class | Entries | Standing instrument |
|---|---|---|
| visibility-contract | V-001, V-002, V-005, V-006, V-016 (5) | Paired-raster equivalence required for every hidden-component policy; AGENTS.md, test_layout.py and M3 prepared-source admission tripwire |
| provenance | V-003, V-014, V-018, V-020, V-021, V-022, V-024 (7) | Standing artifact-derived identity rule in AGENTS; computed hashes/byte comparison, frozen execution, immutable controller origin, unwrapped callable origin, actual receipt status and checkpoint architecture/config/aggregate tripwire |
| test-instrument | V-004, V-011, V-012, V-017, V-019, V-023 (6) | Native admission/raw counters; warm backward; standing pre-import CPU-library policy/live child tripwire; validated fixtures before model gates; named queue anchor flags; raw versus normalized evidence-copy tripwire |
| resource-identity | V-007 (1) | Exact named Job plus kernel membership/readback |
| corpus-admission | V-008, V-009 (2) | Malformed/duplicate records and off-image known-cell fixtures |
| resource-pressure | V-010 (1) | OS commit admission and pressure/failure process-tree tripwires |
| resource-profile | V-013 (1) | Shape-diverse native memory trajectory, bounded native cache and resumable finite runs |
| windows-persistence | V-015 (1) | Bounded sharing-error retry, prior-pointer preservation and fail-closed launch |

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
