# Contributor handoff — 2026-10-30

## Initial tensor admission — 2026-11-07

The trainer now saves and reads back actual pre-evaluation initial state tensors;
their file identity is checked before completion and by independent curators.
Focused tests passed. Candidate gate r9 is planned with explicit output/receipt/
log preflight and free-disk check: parent 1024 MiB / 600 s / 14 updates,
child 640 MiB / 90 s, CPU1/cache16 and existing pressure guards, expected
under 800 MiB. Command `scripts/probe_stream_recovery.py --output
runs/m2-stream-recovery-gate-r9 --curriculum all-class-fit-v1
--type-head-rate-factor 4`; receipt/log prefix `runs/m2-stream-recovery-gate-r9`.
It performs profile2 then admitted continuous6 and partial3+resume3, not the
1,024-update study. Control r10 requires a separate same-source gate.
r9 passed 147.947 s / 599,404,544 bytes; terminal receipt and all matching
processes must be checked before control. Control r10 uses the same limits,
ordered curriculum, factor1 and namespace/disk preflight, expected under
800 MiB. Receipt, output and logs use `runs/m2-stream-recovery-gate-r10`.
r10 passed 142.762 s / 600,817,664 bytes, pending process-death checks.
Curate r9 and r10 sequentially at 128 MiB / 30 s / 181 items, expected under
100 MiB with CPU1 and existing guards; use new `-curation` receipt/log prefixes
and preflight them. Then compare actual initial tensors with a separate 640 MiB /
30 s / two-snapshot collector, expected under 400 MiB. No full fit yet.
r9/r10 curators passed, each replaying 17 records and 164 saved probes. The
matched r2 collector verified actual initial tensor bytes identical, initial
probe metrics, first loss/gradient norm, six data recipes and optimizer groups;
2.259 s / 257,298,432 bytes under 640 MiB. Source catalogue 18ce798e...;
evidence `docs/evidence/type-head-initial-admission-r2/` and gate r9/r10 scopes.
The finite recovery and start-comparison admissions pass; full-study initial
artifacts still need comparison to these starts before final claims.

## Joint type-head optimization preparation — 2026-11-07

The predeclared comparison is `docs/TYPE_HEAD_JOINT_OPTIMIZATION.md`. An opt-in
fourfold type-head optimizer factor is implemented with default single-group
behavior preserved, profile/recovery identity bound, and segment routing guarded.
Focused optimizer/admission/routing checks passed; these do not admit training.

Candidate recovery admission r6 planned: `scripts/probe_stream_recovery.py
--output runs/m2-stream-recovery-gate-r6 --curriculum all-class-fit-v1
--type-head-rate-factor 4`, parent Windows Job 1024 MiB / 600 s / 14 updates,
serialized children each 640 MiB / 90 s, CPU1/cache16 and unchanged OS
pressure guard. Expected aggregate under 800 MiB from prior pilot gate; first
two updates are the full-square warm profile, followed only if admitted by
continuous6 and partial3+resume3. Receipt `runs/m2-stream-recovery-gate-r6-job.json`,
logs same prefix `.log`/`.stderr.log`, progress in the output directory. More
than 40 GB free was checked. This gate does not start the 1,024-update study.
Correction: r6 already existed. The runner refused its existing receipt before
work; redirect opened historical local log names. Failed-attempt copies are
`runs/type-head-recovery-name-collision{,.stderr}.log`; historical local logs
were restored byte-for-byte from `docs/evidence/m2-stream-recovery-gate-r6/`.
See V-029. Use r7 only after checking all output, receipt and log names absent;
same command, limits, output/log/receipt prefix, and work cap as above.
r7 passed 57.769 s / 602,124,288 bytes, exact continuous6 == partial3+resume3,
profile child peak 558,534,656 bytes under 640 MiB; all matching processes
gone. Independent curator planned under 128 MiB / 30 s / 181 replay items,
expected under 100 MiB, same CPU/pressure guards. Receipt/log prefix
`runs/m2-stream-recovery-gate-r7-curation`.
r7 curator passed 2.059 s / 87,764,992 bytes; replayed 17 records and 164
saved probes, evidence `docs/evidence/m2-stream-recovery-gate-r7/`. Control r8
planned using the same gate, ordered curriculum and factor1, parent 1024 MiB /
600 s / 14 updates, serialized child 640 MiB / 90 s, expected under 800 MiB,
CPU1/cache16 and unchanged pressure guard. Preflight output, receipt and both
log paths; more than 40 GB free must remain. This is not the 1,024-update fit.
r8 passed 118.063 s / 600,768,512 bytes; matching processes terminated.
Control curator planned under 128 MiB / 30 s / 181 items, expected under
100 MiB with the same guard, receipt/log prefix
`runs/m2-stream-recovery-gate-r8-curation`. Match comparator planned at
640 MiB / 30 s / two snapshots, expected under 400 MiB, after curation.
r8 curator passed 2.414 s / 88,158,208 bytes; 17 records/164 images replayed.
Matched first-step comparator passed 2.219 s / 256,782,336 bytes; initial
probe metrics and first loss/gradient norm match, non-head first-step weights
match and group rates are .001 versus .004 for the candidate head. Exact
initial tensor bytes were not recorded. The predeclared identical-initial-
weights criterion remains open; add source-bound capture/verification and
repeat affected admissions before launching any 1,024-update study. Receipts:
`docs/evidence/type-head-admission-contracts-r1/`. No learning claim follows.

## Pixel replay work in progress — 2026-11-07

The conditional pixel replay and independent collectors are complete; see
`docs/CONDITIONAL_PIXEL_REPLAY.md`. Both joint-fit gates fail and both development
type scores remain 0/18. The next research action is to predeclare a controlled
joint-optimization comparison, including resource and recovery admission; no
new training is admitted yet. `scripts/pixel_replay_contract.py`
adds a strict byte comparison of every non-type output at each stage and final
output, with only the designated stage type logits exempted. Three focused
checks pass under the bounded CPU runner; receipt and log are in
`docs/evidence/pixel-replay-contract-r2/` (peak 246,312,960 bytes, 640 MiB cap,
30 seconds, work 3). This is a comparison instrument, not a pixel replay or
model-quality result on its own. The original runs remain frozen.

Ordered replay attempt r1 planned: `docs/experiments/conditional_pixel_replay.py
--study ordered --output runs/conditional-pixel-ordered-r1`, supervised by
`scripts/run_bounded.py` at 640 MiB / 120 s / 82 forwards; expected under
600 MiB, CPU1/cache16, Windows Job and existing pressure guard. Receipt
`runs/conditional-pixel-ordered-r1-job.json`, stdout/stderr same prefix with
`.log`/`.stderr.log`. This is an experimental attempt, not an audited result.
Ordered r1 terminated passed at 12.606 s / 426,795,008 bytes; all 82 forwards,
saved feature and original-report checks pass; independent audit remains open.
All startup processes 8856/23544/8412/17196 are gone. Interleaved r1 planned
with the same guard, 640 MiB / 120 s / 82 forwards, expected under 600 MiB:
`docs/experiments/conditional_pixel_replay.py --study interleaved --output
runs/conditional-pixel-interleaved-r1`, receipt and logs under that prefix.
Interleaved r1 passed 11.734 s / 426,926,080 bytes; startup processes
2808/22148/12384/4840 gone. Independent collectors planned sequentially:
`docs/experiments/record_conditional_pixel_replay.py --study STUDY`, 41 cases,
256 MiB / 30 s, expected under 150 MiB, CPU1 and unchanged pressure guard.
Receipts and logs use `runs/conditional-pixel-STUDY-curation-r1` prefix.
Both collectors passed, with raw evidence under `docs/evidence/conditional-pixel-
{ordered,interleaved}-r1/`. All known replay and collector processes exited.

## Owner request and delivery

The owner asked for a clean project, a repository under `sieradni`, a ZIP to send
and a document of the core goals. This is the active delivery task. No further
research training is selected during handoff. No research jobs are running.

- Repository: https://github.com/sieradni/tetris-state-vision (PRIVATE; owner/account verified).
- Core brief: docs/CORE_GOALS.md. Contributor workflow/setup: CONTRIBUTING.md.
- The incoming contributor's next research task is conditional pixel replay below.
- GitHub holds source/history; the ZIP is a snapshot of the same source commit
  plus current ignored model/run artifacts. ZIP extraction alone creates no remote.
- No collaborator was invited because their GitHub username was not supplied.
  The owner must grant access before that person can clone/push a branch.
- Original screenshots, private contact sheets, `.venv`, `.git`, the neighboring
  engine and unselected historical runs are excluded. Original local data remains.
- Bundle contains current image runs, required warm admissions, fitted heads,
  saved synthetic probes, exact frozen sources and one latest optimizer generation
  per selected run. The archive manifest is the exact file/commit inventory.

Local research results were committed at64939f2, then the contributor tooling/docs
at e3312ef. Six transport tests passed:7.743s(native8.231s/120,905,728B under128MiB).
They verify integrity,latest snapshot selection,non-destructive/idempotent restore,
source preservation,conflict preflight and unsafe/duplicate members. Source-bound
raw evidence:docs/evidence/handoff-contracts-r1/. This is not a whole-suite gate.
Actual artifact preflight passed .736s/74,731,520B:925 artifact files/313,244,976B,
19 explicit roots. Counts are preliminary;final archive manifest is authoritative.
Private repository creation passed6.059s/138,231,808B under256MiB/90s. Raw receipts,
logs,startup/death and verified owner/visibility:docs/evidence/handoff-preflight/.

Export verification is intentionally outside the source commit to avoid a
self-referential archive hash. Durable owner-side jobs/results are in ignored
runs/handoff*/ and runs/handoff/. Final ZIP,SHA256 and adjacent verification JSON
are delivered in Downloads. They name the exact exported Git commit. The final
acceptance is local main==origin/main==GitHub main==ZIP source commit,clean source,
verified every archive file,restoration into a fresh GitHub clone,latest optimizer
payload hashes,relocated original-model pixel smoke for all3models and copied-head
vector replay. The pinned existing runtime is reused;no fresh install or full
conditional pixel intervention is claimed by that transport check.

Before each export job: state expected RAM/cap/guard,verify actual processes,
record receipt/log/command and inspect terminal status. No force pushes or
history rewrites. Local main is advanced only by a normal fast-forward merge.
Current one-off publisher:runs/handoff/export_remote.py (no credential output).
Push:expected<300MiB,512MiB/180s/work1,unchangedpressureguard:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 512 --wall-seconds 180
--work-items 1 --receipt runs/handoff-push-r1-job.json --
.venv/Scripts/python.exe runs/handoff/export_remote.py push
Logs sameprefix.log/.stderr.log;result runs/handoff/push-result.json.
Build:expected<200MiB,256MiB/180s/work12000:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 256 --wall-seconds 180
--work-items 12000 --receipt runs/handoff-build-r1-job.json --
.venv/Scripts/python.exe scripts/handoff_bundle.py build --zip ZIP_IN_DOWNLOADS
Logs sameprefix.log/.stderr.log;ZIP plus SHA256 and native receipt are sentinels.
Relocation:expected<800MiB,1024MiB/240s/work12000:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 1024 --wall-seconds 240
--work-items 12000 --receipt runs/handoff-verify-r1-job.json --
.venv/Scripts/python.exe scripts/verify_handoff_checkout.py --zip ZIP_IN_DOWNLOADS
--checkout runs/handoff-check-COMMIT_PREFIX
Logs sameprefix.log/.stderr.log;adjacent *.verification.json is result sentinel.
Do not launch duplicate export operations because a read/poll timed out.
Exact post-commit export paths/command/PID/log/limits/results are recorded before
launch in ignored runs/handoff/export-state.json. This fixed source checkpoint
points there so the delivered commit need not contain its own later ZIP checksum.
Before final source commit:local verification-script check,expected<800MiB,
1024MiB/60s/work167(3ordinarypixels+2x82headvectors),no training/sourcewrites.
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 1024 --wall-seconds 60
--work-items 167 --receipt runs/handoff-local-smoke-r1-job.json --
.venv/Scripts/python.exe runs/handoff/local_smoke.py
Hiddenlogs sameprefix.log/.stderr.log;native receipt terminal sentinel.
Local verifier smoke PASS12.893s/525,479,936B under1GiB/60s/work167:all3frozen
model origins and original pixel logits exact,both82-row fitted-head replays
exact. Actualsource/rawreceipt/log/startup/death in docs/evidence/handoff-local-smoke-r1/.
This verifies the program locally;full fresh-clone ZIP verification is separate.

## Incoming contributor: selected next experiment

Read AGENTS.md,docs/CORE_GOALS.md,docs/ROADMAP.md,docs/TYPE_HEAD_FIT.md and BUG_LOG.md.
There is no validated complete screenshot parser. The original4096baseline and
both1024controlled image studies remain complete/frozen and fail adequate/joint
learning. API remains needs_review/unsupported. The full release goal stays open.

Implement conditional pixel replay for the ordered and interleaved copied raw
float32 heads. No additional image-model training or vector-fit extension is
selected. Acceptance:

1. Verify curated model/feature/headfit source,config,weights,payload and hashes.
2. Replay all41saved rasters/public targets per model;original ordinary metrics
   and per-image type-head inputs/logits must match independent saved reports.
3. Replace only layer_type weights/bias in a MEMORY COPY. Pixel/validity only
   to model.forward. Exactly2forwards/image=82/model;no oracle geometry/labels.
4. Require every non-type tensor at EVERY stage to remain bitwise equal and
   final head features to remain exact. Record honest float32 matrix-product
   roundoff versus vector-batch logits;actual class counts must match headfit.
5. Restore original head in finally/percase,verify full model state/source/file
   identity. Never overwrite original model/checkpoint/optimizer/data.
6. Independently derive/audit type,pose,absence,locked,geometry and joint-fit
   metrics from the actual changed pixel outputs. Keep the intervention separate
   from original ordinary reports and historical study tags.

Expected inference<600MiB based on measured425MBobservers;enforce640MiB/120s/
82forwards,CPU1/cache16,existing pressure guard. Declare/bound an independent
collector separately. Implementation and those conditional runs are still pending.
Transport smoke performs just1ORIGINAL pixel forward/model;it does not meet this gate.

## Latest verified findings and artifact map

TYPE_HEAD_FIT.md: both finite copied-head fits complete at1024updates with per-bank
native warm/reload admission and independent collectors. Original image models,
source and optimizer files unchanged. New diagnostic optimizer is explicitly reset;
fullbatch64known vectors,AdamWLR.01/decay0,training-onlymean/popstd floor1e-6,
float64 conditioning,converted shared8x64+bias rawfloat32 head. Initial per-image
float32 logits exact;first2fit records match warm prefix;head/Adam reload exact.

- Ordered types20/64->62/64(active32,ghost30):64/64 criterionFAILS. Remaining two
  errors:T->S,both views of all-class-04(one ancestry). Dev0/18;CE3.279->58.590.
- Interleaved types37/64->64/64(active32,ghost32):vector criterionPASSES.
  Dev0/18;CE4.933->25.486. Rawfloat64/float32 counts agree.
- This proves representation on these saved training vectors,not image reading
  or generalization,and does not identify the cause of joint-training failure.

Evidence/runtime names:type-head-{ordered,interleaved}-{profile,fit}-r1.
Profiles:2warm+continuousnext1+reloadednext1(work4),640MiB/60s.
Fits:1024updates,640MiB/120s. Peak333MBboth;native~3s;independent collectors
~.35-.4s/78-80MB under128MiB. All known PIDs/commands gone;raw source/payload/log/
receipt/startup/death scopes in docs/evidence/<name>/. Six pure contracts in
handoff source:docs/evidence/type-head-contracts-r1/. No new defects;V001..V028
closed on named receipts. Do not attach those green scopes to changed sources.

Ordered model:runs/m2-stream-allclass1024-s3;interleaved:runs/m2-stream-interleave1024-s3.
Fitted payloads:runs/type-head-{ordered,interleaved}-fit-r1/head-state.pt.
Raw32values are in result.json/head.weight_raw32,bias_raw32;rederive from the
normalized payload+mean/scale and verify. Curated inputs_sha256 carries model,
feature and source identities;do not hand-rewrite hashes/absolute historical paths.
Payload SHA ordered6749fa5c64c5f6e5dce9fd820c9f1b94854ed2bc14abea931b10fb79545fc696;
interleaved58b417a8c50e434462c5d728c2828b564b33d74c3bdf98391cb7d178e66f86e9.

INTERLEAVED_FIT.md original image studies:ordered32/32boards,12/56visible roles,
8/8absences,geometry.21059;interleaved28/32boards,30/56visible roles,7/8absences,
geometry.38618. BothFAIL joint32boards/64roles/128semantics/geometry<.25.
Devboth2/9boards(the2empty),0/18roles,only3ancestries. One seed,16trainingancestries,
lockedonlyempty/fixed8cellrow. Do not count views as independent accuracy samples.

ACCURACY4096_BASELINE.md original larger7.69Mmodel complete and frozen. Final
ordinary27-image replay/curator pass;dev0/18boards,0/36roles,369/401positivecells,
32misses,813inventedcells,geometry2.01544. All9correct training-probe ghosts are
absence. Run/source/optimizer integrity is not adequate learning. Current model
runs/m2-stream-accuracy4096-s10,original source121fd05a... and weights06751638....

LAYER_FEATURE_DIAGNOSTIC.md actual inputs/logits and unchanged ordinary outputs
verified for all3models. Centroid failure did not prove absent class information;
subsequent copied-head fits above are more informative. Original raw references,
source catalogs,summary/probe files and current model files are in the handoff.
Older unselected tensors/run folders remain local;Git preserves curated evidence.

## Safety and full scope

Windows kernel Job/kill-on-close,CPU1/cache16,GetPerformanceInfo/.2spressure guard
remain armed. Never disable/raise machine ceilings;before detached work record
RAM/wall/work caps and expectedpeak,verify actual startup/termination and keep logs.
Oneheavy job at a time;>40GBfree beforelongtraining. Reference machine31.2GBRAM,
stopcommit31,368,351,744B. NewhostcrashrequiresSystem41/1001anddump audit first.
Do not delete user run data,closeapps,change power or touch neighboring engine/stage.
Historical pressure stops/restarts retain their evidence;no crash was observed
in this handoff. Read RESOURCE_GUARD.md and TRAINING_RECOVERY.md.

The166real screenshots are unreviewed and excluded. Broad rendering/occlusion/
association,reviewedreal/independentsplits,adequateM2/M3jointlearning,scratch/
pretrained/directdecoder comparison,calibration/export/CPU/adapters and>=95%
readable-real exactcomplete-state remainOPEN. See CORE_GOALS.md for acceptance.
No claim that the research goal or release gate is complete.
