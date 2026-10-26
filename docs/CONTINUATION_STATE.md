# Current checkpoint — 2026-10-26

Repo C:/Users/micha/Code/tetris-state-vision, local Git, no remote. Goal ACTIVE.
No vision process running before the next launch. No run data deleted.
Previous goal turn made concrete progress (warm profile verified, recovery draft).
Current continuation: no implementation progress; same RAM blocker revalidated.
Fresh consecutive no-progress blocked audit count1; goal remains ACTIVE.

Verified checkpoints:
- 421de02 controlled128: exact16-step prefix, training boards2/6, development0/6.
- 331ac58 saved511/512 inner-wall partial: exact128 prefix, training boards6/6,
  layers10/12 and geometry.4106 cells fail joint fit; development0/6 and0/12.
  Weight ca4c0f34a81b8c7591fab7db0e92fd89d652a27dea015cf63473c3f7b780bcdf
  retained at runs/m2-learn512/checkpoint/. Do not call completed512.
- 72cde65 local reading: 36 changes/18 matched groups, direction36/36,
  targeted20/36, exact changed boards18/36; two unrelated cells changed.
- 6970ca8 streaming: fresh state/position triplets, frozen verified imports,
  zero tensor cache, all-final-tensor checkpoint parity. One512-square step:
  9.717s /549,154,816B under1GiB.14 source/sampler/admission checks. Source
  catalogue now changed; this historical probe cannot admit current long runs.
- f9ab4a6 pressure candidate; 53abb77 VERIFIED and V-010 CLOSED: pressure5,
  native-resource5, strengthened stream-admission3, all green on named logs in
  docs/evidence/v010/. Actual pressure park also preserved separately.
  Supervisor SHA635e764afdd1cac41b46057122deeed987f1a8345c262477cf1782671088867b.
- 64e582c component target source; now8 fixtures GREEN, .013s /41,533,440B
  under128MiB/45s. docs/evidence/m3-component-targets/scope.json. No component
  model/head/loss/runtime integration yet. M3 is not complete.

V-001..V-011 CLOSED on named gates. V-011 protocol4 green on322e055, actual
warm512-square2-step profile GREEN:9.553s /594,829,312B under640MiB,142
initialized parameter tensors,5,177,352B resident Adam state, all-final-tensor
reload exact. docs/evidence/m2-stream-warm-probe/scope.json; catalogue
42148a03905297502d6a36e02622087f808b8c36cc10883474a4b1bd84b8770b.
Both training samples and27 saved probes replayed exactly. PIDs27800/31056/
26384/23216 verified gone. Curator no longer attaches old fixture logs to new runs.
Default runner reserves launch headroom
and polls real GetPerformanceInfo counters every.2s, with native Job cap/wall/
work limits. Threshold31,368,351,744B commit and2GiB minimum physical available
on this machine; never bypass/relax it or use a pre-pressure archived supervisor.
Five pressure fixtures proved actual child/grandchild/venv-worker death on
pressure and counter failure, using tiny Jobs, plus real global counters.
The13 green checks and8 component checks are separate scoped runs, not a
whole-suite or accuracy claim. Current total distinct checks151 across scopes.

A safe window opened (no cargo/rustc at last check; committed~30.6GB). User async
question about lower-memory availability is still pending; no app-close or
other-job modification permission assumed. Actual OS state, not elapsed time
or an old receipt, governs launch. All test supervisors/children terminal.

Latest one-step attempt is terminal: runs/m2-stream-guard-probe-job.json,
supervisor3684 verified gone, memory_pressure125 before any child at32.63GB
commit. No model work ran; receipt/logs retained. No receipt path reuse.

Selected acceptance-gated task: actual fresh-state learning,768steps/256ancestry
groups. No quality threshold asserted from this engineering diagnostic. Compare
initial/final fixed27probes with exact board/layer/geometry metrics; require finite
ledger replay, native pass and checkpoint parity before curating as complete.
Expected peak <600MiB, same admitted640MiB envelope and one CPU thread.
Hidden detached command, logs runs/m2-stream-learn768.log/.stderr.log:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib640
--wall-seconds660 --work-items768 --receipt runs/m2-stream-learn768-job.json
-- .venv/Scripts/python.exe scripts/train_m2_stream.py
--output runs/m2-stream-learn768 --model pilot --steps768 --wall-seconds600
--development-groups6 --admission runs/m2-stream-warm-probe.
Do not launch if actual OS headroom insufficient. Record/verify PID after spawn.
Attempt terminal: shim29620/supervisor30048 verified gone; admission refused
before child at30,864,125,952B commit plus640MiB reservation. Receipt125,
runs/m2-stream-learn768-job.json; no output directory. No retry loop.
Prior V-003 catalogue/weight identity,V-004 raw counters,V-007 exact Job and
V-010 pressure closure remain armed. No neural/loss math change.

Next source task: isolated exact CPU training-snapshot primitive, not yet wired
into the source-frozen driver. Acceptance: interrupted+resumed6-step AdamW/dropout
matches continuous6-step model/optimizer/RNG; reject source/recipe/hash/cursor
mismatch; publish latest pointer only after complete artifact and ledger hashes.
Todo: implement helper; adversarial contracts; native bounded gate; review/commit.
Helper and7 fixtures now drafted; NOT integrated/verified. First harness
runs/training-snapshot-tests.json parked before child125 at31.014GB commit plus
512MiB reservation; supervisor21112 terminal, no fixture ran. No retry loop.
docs/TRAINING_RECOVERY.md describes proposed contract. Next actual command is
native512MiB/45s/7cases unittest discover -s tests -p test_training_snapshot.py -v,
with NEW receipt/log stem training-snapshot-tests-r2 after genuine headroom.
Even96MiB syntax harness parked before child: runs/training-snapshot-syntax.json.
No syntax/runtime success is claimed. Preserve this isolated source as a draft
checkpoint; do not integrate or admit it until its seven actual fixtures pass.
Latest continuation also refused a64MiB/15s/two-file syntax-only harness before
child: runs/training-snapshot-syntax-r2.json, supervisor28620, receipt125,
commit31,379,595,264B >stop31,368,351,744B; follow-up actual31,427,129,344B.
No vision workers live at the initial scan; supervisor28620 verified gone. No job
started and no code changed. All substantive training/recovery gates require a
lower-memory external state; no guard bypass, no unverified integration.
Architecture review: V-003 hashes/identity require source+recipe and byte-prefix
binding; V-004/V-007 preserve actual native readback/accounting; V-010 pressure
kill can interrupt writes, so pointer is the last commit; V-011 requires a new
warm resource profile after eventual driver integration. Visibility bugs remain
untouched: no model inputs/targets or active/locked/queue rules change.

Then characterize actual multi-step admission/streaming and adequately train on
fresh states. A long run may be interrupted by renewed host pressure; periodic
resumable checkpoints are the next mechanism if needed, not a guard opt-out.
Accuracy-sized configuration is explicit but still unprofiled/untrained. Give it
a separate bounded characterization; do not select the pilot as release model.

M3 next: bounded shared role/index queries and losses after verified labels;
legacy artifact parity and review-only output remain required. Prior visibility
bugs V-001/002/005/006/009 require paired-raster masking, no guessed active-cell
subtraction and no hidden queue length. V-003 source/weight identity and
V-004/V-007/V-010 native/global resource instruments stay armed.
Old neural math files unchanged; added component_targets.py changes aggregate
source digest. Past artifacts retain named commits/frozen source catalogues.

User-approved166 uncertain-answer screenshots remain unlabeled drafts at
data/real-pilot-drafts/manifest.jsonl, zero training/scoring eligibility. Seven
private overview pages in runs/real-drafts-overview/, aggregate receipts in
docs/evidence/real-drafts/scope.json. Never use legacy guesses as board truth.
Env .venv0.3.0, Torch2.14.1+cpu/NumPy2.5.4/Pillow12.3.0.
M2 broad learning, M3 full joint state, broad data, real reviewed pilot/baseline,
calibration/export/release remain OPEN. Whole goal not achieved.
