# Current checkpoint — 2026-10-26

Repo C:/Users/micha/Code/tetris-state-vision; branch codex/layout-diversity, no
remote. Goal ACTIVE and unfinished. Neighboring engine/apps/data untouched.
Local milestones committed; no push/deploy/publish/deletion authorized.

RUNNING: runs/m2-stream-accuracy4096-s2, resumed ORIGINAL M2 from bank192,
fixed global4096, at most512 new updates (up to704). Hidden root24896/20316,
controller23208/21624, training supervisor18008/15176, actual worker9836
(launcher26436). Exact command lines verified. Startup passed: global199 at
55.183s, original archived trainer, no stderr. Later290/4096 at242.158s, bank256
published. Only heavy vision Job live.
Check live PID and terminal receipts; stale running/progress alone proves nothing.

Native parent1280MiB/2100s/512new updates, child1024MiB/1860s outer/1800s inner,
one CPU, cache16, checkpoints64. Serial curator128MiB/120s. Expected<1100MiB
aggregate,<1GiB new disk; last disk50.07GiB free (>40GiB). Native named Job /
kill-on-close and GetPerformanceInfo/.2s pressure guard remain armed.
Exact launch command from repo root:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 1280
--wall-seconds 2100 --work-items 512
--receipt runs/m2-stream-accuracy4096-s2-pipeline-job.json --
.venv/Scripts/python.exe scripts/run_stream_segment.py
--name m2-stream-accuracy4096-s2 --target-steps 4096 --updates 512
--admission runs/m2-stream-accuracy-probe --resume-from runs/m2-stream-accuracy4096-s1
Root logs/receipt prefix runs/m2-stream-accuracy4096-s2-pipeline; controller
runs/m2-stream-accuracy4096-s2-pipeline/{progress,result}.json. Worker
runs/m2-stream-accuracy4096-s2/{progress,latest-training,summary}.json and
runs/m2-stream-accuracy4096-s2-job.json/.log/.stderr.log.

NEXT inspect these same handles/files, not a fresh launch. On segment_verified,
verify actual new/global counts, partial tag, source/artifact and ancestor-stop
provenance, terminal tree and curated evidence. Commit narrow milestone, then
run error probe on actual saved segment before selecting learning changes.
On pressure/timeout/crash preserve everything, inspect latest valid bank and
reason before any new resume. If only a long wait remains, leave the finite
controller detached. No auto-restart/scheduler/notification.

CRITICAL source separation: current M3 package changed after original M2 study.
All4096 continuations MUST execute the parent's verified archive, original recipe,
Adam/RNG/global cursor, not current source or completed128 diagnostic. Original
catalogue121fd05a2f75e30b0c738dac592c67c6d015aa6043a43e4032d91ac662de794b.
Supervisor SHA d2fb1478940b7b606073dbd75aa3746098fe7ede15f306082ec1f762d09c20db.
Pre-import CPU pools1, finite native cache16; no opt-out, ceiling increase or deletion.

Prior s1 TERMINAL pressure125 at748.552s,956,899,328B/1280MiB; OS commit
31,456,382,976B > stop31,368,351,744B. All matching processes verified gone.
229 updates logged; bank192 valid,37 later updates not restored. Raw child receipt
remains running/null because ancestor killed it before final publication; preserve
terminating ancestor receipt alongside it. No summary/final weights/segment pass.
docs/evidence/m2-stream-accuracy4096-s1-stop/scope.json verifies archived source,
tensor hash and all3 committed prefixes. Bank generation:
step-000192-e7873277a86a4428a7c8aa71898e44c0; tensor SHA
2d0e8e86f485baa1761842978c4d39ebea90c48b40fbc977bf1a7d7044f3927b.
Curator now retains ancestor receipt/logs and explicitly marks stale child status
on resumed histories. Three partial-evidence boundary fixtures reran GREEN;
actual s2 final curation will exercise ancestor metadata. Hashes alone are not
semantic restoration; real resumed updates now run from bank192.
Actual pressure replay GREEN: docs/evidence/m2-stream-accuracy4096-pressure-replay/
scope.json; all229 recorded data/loss/terms/gradient norms match (elapsed excluded),
including37 updates recomputed after bank192. All3 committed prefixes identical.
No old229 tensor artifact survived, so no final-model tensor equality claim.

M2 evidence: pilot768 complete, resource/recovery gate GREEN at640MiB,
519,782,400B peak,2.6MiB second-half growth, exact565 old records; dev0/18 exact
boards,0/36 layers,geometry1.8522cells. Larger7,694,010-parameter profile GREEN
at1GiB/910,176,256B. Larger128 completed93+resume35; all128 data recipes match
pilot128. Dev4/18 boards all zero-locked,nonempty0/14,layers0/36,geometry2.4785.
docs/evidence/m2-stream-accuracy128-r2/{scope,analysis}.json. Neither converged.

NEW native error report on old larger128: docs/evidence/m2-stream-accuracy128-errors/
scope.json/result.json,27 ordinary forwards,32.419s outer,546,164,736B/1GiB.
All saved metrics reproduced, weights unchanged, no labels passed to model;
all probe processes verified gone before s2. Dev401 positive cells:15 TP,386 FN,
8 FP; active18/18 and ghost18/18 predicted unknown. Mean origin error2.1934,
per-column basis error.4096,per-row.1366 source-cell units; no causal/accuracy
claim. Six dev ancestry groups with dependent triplets. Pure analyzer12 fixtures
GREEN on1531769. Pilot oracle-grid intervention separately still0 exact boards.

NEW optional M3 model/loss milestone fdb0c81: shared role/index queries/chunks,
capacity1..64 plus EOS64, independent presence/slot/status/start/end/hold heads,
masked semantic losses and shared encoder gradients. Capacity0 retains M2 state/
outputs; explicit M3 checkpoint identity/legacy loading tested. Source-bound
44 scoped checks:15 new,14 M2 numerical,7 inspection,8 labels, in
 docs/evidence/m3-joint-contracts/scope.json. No whole-suite/quality claim.
V-016 unsafe prepared-component admission and V-017 bad fixture CLOSED on15-case
r2 gate. V-001..V-017 now closed on named source-scoped evidence; no new crash.

M3 LIMITS: source Icon fields lack resized support geometry. M3 targets currently
refuse resized or invalid-alpha pixels; unchanged valid source/padding only.
Need explicit resized component observer + paired-raster gate, fresh full-square
warm optimizer/resource profile, actual M3 training and reviewed runtime decoding.
API always needs_review/unsupported, queue/hold unknown. No old M2 admission
covers new M3 heads/current package. See docs/M3_COMPONENT_TARGETS.md.

Whole goal still open: adequate M2/joint learning, broad rendering/crops/layouts,
reviewed real pilot and controlled scratch/pretrained/direct comparisons,
calibration, export and release. User's166 images are unlabeled drafts, zero
training/scoring eligibility; seven private sheets. Unknown is not empty;
locked/active/ghost separate, bottom-first. Torch2.14.1+cpu, no GPU.
