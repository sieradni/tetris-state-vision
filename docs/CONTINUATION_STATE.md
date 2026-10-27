# Current checkpoint — 2026-10-26

Repo C:/Users/micha/Code/tetris-state-vision; local branch codex/layout-diversity,
no remote. User's lower-memory reply enabled substantial verified work. Goal
service last reported BLOCKED (resume is user-controlled); actual work resumed.
Neighboring engine, user apps and run data untouched. Whole goal remains open.

RUNNING: first larger4096-target segment, at most512 new updates. Hidden detached
root26644/20696, controller31560/21424, training supervisor6232/28724 and actual
worker8988 (launcher15820), exact command lines verified. Source145b76a; only
heavy vision job. Startup verified training step10/4096 at61.093s, PID8988 alive,
no stderr, host commit27.37GB. Inspect paths below before any new launch;
progress/receipt marked running can be stale, verify process tree.

Completed finite pilot: runs/m2-stream-learn768-r3, COMPLETE768,458.494s inner,
459.718s outer, one CPU,640MiB native cap,519,782,400B raw peak. Exact565 prior
loss/gradient/data records; second-half growth2,748,416B <=32MiB; recovery768
and all final tensors reload exactly. Curator replayed768 samples+27 probes.
docs/evidence/m2-stream-learn768-r3/{scope,analysis}.json. Learning FAILS:
training0/9 boards,12/18 layers,geometry3.0099cells; dev0/18 boards,0/36 layers,
geometry1.8522. Six dev ancestry groups with dependent triplets, no real score.
Weights d183a22afcbe0e08d64b59bc1bd3cb2c337bd58e1d4d048f6406407943f43a74.

Larger7,694,010-parameter model profile GREEN:2 warm/recovery updates,1GiB,
910,176,256B peak,238 initialized Adam parameter states. First learning study
completed128 as wall_partial93 + exact resume35, not128 updates in second Job.
Peaks917,159,936B and910,688,256B. All128 data recipes match pilot128; parent
snapshot/three prefixes retained and verified. Dev4/18 boards are ALL zero-locked;
nonempty0/14,layers0/36,geometry2.4785. No convergence/architecture ranking.
docs/evidence/m2-stream-accuracy128-r2/{scope,analysis}.json. Weights:
1ed5a3113ec3322a80f3b3d56f8a5c88df7e9c4e1813de67ecf8a555a766dc78.

Oracle-grid diagnostic on saved pilot768:54 forwards, ordinary metrics reproduced
exactly, weights unchanged, decoder restored. Correct synthetic grid still gives0
exact boards; train cell.7950->.8333,dev.8822->.8583. Deliberate label intervention
only, not automatic accuracy/upper bound or production correction.
docs/evidence/m2-stream-pilot-geometry/scope.json.

All V-001..V-015 CLOSED on named gates.29 scoped resource/recovery contracts:
docs/evidence/v015-contracts/scope.json. Native2+6+3+3 exact recovery:
docs/evidence/m2-stream-recovery-gate-r2/scope.json. Three separate partial-evidence
boundary cases: docs/evidence/stream-evidence-contracts/scope.json. No unified
suite claim. V-013 closes observed565-update memory failure on finite768/cache16
gate, not every untested shape/run. Raw failed565/cache evidence retained.

Frozen training catalogue (neural package/sampler/lock unchanged):
121fd05a2f75e30b0c738dac592c67c6d015aa6043a43e4032d91ac662de794b.
Supervisor SHA d2fb1478940b7b606073dbd75aa3746098fe7ede15f306082ec1f762d09c20db.
Seven archived helpers; pre-import CPU pools1; native cache16; exact named
Windows Job, kill-on-close and GetPerformanceInfo/.2s pressure guard stay armed.
No opt-out, higher cap, receipt overwrite or deletion. Actual CPU Torch2.14.1.

Selected task: verified finite segment controller, now begin fixed4096-step larger
scratch baseline, at most512 new updates/segment. Completed128 stays a separate
diagnostic; do not silently extend its recipe. No scheduler/restart/notification.

First trial runs/m2-stream-accuracy-segment-probe: pressure-parked exit125 during
final evaluation after recovery1/target3. External Rust builds coincided with OS
commit31,370,371,072B > stop31,368,351,744B. Peak793,481,216B/1GiB,60.648s;
parent829,616,128B/1.25GiB,61.244s. No final summary/pipeline result; stale progress
is not live state. All matching processes verified gone. First trial incomplete.
Actual saved model/Adam/RNG/ledgers restored without new forward/update under
512MiB/30s/one restore: peak467,935,232B,5.177s,238 states all at step1.
docs/evidence/segment-pressure-recovery/scope.json. Generation:
step-000001-df559785b6354034af8e1d2ed89f76f0; tensors:
c60115c5dea3aca0cbc730a4ab2f74a63f56fb2521199242a4a64a208d6b0122.

Resume trial GREEN: runs/m2-stream-accuracy-segment-probe-r2-pipeline/result.json,
segment_verified,global2/target3,new1,step_partial,2 recipes+27 probes replayed.
Parent exit125/snapshot retained. Native parent68.815s,878,436,352B/1280MiB;
training842,248,192B/1GiB,curator82,022,400B/128MiB. All matching Python processes
verified gone. docs/evidence/m2-stream-accuracy-segment-probe-r2/controller-gate.json
pins actual controller source89138cd8... and receipts. This is not target3 complete.

ACTIVE first larger long baseline: expected<1100MiB total,<1GiB new disk; latest
OS commit26.4GB,disk50.33GiB free (>40GiB). Native parent1280MiB/2100s/512updates;
child1GiB/1860s outer/1800s inner,checkpoint64; serial curator128MiB/120s.
Hidden detached command from repo root:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 1280
--wall-seconds 2100 --work-items 512
--receipt runs/m2-stream-accuracy4096-s1-pipeline-job.json --
.venv/Scripts/python.exe scripts/run_stream_segment.py
--name m2-stream-accuracy4096-s1 --target-steps 4096 --updates 512
--admission runs/m2-stream-accuracy-probe
Root receipt/log prefix runs/m2-stream-accuracy4096-s1-pipeline; controller state
runs/m2-stream-accuracy4096-s1-pipeline/{progress,result}.json; actual training
runs/m2-stream-accuracy4096-s1/{progress,latest-training,summary}.json. Preserve
partial/pressure/timeout honestly. Only controlled segment results enter curation.
NEXT inspect pipeline result/receipts and training latest pointer, verify live or
terminal process tree. On segment_verified, review actual counts/memory/metrics
and curated source/artifact identity; commit narrow evidence, then resume the
same global4096 source/recipe/cursor in a NEW segment name if headroom permits.
On pressure/timeout/crash, preserve receipts/logs/latest state, diagnose first;
do not promote an uncurated result to success. No auto-restart. A missing result
is not a completed segment. Leave this finite controller detached while waiting.

Then adequately train and repair geometry/layer/board errors. M3 labels8 green,
heads/loss/runtime open. API review-only, queue/hold unknown. User's166 images
remain unlabeled drafts, zero training/scoring eligibility, seven private sheets;
legacy guesses are not truth. Broad rendering, reviewed real labels, controlled
initialization comparisons, calibration and export/release remain open.

Selected concurrent source task: decompose existing M2 prediction errors without
changing neural math, labels, training recipes or the live archive. Add a pure
bounded analyzer and adversarial fixtures, then a separate post-run probe that
replays saved rasters/labels and reproduces existing metrics before reporting
missed/invented cells, grid translation/scale and active/ghost causes. No second
heavy job while this training process lives. Pure12 fixtures GREEN in0.014s,
33,021,952B/128MiB; driver syntax GREEN separately. Actual model replay remains
pending until training is terminal; exact diagnostic command and limits are in
docs/M2_LEARNING_ACCEPTANCE.md. Do not claim runtime gate from pure fixtures.
Source-pinned12-case receipt: docs/evidence/m2-error-analysis-contracts/scope.json
(1531769). No package/training-source files changed; catalogue121fd05a... remains
unchanged. Review detailed errors after this segment before altering model math.
Prior contracts: V-001/002/005/006 unknowns and separate layers; V-003/014 frozen
sources/weights; V-004 raw resource counters. No architectural change/fix claim.
