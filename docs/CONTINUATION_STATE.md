# Current checkpoint — 2026-10-26

Repo C:/Users/micha/Code/tetris-state-vision; local branch codex/layout-diversity,
no remote. User's lower-memory reply enabled substantial verified work. Goal
service last reported BLOCKED (resume is user-controlled); actual work resumed.
Neighboring engine, user apps and run data untouched. Whole goal remains open.

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

Selected task: verify finite segment controller, then begin fixed4096-step larger
scratch baseline, at most512 new updates/segment. Completed128 stays a separate
diagnostic; do not silently extend its recipe. No scheduler/restart/notification.

First trial runs/m2-stream-accuracy-segment-probe: pressure-parked exit125 during
final evaluation after recovery1/target3. External Rust builds coincided with OS
commit31,370,371,072B > stop31,368,351,744B. Peak793,481,216B/1GiB,60.648s;
parent829,616,128B/1.25GiB,61.244s. No final summary/pipeline result; stale progress
is not live state. All matching processes verified gone. Wrapper UNVERIFIED.
Actual saved model/Adam/RNG/ledgers restored without new forward/update under
512MiB/30s/one restore: peak467,935,232B,5.177s,238 states all at step1.
docs/evidence/segment-pressure-recovery/scope.json. Generation:
step-000001-df559785b6354034af8e1d2ed89f76f0; tensors:
c60115c5dea3aca0cbc730a4ab2f74a63f56fb2521199242a4a64a208d6b0122.

NEXT fresh resume trial (only when baseline plus1.25GiB fits OS guard; latest
baseline~28.3GB). Expected<1100MiB; parent1280MiB/120s/one update, serial training
1024MiB and curation128MiB. Hidden detached command from repo root:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 1280
--wall-seconds 120 --work-items 1
--receipt runs/m2-stream-accuracy-segment-probe-r2-pipeline-job.json --
.venv/Scripts/python.exe scripts/run_stream_segment.py
--name m2-stream-accuracy-segment-probe-r2 --target-steps 3 --updates 1
--admission runs/m2-stream-accuracy-probe
--resume-from runs/m2-stream-accuracy-segment-probe
Logs runs/m2-stream-accuracy-segment-probe-r2-pipeline.log/.stderr.log;
pipeline/progress.json/result.json. Verify exact supervisor/worker PID/tree,
global2/new1, explicit step_partial and curated parent exit125, then tree death.
Native refusal/pressure stop is not a complete segment.
LAUNCHED resume trial: root27516/25860 and controller20136/7012 exact command
lines verified. Source8f6c1e8; inspect these logs/receipt and real worker PID
before waiting. No other vision heavy job live; original pressure trial is dead.

Only after green, launch fresh runs/m2-stream-accuracy4096-s1 with controller
--target-steps 4096 --updates 512 --admission runs/m2-stream-accuracy-probe under
parent1280MiB/2100s/512updates. Child1GiB/1860s outer/1800s inner; checkpoint64;
serial curator128MiB/120s. Expected<1100MiB total,<1GiB new disk; verify >40GiB
free (last50.79GiB). Root receipt/log prefix runs/m2-stream-accuracy4096-s1-pipeline.
Record launch/PIDs here, commit before waiting. If only long wait remains, leave
finite controller detached with discoverable logs/result and end turn.

Then adequately train and repair geometry/layer/board errors. M3 labels8 green,
heads/loss/runtime open. API review-only, queue/hold unknown. User's166 images
remain unlabeled drafts, zero training/scoring eligibility, seven private sheets;
legacy guesses are not truth. Broad rendering, reviewed real labels, controlled
initialization comparisons, calibration and export/release remain open.
