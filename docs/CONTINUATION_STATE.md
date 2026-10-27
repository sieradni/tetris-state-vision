# Current checkpoint — 2026-10-26

Repo C:/Users/micha/Code/tetris-state-vision, local Git, no remote. Working from
user's lower-memory reply; host commit~21GB. Goal service last reported BLOCKED
(user controls its resume); actual work resumed and made substantial progress.
No vision job running before next launch; all r3 processes verified gone.
No data deleted; neighboring engine untouched. Neural package/sampler/lock unchanged.

Completed milestone: runs/m2-stream-learn768-r3, COMPLETE768,458.494s inner/
459.718s outer, one CPU, native640MiB. Peak519,782,400B (~496MiB). Predeclared
resource gate PASSED: exact565 prior loss/term/gradient-norm and data/pixel/label
records; second-half peak growth2,748,416B <=32MiB; final recovery768 and every
final-tensor reload exact. Curator replayed768 samples+27 saved probes.
docs/evidence/m2-stream-learn768-r3/{scope,analysis}.json. Source catalogue:
121fd05a2f75e30b0c738dac592c67c6d015aa6043a43e4032d91ac662de794b.
Weights d183a22afcbe0e08d64b59bc1bd3cb2c337bd58e1d4d048f6406407943f43a74,
runs/m2-stream-learn768-r3/checkpoint/. Recovery64..768 retained.

Learning NOT accepted: training0/9 exact boards,12/18 known layers,geometry3.010;
development0/18 boards,0/36 layers,geometry1.852cells. Dev loss6.643->3.756 and
cell accuracy.724->.882 do not establish exact state. Six development ancestry
groups only; matched views are dependent. No real data trained/scored/released.

All V-001..V-015 CLOSED on named gates. V-013 finite failure/cache evidence:
docs/evidence/v013-failure/. Original565 failure had no final weights; no old
model-tensor equality claim. New native cache16/recovery/memory-curve resolves
the observed finite failure, not every untested run. V-014/V-015 close on:
- docs/evidence/v015-contracts/scope.json:29 scoped checks on f583792.
- docs/evidence/m2-stream-recovery-gate-r2/scope.json: native2+6+3+3 exact data,
  losses/gradients/evaluation/model/Adam/RNG and archived main;17 recipes+72 probes.
Guard SHA d2fb1478940b7b606073dbd75aa3746098fe7ede15f306082ec1f762d09c20db;
pre-import OMP/MKL/OpenBLAS1, kernel cap/kill-on-close, exact named Job and OS
GetPerformanceInfo/.2s pressure remain armed. No opt-out or receipt overwrite.

Selected next task: accuracy-sized model FIRST bounded characterization.
Peak unknown: experiment at1GiB aggregate cap,300s outer/240s inner,
2 square optimizer updates, one CPU, native cache16, no CUDA. Native refusal
is an unadmitted experiment, not model-quality evidence. Command:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib1024
--wall-seconds300 --work-items2 --receipt runs/m2-stream-accuracy-probe-job.json
-- .venv/Scripts/python.exe scripts/train_m2_stream.py
--output runs/m2-stream-accuracy-probe --model accuracy --steps2 --profile-square
--wall-seconds240 --development-groups6.
Logs runs/m2-stream-accuracy-probe.log/.stderr.log. Launch hidden/detached,
verify exact PID/tree and record it before waiting. Inspect receipt/progress,
verify terminal tree, preserve failure/denial. Curate only native/warm/recovery/
all-final-tensor pass on actual config; never silently substitute pilot.
Contract review: V-003 source/artifact; V-004 raw counters; V-007 exact Job;
V-010 pressure; V-011 warm state; V-012 pools; V-013 native caches/long shape;
V-014 archived source; V-015 bounded atomic retries. Neural/visibility math unchanged.

LAUNCHED accuracy experiment:12556/25988 exact supervisor commands verified;
host commit21.80GB. Inspect same receipt/progress and verify actual worker PID
before any wait. No other heavy vision job live; r3 milestone committed229efde.
Accuracy profile COMPLETE: all matching processes verified gone;28.860s inner/
29.676s outer,910,176,256B under1GiB. Actual7,694,010 parameters,238 initialized
tensor states/61,553,032B Adam; widths48/96/192/256,D192,heads6,blocks6. Full
recovery/warm/every-final-tensor parity. Catalogue121fd05a... unchanged.
docs/evidence/m2-stream-accuracy-probe/scope.json;2 samples+27 probes replayed.

NEXT first larger learning diagnostic:128fresh updates, same recipes as pilot's
first128, global target128; not an adequate-convergence or architecture ranking.
Expected<950MiB, native1GiB/360s outer/300s inner, one CPU, snapshots64.
Hidden detached command:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib1024
--wall-seconds360 --work-items128 --receipt runs/m2-stream-accuracy128-job.json
-- .venv/Scripts/python.exe scripts/train_m2_stream.py
--output runs/m2-stream-accuracy128 --model accuracy --steps128 --wall-seconds300
--development-groups6 --checkpoint-every64 --admission runs/m2-stream-accuracy-probe.
Logs .log/.stderr.log; latest-training.json/state generations in output. Record
PID/tree, then inspect same live handle and preserve any partial/denial. Require
native pass,128-step replay/inference+recovery identity for completed curation;
compare data-only prefix with pilot, not different model optimizer curves.
LAUNCHED larger128:20552/6384 exact supervisor commands verified. Inspect same
progress/receipt and verify real worker before waiting. Model/config/source
121fd05a...; no parallel heavy vision job. Independent source work: diagnostic
oracle-lattice intervention to separate geometry/local-reading failure, only on
the saved synthetic pilot artifact; never production inference/training truth.
V-003/V-014 bind frozen source/weights; V-001/002/005/006 require full supported
geometry and keep layers separate; no hidden targets or model weights changed.
First larger segment TERMINAL: wall_partial93/128, checkpoint93 saved,
300.558s update-loop/331.040s outer,917,159,936B under1GiB. Verify process
tree gone before next heavy job; never call it complete128. Resume35 updates
into a fresh output after the short pilot geometry diagnostic.
Oracle source is syntax-green, runtime pending. Native768MiB/120s/54forwards,
expected<600MiB, no training, fixed27 saved pilot probes, source121fd05a...:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib768
--wall-seconds120 --work-items54 --receipt runs/m2-stream-pilot-geometry-job.json
-- .venv/Scripts/python.exe scripts/probe_stream_geometry.py
--run m2-stream-learn768-r3 --output runs/m2-stream-pilot-geometry.
Hidden logs .log/.stderr.log. Check ordinary metrics reproduce saved values,
weights unchanged and intervention restored. Any oracle score is diagnostic
only, never automatic accuracy. After it ends, resume larger93+35 exactly.
Oracle COMPLETE:54 forwards/15.506s inner/16.816s outer,415,694,848B/768MiB;
ordinary metrics exactly reproduced, weights unchanged, temporary decoder
restored. Correct grid still0 exact boards; train cell.795->.8333, dev.8822->
.8583; layers unchanged12/18 and0/36. Conditional intervention only, not an
accuracy ceiling or deployed correction. All matching processes verified gone.

NEXT resume35 into runs/m2-stream-accuracy128-r2: same source/recipe/model/Adam/
RNG and global target128. Expected<950MiB, native1GiB/360s/35updates,
worker inner300s, one CPU. Command:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib1024
--wall-seconds360 --work-items35 --receipt runs/m2-stream-accuracy128-r2-job.json
-- .venv/Scripts/python.exe scripts/train_m2_stream.py
--output runs/m2-stream-accuracy128-r2 --model accuracy --steps128
--max-new-steps35 --wall-seconds300 --development-groups6 --checkpoint-every64
--admission runs/m2-stream-accuracy-probe --resume-from runs/m2-stream-accuracy128.
Hidden logs .log/.stderr.log; verify exact PID/tree and saved parent cursor93.
Curator must preserve parent wall_partial93 receipt, snapshot/hash/ledger-prefix
provenance and resumed35 count, not describe second Job as128updates.
Resume COMPLETE:7624/23788/28444/22504 verified gone, global128, new35,
183.639s inner/184.763s outer,910,688,256B/1GiB. Parent remainedwall_partial93
with331.040s outer/917,159,936B. Curator verified parent tensor/manifest and
all3 carried prefixes;128 new-data records and27 probes replayed. Weights
1ed5a3113ec3322a80f3b3d56f8a5c88df7e9c4e1813de67ecf8a555a766dc78.
Data prefix matches pilot128 exactly. Large dev4/18 boards are all zero-locked;
nonempty0/14, layers0/36, geometry2.4785. No model ranking/convergence claim.

Selected next work: fixed4096-update larger scratch baseline, at most512 per
native segment with periodic64 recovery and automatic explicit-status curation.
The128 run was diagnostic; a fresh fixed4096 target preserves a single optimizer
history across future segments, without changing completed128's contract.
Controller run_stream_segment.py is syntax-green; native integration pending.
Partial-curation3 boundary tests + real93-step replay GREEN; source/evidence
milestones must be committed before the trial and before any wait.

NEXT controller trial: parent1280MiB/120s/one update, expected<1100MiB. It runs
one larger update toward target3 in a child1GiB Job, then serial128MiB curation.
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib1280
--wall-seconds120 --work-items1 --receipt runs/m2-stream-accuracy-segment-probe-pipeline-job.json
-- .venv/Scripts/python.exe scripts/run_stream_segment.py
--name m2-stream-accuracy-segment-probe --target-steps3 --updates1
--admission runs/m2-stream-accuracy-probe.
Hidden root logs runs/m2-stream-accuracy-segment-probe-pipeline.log/.stderr.log;
pipeline/progress.json and result.json distinguish segment success from an
unfinished global target. Verify PID/tree, exact child status and death. Only
after green launch512 toward4096 at parent1280MiB/2100s, child1GiB/1860s outer/
1800s inner. Expected<1100MiB total, disk50.79GiB free (>40GiB), <1GiB new run
state expected. No automatic restart or message/notification; one finite segment.

Then adequately train larger scratch candidate and resolve geometry/layer/board
errors. M3 labels8 green, heads/loss/runtime open. Unknown is not empty;
locked/active/ghost separate and bottom-first. API stays review-only. User's166
screenshots remain unlabeled drafts, zero training/scoring eligibility, seven
private sheets. No legacy guess is truth. Env .venv0.3.0/Torch2.14.1+cpu/
NumPy2.5.4/Pillow12.3.0. Broad data, reviewed real pilot/baselines, calibration,
export/release remain OPEN. Whole goal not achieved.
