# Current checkpoint — 2026-10-27

Repository C:/Users/micha/Code/tetris-state-vision; branch codex/layout-diversity;
no remote. Goal ACTIVE, unfinished. Local narrow verified commits authorized.
No push/deploy/publish/user-data deletion. Neighboring tetrguide untouched.
No subagents authorized. Draft166 screenshots remain ineligible for training/scoring.

## Selected task and acceptance gates

Implement and measure the optional dense active/ghost evidence experiment;
docs/LAYER_EVIDENCE_EXPERIMENT.md records hypothesis, masks, prior-bug review
and gates. Default off preserves original M2 numerics. New source is a candidate,
not a selected release model. No real-image or complete-state quality claim.

- [x] Optional model pooling/targets/loss and explicit visible-grid metrics.
- [x] Original-reference equality, paired-raster/visibility/gradient/reload gates.
- [x] Frozen trainer/controller/configuration/diagnostic routing.
- [x] 139 selected checks with source-pinned scopes (not a whole-suite run).
- [ ] Fresh true and false two-full512-square-step warm-Adam/recovery profiles.
- [ ] Actual new-path finite learning and continuous/resumed recovery evidence.
- [ ] Matched-source/data comparison with ordinary exact-state denominators.
- [ ] Original fixed4096 baseline completion and adequate learning.

Verified scopes: docs/evidence/layer-evidence-core-contracts/ (60 cases),
layer-controller-contracts/ (18), layer-diagnostic-contracts/ (26),
layer-error-contracts/ (35). Three latter curators passed under128MiB/30s:
.419s/76,447,744B; .437s/76,771,328B; .429s/76,550,144B. All source hashes
rechecked, native receipt/log copies byte-verified, supervisor/child PIDs gone.
V-001..V-025 CLOSED on named historical gates in BUG_LOG; no unified-suite claim.

## Next detached job (no owned model job live)

Expected peak<600MiB; cap640MiB,120s,2 updates. Native readback-verified Windows
Job kill-on-close, GetPerformanceInfo/.2s stop31,368,351,744B, pre-import CPU1,
cache16. Recheck>40GB disk before long work. No cap bypass or app closure.

.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 640
--wall-seconds 120 --work-items 2
--receipt runs/m2-stream-layer-pilot-profile-r1-job.json --
.venv/Scripts/python.exe scripts/train_m2_stream.py
--output runs/m2-stream-layer-pilot-profile-r1 --model pilot --steps 2
--wall-seconds 60 --development-groups 6 --profile-square --layer-evidence

Logs runs/m2-stream-layer-pilot-profile-r1.log/.stderr.log;
state runs/m2-stream-layer-pilot-profile-r1/progress.json and summary.json;
archived code/source.json, checkpoint/manifest.json and latest-training.json.
Launch hidden, verify exact process command; record launcher/supervisor/worker IDs.
On terminal receipt verify all process death, warm resident Adam all parameters,
recovery and every final output parity; curate under128MiB/30s/work29 using
scripts/record_stream_run.py --name m2-stream-layer-pilot-profile-r1. Preserve
curator receipts/logs. Then identical false control profile, omit --layer-evidence,
name m2-stream-layer-control-profile-r1. Never reuse old fresh-code admissions.

Current controller dadaee6df6761ce0f32f494979758111d56329466290ca3d408c2cfd8bcc46ad;
supervisor d2fb1478940b7b606073dbd75aa3746098fe7ede15f306082ec1f762d09c20db.
Do not edit active controller. Original archives keep historical scopes valid.

## Original baseline (separate frozen study)

runs/m2-stream-accuracy4096-s6 VERIFIED step_partial2688/4096,512 new from2176.
All2688 recipes+27 saved probes replayed; inference/training artifact hashes
and native parent/child/curator preserved in docs/evidence/m2-stream-accuracy4096-s6/.
Parent690.480s/1,014,280,192B under1280MiB; child671.384s/978,317,312B under1GiB;
curator18.340s/113,209,344B under128MiB. Entire process tree gone before controller edit.
Weights dd93e38745abb4aa2b02ddf47cc4e79a6dd835e7e037b747b0878fa51d643702.
Dev2/18 exact boards BOTH EMPTY; nonempty0/14, layers0/36, geometry2.052814543cells.
Learning inadequate. Original source catalogue121fd05a2f75e30b0c738dac592c67c6d015aa6043a43e4032d91ac662de794b;
accuracy/capacity0, seed20261026, LR.001, eval128, checkpoint64, groups6, CPU1/cache16.
Future s7 must use original admission runs/m2-stream-accuracy-probe and resume s6;
max512 new ->3200/4096. Parent1280MiB/2100s/work512, original child1GiB.
New controller derives false and omits new keyword for original archived trainer.

1664-prefix exact target audit shows all7 observed classes taught, but ghost row10
only and fixed probes omit classes/statuses. docs/TARGET_EXPOSURE.md and
m2-stream-accuracy1664-exposure-r1/ preserve exact masks/counts. Ordinary/conditional
geometry diagnostics do not establish a learning cause. Do not call empty-board
success or lower loss complete-state accuracy.

## Remaining objective

Broad synthetic coverage; adequate scratch learning; complete queue/hold joint
learning; reviewed real labels/independent family/session/ancestry splits; pretrained/
direct decoder comparison; calibration/export/latency; >=95% exact readable real
complete-state, abstentions failures. All OPEN. User screenshots are draft references,
not truth. No model acceptance/public policy change; API remains needs_review/unsupported.

LIVE true-profile launcher23064, exact command verified; inspect same named receipt/progress before any next Job.

True profile PASS9.535s/562,204,672B under640MiB; catalogue
f095d3526162e5bbee45e0800c017296d05d45ec78cd76f4583642c2c5b7ce47;
launcher23064/supervisor32088/child25364/worker35424 gone. Next:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 128 --wall-seconds 30
--work-items 29 --receipt runs/layer-pilot-profile-curation-r1.json --
.venv/Scripts/python.exe scripts/record_stream_run.py --name m2-stream-layer-pilot-profile-r1
Logs runs/layer-pilot-profile-curation-r1.log/.stderr.log; scope in same named docs/evidence.
Expected<96MiB, native guards. Then false profile command exactly as above true,
omit --layer-evidence, replace run stem with m2-stream-layer-control-profile-r1.
Curator launcher25376 exact command verified.
True curator PASS; exact raw copies preserved; all3 PIDs gone. LIVE false profile launcher16012, exact command verified; same documented640MiB/120s/2 envelope.
False control profile PASS9.434s/561,950,720B under640MiB,same cataloguef095d352;
all launcher/supervisor/child/worker PIDs gone. NEXT curator:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 128 --wall-seconds 30
--work-items 29 --receipt runs/layer-control-profile-curation-r1.json --
.venv/Scripts/python.exe scripts/record_stream_run.py --name m2-stream-layer-control-profile-r1
Logs runs/layer-control-profile-curation-r1.log/.stderr.log. Expected<96MiB,native guards.
New recovery-helper optional flag is drafted; actual2+6+3+3 native gate pending.
False curator launcher32132 exact command verified.
Both current-source true/false pilot warm profiles curated. True647,228params,
false647,098; same cataloguef095d352. Old profile source scopes unchanged.
NEXT native actual recovery:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 1024 --wall-seconds 420
--work-items 14 --receipt runs/m2-stream-recovery-gate-r3-job.json --
.venv/Scripts/python.exe scripts/probe_stream_recovery.py
--output runs/m2-stream-recovery-gate-r3 --layer-evidence
Expected<700MiB aggregate,1GiB parent; children640MiB/90s each,
profile2+continuous6+partial3+resumed3. Windows Job/pressure/thread guards unchanged.
Logs runs/m2-stream-recovery-gate-r3.log/.stderr.log;
state same named run/progress.json; final proof.json. Freeze probe until terminal.
Next inspect proof/receipt and entire process death, then128MiB/30s curator
scripts/record_stream_recovery.py --name m2-stream-recovery-gate-r3, work92
(17 replayed training rows including resumed prefix +72 saved probe images=89;
use exact curator count after inspecting; no accuracy claim).
Recovery launcher25328, exact command verified.
Recovery r3 PASS38.156s/593,793,024B under1GiB; all recorded PIDs and matching
commands gone. Exact source/config/data/loss/terms/grad/evaluations/model/Adam/RNG.
NEXT curator expected<96MiB,128MiB/30s/work89 (17 rows+72 saved images):
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 128 --wall-seconds 30
--work-items 89 --receipt runs/layer-recovery-curation-r1.json --
.venv/Scripts/python.exe scripts/record_stream_recovery.py --name m2-stream-recovery-gate-r3
Logs runs/layer-recovery-curation-r1.log/.stderr.log. Native guards unchanged.
Curator launcher11744 exact command verified.
