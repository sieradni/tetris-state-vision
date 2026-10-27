# Current checkpoint — 2026-10-27

Repo C:/Users/micha/Code/tetris-state-vision; codex/layout-diversity; no remote.
Goal ACTIVE, unfinished. Local narrow verified commits authorized; no push,
deploy, publication or user-data deletion. Neighboring tetrguide unchanged.
No subagents authorized. The166 user screenshots are drafts, not training truth.

## Selected work / acceptance todo

Optional dense active/ghost evidence experiment, default off; design and all
prior-bug interactions in LAYER_EVIDENCE_EXPERIMENT.md. New source catalogue
f095d3526162e5bbee45e0800c017296d05d45ec78cd76f4583642c2c5b7ce47.

- [x] Optional heads/pooling, masked independent targets/loss and visible-grid metrics.
- [x] 139 selected source-bound compatibility/protocol cases, not a unified suite.
- [x] True/false warm full512-square profiles, resident Adam/recovery/output parity.
- [x] Native true continuous6 == partial3+resume3, exact model/Adam/RNG/data/curve.
- [x] Matched fresh128-update pilot, source/data equal; both dev0/18 boards,0/36 layers.
- [ ] Finish both fixed768-update pilot studies, preserve pressure-recovery evidence.
- [ ] Matched768 report + predeclared disjoint192-image/64-ancestry diagnostic.
- [ ] Original fixed4096 baseline completion and adequate learning.

Source scopes: docs/evidence/layer-evidence-core-contracts/(60),
layer-controller-contracts/(18), layer-diagnostic-contracts/(26), layer-error-contracts/(35).
Later recovery-helper CLI extension is covered by actual native recovery r3;
these earlier unit scopes retain their own hashes. V-001..V-025 closed on named
gates in BUG_LOG; no model quality or fresh unified-suite claim.

Resource gates: true647,228 parameters/562,204,672B; false647,098/561,950,720B,
both640MiB. docs/evidence/m2-stream-layer-{pilot,control}-profile-r1/.
True native recovery2+6+3+3:38.156s/593,793,024B under1GiB; exact states/modes,
curator17 rows+72 images passed. docs/evidence/m2-stream-recovery-gate-r3/.

## LIVE true pilot s3 — inspect same receipts before any new model Job

Launcher35152, ancestor16448, controller26364, child supervisor30276,
child31596, worker13660. Exact launch command verified. Last observed step611.
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 896 --wall-seconds 420
--work-items 256 --receipt runs/m2-stream-layer-evidence768-s3-pipeline-job.json --
.venv/Scripts/python.exe scripts/run_stream_segment.py
--name m2-stream-layer-evidence768-s3 --target-steps 768 --updates 256
--admission runs/m2-stream-layer-pilot-profile-r1
--resume-from runs/m2-stream-layer-evidence768-s2
Expected<700MiB aggregate, child640MiB. Windows Job/readback/kill-on-close,
GetPerformanceInfo/.2s stop31,368,351,744B, CPU1/cache16. One heavy Job only.
Logs same-prefix-pipeline.log/.stderr.log; result/progress same-prefix-pipeline/;
training progress/summary/snapshots in same-prefix run; child same-prefix-job.json.
Freeze controller/imported helpers while live. Supervisor d2fb1478940b7b606073dbd75aa3746098fe7ede15f306082ec1f762d09c20db;
controller dadaee6df6761ce0f32f494979758111d56329466290ca3d408c2cfd8bcc46ad.

s1 true/false verified128/768, same data/source/config except boolean.
Both0/18 dev boards,0/36 layers; nonempty0/14. True cellacc.90222/geometry1.94832,
false.83056/2.21219. No exact-state improvement. Named comparison128 evidence.
True s2 pressure-stopped: logged541,banked512,29unbanked; ancestor125,
106.503s/554,323,968B; OS commit31,445,913,600B. Entire tree gone before s3.
Hash-verified actual bank512 pointer/manifest/payload; no saved inference summary.
docs/evidence/layer-learning-pressure-stop-r1/ preserves raw541-row prefixes and
stale child receipt. s3 must restore512, not541. Before final comparison verify
all541 old/new data/loss/terms/grad rows equal excluding elapsed time; old541
tensors unavailable. No crash occurred; pressure guard acted as designed.

## Next after terminal s3

Capture exact terminal receipts/artifact hashes/process death with
./docs/experiments/capture_layer_segment.ps1 -Name m2-stream-layer-evidence768-s3 -Launcher 35152
Then false s2: same896MiB/420s,work512 command, name m2-stream-layer-control768-s2,
admission runs/m2-stream-layer-control-profile-r1, resume runs/m2-stream-layer-control768-s1,
updates512,target768. Expected saved640/768. Then false s3,128new ->768.
Record exact command/log/PIDs before launch and verify native admission. Recheck
>40GB disk. Recent host commit29,136,605,184B,available11,587,985,408B,free46.89GB.
Never disable guards, close apps, delete data or retry old output paths.

Versioned read-only diagnostics under docs/experiments/:
compare_layer_studies.py --control FINAL_FALSE --candidate FINAL_TRUE --output docs/evidence/layer-learning768-comparison-r1
Run under128MiB/30s/work1590 (1536 curves+54 probe states); expected<96MiB.
probe_layer_generalization.py same --control/--candidate/--output options:
output docs/evidence/layer-generalization768-r1. UNRUN candidate source, not a green gate.
Predeclared development groups100..163/views0..2/seed20261026:192 images/model,
384 forwards,64 ancestries separate from fixed6 and training seed domain.
Expected<600MiB,640MiB cap/180s/work384; native guards. Preserve raw predictions,
recipes/class/status/pose denominators; actual observed dense grids separate from
absent grids. This still tests same clean/no-grid generator, not real/OOD skins.
Any pressure stop requires fresh receipt/cursor audit, never blind restart.

## Original larger baseline / remaining goal

Original runs/m2-stream-accuracy4096-s6 verified2688/4096,512new; all2688+27 recipes
replayed, actual inference/training hashes and native tree death verified.
Evidence same named docs/evidence directory. Weights
dd93e38745abb4aa2b02ddf47cc4e79a6dd835e7e037b747b0878fa51d643702.
Dev2/18 exact boards BOTH EMPTY; nonempty0/14,layers0/36,geometry2.052814543cells.
Must resume original archive121fd05a2f75e30b0c738dac592c67c6d015aa6043a43e4032d91ac662de794b
using admission runs/m2-stream-accuracy-probe, not current neural code. s7 max512
->3200/4096;1280MiB/2100s parent,1GiB child; CPU1/cache16, fixed recipe unchanged.

All quality gates OPEN: adequate scratch learning, broader rendering/visibility,
complete joint queue/hold, reviewed real labels/independent splits, initialization/
direct decoder comparisons, calibration/export/CPU latency and >=95% exact readable
real complete-state (abstentions fail). No selected release model; public API
remains needs_review/unsupported. Target exposure/ordinary/oracle diagnostics are
evidence, not proof of learning cause or whole-state success.
True s3 VERIFIED complete768/768,256new frombank512. Parent77.074s/558,313,472B;
all known PIDs/command tree gone, actual inference/training hashes verified and
raw controller/curator receipts copied. Pressure-prefix equality still pending.
NEXT false s2, expected<700MiB,896MiB aggregate/420s/work512,child640MiB:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 896 --wall-seconds 420
--work-items 512 --receipt runs/m2-stream-layer-control768-s2-pipeline-job.json --
.venv/Scripts/python.exe scripts/run_stream_segment.py
--name m2-stream-layer-control768-s2 --target-steps 768 --updates 512
--admission runs/m2-stream-layer-control-profile-r1
--resume-from runs/m2-stream-layer-control768-s1
Logs same-prefix-pipeline.log/.stderr.log, state same-prefix-pipeline/progress.json,
result.json; child same-prefix-job.json. Expected step_partial640/768. Freeze
controller/imported helpers; native admission/pressure/CPU1 guards unchanged.
LIVE false s2 launcher32316, exact command verified.
NEXT modest read-only pressure-prefix audit, expected<96MiB,128MiB/30s/work2164:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 128 --wall-seconds 30
--work-items 2164 --receipt runs/layer-pressure-prefix-r1-job.json --
.venv/Scripts/python.exe docs/experiments/verify_layer_pressure.py
Logs runs/layer-pressure-prefix-r1.log/.stderr.log; final
 docs/evidence/layer-learning541-recomputed-r1/scope.json. Four541-row finite inputs.
May coexist with the one640MiB training child; no second heavy workload.
Prefix audit launcher19788, exact command verified.
Pressure-prefix gate PASS.308s/42,455,040B under128MiB; native receipt/log copies
byte-exact, all3 PIDs gone. ALL541 data recipes and sample/loss/term/gradient rows
match excluding elapsed time, including29 recomputed updates513..541. Evidence
layer-learning541-recomputed-r1/; no old541 tensor comparison claimed.
Broader diagnostic source review now adds full recipe/seed/config/frozen-model
origin checks and independent ordinary/dense metric recomputation; still UNRUN.
