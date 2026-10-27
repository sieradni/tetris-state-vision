# Current checkpoint — 2026-10-26

Repo C:/Users/micha/Code/tetris-state-vision;codex/layout-diversity,no remote.
Goal ACTIVE,unfinished. Local commits authorized;no push/deploy/publish/user-data
deletion. Neighboring engine/apps/data untouched. One heavy model Job at a time;
native RAM/pressure/CPU guards armed. No subagents authorized.

S5 TERMINAL verified step_partial2176/4096,512 new from1664. Root PASS707.301s,
995,926,016B/1280MiB;child controlled2;curator passed. Actual inference/training
hashes and all recorded PIDs/commands verified dead. Scope and raw parent/curator
receipts copied to docs/evidence/m2-stream-accuracy4096-s5/. Weights
2a4eeeb93975a0aab43673cdaf631875e9a62e984ca89f973c0e933d43b43fb5;
original121fd05a recipe. All2176 recipes+27 probes replayed. Dev0/18boards,
0/36 layers,geometry1.85661909cells. Next job is planned1664 target replay below;
no model Job live. Old LIVE s5 block below is historical pending cleanup.
## Next action: original baseline s5 from verified1664

LIVE s5 launch18820/root20148/controller29040 verified;exact command:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 1280
--wall-seconds 2100 --work-items 512
--receipt runs/m2-stream-accuracy4096-s5-pipeline-job.json --
.venv/Scripts/python.exe scripts/run_stream_segment.py
--name m2-stream-accuracy4096-s5 --target-steps 4096 --updates 512
--admission runs/m2-stream-accuracy-probe --resume-from runs/m2-stream-accuracy4096-s4

Hidden logs runs/m2-stream-accuracy4096-s5-pipeline.log/.stderr.log;
state runs/m2-stream-accuracy4096-s5-pipeline/progress.json,result.json;
child runs/m2-stream-accuracy4096-s5-job.json and s5.log/.stderr.log;
evidence sentinel docs/evidence/m2-stream-accuracy4096-s5/scope.json.
Expected<1100MiB aggregate,parent1280MiB/2100s/work512,child1024MiB/1860s outer/
1800s inner;serialized curator128MiB/120s. Native pressure/Job/thread guards on.
Last host commit26,567,606,272B,available14,223,953,920B,C:49,119,178,752B free;
recheck before launch and require>40GB disk. No resource bypass or data deletion.

Start1664,at most512new ->2176,fixed target4096. MUST execute s4 archived trainer,
ORIGINAL catalogue121fd05a2f75e30b0c738dac592c67c6d015aa6043a43e4032d91ac662de794b,
accuracy/capacity0,seed20261026,LR.001,eval128,checkpoint64,groups6,CPU1/cache16.
Controller SHA11a0a81aeccddbb62451c46d24d2d4a844d762653b44cdb7b5edbb168bec0ac4
must remain unchanged while active. Native supervisor SHA
 d2fb1478940b7b606073dbd75aa3746098fe7ede15f306082ec1f762d09c20db unchanged.
Verified chain:18820/20148/29040+31004/13840+28636/6908+22228;worker22228
executes s4/code/scripts/train_m2_stream.py with original recipe. Last observed
step1687 at41.032s. No new separately verified snapshot yet;last final artifact1664.
Do not restart from a stale receipt or observation timeout. At controlled finish
verify entire process death,status/actualartifact/curation;at pressure stop first
record true bank/logged cursor and verify recovery. No automatic retry.

Selected next independent source/analysis work: audit actual observed/absent
piece-class exposure and local reading failures in the original curriculum,
without editing current model/data/optimizer sources or treating hidden state as
known. Current active-head development collapse may reflect learning/data issues;
do not assume root cause or claim model capacity from the conditional test.
Do not infer actual masked training exposure from source-visible state counts;
the exact frozen targets_for audit requires Torch and belongs in a serialized
model-free gap. Pure metadata/code review can proceed while s5 trains.
Do useful read-only/modest metadata work between polls;only long wait remaining
means checkpoint/commit/end turn,not prolonged babysitting.

Selected task: exact target-exposure audit for saved1664 using its original
frozen sample_case/prepare_image/targets_for,not source-visible counts. Todo:
implement pure mask-first layer packing/counting and protocol fixtures;bounded
metadata tests;commit;after s5 exits,serial native1691-target replay (1664updates+
9trainingprobes+18devprobes),then independent metadata/aggregate/source curator.
No model forward,training/source-catalogue/label-policy changes. V-001/V-005/V-006
require unknown masks and independent roles;V-003/V-014/V-018/V-020 bind actual
arch/config/archive/callables;V-021 original partial status;V-024 recomputed
unique populations/command/aggregate;V-023 raw copies byte-checked. Native runner
and all active files stay unchanged. Acceptance: unknown padded values cannot
change counts,absent distinct from observed,poses only when known,unique group
exposures distinct from repeated views,exact native recipe/target replay and
artifact/receipt provenance. Class balance alone cannot explain model failure.
Target-exposure candidate ready: scripts/target_exposure.py (mask-first pure),
probe_stream_exposure.py (frozen resize/targets,no model),record_stream_exposure.py
(independent metadata/aggregate/recipe/source curator). NEXT14 pure checks:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 128
--wall-seconds 30 --work-items 14 --receipt runs/target-exposure-tests-r1.json --
.venv/Scripts/python.exe -m unittest discover -s tests -p test_target_exposure.py -v
Logs runs/target-exposure-tests-r1.log/.stderr.log;expected<96MiB,native guards.
No Torch import;safe modest work beside verified live s5. Actual1691-target audit
on saved1664 waits for s5 entire process tree to exit;640MiB/120s/work1691,
expected<550MiB. Native output/receipt paths must be checkpointed before launch.
Target-exposure14 pure cases GREEN.040s,35,467,264B/128MiB/30s. Testlaunch26752/
supervisor20532/child36000 gone. Sourcee454345;no Torch/model/optimizer.
NEXT source curator exact command array runs/target-exposure-curation-r1-command.json:
128MiB/30s/work14,record_test_gates --name target-exposure-contracts --base e454345
--gate target-exposure-tests-r1:14 and12 sources. Logs
runs/target-exposure-curation-r1.log/.stderr.log,receipt same-prefix.json;
sentinel docs/evidence/target-exposure-contracts/scope.json. Expected<96MiB,
native guards;modest metadata job beside s5. Actual target audit still unrun.
Target14 source scope curated: docs/evidence/target-exposure-contracts/scope.json,
12 hashes one454345;curator.473s,74,833,920B/128MiB,24584/36156/5152 all gone.
S5 reached2176 but is finalizing;do not start target replay until entire tree exits.
PLANNED exact target replay command,after s5 terminal/dead verification:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 640
--wall-seconds 120 --work-items 1691
--receipt runs/m2-stream-accuracy1664-exposure-r1-job.json --
.venv/Scripts/python.exe scripts/probe_stream_exposure.py
--run m2-stream-accuracy4096-s4 --output runs/m2-stream-accuracy1664-exposure-r1
Hidden logs runs/m2-stream-accuracy1664-exposure-r1.log/.stderr.log;
state output/progress.json,result.json,targets.jsonl;expected<550MiB,native guards,
zero model forwards/updates. Frozen1664 source matches investigated error artifact;
new2176 is a different artifact. Do not edit audit code during actual replay.
After terminal/dead verification,curate under128MiB/30s/work1691 with fresh
runs/accuracy1664-exposure-curation-r1.json/.log/.stderr.log and
record_stream_exposure.py --name m2-stream-accuracy1664-exposure-r1;
evidence sentinel docs/evidence/m2-stream-accuracy1664-exposure-r1/scope.json.
Native target replay PASS33.834s,361,013,248B/640MiB/120s/work1691;
launch18156/supervisor18344/worker20180 and all matching commands gone.
Actual frozen1664 updates+9training+18dev targets,zero model forwards/updates;
weights unchanged. All training labels known;active1376observed/288absent,
ghost1046observed/618absent;all7 observed classes supervised. Ghost training
poses allrow10;training probe ghosts allabsent. Independent curation pending.
NEXT exact collector: .venv/Scripts/python.exe scripts/run_bounded.py
--memory-mib 128 --wall-seconds 30 --work-items 1691
--receipt runs/accuracy1664-exposure-curation-r1.json --
.venv/Scripts/python.exe scripts/record_stream_exposure.py
--name m2-stream-accuracy1664-exposure-r1
Hidden logs runs/accuracy1664-exposure-curation-r1.log/.stderr.log;
sentinel docs/evidence/m2-stream-accuracy1664-exposure-r1/scope.json.
Expected<96MiB,native pressure/Job guard. No model/target Job live.
## Verified s4 and ordinary/conditional diagnostics

M2_BASELINE_1664.md explains latest evidence and limits. s4 is verified
step_partial1664/4096,512new frombank1152. All1664 recipes+27 probes replayed;
training/inference actual hashes checked,all owned processes gone:
docs/evidence/m2-stream-accuracy4096-s4/scope.json and controller-receipt-scope.json.
Parent713.417s,993,869,824B/1280MiB;child controlled2,700.086s,957,689,856B/1GiB;
curator12.593s,103,497,728B/128MiB. Inference
7b1e4587437514db77cc58b19463ecbc1c0e23ed6bd9ff19b697b23cee70f4fb.
Dev0/18 exactboards,2/36 layers,2.22941-cell geometry. Training0/9,8/18 layers,
but all8 exact layers are absent ghosts;observed active0/6. Learning inadequate.

Ordinary format2 replay27 forwards PASS15.821s,573,911,040B/640MiB;
curator.447s,75,657,216B/128MiB. All saved metrics reproduce,weights unchanged;
docs/evidence/m2-stream-accuracy1664-errors-r1/scope.json. Both probe ledgers
byte-identical to704;comparison704.json preserves actual source/config and hashes.
Dev384/401TP,270FP,17FN (704:384TP,326FP,17FN);geometry worsened. Active predicts
one piece class in all18 dev images,15 identity errors and18 column errors.
All ordinary/curator PIDs and commands gone. No real/complete-state accuracy.

Conditional54-forward geometry PASS23.894s,585,072,640B/1GiB/120s;curator.288s,
64,712,704B/128MiB. docs/evidence/m2-stream-accuracy1664-geometry-r1/scope.json.
Actual step_partial1664/4096,architecture/config/source/weights bound. Supplied
correct lattice still0/18 devboards,cell accuracy.920278->.904444,layers2/36->0/36.
Weights unchanged,all ordinary metrics reproduced;labels intentionally enter
separate diagnostic forward. No capacity bound,automatic accuracy or runtime fix.
Launch9692/supervisor7656/worker12532 and curator2984/14444/12752 all gone.

New geometry sourceeaf5bee has10 metadata cases PASS.113s,35,180,544B/128MiB:
docs/evidence/geometry-evidence-contracts/scope.json,18 selected source/fixture
hashes;curator.493s,76,550,144B. Failed r1 retained:valid historical768MiB envelope
was omitted during candidate validation;640/768/1024 now accepted,live caps unchanged.
V-024 CLOSED on those tests and actual new native geometry replay+curation.
Old scopes retain old sources. All V-001..V-024 CLOSED;no whole-suite claim.

## Preserved foundation and remaining objective

s3 logged1172/banked1152 under OS pressure125,no crash. Actual frozen model/238
Adam states/RNG restore passed;all20 unbanked updates1153..1172 and1172 data/loss/
gradient prefixes reproduced exactly. Named scopes m2-stream-accuracy4096-s3-stop,
m2-stream-accuracy1152-recovery-r1,m2-stream-accuracy1172-recomputed-r1. No old1172
tensors or final1152 inference claim. New1664 is the current final artifact.

M3 neural sourcef542096/catalogueb6bb05f3 has68 scoped checks,pilot/larger warm
profiles and exact driver6==3+3 recovery. Controller3==1+2 plus16 controller and
4 copy-contract checks are in m3-controller-recovery-r1/segment-controller-contracts.
These are finite integration/resource proofs,not trained joint quality.

M2 adequate learning,M3 joint learning/long varied-shape memory,broad rendering,
real review,scratch/pretrained/direct comparisons,calibration/export and>=95%
exact complete-state release remain OPEN.166 user images unlabeled drafts,zero
training/scoring eligibility. API needs_review/unsupported. Unknown!=empty,layers
independent,rows bottom-first,no guessed footprint erasure. Preserve full scope.
