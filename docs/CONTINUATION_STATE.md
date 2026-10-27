# Current checkpoint — 2026-10-26

Repo C:/Users/micha/Code/tetris-state-vision;codex/layout-diversity,no remote.
Goal ACTIVE and unfinished. Local commits authorized;no push/deploy/publish or
user-data deletion. Neighboring engine/apps/run data untouched. One heavy model
Job at a time;native memory/OS pressure/CPU thread guards remain armed.

## Next task: original baseline from bank1152

LIVE launch31508 verified;root supervisor33104/controller13492. Exact detached command:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 1280
--wall-seconds 2100 --work-items 512
--receipt runs/m2-stream-accuracy4096-s4-pipeline-job.json --
.venv/Scripts/python.exe scripts/run_stream_segment.py
--name m2-stream-accuracy4096-s4 --target-steps 4096 --updates 512
--admission runs/m2-stream-accuracy-probe --resume-from runs/m2-stream-accuracy4096-s3

Logs runs/m2-stream-accuracy4096-s4-pipeline.log/.stderr.log;
state runs/m2-stream-accuracy4096-s4-pipeline/progress.json,result.json;
child runs/m2-stream-accuracy4096-s4-job.json and s4.log/.stderr.log;
sentinel docs/evidence/m2-stream-accuracy4096-s4/scope.json if controlled finish.
Child1024MiB/1860s outer/1800s inner,checkpoint64,CPU1/cache16;serialized
curator128MiB/120s. Expected<1100MiB aggregate;native Job/pressure guard stays on.
Launchcheck commit26,539,560,960B,available14,388,453,376B,C:free51,524,427,776B;
recheck before launch,require>40GB disk. User freed disk;no data deleted here.

Controller b8eb48a,SHA11a0a81aeccddbb62451c46d24d2d4a844d762653b44cdb7b5edbb168bec0ac4
MUST remain unchanged while active. Native supervisor SHA
 d2fb1478940b7b606073dbd75aa3746098fe7ede15f306082ec1f762d09c20db unchanged.
Execute ORIGINAL s3/code/scripts/train_m2_stream.py,catalogue
121fd05a2f75e30b0c738dac592c67c6d015aa6043a43e4032d91ac662de794b.
Actual1152 preflight PASS.353s,34,402,304B/128MiB/30s/work1:
docs/evidence/m2-s4-preflight-r1/scope.json. Start1152,target4096,budget512;
accuracy/capacity0,seed20261026,LR.001,eval128,checkpoint64,groups6,no componentflag.
At most512new ->1664;target stays4096. Launch/PIDs and20 recomputed rows verified. If only long wait
remains,commit live discovery state and end turn,do not babysit indefinitely.

LIVE s4: verified worker30892 (redirector4208),child supervisor31064 (launcher27512),
controller13492/30656,root33104/launch31508. Last observed training1235 at111.993s;
new bank1216 manifest/payload hashes verified. Pointer manifest
59e01068f65e5bc3b28696dffbe889c6110ce27cf100e837570359f4bc9e45f8;
payload a8f33371ec362b9e5d1010d3c30325d80ea1a81323f1838712fd4e76b218941b.
This is a saved training snapshot,not a final inference/accuracy result. Latest
resource check commit27,967,053,824B,available13,263,626,240B,disk51,421,515,776B.

Recomputation PASS.335s,52,961,280B/128MiB/30s/work1172:
docs/evidence/m2-stream-accuracy1172-recomputed-r1/scope.json. All1172 data rows
byte-exact;all1172 loss/terms/gradient/sample rows equal excluding elapsed,
including20 unbanked s3 updates1153..1172. Original1172 tensors unavailable.
Raw old/new finite prefixes,verifier,receipt/logs preserved with hashes. Launch
25364/supervisor27564/child17992 and matching proof commands gone. No live-file
edits,new forward/update or completed-target claim from the read-only proof.

NEXT inspect existing pipeline receipt and exact PID30892 before doing anything.
While confirmed live,do not restart or edit controller/training catalogue. If
controlled finish,inspect result.json,child/curator/parent terminal receipts,
summary/actual architecture,curated scope and process death;preserve ancestor
logs/results/receipts and inspect updated learning metrics. If pressure125,
record actual bank/logged cursor and raw statuses,verify all owned processes
are gone,then verify recovery rather than relaunching automatically. Target4096
remains unfinished. Only long wait remains;live discovery state is committed.
Selected independent task while s4 is confirmed live: admit explicit controlled
partial M2 artifacts to oracle-lattice diagnostic,with artifact-derived identity,
frozen callable origin and verified metadata/aggregate evidence. Todo: add pure
contract helpers and regressions,run bounded metadata tests,commit;after s4 ends,
run ordinary errors then conditional geometry under serialized native caps.
V-003/V-014/V-018 require actual architecture/config/weights/frozen imports;
V-020 uses inspect.unwrap;V-021 preserves original partial/receipt status;
V-023 raw diagnostic copies stay byte-checked. Visibility/labels/optimizer and
running controller/training catalogue stay unchanged. Historical geometry scope
retains old sources;new format2 scope required. Diagnostic label input must remain
explicit and cannot be described as automatic accuracy or model capacity bound.
Candidate conditional geometry format2 ready outside frozen neural catalogue;
V-024 OPEN: old collector trusted oracle totals and did not bind full native
command/output. Ten metadata regressions use adapted preserved pilot data;
no neural forwards. NEXT bounded gate: .venv/Scripts/python.exe
scripts/run_bounded.py --memory-mib 128 --wall-seconds 30 --work-items 10
--receipt runs/geometry-evidence-tests-r1.json -- .venv/Scripts/python.exe
-m unittest discover -s tests -p test_geometry_evidence.py -v
Logs runs/geometry-evidence-tests-r1.log/.stderr.log. Expected<96MiB;native
pressure/Job/CPU guards;modest read-only metadata job can coexist with s4.
Acceptance: explicit partial/source/command/population/aggregate/masking refusal
fixtures,then actual serialized saved-artifact conditional forwards/curation.
Geometry r1:10 cases,1failure+2errors because candidate omitted preserved valid
768MiB diagnostic envelope. Production contract now accepts640/768/1024MiB;
no training cap changed. Failed r1 retained;14216/22760/16004 all dead.
NEXT geometry-evidence-tests-r2.json/.log/.stderr.log with same128MiB/30s/work10,
expected<96MiB and native guards. Source checkpointed before rerun.
Geometry metadata r2 GREEN10/10,.113s,35,180,544B/128MiB/30s;4244/36504/5068
and matching tests gone. Sourceeaf5bee. NEXT curator exact command array in
runs/geometry-contract-curation-r1-command.json: bounded128MiB/30s/work10,
record_test_gates --name geometry-evidence-contracts --base eaf5bee
--gate geometry-evidence-tests-r2:10 and18 source/fixture selections. Logs
runs/geometry-contract-curation-r1.log/.stderr.log,receipt same-prefix.json;
sentinel docs/evidence/geometry-evidence-contracts/scope.json. Expected<96MiB,
native guards;no model imports. Preserve r1 failure in new scope folder.
## Verified controller/recovery milestone

Final16 source-bound controller unit cases PASS1.275s,57,405,440B/192MiB/60s:
docs/evidence/segment-controller-contracts/scope.json,b8eb48a,20 source/fixture
hashes. Metadata fixtures isolate source-byte checks;no model forward/update.

Actual native joint controller proof:
docs/evidence/m3-controller-recovery-r1/scope.json. Partial1/3,resumed3/3(two new),
continuous3/3;parent times13.512/11.658/11.088s and aggregate peaks
526,790,656/531,968,000/532,938,752B under1280MiB/120s;children640MiB.
Source b6bb05f3a82b68ae41f5fb6ba8a256a89ddb69ed94c917807511bb636cf34255;
pilot/M3/capacity64. Exact data/loss/terms/grad/eval/probes,modes/signature;
model/Adam/CPU-RNG payload byte-identical510be7353c6819de3a79aa4d9c8a5224945cb8bce8b5ad996ef0510d658767d9;
inference5a69097af40f0660ec464e795e40c4b7085ecc8d69bfb63b588ec7e6322b491c.
Seven data rows+81 saved probes replayed across three cases;all processes dead.
Three updates establish integration only,not accuracy/long-memory acceptance.

Native proof-r2 PASS.413s,72,044,544B/128MiB/30s/work7;source/test scope in same
folder. Four copy-contract cases PASS.076s,35,454,976B/128MiB. V-022/V-023 CLOSED
on named gates. Initial proof-r1 failure preserved: new checker had mistaken
normalized current manifest JSON for raw bytes and omitted parent mappings.
Current manifests compare exact JSON values;other selected artifacts raw bytes.
No training rerun. All V-001..V-023 CLOSED;no whole-suite claim. M2 s4 is live above;no other owned heavy Job.

## Prior pressure recovery and current quality

s3 ancestor/child both memory_pressure125;root1215.374s,978,792,448B/1280MiB;
child1214.819s,942,940,160B/1GiB. Host31,537,569,792B crossed31,368,351,744B;
no crash;all related processes gone.1172 logged,1152 banked;20 later updates
lack committed optimizer state. No final1152 inference/accuracy result.
File proof docs/evidence/m2-stream-accuracy4096-s3-stop/scope.json.
Actual frozen restore docs/evidence/m2-stream-accuracy1152-recovery-r1/scope.json,
6.428s,477,442,048B/512MiB;7,694,010params,238 Adam states at1152,RNG/modes/prefixes,
zero forwards/updates. Tensor SHA c27f05730eae13ca75b56ddf33aea6b9d96c19fc36755a66024d3d40acadf861;
pointer manifest c51fb2d140e201e63e7d915249f13232eff93f71fecc6bb44ad5e1bfc0e0d23b.

Last FINAL M2 inference s2 step_partial704/4096,weights36f3dcf78ed807d021a5c8c0c5de47440a9032d208cc44dd396aa762fa04adda.
Actual27-case format2 report docs/evidence/m2-stream-accuracy704-errors-r3/scope.json:
all saved metrics reproduced,weights unchanged;dev384/401TP,17FN,326FP,
0/18 exactboards,0/36 layers,geometry1.84647cells. Learning inadequate.

M3 sourcef542096 has68 scoped integration checks;pilot/larger full-square warm
profiles and driver6==3+resume3 recovery pass. Actual larger two-update format2
report docs/evidence/m3-stream-large-profile-errors-r1/scope.json reproduces all27
metrics;18 readable components,0exact;plumbing only. Format2 diagnostics35 source-
bound cases pass in error-evidence-contracts. API stays needs_review/unsupported.

M2 adequate learning,M3 joint learning/long varied-shape memory,broad rendering,
real review,scratch/pretrained/direct comparisons,calibration/export and>=95%
exact complete-state release evidence remain OPEN.166 user images are unlabeled
drafts,zero training/scoring eligibility. Unknown!=empty,independent layers,
bottom-first rows,no guessed footprint erasure. Preserve full objective.
