# Current checkpoint — 2026-10-26

Repo C:/Users/micha/Code/tetris-state-vision;codex/layout-diversity,no remote.
Goal ACTIVE,unfinished. Local commits authorized;no push/deploy/publish/user-data
deletion. Neighboring engine/apps/data untouched. One heavy model Job at a time;
native RAM/pressure/CPU guards remain armed. Do not spawn subagents.

## Live original baseline: inspect existing job before any new model work

s4 last observed step1618,elapsed619.636s,training;worker30892 verified. Process
chain:launch31508/root33104/controller13492+30656/child launcher27512+supervisor31064/
worker4208+30892 (Python redirectors included). Original s3 archive executes.
Controller SHA11a0a81aeccddbb62451c46d24d2d4a844d762653b44cdb7b5edbb168bec0ac4
MUST stay unchanged while active. Native supervisor SHA
 d2fb1478940b7b606073dbd75aa3746098fe7ede15f306082ec1f762d09c20db unchanged.

Exact detached command:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 1280
--wall-seconds 2100 --work-items 512
--receipt runs/m2-stream-accuracy4096-s4-pipeline-job.json --
.venv/Scripts/python.exe scripts/run_stream_segment.py
--name m2-stream-accuracy4096-s4 --target-steps 4096 --updates 512
--admission runs/m2-stream-accuracy-probe --resume-from runs/m2-stream-accuracy4096-s3

Logs runs/m2-stream-accuracy4096-s4-pipeline.log/.stderr.log;
state runs/m2-stream-accuracy4096-s4-pipeline/progress.json,result.json;
child runs/m2-stream-accuracy4096-s4-job.json and s4.log/.stderr.log;
evidence sentinel docs/evidence/m2-stream-accuracy4096-s4/scope.json.
Child1024MiB/1860s outer/1800s inner,checkpoint64,CPU1/cache16;serialized
curator128MiB/120s. Expected<1100MiB aggregate. Start1152,at most512new ->1664,
fixed target4096. Source121fd05a2f75e30b0c738dac592c67c6d015aa6043a43e4032d91ac662de794b;
seed20261026,LR.001,eval128,checkpoint64,groups6,capacity0. No training edits.
Current memory window good,latestcommit27.3GB/free disk51.0GB;recheck beforelaunch,
require>40GB disk. Never close apps/delete data/alter guards to bypass admission.

At1280 dev4/18 exactboards,0/36 layers,2.07288-cell geometry;intermediate logged
evaluation only,no final1280 inference result. New saved snapshots observed up
to1600;last separately hashed snapshot1216. Final accuracy must use s4 artifact.
Exact20 recomputed updates1153..1172 and all1172 prefix records verified:
docs/evidence/m2-stream-accuracy1172-recomputed-r1/scope.json. No old1172 tensors.

NEXT poll existing root receipt and worker30892. If controlled completion,verify
all owned PIDs/commands gone,child/curator/root statuses,actual summary/config/
weights/source and curated scope;preserve parent/curator receipts/logs/results.
If pressure125,record actual logged/banked cursor and raw statuses,verify death,
then verify recovery;do not restart merely from a stale receipt/observation.

## Selected diagnostic task after s4 terminates

Ordinary error replay first: scripts/probe_stream_errors.py
--run m2-stream-accuracy4096-s4 --output runs/m2-stream-accuracy1664-errors-r1
under640MiB/120s/work27,expected<600MiB from same model's measured576,446,464B.
Then record_stream_errors.py --name m2-stream-accuracy1664-errors-r1 under
128MiB/30s/work27,expected<96MiB. All outputs/receipts/logs must be new;record
exact launch/checkpoint/PID before waiting. Output label never substitutes for
actual summary cursor. No model job alongside live s4.

New conditional geometry format2 (sourceeaf5bee) is ready outside training
catalogue: explicit --allow-partial,actual M2 architecture/config/weights/status,
frozen callable origins incl inspect.unwrap,full native command/output/work,
unique cases/ancestry and recomputed totals;oracle labels remain diagnostic only.
See CONDITIONAL_GEOMETRY.md. After ordinary replay,run
scripts/probe_stream_geometry.py --run m2-stream-accuracy4096-s4
--output runs/m2-stream-accuracy1664-geometry-r1 --allow-partial
under1024MiB/120s/work54,expected<850MiB. Curate with record_stream_geometry.py
--name m2-stream-accuracy1664-geometry-r1 under128MiB/30s/work54,expected<96MiB.
Serialize all model work;no final artifact means no ordinary/oracle replay.

V-024 OPEN: historical geometry collector trusted oracle totals and incompletely
bound native command/output. Ten new metadata cases GREEN.113s,35,180,544B/128MiB:
docs/evidence/geometry-evidence-contracts/scope.json,18 source/fixture hashes on
eaf5bee. Curator.493s,76,550,144B/128MiB. All test/curator PIDs gone. Initial
r1 rejection of valid historical768MiB envelope retained;now640/768/1024 admitted.
No live ceiling changed. Actual new paired-forward/replay/curation gate still
required to close. Historical pilot geometry keeps its old source scope.
V-003/V-014/V-018/V-020/V-021 provenance,partial status and frozen import rules
apply;V-023 raw copies remain byte-checked. No labels/model/optimizer changed.

## Preserved evidence and remaining objective

Prior s3 pressure stop:1172 logged/1152 banked,both receipts125;file/restore
proofs in m2-stream-accuracy4096-s3-stop and m2-stream-accuracy1152-recovery-r1.
Actual frozen model/238 Adam states/RNG restore PASS6.428s/477,442,048B;no final
1152 inference. Last FINAL M2 inference is s2 step_partial704/4096,weights
36f3dcf78ed807d021a5c8c0c5de47440a9032d208cc44dd396aa762fa04adda.
Verified format2 errors at m2-stream-accuracy704-errors-r3:dev384/401TP,17FN,
326FP,0/18 exactboards,0/36 layers,geometry1.84647cells. Learning inadequate.

M3 neural sourcef542096/catalogueb6bb05f3 has68 scoped checks,pilot/larger warm
profiles and exact driver6==3+3 recovery. Native controller3==1+2 proof plus16
source-bound controller and4 copy-contract checks are committed in
m3-controller-recovery-r1 and segment-controller-contracts. All V-001..V-023
CLOSED on named scopes;V-024 OPEN. No whole-suite or quality claim.

M2 adequate learning,M3 joint learning/long varied-shape memory,broad rendering,
real review,scratch/pretrained/direct comparisons,calibration/export and>=95%
exact complete-state release remain OPEN.166 user images are unlabeled drafts,
zero training/scoring eligibility. API needs_review/unsupported. Unknown!=empty,
layers independent,rows bottom-first,no guessed footprint erasure.
