# Current checkpoint — 2026-10-26

Repo C:/Users/micha/Code/tetris-state-vision; codex/layout-diversity,no remote.
Goal ACTIVE and unfinished. Neighboring engine/apps/run data untouched. Local
commits authorized; no push/deploy/publish/deletion. One heavy model Job at a time.
Native Job cap, OS commit guard and one-thread import policy remain armed.

## Immediate state and next action

M2 s3 STOPPED by actual host commit pressure, not complete. Both ancestor and
child receipts terminal memory_pressure/125. Root1215.374s,978,792,448B peak
under1280MiB; child1214.819s,942,940,160B under1GiB. OS commit31,537,569,792B
crossed stop31,368,351,744B. All matching s3 Python processes absent. Last
progress1172; latest-training pointer selects1152 (manifest c51fb2d1...),20 later
updates lack committed optimizer state. No final summary/inference checkpoint.
Never relabel1172 as saved or run an inference report on this incomplete output.

Selected task: preserve/verify the pressure-stop bank with existing
scripts/record_stream_stop.py --name m2-stream-accuracy4096-s3 under128MiB/30s/
work1172,expected<96MiB. Receipt/logs runs/m2-stream-accuracy4096-s3-stop-curation-r1
(.json/.log/.stderr.log). Evidence target docs/evidence/m2-stream-accuracy4096-s3-stop/.
Require exact snapshot/tensor/prefix hashes and terminal saved PIDs; this alone
is not semantic optimizer restoration. Check metadata gate before any new launch.
Stop metadata gate PASS.441s,36,564,992B/128MiB; launcher22556/supervisor15916/
child9664 and matching processes gone. Bank1152 tensor SHA
c27f05730eae13ca75b56ddf33aea6b9d96c19fc36755a66024d3d40acadf861 and all three
ledger prefixes verified; five saved training PIDs dead. V-021 CLOSED on actual
scope with terminal child receipt accurately preserved. No semantic restore yet.
Fresh host commit29,225,660,416B after stop,available9589MiB; pressure can recur.
Do not restart training automatically or change any guard/cap. No host crash seen.

NEXT semantic restore only:512MiB/45s/work1,expected<480MiB from earlier same-size
446MiB restore. script probe_parked_recovery.py --run runs/m2-stream-accuracy4096-s3
--output runs/m2-stream-accuracy1152-recovery-r1.json. Native receipt
runs/m2-stream-accuracy1152-recovery-r1-job.json and matching.log/.stderr.log.
No new forward/update. Current commit30,270,418,944B fits512MiB reservation but
does not provide a stable1GiB model window. Verify exact command/PIDs/terminal
receipt and freeze its proof before considering any model report.
Semantic restore PASSED6.428s,477,442,048B/512MiB,238 Adam parameter states,
all at1152,7,694,010 model parameters, original121fd05a source/pointer/tensor
identities,0 forwards/updates. No matching recovery processes after exit.
Evidence docs/evidence/m2-stream-accuracy1152-recovery-r1/scope.json. No final
inference checkpoint or1172 tensor-equality/quality claim. Training remains stopped.

Next serialized model checks, only if fresh native headroom admission succeeds:
1. New format2 ordinary M2 report on ORIGINAL saved704 runs/m2-stream-accuracy4096-s2,
   output m2-stream-accuracy704-errors-r3.640MiB/120s/27forwards,expected<600MiB.
   Native runs/m2-stream-accuracy704-errors-r3-job.json/log/stderr. Compare ordinary
   metrics/diagnostics/totals to prior704 report, then scripts/record_stream_errors.py
   --name m2-stream-accuracy704-errors-r3 under128MiB/30s/work27 (expected<96MiB).
2. Actual M3 replay on runs/m3-stream-large-profile-r1, NEW output
   m3-stream-large-profile-errors-r1, same1GiB/120s/27forwards,expected<600MiB but
   experimentally bounded. Then same128MiB/30s/work27 collector. These2-step
   weights establish replay/plumbing only, not trained accuracy or learning.
Use fresh receipts/output, hidden logs and checkpoint entry/PID verification.
M2 report cap lowered from planned1GiB to640MiB because prior same-model reports
peaked551,600,128B (~526MiB); full model/all27 probes unchanged. Host around
30.30GB commit fits640MiB reservation while1GiB is marginal. Kernel/OS guards
remain armed. NEXT launch this M2 report; verify process and terminal outcome.
If a headroom gate parks, preserve receipt; do useful source work rather than
repeatedly launching into unchanged pressure. No model process currently live.

## New verified diagnostic milestone

Source77128c4: format2 stores every ordinary per-case metric; new
scripts/record_stream_errors.py binds native command/output/work/memory, actual
architecture/config/weights, partial tag, every sample/ancestry/recipe/metric and
recomputed totals. No training source/labels/model/guard changes. Unit gate
runs/error-evidence-tests-r1:35/35 in1.906s,248,700,928B/512MiB/60s, launcher28252,
supervisor15636,child25044 gone.8 new metadata cases+27 diagnostic cases.
Source scope docs/evidence/error-evidence-contracts/scope.json:32 hashes,
metadata curator runs/error-evidence-curation-r1 PASS.651s/72,331,264B under
128MiB/30s; launch2804/supervisor30268/child31220 gone. Preserve/commit its raw
receipts and docs. Actual new-format M2/M3 forward/curation integration pending.

Prior source577c7ec diagnostics27 pass:12 M2+15 joint pure/tensor/mask/legacy/
decorator-origin cases, docs/evidence/joint-error-diagnostics-full/scope.json.
Earlier12-source scope refers to SAME27, not54. V-019/V-020 CLOSED on named27.
All V-001..V-021 CLOSED on named historical gates, not a fresh whole-suite claim.
See docs/SAVED_ERROR_REPORTS.md and M3_COMPONENT_TARGETS.md for contracts/limits.

## Immutable training histories

Original M2 catalogue121fd05a2f75e30b0c738dac592c67c6d015aa6043a43e4032d91ac662de794b.
Any continuation MUST execute s3/code/scripts/train_m2_stream.py, original fixed
4096 recipe/Adam/RNG/cursor/checkpoint64, not current M3 package. Next candidate
s4 resumes s3 bank1152 only after stop audit/semantic recovery/headroom review;
use NEW names, bounded finite segment, no automatic restart. Preserve20 later
unbanked data/loss rows for a subsequent deterministic trajectory comparison.
Supervisor d2fb1478940b7b606073dbd75aa3746098fe7ede15f306082ec1f762d09c20db unchanged.

Last FINAL M2 model is s2 at704/4096,step_partial,weights36f3dcf78ed807d021a5c8c0c5de47440a9032d208cc44dd396aa762fa04adda.
Docs/evidence/m2-stream-accuracy4096-s2/:704recipes+27probes replayed. Ordinary704
error report:384/401 devpositive cells found,17FN,326FP;0/18 exact boards,0/36
layers,each type18/18 wrong;geometry1.84647cells. No adequate learning/real claim.

M3 catalogue b6bb05f3a82b68ae41f5fb6ba8a256a89ddb69ed94c917807511bb636cf34255.
Pilot capacity64 warm/reload profile558,497,792B/640MiB; larger7,997,016 parameters,
264 initialized tensors/63,977,184B Adam,942,903,296B/1GiB,43.663s. Source-pinned
scope docs/evidence/m3-stream-large-profile-r1/. Native pilot6==3+resume3 exact
model/Adam/data/loss/grad/eval/RNG/modes,95.148s; m3-stream-recovery-gate-r1/.
Joint learning and long varied-shape memory remain OPEN. Original model source
f542096 has68 scoped unit/contract checks; later analysis changes are separate.

## Remaining objective

API remains needs_review/unsupported,queue/hold unknown.166 approved screenshots
are unlabeled drafts with zero training/scoring eligibility. Unknown!=empty;
layers independent,rows bottom-first,no guessed footprint erasure. Broad render
coverage, real review, scratch/pretrained/direct comparisons, calibration/export
and>=95% exact complete-state release evidence remain OPEN. Do not mark complete.
