# Current checkpoint — 2026-10-26

Repository C:/Users/micha/Code/tetris-state-vision, local Git, no remote.
Goal ACTIVE: complete the standalone model thoroughly, commit good checkpoints.
No vision background process is running. No run data was deleted.
Latest commits: 2486cb7 corpus integrity; c2572bd user-approved draft stress set.

Latest verified milestone: corpus preparation/audit/seal/verify, 26/26 checks in
3.620 s, raw peak 62,005,248 B under 256 MiB / 60 s / 26 finite cases.
Source hashes/logs: docs/evidence/corpus-integrity-final/scope.json. V-008 and
V-009 closed on their named gates; V-001..V-009 all closed. Supervisor 27196
and child 29160 verified exited. Curator passed under 128 MiB;
runs/corpus-curation-r3.json. The earlier 25-case scope is historical.
Todo complete: bounded admission, draft/review/permission policy, grouping and
duplicates, externally pinned held-out seals, adversarial/actual CLI gates.
See docs/CORPUS.md. Model/math/pilot code was not changed.
User authorized the 166 images in neighboring tetrguide/data/raw_screenshots,
warning correct board answers are uncertain. Imported as draft references to
data/real-pilot-drafts/manifest.jsonl; 35,553,531 B, no exact pixel duplicates,
0 eligible and no annotations/permissions. Import 4.504 s / peak 118,034,432 B;
audit 3.966 s / 117,944,320 B; profile 13.398 s / 117,035,008 B. Each ran under
512 MiB with wall/work caps. Supervisor/child PIDs 7260/29604, 23216/29052 and
20268/2572 verified exited. Seven private pages + metadata at
runs/real-drafts-overview/. Representative pages 1, 4, 7 visually checked.
All import/audit/profile/curation todos complete. Aggregate receipts/source/page
hashes: docs/evidence/real-drafts/scope.json; raw per-frame data/images stay
ignored/private. See docs/REAL_STRESS_SET.md. No board correctness claim.
No real images admitted for training. Never reuse legacy
detector labels as full-state ground truth; they mix active and locked in some
cases. Tool tests contain invented fixtures only. Real training requires a
separate reviewed-data integration; the existing optimizer is synthetic-only.

Next acceptance-gated task: M2 128-step controlled learning study. Another
owner's release build remains live (cargo 30852/15708, rustc 20672 last check).
Host check: 29,079,207,936 B committed; rustc private 1,927,860,224 B. Keep
the conservative serialized learning window; no new optimizer launched.
Do not kill or modify it. The 128-step study has NOT launched; output/receipt
do not exist. Immediately recheck exact cargo/rustc/fusion-bridge processes.
If the resource window is free: expected <768 MiB, kernel 1 GiB ceiling,
180 s outer / 150 s inner / 128 steps / one thread / <=64 MiB tensor cache.
Command through hidden detached Start-Process with bootstrap Python:

scripts/run_bounded.py --memory-mib 1024 --wall-seconds 180 --work-items 128
--receipt runs/m2-learn128-job.json -- .venv/Scripts/python.exe
scripts/train_m2_pilot.py --output runs/m2-learn128 --steps 128 --states 3
--max-side 384 --threads 1 --wall-seconds 150 --evaluate-every 16

Logs runs/m2-learn128.log / .stderr.log; progress/summary/checkpoint in output.
Verify exact PID after launch; record it here before waiting. Measured 16-step
throughput projects ~59 s. Compare manifest/data hashes and first 16 losses
with the retained run to establish a controlled repeat. Do not change model.py,
losses.py, preprocess.py, checkpoint.py or the pilot driver before comparison.
The core four modules remain byte-identical to 8d9d0f2; driver to e6adce8.
New unused runtime.py changed aggregate source hash, not optimizer computation.
Read docs/M2_LEARNING_ACCEPTANCE.md; do not wait long on the external build.

Evidence so far: 85 numerical/core/resource checks at 8d9d0f2; 2 optimizer
admission checks; 7 runtime checks plus actual one-image checkpoint call;
26 corpus checks. These are separate scoped gates, not one unified 120 run.
16-step learning: loss 4.4767 -> 1.6126; geometry 2.63 -> 1.43 cells;
train exact locked boards 2/6, development 0/6. Not precise/general parsing.
Receipt docs/evidence/m2-learn16/receipt.json; weights retained at
runs/m2-learn16/checkpoint/weights.pt, SHA256
56d23804eec0861d11ab1c08941a95cab4d38516e2db4e56ebc9d126e9bb0b4a.
Runtime is review-only, queue/hold unknown; docs/evidence/m2-runtime/scope.json.
Local .venv synchronized to 0.3.0; Torch 2.14.1+cpu, NumPy 2.5.4, Pillow 12.3.0;
docs/evidence/m2-environment/ records the frozen install/version checks.

M2 learning remains OPEN. M3 complete queue/hold state, accuracy-sized model,
broad synthetic/real data, human reviewed pilot/baseline, calibration, export
and release remain. Do not mark the full goal complete on these diagnostics.
