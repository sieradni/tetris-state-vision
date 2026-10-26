# Current checkpoint — 2026-10-26

Repository C:/Users/micha/Code/tetris-state-vision, local Git, no remote.
Goal ACTIVE. No vision background process running. Previous turn made progress.
No run data deleted. User-approved screenshot folder remains unlabeled only.

M2 128-step study COMPLETE, not a learning/quality acceptance. Curated evidence
docs/evidence/m2-learn128/receipt.json and comparison.json. 50.537 s, raw peak
479,604,736 B under 1 GiB / 180 s / 128 steps / one CPU thread. Weight SHA256
46787c4f236c421dc7b3eb693665c1c8d09688c28f7330a28c2d0abe8907967e
at retained runs/m2-learn128/checkpoint/. 12 image/label identities and every
optimizer step verified. Exact 16-step prefix/data/initial metrics confirmed.
Only src/ delta since e6adce8 is unused runtime.py; core/driver unchanged.

Training loss 4.4767 -> .34977, geometry .3645 cells, exact boards 2/6 (empty
pair only), exact known active/ghost 10/12. Development 0/6 and 0/12 layers,
geometry 1.3768 cells. Independent checkpoint confusion counts and per-case
layers/stages agree with summary; examples/m2-learn128-locked-errors.png visually
inspected along with lattice preview. Analysis peak 811,859,968 B exceeded
<512 MiB forecast but stayed within 1 GiB; future analysis forecast <900 MiB.
All PIDs 10268/18248/25716 (training), 25200/17312 (analysis) verified exited.
Current task complete: finite run, exact controlled comparison, per-case
diagnostics, receipts and local commit. Read M2_LEARNING_ACCEPTANCE.md.

Next selected task: same-initialization 512-step finite memorization diagnostic.
Predeclared fit criterion: train exact locked boards 6/6, known layers 12/12,
mean geometry <.25 cells. Still no generalization/full-state/real quality claim.
If it fails, inspect model sampling/targets/gradients before longer training.
Todo: [ ] bounded run; [ ] compare first128 prefix/data; [ ] independent per-case
analysis; [ ] acceptance verdict and local verified commit.
No architectural/label change; V-001/002/005/006/009 observation masking,
V-003 computed identity and V-004/V-007 named native resource guard apply.

Before launch recheck no cargo/rustc, no bridge >256 MiB, host committed <28e9 B.
Do not touch another owner's process. Expected <768 MiB from measured 128 run,
native 1 GiB aggregate Job cap, 300 s outer / 240 s inner / 512 optimizer steps,
one thread, <=64 MiB cache. Measured ~.36 s/step projects <200 s.
Hidden detached bootstrap Python command:
scripts/run_bounded.py --memory-mib 1024 --wall-seconds 300 --work-items 512
--receipt runs/m2-learn512-job.json -- .venv/Scripts/python.exe
scripts/train_m2_pilot.py --output runs/m2-learn512 --steps 512 --states 3
--max-side 384 --threads 1 --wall-seconds 240 --evaluate-every 32
Logs runs/m2-learn512.log/.stderr.log; progress/summary/checkpoint in output.
Record source/host admission to runs/m2-learn512-admission.json, verify PID and
record it here. The output/receipt do not exist yet. Keep model/math/driver
unchanged through this comparison. While it runs: plan diverse synthetic
training input path; do not edit runtime src while worker hashes are pinned.

Current numerical/evidence gates: 85 core/model/resource at 8d9d0f2; 2 training
admission; 7 runtime plus actual single-image call; 26 corpus. Separate scopes,
not one unified120. V-001..V-009 CLOSED on named receipts.
User-approved 166 images: data/real-pilot-drafts/manifest.jsonl, 0 labels or
training/scoring eligibility, seven pages at runs/real-drafts-overview/.
Aggregate evidence docs/evidence/real-drafts/scope.json; overview1/4/7 visually
checked. Never treat legacy labels/proposals as truth. User warned board
answers uncertain. Real-data training requires a separate reviewed integration.

M2 broad learning, M3 queue/hold joint state, accuracy-sized model, diverse data,
reviewed real pilot/baseline, calibration/export/release remain OPEN. Whole goal
is not complete. Local env .venv 0.3.0, Torch2.14.1+cpu/NumPy2.5.4/Pillow12.3.0.
