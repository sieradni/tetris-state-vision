# Current checkpoint — 2026-10-26

Repository C:/Users/micha/Code/tetris-state-vision, local Git, no remote.
Goal remains ACTIVE. Continue the standalone image-to-state model thoroughly;
commit verified milestones. No complete-state or real-image release claim.

Verified source milestones: f5ed97f foundation; 3cd063b layout/panel diversity;
b06cdda canonical-frame repair; 8d9d0f2 scratch model/numerical gates (85 checks).
Bug ledger has V-001..V-007 closed on named green runs. CPU-only PyTorch is an
optional dependency. Current runtime source digest remains f118a239... from the
numerical gate. Accuracy config is separate and still untrained/unprofiled.

First optimizer pilot is verified:
- runs/m2-learn16, 16/16 steps, 3 latent training states x clean/no-grid = 6 images;
  3 separate development states x 2 appearances = 6 images. Variable board origin
  and spacing, density strata dense/sparse/empty; full-image learned lattice.
- One CPU thread, no workers/prefetch, max side 384, tensor cache 17,252,352 B
  (64 MiB budget). Kernel cap 1 GiB / outer wall 120 s / inner 90 s / 16 steps.
- Runner 19.414 s, raw peak 484,966,400 B. Saved/reloaded predictions identical.
- Loss 4.4767 -> 1.6126; occupancy 0.5969 -> 0.2248; mean geometry error
  2.63 -> 1.43 cells. Training raw exact boards 2/6; development 0/6. Geometry is
  still visibly misregistered. These are uncalibrated head diagnostics only.
- Curated receipt/data hashes/curve/progress: docs/evidence/m2-learn16/receipt.json.
  Actual weights stay in retained runs/m2-learn16/checkpoint/weights.pt,
  SHA256 56d23804eec0861d11ab1c08941a95cab4d38516e2db4e56ebc9d126e9bb0b4a.
  Preview examples/m2-learn16.png inspected. Driver's two admission fixtures pass
  (m2-train-admission.log); the shared runtime's 85-case gate remains unchanged.

Next acceptance-gated action: fresh 128-step study, same seed/data/config/initial
weights, evaluate every 16 steps. Measured 0.365 s/step including intermediate
checks projects ~59 s total; use 1 GiB / 180 s outer / 150 s inner / 128 steps,
one thread, same 64 MiB cache cap. Recheck other heavy jobs before launch.
Command planned: run_bounded.py --memory-mib 1024 --wall-seconds 180 --work-items
128 --receipt runs/m2-learn128-job.json -- .venv/Scripts/python.exe
scripts/train_m2_pilot.py --output runs/m2-learn128 --steps 128 --states 3
--max-side 384 --threads 1 --wall-seconds 150 --evaluate-every 16.
Log runs/m2-learn128.log / .stderr.log; progress inside output, final summary and
checkpoint sentinel likewise. It has NOT been launched yet. Commit the verified
16-step milestone before launching. Read M2_LEARNING_ACCEPTANCE.md for verdict.

No vision background process is running; all measured supervisors/children exited.
M2 learning acceptance remains OPEN; queue/hold joint state, broader data,
accuracy-model training, reviewed real images, calibration/export/release still
remain. Preserve the full objective and retain every existing run/artifact.
