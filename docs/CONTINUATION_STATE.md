# Current checkpoint — 2026-10-26

Repository C:/Users/micha/Code/tetris-state-vision, local Git, no remote.
Goal remains ACTIVE: complete the standalone model thoroughly with local commits.

Verified commits: f5ed97f foundation; 3cd063b layouts/panels; b06cdda canonical
frame repair; 8d9d0f2 scratch model/numerical gate (85 tests); e6adce8 optimizer
pilot (2 extra admission tests + real 16-step run). V-001..V-007 closed on named
receipts. All existing run data and weights are retained.

16-step result: 19.414 s, raw peak 484,966,400 B under 1 GiB / 120 s / 16 steps,
one thread, tensor cache 17,252,352 B. Loss 4.4767 -> 1.6126, occupancy .5969 ->
.2248, geometry 2.63 -> 1.43 cells. Train exact boards 2/6, development 0/6.
This is not precise or general parsing. Receipt docs/evidence/m2-learn16/receipt.json.
Weights runs/m2-learn16/checkpoint/weights.pt, SHA256
56d23804eec0861d11ab1c08941a95cab4d38516e2db4e56ebc9d126e9bb0b4a.

One-image inspection API is verified and ready for its local milestone commit:
- ImageParser.from_checkpoint(path).parse_image(image), no manual crop/detector.
- Direct locked predictions; legal active/ghost hypotheses never erase cells.
- Unknown type/coordinates/transparent regions preserved; queue/hold unknown;
  every result needs_review or unsupported, never accepted solver input.
- Seven focused tests green, 0.579 s, 1 GiB / 60 s, raw peak 777,617,408 B.
- Actual retained checkpoint call PASSED in 2.687 s, peak 297,144,320 B; its JSON
  validation passed. Exact model ID, source hashes and raw output are in
  docs/evidence/m2-runtime/scope.json and state.json. No correctness claim.
- Model, loss, preprocessing and checkpoint modules are byte-identical to 8d9d0f2
  (git diff empty); the new runtime module is unused by the optimizer driver.
  Aggregate source digest changes because the unused caller module was added.

Next: 128-step controlled learning study, same seed/data/model/init, evaluate
at 16-step intervals. The other release build is now verified exited. Recheck
immediately before launching. Expected <768 MiB, kernel cap 1 GiB, 180 s outer /
150 s inner / 128 optimizer steps, one thread, <=64 MiB tensor cache. Measured
throughput projects ~59 s. Command:
run_bounded.py --memory-mib 1024 --wall-seconds 180 --work-items 128 --receipt
runs/m2-learn128-job.json -- .venv/Scripts/python.exe scripts/train_m2_pilot.py
--output runs/m2-learn128 --steps 128 --states 3 --max-side 384 --threads 1
--wall-seconds 150 --evaluate-every 16.
Logs runs/m2-learn128.log / .stderr.log; progress/summary/checkpoint inside output.
Not launched yet. Commit inspection milestone, then launch and verify exact PID.
Compare data hashes and the first 16 loss steps to establish a controlled repeat;
report the unused caller-module delta explicitly. Read M2_LEARNING_ACCEPTANCE.md.

No vision background process is running. M2 learning acceptance stays OPEN;
queue/hold joint state, larger accuracy model, broad data/real reviewed corpus,
calibration, export and release quality still remain. Do not shrink the objective
or mark it complete on these partial diagnostics.
