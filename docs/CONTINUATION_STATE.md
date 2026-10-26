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

One-image inspection API is verified and committed at 4b1e51e:
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
Inspection milestone committed as 4b1e51e. The immediate launch guard found a NEW
release build (cargo 29028/30064, rustc 24304) and refused before Start-Process.
The 128-step run was NOT launched; no receipt/output was created. Wait for an
actually free resource window, recheck exact processes, then use the command
above and verify PID. Do not kill or modify another owner's build.
Compare data hashes and the first 16 loss steps to establish a controlled repeat;
report the unused caller-module delta explicitly. Read M2_LEARNING_ACCEPTANCE.md.

No vision background process is running. M2 learning acceptance stays OPEN;
queue/hold joint state, larger accuracy model, broad data/real reviewed corpus,
calibration, export and release quality still remain. Do not shrink the objective
or mark it complete on these partial diagnostics.

Last housekeeping PASSED: synchronize the local editable install metadata to source
version 0.3.0 (core library bytes/dependency versions unchanged): run_bounded.py
--memory-mib 512 --wall-seconds 120 --work-items 1 --receipt runs/m2-sync-final.json
-- uv sync --frozen --extra train. Logs runs/m2-sync-final.log / .stderr.log.
Measured 3.154 s / 115,916,800 B peak, within 512 MiB / 120 s / one frozen sync.
Distribution/source version equality 0.3.0 verified by runs/m2-env-validate.json;
torch 2.14.1+cpu, numpy 2.5.4, Pillow 12.3.0 match the lock. Environment receipts
are retained under docs/evidence/m2-environment/.
Goal remains active for the next continuation. No training or vision background
process is running. The next required learning evidence is the 128-step study
when resources permit. If the release build is still live, a safe independent
source task is the real-image corpus annotation/review auditor under scripts/:
draft vs reviewed labels, opt-in provenance, session/family split integrity and
frozen image/state hashes. Do not modify the model/loss/preprocessing/pilot driver
before the controlled comparison, and do not turn user images into unchecked
training labels. That source task advances M0/M4 while the M2 resource window waits.
