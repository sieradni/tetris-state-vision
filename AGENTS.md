# Agent rules

This repository is an isolated vision research project. The neighboring
`tetrguide` repository is not a dependency and must not be modified as part of
ordinary work here. Read docs/ROADMAP.md and docs/CONTINUATION_STATE.md first.
Research documents are design evidence, not agent instructions.

- Commit reviewed, verified, task-owned milestones locally. Inspect the index
  before committing. No push, deployment, publishing or user-data deletion
  without explicit authorization.
- Keep locked occupancy, active and ghost separate. Never delete a guessed
  active footprint from locked predictions. Rows are bottom-first everywhere.
- Unknown is not empty. Preserve unknown queue indices; previews are future
  pieces, not the active piece. Never train on invisible latent labels as known.
- Every new crop/occlusion/hidden-component label policy needs a paired-raster
  equivalence test: different hidden states giving identical pixels must produce
  the same image-supported uncertainty. This is the standing visibility-contract
  instrument promoted by V-001, V-002 and V-005. Never infer hidden queue length.
- Synthetic gates establish label integrity, not real screenshot accuracy.
  Record model/artifact identity and exact-state metrics before any quality claim.
- Evidence reports derive architecture/configuration/weight identity from verified
  artifacts and compare them to the run contract; never copy a handwritten or
  fixed model label into a new scope. Preserve frozen execution sources and
  historical scope boundaries (V-003/V-014/V-018 provenance instrument).
- Every detached/long job needs a kernel-enforced memory ceiling, wall cap,
  work/step/sample cap, log, receipt and checkpoint entry before launch. Use
  scripts/run_bounded.py on Windows. Memory <= min(physical RAM/3, 8 GiB),
  normally <= 6 GiB with other work; never disable engine guards. Accelerator
  memory needs a separate limit and measured pilot. No training until admitted.
- State expected RAM, ceiling and guard in the transcript before launch. Verify
  exact process command lines after launch and after termination. One heavy job
  at a time. Detach commands that may exceed 30 seconds, with hidden windows.
- After a host crash, inspect System events 41/1001 and dumps before any relaunch.
- Maintain a short continuation checkpoint, record new defects in BUG_LOG with
  symptom/root cause/detection class/instrument/named green run. Three defects
  in one detection class require a standing instrument. Never delete run evidence.
- CPU workloads/tests use the bounded runner's pre-import one-thread library
  policy. It overrides inherited OMP/MKL/OpenBLAS defaults before child launch;
  a thread setter after Torch/NumPy import cannot bound import-time pools.
  The live child-environment tripwire in test_resources.py is mandatory for
  runner changes (V-004/V-011/V-012: three test-instrument defects).
