# Current checkpoint — 2026-10-26

Repo C:/Users/micha/Code/tetris-state-vision, local Git, no remote. Working from
user's lower-memory reply; goal service last reported BLOCKED (user-controlled
resume). Previous blocked audit reset: actual host commit fell from31.7GB to
20.6GB. No project job running; no run data deleted. Neural math unchanged.

Selected task: fresh warm admission, then768-step fresh-state learning study.
Todo: commit/curate22 scoped gates; current-runner square profile; finite study;
exact ledger/probe replay and metrics; commit. Then integrate verified recovery
primitive with an exact uninterrupted/resumed stream gate and fresh resource
characterization. No recovery is wired into the trainer yet.

Latest verified work:
- Recovery7 GREEN: runs/training-snapshot-tests-r3.stderr.log,4.103s,
  332,562,432B raw Job peak under512MiB/45s. Exact6 continuous vs3+resume3
  AdamW/dropout gives identical model, optimizer and RNG. Interrupted publication,
  corrupted tensors/manifest/ledger prefixes and source/recipe/order mismatch fail.
- V-012 CLOSED: r2 failed during OpenBLAS import before fixtures; pre-import
  OMP/MKL/OpenBLAS defaults1 repair it at the same512MiB cap. New standing
  instrument (third test-instrument defect): AGENTS.md plus host8->child1 probe.
  Native resources6 GREEN3.116s/151,027,712B; pressure5 GREEN2.147s/75,378,688B;
  admission4 GREEN.275s/52,604,928B. Runs/v012-*-tests.{json,log,stderr.log}.
  These22 cases are separate scopes, not a whole-suite or accuracy claim.
- Supervisor SHA6f46c8389de696d9fcddc00803675e3eb1898494369bb6e0c3fec1034d912a6f.
  Kernel aggregate cap, kill-on-close, exact named Job and GetPerformanceInfo/.2s
  pressure guard remain armed. Never use a historical pre-pressure supervisor.
  Fresh source catalogue/profile required: old admission cannot admit new runner.
- All V-001..V-012 closed on named gates. Gate curation pending current commit.

Next hidden detached launch, expected<600MiB, native640MiB/150s outer/110s inner,
2 optimizer steps, one CPU thread; guard parks on OS pressure or counter failure:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib640
--wall-seconds150 --work-items2 --receipt runs/m2-stream-thread-probe-job.json
-- .venv/Scripts/python.exe scripts/train_m2_stream.py
--output runs/m2-stream-thread-probe --model pilot --steps2 --profile-square
--wall-seconds110 --development-groups6.
Logs runs/m2-stream-thread-probe.log/.stderr.log. Record exact PID/tree after
spawn; verify terminal receipt and death. Curate only native pass plus warm
backward with full Adam state and all-final-tensor reload parity. Then use that
matching admission for a new768-step run name (previous learn768 parked125,
no child; never reuse receipt). Frozen code archive permits independent source
work while a run uses those exact archived imports.

Retained learning evidence:
- 421de02 controlled128: exact16-step prefix, training boards2/6, development0/6.
- 331ac58 saved511/512 inner-wall partial: training boards6/6, layers10/12,
  geometry.4106 cells fail joint fit; development0/6 and0/12. Not completed512.
  Weights ca4c0f34a81b8c7591fab7db0e92fd89d652a27dea015cf63473c3f7b780bcdf,
  runs/m2-learn512/checkpoint/. Neural math byte-identical to original M2.
- 72cde65 local reading:36 changes/18 matched groups, direction36/36,
  targeted20/36, exact changed boards18/36; two unrelated cells changed.
- eb11572 warm512-square2-step profile:9.553s/594,829,312B under640MiB,
  142 initialized tensor states,5,177,352B Adam state, every final tensor reload
  exact. Historical catalogue42148a... no longer admits current runner.
- M3 component labels8 GREEN incl paired hidden queue/hold; no heads/loss/runtime
  integration. docs/evidence/m3-component-targets/scope.json. Unknown is not empty;
  keep locked/active/ghost separate, bottom-first; never infer hidden queue length.

User-approved166 uncertain-answer screenshots are unlabeled drafts only:
data/real-pilot-drafts/manifest.jsonl; zero training/scoring eligibility. Seven
private overview pages in runs/real-drafts-overview/. No legacy guess is truth.
Env .venv0.3.0, Torch2.14.1+cpu/NumPy2.5.4/Pillow12.3.0. Accuracy configuration
still unprofiled/untrained. M2 broad learning, M3 joint state, broad data, reviewed
real pilot/baseline, calibration/export/release remain OPEN; goal not achieved.
