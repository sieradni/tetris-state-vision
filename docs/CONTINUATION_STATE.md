# Current checkpoint — 2026-10-26

Repo C:/Users/micha/Code/tetris-state-vision, local Git, no remote. Goal ACTIVE.
No vision background process running; no run data deleted.
This turn made concrete progress: controlled128 and saved511/512 studies.

M2 study128 committed at421de02. Same data/initial metrics/first16 optimizer
records exactly reproduced. Training exact2/6 (empty pair); development0/6.
Receipt/source scope docs/evidence/m2-learn128/{receipt,comparison}.json.

Second study reached inner240s wall after511/512 steps, saved checkpoint and
returned wall_partial/exit2. Outer244.005s, raw peak480,751,616B under1GiB/300s/
512-step cap/one thread. Do not call it completed512. Exact first128 curve
records/data/runtime/driver/initial metrics reproduce. Training locked6/6 but
known layers10/12 and geometry.4106 cells fail predeclared joint fit; dev0/6,
layers0/12, geometry1.3112 cells. Ghost row15 vs target10 on dense clean/no-grid
is the remaining training layer error. Source/weight/progress/logs/3 passing
partial-evidence fixtures: docs/evidence/m2-learn512/. Weight SHA256
ca4c0f34a81b8c7591fab7db0e92fd89d652a27dea015cf63473c3f7b780bcdf
at retained runs/m2-learn512/checkpoint/. Reload exact.

Training supervisor31360/launcher11480/worker9424 and analysis7284/child28924
verified exited. Analysis12images passed4.717s, raw peak811,851,776B under1GiB/
60s; forecast future analysis <900MiB, not <512. Current study/curation/diagnostic/
verdict todos complete. See M2_LEARNING_ACCEPTANCE.md; goal not achieved.

Next selected diagnostic: fixed-layout single-cell interventions on the actual
511-step artifact. Re-render training cases byte/pixel-identically; change one
visible locked cell away from active/ghost; assert exactly one supported label
and local raster patch changes, hold other components/layout fixed. No target
geometry in model.forward. Measure targeted change, unrelated changes, exact
counterfactual boards and lattice drift. This distinguishes memorized states
from local reading before more training or architectural changes.
Todo: [ ] guarded finite probe; [ ] per-case evidence; [ ] review/commit; then
choose targeted geometry/layer/data work. Prior bug interactions: V-001/002/005/
006/009 visible-only labels and no guessed layer subtraction; V-003 re-render/
weight/data hashes; V-004/V-007 named native guard and raw counters preserved.
No new observation policy. Keep core model/loss/preprocess/checkpoint unchanged
until the diagnostic is scoped. Legacy runtime stays review-only.

A new unrelated release build is live: cargo28392/8656, rustc18496/3460 last
checked. Recheck authority before any next training; don't touch their processes.
Small one-thread diagnostic can run within1GiB/120s/<=54 forwards; expected
<900MiB from measured checkpoint analysis, with a hard1GiB ceiling. No job
launched yet; write exact command/log/receipt/PID before waiting.

Earlier gates:85 core/model/resource at8d9d0f2;2 optimizer admission;7 runtime
plus actual one-image API;26 corpus;3 partial-evidence. Separate scoped runs,
not a unified123. V-001..V-009 CLOSED on named logs. Env .venv0.3.0,
Torch2.14.1+cpu/NumPy2.5.4/Pillow12.3.0.

User-approved166 uncertain-answer images remain unlabeled drafts only at
data/real-pilot-drafts/manifest.jsonl, zero training/scoring eligibility.
Seven private overview pages in runs/real-drafts-overview/. Aggregate receipts
docs/evidence/real-drafts/scope.json. Never treat legacy labels/proposals as truth.

M2 broad learning, M3 queue/hold joint state, accuracy-sized model, diverse data,
reviewed real pilot/baseline, calibration/export/release remain OPEN.
