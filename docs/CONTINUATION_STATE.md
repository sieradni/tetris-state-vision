# Current checkpoint — 2026-10-26

Repo C:/Users/micha/Code/tetris-state-vision, local Git, no remote. Goal ACTIVE.
No vision background process running; no run data deleted. This turn made
concrete progress (controlled learning, local-reading probe, streaming pipeline).

Verified milestones:
- 421de02: 128 steps, exact first-16 optimizer prefix; training exact boards2/6,
  development0/6. docs/evidence/m2-learn128/comparison.json.
- 331ac58: inner wall saved511/512 steps, explicitly wall_partial/exit2. Exact
  first128 prefix. Training boards6/6 but layers10/12, geometry.4106 cells fail
  declared joint fit; development0/6 and0/12. Ghost row15 vs target10 on dense
  clean/no-grid. docs/evidence/m2-learn512/comparison.json. Weight SHA256
  ca4c0f34a81b8c7591fab7db0e92fd89d652a27dea015cf63473c3f7b780bcdf.
- 72cde65: fixed-layout reading probe, 42 forwards, 36 visible one-cell changes/
  18 matched groups. Direction36/36, targeted20/36, exact changed boards18/36,
  two unrelated cells changed. docs/evidence/m2-local-reading/scope.json.
- Streaming pipeline now verified, ready to commit: 7 sampler +4 archive +3
  admission fixtures, plus real 512x512 one-step profile; source/model imports
  bind to the verified run archive, no tensor cache, all-final-tensor reload
  exact. 9.717s / raw peak549,154,816 B under1GiB/150s/one optimizer step.
  Curator re-rendered1 training +27 saved probe frames/labels exactly.
  docs/evidence/m2-stream-probe/scope.json; docs/M2_STREAMING.md. Model/loss/
  preprocess/render/visibility unchanged. Accuracy configuration not profiled.
  Supervisor24324/launcher7672/worker29788 verified exited.

Next selected task: supervisor system-wide commit-pressure admission and polling
before longer learning studies. Host at last check: committed30,895,226,880 B,
available11,370,631,168 B; another owner's Rust build remains live (last cargo
30968/31372, rustc24260, recheck authority). Do not touch their processes.
Do not start another substantial learning study at this pressure without a
guard. Small bounded tests/source work remain available.

Plan: read docs/BUG_LOG.md and official Windows GetPerformanceInfo contracts;
query OS commit independently of nested Jobs; reserve configured headroom at
launch, poll well before wall/snapshot cadence, write a terminal receipt then
kill the complete owned Job on pressure. No env/CLI opt-out. Preserve native
aggregate cap and all wall/work limits. Add a synthetic pressure tripwire using
tiny real Jobs, prove child/grandchild death without allocating large memory,
and ensure global OS counters aren't mistaken for nested Job counters.
Prior interactions: V-003 computed source hashes; V-004 current OS pressure is
distinct from historical Job peak; V-007 exact named Job still verifies worker
membership, global counters must not use implicit Job identity. V-001/002/005/
006/009 targets unchanged; V-008 malformed declarations stay fail closed.
Record any fixed resource defect in BUG_LOG with named green gate.

The new streaming driver's source catalogue includes run_bounded.py, so a guard
change invalidates the old one-step admission for future studies. Re-profile
with a NEW output/receipt after guard verification; preserve the existing run.
Then use matching --admission for a finite fresh-state study, with one CPU thread,
native <=1GiB for pilot, explicit work/wall caps. Actual 512-square pilot peak
~524MiB provides the first estimate; accuracy-sized candidate needs separate
bounded characterization. Source archives let parent model work continue during
a longer run; archived helpers and lock are pinned, no live-source drift.

Earlier gates:85 core/model/resource at8d9d0f2;2 optimizer admission;7 runtime
plus actual one-image API;26 corpus;3 partial-evidence;14 stream contracts.
These are separate scopes, not one unified137. V-001..V-009 CLOSED on named logs.
Env .venv0.3.0, Torch2.14.1+cpu/NumPy2.5.4/Pillow12.3.0.

User-approved166 uncertain-answer screenshots remain unlabeled drafts only at
data/real-pilot-drafts/manifest.jsonl, zero training/scoring eligibility.
Seven private pages in runs/real-drafts-overview/. Aggregate receipts in
docs/evidence/real-drafts/scope.json. Never treat legacy labels/proposals as truth.

M2 broad learning, M3 queue/hold joint state, accuracy-sized model, diverse data,
reviewed real pilot/baseline, calibration/export/release remain OPEN.
