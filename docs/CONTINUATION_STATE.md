# Current checkpoint — 2026-10-25

Repository C:/Users/micha/Code/tetris-state-vision; local Git, no remote.
Active goal: complete as much of the standalone model project as possible,
thoroughly, with verified local commits. Goal remains active.

Completed milestones:
- f5ed97f: state/data foundation (46 checks on those sources, historical).
- 3cd063b: variable layouts/panels and observation-only queue spans (67 checks,
  30-image montage + three explicit visible queue flows; docs/evidence/m1/).
- V-006 repair: canonical active/ghost poses require supported board geometry.
  Reproduction 1 failed / 21 at docs/evidence/v006/v006-before.stderr.log;
  fixed full gate 69/69 in 3.685 s at v006-after.stderr.log. Limits 512 MiB /
  90 s / 69 cases, raw peak 154,918,912 B. Source hashes in scope.json.
  Legacy 1.0 cropped annotation masks cells/layers; invalid 1.1 pose is refused.
  These source changes supersede M1's cropped annotation policy; retain old
  receipts on their own sources, never as verification of the repair.

Active M2 task: scratch lattice-query prototype, docs/MODEL_PROTOTYPE.md.
Optional CPU dependency setup PASSED: runs/m2-setup.json, rc=0 in 70.658 s,
190,705,664 B peak under 1 GiB / 600 s / one uv sync. torch 2.14.1+cpu,
numpy 2.5.4 installed in .venv; pyproject.toml, uv.lock and prototype design remain
owned M2 work pending their own verified checkpoint. No model source/training yet.

Next: implement preprocessing + valid-pixel mask, shared multiscale encoder,
learned one-board lattice, bounded cell samples/axial context and independent
occupancy/type/observability/active/ghost heads. Characterize import and tiny
forward/backward under a small Windows Job before any longer learning pilot.
Queue/hold, controlled real data, export and release gates remain open.

Resource note: other owner's tetrguide release build was live at last check
(cargo 30444/18252, rustc 23704). Recheck exact current process state before a
substantial training job; do source/modest verification work meanwhile. Do not
kill or alter that work. Setup/reproduction/full-gate supervisors and children
are verified exited; no vision background work is running.

Bug interactions: V-001/V-006 require one supported coordinate system for all
canonical targets; V-002 keeps occupancy/type/layers independent; V-003 requires
computed hashes; V-004 retains native memory enforcement and raw counters;
V-005 requires observation-only targets and paired-raster equivalence tests.
No engine files, binaries or run data changed. Real-image accuracy unmeasured.
