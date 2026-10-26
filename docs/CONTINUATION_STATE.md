# Current checkpoint — 2026-10-26

Repository C:/Users/micha/Code/tetris-state-vision, local Git, no remote.
Goal remains ACTIVE: complete the standalone model project thoroughly and commit
verified milestones. No real-image/release completion is claimed.

Verified checkpoints:
- f5ed97f foundation; 3cd063b layout/panel diversity (historical scopes).
- b06cdda: V-006 canonical pose/registration repair, 69 green checks.
- M2 source/numerical milestone: 85/85 in 6.992 s, no skips. Current source and
  lock hashes in docs/evidence/m2-numerics/numerical.json. Scratch encoder at
  strides 2/4/8/16, learned axis-aligned lattice, bounded cell sampling/axial
  attention; independent occupancy/type/visibility/active/ghost heads; tensor-only
  hash-checked checkpoints. Default accuracy config and small pilot are distinct.

Measured resource admission:
- Optional CPU-only torch 2.14.1+cpu / numpy 2.5.4 install, peak 190,705,664 B
  under 1 GiB / 600 s / one sync. Package source version 0.3.0.
- One 647,098-parameter pilot forward/backward: 15.291 s with cold import,
  peak 389,275,648 B under 1 GiB / 60 s / one step; finite gradients and losses.
- Full 85-case suite: 9.585 s runner, raw peak 1,006,940,160 B under 1 GiB /
  120 s / 85 cases. Counter is near the cap and includes denied-allocation probes;
  raw values retained. Native allocation denial and process-tree death green.
- V-007 closed: implicit worker Job query saw an incidental nested object.
  Supervisor now names its Job; worker proves membership and reads that exact
  ceiling before Torch import. No cap disabled/raised for the repair. Evidence
  retains the initial failure, diagnostic and green exact-identity fixtures.

Next acceptance-gated task: M2 clean/no-grid learning pilot. Implement a finite
CPU optimizer driver with >=3 density strata, variable board origin/spacing,
clean/no-grid matched groups, observation-only targets, <=64 MiB tensor cache,
no workers/prefetch, per-step durable progress and hash-identified checkpoints.
First characterize 16 optimizer steps at max side 384 and one CPU thread under
1 GiB / 120 s; measure loss/throughput/memory before scaling. Check for any other
heavy process before launch. Code/data/config/seed/weights identity is required.
Then run a longer bounded diagnostic if the small pilot warrants it; do not
call learning complete on a merely finite gradient or favorable average score.

No background vision process or training is running. Other release-build/search
processes were last verified absent; recheck before longer work. Queue/hold joint
heads, larger accuracy model training, broad renderer coverage, human-reviewed
real-image pilot/baseline, calibration, export and release accuracy remain open.
Read MODEL_PROTOTYPE.md and BUG_LOG.md before the next architectural change.
