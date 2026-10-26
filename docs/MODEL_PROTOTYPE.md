# M2 scratch lattice-query prototype

Acceptance is a bounded engineering/learning pilot, not real screenshot quality.
M1 is committed at 3cd063b (67 green checks); its coverage limits remain explicit.

Implement one image encoder at strides 2/4/8/16, a coarse global context path,
learned one-board lattice, and bounded sub-cell sampling. Canonical queries use
row/column identity and axial attention with global context. Independent heads
predict locked occupancy, conditional type, observability, active and ghost
piece/footprint. Never subtract an active prediction from the board.

The accuracy configuration starts at 48/96/192/256 channels, decoder dimension
192 and six blocks. A small 16/32/64/96, dimension-64 configuration is for resource
characterization and numerical tests only. Its pilot cannot decide the final
architecture or scratch/pretrained comparison. No pretrained weights are used.
Queue/hold joint decoding, appearance-memory experiments, multi-board association
and release inference remain M3+ gates, not placeholders advertised as complete.

Preserve aspect ratio and valid-pixel mask; pad, rather than stretch. Geometric
targets follow actual resize/padding. Spatial sampling uses pixel boundaries
normalized for `grid_sample(..., align_corners=False)`, with offsets bounded by
fractions of cell basis vectors. Recognition gradients must reach geometry.
Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html

Losses consume only image-supported targets: unknown geometry/cells/type/layers
are masked. Include registered clean/no-grid and density-balanced cases in the
learning pilot. Initial geometry is axis-aligned affine for the current digital
renderer; skew/perspective are future controlled extensions. Explicit limits:
one board, 10x20 locked cells and six spawn rows; source image/work bounds are
validated. Untrained or uncalibrated outputs must remain needs_review.

Required gates: exact preprocessing geometry; valid masks; nonfolding/bounded
lattice and sampling; complete finite forward/backward; occupancy/type independence;
all-unknown loss without NaN; gradients through lattice sampling; checkpoint
identity/round trip; measured tiny CPU pilot before scaling; a clean/no-grid
learning curve and held-out synthetic diagnostic. Real images remain a separate
human-reviewed acceptance gate.

Resource admission: CPU-only optional dependency, two compute threads, no worker
processes/prefetch, finite steps/samples/cache/pixels, verified Windows Job and
wall limit, durable per-step log and atomic completion/checkpoint records. The
first import/forward pilot characterizes RAM at a small hard ceiling. Accelerator
memory is not used or inferred from the machine's video-controller listing.

Prior defect interactions: V-001 masks unregistered geometry/canonical cells;
V-002 keeps occupancy/type/layers independent; V-003 computes source/config/data/
checkpoint hashes; V-004 retains native admission enforcement and raw peaks;
V-005 consumes observation-only targets and never invents queue start/count.
V-006 gates all canonical layer poses on registration. V-007 identifies the
exact named supervisor Job and proves membership before importing the runtime.

Numerical gate: 85 checks green, one CPU forward/backward probe, source identities
in docs/evidence/m2-numerics/numerical.json. Learned quality remains pending.
