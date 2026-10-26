# Current checkpoint — 2026-10-25

Repository: C:/Users/micha/Code/tetris-state-vision, local Git only, no remote.
Goal: complete as much of the standalone model project as possible, thoroughly,
with local commits at verified milestones. Goal remains active.

M1 layout/panel diversity is verified:
- [x] Variable canvas, board position/scale, panel geometry and icon scale.
- [x] Vertical/horizontal/wrapped queues; all piece types/rotations; lengths 0..10.
- [x] Independent preview palettes/skins; absent panels; explicit occlusions.
- [x] Image-supported queue span/start; no hidden count or type support leakage.
- [x] Schema 1.1, conservative 1.0 reader and solver refusal on unknown queue start.
- [x] Standing identical-raster equivalence instrument promoted by V-005.
- [x] 67 tests green; 30-image montage and three visible-flow specimens inspected.

Current receipt: docs/evidence/m1/layout.json, named sources and raw run logs.
Tests: m1-fixtures-r2, 67/67 in 3.497 s, 512 MiB / 90 s / 67 cases,
raw Job peak 155,123,712 B. Demo: m1-demo, 30 images / 6 groups, 256 MiB /
60 s / 30 images, 1.106 s, raw peak 64,815,104 B. Curation m1-evidence:
256 MiB / 30 s / 33 image records/specimens, rc=0 in 0.494 s,
raw peak 79,020,032 B. All recorded supervisors/children have exited.
Preview examples/layout-diversity.png and examples/queue-flows.png.
V-005 closes on its own named green run. Original f5ed97f/foundation evidence is
history on its own sources; do not cite it as verification of the new renderer.

Next acceptance-gated task: M2 scratch-model prototype. Add optional CPU PyTorch
runtime, preserve full-image aspect ratio and valid-pixel mask, implement shared
multiscale encoder, learned lattice and bounded cell sampling, independent
occupancy/type/active/ghost heads, then a tiny memory/wall/step-capped numerical
and learning pilot before scaling. Queue/hold joint heads follow, with unknown
labels masked. Record code/data/config/checkpoint identity and finite resources.

Pre-change bug review for M2: V-001 requires correctly transformed pixel centers
and no geometry/canonical-cell training on unregistered crops. V-002 requires
independent occupancy/type and locked/active/ghost targets. V-003 requires computed
checkpoint/source hashes. V-004 requires the native Job guard and raw counters.
V-005 requires observation-only losses and unknown queue starts/counts preserved.

No model training or background work is running. Real-image pilot/baseline,
calibration, sealed accuracy gate, export and inference latency remain unmeasured.
Source brief remains byte-exact under docs/research/Tetris_Vision_Research.md.
No neighboring engine files, binaries or run data were changed.
