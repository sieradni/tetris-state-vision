# Current checkpoint — 2026-10-25

The user asked to begin the supplied standalone Tetris vision project in a new
repository. This repository is C:/Users/micha/Code/tetris-state-vision, local Git
only, no remote. The neighboring tetrguide repository was left untouched.

Completed initial M0/M1 milestone:
- [x] Rich state contract, explicit unknown/empty/absent/cropped semantics.
- [x] Bottom-first lattice; independent locked/active/ghost outputs.
- [x] Strict solver adapter: active once; explicit temporal hold availability.
- [x] Exact solver/colored/all-layer evaluation; readable refusals fail.
- [x] Deterministic layered renderer; same-state appearance groups; masked
  ambiguous targets; separate latent records; finite generation CLI.
- [x] Independent 200-cell pixel oracle, crop/resize and ambiguity fixtures,
  CLI integration and real Windows memory/deadline/process-tree enforcement.
- [x] Fifteen demo images / three ancestry groups; montage visually inspected.
- [x] Exact source/lock/brief checksums, raw logs and named bug closure gates.

Verification: 46/46 tests in 2.462 s. Receipt docs/evidence/fixtures-final.json;
log docs/evidence/fixtures-final.stderr.log. Aggregate ceiling 512 MiB, wall 90 s,
46 finite checks; raw Windows peak 155,201,536 B. Memory/deadline probes use
nested small Jobs; a historical peak can include disputed accounting, so raw
values are preserved and native allocation denial is verified independently.
Demo: docs/evidence/demo-r1.json, 256 MiB / 60 s / 15 images, 0.623 s,
raw peak 48,103,424 B. Preview examples/montage.png; complete/cropped JSON examples
alongside it. Overall record: docs/evidence/foundation.json. Evidence curation
passed in 0.285 s at a 256 MiB / 30 s / 15-record budget.

No background processes or training are running. Failed first fixture runs are
retained, their four defects closed on named green evidence in BUG_LOG.md.
No model weights, real-image accuracy, reviewed screenshot pilot, legacy baseline,
training throughput or inference latency have been measured. This is one fixed
layout with five controlled appearance families, not full training coverage.

Next acceptance-gated task: M1 layout/panel diversity. Parameterize board/cell
scale and panel geometry; add horizontal/wrapped and repeated queues, independent
preview skins and absence/crop counterfactuals. Gate transformed coordinates and
semantic layer masks with independent pixel fixtures and inspect new montages.
Then implement the scratch lattice-query prototype and characterize a tiny
memory/wall/step-capped learning pilot before scaling. Read ANNOTATION.md and
BUG_LOG.md before changing geometry, visibility or evaluator semantics.

Source brief: docs/research/Tetris_Vision_Research.md, byte-identical to supplied
Downloads document. SHA256:
`4FE67034E8FD183849E27B8DB099A0E12798DB6E5E9CA6FBBFA94493990658F3`.
Git history records the local milestone identity; nothing is pushed or published.
