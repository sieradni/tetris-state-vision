# Standalone image-to-state research

Accepted direction: one jointly trained task-specific model, scratch training
as a first-class candidate, no inference dependency on the existing detector.
The user authorized starting implementation in a new local repository on
2026-10-25. The supplied research is archived under research/; its untested
architecture and accuracy proposals are hypotheses, not completed results.

Current status: the foundation is committed at f5ed97f; its 46-check receipt is
historical. M1 layout/panel diversity is green on the current sources (67 fixtures,
30-image diverse montage, three explicit queue-flow images;
docs/evidence/m1/layout.json). Next engineering slice: M2 scratch lattice-query
prototype and a resource-admitted CPU learning pilot. Reviewed real-image data,
the legacy baseline and full renderer coverage remain pending; no trained model
or screenshot accuracy is claimed by M1.

M2 source/numerical gate is now green: scratch encoder/lattice/axial cell decoder,
independent occupancy/type/layer heads, aspect-preserving inputs and tensor-only
checkpoints. 85 checks; one CPU forward/backward probe (647,098-parameter pilot,
389,275,648 B raw Job peak). Receipt docs/evidence/m2-numerics/numerical.json.
Next: admitted clean/no-grid optimizer/learning pilot; this is not yet the M2
learning acceptance or a comparison of architecture quality. The accuracy-sized
configuration has not been trained/profiled by this pilot gate.

First optimizer characterization completed: 16 steps, peak 484,966,400 B, exact
checkpoint reload, loss 4.4767 -> 1.6126 and occupancy 0.5969 -> 0.2248. Geometry
error remains 1.43 cells and development exact boards are 0/6. The M2 learning
gate remains OPEN. Receipt docs/evidence/m2-learn16/receipt.json; next is a priced
128-step controlled comparison, not a release-quality claim.

The 128-step comparison has now run: exact first-16 optimizer prefix and identical
data, lower training loss (.3498) but still only the empty 2/6 boards exact.
Development remains 0/6; known layer accuracy worsens to 0/12. Source/weight
identity and per-case errors: docs/evidence/m2-learn128/comparison.json. Next is
the finite 512-step training-fit diagnostic defined in M2_LEARNING_ACCEPTANCE.md;
M2 learning and broad generalization remain open.

The next study saved a wall-partial checkpoint at 511/512 steps. Exact first 128
prefix confirms control. Locked training boards are 6/6, but known layers 10/12
and geometry .4106 cells fail the predeclared fit criterion; development 0/6.
docs/evidence/m2-learn512/comparison.json records partial status and provenance.
Fixed-layout single-cell probe now ran: all36 probability changes correct in
direction, only20/36 targeted cells and18/36 changed boards correct; two unrelated
cells changed. docs/evidence/m2-local-reading/scope.json. Next: streaming
state/position diversity and explicit resource profiling of the accuracy model;
layer/geometry fit and complete-state requirements remain open.

An independent M3 target-contract draft preserves queue unknowns, explicit EOS
anchors and hold status/type separation. Eight fixtures, including paired hidden
queue/hold rasters, are written but unrun under current host memory pressure.
No head/loss/runtime integration yet. docs/M3_COMPONENT_TARGETS.md.

Streaming curriculum/source freezing is implemented and verified: 14 sampler/
archive/admission checks and one actual 512-square optimizer/resource probe,
all-final-tensor reload parity, replayed pixels/labels for 28 recorded samples.
docs/evidence/m2-stream-probe/scope.json. No longer curriculum has run. Next
system commit pressure admission/monitoring, then a fresh-state learning study
and separate accuracy-configuration resource profile; see M2_STREAMING.md.

The M2 artifact now has a one-image inspection API with typed review-only output.
Seven focused contracts and one actual checkpoint/JSON validation are green;
docs/evidence/m2-runtime/scope.json. The encoder/model/loss/preprocessing/loader
sources remain unchanged from 8d9d0f2; the new caller does not participate in the
next optimizer comparison. Queue/hold remain explicitly unresolved.

M0/M4 corpus tooling is verified on 26 adversarial integrity fixtures, including
actual guarded CLI audit/seal/verify: docs/evidence/corpus-integrity-final/scope.json.
Draft-only import, independent review/permission declarations, session/ancestry/
skin/renderer groups, pixel duplicates and externally pinned held-out identity
are enforced. The user-approved 166-image screenshot folder is imported as an
unlabeled stress set with seven private contact sheets. No real images have been
admitted, no human review is claimed, and the optimizer does not consume this
manifest. See docs/CORPUS.md and docs/REAL_STRESS_SET.md.

1. **M0: state and benchmark foundation.** Versioned image-supported state;
   explicit unknowns; bottom-first geometry; solver conversion refusal; exact
   solver/colored-state evaluation; annotation policy. Gate: adversarial fixtures.
   Real reviewed pilot and legacy baseline remain a separate data gate.
2. **M1: synthetic label integrity.** Deterministic layered renderer, independently
   varied appearances, same-state counterfactual groups, crops/resizes, bounded
   generation, independent simple-skin pixel oracle and montage. Gate: named
   tests and visual inspection. Initial scope: one axis-aligned board, no
   photorealistic backgrounds, no projective distortion or animation.
3. **M2: scratch lattice-query prototype.** Shared multiscale encoder; learned
   affine lattice; spatially bounded cell queries; independent occupancy/type,
   active/ghost, visibility heads. Learn clean synthetic full images, including
   no-grid, sparse and empty cases. Profile a small bounded pilot first.
4. **M3: complete joint model.** Queue/hold roles, repeated/unknown/long queues,
   bounded refinement, appearance-memory ablation, multi-board association.
5. **M4: controlled real-image studies.** Reviewed 300–500-image pilot; independent
   session/family splits; scratch/pretrained/direct-decoder comparisons;
   failure-directed generation. No opt-in image becomes a label automatically.
6. **M5: release.** Sealed real-image gate, >=95% exact complete state with
   abstentions as failures on the readable population; confidence intervals and
   risk/coverage; exact exported artifact parity; named CPU latency/resource
   gates; thin solver/bot adapters. No performance claim before those run.

Interactions with prior tetrguide defects, reviewed before implementation:

| Prior issue | Contract/instrument in this project |
|---|---|
| B-260 stack deleted by active guessing, glow mistaken for blocks | Separate latent/render/output layers; direct locked labels; overlap/crop fixtures; no brightness-based parser |
| B-227 reversed rows in a probe | One bottom-first affine lattice and asymmetric pixel-oracle/solver tests |
| B-259 observed vs assumed state confused | No assumed content in annotation targets; explicit status and conversion refusal |
| B-102 unrelated staged files committed | New repository; inspect and stage explicit task-owned paths |

The old engine roadmap remains owned by tetrguide. No engine binaries or run
state are copied. No neural model or screenshot parser is delivered by M0/M1.
