# Standalone image-to-state research

Accepted direction: one jointly trained task-specific model, scratch training
as a first-class candidate, no inference dependency on the existing detector.
The user authorized starting implementation in a new local repository on
2026-10-25. The supplied research is archived under research/; its untested
architecture and accuracy proposals are hypotheses, not completed results.

Current status: the initial M0/M1 code and scoped synthetic integrity pilot are
green (46 fixtures; docs/evidence/foundation.json). The reviewed real-image pilot
and baseline are still pending. Next engineering slice: expand M1 layout and
panel-order diversity before an admitted M2 learning pilot. The full M1 coverage
described by the research is not claimed by this initial fixed-layout renderer.

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
