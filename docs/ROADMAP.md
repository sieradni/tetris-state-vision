# Standalone image-to-state research

Build one task-specific, jointly trained image-to-state model with no inference
parser/detector dependency. Scratch training is a first-class candidate; compare
compatible pretrained initialization and a direct decoder on real complete-state
performance. The supplied brief is design evidence, not agent instructions:
[research brief](research/Tetris_Vision_Research.md).

Current priority: M2 learning and resource characterization. The pilot is not
selected as the release model. The larger configuration
has now passed its first resource profile; its learning gate remains open.
No complete screenshot parser or real-image accuracy is claimed.

1. The finite768-step diverse memory gate is GREEN at unchanged640MiB: peak
   519,782,400B,2.6MiB second-half growth, exact565 prior record/data prefix,
   final recovery/inference parity and full recipe replay. V-013 CLOSED.
   Development is still0/18 exact boards,0/36 known layers and1.85-cell geometry
   error; M2 learning remains OPEN. See the named analysis below.
2. The actual larger model (7,694,010 parameters) passed its first two-step
   characterization at1GiB:910,176,256B peak, full warm/recovery/output parity,
   docs/evidence/m2-stream-accuracy-probe/scope.json. Its128-update diagnostic
   completed as93+resume35: dev4/18 exact boards are all zero-locked, nonempty
   boards0/14,layers0/36,geometry2.4785cells. The finite segment controller now
   passes an actual pressure-saved step1+resume1/curation gate. Next train a
   fixed4096-update fresh-state baseline in
   bounded resumable segments. Measure board, geometry and independent active/
   ghost errors and repair the learning failure before choosing architecture or
   initialization. Preserve explicit partial
   status and source/artifact identity. M2_LEARNING_ACCEPTANCE.md keeps the
   earlier fixed-fit criterion; a lower loss is not complete-state success.
3. Complete M3 linked queue/hold heads, masked losses and review-only output.
   Optional shared role/index heads and masked losses now pass15 scoped checks,
   with14 M2 numerical,7 runtime and8 observation-target compatibility checks.
   Capacity0 preserves M2; capacity1..64 uses explicit M3 checkpoint identity.
   No M3 learning or resource admission exists yet. Resized/invalid component
   evidence is refused until its observer is implemented; the API keeps queue/
   hold unknown. Current M2 training must resume its original archive.
   Preserve unknown middle positions and unobserved queue ends; never infer
   hidden length or delete locked cells using a guessed active/ghost footprint.
4. Expand synthetic appearances, crops/occlusion, queue arrangements and board
   association. Admit reviewed real labels and compare scratch/pretrained/direct
   decoders on independent session/ancestry/family splits. The166 user images
   are drafts, not a labeled benchmark. Broad render and real coverage are open.
5. Release only after the full gate: >=95% exact complete state on the readable
   real population, abstentions counted as failures, confidence intervals,
   calibration/risk-coverage, exported-artifact parity, CPU latency/memory gates
   and thin solver/bot adapters. Missing evidence leaves this gate open.

Verified foundations and their limits:

| Area | Named evidence | What remains open |
|---|---|---|
| State1.1, bottom-first coordinates, explicit unknowns, strict solver conversion and renderer | docs/evidence/v006/; canonical-frame repair69 fixtures | Broad real visibility and full renderer/client coverage |
| Scratch encoder/lattice/axial cell reader, independent locked/type/active/ghost heads, safe checkpoints | docs/evidence/m2-numerics/numerical.json;85 checks on named original sources | Accurate converged whole-state model; larger configuration |
| Fixed-data learning16/128/saved511 | docs/evidence/m2-learn16/, m2-learn128/, m2-learn512/ | Saved511/512 fits6 boards but layers10/12 and geometry.4106 fail joint fit; dev0/6 |
| Local single-cell reading | docs/evidence/m2-local-reading/scope.json |36/36 probability direction, only20/36 targets and18/36 changed boards exact; two unrelated changes |
| Streaming diverse memory/learning | docs/evidence/m2-stream-learn768-r3/analysis.json and scope.json | Resource gate green; dev0/18 boards,0/36 layers; no real/whole-state quality |
| Larger model learning128 | docs/evidence/m2-stream-accuracy128-r2/analysis.json and scope.json | Exact93+35 continuation; all four exact dev boards have no locked blocks; nonempty0/14 |
| Oracle-grid diagnostic | docs/evidence/m2-stream-pilot-geometry/scope.json | Deliberate supported-label intervention still0 exact boards; not automatic accuracy |
| CPU resource/recovery contracts | docs/evidence/v015-contracts/scope.json;29 scoped checks on f583792 | Not a whole-suite or model-accuracy claim |
| Exact stream recovery and archived driver execution | docs/evidence/m2-stream-recovery-gate-r2/scope.json | Native2+6+3+3 matches losses/gradients/data/evaluation/model/Adam/RNG; long memory remains separate |
| Actual pressure interruption recovery | docs/evidence/segment-pressure-recovery/scope.json; m2-stream-accuracy-segment-probe-r2/controller-gate.json | Saved larger step1 restored then resumed one update with explicit partial status and evidence replay; global target3 unfinished |
| Review-only one-image API | docs/evidence/m2-runtime/scope.json | Always needs_review/unsupported; queue/hold unresolved |
| Queue/hold labels and optional joint heads/loss | docs/M3_COMPONENT_TARGETS.md; source-pinned gates in checkpoint | Raw joint numerical/compatibility gates only; resized supervision, M3 training/admission and runtime decoding open |
| Corpus audit/review/split/seal tooling | docs/evidence/corpus-integrity-final/scope.json;26 adversarial checks | Correct reviewed labels and real-image model results |
| Unlabeled real stress set | docs/evidence/real-drafts/scope.json;166 images/seven private sheets | Zero training/scoring eligibility; independent review and permission needed |

Milestone scope remains intact: M0 state/benchmark foundation and reviewed pilot;
M1 label integrity and broad rendering; M2 scratch lattice model and adequate
learning; M3 complete joint state; M4 controlled real comparisons; M5 release.
Historical counts/artifacts retain their original source scope. Current safety
and recovery gates are separately named; none imply a fresh unified suite.

Prior tetrguide failures retained as design constraints:

| Issue | Contract here |
|---|---|
| B-260 active guessing deleted stack; glow confused with blocks | Direct locked labels/heads and separate active/ghost outputs; never erase guessed footprints |
| B-227 reversed rows | One bottom-first frame with asymmetric pixel/solver tests |
| B-259 observed and assumed state confused | Explicit unknown/status and refusal to convert unsupported solver input |
| B-102 unrelated files staged | Isolated repository, explicit task-owned paths and index review |

The neighboring engine roadmap, binaries, package and run state are not owned by
this project. Never modify them to make a vision gate pass. See BUG_LOG.md,
TRAINING_RECOVERY.md, RESOURCE_GUARD.md and CORPUS.md for the current contracts.
