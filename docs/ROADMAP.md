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

Latest: the input representation (model capture side 512 -> 640) is a declared negative
([M3_IDENTITY_REPRESENTATION.md](M3_IDENTITY_REPRESENTATION.md), predeclared at
`b229972`, `docs/evidence/m3-identity-factor-representation-r1-errors-r1/scope.json`):
the primary does not rise (exact queue piece tokens 3/54 training and 9/90 development,
equal to `m3-identity-factor-r1`) while every structure term holds, and the independent
replay measures the same 54/90 tokens and 9/12 hold cases, so the finer capture did not
change the supervised population. The fit gate is unchanged and FALSE (0/9 and 0/18
boards, 9/18 and 0/36 roles, 0/9 and 0/18 components, 3.82 and 2.28 cells). With the
coefficient, read-out, budget and representation all measured, no executable identity
lever remains: what is left is the owner-decision real-image chain and the FALSE fit
gate.

Latest: the 512-update identity budget is a miss on its own criterion
([M3_IDENTITY_BUDGET.md](M3_IDENTITY_BUDGET.md),
`docs/evidence/m3-identity-factor-budget-r1-errors-r1/scope.json`): exact queue piece
tokens rise 3/54 -> 12/54 training and
9/90 -> 12/90 development at the same
capacity, factor, data and order, but training queue EOS falls
8/9 -> 5/9 and development hold identity
3/12 -> 0/12, training roles fall to
9/18, and the fit gate stays FALSE. Identity is budget-sensitive but
the budget multiplies a structural trade-off, so the next ranked lever is the input
representation (pixels per icon), not another budget extension. V-044 fixes the
predeclaration rule that had no branch for this state.

Latest: identity supervision is class-complete but label-scarce
([M3_IDENTITY_EXPOSURE.md](M3_IDENTITY_EXPOSURE.md),
`docs/evidence/m3-identity-exposure-r4/scope.json`): all seven piece classes are
supervised in both splits (54 training and 90 development queue tokens), yet the only
identities ever exact are queue `S` (3/3 and 9/18) and hold `T` (0/3 and 3/3), with six
queue classes and three hold classes at zero and the development successes spread over
slots 1, 3, 5 and 7. The scored probes are not the training stream, though: the factor-4 run's 128
updates drew 128 distinct images from 43 ancestries, so what remains is optimization
budget versus the input representation. The budget lever is predeclared in
[M3_IDENTITY_BUDGET.md](M3_IDENTITY_BUDGET.md) and is running next.

Latest: tying the identity read-out into the cell-type classifier is a measured
negative ([M3_IDENTITY_READOUT.md](M3_IDENTITY_READOUT.md), predeclared at `e83d70e`,
shared mode admitted at`runs/m3-stream-identity-shared-profile-r2-job.json`):matched128
updates atcapacity64 move exact queue pieces0/54->9/54 training and0/90->6/90
development and development hold0/12->3/12, but the predeclared structure-held
condition fails (training presence63/63->54/63, development EOS15/15->13/15) and
training roles fall12/18->0/18, so the fit gate stays FALSE(0/9 and0/18 boards,0/9 and0/18
components,3.32 and2.55 cells). Independent27-forward replay(weights`55a35243...`
unchanged) and curator: `docs/evidence/m3-stream-identity-shared-r1-errors-r1/`. Both identity-side levers(coefficient and read-out)are now measured and closed: the
coefficient moved localization at a measured structural cost, and the read-out tying is
a negative. The fit gate stays FALSE, and the open items narrow to the supervision
signal, the input representation(pixels per icon)and the owner decision on admitting
real screenshots. The whole contract suite is green on this tree(407 tests,
`docs/evidence/m3-identity-readout-fullsuite-r11/`)after the supervisor's commit stop line
was corrected from physical RAM to the commit limit(V-040).

Latest: a declared4x identity-loss scale is a measured positive on the M3 identity
localization ([M3_COMPONENT_IDENTITY.md](M3_COMPONENT_IDENTITY.md), predeclared at
`65ee2f1`): matched128 updates atcapacity64 raise exact queue piece tokens0/54->3/54
training and0/90->9/90 development, hold0/12->3/12, slot status still exact, while the
fit gate stays FALSE(0/9 boards,12/18 roles,0/9 components,3.83 cells) and development
EOS falls15/15->3/15. Independent27-forward replay and curator:
`docs/evidence/m3-identity-factor-r1-errors-r3/`.

Latest: the first joint M3 fit (capacity64,128 updates) is a measured negative
([M3_JOINT_FIT.md](M3_JOINT_FIT.md), predeclared at `90218b3`): training probes
0/9 exact boards,12/18 roles (ghost9/9, active3/9) and0/9 readable components with
queue tokens0/54 andhold0/9, while EOS9/9, slot status54/54 andpresence63/63 are
exact; geometry19.82->4.04 cells and loss29.97->7.74 show the optimizer trains,
but development stays0/18 boards and0/36 roles. Independent27-forward replay
(weights`b2e44c9c...` unchanged) and its curator are in
`docs/evidence/m3-stream-joint-fit-r1-errors-r2/scope.json`. Items4-5 need
reviewed, permissioned real labels and stay unreachable from repo data.

The staged frozen-backbone type-head refinement is also a measured negative
([STAGED_HEAD_REFINEMENT.md](STAGED_HEAD_REFINEMENT.md), predeclared at `e601182`):
head-only training on the ordered artifact raises exact training roles 20/64 to
36/64 with boards32/32, cells128/128 and geometry.21059 bit-identical, still short
of the64/64 gate and26 short of the62/64 copied-head reference, and an independent
50-forward replay (unchanged weights`e02a02e7...`) plus curator reproduce every
saved metric; development on the canonical18 images is the same4/18 boards and0/36
roles for both stages, with stage2 loss higher (11.8129 vs11.0885, replay
`m2-stream-headrate-control-s3-dev6-errors-r1`); do not re-tune this stage. Conditional copied-head pixel
replay and independent audits are complete for both controlled models. Ordered training roles reach 62/64; interleaved
64/64, but both joint-fit criteria fail and development types remain 0/18.
See [CONDITIONAL_PIXEL_REPLAY.md](CONDITIONAL_PIXEL_REPLAY.md). Next predeclare
a controlled joint-optimization lever and its resource/recovery admission.
[TYPE_HEAD_JOINT_OPTIMIZATION.md](TYPE_HEAD_JOINT_OPTIMIZATION.md) now defines
the fixed candidate and records the completed matched comparison: both fail the
joint fit; increasing the type-head rate reduces exact training roles 20/64 to
8/64, despite improving geometry, and development roles remain 0/18 for both.
The checkpoint diagnostic in [TYPE_FORGETTING_DIAGNOSTIC.md](TYPE_FORGETTING_DIAGNOSTIC.md)
finds recurrent type-correctness turnover on the fixed fixtures and local
class-gradient conflicts, with significant limits on causal interpretation.
The matched design and focused grouping/gradient/recovery instrument are in
[BALANCED_ACCUMULATION.md](BALANCED_ACCUMULATION.md). Both1,024-image arms now
have independent ordinary replays and curators
(`docs/evidence/accumulation-arm-{balanced,online}-errors-r1/scope.json`):
balanced-eight reaches16/32 exact training boards,34/64 roles and.7507 geometry,
while online interleaved reaches28/32,37/64 and.3862; both fail the predeclared
fit and both stay at4/18 development boards(all four empty)with0/36 roles and
rising development loss. Batching eight images per update is a measured negative
result,not a mechanism claim:the ordered-eight arm was never run and grouping is
still confounded with update count and clipping frequency. Explicit image and
optimizer cursors are now integrated into the actual trainer behind an explicit
accumulation policy, with image-cadence evaluation/snapshots and separate counts in
every ledger and summary. Measured two-update native profiles now exist for both
predeclared group sizes under the unchanged 640 MiB ceiling and supervisor
(size1: 519,929,856 B peak,33.3 s; size8: 561,786,880 B peak,29.35 s, both with
exact boundary reload), and each arm is admitted only through those measured
numbers. Both 1,024-image arms now complete across two bounded segments each with exact
journal-prefix resume, byte-identical Adam/RNG continuation and measured peaks
under640MiB (`docs/evidence/accumulation-arm-r1/`); automated controller/curator
routing remains open; that independent pixel replay and both curators are now complete for both arms (see the accumulation paragraph above). The pure journal and ownership boundary pass their own scoped
tests, and the whole-group arm also passes scoped driver checks including a forced
in-group interruption replayed to identical bytes (toy network; V-030 records that
bounded children need an absolute interpreter path). Retain the original joint-fit
and generalization gates.

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
   passes an actual pressure-saved step1+resume1/curation gate. The fixed4096
   baseline is COMPLETE4096 through bank3840+256 verified updates: dev0/18 boards,
   0/36 layers,geometry2.01544cells. All4096 recipes and27 probes replayed;
   docs/evidence/m2-stream-accuracy4096-s10/scope.json preserves complete4096/4096
   and native controller/child/curator receipts. The prior host-pressure stop
   logged1172/banked1152; all20 recomputed updates and1172-row prefixes match
   exactly,excluding time. New inference weights06751638... retain original
   catalogue121fd05a; adequate learning remains open. Keep this study complete. Its
   final ordinary27-forward replay and independent curator now pass,with813
   invented development cells,32 misses,0/18boards and0/36roles. The entire9/18
   training-probe layer score is ghost absence,not visible-ghost reading.
   ACCURACY4096_BASELINE.md records source/artifact/native scope and limits.
   Use those errors to track independent active/
   ghost errors and repair the learning failure before choosing architecture or
   initialization. Preserve explicit partial
   status and source/artifact identity. M2_LEARNING_ACCEPTANCE.md keeps the
   earlier fixed-fit criterion; a lower loss is not complete-state success.
   Exact masked-target audit of the investigated1664 prefix confirms all seven
   observed piece classes were supervised,but training ghosts only occur at
   row10 and fixed probes omit classes/statuses. TARGET_EXPOSURE.md records
   counts and limits. A separate optional dense layer-evidence experiment is
   now a completed matched768-update pilot: both studies pass resource/recovery,
   but broader192-image evaluation gives0/148 exact nonempty boards for both,
   only4/165 observed active poses and0/144 ghosts for the candidate, with more
   locked false positives. It remains default off. LAYER_LEARNING_PILOT.md records
   raw source/data/artifact-bound results and coverage limits. Retain the original
   baseline; investigate class collapse with an all-class positive-control fit
   before selecting another learning mechanism. Its32 balanced public-observer
   fixtures now pass5 source-bound checks and exact file/pixel/recipe replay;
   ALL_CLASS_FIT.md predeclares the fixed1,024-update fit criterion. The frozen
   driver, fresh warm admission and exact2+6+3+3 recovery are verified on
   catalogue6147a558. The fixed1,024 updates and independent final replay are
   complete:32/32 boards,12/56 observed role labels,8/8 absences,128/128 semantic
   cells and.211-cell geometry. All observed positions/rotations fit, but types
   collapse to Z/J/L and the fit gate FAILS. The next selected lever is the
   order-only comparison in INTERLEAVED_FIT.md. Its sampler and policy are
   integrated, with88 scoped checks plus the8-case profile-ancestry regression.
   Fresh640MiB warm admission and independent replay pass; the same-input
   comparison has identical actual weights,losses,gradients,recipes and probes.
   New-policy recovery and the full matched1,024-update comparison are now
   complete:visible labels12/56->30/56,but boards32/32->28/32 and geometry
   .211->.386. Both fit gates fail;INTERLEAVED_FIT.md records controls,raw evidence
   and limits. Do not extend either finished study or adopt this ordering as a
   release fix. The original larger4,096 baseline is now complete with its
   original frozen source/optimizer and independently replayed final errors.
   Read-only actual type-head feature observation and independent collectors
   now pass on all three completed models,with ordinary outputs/weights unchanged.
   The ancestry-excluded centroid readouts provide no obvious replacement;they
   do not prove absence of class information. LAYER_FEATURE_DIAGNOSTIC.md records
   actual class coverage/counts/receipts and limits. The subsequent fixed1,024-update
   conditioned full-batch linear-head fits now complete with separate native warm/
   reload admission and independent audits. Ordered training types20/64->62/64,
   interleaved37/64->64/64,development0/18both;TYPE_HEAD_FIT.md records criteria,
   worsened development losses and scope. Original image models/source/optimizers
   stay frozen. Conditional pixel replay now confirms the copied-head training
   counts while leaving other outputs unchanged; both joint-fit gates fail.
   No generalization gain follows from this intervention.
   More than40GB free is required before each long training window.
3. Complete M3 linked queue/hold heads, masked losses and review-only output.
   Optional shared role/index heads and masked losses now pass15 scoped checks,
   with14 M2 numerical,7 runtime and8 observation-target compatibility checks.
   Capacity0 preserves M2; capacity1..64 uses explicit M3 checkpoint identity.
   Joint streaming integration passes68 scoped unit/contract checks onf542096.
   Native pilot recovery6==3+3 is exact, and both pilot and larger capacity64
   configurations pass full512-square warm/reload resource profiles. The finite
   controller now also passes actual continuous3==partial1+resume2 routing,
   curation and byte-identical model/Adam/RNG recovery on the admitted pilot;
   docs/evidence/m3-controller-recovery-r1/scope.json. The larger
   has7,997,016 parameters and942,903,296B peak under1GiB. Joint learning now has its first measured datum: a128-update joint fit atcapacity64
   on the run's nine training probes is complete and fails its predeclared criterion
   (0/9 boards,12/18 roles withghost9/9 andactive3/9,0/9 readable components with
   queue tokens0/54 andhold0/9 against exact EOS9/9, slot status54/54, presence63/63)
   while development stays0/18 boards and0/36 roles. Geometry19.82->4.04 and loss
   29.97->7.74 show the optimizer trains; component identity does not.
   docs/M3_JOINT_FIT.md; curator docs/evidence/m3-stream-joint-fit-r1-errors-r2/.
   A declared4x identity-loss scale (matched, predeclared `65ee2f1`) moves exact queue
   piece tokens0/54->3/54 and development0/90->9/90 with hold0/12->3/12 while the fit
   gate stays FALSE and development EOS falls15/15->3/15: identity is gradient-starved,
   not unreachable.Identity representation at that scale (pixels per icon, or the component targets
themselves) and varied-shape long memory remain pending; no quality admission follows.
   Synthetic final-model-size
   captures now reobserve labels at that exact transform (9 checks). Arbitrary
   post-labeled resize/invalid alpha still refuses; the API keeps queue/
   hold unknown. Current M2 training must resume its original archive.
   Preserve unknown middle positions and unobserved queue ends; never infer
   hidden length or delete locked cells using a guessed active/ghost footprint.
4. Expand synthetic appearances, crops/occlusion, queue arrangements and board
   association. Admit reviewed real labels and compare scratch/pretrained/direct
   decoders on independent session/ancestry/family splits. The166 user images
   are drafts, not a labeled benchmark. Broad render and real coverage are open.
   This chain is not executable from repo data: it needs reviewed, permissioned
   labels, so it is skipped on purpose rather than treated as a blocker.
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
| Queue/hold joint heads, masked losses and final synthetic captures | docs/M3_COMPONENT_TARGETS.md; joint-stream-contracts/ and joint-recovery-contracts/ | Joint learning, general resized real supervision and runtime decoding open |
| Native M3 resource/recovery | docs/evidence/m3-stream-pilot-profile-r1/, m3-stream-large-profile-r1/, m3-stream-recovery-gate-r1/ | Larger/pilot warm profiles and finite pilot recovery only; long varied-shape memory and quality open |
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
