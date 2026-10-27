# Optional cell evidence for active and ghost learning

This is a controlled learning candidate,not a selected successor. The original
4096-update baseline continues with its original source/optimizer/recipe. At2176,
it still has0/18 exact development boards and0/36 exact layers. Exact target
replay of the investigated1664 prefix confirms every observed class was taught;
the fixed probes and ghost-position distribution have substantial coverage gaps.
Neither those facts nor the conditional lattice result proves a root cause.

Hypothesis: the global layer reader needs more direct local visual supervision.
Add optional `ModelConfig.layer_evidence=False`. When true,each decoder stage
predicts two independent20x10 visible-cell logit maps,one for active and one for
ghost. Predicted map weights pool visible cell features into their respective
layer readouts. Global context/original layer attention remain available,including
spawn rows. The public forward still accepts only pixels and validity. Ground
truth never enters normal forward or evaluation. Locked predictions are unchanged
by any explicit layer subtraction,and overlapping active/ghost targets are legal.

Targets derive solely from the supported public state and existing prepared
canonical image-coverage/layer masks,independent of locked occupancy certainty.
Unknown layer or unregistered cells supply no negatives.
Known absence supplies supported zeros. An observed layer supplies supported
positive cells within the visible20 rows;spawn cells remain the existing pose
head's responsibility. This visible-grid auxiliary target is not a complete
layer/state label. Independent balanced binary losses receive a predeclared.5
weight,averaged across stages,alongside existing losses. Report fully known
visible-grid metrics separately from existing exact type/pose layers.

Default false must preserve legacy parameter keys,initialization,forward tensors,
targets and losses exactly. Create new heads after existing initialization so a
paired random seed keeps common tensors identical. Checkpoint/configuration and
source identity distinguish the new path. Fresh studies only;never resume an old
optimizer into a changed topology. The boolean travels through training,resource
admission,archived execution,segment routing and diagnostic target preparation.
Old archived trainers must never receive its new keyword unless enabled in their
verified configuration. Historical source scopes remain historical.

Acceptance-gated todo:

1. Implement optional heads/pooling,masked targets/loss and explicit metrics.
2. Paired identical rasters with different hidden layers produce identical masked
   dense targets;also test absence,overlap,crops,invalid preparation,visible/spawn
   boundaries,masked zero gradients and dense-to-encoder/geometry gradients.
3. Prove default-off numerical/target/checkpoint compatibility with the actual
   original archive. Run relevant M2/M3/runtime/capture checks on named new sources.
4. Route explicit option through frozen trainer/config/segment/diagnostics;do not
   edit an active controller. Tests refuse profile/config/recipe mismatches.
5. Fresh two-full512-square-step warm-Adam/recovery characterization before any
   study,at measured640MiB pilot/1GiB larger ceilings. Caps/pressure/cache/thread
   policy remain unchanged. A warm profile does not prove long varied memory.
6. A finite fresh matched-source/data learning diagnostic reports ordinary exact
   boards,type/pose layers and visible masks. Preserve class/status/ancestry
   denominators;include broader class coverage before drawing comparisons.
   Lower auxiliary loss or a green profile does not select a release model.

Prior-bug interaction review (entire BUG_LOG read before this design):

- V-001,V-002,V-005,V-006,V-016: use observed/prepared masks;paired-raster policy
  gate;no latent cells,unregistered coordinates,hidden component presence or
  post-resize source certainty. No queue/hold observation policy change.
- V-003,V-014,V-018,V-020,V-021,V-022,V-024: computed artifact/config/source
  identity,frozen callable origins,original partial status,immutable active
  controller,full command/aggregate validation. No old scopes relabeled.
- V-004,V-011,V-012,V-017,V-019,V-023,V-025: preserve raw counters;warm resident-Adam
  profile;pre-import thread limits;validated fixtures/named role arguments;
  distinguish raw copies from normalized metadata. New gates must actually run.
  Absence fixtures must verify that the public observer sees the full spawn
  domain;latent missing poses never establish known absence.
- V-007,V-010,V-013,V-015: named kernel Job/pressure guards,finite memory/time/work,
  cache16,periodic recovery and atomic publication unchanged. No cap opt-out.
- V-008,V-009: real-corpus admission is untouched;no new real labels/eligibility.

Implementation is present with the option default off. Sixty selected model,
component,runtime and capture checks pass on32 hashed sources:
`docs/evidence/layer-evidence-core-contracts/scope.json`. They include exact
agreement with the byte-verified original M2 reference,real paired rasters,
independent overlap/locked uncertainty,masks/gradients and checkpoint reload.
The initial absence-fixture failure and its corrected public-status check remain
in that scope (V-025). Another79 protocol cases pass in source-pinned scopes
`layer-controller-contracts` (18), `layer-diagnostic-contracts` (26), and
`layer-error-contracts` (35). Fresh native warm admission and actual learning/
recovery comparisons remain pending. These unit results do not establish that
the new mechanism improves learning.

Native pilot gates now pass on catalogue
`f095d3526162e5bbee45e0800c017296d05d45ec78cd76f4583642c2c5b7ce47`.
True has647,228 parameters and562,204,672B peak; false has647,098 parameters and
561,950,720B peak, both under640MiB. Both full-square two-update profiles include
resident Adam, training restore and every final-output reload parity. The true
continuous6 versus3+resume3 gate also passes under1GiB aggregate (593,793,024B):
data, losses/terms/gradients, evaluations, model/Adam/RNG and modes are exact.
Evidence: `m2-stream-layer-pilot-profile-r1/`, `m2-stream-layer-control-profile-r1/`
and `m2-stream-recovery-gate-r3/`. These admit a finite learning pilot only.

Predeclared matched learning pilot (before any learning segment):

- Two separate fresh studies, global target768 updates each; first segments128.
  Names `m2-stream-layer-evidence768-s1` and `m2-stream-layer-control768-s1`.
  Preserve `step_partial128/768`; never silently extend a completed128 study.
- Both execute the above frozen source, pilot/capacity0, seed20261026, LR.001,
  weight decay.01, evaluation128, snapshot64, development groups6, CPU1/cache16.
  Profiles admit their respective actual boolean configuration. All common
  seeded parameter tensors were proven identical by the original-reference gate.
  The option adds130 parameters, local pooling and an auxiliary loss; this
  combined intervention does not isolate pooling from supervision.
- Require byte-identical training recipes and fixed-probe identities for the
  comparison. Report ordinary exact locked boards, nonempty boards and exact
  type/pose layers with denominators, plus geometry and common loss terms.
  True additionally reports observed visible-grid exactness separately from
  known-absent grids. Total joint loss is not comparable across the two objectives.
- The existing18-image development probe omits classes/statuses and has only
  six ancestry groups. It is a diagnostic, not independent model selection.
  No advantage/convergence claim from this segment without broader class/status/
  position coverage; no real screenshots enter the study.
- Each serialized segment uses a640MiB training Job,128-update work cap and an
  aggregate896MiB/420s ancestor deadline. Expected peak<700MiB from the native
  warm/recovery gates. Native pressure guards and source/recipe replay remain
  mandatory; inspect the saved result before choosing the next bounded segment.

The first128-update comparison verifies identical recipes, probe states and
source/configuration except the option. Both are0/18 development boards and
0/36 layers. True has.90222 cell accuracy and1.94832-cell geometry versus
false.83056 and2.21219; neither is adequate. Exact training layers are3/18 versus
7/18; that probe's nine ghosts are all absent. Named raw report and analysis:
`docs/evidence/layer-learning128-comparison-r1/`. Continue the fixed768 target,
not selection from these results.

Predeclared additional diagnostic before seeing its predictions: at768, evaluate
both actual frozen artifacts on development groups100..163, all three original
views (192 images from64 ancestries), seed20261026. These are distinct from the
fixed six development groups and every training ancestry (the frozen generator
uses a disjoint development seed domain). Require both models see identical
pixels/validity and no truth enters forward. Preserve every recipe/pixel/state
hash and raw predictions/targets. Report empty/nonempty boards, separate active/
ghost class/status denominators and type/pose errors, plus observed visible-grid
exactness for the enabled model. Do not count absent-grid success as observed
localization. Explicitly audit actual class coverage and position concentration.
This expands ancestry/class sampling within the same clean/no-grid generator;
it does not test unseen appearance families, realistic occlusion or real images.
No confidence interval treating three related views as independent trials.
