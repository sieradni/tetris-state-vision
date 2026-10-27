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
- V-004,V-011,V-012,V-017,V-019,V-023: preserve raw counters;warm resident-Adam
  profile;pre-import thread limits;validated fixtures/named role arguments;
  distinguish raw copies from normalized metadata. New gates must actually run.
- V-007,V-010,V-013,V-015: named kernel Job/pressure guards,finite memory/time/work,
  cache16,periodic recovery and atomic publication unchanged. No cap opt-out.
- V-008,V-009: real-corpus admission is untouched;no new real labels/eligibility.

All implementation/testing/learning gates are pending at this design checkpoint.
