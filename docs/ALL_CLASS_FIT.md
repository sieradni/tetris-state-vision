# All-class positive-control fit

The completed layer pilot shows class collapse despite supported class exposure.
This next diagnostic separates basic training-set fit from varied-layout learning.
It does not select a release model or replace the original 4096-update baseline.

First prepare 16 synthetic states, each with clean and no-grid views: 32 full
512-square images. Every active/ghost class, including known absence, has exactly
four views per role. Each observed piece has two poses. Geometry is fixed and
the full visible board and six-row spawn domain are inside the image. Half the
images have an eight-color bottom-row stack; half have no locked cells. Queue
and hold appearances are constant across classes. Ghost rows 2 and 6 replace
the previous single-row prior. These are controlled drawings, not physics or
real-game coverage. Related views share an explicit ancestry.

Labels must come from the existing public observer, with all 200 locked cells
and both role statuses supported. Do not use a latent pose to fill unknowns.
The current crop/visibility/semantic policy is unchanged. Require actual public
status/cell checks, complete-spawn absence, determinism, clean/no-grid label
agreement, all-class/pose coverage and bounds before any learning run.

Predeclared fit criterion: 32/32 exact locked boards, 64/64 known type/pose layers
(56 observed, 8 absent), every supported locked semantic label correct, and mean
geometry error below 0.25 source cells. Report every class and absence separately.
Only ordinary pixels/validity enter forward. Any supplied-geometry intervention
is a separate conditional diagnostic, never ordinary accuracy. Passing this
memorization criterion still does not demonstrate independent generalization.

The fixture and curriculum modules are now in frozen catalogue6147a558. Fresh
two-full-square-step warm-Adam/restore/output parity, default numerical
compatibility and actual2+6+3+3 recovery are verified; the new fit is admitted
by runs/m2-stream-recovery-gate-r5/profile. Historical profiles cannot admit
this changed recipe/catalogue. Use bounded streaming/checkpoint mechanisms
with explicit partial/recovery handling. No tensor cache, CPU1/cache16,
unchanged kernel/pressure guards, and more than40GB free before long training.

The fixture gate is now green: five public-label/coverage/replay/file-byte tests
on26 hashed sources, and a separate native32-image generation plus exact PNG/
state/recipe/source replay. Every role class, including absence, has four views.
Evidence: `docs/evidence/positive-control-fixture-contracts/scope.json` and
`docs/evidence/positive-control-fixtures-r1/scope.json`. These establish supported
fit data; separate fresh driver/resource/recovery evidence admits training.

Driver integration contract: add an explicit `curriculum` identity, default
`stream-v1`; `all-class-fit-v1` is M2/capacity0 only. Preserve every existing
stream recipe byte and its three-view sampling order. The fit cycles32 cases
in index/view order, with32 training-probe views and the existing independent
development generator. Its32-case population is reused, not fresh diversity.
The full-square resource profile repeats the first512-square case twice.
Record the curriculum in admission, recovery, summaries and controller routing;
refuse a resource profile or recovery state from another policy. Archive both
the adapter and fixture module. Old archived drivers never receive a new flag.
Readers replay the declared policy and compare the complete serialized record,
not merely a sample ID. Preserve all default numerical behavior; obtain fresh
source-bound gates/admissions before using the new catalogue.

The first fit study has a fixed1,024-update target, fresh initialization
seed20261026, pilot/capacity0, layer evidence off, AdamW LR.001/decay.01,
evaluation128 and snapshots64. Start with128 new updates and preserve explicit
partial128/1024. Use the admitted all-class recovery gate's nested profile,
including its three independent development groups (nine views); these are a
narrow diagnostic, not an all-class generalization population. Training cycles
the32 fit images, without cached tensors. If the criterion fails at1,024, record
the failure and inspect raw errors before declaring any extended study.

Saved fit points also need an independent ordinary-forward error report. Replay
the frozen32-case fit recipes and development recipes, and derive semantic/role
scores from raw head predictions and supported masks. Reproduce every saved
per-case field, including the fit metrics. Keep the old45-case report bound as
the default; an explicit fit policy may use a bounded68-case population. Bind
policy, family metadata, actual checkpoint and all aggregates in the collector.
These external diagnostic helpers are outside the frozen neural catalogue and
must not alter a live trainer, controller, or its imported helpers.

At128/1,024 updates, independent ordinary replay reproduces every saved score:
28/32 exact boards (all16 empty and12/16 nonempty),128/128 semantic head labels,
and.2159634-cell mean geometry. Both roles predict known absence on every fit
image:8/64 exact layers are the eight real absences;0/56 observed poses succeed.
Each role has28 type errors and14 column errors; active has28 row errors,ghost14,
and neither has an equivalent-rotation error. Four locked occupied cells are
missed, with zero locked false positives. The fit criterion is FALSE.

On the narrow9-image/3-ancestry development population, only the two empty boards
are exact;nonempty0/7 and roles0/18. This is not independent all-class coverage.
Evidence: docs/evidence/m2-stream-allclass128-errors-r1/scope.json, actual weights
f880c24d..., catalogue6147a558..., explicit step_partial128/1024. The41-forward
worker used440,119,296B/640MiB in17.475s; independent collector used77,279,232B/
128MiB in.731s. All known processes terminated; raw receipts/logs are preserved.
The39 error-report checks bind actual historical fixture/source hashes in
docs/evidence/allclass-error-contracts/scope.json. Continue the unchanged finite
study before choosing another mechanism or extending its target.

If the final fixed study still fails, keep two questions distinct. First, the
declared32-step cycle presents each class in four consecutive views and always
ends with the four absences; every64-step snapshot is at that same cycle phase.
The step97–128 curve's layer loss falls from4.24 on the first I view to.52 on
the final absent view. This is an order/recency hypothesis, not proof that data
order caused the failure. Second, the shared pose heads consume the output of
layer attention without a query residual; active/ghost identity enters through
attention weights. Their ability to retain separate row information is an
unmeasured representation hypothesis. Neither observation alone is a defect.

After the final ordinary error replay, inspect per-class predictions and role
feature separation before a new learning experiment. Any order-only comparison
must keep the same32 pixel/state pairs, initialization, optimizer,1,024-update
budget and acceptance criterion; change only a declared deterministic order,
then obtain fresh source/resource/recovery admission. A mechanism comparison is
a separate source-bound study. Do not silently shuffle or extend this live fit.

Prior-bug interaction review (current ledger read before this diagnostic):

- V-001/002/005/006/016: existing observer/prepared masks remain authoritative;
  full source coverage, distinct role cues and no resize/crop policy change.
- V-003/014/018/020/021/022/024: compute source/recipe/artifact hashes, execute
  frozen callables, bind actual configuration and native command, preserve
  controller identity and recompute aggregates. Old scopes remain historical.
  V-026 additionally requires the fixture writer's actual UTF-8 file bytes to
  match its recorded string/manifest identities across platforms.
- V-004/011/012/017/019/023/025: preserve raw native counters, warm optimizer,
  pre-import thread guard, observer-validated full-spawn fixtures and byte-exact
  evidence copies. No green claim for an unrun gate.
- V-007/010/013/015: named Job membership/readback, OS pressure monitoring,
  bounded native cache, periodic atomic recovery and finite work/wall/RAM remain.
- V-008/009: real corpus admission is unchanged; synthetic fit data do not make
  the 166 draft screenshots eligible for training or scoring.
