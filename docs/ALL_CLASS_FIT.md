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

No fit is admitted yet. The new fixture module must be included in a frozen
source archive before a driver can train on it; old resource profiles cannot
admit that changed recipe/catalogue. Reuse the bounded streaming/checkpoint
mechanisms, not the historical unarchived fixed-pilot driver. Require fresh
two-full-square-step warm-Adam/restore/output parity, then a predeclared finite
fit with explicit partial/recovery handling. No tensor cache, CPU1/cache16,
unchanged kernel/pressure guards, and more than 40 GB free before long training.

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
