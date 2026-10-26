# Current checkpoint — 2026-10-26

Repo C:/Users/micha/Code/tetris-state-vision, local Git, no remote. Goal ACTIVE.
No vision background process running; no run data deleted. This turn made
concrete progress (controlled learning, local-reading probe, streaming pipeline).

Verified milestones:
- 421de02: 128 steps, exact first-16 optimizer prefix; training exact boards2/6,
  development0/6. docs/evidence/m2-learn128/comparison.json.
- 331ac58: inner wall saved511/512 steps, explicitly wall_partial/exit2. Exact
  first128 prefix. Training boards6/6 but layers10/12, geometry.4106 cells fail
  declared joint fit; development0/6 and0/12. Ghost row15 vs target10 on dense
  clean/no-grid. docs/evidence/m2-learn512/comparison.json. Weight SHA256
  ca4c0f34a81b8c7591fab7db0e92fd89d652a27dea015cf63473c3f7b780bcdf.
- 72cde65: fixed-layout reading probe, 42 forwards, 36 visible one-cell changes/
  18 matched groups. Direction36/36, targeted20/36, exact changed boards18/36,
  two unrelated cells changed. docs/evidence/m2-local-reading/scope.json.
- Streaming pipeline now verified, ready to commit: 7 sampler +4 archive +3
  admission fixtures, plus real 512x512 one-step profile; source/model imports
  bind to the verified run archive, no tensor cache, all-final-tensor reload
  exact. 9.717s / raw peak549,154,816 B under1GiB/150s/one optimizer step.
  Curator re-rendered1 training +27 saved probe frames/labels exactly.
  docs/evidence/m2-stream-probe/scope.json; docs/M2_STREAMING.md. Model/loss/
  preprocess/render/visibility unchanged. Accuracy configuration not profiled.
  Supervisor24324/launcher7672/worker29788 verified exited.

Next selected task: supervisor system-wide commit-pressure admission and polling
before longer learning studies. Host at last check: committed30,895,226,880 B,
available11,370,631,168 B; another owner's Rust build remains live (last cargo
30968/31372, rustc24260, recheck authority). Do not touch their processes.
Do not start another substantial learning study at this pressure without a
guard. Small bounded tests/source work remain available.

Plan: read docs/BUG_LOG.md and official Windows GetPerformanceInfo contracts;
query OS commit independently of nested Jobs; reserve configured headroom at
launch, poll well before wall/snapshot cadence, write a terminal receipt then
kill the complete owned Job on pressure. No env/CLI opt-out. Preserve native
aggregate cap and all wall/work limits. Add a synthetic pressure tripwire using
tiny real Jobs, prove child/grandchild death without allocating large memory,
and ensure global OS counters aren't mistaken for nested Job counters.
Prior interactions: V-003 computed source hashes; V-004 current OS pressure is
distinct from historical Job peak; V-007 exact named Job still verifies worker
membership, global counters must not use implicit Job identity. V-001/002/005/
006/009 targets unchanged; V-008 malformed declarations stay fail closed.
Record any fixed resource defect in BUG_LOG with named green gate.
Before-change test now written: tests/test_pressure_guard.py synthetic critical
OS counter must refuse child launch, with a real96MiB native cap. Old runner
never queries it. Planned reproduction: one filtered test under128MiB/30s,
runs/v010-before.json/.log/.stderr.log. Expected <96MiB; no large allocations.
After fix: default threshold min(physical minus min(2GiB,physical/8),90% OS
commit limit), available-physical floor same reserve; reserve full job ceiling
before launch; poll every .2s. Counter failures fail closed. No opt-out.
V-010 OPEN: before gate reproduced child launch despite synthetic critical
pressure (runs/v010-before.stderr.log, one assertion failure). Supervisor fix
written; native cap unchanged, OS snapshots/default reservation/.2s polling
and source/guard identity exported. Next: pressure5 + existing resource5 suites,
small real Jobs; don't launch a learning job until these are green.
Streaming admission now also requires supervisor SHA/OS-guard identity to match
the source catalogue and actual probe receipt; old/no-pressure supervisors fail.
Next gate: pressure5 under256MiB/45s/5 cases, expected <192MiB, receipt
runs/v010-pressure-tests.json/.log/.stderr.log. Existing resource5 follows under
same envelope at runs/v010-resource-tests.*. Driver3 re-run after protocol change
under128MiB/30s at runs/v010-stream-admission-tests.*. No neural study yet.
Pressure harness launch PARKED at actual32,116,695,040 B commit against stop
31,368,351,744 B; no child created, zero tests executed. Supervisor30924 verified
exited. docs/evidence/v010-admission-only/scope.json preserves before-red + actual
refusal, status partial. V-010 remains OPEN; no native runtime kill gate green.
Current observed commit later32,797,134,848 B; cargo20160/22104 and rustc30468
last confirmed live. User async question pending about a lower-memory window;
no authorization to close apps, change another job or relax the guard assumed.
Source candidate can be checkpointed with its explicit partial verification;
remaining pressure5/resource5/stream-admission3 tests must run in a real admitted
window. No retries while above threshold; keep useful source work independent.
Source candidate checkpointed locally with only admission verification; V-010
remains OPEN. Byte-exact before-failure stderr intentionally includes one trailing
space; source/docs whitespace check excludes raw evidence and passed.

Independent M3 source slice while RAM is unavailable: pure queue/hold target
projection, not model training or a public schema change. Pre-change review:
V-005 requires no canonical index/EOS/count from unknown queue start or unseen
tail; absent differs from visible-empty; unknown middle indices retained.
V-001/002/006/009 preserve current image-supported State values and never infer
hidden labels or use board geometry to fill panel identities. V-003 reproducible
source remains recorded; V-004/007/V-010 resource gates are untouched.
Plan: capacity64 plus one EOS position, masked slots, independent status/identity
targets, explicit refusal of undersized capacity; paired identical-raster hidden
queue/hold fixtures required by standing visibility contract. Then tests under
128MiB/30s when the real guard admits them. No component head/loss wired yet.
Pure src/tetris_vision/component_targets.py and8 tests now written, not run.
Targets retain unknown-middle positions/presence, mask unseen-start indices,
supervise EOS/count only with both observed endpoints, distinguish absent/empty,
separate hold status/type, and refuse identified model output as labels.
Paired-raster fixture changes wholly hidden queue length/types and hold identity;
output targets must remain equal and unknown. These new labels are not consumed
by the current M2 trainer. Added module changes aggregate source digest only;
past artifacts keep their named commits/frozen source catalogue.
M3 source contract manually reviewed and documented in M3_COMPONENT_TARGETS.md;
eight tests remain UNRUN, no model head/loss integration. Preserve as a draft.
Next admitted sequence: v010-pressure-tests-r2 (5 cases,256MiB/45s), then
v010-resource-tests (5,256MiB/45s), v010-stream-admission-tests (3,128MiB/30s),
m3-component-targets-tests (8,128MiB/30s), each with matching .log/.stderr.log.
The first pressure receipt name is already occupied by a safe pre-child park;
never overwrite/reuse it. Recheck real OS memory and exact heavy process state,
not an old receipt or intent. Current host is still above31.4GB stop point.

The new streaming driver's source catalogue includes run_bounded.py, so a guard
change invalidates the old one-step admission for future studies. Re-profile
with a NEW output/receipt after guard verification; preserve the existing run.
Then use matching --admission for a finite fresh-state study, with one CPU thread,
native <=1GiB for pilot, explicit work/wall caps. Actual 512-square pilot peak
~524MiB provides the first estimate; accuracy-sized candidate needs separate
bounded characterization. Source archives let parent model work continue during
a longer run; archived helpers and lock are pinned, no live-source drift.

Earlier gates:85 core/model/resource at8d9d0f2;2 optimizer admission;7 runtime
plus actual one-image API;26 corpus;3 partial-evidence;14 stream contracts.
These are separate scopes, not one unified137. V-001..V-009 CLOSED on named logs.
Env .venv0.3.0, Torch2.14.1+cpu/NumPy2.5.4/Pillow12.3.0.

User-approved166 uncertain-answer screenshots remain unlabeled drafts only at
data/real-pilot-drafts/manifest.jsonl, zero training/scoring eligibility.
Seven private pages in runs/real-drafts-overview/. Aggregate receipts in
docs/evidence/real-drafts/scope.json. Never treat legacy labels/proposals as truth.

M2 broad learning, M3 queue/hold joint state, accuracy-sized model, diverse data,
reviewed real pilot/baseline, calibration/export/release remain OPEN.
