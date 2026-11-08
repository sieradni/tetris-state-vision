# Contributor handoff — 2026-10-30

## M3 input representation (capture side 640): a declared negative — 2026-11-08

`docs/M3_IDENTITY_REPRESENTATION.md` predeclared the lever at `b229972`, before any
receipt: one value, the model capture side, 512 -> 640, threaded through
`model_capture_style` and the trainer's input budget, with every target, mask, order,
loss, architecture and curriculum term unchanged. The control is
`m3-identity-factor-r1` (128 updates, capacity 64, factor 4, separate read-out, seed
`20261026`, LR .001, `stream-v1`, eval/checkpoint 64, six development groups).

The rule is total over the rise x structure square and the observed state lands in its
**NEGATIVE** cell: the primary did not rise (exact queue piece tokens 3/54 training and
9/90 development, equal to control on both) while structure held on all eight declared
terms (training EOS 9/9, slot 54/54, presence 54/63, hold 0/9; development EOS 3/15,
slot 90/90, presence 90/105, hold 3/12). The replay also measures the control's exact
54 and 90 queue tokens and 9/12 hold cases, so the finer capture did not change the
supervised component population on these probes. The fit gate stays FALSE (0/9 and
0/18 boards, 9/18 and 0/36 roles, 0/9 and 0/18 exact readable components, geometry 3.82
and 2.28 cells).

Receipts: admission `runs/m3-identity-representation-profile-r3-job.json` (passed,
87.4 s, 944,017,408 B under 2,048 MiB), study
`runs/m3-identity-factor-representation-r1-job.json` (passed, 128/128 steps, 542.6 s,
986,275,840 B, checkpoint `acb6a8a9...`, catalogue `2d3623a1...`), independent replay
`runs/m3-identity-factor-representation-r1-errors-r1-job.json` (passed, 27 forwards,
39.0 s, 631,767,040 B, every saved metric reproduced, weights unchanged) curated in
`docs/evidence/m3-identity-factor-representation-r1-errors-r1/scope.json` (`passed`),
and the capture-side contract tests inside the whole suite (410 tests, 31.2 s,
540,966,912 B, `docs/evidence/m3-capture-side-tests-r2/`, base `73f0452`), which is
the suite re-run on the final code after the admission-guard repair; the earlier
run at `a892dfb` is preserved as `m3-capture-side-tests-r1`.

Three instrument defects surfaced and are recorded: V-045 (a predeclared ceiling above
the engine's 2 GiB bounded-job cap; lowered to 2,048 MiB, never the capture side),
V-046 (the profile admission compared its observed maximum to the literal 512^2,
refusing every other side; now `512**2 <= observed <= max_side**2`, green on the r3
profile), V-047 (retry receipts with suffixed names are invisible to readers that hard
require `runs/<name>-job.json`; the passing receipt now occupies the canonical name and
every refused attempt is preserved beside it).

Re-ranked, the executable identity levers are exhausted: the coefficient moves identity
at a structural cost, the read-out tying is a negative, the budget is a miss, and the
representation is a negative with structure held. What remains is the owner-decision
real-image chain (admitted, permissioned, reviewed labels) and the unchanged FALSE fit
gate, which no identity lever has moved. The next pass should close the fit gate on its
own terms or take the owner decision on real screenshots; another identity lever is not
ranked.

## M3 identity budget lever: identity rises, structure regresses — 2026-11-08

The budget lever is executed and judged against its own predeclaration (`9970e5a`).
Fresh admission `runs/m3-identity-budget-profile-r1-job.json` (passed, 943,157,248 B),
the 512-update study `runs/m3-identity-factor-budget-r1-job.json` (passed, 512/512 steps,
1729.7 s, 985366528 B under 1,152 MiB, checkpoint
`9b79863b`), its independent 27-forward replay
(34.9 s, 563867648 B, every saved metric
reproduced) and curation curate in
`docs/evidence/m3-identity-factor-budget-r1-errors-r1/scope.json` (`passed`).

By the predeclared rule this is a miss. Identity does rise with budget: exact queue
piece tokens 3/54 -> 12/54 training and
9/90 -> 12/90 development. But the
structure-held condition fails on two declared terms — training queue EOS
8/9 -> 5/9 (declared >= 8/9) and development
hold identity 3/12 -> 0/12 (declared
>= 3/12) — while slot status and presence hold and the fit gate stays FALSE
(0/9 and 0/18 boards,
9/18 and 0/36 roles,
0/9 and 0/18 exact readable
components, geometry 3.03 and 2.07).
Training roles also fall 12/18 -> 9/18. The observed state matched none of the three
declared branches, which is its own defect, V-044, now fixed by making the rule total
over the primary-rise/structure-held square.

Ranked next is therefore the input representation (pixels per icon, the model capture
side): budget moves identity but multiplies a structural trade-off, so extending the
budget again is not the next lever and is not ranked. That lever needs its own
predeclaration, admission and criteria; nothing about it is admitted here.

## M3 identity supervision measured, and the guard policy text corrected — 2026-11-08

The ranked open item after both identity levers closed was "the supervision signal",
so this pass measured it instead of declaring another lever. `docs/M3_IDENTITY_EXPOSURE.md`
records it: `docs/experiments/audit_identity_exposure.py` replays the 27 recorded probes
of `m3-identity-factor-r1` from the run's archived code with no model forward, refuses
unless recipe, raster and annotation bytes match, rebuilds the capacity-64 queue/hold
targets, tallies masked identity by class and slot, and joins the published per-case
diagnostics; `docs/experiments/record_identity_exposure.py` re-derives every count from
the per-case rows. Audit `runs/m3-identity-exposure-r4.json` (passed, 27 cases, 3.248 s,
266,608,640 B under 640 MiB) and curation `runs/m3-identity-exposure-r4-curation.json`
(passed, 128 MiB) curate in `docs/evidence/m3-identity-exposure-r4/scope.json`; the first
three attempts are retained beside it as receipts, each refused by the audit tooling or
its log capture rather than by a measured disagreement.

The counts: all seven piece classes are supervised in both splits (54 training tokens
across slots 0-5, 90 development across slots 0-8, 9 and 15 EOS labels), so coverage is
complete; per-class exposure is 3-18 unique labels, and hold identity sees four of seven
classes. Correctness is class-selective, not positional: the only identities ever exact
are queue `S` (3/3 training, 9/18 development) and hold `T` (0/3 training, 3/3
development), while six queue classes and three hold classes are at zero; development's
nine successes sit at slots 1, 3, 5 and 7, and all 15 development slot-0 labels are
wrong. With the coefficient lever having already moved the totals and the stream drawing a
new ancestry every three steps (128 distinct images from 43 ancestries in the
factor-4 run), what remains is optimization budget versus the input representation —
not global starvation and not class coverage. The budget lever is predeclared in
`docs/M3_IDENTITY_BUDGET.md`.

Also fixed: `docs/RESOURCE_GUARD.md` still described V-040's retired physical stop line
as binding (V-043, green on this audit receipt's `stop_commit_bytes`), so the policy record
now quotes the expressions the supervisor evaluates.

## M3 identity read-out tying: a measured negative, with the role head paying — 2026-11-08

`docs/M3_IDENTITY_READOUT.md` predeclared the lever at commit `e83d70e`, before any
receipt: replace the two sparse identity heads with the cell-type classifier plus one
learned EOS logit (`--component-identity-readout shared`), matched on capacity 64, 128
updates, seed `20261026`, learning rate .001, identity factor 1. `b80d82d` fixed the
first admission refusal at its root (V-039: the shared mode kept dead identity heads,
which the resident-Adam guard rejects); admission r2 passed on the exact mode
(`runs/m3-stream-identity-shared-profile-r2-job.json`, 80.218 s, 945,143,808 B,
checkpoint `c9e8f3f8...`, 262 tensors with warm Adam state), and the study completed
(`runs/m3-stream-identity-shared-r1-job.json`, 128 steps, 573.629 s of recorded
updates inside 574.770 s outer, checkpoint `55a35243...`, catalogue `2f69689a...`,
peak 949,555,200 B under the declared 1,152 MiB ceiling).

By the predeclared rule this is a negative. Identity does move: exact queue piece
tokens 0/54 -> 9/54 training and 0/90 -> 6/90 development, hold 0/12 -> 3/12
development. But the rule required structure held, and it is not: training presence
63/63 -> 54/63, development EOS 15/15 -> 13/15, and — the new cost — training roles
collapse 12/18 -> 0/18 where the coefficient lever had held them. Fit gate stays
FALSE (0/9 and 0/18 boards, 0/9 and 0/18 exact readable components), queue EOS 9/9
and slot status 54/54 and 90/90 still exact, geometry 3.32 training (better than 4.04)
and 2.55 development (worse than 2.03). Independent 27-forward replay (35.867 s,
554,905,600 B, weights `55a35243...` unchanged, every saved per-case field reproduced)
curates in `docs/evidence/m3-stream-identity-shared-r1-errors-r1/scope.json` (`passed`).

The verification gap is closed. The complete contract suite is green on this tree in
`docs/evidence/m3-identity-readout-fullsuite-r11/` (407 tests collected, `Ran 407 tests
in 72.290s`, `OK`, exit 0, receipt peak 545,845,248 B, admitted at 38,347,567,104 B
commit against a 45,626,134,118 B stop line). That count is auditable rather than
asserted: `git diff 5c6b41f HEAD -- tests/` adds exactly 9 test methods and removes
none, so 407 = the previous green run's 398 (its log is preserved at
`docs/evidence/audit-398-baseline-derivation/identity-factor-fullsuite-r1.stderr.log`,
which records `Ran 398 tests in 59.453s` and `OK`) + the 4 identity-read-out tests + the 5
pressure-decision tests; the 32 component-identity/component/joint-stream tests
sit inside it (`test_component_identity` 10, `test_components` 15, `test_joint_stream`
7). The collection is preserved as `collection-job.json`/`collection.log` beside the
suite receipt.

The 398 log deliberately sits in `audit-398-baseline-derivation/`, not beside the
frozen run's receipts: an evidence directory is an atomic record keyed by its name
(the recorders refuse an existing evidence path), so a frozen run's directory cannot
gain files (V-041). Closing it needed no test change and no
weakened receipt: the five earlier refusals (`-r1` mid-run, `-r2`–`-r5` at admission)
were a supervisor defect, V-040, because the commit stop line was derived from physical
RAM (`physical − 2 GiB` = 31.4 GB) while the exhaustible budget is the commit limit
(50.7 GB with this page file). The line is now the commit limit's 10% headroom and the
2 GiB available-physical floor still refuses genuine physical pressure (`-r4` was such a
refusal and remains correct). All refusal receipts are preserved; nothing was re-scored.

So both identity-side levers reachable from current data are measured and closed: the
loss coefficient (positive on localization, structural cost) and the read-out
tying (negative — identity moves, structure does not hold and the role head collapses).
The fit gate stays FALSE. Open items narrow to the supervision signal (54 queue tokens
and 9 hold cases against ~1,800 supervised cells per pass), the input representation
(pixels per icon), and the owner decision on admitting real screenshots.

So read-out sharing is not the identity lever either: tying supervision into the cell
classifier moves the sparse identity targets but buys nothing readable and costs the
role head. Per the predeclaration the next lever is the input representation (pixels
per icon) or the component targets themselves; the coefficient and read-out questions
are closed. Real-image items 4–5 remain an owner decision (reviewed, permissioned
labels).

## M3 component-identity lever: the identity scale is a measured positive — 2026-11-08

`docs/M3_COMPONENT_IDENTITY.md` predeclares the lever at commit `65ee2f1`, before any
receipt: 128 matched updates at capacity 64 with the queue-token and hold-piece terms
scaled by a declared factor 4, everything else identical to `m3-stream-joint-fit-r1`.
Fresh admission `runs/m3-identity-factor-profile-r1-job.json` (passed, 62.276 s,
952,553,472 B under 1,152 MiB, checkpoint `e1ba82fc...`, catalogue `b3caba97...`), then
`runs/m3-identity-factor-r1-job.json` (passed, 494.121 s outer, 436.42 s of updates,
982,077,440 B peak, 128 steps, checkpoint `acb6a8a9...`).

By the predeclared rule the lever is a measured positive on its localization: exact
queue piece tokens rise 0/54 -> 3/54 on the training probes and 0/90 -> 9/90 on
development, and development hold identity 0/12 -> 3/12, with slot status still exact
(54/54 and 90/90). The fit gate stays FALSE (0/9 and 0/18 boards, 12/18 and 0/36 roles,
0/9 and 0/18 exact readable components, 3.83 and 2.21 cells). The declared structural
regression also shows: development EOS falls 15/15 -> 3/15 and training presence
63/63 -> 54/63, so the scale buys identity at a measurable structural cost. Losses are
not comparable across factors because the scaled terms enter them.

Independent replay `runs/m3-identity-factor-r1-errors-r3-job.json` (27 forwards,
36.492 s, 551,710,720 B, weights `acb6a8a9...` unchanged) reproduces every saved
per-case field and curates in
`docs/evidence/m3-identity-factor-r1-errors-r3/scope.json` (`passed`).

The chain exposed two instrument defects, both fixed at their root (V-038): the replay
scored the published factor-1 objective and so could not reproduce a factor-4 run's
saved losses, and the owned-run-name guard did not list the predeclared
`m3-identity-factor-*` family. Real-image items 4–5 remain unreachable without
reviewed, permissioned labels.

Next: identity targets or representation at this scale (the coefficient question is
closed). The staged head refinement stays a measured negative, and development stays
comparable across stages on the canonical 18 images at 4/18 boards, 0/36 roles,
.901944 cells and 3.530471 geometry with loss 11.0885 versus 11.8129.

## First M3 joint learning is a measured negative — 2026-11-08

Roadmap item 3's open half is measured for the first time. Items 4–5 were checked
first and skipped on purpose: they need admitted, permissioned, independently
reviewed real labels, and the tree holds only 166 real drafts with zero
training/scoring eligibility. Nothing in the repo converts those pixels into
admitted labels, so the preferred real-image chain is an owner decision (review
plus permission), not an executable pass. `docs/M3_JOINT_FIT.md` records it.

The next ranked item repo data supports is the joint queue/hold heads, which had
never been trained. `docs/M3_JOINT_FIT.md` predeclares the study at commit
`90218b3`, before any receipt existed: 128 updates of the accuracy model with
`--component-capacity 64` on `stream-v1`, seed `20261026`, evaluation and
checkpoint every 64, canonical 6 development groups, criterion = 9/9 exact boards,
18/18 exact roles, 9/9 exact readable components and <= 0.25-cell geometry on the
run's own nine training probes.

Fresh capacity-64 admission `runs/m3-stream-joint-fit-profile-r2-job.json`
(passed, 82.759 s, 943,411,200 B under 1,152 MiB, checkpoint `ad14d69a...`,
catalogue `7074e6a6...`, work-item bound 2), then
`runs/m3-stream-joint-fit-r1-job.json` (passed, 417.353 s outer, 360.29 s of
updates, 984,649,728 B peak, 128 steps, checkpoint `b2e44c9c...`).

Result: the criterion is FALSE. Training probes 0/9 exact boards, 12/18 exact
roles (ghost 9/9, active 3/9), .8494 cell accuracy, 4.0417-cell geometry, loss
7.739; development 0/18 boards, 0/36 roles, .8481 cells, 2.0294 geometry, loss
7.332. Against the two-update profile (roles 3/18, geometry 19.82, loss 29.97
training; geometry 16.18, loss 30.56 development) the optimizer clearly trains:
geometry and loss collapse, ghost identity fits. Components do not: queue EOS
9/9, slot status 54/54 and presence 63/63 are exact while piece tokens are 0/54
and hold identity 0/9, so every readable component fails on identity alone. That
is the M2 class-collapse signature localized to the queue-token and hold-type
heads.

Independent replay `runs/m3-stream-joint-fit-r1-errors-r2-job.json` (27 forwards,
35.184 s, 546,545,664 B, weights `b2e44c9c...` unchanged,
`automatic_metrics_reproduced`) is curated in
`docs/evidence/m3-stream-joint-fit-r1-errors-r2/scope.json` (`passed`). Three
launches were refused for authored rather than derived bounds (wall 2,700 s vs the
trainer's (0, 1800]; admission work items 256 vs the required 2; probe work items
64 vs the required 27 forwards) and all receipts are retained; no code, fixture or
gate defect was found.

The staged head refinement remains a measured negative, and development stays
comparable across stages on the canonical 18 images: stage 1 (ordinary replay of
its artifact) and stage 2 both reach 4/18 exact boards, 0/36 exact roles, .901944
cell accuracy and 3.530471-cell geometry, with development loss 11.0885 versus
11.8129. The M3 study does not change those numbers.

Next: component-identity learning is now the narrowest open M3 lever (targets,
loss weighting or a longer fit at the same criterion), each needing its own
predeclaration and admission; varied-shape long memory and review-only runtime
decoding stay open, and the real-image chain still needs admitted labels.

## Staged frozen-backbone head refinement is a measured negative — 2026-11-08

The roadmap's next joint-learning lever is done and closed as a measured negative.
`docs/STAGED_HEAD_REFINEMENT.md` predeclares it at commit `e601182`, and every
staged receipt is later. Stage 2 loads the ordered artifact
`runs/m2-stream-headrate-control-s3` (`1f20007c...`), freezes everything but the
shared type head and starts a fresh AdamW over the same 1,024 ordered pairs.

The fresh native admission for supervisor `d9382fa2...` passed on the exact
staged mode: `runs/staged-head-refine-profile-r3-job.json`, 442,085,376 B peak
under the 671,088,640 B ceiling, 36.287 s, warm backward over exactly 2
parameter tensors, source catalogue `7074e6a6...`. The study then ran as one
interrupted first attempt (`runs/staged-head-refine-interrupted-s1`, outer command
cap at step 716, retained), a 512-step segment stopped by the OS commit guard at
430,661,632 B, a 404-step resume stopped the same way at 429,887,488 B (both
external pressure, not training failure), and `runs/m2-stream-staged-head-r1-s3`,
which restored the step-384 snapshot and finished 640 updates to step 1,024:
`passed`, peak 433,860,608 B, 268.64 s, checkpoint `e02a02e7...`.

Result: exact locked boards 32/32, semantic cells 128/128 and geometry
.21058594 all stay identical to the digit, while exact training roles rise from
20/64 to 36/64 (observed 12/56 to 28/56, absences 8/8). The predeclared 64/64 role
gate fails, and the run stops 26 roles short of the 62/64 copied-head reference, so
head-only refinement recovers about a quarter of that gap and then stalls. That
retires the "copied head is reachable from pixels" hypothesis without proving the
joint forward lacks class information. Development is now comparable on the contract's canonical six-group set: an ordinary
replay of the stage-1 artifact over the same 18 images
(`m2-stream-headrate-control-s3-dev6-errors-r1`, 50 forwards, weights `1f20007c...`
unchanged, every saved metric reproduced, curator in
`docs/evidence/m2-stream-headrate-control-s3-dev6-errors-r1/`) reaches the same
4/18 exact boards, 0/36 roles, .901944 cells and 3.530471 geometry, with stage 2's
loss higher (11.8129 vs 11.0885). V-036 closes the denominator discrepancy; stage
1's published 9-image row (2/9 boards, 0/18 roles) is a saved subset, not a
different result.
Independent evidence: `runs/m2-stream-staged-head-r1-s3-errors-r1/result.json`
(50 forwards, weights unchanged, every saved per-case metric reproduced, 425,635,840 B
/ 20.032 s) and the curator's
`docs/evidence/m2-stream-staged-head-r1-s3-errors-r1/scope.json`.

The chain exposed and the fix log now closes two defects. V-034: the ownership
test for a staged start was written against `ROOT/runs`, but the driver re-executes
from its verified archive at `runs/<name>/code`, so the stage refused its own
source; ownership is now the written run's exact sibling and is still re-read from
source bytes. V-035: the flat exclusion between initial weights and resume blocked
the only valid resumed form, since a staged run must restate the frozen mode and
start artifact that its recovery contract stores. Carried limit: a segment cannot
be resumed after the repository source changes, because the recovery contract
pins the source catalogue by design.

Next: with accumulation, interleaved ordering, type-head rate and head-only
refinement all closed as measured negatives, the next candidate is a
representation/architecture or data-coverage lever predeclared from scratch, and
every new mode needs its own fresh native admission. The real-image and complete
state measurement chain remains the top open quality item, and the M2/M3 gates are
unchanged.

## Accumulation arms replayed, V-030 closed — 2026-11-08

The two completed accumulation arms now have independent ordinary replays and
curators, so their learning outcome is citable rather than trainer self-report:
`docs/evidence/accumulation-arm-balanced-errors-r1/scope.json` (50 forwards,
424,710,144 B, 19.523 s under the unchanged 640 MiB ceiling, weights
`a5e6bc56...`) and `docs/evidence/accumulation-arm-online-errors-r1/scope.json`
(424,316,928 B, 19.202 s, `93e79820...`), both from base commit `64516ad`. Each
reproduces every saved per-case metric exactly and leaves the model tensors
unchanged. Balanced-eight reaches 16/32 exact training boards, 34/64 roles and
.7507 cells geometry; online interleaved reaches 28/32, 37/64 and .3862. Both
fail the predeclared fit, and development stays 4/18 boards (all four empty) and
0/36 roles for both, with development loss rising from 6.6425 to 11.7978 and
13.3888. Batching eight images per update is therefore a measured negative
result; the ordered-eight arm was never run, so no mechanism claim follows.
`BALANCED_ACCUMULATION.md` holds the full tables and limits.

`scripts/run_bounded.py` now resolves the child executable before the Job is
armed and refuses a relative forward-slash path, closing V-030; receipts record
`child_executable`, and a refusal is marked `launch_refused` with no `child_pid`
so it reads as an instrumentation failure rather than a failed gate. The
supervisor hash is now `d9382fa2...` (was `d2fb1478...`), so fresh native
admission is required before any further long training window, and copy-paste
commands must name the interpreter with an absolute or backslash path. The
shared `require_owned_run` policy also admits the accumulation arms to the error
probes and curators (V-033). One 68-case gate passes at 355,688,448 B under
640 MiB: `docs/evidence/v030-launch-preflight-r1/scope.json`.

Next: with the accumulation direction closed as a negative result, predeclare the
next joint-learning lever and its resource/recovery admission. Controller/curator
routing for the accumulation cursors and an in-group interruption at arm scale
stay open but are no longer on the critical path. Every V-entry in the ledger is
now closed on a named receipt; the M2/M3 quality and real-image chain remains
fully open.

## Accumulation integration — 2026-11-08

`scripts/train_m2_stream.py` now executes the predeclared whole-group arms when
`--accumulation-policy` is given: each group is validated and journaled only
after its optimizer step, the stepped gradients are cleared before any boundary,
evaluation fires only at the declared image cadence, and a boundary snapshot is
published only at a complete-group boundary. Image and optimizer cursors stay
separate in `curve.jsonl`, `training.jsonl`, progress and summary, and the
default streaming path is unchanged. Five scoped driver checks pass under448MiB
(`docs/evidence/accumulation-driver-integration-r1/`): accounting/cadence,
per-group-size admission, the profile CLI coupling, a forced in-group
interruption replayed to byte-identical model/Adam/RNG/journal bytes, and the
native profile characterization. 29 accumulation,43 stream and5 type-head
regression tests also pass. The pilot model's
`--profile-square --accumulation-policy` path ran natively for both predeclared
group sizes at640MiB: size1 519,929,856B /33.30s /0.060 updates/s, size8
561,786,880B /29.35s /0.068 updates/s, each with a byte-exact boundary reload
(0.212s and0.086s) and a zero reload delta because the warm peak was already
higher; evidence in `docs/evidence/accumulation-profile-r1/`. Admission now
requires the matching policy, group size, `2*group_size` microsteps and a
positive measured update rate. Both arms then ran natively to their full1,024-image budgets across two bounded
segments each, every segment admitted from the measured two-update profile:
balanced-eight 64+64 updates (175.3s/179.8s in-child, receipts175.944s/180.456s,
peaks565,817,344B and567,971,840B, peak difference+2,154,496B), online-interleaved
512+512 updates (216.1s/289.6s, receipts216.863s/290.393s, peaks565,100,544B and
569,487,360B, peak difference+4,386,816B). The first journal is an exact byte
prefix of the resumed one and the resume pointer matches the published boundary;
measured reload cost0.219s/0.200s. Every published number is re-derived from those
bytes by `docs/evidence/accumulation-arm-r1/audit/audit_arm_evidence.py` (no
mismatch, result in that folder's `audit/audit.json`), which also
records the wall-clock and rate-denominator definitions and the
receipt-admitted headroom (tightest101,601,280B under the671,088,640B ceiling)
(`docs/evidence/accumulation-arm-r1/`). Resume now goes through
`restore_boundary`. A bounded partial segment exits2 with `status: step_partial`,
so its supervisor receipt says `failed` while the summary carries the partial
status. Still open: automated controller/curator routing, an independent final
pixel replay, and an interruption inside a group at arm scale.

## State recovery — 2026-11-08

Recovered an unrecorded change set on local `main` 86f7362 (8 unpushed commits;
three untracked files, no tracked modifications). `scripts/accumulation_checkpoint.py`
is a durable group-boundary bridge: fresh exclusive journal creation, append only
of validated whole groups after a successful optimizer step, boundary-only
model/Adam/RNG publication with no pending gradients, refusal of a mismatched or
wrongly advanced optimizer, and restore into a new directory that excludes every
uncommitted suffix record. Eight no-Torch contract checks pass under128MiB:
`runs/accumulation-checkpoint-pure-r6-job.json`,0.522s/37,761,024B, and one Toy
Torch group-interruption check passes at640MiB/332,423,168B; receipts and logs
are in `docs/evidence/accumulation-checkpoint-contracts-r1/`.
The first Torch attempt was refused with `os_commit_reservation` before any child
launch (`runs/accumulation-checkpoint-torch-r1-job.json`, memory_pressure,125,no
child) and is retained as an admission refusal, not a test result; r2 passed once
pressure cleared. A launch trap is recorded as V-030: a relative
forward-slash child interpreter fails in `CreateProcess`, so bounded children
must use an absolute or backslash path form; the supervisor is unchanged.
The bridge is used by the trainer's accumulation arm only (see the new
checkpoint above): the controller and independent curator still do not route
these cursors, ordinary fixed-fit recipes only; two-update full-square group profiles and native
segment-boundary resume now exist for group sizes1 and8; both1,024-image arms are
complete, and no interruption has been forced inside a group at arm scale. At
that recovery point no training had been admitted and no research job was running.
Next: route the image/update cursors through the controller and independent
curator, wire accumulated resume to the boundary restore, then obtain fresh
native full-square warm/reload profiles per group size and an actual in-group
interruption recovery before any arm of the predeclared comparison. Commit pressure was31.2GB against the31.368GB stop
at review and briefly cleared to1.12GB of headroom, so long training windows stay
blocked. Local commits remain unpushed;
no push without explicit authorization.

## Accumulation journal contracts — 2026-11-07

System commit was 30,986,448,896 B against 31,368,351,744 B stop when a
512 MiB reservation was checked, so no heavy work was relaunched. Added
`scripts/accumulation_journal.py`: validates whole public-recipe groups,
separate image/update cursors, complete-group snapshot positions and exact
optimizer ownership; its execution boundary preflights before calling the
existing group primitive and only returns complete records. Eight no-Torch
checks pass in1.565 s / 41,299,968 B under128 MiB / 30 s / work8; raw scope
`docs/evidence/accumulation-journal-contracts-r1/`. Execution is mocked in
these tests and ordinary fit recipes only are supported; full-square profiles
need explicit handling. The actual trainer, durable journal publisher,
controller and independent curator remain unintegrated; no neural workload or
training admission follows. Continue those changes and fresh native gates when
resource pressure safely permits; never disable the existing guard.

## Balanced accumulation preparation — 2026-11-07

`docs/BALANCED_ACCUMULATION.md` predeclares online-interleaved, balanced-eight
and ordered-eight arms at1,024 images each, explicitly distinguishing their
1,024 versus128 optimizer updates. `scripts/accumulation_contract.py` contains
finite group mapping, boundary validation and a streamed mean-gradient primitive;
seven bounded checks pass, including actual public-label mapping, full-batch
gradient parity, rejected partial groups and toy interruption/recovery parity.
Evidence `docs/evidence/accumulation-contracts-r1/`: 5.806 s / 332,820,480 B,
640 MiB / 30 s / work7. This is not integrated or native admission. Next
integrate image/update accounting, group-boundary checkpoint/replay and
controller/curator routing, including an optimizer-owns-all-parameters check,
then obtain fresh full-square resource and actual
in-group interruption/recovery gates before any study. No job remains active.
An additional ownership-check test run was declined by the OS reservation guard
(r4), then its 512 MiB retry was stopped by OS commit pressure after two tests
(r5). Both raw receipts/logs are preserved in the named evidence. The unverified
ownership edit was reverted byte-for-byte to the seven-test r3 source; do not
claim the extra check passed or relaunch under unchanged pressure.

## Checkpoint forgetting diagnostic completed — 2026-11-07

`docs/TYPE_FORGETTING_DIAGNOSTIC.md` and the source-bound scopes in
`docs/evidence/type-forgetting-{control,candidate,comparison}-r1/` record 17
checkpoints/544 ordinary pixel forwards per arm, independently recomputed.
Control type correctness ranges8..20/64; fourfold candidate stays8/64 at
every sampled checkpoint despite80 correct-to-wrong and80 wrong-to-correct
role transitions across 16 intervals. The control records86/98 transitions.
These are repeated dependent roles, not independent events, and intervals may
hide intermediate reversals. Local final-head class-gradient cosines are not
full joint/Adam gradient evidence. No new real-image or generalization claim.
Both observers, curators and comparator passed; no diagnostic process remains.
Next predeclare and safely admit a matched class-balanced gradient accumulation
test with equivalent example exposure and explicit optimizer budgets; do not
train or alter frozen studies until its design and admission gates pass.

## Type forgetting diagnostic in progress — 2026-11-07

The completed rate comparison remains frozen. A new read-only diagnostic
replays all32 supported training images at initialization and each64-update
checkpoint through1024 (17 checkpoints, 544 forwards per model). It captures
actual final type-head features/logits and supported public targets, verifies
each retained snapshot's tensor and ledger-prefix identity, replays exact
fixture pixels, and checks final outputs against the independent ordinary
report. The separate pure analyzer computes per-class mean final-head CE
gradients, cosine and checkpoint-to-checkpoint type changes; this is not the
complete joint gradient, Adam update, feature-gradient or real-image evidence.
Two mathematical/unknown/refusal tests passed under128 MiB / 30 s / work2.

Control observer planned: `docs/experiments/probe_type_forgetting.py --arm
control --output runs/type-forgetting-control-r1`, 640 MiB / 600 s / 544
forwards, expected below600 MiB, CPU1/cache16 and existing pressure guard.
Receipt/log prefix `runs/type-forgetting-control-r1`; preflight output, log,
receipt names and capture process identity. One heavy job at a time. Candidate
follows under the same bounds only after control terminates. An independent
bounded collector is required before interpreting the outputs.
Control observer passed 166.366 s / 427,859,968 B, 544 forwards and final
independent-report agreement. After verifying process death, candidate observer
uses `docs/experiments/probe_type_forgetting.py --arm candidate --output
runs/type-forgetting-candidate-r1`, same 640 MiB / 600 s / 544 forwards,
expected below600 MiB, CPU1/cache16 and pressure guard. Receipt/log prefix
`runs/type-forgetting-candidate-r1`; preflight all paths.
Candidate passed 170.071 s / 423,342,080 B, 544 forwards and final independent
report agreement; all observer processes gone. Stronger analyzer tests r2
passed. Sequential independent curators planned: `docs/experiments/
record_type_forgetting.py --arm {control,candidate}`, 128 MiB / 30 s /
544 role-image forward records per arm, expected below100 MiB, CPU1 and
pressure guard. Receipt/log prefixes `runs/type-forgetting-{arm}-curation-r1`,
new evidence directories `docs/evidence/type-forgetting-{arm}-r1/`.
Both curators passed; control 42,598,400 B, candidate 42,672,128 B under
128 MiB, with actual checkpoint/logit reconstruction and public-target checks.
Matched class-distribution comparator planned `docs/experiments/
compare_type_forgetting.py`, 128 MiB / 30 s / 1088 image-checkpoint cases,
expected below100 MiB, CPU1/pressure guard, receipt/log prefix
`runs/type-forgetting-comparison-r1`; preflight new evidence path.

## Joint type-head fit completed — 2026-11-07

Both fresh same-source fixed1024 studies and independent ordinary41-image
replays/collectors are complete. `docs/TYPE_HEAD_JOINT_OPTIMIZATION.md` and
`docs/evidence/type-head-joint-comparison-r1/scope.json` supersede the earlier
in-progress entries below. Both joint-fit gates FAIL: control32/32 boards,
20/64 roles,128/128 semantics, geometry.211; candidate32/32 boards,8/64 roles,
128/128 semantics, geometry.108. Development both2/9 boards (both empty),
0/18 roles. Source18ce798e..., actual initial tensors and full1024-row data
ledgers match; control weights1f20007c..., candidate8ec2cfab.... The fourfold
head rate is not selected. One seed, 16 training/3 dev ancestries; no real or
generalization claim. Do not extend either finished study. Next investigate
per-class gradient direction and forgetting on supported fixtures and declare
any new intervention before training. No research job remains active.

## Initial tensor admission — 2026-11-07

The trainer now saves and reads back actual pre-evaluation initial state tensors;
their file identity is checked before completion and by independent curators.
Focused tests passed. Candidate gate r9 is planned with explicit output/receipt/
log preflight and free-disk check: parent 1024 MiB / 600 s / 14 updates,
child 640 MiB / 90 s, CPU1/cache16 and existing pressure guards, expected
under 800 MiB. Command `scripts/probe_stream_recovery.py --output
runs/m2-stream-recovery-gate-r9 --curriculum all-class-fit-v1
--type-head-rate-factor 4`; receipt/log prefix `runs/m2-stream-recovery-gate-r9`.
It performs profile2 then admitted continuous6 and partial3+resume3, not the
1,024-update study. Control r10 requires a separate same-source gate.
r9 passed 147.947 s / 599,404,544 bytes; terminal receipt and all matching
processes must be checked before control. Control r10 uses the same limits,
ordered curriculum, factor1 and namespace/disk preflight, expected under
800 MiB. Receipt, output and logs use `runs/m2-stream-recovery-gate-r10`.
r10 passed 142.762 s / 600,817,664 bytes, pending process-death checks.
Curate r9 and r10 sequentially at 128 MiB / 30 s / 181 items, expected under
100 MiB with CPU1 and existing guards; use new `-curation` receipt/log prefixes
and preflight them. Then compare actual initial tensors with a separate 640 MiB /
30 s / two-snapshot collector, expected under 400 MiB. No full fit yet.
r9/r10 curators passed, each replaying 17 records and 164 saved probes. The
matched r2 collector verified actual initial tensor bytes identical, initial
probe metrics, first loss/gradient norm, six data recipes and optimizer groups;
2.259 s / 257,298,432 bytes under 640 MiB. Source catalogue 18ce798e...;
evidence `docs/evidence/type-head-initial-admission-r2/` and gate r9/r10 scopes.
The finite recovery and start-comparison admissions pass; full-study initial
artifacts still need comparison to these starts before final claims.

Control study first segment planned: `scripts/run_stream_segment.py --name
m2-stream-headrate-control-s1 --target-steps 1024 --updates 128 --admission
runs/m2-stream-recovery-gate-r10/profile`. Plan-only receipt/log
`runs/type-head-control-plan-r1*` passes and binds catalogue18ce798e.... Parent
cap 1280 MiB / 2100 s / work297 (128 training updates plus 169 curation
items), child 640 MiB / 1860 s / 128 updates, serialized curator 128 MiB /
120 s / 169 items; expected parent under 800 MiB, CPU1/cache16 and existing
pressure guard. Parent receipt/log prefix `runs/type-head-control-s1-controller`;
child receipt/logs use the run name, and controller progress is in its pipeline
directory. Verify all output/receipt/log paths absent, >40 GB free and exact
startup/death process command lines. This segment is partial128/1024, not a
fit verdict. Afterward compare its actual initial tensors to the admitted
control, and conduct the matched candidate segment with its own profile.
Control s1 completed128/1024, explicit `verified_partial`, 140.772 s /
600,555,520 bytes, exact saved initial file SHA256
51c58ee18ca9df6de3f03ba5430251ed831b662da360e0446b2eeced7e1e2aa8 matches
the admitted control. No matching processes remain. Candidate s1 plan-only
receipt/log `runs/type-head-candidate-plan-r1*` confirms factor4 and same
source/config/seed/recipe. Planned command `scripts/run_stream_segment.py
--name m2-stream-headrate-candidate-s1 --target-steps 1024 --updates 128
--admission runs/m2-stream-recovery-gate-r9/profile`, parent 1280 MiB /
2100 s / work297, child 640 MiB / 1860 s / 128 updates, curator 128 MiB /
120 s / 169 items. Expected parent under 800 MiB; CPU1/cache16, kernel and
pressure guards unchanged. Preflight every run/receipt/log/evidence path and
>40 GB free. Parent receipt/log prefix `runs/type-head-candidate-s1-controller`.
Candidate s1 verified partial128/1024, parent 136.545 s / 602,480,640 B;
no matching processes remain. Both actual initial files have SHA256 51c58ee1...
and their complete 128-row data ledgers have SHA256 9e4d4248...; independent
segment scopes are `docs/evidence/m2-stream-headrate-{control,candidate}-s1/`.
No accuracy verdict follows. Next control s2 plan-only passes: resume s1 at
128, cap448 new updates, target1024, factor1 and identical source identity.
Command `scripts/run_stream_segment.py --name m2-stream-headrate-control-s2
--target-steps 1024 --updates 448 --admission runs/m2-stream-recovery-gate-r10/profile
--resume-from runs/m2-stream-headrate-control-s1`. Parent 1280 MiB / 2100 s /
work1065 (448 updates plus 617 curation items); training child 640 MiB / 1860 s /
448 updates; curator 128 MiB / 120 s / 617 items. Expected parent under 800 MiB,
CPU1/cache16 and pressure guard unchanged. Preflight all paths and >40 GB free;
parent receipt/log prefix `runs/type-head-control-s2-controller`. One heavy job.
Control s2 verified partial576/1024, 414.573 s / 601,214,976 bytes; process
tree gone and checkpoint at576. Candidate s2 plan-only passed; resume its s1
at128 for448 updates to576 with factor4 and same frozen source/target. Planned
`scripts/run_stream_segment.py --name m2-stream-headrate-candidate-s2
--target-steps 1024 --updates 448 --admission runs/m2-stream-recovery-gate-r9/profile
--resume-from runs/m2-stream-headrate-candidate-s1`. Parent 1280 MiB / 2100 s /
work1065; child 640 MiB / 1860 s / 448 updates; curator 128 MiB / 120 s /
617 items; expected parent under800 MiB, CPU1/cache16 and pressure guard.
Preflight all paths and >40 GB free. Parent receipt/log prefix
`runs/type-head-candidate-s2-controller`. No fit verdict at576.
Candidate s2 verified partial576/1024, 394.045 s / 601,661,440 bytes;
process tree gone. Control/candidate training ledgers at576 and initial state
files hash identically. Control s3 plan-only passes: resume576, 448 updates
to fixed1024. Planned `scripts/run_stream_segment.py --name
m2-stream-headrate-control-s3 --target-steps 1024 --updates 448 --admission
runs/m2-stream-recovery-gate-r10/profile --resume-from
runs/m2-stream-headrate-control-s2`. Parent 1280 MiB / 2100 s / work1513
(448 updates + 1065 replay items); child 640 MiB / 1860 s / 448 updates,
curator 128 MiB / 120 s / 1065 items. Expected parent under800 MiB, CPU1/
cache16, unchanged pressure guard. Preflight all paths and >40 GB free;
parent receipt/log prefix `runs/type-head-control-s3-controller`. Final
ordinary pixel replay remains separate and mandatory.
Control s3 completed1024/1024 and native curator passed, 401.226 s /
605,925,376 bytes. Final weights 1f20007c... match the historical ordered
baseline, but new source/evidence scope remains distinct. All matching
processes gone; independent ordinary pixel replay remains pending.
Candidate s3 plan-only passes: resume576, 448 updates to fixed1024. Planned
`scripts/run_stream_segment.py --name m2-stream-headrate-candidate-s3
--target-steps 1024 --updates 448 --admission runs/m2-stream-recovery-gate-r9/profile
--resume-from runs/m2-stream-headrate-candidate-s2`. Parent 1280 MiB / 2100 s /
work1513; child 640 MiB / 1860 s / 448 updates; curator 128 MiB / 120 s /
1065 items; expected parent under800 MiB, CPU1/cache16 and pressure guard.
Preflight all paths and >40 GB free; parent receipt/log prefix
`runs/type-head-candidate-s3-controller`. Independent final pixel replay and
matched comparison remain required for both studies.
Candidate s3 completed1024/1024, curator passed, 398.877 s / 606,023,680 B;
all matching processes gone. Both complete training ledgers SHA256
91c5bb06c1b209477c2c0342ba4c27fd69cbb93ef49a62b9bf8bdda01509ae5c.
Independent control pixel replay planned: `scripts/probe_stream_errors.py
--run m2-stream-headrate-control-s3 --output runs/m2-stream-headrate-control-errors-r1`,
640 MiB / 120 s / 41 forwards, expected below600 MiB, CPU1/cache16 and
pressure guard. Receipt/logs prefix `runs/m2-stream-headrate-control-errors-r1`;
check output, receipts and logs absent first. Candidate follows only after
control process death, with its own analogous output. Separate 128 MiB /
30 s / 41-case collectors and final matched comparison remain mandatory.
Control replay passed 15.742 s / 425,357,312 B under640 MiB. After confirming
its process tree gone, candidate replay uses `scripts/probe_stream_errors.py
--run m2-stream-headrate-candidate-s3 --output
runs/m2-stream-headrate-candidate-errors-r1`, same640 MiB / 120 s / 41 forwards,
expected under600 MiB, CPU1/cache16 and pressure guard; receipt/log prefix
`runs/m2-stream-headrate-candidate-errors-r1`. Preflight all paths.
Candidate pixel replay passed 15.071 s / 424,677,376 B; both process trees
gone. Curators run sequentially `scripts/record_stream_errors.py --name
m2-stream-headrate-{control,candidate}-errors-r1`, 128 MiB / 30 s / 41 cases,
expected under100 MiB, CPU1/pressure guard; receipt/log prefixes
`runs/type-head-{control,candidate}-errors-curation-r1`. Preflight names.
The final comparison remains pending their independent scopes.
Both ordinary replay curators passed (control 78,360,576 B; candidate
77,758,464 B under128 MiB). Final matched comparator planned:
`docs/experiments/verify_type_head_finish.py`, 128 MiB / 30 s / 2130 items
(two1024-row ledgers + 82 pixel cases), expected below100 MiB, CPU1/pressure
guard. Receipt/log prefix `runs/type-head-joint-comparison-r1`; output
`docs/evidence/type-head-joint-comparison-r1/`, preflight all paths.

## Joint type-head optimization preparation — 2026-11-07

The predeclared comparison is `docs/TYPE_HEAD_JOINT_OPTIMIZATION.md`. An opt-in
fourfold type-head optimizer factor is implemented with default single-group
behavior preserved, profile/recovery identity bound, and segment routing guarded.
Focused optimizer/admission/routing checks passed; these do not admit training.

Candidate recovery admission r6 planned: `scripts/probe_stream_recovery.py
--output runs/m2-stream-recovery-gate-r6 --curriculum all-class-fit-v1
--type-head-rate-factor 4`, parent Windows Job 1024 MiB / 600 s / 14 updates,
serialized children each 640 MiB / 90 s, CPU1/cache16 and unchanged OS
pressure guard. Expected aggregate under 800 MiB from prior pilot gate; first
two updates are the full-square warm profile, followed only if admitted by
continuous6 and partial3+resume3. Receipt `runs/m2-stream-recovery-gate-r6-job.json`,
logs same prefix `.log`/`.stderr.log`, progress in the output directory. More
than 40 GB free was checked. This gate does not start the 1,024-update study.
Correction: r6 already existed. The runner refused its existing receipt before
work; redirect opened historical local log names. Failed-attempt copies are
`runs/type-head-recovery-name-collision{,.stderr}.log`; historical local logs
were restored byte-for-byte from `docs/evidence/m2-stream-recovery-gate-r6/`.
See V-029. Use r7 only after checking all output, receipt and log names absent;
same command, limits, output/log/receipt prefix, and work cap as above.
r7 passed 57.769 s / 602,124,288 bytes, exact continuous6 == partial3+resume3,
profile child peak 558,534,656 bytes under 640 MiB; all matching processes
gone. Independent curator planned under 128 MiB / 30 s / 181 replay items,
expected under 100 MiB, same CPU/pressure guards. Receipt/log prefix
`runs/m2-stream-recovery-gate-r7-curation`.
r7 curator passed 2.059 s / 87,764,992 bytes; replayed 17 records and 164
saved probes, evidence `docs/evidence/m2-stream-recovery-gate-r7/`. Control r8
planned using the same gate, ordered curriculum and factor1, parent 1024 MiB /
600 s / 14 updates, serialized child 640 MiB / 90 s, expected under 800 MiB,
CPU1/cache16 and unchanged pressure guard. Preflight output, receipt and both
log paths; more than 40 GB free must remain. This is not the 1,024-update fit.
r8 passed 118.063 s / 600,768,512 bytes; matching processes terminated.
Control curator planned under 128 MiB / 30 s / 181 items, expected under
100 MiB with the same guard, receipt/log prefix
`runs/m2-stream-recovery-gate-r8-curation`. Match comparator planned at
640 MiB / 30 s / two snapshots, expected under 400 MiB, after curation.
r8 curator passed 2.414 s / 88,158,208 bytes; 17 records/164 images replayed.
Matched first-step comparator passed 2.219 s / 256,782,336 bytes; initial
probe metrics and first loss/gradient norm match, non-head first-step weights
match and group rates are .001 versus .004 for the candidate head. Exact
initial tensor bytes were not recorded. The predeclared identical-initial-
weights criterion remains open; add source-bound capture/verification and
repeat affected admissions before launching any 1,024-update study. Receipts:
`docs/evidence/type-head-admission-contracts-r1/`. No learning claim follows.

## Pixel replay work in progress — 2026-11-07

The conditional pixel replay and independent collectors are complete; see
`docs/CONDITIONAL_PIXEL_REPLAY.md`. Both joint-fit gates fail and both development
type scores remain 0/18. The next research action is to predeclare a controlled
joint-optimization comparison, including resource and recovery admission; no
new training is admitted yet. `scripts/pixel_replay_contract.py`
adds a strict byte comparison of every non-type output at each stage and final
output, with only the designated stage type logits exempted. Three focused
checks pass under the bounded CPU runner; receipt and log are in
`docs/evidence/pixel-replay-contract-r2/` (peak 246,312,960 bytes, 640 MiB cap,
30 seconds, work 3). This is a comparison instrument, not a pixel replay or
model-quality result on its own. The original runs remain frozen.

Ordered replay attempt r1 planned: `docs/experiments/conditional_pixel_replay.py
--study ordered --output runs/conditional-pixel-ordered-r1`, supervised by
`scripts/run_bounded.py` at 640 MiB / 120 s / 82 forwards; expected under
600 MiB, CPU1/cache16, Windows Job and existing pressure guard. Receipt
`runs/conditional-pixel-ordered-r1-job.json`, stdout/stderr same prefix with
`.log`/`.stderr.log`. This is an experimental attempt, not an audited result.
Ordered r1 terminated passed at 12.606 s / 426,795,008 bytes; all 82 forwards,
saved feature and original-report checks pass; independent audit remains open.
All startup processes 8856/23544/8412/17196 are gone. Interleaved r1 planned
with the same guard, 640 MiB / 120 s / 82 forwards, expected under 600 MiB:
`docs/experiments/conditional_pixel_replay.py --study interleaved --output
runs/conditional-pixel-interleaved-r1`, receipt and logs under that prefix.
Interleaved r1 passed 11.734 s / 426,926,080 bytes; startup processes
2808/22148/12384/4840 gone. Independent collectors planned sequentially:
`docs/experiments/record_conditional_pixel_replay.py --study STUDY`, 41 cases,
256 MiB / 30 s, expected under 150 MiB, CPU1 and unchanged pressure guard.
Receipts and logs use `runs/conditional-pixel-STUDY-curation-r1` prefix.
Both collectors passed, with raw evidence under `docs/evidence/conditional-pixel-
{ordered,interleaved}-r1/`. All known replay and collector processes exited.

## Owner request and delivery

The owner asked for a clean project, a repository under `sieradni`, a ZIP to send
and a document of the core goals. This is the active delivery task. No further
research training is selected during handoff. No research jobs are running.

- Repository: https://github.com/sieradni/tetris-state-vision (PRIVATE; owner/account verified).
- Core brief: docs/CORE_GOALS.md. Contributor workflow/setup: CONTRIBUTING.md.
- The incoming contributor's next research task is conditional pixel replay below.
- GitHub holds source/history; the ZIP is a snapshot of the same source commit
  plus current ignored model/run artifacts. ZIP extraction alone creates no remote.
- No collaborator was invited because their GitHub username was not supplied.
  The owner must grant access before that person can clone/push a branch.
- Original screenshots, private contact sheets, `.venv`, `.git`, the neighboring
  engine and unselected historical runs are excluded. Original local data remains.
- Bundle contains current image runs, required warm admissions, fitted heads,
  saved synthetic probes, exact frozen sources and one latest optimizer generation
  per selected run. The archive manifest is the exact file/commit inventory.

Local research results were committed at64939f2, then the contributor tooling/docs
at e3312ef. Six transport tests passed:7.743s(native8.231s/120,905,728B under128MiB).
They verify integrity,latest snapshot selection,non-destructive/idempotent restore,
source preservation,conflict preflight and unsafe/duplicate members. Source-bound
raw evidence:docs/evidence/handoff-contracts-r1/. This is not a whole-suite gate.
Actual artifact preflight passed .736s/74,731,520B:925 artifact files/313,244,976B,
19 explicit roots. Counts are preliminary;final archive manifest is authoritative.
Private repository creation passed6.059s/138,231,808B under256MiB/90s. Raw receipts,
logs,startup/death and verified owner/visibility:docs/evidence/handoff-preflight/.

Export verification is intentionally outside the source commit to avoid a
self-referential archive hash. Durable owner-side jobs/results are in ignored
runs/handoff*/ and runs/handoff/. Final ZIP,SHA256 and adjacent verification JSON
are delivered in Downloads. They name the exact exported Git commit. The final
acceptance is local main==origin/main==GitHub main==ZIP source commit,clean source,
verified every archive file,restoration into a fresh GitHub clone,latest optimizer
payload hashes,relocated original-model pixel smoke for all3models and copied-head
vector replay. The pinned existing runtime is reused;no fresh install or full
conditional pixel intervention is claimed by that transport check.

Before each export job: state expected RAM/cap/guard,verify actual processes,
record receipt/log/command and inspect terminal status. No force pushes or
history rewrites. Local main is advanced only by a normal fast-forward merge.
Current one-off publisher:runs/handoff/export_remote.py (no credential output).
Push:expected<300MiB,512MiB/180s/work1,unchangedpressureguard:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 512 --wall-seconds 180
--work-items 1 --receipt runs/handoff-push-r1-job.json --
.venv/Scripts/python.exe runs/handoff/export_remote.py push
Logs sameprefix.log/.stderr.log;result runs/handoff/push-result.json.
Build:expected<200MiB,256MiB/180s/work12000:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 256 --wall-seconds 180
--work-items 12000 --receipt runs/handoff-build-r1-job.json --
.venv/Scripts/python.exe scripts/handoff_bundle.py build --zip ZIP_IN_DOWNLOADS
Logs sameprefix.log/.stderr.log;ZIP plus SHA256 and native receipt are sentinels.
Relocation:expected<800MiB,1024MiB/240s/work12000:
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 1024 --wall-seconds 240
--work-items 12000 --receipt runs/handoff-verify-r1-job.json --
.venv/Scripts/python.exe scripts/verify_handoff_checkout.py --zip ZIP_IN_DOWNLOADS
--checkout runs/handoff-check-COMMIT_PREFIX
Logs sameprefix.log/.stderr.log;adjacent *.verification.json is result sentinel.
Do not launch duplicate export operations because a read/poll timed out.
Exact post-commit export paths/command/PID/log/limits/results are recorded before
launch in ignored runs/handoff/export-state.json. This fixed source checkpoint
points there so the delivered commit need not contain its own later ZIP checksum.
Before final source commit:local verification-script check,expected<800MiB,
1024MiB/60s/work167(3ordinarypixels+2x82headvectors),no training/sourcewrites.
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 1024 --wall-seconds 60
--work-items 167 --receipt runs/handoff-local-smoke-r1-job.json --
.venv/Scripts/python.exe runs/handoff/local_smoke.py
Hiddenlogs sameprefix.log/.stderr.log;native receipt terminal sentinel.
Local verifier smoke PASS12.893s/525,479,936B under1GiB/60s/work167:all3frozen
model origins and original pixel logits exact,both82-row fitted-head replays
exact. Actualsource/rawreceipt/log/startup/death in docs/evidence/handoff-local-smoke-r1/.
This verifies the program locally;full fresh-clone ZIP verification is separate.

## Incoming contributor: selected next experiment

Read AGENTS.md,docs/CORE_GOALS.md,docs/ROADMAP.md,docs/TYPE_HEAD_FIT.md and BUG_LOG.md.
There is no validated complete screenshot parser. The original4096baseline and
both1024controlled image studies remain complete/frozen and fail adequate/joint
learning. API remains needs_review/unsupported. The full release goal stays open.

Implement conditional pixel replay for the ordered and interleaved copied raw
float32 heads. No additional image-model training or vector-fit extension is
selected. Acceptance:

1. Verify curated model/feature/headfit source,config,weights,payload and hashes.
2. Replay all41saved rasters/public targets per model;original ordinary metrics
   and per-image type-head inputs/logits must match independent saved reports.
3. Replace only layer_type weights/bias in a MEMORY COPY. Pixel/validity only
   to model.forward. Exactly2forwards/image=82/model;no oracle geometry/labels.
4. Require every non-type tensor at EVERY stage to remain bitwise equal and
   final head features to remain exact. Record honest float32 matrix-product
   roundoff versus vector-batch logits;actual class counts must match headfit.
5. Restore original head in finally/percase,verify full model state/source/file
   identity. Never overwrite original model/checkpoint/optimizer/data.
6. Independently derive/audit type,pose,absence,locked,geometry and joint-fit
   metrics from the actual changed pixel outputs. Keep the intervention separate
   from original ordinary reports and historical study tags.

Expected inference<600MiB based on measured425MBobservers;enforce640MiB/120s/
82forwards,CPU1/cache16,existing pressure guard. Declare/bound an independent
collector separately. Implementation and those conditional runs are still pending.
Transport smoke performs just1ORIGINAL pixel forward/model;it does not meet this gate.

## Latest verified findings and artifact map

TYPE_HEAD_FIT.md: both finite copied-head fits complete at1024updates with per-bank
native warm/reload admission and independent collectors. Original image models,
source and optimizer files unchanged. New diagnostic optimizer is explicitly reset;
fullbatch64known vectors,AdamWLR.01/decay0,training-onlymean/popstd floor1e-6,
float64 conditioning,converted shared8x64+bias rawfloat32 head. Initial per-image
float32 logits exact;first2fit records match warm prefix;head/Adam reload exact.

- Ordered types20/64->62/64(active32,ghost30):64/64 criterionFAILS. Remaining two
  errors:T->S,both views of all-class-04(one ancestry). Dev0/18;CE3.279->58.590.
- Interleaved types37/64->64/64(active32,ghost32):vector criterionPASSES.
  Dev0/18;CE4.933->25.486. Rawfloat64/float32 counts agree.
- This proves representation on these saved training vectors,not image reading
  or generalization,and does not identify the cause of joint-training failure.

Evidence/runtime names:type-head-{ordered,interleaved}-{profile,fit}-r1.
Profiles:2warm+continuousnext1+reloadednext1(work4),640MiB/60s.
Fits:1024updates,640MiB/120s. Peak333MBboth;native~3s;independent collectors
~.35-.4s/78-80MB under128MiB. All known PIDs/commands gone;raw source/payload/log/
receipt/startup/death scopes in docs/evidence/<name>/. Six pure contracts in
handoff source:docs/evidence/type-head-contracts-r1/. No new defects;V001..V028
closed on named receipts. Do not attach those green scopes to changed sources.

Ordered model:runs/m2-stream-allclass1024-s3;interleaved:runs/m2-stream-interleave1024-s3.
Fitted payloads:runs/type-head-{ordered,interleaved}-fit-r1/head-state.pt.
Raw32values are in result.json/head.weight_raw32,bias_raw32;rederive from the
normalized payload+mean/scale and verify. Curated inputs_sha256 carries model,
feature and source identities;do not hand-rewrite hashes/absolute historical paths.
Payload SHA ordered6749fa5c64c5f6e5dce9fd820c9f1b94854ed2bc14abea931b10fb79545fc696;
interleaved58b417a8c50e434462c5d728c2828b564b33d74c3bdf98391cb7d178e66f86e9.

INTERLEAVED_FIT.md original image studies:ordered32/32boards,12/56visible roles,
8/8absences,geometry.21059;interleaved28/32boards,30/56visible roles,7/8absences,
geometry.38618. BothFAIL joint32boards/64roles/128semantics/geometry<.25.
Devboth2/9boards(the2empty),0/18roles,only3ancestries. One seed,16trainingancestries,
lockedonlyempty/fixed8cellrow. Do not count views as independent accuracy samples.

ACCURACY4096_BASELINE.md original larger7.69Mmodel complete and frozen. Final
ordinary27-image replay/curator pass;dev0/18boards,0/36roles,369/401positivecells,
32misses,813inventedcells,geometry2.01544. All9correct training-probe ghosts are
absence. Run/source/optimizer integrity is not adequate learning. Current model
runs/m2-stream-accuracy4096-s10,original source121fd05a... and weights06751638....

LAYER_FEATURE_DIAGNOSTIC.md actual inputs/logits and unchanged ordinary outputs
verified for all3models. Centroid failure did not prove absent class information;
subsequent copied-head fits above are more informative. Original raw references,
source catalogs,summary/probe files and current model files are in the handoff.
Older unselected tensors/run folders remain local;Git preserves curated evidence.

## Safety and full scope

Windows kernel Job/kill-on-close,CPU1/cache16,GetPerformanceInfo/.2spressure guard
remain armed. Never disable/raise machine ceilings;before detached work record
RAM/wall/work caps and expectedpeak,verify actual startup/termination and keep logs.
Oneheavy job at a time;>40GBfree beforelongtraining. Reference machine31.2GBRAM,
stopcommit31,368,351,744B. NewhostcrashrequiresSystem41/1001anddump audit first.
Do not delete user run data,closeapps,change power or touch neighboring engine/stage.
Historical pressure stops/restarts retain their evidence;no crash was observed
in this handoff. Read RESOURCE_GUARD.md and TRAINING_RECOVERY.md.

The166real screenshots are unreviewed and excluded. Broad rendering/occlusion/
association,reviewedreal/independentsplits,adequateM2/M3jointlearning,scratch/
pretrained/directdecoder comparison,calibration/export/CPU/adapters and>=95%
readable-real exactcomplete-state remainOPEN. See CORE_GOALS.md for acceptance.
No claim that the research goal or release gate is complete.
