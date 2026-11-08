# Matched class-balanced gradient accumulation experiment

## Question and predeclared comparisons

The checkpoint diagnostic found recurring changes in which fixed training
roles are classified correctly. Its local head-gradient cosines do not prove
the cause. Test whether averaging gradients across a class-balanced group of
eight supported images improves the fixed joint fit at the same image exposure.
This is a memorization diagnostic, not a release model or real-image test.

Use the existing 32 all-class public-observer fixtures, pilot M2, seed
20261026, no optional heads, AdamW LR .001/decay .01, CPU1/cache16, the
existing joint loss and global norm clipping at 1. Each arm consumes exactly
1,024 images: 32 complete cycles of the same 32 image/target pairs. Compare:

1. **Online interleaved:** the existing interleaved order, one image per update,
   1,024 optimizer updates. It is an exposure-matched reference, not an
   optimizer-update-matched comparison.
2. **Balanced accumulated:** the identical interleaved order, eight successive
   images per update, average their full joint-loss gradients, clip once, then
   step AdamW once: 128 optimizer updates. Each group contains one image of
   each of the seven piece classes and known absence.
3. **Ordered accumulated:** the original ordered sequence grouped in eights,
   otherwise identical to arm 2: 128 optimizer updates. Each group covers only
   two classes, while each 32-image cycle contains the same 32 pairs as arm 2.

The accumulated arms isolate grouping/order at equal example and update
budgets. Online versus balanced accumulated also changes update count,
clipping frequency and effective AdamW decay; it cannot isolate batching from
those effects. There is no claim of equivalent optimization or effective
learning rate. No extra tuning, early stopping or extension of finished studies
is selected. Evaluation cadence is every 128 **images**, checkpoints every
64 **images**, with boundaries restricted to completed optimizer updates.
Adam, RNG, model, exact committed data-prefix and next global image/update
cursor must recover exactly; an interruption within a group discards the
uncommitted partial group and replays from the preceding committed boundary.

Predeclared fit: 32/32 locked boards, 64/64 supported active/ghost type-and-pose
roles including eight absences, 128/128 supported semantic cells, and mean
geometry error below .25 cells. Report role/class/absence, pose, false positive,
miss, geometry and development metrics for all three arms. The nine development
views represent only three ancestries and are descriptive, not a tuning set.
Passing this fixed fit does not establish generalization or real accuracy.

## Admission before training

Integrate a versioned policy without changing default streaming semantics.
Record image count and optimizer updates separately in every ledger, snapshot,
summary, controller and independent curator. Enforce a finite microbatch
count and only checkpoint after the full group and optimizer step. Verify
public supported labels, all-class group coverage, exact same ordered input
sequence for arms 1/2, same multiset per 32-image cycle for arms 2/3, and
loss/gradient accumulation equivalence on an independently constructed small
model. Unknown targets stay masked and cannot be filled from latent labels.

Before any study, require fresh source-bound full-square warm/reload profiles
for each actual group size and a continuous-versus-interrupted/recovered
trajectory including group boundaries and a forced interruption inside a group.
Characterize peak memory with all gradients and Adam state resident, count
all forward/backward microsteps in the native work budget, and retain the
pre-import one-thread policy and Windows Job/pressure guards. Verify identical
actual initial tensors for all arms and compare data ledgers independently.
Use one heavy job at a time and check more than 40 GB free before long training
windows. Only then run finite, separately bounded segments and independent
final ordinary pixel replays. Existing completed studies and their frozen
artifacts are never extended or relabeled.

The pure grouping and accumulation instrument was preparation only at that
milestone; integration, fresh native admissions and the studies were still open
then, and the sections below record them landing.
The integrated driver must also verify that its optimizer owns every model
parameter exactly once before the group executes.

Seven focused checks now pass: all 1,024 image cursors and 32 cycles are
covered, the two accumulated policies have the same per-cycle population,
balanced groups match the actual curated public active/ghost labels, streamed
gradients match a small full-batch reference, and partial-group/failure paths
cannot step through the guarded primitive. A toy AdamW snapshot test verifies
discarding and replaying a partial group reaches the same state. Evidence and
the exact source/receipt are in [accumulation contracts](evidence/accumulation-contracts-r1/scope.json):
5.806 seconds, 332,820,480 bytes under 640 MiB / 30 s / work7.
This is not an integrated trainer, native interruption gate or training admission.
An additional ownership-check run was refused or stopped by the system pressure
guard (r4/r5 receipts retained); that unverified edit was reverted and the
ownership check remains an integration requirement, not a passed test.

The subsequent [journal contracts](evidence/accumulation-journal-contracts-r1/scope.json)
add a pure boundary for exact public recipe/image cursors, separate image and
optimizer counts, finite complete-group prefixes and snapshot resume cursor
refusal. They check optimizer ownership before invoking the grouped update
primitive, and return a journal entry only after a complete outcome. Eight
small, no-Torch checks pass, including malformed/partial records, mutated
recipes, missing/duplicate optimizer parameters and mocked execution failures;
the native receipt records 1.565 seconds and 41,299,968 bytes under 128 MiB /
30 s / work8. These tests do not execute the real update through this wrapper.
The journal accepts ordinary fixed-fit recipes only; full-square characterization
needed its own explicit profile handling. At that milestone neither the trainer
nor the checkpoint publisher used this boundary. Fresh neural integration,
pressure-safe testing, full-square admission and real interruption recovery were
still required there; each has since landed in the sections below.

## Durable boundary bridge and integrated arms (measured)

[`scripts/accumulation_checkpoint.py`](../scripts/accumulation_checkpoint.py)
binds the journal to the existing snapshot tooling: it creates a journal only in a
fresh owned directory, appends only a validated whole group after its successful
optimizer step, refuses to publish a snapshot with pending gradients or with an
optimizer whose AdamW step or hyperparameters disagree with the committed cursor,
and restores into a new directory that copies only the snapshot's committed
journal prefix and excludes every uncommitted suffix record. Restore verifies the
pointer, manifest digest, contract, journal set and prefix bytes before any model
state is loaded, so a corrupt or drifted artifact refuses before mutation.

Eight no-Torch contract checks pass in 0.522 s / 37,761,024 B under 128 MiB /30 s /
work8 (`runs/accumulation-checkpoint-pure-r6-job.json`, evidence
[accumulation checkpoint contracts](evidence/accumulation-checkpoint-contracts-r1/scope.json)).
They cover exclusive journal creation, non-contiguous group refusal that leaves
the journal unchanged, uncommitted-suffix exclusion, contract/prefix corruption,
optimizer ownership, wrong Adam step and hyperparameters, and pending gradients.
The Toy Torch test in `tests/test_accumulation_checkpoint_torch.py` also passes
at 640 MiB / 60 s / work1 after an earlier admission refusal: real AdamW with
dropout, a forced interruption inside group2, restore into a fresh directory and
replay to byte-identical model, Adam, RNG and journal bytes (1.422 s test,
332,423,168 B peak).

The trainer now calls this boundary for every whole group: `--accumulation-policy`
in `scripts/train_m2_stream.py` runs the predeclared arm through
`run_accumulation_loop`, which validates and journals each group, clears the
stepped gradients, evaluates only at image boundaries, and snapshots only at a
complete-group boundary. Image and optimizer cursors stay separate in
`curve.jsonl`, `training.jsonl`, progress and summary. Five scoped driver checks
cover the image/update accounting, the evaluation and snapshot cadence, the
per-group-size admission rules, the profile CLI coupling, and a forced in-group
interruption that is discarded and replayed to byte-identical
model/Adam/RNG/journal bytes; they pass with 29 accumulation, 43 stream and 5
type-head regression tests under 448 MiB and below
([accumulation driver](evidence/accumulation-driver-integration-r1/scope.json)).
Those checks use a toy network, not the pilot model. The pilot model's
`--profile-square --accumulation-policy` path then ran natively for both
predeclared group sizes under the unchanged supervisor at 640 MiB, measuring warm
Adam state and an actual boundary reload with byte-exact model/Adam/RNG parity
([accumulation profiles](evidence/accumulation-profile-r1/scope.json)):

| Group | Wall | Updates/s | Peak commit | Reload | Reload delta |
|---|---|---|---|---|---|
| 1 (online interleaved) | 33.30 s | 0.060 | 519,929,856 B | 0.212 s | 0 B |
| 8 (balanced eight) | 29.35 s | 0.068 | 561,786,880 B | 0.086 s | 0 B |

Both are 142 parameter tensors with 5,177,352 B of resident Adam state at
262,144 input pixels; the second run's journal holds two groups of eight images
and 16 image rows, matching `images_per_update`. The zero reload delta means the
warm peak already exceeded the reload allocation, not that reload is free. Each
arm is accepted by `validate_admission` only when its profile names the same
policy and group size, counts `2*group_size` microsteps in the native work
budget, and carries a positive measured update rate.

Both predeclared arms then ran natively to their full 1,024-image budgets across
two bounded segments each, every segment admitted from its own measured two-update
profile at the same 640 MiB ceiling
([segmented arms](evidence/accumulation-arm-r1/scope.json)). Every figure below is
re-derived from the committed summaries, supervisor receipts, journals and
boundary manifests by the folder's
[audit instrument](evidence/accumulation-arm-r1/audit/audit_arm_evidence.py), whose
[result](evidence/accumulation-arm-r1/audit/audit.json) fails if any published
number, table cell or scope field disagrees with those bytes:

| Arm | Segment | Updates | Images | Wall | Updates/s | Peak commit | Peak Δ vs seg 1 |
|---|---|---|---|---|---|---|---|
| balanced-eight | 1 | 64 | 512 | 175.3 s | 0.365 | 565,817,344 B | - |
| balanced-eight | 2 | 64 | 512 | 179.8 s | 0.356 | 567,971,840 B | +2,154,496 B |
| online interleaved | 1 | 512 | 512 | 216.1 s | 2.369 | 565,100,544 B | - |
| online interleaved | 2 | 512 | 512 | 289.6 s | 1.768 | 569,487,360 B | +4,386,816 B |

`Wall` is the trainer's own `elapsed_seconds`; the supervisor receipts for the same
jobs measure 175.944, 180.456, 216.863 and 290.393 s, each 0.649-0.832 s longer for
process start and teardown, which is why an earlier poll quoting the receipts
printed 175.9 s and 290.4 s where this table prints 175.3 s and 289.6 s. `Updates/s`
is the segment's own updates over its own wall time and its `Images` are the 512
fresh images it consumed, not the arm total; the trainer summary field of the same
name is cumulative over the arm (0.711981 and 3.536386 for the second segments).
`Peak Δ vs seg 1` is the arithmetic difference between the two supervised peaks
(567,971,840 - 565,817,344 and 569,487,360 - 565,100,544), not the cost of
restoring a boundary, which is measured separately below.

Each peak sits below the 671,088,640 B Job commit limit that
`scripts/run_bounded.py` sets, and that limit terminates the tree on reaching it,
so the guard requires strict inequality and no further fractional margin: the
tightest is 101,601,280 B (15.14 %) under the ceiling on online segment 2, the
widest 105,988,096 B (15.79 %) on online segment 1, and the two admission profiles
also ran under it at 563,216,384 B and 527,622,144 B.

Segment 2 restores the committed boundary with `restore_boundary`: the first
segment's journal is an exact byte prefix of the resumed one (512 images each),
the resume pointer equals the first segment's published boundary, and the resumed
cursors agree (64 and 512 updates). The prefix check compares bytes: the first
539,646 B of the 1,079,495 B resumed journal hash to
c9bd56f0e4d6ce0969315462e9d2c68c38153d248f1faa37dad988962d5e099a, equal to the
whole 539,646 B first journal, and the first 595,337 B of the 1,191,099 B resumed
journal hash to
4e0511b888a820334b22b5c8457c3d2cb616b6dd9160415841307fa027428a75, equal to the
whole 595,337 B online prefix. An independent bounded restore of that boundary
costs 0.219 s (eight-image arm) and 0.200 s (online arm). Both final segments
report `complete` at 128/128 and 1,024/1,024 updates. A bounded partial segment
reports `status: step_partial` and exits 2 by the trainer's existing convention,
so its supervisor receipt reads `failed` while the summary carries the partial
status.

The earlier 0.068-updates/s figure is superseded rather than contradicted: two
updates over 29.35 s is a warm square profile dominated by fixed model, data and
optimizer startup, and dividing the budget by the printed rate gives 15,059 s
(17,067 s for the printed 0.060), which these runs falsify at 355.1 s
(175.3 + 179.8) and 505.7 s (216.1 + 289.6) of measured, startup-inclusive wall
across two bounded segments each, about 42x and 34x below the estimate. The
marginal rates above account for the difference: 2/0.365 = 5.48 s and
2/2.369 = 0.84 s of training per two updates, so startup is 29.35 - 5.48 = 23.87 s
and 33.30 - 0.84 = 32.46 s of those profile windows. No predeclared arm remains
impossible to run; the two segments exist because the study requires separately
bounded windows with a resume contract verified at arm scale, not because a
1,024-image budget is unreachable.

Still open: automated controller and independent-curator routing for these
cursors (segments were launched explicitly with `--resume-from`), an independent
ordinary pixel replay of the final artifacts, and an interruption *inside* a
group at arm scale. Bounded children must use an absolute or backslash interpreter
path; see V-030 in [BUG_LOG.md](BUG_LOG.md).
