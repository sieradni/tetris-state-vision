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

The pure grouping and accumulation instrument is preparation only; integration,
fresh native admissions and the studies remain open.
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
needs its own explicit profile handling. Neither the trainer nor checkpoint
publisher uses this boundary yet. Fresh neural integration, pressure-safe
testing, full-square admission and real interruption recovery remain required.

## Durable boundary bridge and integrated arm (no native run)

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
`curve.jsonl`, `training.jsonl`, progress and summary. Two scoped checks cover the
image/update accounting, the evaluation and snapshot cadence, and a forced
in-group interruption that is discarded and replayed to byte-identical
model/Adam/RNG/journal bytes; they pass with 26 accumulation and 43 stream
regression tests under 640 MiB
([accumulation driver](evidence/accumulation-driver-integration-r1/scope.json)).
Those checks use a toy network, not the pilot model. No accumulation arm has run
natively: there is no per-group-size resource admission, no controller or
independent-curator routing for these cursors, and accumulated resume is refused
until the boundary restore is wired into the driver. So no arm of the comparison
is admitted. Bounded children must use an absolute or backslash interpreter path;
see V-030 in [BUG_LOG.md](BUG_LOG.md).
