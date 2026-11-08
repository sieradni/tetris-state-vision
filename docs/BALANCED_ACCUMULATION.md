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
