# Exact CPU training recovery

`scripts/training_snapshot.py` is the verified primitive. Its streaming-driver
integration now passes the native2+6+3+3 equivalence gate.
Periodic snapshots, explicit global-step resume and a finite new-update cap
are verified on that finite gate; long diverse-shape memory remains a separate
open requirement (V-013).
Seven runtime fixtures now pass: runs/training-snapshot-tests-r3.stderr.log,
7/7 in4.103s,332,562,432B raw peak under512MiB/45s. Earlier attempts parked
on host pressure; r2 hit an OpenBLAS import allocation failure before tests.
V-012's pre-import thread policy repaired that harness at the same ceiling.
Committed-source scope and raw logs: `docs/evidence/v012/scope.json` (7cbc68c).

Optional M3 heads now pass the same native continuous6 versus3+resume3 pilot
gate: all model/Adam tensors, data/losses/gradient norms, evaluations, CPU RNG
and module modes match exactly. Sourceb6bb05f3..., capacity64. Named evidence
docs/evidence/m3-stream-recovery-gate-r1/scope.json/proof.json. The actual larger
configuration separately passes its two full-square warm/reload profile in
docs/evidence/m3-stream-large-profile-r1/scope.json. Neither finite check proves
long varied-shape stability or complete-state learning.

The verified primitive's recovery boundary contains model tensors, complete AdamW state,
Torch CPU randomness, module training modes and the completed global step.
The caller supplies the exact source catalogue and data/training recipe as a
JSON contract. Recovery requires equal recipe, Torch version, tensor shapes/
dtypes and ordered optimizer parameter groups. Data must be a deterministic
function of global step and recipe. Python/NumPy/GPU randomness is outside this
primitive's scope; the current synthetic stream uses explicit local seeds.

Each generation is new and retained. Tensor payload and manifest are flushed,
synced and hashed before atomic replacement of `latest-training.json`, which is
the final publication point. A process interrupted earlier leaves the preceding
complete generation selected. This is a process-interruption contract, not a
claim about storage hardware durability after power loss. No generations or run
data are automatically removed.

Up to three local ledgers are bound by exact byte-prefix lengths and SHA256.
Callers must flush their writers first. Recovery returns only those committed
prefixes; later uncheckpointed appends stay in the source run but are excluded
from recovery. Changed or truncated prefixes fail. A fresh output directory and
explicit global-step scheduling are the future driver's responsibility.

Budgets:1..4096 steps,64KiB JSON,16MiB per ledger,256MiB tensor artifact (also
checked conservatively before serialization), CPU AdamW only. Loading uses
`weights_only=True` under the native Job ceiling. Artifact hashes are checked
before tensor loading or model mutation. File identities come from local
receipts; hashes do not authenticate arbitrary third-party downloads.

Acceptance: continuous six-step AdamW/dropout training and three steps plus
recovery plus three steps must give identical model tensors, optimizer state
and next Torch randomness. Other fixtures inject failed pointer publication,
corrupt tensors/manifest, changed recipe/parameter order and modified/truncated
ledger prefixes, and check finite budgets. Driver integration then needs its
own exact data-prefix equivalence and a new warm resource characterization;
the existing frozen streaming admission does not cover that future change.

The driver candidate writes a snapshot every64updates and at a controlled exit.
`--resume-from RUN` restores the committed global step and three ledger prefixes
into a fresh output; `--max-new-steps N` creates explicit `step_partial` status
when a finite segment ends before the global `--steps` target. Wall exits keep
`wall_partial`. Source/recipe/runtime mismatch still refuses recovery. Snapshot
publication during a native kill leaves the previous pointer selected.

Version3 resource characterization includes actual snapshot save/load parity and
a second full-square warm backward. Source execution itself is bound to the
archive, fixing V-014's misleading parent-file traceback lines. The CPU worker
requests a16-entry oneDNN cache before import; controlled cache-hit/miss logs
verify eviction behavior. V-013's finite768-update diverse-shape gate now passes
at640MiB; broader untested shape/run behavior remains uncharacterized.

Native gate: runs/m2-stream-recovery-gate-r2/proof.json, source catalogue
121fd05a2f75e30b0c738dac592c67c6d015aa6043a43e4032d91ac662de794b.
Six continuous updates and3+resume3 match data, loss terms, gradients, evaluation
results, every model/optimizer tensor and CPU RNG exactly. Inference weight SHA
f5145621be3ea9650e5126250d80baa568aec1e7748f66e9f2dd5834e5730f51 is identical
for both. Partial status remains explicit. Child max554,541,056B under640MiB;
outer590,663,680B under1GiB. Each native child process was verified dead.

The larger model now demonstrates a real wall-limited continuation:93/128 saved
at the300s inner limit, then35 updates from that exact Adam/RNG/data cursor.
The completed artifact retains its parent's `wall_partial` receipt and all three
ledger-prefix identities. Curated source/snapshot/tensor/prefix verification:
docs/evidence/m2-stream-accuracy128-r2/scope.json. This does not relabel the first
Job as complete or claim that the second Job performed128updates.

An actual OS-pressure interruption also saved recovery step1 before final
evaluation was killed (native exit125). A separate512MiB restore loaded the
larger model,238 Adam parameter states, CPU RNG and committed ledgers without
another forward/update. It passed at467,935,232B raw peak. The interrupted run
has no final summary; this verifies saved-state recovery, not successful segment
completion. Evidence: docs/evidence/segment-pressure-recovery/scope.json.

A later4096-target segment was killed by its ancestor Job at OS commit pressure:
229 updates logged but only192 banked. The child receipt remained raw running/
null; the terminating ancestor receipt125 is retained separately. Resume from192
recomputed37 later updates with all229 recorded data/loss/terms/gradient norms
equal (time excluded), and all three committed prefixes unchanged. Evidence:
docs/evidence/m2-stream-accuracy4096-pressure-replay/scope.json. No old229 tensor
artifact survived; this proves the recorded trajectory, not full final-model
tensor identity or target completion. All original/uncheckpointed data remain.

`record_stream_run.py --allow-partial` now also preserves controlled `wall_partial`
and `step_partial` artifacts explicitly. Default completion remains strict;
outer timeout, pressure kill, crash and missing checkpoint/reload parity refuse.
Three boundary fixtures and actual93-step partial replay pass. Segmented
longer targets can therefore publish evidence without a completion claim.
