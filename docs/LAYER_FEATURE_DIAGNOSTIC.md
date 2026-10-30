# Read-only piece-type feature diagnostic

The completed controlled studies fit pose more readily than type. Before another
architecture or training change, inspect the actual vectors entering the existing
type head. This is a diagnostic of saved artifacts, with no training, replacement
decoder, oracle geometry, model-output changes or real-image accuracy claim.

Execution waits for the original larger baseline's fixed4,096 updates to finish.
Then independently replay its ordinary errors using the existing worker and
curator. Keep that result separate from these extra observations. Inspect the
completed ordered and interleaved1,024 models with the same diagnostic afterward.
They differ in capacity/population from the larger baseline; do not rank those
three models as a controlled architecture comparison.

`docs/experiments/probe_layer_features.py` calls the existing ordinary saved-probe
worker. A Torch observer returns None on every call. It records actual Linear
inputs/outputs, verifies the module is exactly the frozen model's `layer_type`,
checks every stage and reconstructs its logits exactly. The existing worker must
reproduce all saved ordinary metrics, exact pixels/public annotations, frozen
source/configuration/runtime and unchanged full weights. No target enters forward.
Record all final role features, logits and the actual head weights/bias for audit.
Capture execution sources, native limits, raw logs and exact process identities.

Analysis has one fixed descriptive rule: Euclidean nearest class centroid,
separately for active and ghost, using only known training-probe labels and
excluding **every view of the query ancestry**. Development never supplies a
reference. Missing target classes make a query ineligible; exact ties abstain.
Report denominators, unknown exclusions, class coverage and all distances, alongside
the original native type head's correctness. No fitting iterations or new model
artifact are created. Training views were used to train the network, so even
ancestry-excluded readout is not an independent held-out representation test.

A higher centroid count suggests that some class signal is available to this
particular distance rule; a lower count does not prove an information bottleneck.
Neither result establishes a causal repair. Do not select a new mechanism from
an unexecuted feature probe or treat this readout as an inference dependency.

This instrument changes no architecture/labels/scheduler/wire format. Relevant
prior defects: V-003/V-014/V-018/V-020..V-024/V-026/V-027 require computed artifact
and source identities, original scope and actual raw bytes; reuse ordinary
replay and bind the observer/helper sources separately. V-004/V-011/V-012/V-017/
V-019/V-023/V-025 require meaningful validated instruments: test sibling exclusion,
unknown invariance, class support, ties and split/role separation, then prove the
actual observer against unchanged neural outputs. V-001/V-002/V-005/V-006/V-016
visibility policy is unchanged; unknown labels remain excluded. Resource controls
stay unchanged. Pure checks may coexist with training under128MiB/30s/6cases;
neural observation is serialized later, at1GiB/120s/27forwards for the larger
model or640MiB/120s/41forwards for either controlled pilot, CPU1/cache16.

Six pure adversarial checks PASS in0.017s(native.234s/32,813,056B under128MiB/30s),
docs/evidence/layer-feature-analysis-contracts/scope.json. Actual source/test and
raw receipt/log/startup bytes are preserved;all known test PIDs/commands gone.
That gate covers analysis math/grouping only;the separate actual neural and
collector passes below provide their own evidence,not a quality claim.
Collector review subsequently exposed integer-to-string class-count keys across
JSON serialization (V-028). The red sixcase r2 reproduces the mismatch; canonical
string keys and exact roundtrip regression pass fresh r3 (six/six,.017s,native
.235s/32,563,200B under128MiB/30s). Raw before/after source and native evidence:
docs/evidence/v028-contracts/. The earlier sixcase gate retains its earlier source
scope; this separate regression closes reporting integrity, not neural replay.

`docs/experiments/record_layer_features.py` is the independent collector. It
requires a separate previously curated ordinary
error report for the same model. Observer cases/totals must equal that report
exactly; all feature rows and grouped analysis are rechecked, along with native
command/cap/count, actual weights/configuration and execution-source byte copies.
Run it under128MiB/30s/work27 or41, after the corresponding observer terminates.
Preserve collector receipts/startup/termination separately as well. Fixed output
names are `m2-stream-{accuracy4096,allclass1024,interleave1024}-features-r1`.

## Larger baseline: native observation and independent collector pass

Evidence: docs/evidence/m2-stream-accuracy4096-features-r1/. The27-forward observer
passes13.768s/590,053,376B under1GiB/120s. Its input vectors reconstruct every
stage's actual type logits exactly;full weights are unchanged. Independent
collector passes.355s/78,213,120B under128MiB/30s. Its ordinary cases and totals
equal the separately curated final ordinary report, and the complete grouped
analysis is recomputed after JSON readback. Execution sources and raw native/
curator/startup bytes are preserved;all known PIDs/matching commands gone.

| Role/population | Known | Eligible ancestry-excluded queries | Correct centroid readout |
|---|---:|---:|---:|
| Active training probes |9|0|0/0(no reference ancestry of same class)|
| Active development |18|3|0/3|
| Ghost training probes |9|9|9/9(all absence)|
| Ghost development |18|0|0/0(no visible class in training references)|

The larger baseline's saved probes cannot establish general type-feature
separation. Most queries lack another ancestry of the target class;ghost
reference labels are absence only. The sole eligible active development class
is J,three matched views. Do not describe an ineligible query as a readout
failure or compare these denominators to the balanced controlled studies.
The two completed controlled models below have training probes covering all eight
classes with two ancestry groups per class. Their actual observer/collector gates
are separately named;the larger-model pass does not substitute for them.

## Completed controlled-model observations

Both actual native observers and independent collectors PASS. Every stage's
captured input reconstructs its actual logits exactly;weights remain unchanged,
and ordinary cases/totals equal the independently curated original reports.
All64 training role queries and18 development queries are eligible in each
study;there are no centroid ties. Counts below are type classification only,
including absence,not exact complete-state or pose scores.

| Model/role | Training native head | Training ancestry-excluded centroid | Development native head | Development centroid |
|---|---:|---:|---:|---:|
| Ordered active |10/32|4/32|0/9|0/9|
| Ordered ghost |10/32|2/32|0/9|0/9|
| Interleaved active |18/32|1/32|0/9|0/9|
| Interleaved ghost |19/32|6/32|0/9|3/9|

These distance-based readouts do not expose an obvious readout-only replacement.
They do **not** prove that the features lack class information or that a different
linear head cannot fit them. Geometry/pose information may dominate Euclidean
distance;this diagnostic does not establish that causal explanation either.
The3/9 development ghost results are only three matched views in a tiny probe
population,not independent generalization or a model-output improvement.

| Operation | Native observer | Independent collector | Evidence |
|---|---|---|---|
| Ordered41forwards |7.819s/424,955,904B under640MiB|.355s/77,811,712B under128MiB|docs/evidence/m2-stream-allclass1024-features-r1/|
| Interleaved41forwards |7.608s/425,201,664B under640MiB|.343s/78,815,232B under128MiB|docs/evidence/m2-stream-interleave1024-features-r1/|

All known processes and matching commands ended;actual four-command observer
startup and curator root startup,raw native/curator logs/receipts,execution sources
and analysis readback are preserved. Wall limits120s/30s and work41 remained armed.
Both original studies remain frozen at1,024 updates and still FAIL joint fit.

The subsequent [fixed linear-head fits](TYPE_HEAD_FIT.md) are now complete and
independently verified:ordered20/64->62/64,interleaved37/64->64/64 training types,
both development0/18. Initial per-image float32 logits and warm/reloaded updates
reproduce exactly;original artifacts stay unchanged. The interleaved vectors can
represent all64 labels with a shared raw float32 affine head. This does not
identify the cause of joint-training failure or improve generalization. Next
verify conditional pixel replay before any image-model fit claim;the broad
project goal and whole-state learning gates remain open.
