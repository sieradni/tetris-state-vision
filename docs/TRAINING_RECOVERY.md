# Exact CPU training recovery

`scripts/training_snapshot.py` is a separate, not-yet-integrated primitive.
The streaming driver still saves its inference checkpoint only at the end.
Seven runtime fixtures are written; their first native harness parked before
launch because Windows could not reserve512MiB. No recovery gate is green yet.

The proposed recovery boundary contains model tensors, complete AdamW state,
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
