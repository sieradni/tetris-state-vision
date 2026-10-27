# Exact masked-target exposure audit

The saved1664 model predicts one active-piece class across all18 development
probes. This audit asks which labels its training stream actually supplied.
It does not infer labels from latent scene fields or treat a visible source
annotation as proof of supervision after resizing.

`probe_stream_exposure.py` verifies a saved ordinary M2 artifact and its original
source archive,then replays each recorded recipe through that archive's
`sample_case`, `prepare_image(max_side=512)` and `targets_for`. Pixel/state/recipe
identity and the trainer's200-known-cell admission must reproduce. It imports
Torch for those exact target operations but constructs no model or optimizer;
it performs zero forwards/updates and verifies unchanged inference weight bytes.
Run it serially after any training/model Job exits.

Each active/ghost target is packed mask-first. Unknown fields become null;
their padded zero types and coordinates never count as absence or a pose.
Known absence is a supervised type0. Known piece identities and known poses
are counted separately. Equivalent rotations retain all allowed values rather
than selecting an arbitrary rotation. The pure bookkeeping and collector checks
use logical fixtures without importing Torch.

Reports distinguish update exposure from distinct ancestry-group type exposure.
Repeated views increase update counts;each group/type contributes once to the
group table. A group can contribute to multiple supported type categories,so
those group counts are not assumed to be mutually exclusive. Separate fixed
training/development probe populations expose any probe-distribution mismatch.
Class counts are diagnostics,not independent observations or a causal explanation
for a failed model.

For saved1664,the finite workload is1691 target operations:1664 recorded updates,
9 fixed training probes and18 development probes. Expected peak below550MiB;
enforce640MiB/120s/work1691 with the native Job,OS pressure and one-thread policy.
Use a fresh output,receipt,hidden logs and checkpoint/PID entry:

```text
.venv/Scripts/python.exe scripts/probe_stream_exposure.py --run m2-stream-accuracy4096-s4 --output runs/m2-stream-accuracy1664-exposure-r1
```

The collector `record_stream_exposure.py --name <output-name>` uses a separate
128MiB/30s/work1691 envelope (expected below96MiB). It verifies actual source,
artifact/config/partial status,runtime,full command/work/guard identities,all
source recipe associations and independently recomputed packed-target counts.
It preserves raw target/input ledgers,summary,source/weight manifests and native
receipts with hashes. Only an actual native audit establishes exposure facts;
unit fixtures alone cannot establish training data quality or model accuracy.

Fourteen pure accounting/protocol checks pass in.040s,35,467,264B under128MiB/30s:
`docs/evidence/target-exposure-contracts/scope.json` pins12 source files on
e454345. Its logical metadata fixtures do not claim actual native target values.

Actual1691-target replay passes in33.834s,361,013,248B under640MiB;independent
curation passes in.535s,90,492,928B under128MiB. Source/config/weights and original
step_partial1664/4096 are bound in
`docs/evidence/m2-stream-accuracy1664-exposure-r1/scope.json`. No model forwards
or updates occurred,and inference weights remained unchanged.

| Supervised layer type | Active updates | Ghost updates |
|---|---:|---:|
| Absent |288|618|
| I |186|138|
| O |200|146|
| T |183|144|
| S |195|168|
| Z |186|138|
| J |207|150|
| L |219|162|

All1664 updates have known layer targets:1376 observed active and1046 observed
ghost targets across555 ancestry groups. Every observed class received
supervision;the active head's fixed-development collapse is not explained by a
missing target class in this prefix. Counts alone do not establish the cause.

All1046 observed training ghosts have bottom-first row10. Active rows are14,15
or16 (470/456/450 updates). The fixed training probes contain only O/J observed
active pieces,three absent active targets,and nine absent ghosts. Development
contains T/S/J/L,with no absence,I,O or Z;its18 views represent six ancestry
groups. These limits constrain interpretation of both the original probe scores
and any future comparison. They do not change the original run's recipe or
qualify the renderer as broadly representative.
