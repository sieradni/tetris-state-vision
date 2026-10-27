# Queue/hold target contract

`component_targets.py` is a pure projection of validated schema 1.1 observation
fields for the optional joint decoder. Eight original fixtures passed in .013 s under a 128 MiB /45 s Job,
including paired identical-raster queue/hold uncertainty. Named logs and source
hashes: docs/evidence/m3-component-targets/scope.json. This proves the target
contract, not complete M3 modeling or accuracy.

Queue token IDs represent seven piece types and EOS; -1 is masked uncertainty.
There are at most 64 visible item positions plus one extra EOS position.
Unknown identity does not erase the slot or shift later indices.

| Observed anchors | Indexed targets |
|---|---|
| Start unseen | All absolute slot/token/presence targets masked; count unknown |
| Start seen, end unseen | Supported visible prefix only; unseen tail masked; no EOS/count |
| Both seen | Supported prefix plus EOS at its observed length; count known |
| Panel absent | No EOS or known-empty count; separate absence status |
| Visible empty panel | EOS at zero with known count zero |

Presence and slot visibility are independent of piece identity. A middle slot
may be known present with unknown/cropped identity. Padding after an observed
endpoint is not trained as the game's hidden future. A 64-item panel can put EOS
at position64; an undersized configured capacity refuses an aligned prefix
instead of dropping known items. Unseen-start spans do not acquire an absolute
count merely because the schema retains their visible relative order.

Hold has independent observed/empty/absent/cropped/unknown status and conditional
piece identity. Unknown type is null, never the first piece class. Identified
model outputs are refused as targets; independent corpus provenance, permission
and review checks remain the training boundary's responsibility.

The paired-raster fixture changes wholly hidden queue length/types and hold
identity while keeping the crop byte-identical. Targets must remain equal, with
no EOS/count or hidden identities. It extends the standing V-001/V-002/V-005/
V-006 visibility instrument. Other fixtures cover unknown middle slots, missing
starts/tails, absence versus visible-empty, all hold statuses, capacity64 and
model-output refusal.

`components.py` now supplies optional shared role/index queries and masked joint
losses. `ModelConfig.component_capacity=0` retains M2;1..64 adds M3 components
at every board decoder stage. Each index uses the same classifier/attention
parameters, with a fixed sinusoidal index code and chunks of at most16 queries.
The capacity includes64 item slots plus one extra EOS position. Queue role and
hold role attend the board/cell/spatial context; component gradients reach the
shared image encoder. Parameter sharing permits additional indices, but does
not establish queue-length extrapolation accuracy.

Queue heads predict token, slot presence, slot status, panel status and observed
start/end independently. Hold has separate status/type heads. Semantic losses
mask unknown identities; presence does not disappear when identity is unknown.
No count loss uses an unseen endpoint. Loss coefficients are untuned hypotheses.

The initial15-case native gate passes,357,052,416B raw peak under512MiB/60s,
1.774s tests/4.938s outer. It checks hidden-label gradients, capacity64, paired
hidden rasters, board-weight compatibility, finite joint/encoder gradients and
legacy/M3 checkpoint round trips. M3 has explicit checkpoint architecture
identity. This is numerical/contract evidence, not trained component accuracy.
Committed-source scope: docs/evidence/m3-joint-contracts/scope.json onfdb0c81,
15 new cases plus14 M2 numerical,7 inspection and8 observation-target cases.
These44 checks are scoped contracts, not a whole-suite or accuracy result.

V-016 input admission: source Icon fields do not contain support geometry for
model resizing/invalid alpha. M3 `targets_for` therefore temporarily requires
unchanged, fully valid source pixels (padding is allowed). It refuses altered
evidence instead of guessing which identities survived. A resized component
observer and its paired-raster gate remain OPEN before broad M3 training.

For synthetic data, `model_capture.render_for_model` now selects a bounded final
capture size and reruns the existing observer at that exact transform. It records
the actual rounded x/y dimensions in the effective Appearance; `prepare_image`
then pads without another resize. Weak final icons lose identity supervision.
This creates a new capture recipe; it does not claim equality to an old PNG
resized a second time. Preserve the original4096 M2 archive/recipe separately.

Nine new label/preparation checks pass: final noninteger geometry, weak previews,
unchanged capture, size/refusal bounds and paired identical rasters for hidden
panels, occluded middle slots and different unseen prefixes. The latter retains
visible T/I order without a hidden absolute count. Prepared tensor targets also
agree for indistinguishable images. Tests took2.057s,255,062,016B raw peak under
512MiB/60s, source milestone09fa4ef. Raw gate is runs/m3-model-capture-tests.json
and its stderr log. The source curator parked125 before child launch at host
commit32.2GB; automated curated scope remains pending, not a failed test run.
This supports an initial synthetic M3 curriculum, not real-image relabeling,
component accuracy or arbitrary post-labeled resizing/invalid alpha.

The public inspection API remains needs_review; it identifies M3 artifacts but
leaves queue/hold unknown. Joint learning, general resized-component supervision
and reviewed runtime decoding are still OPEN. Native resource/recovery profiles
are now verified below and retain their exact configuration/source scope.
No M2 admission covers the changed package or new heads. The partial4096-target
M2 study must continue from its original archive and optimizer history.

The joint streaming driver passes its scoped unit/contract gates; adequate
joint learning remains pending.
`train_m2_stream.py --component-capacity 10..64` selects the optional M3 heads,
final-capture sample policy and component targets/loss. Zero keeps the M2 path.
The lower joint bound covers all10 renderer slots without silent truncation;
the64-slot profile is required before a64-slot study. Ledger records include the
final appearance, capture policy and pixel/state identities. The curator checks
that the replay policy matches the joint contract and can replay old M2 records
without sending new keywords to their frozen sampler.

Per-case raw-head metrics separate matching supported fields from matching fully
readable components. A cropped tail, unknown middle identity or unknown hold can
never become an exact readable-component success. These are prototype diagnostics,
not decoded runtime/whole-state accuracy. Recursive checkpoint parity now covers
all nested component outputs and rejects changed tensor shape/dtype/value.

Current sourcef542096:57 selected cases pass across stream/joint25, component23
and capture9;11 more source/recovery cases pass separately. Source-pinned scopes:
docs/evidence/joint-stream-contracts/scope.json and joint-recovery-contracts/scope.json.
These68 cases are not a whole-suite, trained-model or accuracy claim. Actual M3
configurations were subsequently profiled with2 full-square optimizer steps,
all resident Adam states, recovery parity and nested final-output parity below.
No prior M2 or earlier-source admission permits joint training.

First actual M3 pilot profile now passes: capacity64,682,584 parameters,
168 trainable parameter tensors with5,461,344B resident Adam state, two full
512-square updates including warm backward, exact recovery and all nested final
outputs after checkpoint reload. Native640MiB,558,497,792B peak,22.728s outer.
Frozen source catalogueb6bb05f3a82b68ae41f5fb6ba8a256a89ddb69ed94c917807511bb636cf34255.
docs/evidence/m3-stream-pilot-profile-r1/scope.json pins actual M3 architecture,
configuration, weights, replayed2 recipes and27 probe images. This admission
covers that pilot/config/source envelope, not the larger model or long-run
stability, converged learning, inference decoding or real-image accuracy.

Native joint recovery also passes: continuous6 equals3+resume3 on data, loss
terms/gradient norms, evaluations, all model/optimizer tensors, CPU RNG and
restored module modes. Inference weights SHA2f97b540... identical. The curator
replayed17 training rows including carried prefixes and72 saved rasters/labels;
source/config/architecture and snapshot hashes match. Parent597,655,552B/1GiB,
child max561,909,760B/640MiB,95.148s outer. Named evidence:
docs/evidence/m3-stream-recovery-gate-r1/scope.json and proof.json.
This finite pilot recovery gate does not characterize the larger configuration,
long varied-shape memory or model accuracy.

The larger M3 capacity64 profile also passes:7,997,016 parameters,264 initialized
parameter tensors and63,977,184B resident Adam state. Both optimizer updates use
full512-square inputs; the second backward has warm optimizer state resident.
Native1GiB/300s,942,903,296B peak,43.663s outer, exact training recovery and all
nested final outputs after inference reload. Source remainsb6bb05f3...;
weightsad14d69a...; actual configuration/architecture and replayed2 recipes+27
saved probes are pinned by docs/evidence/m3-stream-large-profile-r1/scope.json.
The metadata replay itself passed under128MiB/30s. This admits that larger
configuration for bounded studies; varied-shape long memory, optimization,
complete-state accuracy and release decoding remain unverified.
