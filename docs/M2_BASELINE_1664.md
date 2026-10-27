# Original baseline at1,664 updates

The original accuracy configuration now has a verified saved inference and
training checkpoint at1,664/4,096 updates. Segment s4 restored bank1,152 and
performed512 new updates using its parent's archived trainer,unchanged seed,
Adam/RNG state and global recipe. It is explicitly `step_partial`;adequate
learning and the fixed4,096-update target remain open.

Evidence: [segment](evidence/m2-stream-accuracy4096-s4/scope.json),
[native receipts and actual artifact hashes](evidence/m2-stream-accuracy4096-s4/controller-receipt-scope.json),
[ordinary replay](evidence/m2-stream-accuracy1664-errors-r1/scope.json),
[matched comparison](evidence/m2-stream-accuracy1664-errors-r1/comparison704.json).
All1,664 training recipes and27 saved probe images replayed. The native parent
completed in713.417s at993,869,824B peak under1,280MiB;training used957,689,856B
under1GiB. The curator completed in12.593s under128MiB. All processes exited.

Source catalogue remains121fd05a2f75e30b0c738dac592c67c6d015aa6043a43e4032d91ac662de794b;
weights7b1e4587437514db77cc58b19463ecbc1c0e23ed6bd9ff19b697b23cee70f4fb.
The newer optional M3 source was not substituted. All20 previously unbanked
updates1153..1172 reproduce exactly in data/loss terms/gradients;no old1172
tensor artifact exists for comparison.

The ordinary replay uses byte-identical probe recipe ledgers to the earlier704
report:9 fixed training probes in3 ancestry groups and18 development probes in6
groups. Matched views are dependent observations. These are synthetic raw-head
diagnostics,with no calibrated confidence,complete-state or real-image claim.

| Development diagnostic | Saved704 | Saved1664 |
|---|---:|---:|
| Exact locked boards |0/18|0/18|
| Positive cells correctly occupied |384/401|384/401|
| False occupied cells |326|270|
| Missed occupied cells |17|17|
| Exact known active/ghost layers |0/36|2/36|
| Mean lattice error,cells |1.84647|2.22941|

Cell false positives decrease,while geometry worsens and no complete board
becomes correct. The two exact development layers are ghosts. The active head
predicts the same piece class in all18 development images;15/18 piece identities
and all18 active columns are wrong. Both layer heads have15/18 development
identity errors. These concrete failures remain targets for learning work.

Training-probe layers score8/18,but all eight successes are absent ghosts.
Every training ghost is absent;none of the six observed active layers is exact.
The score therefore does not demonstrate successful observed-layer recognition.
This distinction is derived from the ordinary report's supported per-case target
types,not guessed latent labels. Development has36 observed layers.

The [conditional lattice diagnostic](evidence/m2-stream-accuracy1664-geometry-r1/scope.json)
uses a separate54-forward run. It reproduces every ordinary saved metric,then
deliberately supplies the correct supported lattice during a second forward.
Weights remain unchanged. Development exact boards stay0/18,cell accuracy changes
from.920278 to.904444,and exact layers from2/36 to0/36. On these weights,the
intervention alone does not resolve the learning failure. This is not automatic
accuracy,a capacity bound or a production correction;the rest of the trained
reader and learned sampling offsets remain unchanged.

Next:continue the original finite baseline toward its unchanged target while
auditing actual observed/absent piece-class exposure and local reading failures.
Do not choose an architecture from an incomplete,poorly converged run. New
architectures/targets require separate frozen studies and resource admission;
they must not overwrite this study's optimizer/history. Keep real images as
unlabeled drafts until independent review supplies supported labels.
