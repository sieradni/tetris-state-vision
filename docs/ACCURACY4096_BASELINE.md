# Completed larger scratch baseline

The original7,694,010-parameter M2 model completed its declared4,096 optimizer
updates with the original archived source, AdamW state and recipe. Resource,
snapshot recovery and full data replay pass. **Adequate learning remains open:**
the final ordinary replay finds no exact locked board in either saved probe split
and no exact active/ghost label on development. Keep this study complete and
frozen; its completion does not justify extending it or selecting this architecture.

Artifact: `runs/m2-stream-accuracy4096-s10/checkpoint/`, weights SHA256
`067516380eb5193f5ab21afd6ae0573d25610d6a9285b18a186871b36b9aadd1`, source catalogue
`121fd05a2f75e30b0c738dac592c67c6d015aa6043a43e4032d91ac662de794b`.
Capacity0/layer-evidence off, full512 inputs, CPU1/cache16, seed20261026, learning
rate.001, AdamW decay.01, evaluation128 and snapshot64 stayed unchanged. The last
segment resumed3,840 and performed256 updates. Earlier pressure stops and exact
recomputed prefixes retain their original receipts; completion does not relabel them.

[Full study replay](evidence/m2-stream-accuracy4096-s10/scope.json) verifies all4,096
training recipes and27 saved images. Controller receipts bind actual inference/
training payloads, limits, raw logs and process termination. [Independent final
errors](evidence/m2-stream-accuracy4096-errors-r1/scope.json) bind one ordinary
pixel-only forward per saved image, exact saved metrics, unchanged tensors and
source/configuration/runtime. A separate collector recomputes row-level totals.
All known native/curator PIDs and matching commands ended.

| Final diagnostic | Training probes | Development probes |
|---|---:|---:|
| Images / ancestry groups |9 /3|18 /6|
| Exact locked boards |0/9|0/18|
| Exact nonempty boards |0/7|0/14|
| Recognized positive cells |117/168|369/401|
| Missed positive cells |51|32|
| Invented positive cells |421|813|
| Exact known active labels |0/9|0/18|
| Exact known ghost labels |9/9|0/18|
| Mean geometry error, cells |3.19118|2.01544|
| Mean loss |4.10707|3.95378|

All nine training-probe ghost targets and predictions are **absence**. This
contributes the entire9/18 aggregate known-layer score; it is not evidence of
reading a visible ghost's type or pose. Training-probe active target groups are
absence,O,J, three views each, whereas every active prediction is L. Development
active targets are T6,S3,J3,L6; predictions are J6,L12. Active type is wrong15/18,
column18/18,row12/18,rotation18/18. Every development ghost target is visible,
but all18 predictions are absence. Correct ghost row18/18 does not rescue a wrong
type/status. Unknown supervision remains excluded.

The four empty development images also fail:228 of813 invented cells come from
them. The14 nonempty images contribute585 invented cells and32 misses. Recognizing
most positive cells therefore does not establish exact occupancy. Geometry has
1.97801-cell mean origin error,.16367 per-column basis error and.08186 per-row
basis error, in the loss's axis-normalized units, not a rotated-board metric.

These fixed synthetic probes share ancestry. They are not a sealed test, unseen
client, confidence interval or real screenshot accuracy estimate. The nine
training probes do not summarize every streamed training state. Their limited
class/status coverage is why the balanced32-image studies were needed. Neither
this study nor the two1,024-update studies establish a converged architecture
comparison or causal bottleneck. Reviewed real labels, adequate joint learning,
queue/hold, calibration/export/runtime and full exact-state release remain open.

| Final bounded operation | Wall time | Raw peak Job commit | Enforced limit |
|---|---:|---:|---:|
| Last256 updates, native child |333.993s|979,279,872B|1GiB|
| Last segment controller aggregate |362.676s|1,015,963,648B|1.25GiB|
| Full4,096-recipe curator |28.066s|127,033,344B|128MiB|
| Ordinary27-forward error replay |13.546s|586,768,384B|1GiB|
| Independent error curator |.406s|76,595,200B|128MiB|

These times describe the final named operations, not the multi-session study.
Kernel enforcement, wall/work limits and host-pressure guard remained armed.
Final training manifest SHA256:
`be66e117a8dd6c1b3a24c7522cf90cda629de78516fd21924b88cb6a6e00d7dd`;payload
`a21c1c56dc7face7eabf49d4cac3f9ee9c1b4a1eaef233af77ef2a97663f1382`.

Next execute the [read-only type-feature diagnostic](LAYER_FEATURE_DIAGNOSTIC.md)
on these artifacts and the two controlled studies. Preserve ordinary outputs and
report ancestry/class support before choosing another mechanism. Its centroid
readout is analysis only, not a parser dependency.
