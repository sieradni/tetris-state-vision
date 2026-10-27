# Optional layer evidence: completed pilot

The optional cell-evidence path did not establish an improvement in joint
image-to-state learning. It remains default off. Both matched768-update studies
are complete; neither is an accepted parser or release model.

The larger diagnostic used192 images from64 development ancestries fixed before
viewing predictions, outside the original six probe groups. Both models saw the
same pixels and supported targets. All seven observed piece classes and known
absence occur for both active and ghost. Every recipe was independently replayed,
384 raw predictions were decomposed again, and every reported aggregate was
recomputed. [Raw result](evidence/layer-generalization768-r1/result.json) and
[verified scope](evidence/layer-generalization768-r1/scope.json) preserve identities.

| Diagnostic | Control | Optional evidence |
|---|---:|---:|
| Exact nonempty locked boards |0/148|0/148|
| Exact empty locked boards |2/44|0/44|
| Exact observed active type/pose |0/165|4/165|
| Exact observed ghost type/pose |0/144|0/144|
| Exact known-absent roles |75/75|27/75|
| Locked false positives,34,519 supported negative cells |6,124|12,940|
| Locked cell accuracy,38,400 supported cells |83.35%|66.09%|
| Mean geometry error, source cell units |1.99669|1.96943|
| Exact observed visible layer grids |not enabled|0/309|

The four correct active poses are I pieces from only two ancestries. The
control's raw type heads predict absence for both roles in every image. The
candidate's active head predicts only I/O/absence, and its ghost head only
I/absence. Thus limited observed-piece progress accompanies worse locked
predictions and more false presence. The two control cases with exact locked
and both supported layers have empty locked boards and absent layers; this
does not establish complete-state performance, including queue/hold.

Coverage is still narrow: active pose rows14/15/16 have63/45/57 views; all144
observed ghosts start at row10. Three views share each ancestry. These results
do not test unseen appearance families, realistic occlusion or real screenshots,
and the related views are not independent trials. The166 user screenshots remain
unreviewed drafts, ineligible for training and scoring.

The original18-image fixed probe also remains0/18 exact boards and0/36 layers
for both models. At768 the optional model's cell accuracy is68.06% versus88.22%
for the control. The early128-update advantage on those partial metrics did not
persist. [Matched768 report](evidence/layer-learning768-comparison-r1/comparison.json)
binds configuration, weights, data, class/status denominators and common loss
terms. Total joint losses are not comparable: the candidate adds a.5-weighted
auxiliary objective. This combined pooling/supervision experiment does not isolate
the effects of its two components, and768 updates do not prove convergence.

Both models execute frozen source catalogue
`f095d3526162e5bbee45e0800c017296d05d45ec78cd76f4583642c2c5b7ce47`, with
pilot/capacity0, seed20261026, AdamW LR.001/decay.01, CPU1/cache16. Configuration
differs only in the explicit option; it adds130 parameters (647,098 to647,228).
The training recipe and all768 image/label records match exactly. The inference
weights are control`d183a22afcbe0e08d64b59bc1bd3cb2c337bd58e1d4d048f6406407943f43a74`
and candidate`a026ec06f2100076b1fa3d16d735cef70035ee80545df91f5f4becc225f622a2`.

Resource/recovery progress is separate from learning. Both two-full-square-step
profiles pass at640MiB with resident Adam and exact save/reload. The enabled
pilot's uninterrupted6 and3+resume3 updates agree on model/optimizer/RNG, modes,
data, losses, gradients and evaluation. An actual host-pressure stop logged541
but banked512; the final study resumed512 and exactly recomputed all29 unsaved
updates. [Pressure proof](evidence/layer-learning541-recomputed-r1/scope.json)
excludes elapsed time and does not claim an unavailable old541 tensor comparison.
The192-image inference gate peaked at432,234,496B under640MiB, and its independent
recipe/raw-case/aggregate audit at79,507,456B under128MiB. All owned processes
were verified gone; native receipts and raw sources are preserved with the scope.

Continue the separate original larger4096-update baseline from its original
archive; its verified position is2688/4096. Do not replace it with this pilot.
The next learning investigation should use an explicit all-class positive-control
fit and distinguish geometry, observed-piece localization and false absence.
Keep ordinary inference free of supplied truth and require joint exact-state
evidence before selecting a mechanism. The full research and release gates remain
open; the public API still returns needs_review/unsupported.
