# Type-head checkpoint diagnostic

The completed matched rate studies were replayed read-only on their same 32
publicly supported training images at initialization and every 64 updates
through 1,024: 17 checkpoints and 544 forwards per model. Final predictions
match the separately audited ordinary pixel reports. The observers verified
each retained snapshot's tensor payload and committed ledger prefixes, exact
saved image/recipe replay and frozen source; independent collectors checked
model identities, reconstructed head logits with a float32 roundoff bound,
and recalculated the analysis. Pixels and validity were the only model inputs.

| Type-only diagnostic | Control | Fourfold head rate |
|---|---:|---:|
| Correct at initialization, out of 64 supported roles | 8 | 8 |
| Correct range over 17 sampled checkpoints | 8–20 | 8 at every checkpoint |
| Correct at update 1,024 | 20 | 8 |
| Correct-to-wrong transitions across 16 intervals | 86 | 80 |
| Wrong-to-correct transitions across 16 intervals | 98 | 80 |
| Sum of sampled head-parameter L2 movements | 2.779 | 11.659 |

Transitions count the same image-role pair again if it changes in another
interval; they are not independent examples. They compare only adjacent
64-update observations, so changes and reversals within an interval are not
observed. The faster-head model's constant total masks repeated turnover in
which roles are correct. At the final checkpoint the control predicts only
absence, Z, J and L across these roles; the candidate predicts only absence, O
and L. These are type-only predictions; exact role results additionally require
known status and supported pose, as reported in the completed fit comparison.

For each checkpoint the collector also derives the mean unweighted
cross-entropy gradient of the final affine type head for each of the eight
supported target classes, holding the captured features fixed. At the final
checkpoint 21 of 28 control class-pair cosines and all 28 candidate class-pair
cosines are negative. This describes local final-head gradient geometry, not
the full multi-stage joint gradient, feature updates, clipping or Adam's actual
update. In particular, negative class-pair cosines by themselves do not prove
that the optimizer caused forgetting. The head-movement sum is a sum of
distances between sampled checkpoints, not the full path length.

Evidence: [matched comparison](evidence/type-forgetting-comparison-r1/scope.json),
[control observer](evidence/type-forgetting-control-r1/scope.json) and
[candidate observer](evidence/type-forgetting-candidate-r1/scope.json). Both use
source catalogue `18ce798eefe37b6d3ae778993fa3c94045312632b12715df5cab80749d258125`;
the actual model weights and individual snapshot identities are derived from
the verified artifacts in those scopes. Native observer peaks were 427,859,968
and 423,342,080 bytes under 640 MiB; 128 MiB independent collectors and the
bounded comparison passed. The two small analytic tests include a known
softmax gradient geometry and unknown-label refusal.

This is one seed on 32 related synthetic views from 16 ancestries. It does not
measure development generalization, real screenshots or complete-state
accuracy. The turnover and persistent collapse motivate testing whether
class-balanced gradient accumulation reduces order sensitivity, with matched
example exposure, explicit optimizer-update budgets, fresh resource/recovery
admission and the existing full joint-fit criterion. That is a hypothesis to
predeclare and test, not a selected release mechanism or an admitted training
run. Preserve the finished rate studies and their original evidence.
