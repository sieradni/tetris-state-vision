# M2 learning evidence, separate from numerical validity

The 85-case numerical gate at 8d9d0f2 proves finite operations, gradient paths,
masking, geometry conventions, checkpoint integrity and native resource contracts.
It does not prove that a network can learn the task or meet screenshot quality.

First optimizer pilot: 16 steps, three density strata, two matched clean/no-grid
appearances each, one CPU thread, input long side 384. Board origin, spacing,
pose and semantic state vary; images remain full scenes. Known target geometry
is supplied only to losses, never to model.forward. Six independent synthetic
development images use separate seeds, with the same renderer family. These are
development diagnostics, not a sealed test or an unseen-client evaluation.

Admission: require all 200 canonical locked cells to be readable at input scale;
<=64 MiB of cached tensors; finite states/steps/evaluation passes; no workers or
prefetch; one verified <=1 GiB Windows Job and outer/inner wall budgets. The
worker logs every optimizer step and publishes progress, data hashes, exact
configuration, raw diagnostics, hash-checked weights and reload parity. Existing
output directories are refused. A wall-partial run never reports complete.

Evaluate the evidence in order:

1. Were all requested optimizer steps completed inside the admitted envelope,
   with finite losses/gradients and exact saved/reloaded predictions?
2. Does the training loss decrease, and which terms account for the decrease?
   Geometry improvement alone does not establish learned occupancy or semantics.
3. For clean and no-grid images, report raw cell accuracy, exact locked boards,
   cell-unit geometry error and known active/ghost footprint correctness. These
   are uncalibrated internal diagnostics, not accepted full-state results.
4. Price a longer finite run from measured throughput and memory. Continue until
   clean/no-grid learning is demonstrated, or record the failure and improve the
   architecture/targets. A short scratch run cannot decide final model quality.
5. Keep the accuracy-sized configuration as a separate measured candidate. The
   small pilot is not selected as the release model by these easy synthetic gates.

Still required later: broad scene/skin/background variation, queue/hold joint
state, ambiguous transitions, reviewed real data, controlled model comparisons,
calibration, exact exported artifact parity and named latency/resource gates.
