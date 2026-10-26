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

## Controlled 128-step result

`docs/evidence/m2-learn128/comparison.json` proves byte-identical data manifest,
unchanged optimizer driver, identical initialization metrics and exact equality
of all first-16 step losses/terms/gradient norms/sample IDs. The aggregate source
digest changed only because the unused runtime caller was added; the model,
loss, preprocessing and checkpoint sources stayed unchanged. Weights SHA256 is
46787c4f236c421dc7b3eb693665c1c8d09688c28f7330a28c2d0abe8907967e.

| Diagnostic | 16 steps | 128 steps |
|---|---:|---:|
| Training loss | 1.6126 | .3498 |
| Training exact locked boards | 2/6 | 2/6 |
| Training mean geometry error, cells | 1.4253 | .3645 |
| Training exact known active/ghost layers | 8/12 | 10/12 |
| Development loss | 5.2027 | 13.7825 |
| Development exact locked boards | 0/6 | 0/6 |
| Development mean geometry error, cells | 1.1470 | 1.3768 |
| Development exact known layers | 4/12 | 0/12 |

The two exact training boards are the empty clean/no-grid pair. The independent
checkpoint analysis finds training TP/FP/FN/TN = 105/36/7/1052; development =
104/71/30/995. High cell accuracy hides incorrect complete boards, particularly
sparse inputs. The [error image](../examples/m2-learn128-locked-errors.png) and
per-case layer/geometry records are curated with the run. Development refinement
can worsen the lattice, so lower training geometry error is not generalization.
Six images in either split are only three matched-appearance ancestry groups.

Resource receipt: 128 steps completed in 50.537 s under 1 GiB / 180 s / one
thread; raw peak Job commit 479,604,736 B. Independent 12-image checkpoint analysis
passed in 5.892 s, raw peak 811,859,968 B under 1 GiB / 60 s. Its peak exceeded
the <512 MiB forecast, not the enforced ceiling; budget later analyses at
<900 MiB with the same cap. The checkpoint reload remains exact.

M2 learning acceptance remains OPEN. Next is a 512-step finite memorization
check on the same cases, initialized from the same seed (not resumed). The
predeclared training-fit diagnostic is 6/6 exact locked boards, 12/12 known
active/ghost layers and mean geometry error <.25 cells. Passing it does not
establish development, unseen-family, complete-state or real-image quality.
If it fails, inspect per-case sampling/gradients/targets before extending time.
