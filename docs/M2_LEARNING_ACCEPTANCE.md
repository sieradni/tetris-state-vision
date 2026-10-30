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

## Saved 511/512-step wall-partial run

The inner 240-second limit stopped the next study after **511 of 512** requested
steps. Exit code 2 and `wall_partial` are preserved in
`docs/evidence/m2-learn512/receipt.json` and comparison.json. This is not a
completed 512-step gate. The saved checkpoint reload is exact; all first 128
optimizer records reproduce the previous run exactly, with identical manifests,
initial metrics, driver and runtime-source digest. The explicit partial-evidence
path has three green refusal/admission fixtures in the same evidence directory.
It refuses outer timeouts, other failures and accidental completion claims.

| Diagnostic at saved step 511 | Training | Development |
|---|---:|---:|
| Exact locked boards | 6/6 | 0/6 |
| Cell accuracy | 1.0 | .9225 |
| Exact known active/ghost layers | 10/12 | 0/12 |
| Mean geometry error, cells | .4106 | 1.3112 |
| Loss | .1690 | 25.5779 |

The predeclared joint training-fit criterion **fails** despite exact locked
boards. The dense clean/no-grid pair's ghost has correct type but row 15 instead
of row 10. Geometry also exceeds .25 cells. Development TP/FP/FN/TN are
92/51/42/1015. This small model can memorize the locked training states; it has
not demonstrated stable layer/geometry fit or general reading. Next inspect
single-cell interventions at fixed geometry/appearance, not a longer blind run.

Resource evidence: 244.005 s outer, raw peak 480,751,616 B within 1 GiB / 300 s;
checkpoint analysis 4.717 s and 811,851,776 B under 1 GiB / 60 s. A new unrelated
release build began after launch; no further training was started alongside it.
Weights remain at runs/m2-learn512/checkpoint/weights.pt, SHA256
ca4c0f34a81b8c7591fab7db0e92fd89d652a27dea015cf63473c3f7b780bcdf.

## Local reading interventions

`docs/evidence/m2-local-reading/scope.json` records a 42-forward probe on the
same saved artifact: six exactly reproduced baselines and 36 single-cell
counterfactuals (18 distinct changes, each with two matched appearances).
The probe verifies the baseline pixels/labels, one changed supported cell,
unchanged other components/geometry and a raster delta contained inside that
cell. It excludes active/ghost footprints. Ground-truth geometry is never passed
to model.forward. No hidden-label policy changes or parser corrections occur.

Every intervention moves the target occupancy probability in the right direction,
but only 20/36 changed cells and 18/36 complete changed boards are correct.
Insertions: 12/24 targeted, 10/24 exact boards; removals: 8/12 for both.
Two unrelated predictions change. These dependent, selected synthetic cases
are a functional probe, not an accuracy estimate. They show partial local
reading with residual state/position bias; exact training boards alone are weak
evidence of a general reader. The 42 forwards took 7.937 s, raw peak 816,402,432 B
under 1 GiB / 120 s / at most 54 forwards, one CPU thread.

Next engineering direction: bounded streaming generation of substantially more
states/positions, with matched clean/no-grid and visible single-cell variants,
separate held-out seeds, no cached tensor growth and no user-image labels.
Keep the accuracy-sized configuration as a separately profiled candidate. The
small pilot, ghost/geometry failures and complete-state requirements remain open.

## Fresh-state study at768 updates

The source-frozen pilot completed768 updates across256 ancestry groups. The
finite resource/recovery gate passes at640MiB:519,782,400B raw peak,2,748,416B
second-half growth, exact565 recorded loss/gradient/data prefix from the prior
failed run, final recovery/inference identity and full sample/probe replay.
Evidence: docs/evidence/m2-stream-learn768-r3/analysis.json and scope.json.

| Final diagnostic | Training probes | Development |
|---|---:|---:|
| Images / ancestry groups |9 /3 |18 /6 |
| Exact locked boards |0/9 |0/18 |
| Cell accuracy |.7950 |.8822 |
| Exact known active/ghost layers |12/18 |0/36 |
| Mean geometry error, cells |3.0099 |1.8522 |
| Loss |4.1976 |3.7555 |

These probes differ from the earlier fixed6-image pilot; their scores are not
an architecture ranking. Lower loss/cell improvement still fail exact state.
The first larger7,694,010-parameter configuration separately passes a1GiB warm
resource profile; its first128-update optimization diagnostic is distinct from
adequately converged scratch evaluation. It must retain explicit wall partial
status and resume the same source/recipe/optimizer rather than reset learning.

The oracle-lattice intervention deliberately supplies supported synthetic
geometry only in a separate diagnostic forward, to distinguish localization
from cell-reading effects. It cannot be reported as automatic parser accuracy,
an upper bound or a production correction. Weights and ordinary output must
remain unchanged, and ordinary scores must reproduce the saved artifact first.
The54-forward gate passed on the saved pilot: all ordinary metrics reproduced,
weights unchanged and decoder restored. Correct geometry gives0 exact boards
in both splits; cell accuracy changes.7950->.8333 training and.8822->.8583 dev,
with layers unchanged. These conditional effects do not isolate a universal
capacity limit or prove that geometry alone is the bottleneck.
docs/evidence/m2-stream-pilot-geometry/scope.json.

The larger7,694,010-parameter model completed128updates as wall_partial93 plus
exact resume35; the parent's partial tag, native receipt, snapshot and committed
prefixes remain attached. All128 data recipes match the pilot's first128.
Development at128: larger4/18 exact locked boards versus pilot0/18; all four
larger successes have zero locked cells (nonempty0/14), with0/36 exact layers
and2.4785-cell geometry. This one-seed, short-budget diagnostic does not select
architecture or establish convergence. Full curation and counts are in
docs/evidence/m2-stream-accuracy128-r2/analysis.json and scope.json.

For the fixed4096-target study, first distinguish missed/invented locked cells,
grid translation/basis errors and active/ghost type/row/column/rotation failures.
`scripts/probe_stream_errors.py` makes one ordinary pixel-only forward per saved
synthetic probe, after verifying the native complete/controlled-partial receipt,
frozen source, weights/config/runtime and replayed pixels/annotations. It must
reproduce the saved metrics before decomposing them; no oracle geometry enters
the model. Empty and nonempty populations derive from actual supported labels,
not the generator's density name (a counterfactual can add a cell to an empty
scene). Unknown labels are excluded and repeated ancestry is counted explicitly.

The pure analyzer has12 adversarial fixtures: empty-cell accuracy hiding one
miss, bottom-first false positives, unknown-label invariance, no-evidence null
scores, analytic translation/scale, symmetric rotations, independent layers,
type/occupancy disagreement, strict zero threshold, malformed contracts and
dependent aggregates. These fixtures do not verify neural replay or accuracy.
Committed-source scope: docs/evidence/m2-error-analysis-contracts/scope.json,
12/12 on1531769,33,021,952B raw peak under128MiB/30s.
Native inference replay now passes on the saved larger128 model:27 forwards,
32.419s outer,546,164,736B under1GiB. Every saved metric reproduces exactly and
model tensors remain unchanged. docs/evidence/m2-stream-accuracy128-errors/scope.json.
Development has401 positive cells:15 recognized,386 missed,8 false positives.
All18 active and18 ghost outputs predict unknown. Origin and basis errors both
remain substantial (2.1934-cell mean origin error,.4096 per-column and.1366
per-row basis error). The high mean cell accuracy mainly reflects empty cells;
this short-budget report does not select a causal architectural repair. Continue
the original longer study before ranking/tuning its alternatives.

At global704 (verified bank192+512new segment), the same saved27 probes now
reproduce with384/401 development positive cells recognized,17 misses and326
false positives. Both piece layers predict known, but each has18/18 wrong types;
no development board or known layer is exact. Mean geometry error1.84647cells.
The change trades missed cells for invented cells; it is not a quality pass or
an adequately converged comparison. Ordinary inference took33.423s under1GiB,
551,600,128B peak, model tensors unchanged. Source/artifact/case evidence:
docs/evidence/m2-stream-accuracy704-errors-r2/scope.json and result.json.

Future4096-target segments need their own saved-artifact report after their
Job ends. Use a NEW output,1GiB/120s/27 forwards on9+18 probes, one CPU.
Expected below the admitted training envelope, experimentally capped; do not
run another heavy model process while training is live. Suggested command:

```
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 1024 --wall-seconds 120 --work-items 27 --receipt runs/m2-stream-accuracy4096-s3-errors-job.json -- .venv/Scripts/python.exe scripts/probe_stream_errors.py --run m2-stream-accuracy4096-s3 --output runs/m2-stream-accuracy4096-s3-errors
```

Hidden logs/state and exact process verification are required. A saved controlled
partial segment may be analyzed with its original partial tag; a pressure stop
without final weights/summary is not eligible. Geometry components use the same
axis-normalized units as the frozen M2 loss; they do not establish rotated-board
accuracy. Confidence values are raw diagnostics, not calibration or API policy.

## Original larger baseline complete at4,096

The fixed original study and independent final27-image ordinary replay are
complete. Adequate learning remains OPEN: development0/18boards and0/36roles,
813 invented cells,32 missed positives;all nine correct training-probe ghost
labels are absence. See ACCURACY4096_BASELINE.md for artifact/source identities,
raw errors,native receipts and scope. The study stays frozen at its declared
budget. The read-only class-feature observations and independent collectors now
pass on all three completed models,with unchanged ordinary outputs/weights.
LAYER_FEATURE_DIAGNOSTIC.md records counts and limited class-reference coverage.
The finite copied linear-head fits now reach62/64(ordered) and64/64(interleaved)
training types on saved features;development remains0/18both with worse losses.
TYPE_HEAD_FIT.md binds native admission,first-two update replay,rawfloat32 heads
and independent audits. Next conditional pixel replay must preserve every
non-type output before any image-model fit claim. These diagnostic
probes do not establish convergence,reviewed-real accuracy or architecture ranking.
