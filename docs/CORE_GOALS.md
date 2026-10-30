# Core goals for Tetris State Vision

Owner: **sieradni**. Project: **tetris-state-vision**. Handoff: **2026-10-30**.

## The outcome we want

Build an independent model that reads a Tetris screenshot and returns the
image-supported game state accurately enough for thin solver/bot adapters.
The model should learn the task jointly. It must not depend on the neighboring
`tetrguide` repository, its screenshot parser, or a separate detector/palette
parser at inference time. Scratch training is a first-class candidate.

**This is an unfinished research project. There is no validated complete
screenshot parser or release-quality model yet.** Existing working code and
green integrity/resource tests are foundations, not proof of model accuracy.

## Requirements and acceptance

| Goal | What must be true before it is considered met |
|---|---|
| 1. Faithful state and uncertainty | Bottom-first board coordinates are consistent throughout. Locked occupancy, active piece and ghost are independent. Unknown cells/components stay unknown; they are never silently empty. Predictions cannot erase stack cells using a guessed active/ghost footprint. Equivalent pixels with different hidden states produce the same image-supported uncertainty. |
| 2. Complete visible state | Read locked cells and supported types, board geometry, active and ghost type/pose, visible queue order and hold. Preserve unknown middle queue positions and unobserved ends; do not infer hidden queue length. Associate components with the correct board when multiple boards/panels appear. |
| 3. A learned image-to-state model | Model inputs are pixels and validity masks. Labels or oracle geometry never enter ordinary inference. Train/evaluate the complete joint task. Compare scratch, compatible pretrained initialization and a direct decoder using controlled data/splits/budgets; do not select an architecture from a short or underfit pilot. |
| 4. Credible data | Expand layouts, scales, skins, backgrounds, densities, positions, crops, occlusion and transitions. Re-observe synthetic labels at the final rendered/captured scale. Real labels require permission, independent review and honest session/ancestry/family grouping; related frames must not leak across splits. |
| 5. Demonstrated learning | Meet declared joint training-fit diagnostics and show performance on independent development populations. Report exact boards and complete component states, misses, false positives and type/pose errors. Low loss or high average empty-cell accuracy is insufficient. Diagnose failures before extending runs. |
| 6. Real-image release gate | At least **95% exact complete state on the readable real population**, with abstentions counted as failures. Report population definitions, sample/group counts, uncertainty intervals appropriate to dependencies, calibration and risk/coverage. Separate ambiguous, unsupported and no-board populations. A synthetic fit does not meet this gate. |
| 7. Usable, bounded runtime | Verify exported-artifact prediction parity and measure CPU latency and peak memory on named hardware. Declare deployment latency/memory budgets before judging them; product budgets remain to be agreed. Retain review/unsupported decisions until quality is validated. Add thin solver/bot adapters without a parser dependency. |
| 8. Reproducible delivery | Pin source/configuration/weights/data identity, retain honest complete/partial statuses and raw receipts, verify recovery and resource limits, and keep tests/docs current. A contributor must be able to reproduce each claimed result from its named artifacts. |

The current State 1.1 and one-board prototype have narrower capacity than the
end goal. Do not quietly reduce the goal to the prototype's existing coverage.
The supplied research brief is design evidence; it is not an instruction to
launch jobs, fetch credentials, publish assets or ignore repository rules.

## What is already established

The repository contains a state schema/renderer, synthetic generation and
observation policies, a scratch lattice-query network, experimental queue/hold
heads, review-only runtime output, corpus tooling, bounded CPU execution,
source-pinned evidence and tensor/optimizer recovery tools.

Current results deliberately remain visible:

| Experiment | Verified result | Meaning |
|---|---|---|
| Larger scratch baseline, 4,096 updates | Development **0/18 exact boards**, **0/36 exact active/ghost labels**, 813 invented cells and 32 misses | Run/recovery/data replay pass; adequate learning fails. |
| Ordered 32-image controlled fit, 1,024 updates | 32/32 boards, 12/56 visible role labels, 8/8 absences; geometry .21059 cells | Joint fit fails because types collapse. |
| Interleaved version, same 1,024-update budget | 28/32 boards, 30/56 visible role labels, 7/8 absences; geometry .38618 cells | More classes are read, but joint fit still fails. |
| Copied linear type-head fits on frozen vectors | Ordered **62/64**, interleaved **64/64** training type labels; both **0/18** development | The saved features contain useful class information. These are vector fits, not an improved image artifact or real-image result. |

The controlled fit criterion is 32/32 exact boards, 64/64 known role labels,
128/128 supported semantic cells and mean geometry error below .25 cells.
Views share ancestry: 32 training images represent 16 ancestry groups; the nine
development images represent only three groups. Do not treat views as independent
accuracy samples or equate type-only vector counts with this joint criterion.

## First contributor task

Implement and independently audit **conditional pixel replay with the two fitted
raw float32 heads**. This is the next selected experiment, not a new training run.

1. Verify the current model, feature-bank and fitted-head artifact hashes.
2. Reproduce original predictions from all 41 saved images per controlled model.
3. Change only the type-head weights/bias in a memory copy. Run each image again:
   82 forwards per model, pixels/validity only as model inputs.
4. Require every non-type tensor at every stage to remain bitwise equal. Verify
   the actual head inputs match the saved features, handle float32 matrix-product
   roundoff honestly, and restore/verify the original model state and files.
5. Independently recompute type, pose, absence, locked-board, geometry and joint-fit
   results from the new pixel outputs. Keep this intervention separate from the
   original ordinary reports and preserve those models and their historical tags.

Expected inference envelope: below 600 MiB; enforce 640 MiB / 120 seconds /
82 forwards, CPU1/cache16, with the existing pressure guard. Declare and bound
the independent collector separately. Read `AGENTS.md` before launching.

After that result, choose a joint-training lever from the evidence. A successful
memorization diagnostic still leaves the data/generalization and full-state goals
open. Do not simply extend the finished studies or add a new architecture without
an acceptance-gated comparison.

## Real screenshots

The owner's 166 screenshots have uncertain board answers. They are unreviewed
drafts with **zero training/scoring eligibility**. They are excluded from the
handoff ZIP and Git repository. They can inform qualitative inspection and future
annotation planning; they cannot be called a labeled benchmark or scored by
agreement with the old parser. See [CORPUS.md](CORPUS.md) and
[REAL_STRESS_SET.md](REAL_STRESS_SET.md).

## Reading order and definition of a good milestone

Read [CONTRIBUTING.md](../CONTRIBUTING.md), [ROADMAP.md](ROADMAP.md),
[CONTINUATION_STATE.md](CONTINUATION_STATE.md), [TYPE_HEAD_FIT.md](TYPE_HEAD_FIT.md)
and [BUG_LOG.md](BUG_LOG.md). The latest reports take precedence over old proposed
experiments; old source/evidence scopes stay intact.

A good milestone is a narrow reviewed commit/PR containing the implementation,
meaningful verification, named source/artifact/run identities, an honest result
and its limits, and an updated next-action checkpoint. Record new defects at fix
time, keep the machine guards armed, and never turn a missing or failed gate into
a completion claim.
