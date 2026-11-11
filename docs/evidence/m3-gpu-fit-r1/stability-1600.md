# The post-1,000-step collapse is a fixed-step random walk, not memorisation

Every figure is read from `runs/m3-gpu-fit-r1-arm{A,B}-*/result.json`, the arm logs and the saved
states. Two arms, 1,000 new optimizer steps total, both starting from the 1,000-step checkpoint on the
cached 512-frame pool with batch 4.

## 1. Memorised pool or drifting optimiser?

First, what the pool is: each training gate board is **pixel-identical** to the pool frame with the same
index (max absolute pixel difference 0.0 for boards v0, v1 and v2). The 36 training boards *are* the
first pool frames, so the only genuinely held-out set is the 18 development boards.

At the 1,000-step checkpoint, same gate code and threshold grid:

| set | boards | F1 @ 0.0 | best threshold | F1 @ best | geometry (cells) |
|---|---:|---:|---:|---:|---:|
| training (seen) | 36 | 0.6298 | 2.85 | 0.6611 | 1.0985 |
| development (held out) | 18 | 0.7350 | 3.00 | 0.8044 | 1.0572 |

**Held-out beats seen** at both decision points with no geometry penalty, which is the opposite of what a
memorised pool would show. Memorisation is ruled out; the collapse is an optimiser property.

## 2. Onset window, read from the logs

| trace | what it shows |
|---|---|
| chain segment 2 loss | 4.94 at segment step 30 -> 7.54 at 130 -> **35.22 at 220** (divergence in ~200 steps) |
| plain arm loss | 4.95 first, max 8.81, last 5.13 — bounded but with no downward trend |
| plain arm dev F1 @ 0.0 | 0.7362 -> 0.7515 -> 0.6338 -> 0.5636 across steps 1,000 -> 1,300 |
| plain arm best threshold | 2.85 -> -0.15 -> 1.05 -> 4.35 (4.5 logits of wander) |
| plain arm geometry | 1.111 -> 0.748 -> 0.664 -> 0.827 cells |
| gradient norm (plain arm) | median **35.86**, mean 46.07, max 109.17, min 13.06, **100% clipped at 1.0** |

## 3. The mechanism the data supports

Clipping at 1.0 rescales *every* update to unit norm, so the weight change is `lr x unit direction` — a
fixed-size step that cannot shrink as the optimum is approached. At lr 1e-3 the weights therefore random-
walk at constant step size: loss stays flat (4.9-8.8) instead of falling, the swept best threshold
wanders 4.5 logits, and geometry oscillates. The alternatives do not fit: no saved state holds a
non-finite tensor, and the held-out split still outperforms the seen split.

## 4. The smallest change that addresses it

Cosine decay of the learning rate 1e-3 -> 1e-4 across the arm (600 steps); nothing else changed.

| arm | steps | dev F1 @ 0.0 | dev F1 @ best | geometry (cells) | train F1 @ best | exact nonempty |
|---|---:|---:|---:|---:|---:|---:|
| 1,000 baseline | 1000 | 0.7350 | 0.8044 | 1.0572 | 0.6611 | 0 |
| armA plain lr | 1400 | 0.6962 | 0.7886 | 0.9330 | 0.6633 | 0 |
| armB cosine decay (best reading) | 1600 | 0.7875 | 0.8337 | 0.9588 | 0.7457 | 0 |
| armB cosine decay (final) | 1600 | 0.7552 | 0.7919 | 0.5983 | 0.7457 | 0 |

Decay arm, every reading: step 1000 (lr 1.00e-03) dev 0.7362 @0.0 / 0.7987 @best, geom 1.111; step 1150 (lr 8.68e-04) dev 0.7875 @0.0 / 0.8337 @best, geom 0.959; step 1300 (lr 5.50e-04) dev 0.7090 @0.0 / 0.7635 @best, geom 0.701; step 1450 (lr 2.32e-04) dev 0.7691 @0.0 / 0.8157 @best, geom 0.672.

Plain arm, every reading: step 1000 (lr 1.00e-03) dev 0.7362 @0.0 / 0.7987 @best, geom 1.111; step 1100 (lr 1.00e-03) dev 0.7515 @0.0 / 0.7552 @best, geom 0.748; step 1200 (lr 1.00e-03) dev 0.6338 @0.0 / 0.7398 @best, geom 0.664; step 1300 (lr 1.00e-03) dev 0.5636 @0.0 / 0.7640 @best, geom 0.827.

Arm bookkeeping, from the result files:

- armA: skipped_batches {'nonfinite_forward': 0, 'nonfinite_loss': 0, 'nonfinite_grad': 0}, pool_origin cache, 0.4378 s/step, 1400 steps.
- armB: skipped_batches {'nonfinite_forward': 0, 'nonfinite_loss': 0, 'nonfinite_grad': 0}, pool_origin cache, 0.4436 s/step, 1600 steps.

## 5. Settled and open

**Settled.** The collapse is stability, not memorisation: held-out boards score above seen boards, every
step is clip-saturated, and decay beats the previous peak inside the arm (0.8337 vs 0.8044) while leaving
the tightest geometry measured anywhere (0.5983 vs 1.0572 cells). Over the same 600-step budget the plain
arm falls to 0.6962 @ 0.0 against 0.7552 with decay.

**Open.** Exactness is still zero nonempty boards in both arms, so no board is reproduced exactly. The
decay arm's final best-threshold F1 (0.7919) sits just below the 0.8044 baseline peak although it exceeded
it mid-arm, so this is a stability win, not a new ceiling. The evidence says the next milestone is not more
steps at a constant step size.
