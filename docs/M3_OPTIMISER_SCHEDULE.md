# Declared lever: the learning-rate schedule (predeclaration)

Status: **predeclared before any receipt exists.** No arm of this pass had run when this file was
written. It fixes the two arms, the matched configuration, the envelope, the read criteria and the
falsifiers so the result cannot be fitted after the fact. It changes no model, no label, no
threshold, no population and no earlier verdict.

## Why this lever

[V-086](BUGLOG.md) and [M3_OPTIMISER_REGIME.md](M3_OPTIMISER_REGIME.md) measured that **40,796 of
40,796** recorded training steps across **90** committed runs exceed the 1.0 gradient clip, and that
no learning-rate schedule exists anywhere in the tree. Because `clip_grad_norm_` rescales, every
update in the campaign's record is `lr * unit_direction` at a constant `lr 1e-3`: a fixed-magnitude
walk. The schedule is the only remaining **non-structural** suspect, and unlike the trunk-width fork
it adds **zero parameters**, so the matched-start contract is satisfiable by loading the same bytes.

The pilot already measured what the schedule does at this regime (V-085): constant rate peaked at
1,000 steps (dev F1 0.8044), decayed, and drifted to 0.52-0.64; cosine `1e-3 -> 1e-4` reached
**0.8337** at step 1,150 and ended with **0.5983**-cell geometry against 0.933.

## The two arms

Both arms continue the published checkpoint `9ae1fef9f649...` from `runs/m3-percell-balance-r1`
(128 steps, catalogue `d87836fdbbeb13c6...`) for **256 steps**, seed `20261026`, at the published
operating configuration:

```
--model accuracy --component-capacity 64 --component-identity-factor 4
--component-identity-readout positional --component-endpoint-term length
--occupancy-factor 4 --occupancy-positive-weight 4 --curriculum stream-v1
--capture-max-side 512 --seed 20261026 --steps 256 --evaluate-every 64
--checkpoint-every 32 --development-groups 6 --init-weights-from runs/m3-percell-balance-r1
```

| arm | run name | schedule flags |
|---|---|---|
| constant | `m3-occsched-constant-r1` | `--learning-rate .001` (no `--lr-final`) |
| cosine | `m3-occsched-cosine-r1` | `--learning-rate .001 --lr-final .0001` |

Nothing else differs: same start bytes, seed, population, objective, curriculum, thresholds,
evaluation cadence and checkpoint cadence. **Zero added parameters** - the configuration is
unchanged, so both arms and the source carry the same tensor set.

**Instrument identity.** `scripts/train_m2_stream.py` sha256
`7ff46b418e13d79be45c311f2df2180ba427251756f2fcb2fcda1f1fec0f4cf6`, `scripts/run_stream_segment.py`
sha256 `6ce85b11de13fcb369d3e4c5cfe0a485a38adb0b7b97cf2cf6b6801fbd1a3926`, suite **800 tests OK**
(`runs/schedule-suite-r1-job.json`). Each arm's own `summary.json` records the source hashes.

**Envelope.** 1,152 MiB and 1,860 s per arm, `work_items` 256; a 2-step square profile at 1,152 MiB
/ 300 s admits each arm; sweep at 640 MiB. One heavy job at a time.

## The schedule must be provable from the run records

Every curve record now carries `learning_rate` per step. This pass asserts, from `curve.jsonl`:

1. the **constant** arm has exactly **one** distinct rate, equal to `1e-3`, at all 256 steps;
2. the **cosine** arm has **256** distinct rates, non-increasing, step 1 exactly `1e-3` and step 256
   exactly `1e-4`;
3. both arms report the same `initial_weights_sha256` and `seed`;
4. step 1 is identical between arms in `sample_id`, `loss` and `gradient_norm` - the schedule is
   applied before the optimizer step, so a matched start cannot change the first forward/backward.

## Read criteria (standing only, no new thresholds)

| gate | source | criterion |
|---|---|---|
| fit gate | `fit_gate.fit_gate_opens` | **>= 3 of 28** exact nonempty training boards at threshold `0.0` |
| attribution | standing floor | per-cell F1 (cosine - constant) **>= +0.05** |
| ranking | `occupancy_sweep` | pooled AUC **>= 0.90** (descriptive; cannot pass at this scale) |
| placement | standing measure | mean placement error, cells: cosine **<= constant + 5%** |

The verdict is `schedule_engaged` only when attribution clears the floor **and** placement does not
regress; `null` when attribution is below the floor; `unreadable` when either arm failed to complete
256 steps or the matched-start proofs fail. The fit gate is reported for both arms and is **not**
expected to open.

## Prediction

- **Predicted:** the cosine arm's per-cell F1 exceeds the constant arm's by **>= +0.05** (the pilot's
  mechanism is a fixed-magnitude walk; decay shortens it late in the segment), with placement no
  worse than +5%.
- **Predicted:** the fit gate stays **FALSE** for both arms (0 of 28). No board is expected to become
  exact; the pilot's decayed arm reached 0.8337 and still produced 0 exact boards.

## Falsifiers

- per-cell F1 improvement `< +0.05` -> **null lever**, prediction falsified, recorded as such;
- placement regression `> +5%` -> **not-cleared** regardless of F1;
- any arm short of 256 steps, a failed receipt, a differing `initial_weights_sha256`, or a
  step-1 mismatch -> **unreadable**, and no verdict is claimed;
- a single fit-gate pass on either arm is reported as a surprise result, not as this lever's claim.

## What this pass cannot do

Overturn any prior lever verdict (both arms share the regime, so pairwise attribution stands), claim
the fit gate is reachable, alter the thresholds, population, gates, source checkpoint or any earlier
evidence byte, or support a real-image, complete-state or quality claim.

## Result (2026-11-11): a measured null at the declared operating point

Both arms completed inside the declared envelope: **256 steps each**, `work_items` 256, 1,152 MiB,
peaks **1,016,332,288 B** (constant) and **1,071,558,656 B** (cosine),
training 520.4 s / 519.9 s. The schedule is proved from the
runs: the constant arm records **1 distinct rate** (0.001) across
256 steps, the cosine arm **256 distinct rates**, strictly
non-increasing from 0.001 to 0.0001. Matched start holds: identical
initial weight bytes (`75da6eed0641f77c...`), seed
`20261026`, and an identical step 1 in `sample_id`, `loss`
(21.6618537903) and `gradient_norm`
(118.768249512).

| read | constant | cosine |
|---|---:|---:|
| pooled AUC | 0.8457 | **0.8864** |
| per-cell F1 at threshold `0.0` | 0.4307 | **0.4357** |
| exact nonempty boards (best threshold) | 0 of 28 | 1 of 28 |
| best swept F1 (threshold) | 0.4307 (0.0) | 0.5173 (1.95) |
| mean geometry error, cells | 3.0612 | 1.8409 |
| final weights | `5d52b3f14c71bda4...` | `28fcd85bbafe9d4f...` |

**Verdict: `null`.** The predeclared attribution read is per-cell F1 at the standing `0.0` point, and
it rose **+0.005041**, far below the **+0.05** floor. The prediction in this document is therefore
**falsified**: cosine decay does not move the standing decision point at this budget. Placement did
not regress - it improved **39.9%** - so the null is the attribution condition alone,
and the fit gate is unchanged at **0 of 28** exact nonempty boards in **both** arms.

**Recorded but not read by the verdict:** the cosine arm's *best swept* F1 is
**0.5173 at `1.95`** against the constant arm's
**0.4307 at `0.0`** (+0.0867), and
pooled AUC rose 0.8457 -> 0.8864. Like the earlier read-out levers, the schedule
moves the operating point and the ordering rather than the precision at the standing point; that
observation reopens nothing here and is not a second criterion.

Limits: one seed, one 256-step segment, synthetic training fit, committed lattice; no real-image,
complete-state or quality claim. The clip regime V-086 measured is shared by both arms, so this
pairwise contrast stands on its own.
