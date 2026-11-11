# A replicate study of the two compute-side levers (predeclaration)

Status: **predeclared before any receipt of this study exists.** Nothing here changes the published
configuration, the checkpoint, the fit gate or any earlier verdict. It is a fresh-baseline block and
its numbers are **not comparable** to the capacity-64 continuation table.

## The question, and why a fresh baseline is the only honest frame

[M3_OPERATING_POINT.md](M3_OPERATING_POINT.md) found that two compute-side refusals reverse under a
boundary chosen out of fold - the 1.5x fit budget (+0.149, interval +0.006..+0.277) and the cosine
schedule (+0.084, +0.010..+0.168) - but both rest on **one trajectory each**, and both interval lower
bounds sit on zero. The campaign's rank order still rests on that closure. This study asks whether
the effect survives replication.

**Seeding is not a label here, it is the data.** `m2_stream_data.sample_case` derives each scene from
`scene_seed = seed + group_index` (plus `10_000_000` on the development split), so a different seed is
a **different synthetic population** - different boards, layouts, piece queues, skins and capture
geometry - not a permutation of the published one. And the trainer pins the seed for continuations:
`train_m2_stream` refuses `--init-weights-from` unless the source summary's seed equals `--seed`
(`"initial-weight source architecture/config/curriculum/seed differs"`). A continuation arm is
therefore deterministic given its start bytes, and **pseudo-replication of the audit's exact arms is
impossible**. Replication at this repository's constraints means fresh-baseline arms, one population
per seed, paired within seed.

That is a real limitation of the frame, stated rather than hidden: a fresh start is weaker than the
curated `9ae1fef9f649...` start, and the width pass already measured a fresh 256-step control at AUC
0.7815 against the curated 0.8255. The study answers "do the compute levers help from a fresh start,
consistently across populations", which is a narrower question than the continuation table asks.

## The block

Three seeds, three arms each, all fresh:

| field | value |
|---|---|
| seeds | `20261101`, `20261102`, `20261103` (three populations, none of them the published one) |
| `base128` | 128 steps, constant `lr 1e-3` - the block's control |
| `budget192` | 192 steps, constant `lr 1e-3` - the **1.5x budget** rung (192/128), same ratio as 384/256 |
| `cosine128` | 128 steps, cosine `1e-3 -> 1e-4` - the **schedule** rung |
| run names | `m3-stream-repl-<arm>-s<seed>` |
| common | `--model accuracy --component-capacity 64 --component-identity-factor 4 --component-identity-readout positional --component-endpoint-term length --occupancy-factor 4 --occupancy-positive-weight 4 --curriculum stream-v1 --capture-max-side 512 --evaluate-every 64 --checkpoint-every 32 --development-groups 6 --wall-seconds 1800` |
| initial state | seeded random, no `--init-weights-from` |

The single declared difference within a seed is the arm: 128 vs 192 steps, or the `--lr-final`
schedule. Every seed runs all three arms, so the block is a paired factorial with seed as the block.

**Why 128 and not 256:** the block must fit three bounded parent jobs under the runner's 3,600 s cap,
and nine arms at the settled 256/384 budget project to roughly 2.9 hours. 128/192 preserves the
lever's ratio at half the budget. This is a declared cost-driven choice, and it is the study's main
limitation: a 64-step budget difference is a smaller absolute walk than 128, so a null here does not
prove the 256-vs-384 contrast null, and a confirmed result here does not by itself confirm +0.149.

## Envelope, stated before launch

Ceiling **2,048 MiB** on every arm, identical across arms and seeds so a memory difference cannot be
read as an arm effect (fresh 256-step arms peaked 991,731,712 B and 1,031,798,784 B at 2,048 MiB).
Internal wall 1,800 s, job wall 2,400 s, work items = steps. Sweeps at the standing 640 MiB with 54
cases, admission profile first. Parent driver jobs at 256 MiB / 3,600 s, one seed per parent, one
heavy job at a time. Expected footprint: under 1.1 GB per arm, well below the ceiling; the pilot and
each admission profile measure it rather than assuming it.

## The primary read, the floor and the rule

**Primary:** the out-of-fold F1 of [M3_OPERATING_POINT.md](M3_OPERATING_POINT.md) - the same
81-point grid, the same F1 criterion and tie-break, the threshold chosen on the other eleven ancestry
groups of that seed's own deciding population - computed per arm, pooled across the block's 36
`(seed, group)` clusters. Every sweep receipt must first reproduce its own published `deciding.auc`
and `threshold 0.0` F1, exactly as the audit requires; a receipt that does not is refused.

**Floor: +0.05**, the campaign's standing attribution floor, kept rather than tuned. It is
resolvable at this design: the audit's paired interval was about `+/-0.15` wide at 12 clusters, and
36 clusters shrink the half-width by roughly `sqrt(3)`, so `+0.05` is about 0.6 half-widths.

**Interval:** a paired whole-group bootstrap over the 36 clusters (1,000 draws, seed `20261026`), one
cluster draw shared by all three arms, so within-cluster variation cancels in the difference.

**Rule, fixed before any receipt** (implemented once, in `compute_replication_verdict`):

- **`floor_limited`** - any seed's control arm has out-of-fold AUC below **0.70**. The block cannot
  speak about the levers at that population and the study reports this instead of a verdict.
- **`confirmed`** - the block's point delta is `>= +0.05` **and** the paired 95% interval's lower
  bound is `> 0` **and** all three per-seed deltas are `>= 0`. All three conditions, no exceptions.
- **`unconfirmed`** - otherwise. The reason field names which condition failed (point below floor,
  interval touching zero, or a seed reversing), and no alternative read, threshold, arm or budget is
  substituted afterwards.

Both levers are read by this rule independently, and the two answers are reported side by side. A
`confirmed` on one and `unconfirmed` on the other is a real outcome, not a partial success to be
re-read.

## Falsifiers

- Any arm short of its declared steps, a failed receipt, a differing board population across the
  three arms of a seed, or a receipt that does not reproduce its published read -> that seed is
  **unreadable** and contributes no cluster; the block is reported short and no verdict is claimed.
- A control below the AUC floor -> `floor_limited`, recorded, no lever claim.
- Both levers `unconfirmed` -> the compute closure is **retired but unconfirmed**: the audit's
  reversal remains an unresolved alternative explanation, and the width verdict stands as the
  remaining actionable lever.
- Either lever `confirmed` -> compute is the ranked lever, and the "capacity is the limit" framing is
  qualified in ROADMAP and the continuation state.

## What this is not

Not a continuation of the capacity-64 table, not a fit-gate reopening, not a quality, real-image or
complete-state claim, and not a licence to re-read a threshold or a population after the numbers are
in. One block, three seeds, two levers, twelve boards per seed.
