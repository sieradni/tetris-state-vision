# A replicate study of the two compute-side levers (predeclaration)

Status: **read - both levers `unconfirmed`** (result at the end; the predeclaration above it is
unchanged). Nothing here changes the published configuration, the checkpoint, the fit gate or any
earlier verdict. It is a fresh-baseline block and its numbers are **not comparable** to the
capacity-64 continuation table.

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

## Operations note: an interrupted runner, recorded rather than hidden (V-093)

The block's first launch completed seed `20261101` (parent receipt `passed`, 1,296.3 s) and was cut
by a **host runner restart** during seed `20261102`. The restart killed the tool call, not the OS
process: the detached chain kept running, finished that seed's `cosine128` arm (304.1 s) and all
three of its sweeps, and wrote `runs/compute-replication-s20261102.json` at 13:00:20 - after a
replacement chain had already been started on the same seed. That replacement reused the two
verified arms and then failed with `PermissionError [WinError 5]` when its `--resume` tried to move
the still-live `cosine128` directory aside; the failed receipt is preserved as
`runs/attempts/m3-repl-restart/resume-s20261102-refused-job.json`. The replacement chain had also
begun seed `20261103`, so the surviving chain's seed-3 step collided with those profile directories
and exited 1 (preserved as `seed3-collision-job.json` with a note). Seed `20261103` was then run to
completion by one chain with `--resume`, which reused its three verified profiles (45.7-45.8 s each),
preserved the partial arm, and finished at 13:18:51.

Two chains against one seed's outputs is the defect: nothing in the driver knew another block for
that seed was alive. The standing instrument is a per-seed lock holding the holder's pid
(`acquire_lock` in the driver): a live holder refuses a second block, a dead holder is replaced
rather than trusted, so a crashed block stays resumable. No arm, seed, budget, ceiling, primary,
floor or rule above changed, and every reused step is recorded in its seed's manifest.

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

## Result: both levers `unconfirmed`, so the compute closure is retired rather than confirmed

Read: `docs/evidence/m3-compute-replication-r1/result.json`, written by
`runs/m3-replread-r1-job.json` (passed, 29.4 s, peak 40,108,032 B of 256 MiB). All nine arms completed
their declared steps with matched starts within each seed (one initial-weights digest and one first
board per seed, three different digests and boards across seeds), and all nine sweeps reproduced
their own published `deciding.auc` and `threshold 0.0` F1 before contributing. No seed was
unreadable and no cluster was dropped: 36 `(seed, group)` clusters, 12 per seed.

| seed | control AUC | `base128` F1 | `budget192` F1 | `cosine128` F1 |
|---|---|---|---|---|
| `20261101` | 0.800414 | 0.400447 | 0.370578 | 0.463932 |
| `20261102` | 0.826166 | 0.312561 | 0.321326 | 0.417742 |
| `20261103` | 0.917477 | 0.673421 | 0.416493 | 0.552129 |
| pooled (36 clusters) | - | **0.443717** | **0.364118** | **0.464992** |

**The 1.5x fit budget: `-0.0796`, interval `-0.1380 .. -0.0250`, `unconfirmed`.** Per seed `-0.0299`,
`+0.0088`, `-0.2569`; the reason string is `point_below_floor+interval_touches_zero+seed_reverses`,
and seed `20261103` carries the reversal. This is not a null: the interval excludes zero **below**
the control, so at this scale and start the extra 64 steps cost pooled F1.

**The cosine schedule: `+0.0213`, interval `-0.0721 .. +0.1136`, `unconfirmed`.** Per seed `+0.0635`,
`+0.1052`, `-0.1213`, the same reason string; two seeds point the audit's way and one reverses hard
enough to pull the interval across zero. It is the better of the two levers and still short of the
floor at four times the audit's cluster count.

Controls clear the AUC floor everywhere (0.800 / 0.826 / 0.917), so `floor_limited` never applied.
The spread is itself the block's most robust observation: control quality moves 0.12 AUC across
populations while both levers move pooled F1 by less than 0.08, so population is a first-order axis
and a single-trajectory delta is not a measurement of a lever. The `20261103` population is also
where `budget192` loses most (`-0.257`), i.e. the extra budget hurt most where the control was
already strongest - a hypothesis worth one predeclared arm later, not a finding from one seed.

**Under the predeclared rule this reads as `unconfirmed` for both levers, therefore the compute
closure is retired but unconfirmed:** "budget is not the limiter" and "more optimisation is not a
substitute" stay retracted to the extent that the audit's reversal is now an unresolved alternative
explanation rather than a live lever, and no further compute arm is ranked on the strength of it. The
width `capacity_null` stands as the remaining actionable verdict. Nothing here is re-read: no
alternative threshold, arm, budget or population is substituted, and the `-0.0796` is reported at the
same floor, interval and rule that were fixed before the first receipt.

**What this does not settle, stated plainly:** the audit measured `+0.149` on continuations of the
curated start at 256 -> 384 steps; this block measures fresh starts at 128 -> 192. The block cannot
arbitrate the continuation contrast (pseudo-replication of those arms is impossible, see above), so
the correct object is: *the compute lever has no replicated support at any scale tested here, and at
the scale that could be replicated it points down.* Confirming or refuting the audit's `+0.149`
itself would need a 256/384 block at three or more seeds under this same rule - roughly 2.9 hours of
arms by the cost model above - and that is a decision for the next pass, not a re-read of this one.
