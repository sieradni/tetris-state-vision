# Occupancy detail stencil — declared 2x2 in-cell sample, matched continuation control

Status: predeclared (2026-11-09) and committed before any profile, study, sweep or evidence
directory for this instrument exists. The change, the control, the envelope, the gates, the floor,
the population and the work-item arithmetic below are fixed; the measured section is appended only
after the canonical receipts exist.

## Where this comes from

`M3_OCCUPANCY_RANKING.md` closes every threshold, conditioned-threshold and class-budget family and
ranks one step after the precision result ([M3_OCCUPANCY_PRECISION.md](M3_OCCUPANCY_PRECISION.md)):
*carry the demonstrated signal with more capacity — a 2x2 in-cell detail stencil per cell through the
same zero-initialized projection*. The precise wording of that entry is "four stride-2 samples
instead of one". That pass's declared mechanism moved as intended (precision at `0.0`
0.1752 -> 0.2363, cluster interval 0.0104-0.0939, recall also up, standing-point F1
0.2848 -> 0.3816) while AUC (0.8226) and the swept optimum (0.4120) stayed short, so the open
question is whether the precision signal is limited by *how much of the cell is sampled*.

This pass declares that one change and nothing broader. The untested stacking of the two read-out
taps (`occupancy_context` and `occupancy_detail`) is **not** part of it and remains open.

## The single declared change

`ModelConfig.occupancy_detail_stencil` (default `0`, published). It is a declared *form of the
detail tap*, not a new read-out: it requires `occupancy_detail == 1` and is refused otherwise. When
`1`, the tap reads the **four quarter-cell samples of a 2x2 in-cell grid** instead of the single
cell-center sample:

```
occupancy_features = cells + W_s . concat(s(u+-1/4, v+-1/4) for the four quadrant points),
W_s : 4 * widths[0] -> dimension,   W_s = 0 at step zero
```

The projection is the same module (`occupancy_detail_tap`), initialized to **zero** exactly as
before: 4 x 48 = 192 inputs to 192, i.e. 36,864 weights plus 192 bias, four times the single tap's
9,216 plus 192, and one grid sample of 4 x 260 = 1,040 points instead of 260. The stencil offsets
are a non-persistent buffer, so both arms carry identical state-dict keys apart from the projection
itself.

Three declared consequences, bound by `tests/test_detail_stencil.py`:

1. Each arm computes **exactly** the published function at step zero (`W_s = 0`), so the change is
   additive rather than a reinitialization.
2. Every shared tensor is drawn from the same seeded initialization and loaded from the same
   checkpoint bytes; the arms differ only by the stencil form of the tap.
3. Relabelling the hidden sample order (permuting the four stencil points *and* their projection
   input blocks together) leaves the image-supported occupancy output unchanged, so a hidden state
   the pixels cannot distinguish cannot manufacture a different decision.

`occupancy_context` stays `0` in both arms: this pass declares one change, not a combination.

One declared consequence of the entry taken literally: the stencil *replaces* the center sample, it
does not add to it. The control therefore keeps the center sample the previous pass demonstrated,
and the measured contrast asks whether the four quadrant appearances carry that signal further
than the center alone does. If precision falls, the measured outcome is that the center sample
carried more than the quadrants; that is a result, not a defect.

## Matched pair, one budget, one catalogue

Both arms are **continuations** of the committed published checkpoint
`runs/m3-percell-balance-r1` (weights `9ae1fef9f649...`) under the same captured source catalogue:

| arm | `occupancy_detail` | `occupancy_detail_stencil` | everything else |
|---|---|---|---|
| lever | `1` | `1` | identical |
| control | `1` | `0` | identical |

Identical means the same initial weights, `seed 20261026`, `learning_rate .001`, curriculum
`stream-v1`, capture side `512`, component capacity `64` with the positional identity read-out and
length endpoint, occupancy loss factor `4` and positive weight `4`, **64 further steps**, the same
image order and the same envelope. Both arms are fresh continuations under this pass's own
catalogue, so the pair is matched by the repo's own source-identity standard.

Because both arms carry the declared detail tap and differ only in its stencil form, the measured
contrast is the **increment of the extra in-cell capacity over the demonstrated tap**, not "detail
versus no detail". The previous pass's arms (catalogue `27a0e917`) are reported as an unmatched
**reference** only: they cannot enter this verdict.

Envelope per arm, declared before admission: **1,152 MiB** aggregate commit, **64 steps**,
**1,200 s** runner wall / 900 s trainer wall, one heavy job at a time. The stencil adds 27,648
parameters over the single tap, so the ceiling is not expected to move; if it does, the outcome is
"not admissible as declared" and nothing is relaxed.

## Declared work-item arithmetic

Declared here because a prior pass declared 36 while performing 54 and its receipt was refused:

| launch | work items | what is counted |
|---|---|---|
| each profile | 2 | 2 optimizer updates |
| each study | 64 | 64 optimizer updates (`steps`); evaluations are not work items |
| each sweep profile | 2 | 2 forwards |
| each measured sweep | **54** | 54 forwards = 36 deciding + 18 development |
| curation | **4** | four receipts read (two studies, two sweeps) |

## Scoring, on the standing widened gates

Each arm is scored read-only on the **widened deciding population** (training, canonical twelve
groups / 36 images / 28 nonempty boards / 667 occupied and 4,933 empty known cells) with the
standing 81-point grid on `[-6, 6]` step 0.15, under its own 640 MiB sweep profile. Population, grid
and thresholds are unchanged from the neighborhood and detail passes, so the numbers sit beside the
standing table.

**The lever CLEARS iff all three hold** (identical to the two preceding passes):

1. **Per-cell gate** — occupied-cell F1 at the replay's operating point (`0.0`) `>= 0.70`.
2. **Ranking gate** — `sweep_verdict` is `informative` (pooled AUC `>= 0.90` **and** some swept
   threshold reaches F1 `>= 0.70`).
3. **Attribution floor** — lever best swept F1 exceeds control best swept F1 by **`>= 0.05`**.

**Otherwise NOT CLEARED**, with the failing condition named mechanically. If either arm's study or
sweep cannot complete inside the declared envelope, the verdict is **NOT ADMISSIBLE AS DECLARED**,
quoting the receipt; no ceiling is raised, no gate widened, no population shrunk, no run repeated
until it passes.

Additional declared **mechanism quantity**, which cannot move the verdict: cell-level **precision**
at the standing point (`0.0`). `precision_improved` iff the lever exceeds the control by `>= 0.05`
**and** the whole-ancestry-cluster interval on that delta excludes zero. Precision, recall, AUC and
F1 deltas are otherwise reported as costs.

Intervals: Hanley-McNeil beside each pooled AUC, Wilson beside each precision and recall, and the
whole-ancestry-cluster paired bootstrap (400 draws, seed `20261026`, twelve clusters) for both arms
and every delta. No gate reads an interval.

## Receipts this pass must produce

1. `runs/m3-occdstencil-lever-r1-profile-r1-job.json` and
   `runs/m3-occdstencil-control-r1-profile-r1-job.json`, 2 steps, 1,152 MiB / 300 s / 2 work items.
2. `runs/m3-occdstencil-lever-r1-job.json` and `runs/m3-occdstencil-control-r1-job.json`, 64 steps,
   1,152 MiB / 1,200 s / 64 work items, each admitted only by its own profile.
3. `...-sweep-r1-profile-r1` (2 forwards) and `...-sweep-r1` (54 forwards) per arm, 640 MiB.
4. Curation `runs/m3-occdstencil-curation-r1-job.json` into
   `docs/evidence/m3-occupancy-stencil-r1/scope.json`, re-deriving every published number, with the
   deciding arithmetic (36 cases, 28 nonempty boards, 667/4,933 cells, 12 clusters) asserted before
   writing.
5. The owned family `m3-occdstencil-*` declared in `record_stream_run.OWNED_RUN_PATTERN` and bound
   by a test.

Any launch that dies to the platform's 295 s command cap or leaves its declared arithmetic is
preserved under `runs/attempts/` with a written defect note and re-recorded canonically; no ceiling,
gate, population or floor moves.

## What each outcome licenses

- **Cleared** — extra in-cell capacity is the precision lever at this budget; the occupancy
  question moves to scale, and the next step is a longer matched budget on this pair.
- **Not cleared, precision improved** — more in-cell sampling carries the demonstrated gain further
  without reaching the bar; the next step moves upstream to encoder resolution.
- **Not cleared, precision flat or worse** — the center sample was already the better in-cell
  statistic; extra quadrant samples do not add separation, and the next step is encoder resolution
  rather than more read-out detail. If precision is *worse*, the reference comparison against the
  prior pass must say so explicitly.
- **Not admissible** — the constraint itself is the measured result.

## Measured result: NOT CLEARED — extra in-cell capacity bought ordering and the swept optimum, not standing-point precision

Receipts: profiles `runs/m3-occdstencil-lever-r1-profile-r1-job.json` (passed, 55.5 s,
1,039,421,440 B) and `runs/m3-occdstencil-control-r1-profile-r1-job.json` (passed, 54.6 s,
1,037,393,920 B); studies `runs/m3-occdstencil-lever-r1-job.json` (passed, 64/64, 202.8 s,
1,032,060,928 B) and `runs/m3-occdstencil-control-r1-job.json` (passed, 64/64, 218.5 s,
1,031,536,640 B), both inside 1,152 MiB, under one captured catalogue (`6ff287a1`) and from
`9ae1fef9f649...`; both sweeps 54 forwards under 640 MiB (peak 562,356,224 / 561,889,280 B);
curation `runs/m3-occdstencil-curation-r1-job.json` into
`docs/evidence/m3-occupancy-stencil-r1/scope.json`.

| widened deciding population | stencil lever | single-sample control | prior single tap (reference) |
|---|---|---|---|
| pooled AUC | **0.8804** (CI 0.8633-0.8975) | 0.8226 (CI 0.8027-0.8426) | 0.8226 |
| precision at `0.0` | 0.2474 | 0.2363 | 0.2363 |
| recall at `0.0` | 0.9805 | **0.9910** | 0.9910 |
| occupied-cell F1 at `0.0` | **0.3952** | 0.3816 | 0.3816 |
| best swept F1 (threshold) | **0.4913** (1.80) | 0.4120 (0.90) | 0.4120 (0.90) |
| exact nonempty boards | 0/28 | 0/28 | 0/28 |
| ranking verdict | `uninformative` | `uninformative` | `uninformative` |

Deltas (lever minus control): AUC **+0.0578** (cluster 0.0385-0.0943); best swept F1 **+0.0793**;
occupied-cell F1 at `0.0` +0.0135 (cluster 0.0057-0.0273); precision +0.0111 (cluster
0.0047-0.0210); recall -0.0105.

**Verdict: NOT CLEARED**, failing `per_cell_f1` (0.3952 against 0.70) and `ranking`
(`uninformative`: AUC 0.020 short of 0.90, best swept F1 0.209 short of 0.70). The **attribution
condition is met**: the stencil's swept optimum exceeds the single-sample control's by 0.0793 against
the declared +0.05 floor, the first arm of this campaign to clear it (neighborhood -0.0138, detail
tap -0.0217).

**Mechanism: `precision_improved = false`.** The declared floor was 0.05 and the measured precision
delta is +0.0111, so the mechanism quantity missed its own bar even though its twelve-cluster
interval excludes zero (0.0047-0.0210). Extra in-cell capacity therefore did **not** carry the
precision gain further: it bought ordering instead.

What the numbers do say: the stencil is the best-ordering arm of the campaign - AUC **0.8804**, 0.020
short of the 0.90 bar, with an interval upper bound of 0.8975, the closest any arm has come - and its
swept optimum **0.4913** is the highest measured here (0.4120 single tap, 0.4338 prior no-tap
control, 0.4200 neighborhood, 0.3704 published). Its standing decision pays for that ordering with
recall (0.9805 against 0.9910) while precision improves only marginally, the same shape every
read-out lever has shown: information rises, separation does not.

**Separation, not ordering, is now the binding constraint.** The AUC deficit fell from 0.076
(neighborhood) and 0.077 (precision tap) to **0.020**, while the swept-F1 deficit has stayed at
0.209-0.288 throughout and no board is exact. The read-out can rank cells; the ranking at this budget
is not sharp enough to separate them.

Reproducibility note, recorded rather than used: the single-sample control reproduces the previous
pass's *lever* exactly - pooled AUC 0.8226433610683003, operating counts 661/2136/6, best F1
0.41203400915631133 at `0.90` - under a different catalogue hash. The captured pixels and the matched
continuation are therefore deterministic across passes, and this pass's contrast is against exactly
the tap the previous pass demonstrated rather than an approximation of it.

**Next-ranked step, in priority order:** (1) move upstream to encoder resolution - a finer native tap
than stride-2 - because every read-out lever has now converted extra capacity into ordering without
separation, so more read-out information is the wrong currency; (2) scale the matched budget on
whichever of the two carries the gain. Threshold, conditioned-threshold and class-budget families stay
closed, and stacking `occupancy_context` with `occupancy_detail` remains untested and out of scope.

`tests/test_detail_stencil.py` re-derives both arms, the twelve-cluster paired bootstrap, the
population, the work arithmetic, the verdict, the mechanism flag, the unmatched reference and the
documents' quoted gaps from the committed receipts.
