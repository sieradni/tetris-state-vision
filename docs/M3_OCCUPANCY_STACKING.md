# Occupancy tap stacking — declared additive composition, matched single-tap control

Status: predeclared (2026-11-09) and committed before any profile, study, sweep or evidence directory
for this instrument exists. The change, the control, the envelope, the gates, the floor, the
population and the work-item arithmetic below are fixed; the measured section is appended only after
the canonical receipts exist.

## Where this comes from

The refusal pass ([M3_OCCUPANCY_RESOLUTION.md](M3_OCCUPANCY_RESOLUTION.md)) ranked this lever next
because it is the last of the read-out levers that changes *what* the occupancy head sees rather than
*how finely* it samples: the 8 x 8 context residual ([M3_OCCUPANCY_REPRESENTATION.md](M3_OCCUPANCY_REPRESENTATION.md),
AUC 0.8244, best swept F1 0.4200) and the center-sample detail tap
([M3_OCCUPANCY_PRECISION.md](M3_OCCUPANCY_PRECISION.md), AUC 0.8226), and then the 2x2 stencil form of
the detail tap ([M3_OCCUPANCY_STENCIL.md](M3_OCCUPANCY_STENCIL.md), AUC 0.8804, best swept F1 0.4913,
attribution floor met). Every one of those passes held the other tap at `0` by declaration. Stacking
was never tested because the code could not express it.

## The single declared change

`ModelConfig.occupancy_context` and `ModelConfig.occupancy_detail` both exist and both default to `0`;
this pass adds **no new configuration knob and no new parameters**. It fixes the composition of the
two declared residuals, which today is not additive:

```
before:  occupancy = cells + block(cells)                    # context path overwrote the detail tap
after:   occupancy = cells + W_d . sample + block(cells)     # both declared residuals summed
```

`W_d` is the existing 48 -> 192 detail projection and `block` the existing 3x3 + 1x1 residual with a
zero-initialized output projection; both are exactly as published, and neither is retrained from
scratch. Three consequences, bound by `tests/test_occupancy_stack.py`:

1. **Additive, not overwriting** — with both taps trained, the stacked arm's logits differ from either
   tap alone, and with one projection zeroed it reproduces that tap's arm exactly.
2. **Identity at step zero** — both projections are zero-initialized, so the stacked arm computes
   exactly the published function at step zero, as does the control.
3. **Byte-identical when either tap is off** — the composition is a sum, so every prior single-tap arm
   keeps the function it had; the change cannot retroactively move an existing result.

Parameters reused, not added: the stack is 0.35 M parameters of context block (331,776 in the 3x3
convolution plus 36,864 in the 1x1) against the control's none, so the declared activation census is
two 26 x 10 x 192 maps plus the block's weights, gradients and Adam state - about 7 MB against 168 MiB
of headroom under the 1,152 MiB ceiling.

## Matched pair, one budget, one catalogue

Both arms are **continuations** of the committed published checkpoint `runs/m3-percell-balance-r1`
(weights `9ae1fef9f649...`) under the same captured source catalogue:

| arm | `occupancy_detail` | `occupancy_context` | `occupancy_detail_stencil` | `occupancy_resolution` |
|---|---|---|---|---|
| lever | `1` | `1` | `0` | `0` |
| control | `1` | `0` | `0` | `0` |

Everything else identical: same initial weights, `seed 20261026`, `learning_rate .001`, curriculum
`stream-v1`, capture side `512`, component capacity `64` with the positional identity read-out and
length endpoint, occupancy loss factor `4` and positive weight `4`, **64 further steps**, the same
image order, gates, floor and population. The single intended difference is the context residual,
i.e. the stacking.

The control is the same configuration as the stencil pass's control, so its numbers also serve as an
independent repetition check of that baseline; the stencil arm (AUC 0.8804) and the context arm
(AUC 0.8244) are reported as unmatched **references** and cannot enter this verdict.

Envelope per arm, declared before admission: **1,152 MiB** aggregate commit, **64 steps**, **1,200 s**
runner wall / 900 s trainer wall, one heavy job at a time. **Declared refusal rule:** if either arm
cannot be admitted inside 1,152 MiB at the matched budget, the outcome is **NOT ADMISSIBLE AS
DECLARED**, recorded with the measured peak and the receipt; no ceiling is raised, no variant is
substituted, and no run is repeated to pass. This is the rule the resolution pass applied at
23,920,640 B over its ceiling.

## Declared work-item arithmetic

Declared here because V-061 exists for exactly this: a receipt must certify the work it performed.

| launch | work items | what is counted |
|---|---|---|
| each profile | 2 | 2 optimizer updates |
| each study | 64 | 64 optimizer updates (`steps`); evaluations are not work items |
| each sweep profile | 2 | 2 forwards |
| each measured sweep | **54** | 54 forwards = 36 deciding + 18 development |
| curation | **4** | four receipts read (two studies, two sweeps) |

The launch, the receipt and the curator are bound to these numbers: the curator refuses unless
`job.work_items == result.forwards == 54`, the deciding population is exactly 36 images / 28 nonempty
boards / 667 occupied / 4,933 empty cells / 12 ancestry clusters, and the sweep evaluated 12 groups on
81 thresholds including `0.0`.

## Scoring, on the standing widened gates

Each arm is scored read-only on the **widened deciding population** (training, canonical twelve
groups / 36 images / 28 nonempty boards / 667 occupied and 4,933 empty known cells) with the standing
81-point grid on `[-6, 6]` step 0.15, under its own 640 MiB sweep profile. Population, grid and
thresholds are unchanged from the neighborhood, detail, stencil and resolution passes.

**The lever CLEARS iff all three hold:**

1. **Per-cell gate** — occupied-cell F1 at the replay's operating point (`0.0`) `>= 0.70`.
2. **Ranking gate** — `sweep_verdict` is `informative` (pooled AUC `>= 0.90` **and** some swept
   threshold reaches F1 `>= 0.70`).
3. **Attribution floor** — lever best swept F1 exceeds control best swept F1 by **`>= 0.05`**.

**Otherwise NOT CLEARED**, with the failing condition named mechanically.

Additional declared **mechanism quantity**, which cannot move the verdict: cell-level **precision** at
the standing point (`0.0`). `precision_improved` iff the lever exceeds the control by `>= 0.05` **and**
the whole-ancestry-cluster interval on that delta excludes zero.

Intervals: Hanley-McNeil beside each pooled AUC, Wilson beside each precision and recall, and the
whole-ancestry-cluster paired bootstrap (400 draws, seed `20261026`, twelve clusters). No gate reads
an interval.

## Receipts this pass must produce

1. `runs/m3-occstack-lever-r1-profile-r1-job.json` and
   `runs/m3-occstack-control-r1-profile-r1-job.json`, 2 steps, 1,152 MiB / 300 s / 2 work items.
2. `runs/m3-occstack-lever-r1-job.json` and `runs/m3-occstack-control-r1-job.json`, 64 steps,
   1,152 MiB / 1,200 s / 64 work items, each admitted only by its own profile.
3. `...-sweep-r1-profile-r1` (2 forwards) and `...-sweep-r1` (54 forwards) per arm, 640 MiB.
4. Curation `runs/m3-occstack-curation-r1-job.json` into
   `docs/evidence/m3-occupancy-stacking-r1/scope.json`, re-deriving every published number, with the
   deciding arithmetic asserted before writing.
5. The owned family `m3-occstack-*` declared in `record_stream_run.OWNED_RUN_PATTERN` and bound by a test.

Any launch that dies to a harness limit leaves its attempt under `runs/attempts/` with a written note
and is re-recorded under a fresh number; no ceiling, gate, population or floor moves.

## What each outcome licenses

- **Cleared** — the two taps are complementary and the occupancy question moves to scale.
- **Not cleared, precision improved** — stacking carries precision but not separation; scale the
  matched budget on this pair.
- **Not cleared, precision flat or worse, but attribution met** — stacking buys ordering only, like
  every other read-out lever, and the remaining question is the target or the data rather than the
  read-out; scaling a lever that only orders would be declared against that evidence.
- **Not cleared with attribution below the floor** — the taps are redundant at this budget; no further
  combination of them is ranked, and the next step leaves the read-out.
- **Not admissible** — the constraint itself is the measured result, recorded with its peak.
