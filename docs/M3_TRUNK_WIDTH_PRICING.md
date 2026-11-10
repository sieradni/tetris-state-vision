# Pricing a wider trunk: what option A would actually cost

Status: **pricing measurement only.** This document changes nothing about the model. It builds scratch copies of
the network at the published widths and at two wider declared widths, runs the trainer's real step at the real
512-pixel input, and reports the cost. No training run, checkpoint, published default or lever-table entry is
touched, and nothing here is comparable to any occupancy result.

The fork it prices is the one `docs/M3_OCCUPANCY_TRUNK_WIDTH.md` left open: authorize a declared trunk-width
lever, or adopt the standing limit that the frozen architecture cannot pass the fit gate.

## What was measured

`docs/evidence/m3-trunk-width-pricing/price_trunk_width.py` reproduces one streaming step exactly as `scripts/train_m2_stream.py` does it:
fresh curriculum case for seed `20261026`, `prepare_image` at `512` pixels,
`targets_for` with capacity `64`, forward, `joint_loss` with the trained endpoint term and
occupancy weights, backward, gradient clip, Adam step. Each trunk gets `1` warm-up step
and `3` timed steps; the median is reported.

It ran through the bounded runner inside a declared **2560 MiB / 600 s**
job (`docs/evidence/m3-trunk-width-pricing/job.json`, exit 0, peak 1,442,033,664 B, 26.6 s). A first attempt under
the settled **1,152 MiB** study ceiling died mid-run: peak **1,229,508,608 B**, i.e. **21,549,056 B over
the ceiling**, at 15.7 s. That is recorded, not hidden
(`docs/evidence/m3-trunk-width-pricing/attempt-1152mib-job.json`): a wider trunk does not fit the resource profile the existing studies use.

## Measured cost

| trunk | widths / dimension | parameters | tensors | model step (s) | ratio | derived whole step (s) | steps inside 1800 s |
|---|---|---|---|---|---|---|---|
| published | `[48, 96, 192, 256]` / `192` | 8,018,505 | 271 | 1.3425 | 1.000 | 4.0867 | 440 |
| wide A | `[64, 128, 224, 256]` / `192` | 8,844,777 | 271 | 1.7082 | 1.272 | 5.2000 | 346 |
| wide B | `[96, 192, 256, 256]` / `240` | 12,870,089 | 271 | 2.5867 | 1.927 | 7.8742 | 228 |

The derived whole step scales the *trainer's* measured per-step cost at capacity 64
(1046.2 s / 256 steps =
4.0867 s) by the measured model-step ratio, so it includes the
curriculum generation and bookkeeping the pricing script does not time. It assumes that non-model share is
unchanged, which makes it a conservative estimate for the wider arms rather than an optimistic one.

## The honest arithmetic of option A

The capacity-64 arms actually consumed **256** steps (1046.2 s)
and **384** steps (1522.4 s). Against those:

- **wide A fits 346 steps.** That is
  above the 256-step baseline and
  below the 384-step budget lever
  (90% of it), so a wider trunk at these widths could receive a
  **comparable** single-job budget at the baseline and a **90%-of-lever** budget.
- **wide B fits 228 steps.** That is
  below the 256-step baseline
  (89% of it), so at this scale the fit budget is **not comparable**
  and the arm would be measuring a different question.
- Both wider trunks **exceed the settled 1,152 MiB ceiling** (21,549,056 B over on the first attempt, peak
  1,442,033,664 B at the larger ceiling), so a wider study needs its own declared resource profile before any
  predeclaration can name an envelope.

The finding in one line: **a moderately wider trunk is affordable at a comparable budget; the widest priced trunk
is not, and neither fits the resource profile the existing studies were admitted under.**

## What authorizing option A would require, contractually

Not an arm bolted onto the current table - a new line of evidence:

1. **A new architecture identity.** The checkpoint, `ModelConfig` defaults and the `TetrisStateNet-M3` identity
   stay exactly as they are; the wider network is a second declared identity with its own width fields.
2. **A width knob in the config**, defaulting to the published widths so no existing artifact changes meaning.
3. **A fresh baseline trained at that identity**, not the curated capacity-64 control. The existing single-control
   reconciliation table cannot host this row; the wider model's numbers are not comparable to it, and the brief
   says so in the record rather than letting a reader assume otherwise.
4. **A predeclaration committed before any receipt**, carrying: the exact widths and dimension, the parameter and
   tensor delta against the published identity, a matched-start policy for *shape-changed* tensors (the published
   checkpoint cannot be loaded into it), a step budget derived from the measured per-step cost above rather than
   carried over from capacity 64, its own memory ceiling and wall cap, the same fit-gate criterion (3 of 28 exact
   nonempty boards at threshold 0.0), an activation floor on the decision-flip rate, and falsifiers.
5. **Its own resource profile**, because the settled 1,152 MiB ceiling already failed at these widths.

## What adopting option B means

The frozen architecture keeps its identity, checkpoint and every existing result. The consequence is concrete:
the fit gate stays FALSE at 0 of 28, the eight spent levers stay refuted, and the capacity branch stays
inexpressible. No further occupancy lever is rankable under this architecture, so subsequent work has to be about
something else - real-image labels and corpus admission, runtime and product scope, or accepting that the
occupancy target is unreachable at this scale - rather than more variants of the same model.

## The decision

**This choice is the user's, and the two options cost different things.** Option A buys the only untested
representational question left, at the price of a new identity, a fresh baseline, a new resource profile and a
record that is explicitly incomparable to the current table; the measurement above says a moderate widening is
affordable at a comparable budget and the widest is not. Option B buys a clean, honest stopping point for the
occupancy campaign, at the price of leaving the fit gate FALSE with no ranked suspect left.

Limits unchanged either way: synthetic training fit, one seed, committed lattice, no real-image, generalization or
complete-state claim, and no change to the default configuration, the published checkpoint or the lever table.
