# M3 identity supervision exposure

Status: measured on the completed identity-factor run. No new lever is opened here.

`docs/M3_COMPONENT_IDENTITY.md` and `docs/M3_IDENTITY_READOUT.md` closed both
identity-side levers: the loss scale moves identity at a measured structural cost,
and the read-out tying moves it while breaking the declared structure-held
condition. Both leave the same open question, recorded as "the supervision
signal", with a single number attached: 54 queue tokens and 9 hold cases per pass
against roughly 1,800 supervised cells. That number says volume is small; it does
not say whether the classes are covered, how the labels are distributed, or which
labels the model actually gets right. This pass measures those before any new
lever is declared, so the next choice is made from counts rather than from the
aggregate accuracy.

## Method

Model-free and read-only. For each of the 27 recorded probes of
`m3-identity-factor-r1` (9 training, 18 development), the audit replays the
recorded recipe from the run's archived code, refuses unless the rebuilt recipe,
raster bytes and annotation digest equal the saved ones, rebuilds
`targets_for(..., component_capacity=64)`, and tallies the masked queue/hold
identity labels by class and slot index. It then joins the published per-case
diagnostics
(`docs/evidence/m3-identity-factor-r1-errors-r3/result.json`) so that
"supervised" and "predicted correctly" are the same labels. Nothing is trained,
inferred, re-sampled or changed: no capacity, mask, target, order, loss or
sampling edit, and no model forward at all.

Every published count is reproduced before a result is written: 54 and 90
supervised queue tokens, 9 and 12 hold labels, 3 and 9 correct queue tokens, 0 and
3 correct holds. A disagreement refuses the run instead of reporting a number.

## Receipts

| Receipt | Status | Numbers |
|---|---|---|
| `runs/m3-identity-exposure-r4.json` | passed | 27 cases, 3.248 s, 266,608,640 B under 640 MiB, work items 27 |
| `runs/m3-identity-exposure-r4-curation.json` | passed | curator re-derives every count from the per-case rows, 27 items, 128 MiB |

Curated scope: `docs/evidence/m3-identity-exposure-r4/scope.json` (`passed`), with
`result.json`, both runner logs and three retained attempt receipts.

The chain's first three attempts are retained rather than discarded, because each
failed in the audit tooling and not in the measurement: `r1` stopped in the new
summarizer on a tuple/dict shape error before writing any result (4.558 s,
266,563,584 B); `r2` and `r3` produced correct results but the curator refused
them because the runner's log capture had been misdirected, which is a launch
mistake rather than a measurement disagreement. `r4` is the green run and the only
curated one.

## Identity supervision in one pass

| Split | Images | Ancestries | Supervised queue tokens | EOS labels | Hold labels | Distinct classes |
|---|---:|---:|---:|---:|---:|---:|
| Training | 9 | 3 | 54 | 9 | 9 | 7 of 7 |
| Development | 18 | 6 | 90 | 15 | 12 | 7 of 7 (queue), 4 (hold) |

Queue token labels by class, and how many the model predicts exactly:

| Class | Training labels | Training correct | Development labels | Development correct |
|---|---:|---:|---:|---:|
| I | 12 | 0 | 15 | 0 |
| O | 9 | 0 | 12 | 0 |
| T | 15 | 0 | 9 | 0 |
| S | 3 | **3** | 18 | **9** |
| Z | 6 | 0 | 18 | 0 |
| J | 6 | 0 | 9 | 0 |
| L | 3 | 0 | 9 | 0 |

Hold identity is a separate head with its own classes: training supervises
J/L/S three times each with zero correct, development supervises J/L/S/T three
times each with **T correct 3/3** and the rest zero.

Slot distribution: training labels fill slots 0–5 exactly nine times each, so all
nine probes present a six-item visible queue with no unknown prefix; development
fills slots 0–8 (15, 15, 15, 12, 9, 9, 9, 3, 3), and its two non-visible queue
statuses are cropped/absent cases that carry no identity labels. Training's three
correct tokens all sit at slot 0, but all 15 development slot-0 tokens are wrong
and the nine development successes are spread across slots 1, 3, 5 and 7, so a
positional prior does not explain the successes.

## What the counts decide

- **Not a coverage gap.** Every piece class appears in both splits, and every
  observed queue slot in the training probes is supervised. Nothing is missing
  from the target vocabulary or left systematically masked.
- **Thin per probe, not thin in the stream.** The scored probes carry 3-18 unique
  labels per class and 9-12 hold labels, with a fivefold imbalance between T (15)
  and S or L (3), and hold identity sees four of seven classes. The training stream
  is not those probes: `training_spec` draws a new ancestry every three steps, so the
  factor-4 run's 128 updates saw 128 distinct images from 43 ancestries
  (`runs/m3-identity-factor-r1/training.jsonl`: 128 distinct `sample_id`, 43 distinct
  `group_id`, 128 distinct `image_pixels_sha256`). Training-wide identity supervision
  is therefore several hundred distinct labels, so per-class label diversity is not
  obviously the binding limit; the probe counts above are what the criterion scores.
- **Identity is class-selective, not uniformly unlearnable.** The only identities
  ever predicted correctly are queue `S` (3/3 training, 9/18 development) and hold
  `T` (0/3 training, 3/3 development). Six queue classes and three of four hold
  classes are at zero everywhere. With a matched comparison already showing that a
  fourfold loss scale moves the totals, the remaining ceiling is not global
  gradient starvation.
- **Consistent with a class prior, not a position prior.** Success attaches to a
  class rather than to a slot; the model appears to collapse onto one identity per
  head and to be scored correct where that class happens to occur.

## What this does and does not decide

It rules out two explanations and therefore two candidate levers: another loss
scale (already measured, structural cost) and more class coverage (already
complete). It does not decide between the two that remain by itself — optimization budget over
more distinct draws, versus the input representation (pixels per icon). The budget
lever is predeclared in `docs/M3_IDENTITY_BUDGET.md` and is the next thing this
project runs; the representation lever stays closed until that verdict.

The next identity step should therefore add supervision diversity or change the
representation, not another coefficient, and should be predeclared with its own
fresh native admission like every prior lever.

## Limits

Nine synthetic training views and eighteen development views drawn from three and
six ancestries; the development views share six ancestries, so the per-class counts are budget
counts rather than statistical estimates. No
real image, no decoded complete state, no release claim, and no calibrated
confidence follows. Masked identity stays unknown and is never counted as a class
or as an empty slot.
