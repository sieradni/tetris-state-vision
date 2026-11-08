# M3 identity discrimination — the heads do not discriminate classes

Status: **measured**. On every closed M3 run the queue-token and hold-piece heads
emit one class (two in one run), so their exact counts are class frequencies in the
targets, not identity reading. Nothing here changes a recorded verdict; it changes
what those verdicts mean.

## Why this had to be measured

Every M3 lever was judged on *how often* queue tokens and hold pieces were exactly
right: 0/54, then 3/54, then 9/54, then 12/54; hold 0/12, then 3/12. No ledger entry
ever asked *which classes the heads emit*. The fit gate stayed FALSE through all of
them, and the last four passes each concluded that identity was "gradient-starved"
or "class-selective". A head that predicts a single class for every supervised
position produces exactly those totals whenever that class happens to appear in the
targets, so the totals alone cannot separate reading from reporting.

## Method (model-free, no forward pass)

`docs/experiments/audit_identity_discrimination.py` reads only committed bytes: the
four published per-case diagnostics and the frozen identity-exposure audit, whose
`published_result_sha256` binds the baseline it was derived from (verified before
anything is read). Unknown positions never enter a tally.

- **Hold piece.** The published `predicted_piece` of every supervised case is tallied
  by class. One predicted class across cases is a constant predictor; its exact count
  is then that class's frequency in the targets.
- **Queue tokens.** Published `error_indices` are exactly the supervised positions
  the head got wrong, so for each case the classes whose constant prediction
  reproduces that error set follow from the audited labels. A class that explains
  every case of a split is a proven constant predictor over that split; a case no
  class explains proves the head is not constant there.

## Measured

Training probes are 9 images / 54 supervised piece tokens and 9 hold pieces;
development is 18 / 90 and 12. Exposure class frequencies are
`I12 J6 L3 O9 S3 T15 Z6` (training) and `I15 J9 L9 O12 S18 T9 Z18` (development).

| Run | Split | Queue tokens | Provable constant class | Hold exact | Hold predicted classes |
|---|---|---:|---|---:|---|
| `m3-identity-factor-r1` (factor 4) | training | 3/54 | `S` (all 9 cases) | 0/9 | `T` ×9 |
| | development | 9/90 | none (3 cases unexplained) | 3/12 | `T` ×12 |
| `m3-identity-factor-representation-r1` (capture 640) | training | 3/54 | `S` (all 9 cases) | 0/9 | `T` ×9 |
| | development | 9/90 | none (3 cases unexplained) | 3/12 | `T` ×12 |
| `m3-identity-factor-budget-r1` (512 updates) | training | 12/54 | `I` (all 9 cases) | 0/9 | `I` ×7, `Z` ×2 |
| | development | 12/90 | none | 0/12 | `I` ×8, `Z` ×4 |
| `m3-stream-identity-shared-r1` (tied read-out) | training | 9/54 | none (6 cases unexplained) | 0/9 | `T` ×9 |
| | development | 6/90 | none | 3/12 | `T` ×12 |

The two exact equalities are the point. Training queue counts are exactly one class's
frequency — `S` is 3 of 54, `I` is 12 of 54 — and hold counts are exactly the
frequency of the class the head emits: `T` is 0 of 9 training and 3 of 12 development.
In those splits no labelled component was read; one class was reported.

## What this corrects

- The coefficient lever's declared identity positive (`0/54 -> 3/54`,
  `0/12 -> 3/12` development) is the head switching its single emitted class to `S`
  and `T`. `3` is the number of `S` tokens in training and `3` is the number of `T`
  hold targets in development.
- The budget lever's `3/54 -> 12/54` "identity rise" is the head switching to `I`;
  `12` is the number of `I` tokens in training.
- The tied read-out's `0/54 -> 9/54` is not constant on the training split, so that
  run does show some discrimination on 3 of 9 images; its hold count is still the
  constant-`T` frequency.
- The fit gate's identity term has therefore never been measured on a discriminating
  head. "Gradient starvation at the published scale" is not the right description:
  at these budgets the heads have not started to discriminate at all.

Recorded verdicts stand as written — each was judged against its predeclared rule
and every receipt is unchanged. The fit gate remains **FALSE** (0/9 and 0/18 exact
boards, 9/18 and 0/36 roles, 0/9 and 0/18 exact readable components), and the
roadmap's "executable identity levers exhausted" statement stays true, because a
lever that only rescales a constant read-out could not have produced anything else.

## Instrument (standing)

`scripts/m3_error_analysis.py` now reports discrimination alongside correctness, per
case and per split: `identity.queue_predicted_classes`,
`identity.queue_distinct_predicted_classes`, `identity.queue_constant_predictor`,
`identity.hold_predicted_piece`, and the aggregated
`queue`/`hold.predicted_classes`, `distinct_predicted_classes`,
`constant_predictor`. A constant head is visible in the same record as its exact
count instead of being inferable only from an audit. Four contract tests cover it
(in `tests/test_m3_error_analysis.py`): a constant token head flagged even when
structure is exact, a discriminating head not flagged, unknown positions excluded
from the tally, and a constant hold head collapsed to one class across cases.

## Receipts

- Audit `runs/m3-identity-discrimination-r2.json` (passed, 4 work items, 0.23 s,
  35.7 MB peak under the declared 128 MiB, supervisor `f5478185...`).
- Curation `runs/m3-identity-discrimination-r2-curation.json` (passed, 128 MiB),
  which re-derives every published count from the audit's own per-case rows and
  writes `docs/evidence/m3-identity-discrimination-r2/scope.json`.
- Instrument tests `runs/m3-identity-discrimination-tests-r1-job-r3.json`: 19 tests,
  `OK`, 1.34 s, 246,984,704 B under 512 MiB.
- Retained, nothing deleted: the first attempt wrote `runs/identity-discrimination-r1`
  and `runs/identity-discrimination-r1-job-r3.json` without its stdout/stderr logs, so
  the curator could not copy them; `runs/m3-identity-discrimination-r1.json` is the
  same situation. Both are kept beside the recorded attempt.

## Limits

- Queue constancy is proved by consistency with the saved error indices and the
  audited labels across a whole split; the frozen runs never saved a per-slot
  predicted token class, so the proof is indirect even though it is exact. The
  instrument now saves the classes directly for future probes.
- Development queue counts are *not* explainable by one constant (3 unexplained cases
  in both factor runs), so the development head does discriminate a little. The claim
  is about the training probes and the hold head, not about every position.
- 27 probes, one seed, synthetic `stream-v1` views. No real-image, complete-state,
  calibration or release claim follows.

## Next

The open identity question is now the read-out's **conditioning**, not its budget,
coefficient, resolution or tying: the separate hold head reads one role vector and has
never left a constant, while the queue-token head reads index/board queries and only
escapes constancy on some images. Any next identity lever must be able to produce
class discrimination by construction and must report these discrimination fields; a
lever that only changes which constant is emitted is not a lever. The real-image
chain (items 4–5) remains the owner decision and is unchanged.
