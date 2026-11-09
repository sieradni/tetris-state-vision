# M3 per-cell decision — occupied-cell class budget lever (predeclaration)

Status: predeclared. This document was committed before any per-cell-decision admission
profile, study or probe receipt existed in the tree.

## The question

The fit gate is measured and closed as a standing negative: at `--occupancy-factor 4` the
locked read-out stops being constant-empty (34 of 1,400 training occupied cells recovered,
230 of 2,800 development) and still reproduces no board, because its errors are per-cell and
bidirectional (59 false positives and 134 misses on 1,400 training cells). The ranked next
lever is therefore the **per-cell decision itself**, not another scale on the term.

The question this pass asks is the class-imbalance question and nothing else: with the
occupancy term's total weight held at its already-measured `4.` and every other ingredient
matched, does a declared **class budget on occupied cells inside the balanced per-cell
term** make the locked decision discriminate between occupied and empty cells - and if not,
which error mass pins it.

Baseline from the fit-gate replay (`m3-fit-occupancy-r1`), computed from the receipt's own
counts as `2TP/(2TP+FP+FN)`:

| split | TP | FP | FN | occupied-cell F1 |
|---|---|---|---|---|
| training nonempty | 34 | 59 | 134 | 0.2605 |
| development nonempty | 230 | 130 | 171 | 0.6045 |

## The lever

`--occupancy-positive-weight 4`: only the occupied-cell mean inside
`balanced_occupancy` is multiplied by four, and the per-image divisor grows with it
(`(w*positive + negative)/(w*(positive>0) + (negative>0))`). The term's total magnitude is
unchanged, so this is a class-balance change and **not** a re-run of the closed
`--occupancy-factor` scale lever: `--occupancy-factor 4` is matched, unchanged, from
`m3-fit-occupancy-r1`. At `w = 1` the computation is bit-identical to the published
objective.

Nothing else belongs to the per-cell question and nothing else changes: capacity 64,
component identity factor 4, capture side 512, seed `20261026`, learning rate `.001`, 128
updates, evaluation/checkpoint 64, six development groups, positional read-out with the
four-query hold bank, length endpoint objective, and every other loss weight. The
`layer_cells` auxiliary term keeps `w = 1`; it is a different head.

## Predeclared rule — one artifact, one split, one quantity, one threshold

- **Artifact**: `docs/evidence/m3-percell-balance-r1-errors-r1/result.json`, written by the
  independent replay of the new run at its final checkpoint, the counts under
  `totals.training.locked_populations.nonempty`.
- **Split**: training only (nine images, three ancestries; seven have nonempty locked
  boards). Development is a diagnostic.
- **Quantity**: occupied-cell F1 on those seven images, `2TP/(2TP+FP+FN)` over known
  cells - a class-imbalance-aware, cell-wise discrimination score, not an exact-match
  count. A constant-empty read-out scores 0; a perfect decision scores 1.
- **Threshold**: **0.70**.

**The lever OPENS iff that F1 is >= 0.70.** Otherwise it **STANDS (FALSE)**, whatever else
moves. An exact tie at 0.70 opens. A movement below 0.70 is recorded as a partial fit and
still STANDS: the gate is not "did anything improve", it is "does the decision
discriminate".

Exact-board counts, exact empty boards, development, active/ghost role exactness, ghost
counts, geometry, exact readable components, queue tokens, EOS, presence, slot status and
hold identity are all **costs**: reported in full and inert for the verdict. No exact-match
count is a criterion here, alone or together with another; the criterion is the per-cell
score above. In particular a run that fits no board still opens if its cells discriminate,
and a run that fits boards by predicting empty everywhere (which cannot reach 0.70) still
stands.

The run is a matched single-seed diagnostic at a fixed budget on synthetic training views:
opening the gate would say the locked read-out discriminates occupied from empty cells on
nine synthetic training views at this budget and nothing more - no real-image,
complete-state, calibration or quality claim, and no re-opening of the closed identity or
fit questions.

## Receipts this pass must produce

1. Admission, 2 steps, `--profile-square`, bounded at 1,152 MiB / 700 s / 2 work items,
   receipt `runs/m3-percell-balance-r1-profile-r1-job.json`, captured stdout/stderr at
   `runs/m3-percell-balance-r1-profile-r1.log` / `.stderr.log`.
2. The matched study, admitted only by that profile, bounded at 1,152 MiB / 1,900 s /
   4,096 work items, receipt `runs/m3-percell-balance-r1-job.json`, logs
   `runs/m3-percell-balance-r1.log` / `.stderr.log`.
3. Independent replay and curation at 640 MiB / 600 s / 27 forwards and 128 MiB, with the
   canonical names the recorder reads: `runs/m3-percell-balance-r1-errors-r1-job.json`,
   `...-errors-r1.log`, `...-errors-r1.stderr.log`, `...-errors-r1-curation.json`.
4. Instrument tests, canonical receipt `runs/m3-percell-balance-tests-r1.json` (re-run
   under a fresh number if a cap kills it).

The per-cell lever needs its own owned name family, so `m3-percell-*` is declared in
`record_stream_run.OWNED_RUN_PATTERN` and bound by a test rather than left to a hand-edited
call.

## Resource plan and discipline

The matched fit run peaked 990,515,200 B with the same model and input shape; this lever is
a scalar on an existing term, so the expected peak is unchanged. Declared ceiling 1,152 MiB
under the engine's 2 GiB bounded-job cap. One heavy job at a time, absolute interpreter
paths (V-030), canonical receipts (V-047), killed stages preserved and re-run to a passing
canonical receipt (V-050), objective provenance bound by `arguments` (V-051) and declared
once (V-052), and the curator's sidecar names read from a single declaration (V-053). Any
defect this pass exposes is fixed at its root with a regression test and recorded in
`docs/BUG_LOG.md`.

## Measured result

(Measured below this predeclaration, in a later commit; the rule above is not revised by
the result.)
