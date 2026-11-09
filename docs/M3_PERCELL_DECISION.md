# M3 per-cell decision — occupied-cell class budget lever (predeclaration)

Status: measured (2026-11-09). The rule below was predeclared and committed at `0d77cef`,
before any per-cell-decision admission profile, study or probe receipt existed in the
tree; the measured result is the section at the end and does not revise the rule.

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

## Measured result: the lever STANDS, and the class budget trades misses for inventions

Receipts: admission `runs/m3-percell-balance-r1-profile-r1-job.json` (passed, 2 work items,
980,099,072 B peak), the matched study `runs/m3-percell-balance-r1-job.json` (passed, 128/128
updates, 459.2 s, 963,743,744 B peak under the declared 1,152 MiB ceiling,
`arguments.occupancy_factor: 4`, `arguments.occupancy_positive_weight: 4`), the independent
replay `runs/m3-percell-balance-r1-errors-r1-job.json` (passed, 27 forwards, 23.0 s, weights
`9ae1fef9f649...` unchanged, every saved metric reproduced), its curation
`runs/m3-percell-balance-r1-errors-r1-curation.json` (passed) into
`docs/evidence/m3-percell-balance-r1-errors-r1/scope.json`, and the instrument tests
`runs/m3-percell-balance-tests-r2.json` (8 modules).

**The lever STANDS on the predeclared rule.** The deciding quantity, occupied-cell F1 on the
seven nonempty training boards, is **0.3109** = `2*67/(2*67+196+101)`, below the threshold
0.70. Development is 0.5350 (275 of 2,800 occupied cells recovered, 352 false positives,
126 misses).

| quantity (independent replay) | fit-gate run, `occupancy-factor 4` | this run, `+ positive weight 4` |
|---|---|---|
| training nonempty occupied-cell F1 | 0.2605 | 0.3109 |
| training nonempty TP / FP / FN | 34 / 59 / 134 | 67 / 196 / 101 |
| development nonempty occupied-cell F1 | 0.6045 | 0.5350 |
| development nonempty TP / FP / FN | 230 / 130 / 171 | 275 / 352 / 126 |
| exact nonempty boards (training, development) | 0/7, 0/14 | 0/7, 0/14 |
| exact empty boards; empty-population FP | 0/2, 23; 0/4, 130 | 0/2, 72; 0/4, 135 |
| exact roles (active, ghost), training | 0/9, 0/9 | 0/9, 0/9 |
| exact roles (active, ghost), development | 0/18, 0/18 | 0/18, 0/18 |
| exact readable components (train, dev) | 0/9, 3/18 | 0/9, 0/18 |
| queue piece tokens (train, dev) | 15/54, 9/90 | 0/54, 0/90 |
| queue EOS (train, dev) | 0/9, 0/15 | 9/9, 15/15 |
| queue presence (train, dev) | 54/63, 90/105 | 54/63, 90/105 |
| queue slot status (train, dev) | 54/54, 90/90 | 54/54, 90/90 |
| hold piece (train, dev) | `T`x9 0/9, `T`x12 3/12 | `T`x9 0/9, `T`x12 3/12 |
| geometry mean cells (train, dev) | 3.3397, 2.6864 | 4.2468, 2.1467 |
| position logit spread (max, train / dev) | 0.0641, 0.0379 | 0.1003, 0.0954 |

**The mechanism is measured and it is a reallocation, not discrimination.** The class budget
moved the bound in the direction it was asked to: occupied-cell recall rose from 34/168 to
67/168 (0.202 to 0.399) while precision fell from 34/93 to 67/263 (0.366 to 0.255), and the
misses it removed came back as inventions (false negatives 134 -> 101, false positives 59 ->
196). F1 therefore rose by only 0.0504 absolute and the decision still separates the classes
no better than a fixed threshold shift would. The empty population makes the same point from
the other side: its false positives tripled from 23 to 72 on 400 known empty training cells
(130 -> 135 in development), so the change is a decision-boundary shift, not a representation
change. `decision_mechanism` labels this population `specificity-limited`, because on the
deciding split the inventions outnumber the misses (196 > 101); a tie between the two would
classify as `both-limited`. This is the same conclusion the fit gate reached from the other
direction: the per-cell decision, not the term's scale or its class split, is what pins fit.

Family instability, none of it a criterion and none of it fit: queue EOS recovered to exact
9/9 and 15/15 (it was 0/9 and 0/15 in the fit-gate run) while piece tokens collapsed from
15/54 and 9/90 to 0/54 and 0/90, hold stayed on one class, ghost exactness stayed 0/9 and
0/18, and the three development exact-readable cases disappeared (3/18 -> 0/18). The queue
read-out swings in both directions across runs of this family; presence and slot status are
the only queue fields unchanged at 54/63, 90/105 and 54/54, 90/90.

`scripts/percell_gate.py` holds the rule as code (`CELL_F1_THRESHOLD`, `occupied_cell_f1`,
`percell_gate_opens`, `decision_mechanism`) and `tests/test_percell_gate.py` reproduces both
the rule (an exact tie at 0.70 opens; a partial fit below it stands; exact-board counts can
neither open nor close the gate; a missing population is refused) and this verdict from the
committed bytes: F1 0.3109, standing gate, `specificity-limited`, 0/7 exact nonempty training
boards, `occupancy_positive_weight: 4`.

What this does not license: no real-image, complete-state, calibration or quality claim, no
re-opening of the closed identity or fit questions, and no further extension of the class
budget as a fit lever - at this budget it reallocates errors without separating the classes.
The ranked next instrument is a read-only **ranking** measurement on this committed
checkpoint: a predeclared threshold sweep of the locked occupancy logits, which separates
"the logits rank occupied cells above empty ones and only the operating point is wrong" from
"the logits do not separate the classes at all". If the ranking turns out to be informative,
fixing the operating point is a precision/recall budget choice that only the owner can make,
and this document names that choice rather than inventing one.
