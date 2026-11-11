# Why the exactness gate reads zero at 400 steps

Population: 54 cases — training 28 nonempty + 8 empty, development 14 nonempty + 4 empty. Every case is assigned exactly one mode; the mode counts sum to the split's full case list.

## Classification at logit 0.0 (the gate's operating point)

| mode | training | development | all 54 |
|---|---:|---:|---:|
| spurious cells only | 22 | 6 | 28 |
| mixed cell errors | 5 | 8 | 13 |
| empty confusion | 8 | 4 | 12 |
| whole board shift | 1 | 0 | 1 |
| **sum** | **36** | **18** | **54** |

At each split's best swept threshold the modes move but exactness does not:
training {"empty_confusion": 6, "spurious_cells_only": 7, "mixed_cell_errors": 20, "whole_board_shift": 1, "exact": 2}, development {"empty_confusion": 4, "whole_board_shift": 1, "mixed_cell_errors": 12, "spurious_cells_only": 1}.

## Dominant mode

**Spurious cells only: 28 of 54 cases (52%)**, with zero missed occupied cells. The model is not misaligned and not under-filling; it over-fills. Raw counts at 0.0 are train FP 1018 / FN 5 and development FP 460 / FN 15, i.e. ~28 and ~26 wrong cells per board that are almost entirely extra fills.

## Per-case error size and offset directionality

Error-size buckets (fp+fn) at 0.0 — training: {"38+0": 3, "3+0": 2, "6+0": 2, "24+0": 2, "35+0": 2, "7+0": 2}; development: {"6+0": 2, "33+3": 2, "19+0": 2, "15+1": 1, "64+0": 1, "46+0": 1}.

training: mean signed offset (column -0.2749, row +0.6538) cells, magnitude 0.8930, spread 0.2661, 18/36 cases sharing the mean sign.
development: mean signed offset (column -0.0810, row +1.1386) cells, magnitude 1.1188, spread 0.3735, 11/18 cases sharing the mean sign.

Only one case in fifty-four is a pure whole-board shift and the offsets are small and incoherent (about half the cases point each way, spread 0.27–0.37 cells), so this is **not** an alignment or decode bug. It is a confidence/decision failure at logit 0.0.

## The fix, scored on the same frozen predictions

Target-free rule: remove occupied cells with no occupied 4-neighbour, and fill empty cells with at least three occupied 4-neighbours (tetrominoes are connected fours). Nothing reads the target.

| split | stage | exact nonempty+empty | F1 | FP | FN | wrong cells/board |
|---|---|---|---:|---:|---:|---:|
| training | logit 0.0 · before | 0+0 | 0.5641 | 1018 | 5 | 28.42 |
|  | logit 0.0 · after_morphological | 0+2 | 0.5668 | 1007 | 5 | 28.11 |
|  | logit 0.0 · after_snap | 0+0 | 0.5599 | 1023 | 10 | 28.69 |
| training | best 2.7 · before | 0+2 | 0.6587 | 561 | 64 | 17.36 |
|  | best 2.7 · after_morphological | 0+4 | 0.6652 | 579 | 46 | 17.36 |
|  | best 2.7 · after_snap | 0+2 | 0.6532 | 566 | 69 | 17.64 |
| development | logit 0.0 · before | 0+0 | 0.6191 | 460 | 15 | 26.39 |
|  | logit 0.0 · after_morphological | 0+0 | 0.6236 | 451 | 15 | 25.89 |
|  | logit 0.0 · after_snap | 0+0 | 0.6191 | 460 | 15 | 26.39 |
| development | best 3.0 · before | 0+0 | 0.7317 | 178 | 67 | 13.61 |
|  | best 3.0 · after_morphological | 0+4 | 0.7420 | 172 | 63 | 13.06 |
|  | best 3.0 · after_snap | 0+0 | 0.7251 | 181 | 70 | 13.94 |

## Verdict

The repair is a real but bounded win: it recovers empty boards (training 0+0 → 0+2 at 0.0 and 0+2 → 0+4 at the best threshold; development 0+0 → 0+4 at the best threshold), lifts F1 by up to 0.0103 and cuts wrong cells per board by up to 0.55. **It never makes a nonempty board exact.** Base-row snapping does not help and slightly hurts, which independently rules out a floor/anchor decode bug.

So no cheap post-hoc fix moves nonempty exactness off zero. The dominant failure is confident over-filling (28/54 cases at 0.0 with zero misses), which is a learned-logit property, not a decode artifact: the fix has to be training-side, which is what the 20,000-step fixed-pool chain is testing.

Instruments: `runs/dump_predictions.py` (frozen-weight re-inference), `runs/exactness_diagnosis.py` (classification), `runs/exactness_decode.py` (the repair and its score). All figures above are recomputed from `predictions-400.json`; no number is hand-entered.
