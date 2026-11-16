# M47 - the pixel-sole occupancy read: does the route or the evidence bind?

Status: **predeclared before any scoring run.** What feeds the head, the absence of a latent route, the expected
parameter delta, the protocol, the falsifier (derived from already-committed receipts) and every branch are below.
The measured section is added after the run and is not part of the declaration.

## What M46 left open, and why this is a routing test

M46 closed the additive raster family: three members cost `-0.021` to `-0.197` AUC for `-0.068` to `-0.232` F1 at
`0.0`, and the two steps that made the statistic more like the diagnosis asked for lost more, not less. The same
pass recorded the fact that makes this arm worth running: the **committed pixel-ceiling read-out scores `0.974763`
F1 at `0.0` and `0.999195` AUC on these same `2,986` cells from the same thirty statistics**, and the winner's own
read receives those statistics too - through a projection added to the decoder latent. So the question is not
whether the raster carries the evidence. It is whether the *route* lets the read use it.

This arm is therefore a restriction, not a feature. **The occupancy head's input becomes the raster projection
itself. No decoder-latent path reaches the decision.** If the pixel-sole read reproduces the ceiling read-out's
quality, the latent route was the problem. If it collapses to M46-like levels, the raster-to-decision path itself is
not the binding constraint and the successor has to be structural somewhere else - a different decision surface, a
different supervision, or a different geometry, not another raster statistic.

## The exact read that feeds the head

In the declared pixel-read block (`occupancy_pixel_query`, the winner's own form `3`), the thirty statistics are the
nine declared in-cell samples per channel at `+/-0.3` cell units plus three per-channel spreads, sampled at the
model's own predicted lattice for the stage - exactly the numbers the ceiling read-out measured. They pass through
the existing projection `occupancy_pixel_query` (30 -> 64 -> `dimension`), and with `--occupancy-pixel-sole 1` that
projection's output **is** the head's input:

```
occupancy_features = projected          # the sole read: assignment, not addition
occupancy_logits   = self.occupancy(occupancy_features)
```

- **No latent path:** the decoded cell features, the detail tap, the neighbourhood block, the column context and
  every additive raster term are excluded by construction - the assignment replaces the accumulated latent features
  rather than joining them. The arm's own test captures the occupancy head's input tensor and the projection's
  output in one forward pass and requires them to be `torch.equal`, and the same probe on the control shows the
  latent path present, so the test is not vacuous.
- **No new tensor:** the arm creates, initialises, zeroes and removes nothing. Its state dict has the same keys and
  the same shapes as the control's, so **the expected parameter delta is exactly `0`**.
- **Not the published function at step zero, by design.** The projection's last layer is zero-initialised in a fresh
  build, but the arm starts from the project's published source checkpoint (`runs/m3-percell-balance-r1`), which
  already carries a trained projection (30 -> 64 weights at norm `4.67`). The arm's first step is therefore the
  trained-projection read, not the winner's read; this is declared here so that the difference is expected rather
  than discovered, and the arm's tests pin non-inertness with a declared non-zero projection.

## Protocol, control and isolation

- **Control**: `m47-control-winner-config` - the winner's recorded configuration (`m30-level-calibrate-r5`:
  `--model accuracy`, `--component-capacity 64`, `--component-identity-readout positional`, `--component-endpoint-term
  length`, `--occupancy-factor 4`, `--occupancy-positive-weight 4`, `--occupancy-pixel-query 3`,
  `--occupancy-read-standardise 1`, `--curriculum stream-v1`, `--capture-max-side 512`, dimension `192`, blocks `6`),
  same seed, same `64` steps, same evaluator, same splits, same `2,986` fixed cells, same catalogue, under this code.
- **Reference**: the control must reproduce `m30-level-calibrate-r5`'s checkpoint `eddec1fd…` bit-exactly; five
  earlier controls of this configuration did (`m42`-`m46`). A miss makes the pass a defect, not an arm.
- **Arm**: `m47-pixel-sole`, differing from the control in exactly one recorded field and **no** parameter, with a
  distinct checkpoint and a test-pinned non-inertness.
- **One run.** If host memory refuses admission, the driver retries with fresh receipt names inside the pass, each
  refusal preserved; admission pressure never ends the pass.

## Falsifier, derived from already-committed receipts

The numbers this bar uses were committed before this arm existed: the winner's arm (`0.372881` F1 at `0.0`,
`0.954529` AUC, best swept `0.689775`, precision at `0.0` `0.230430`) and the ceiling read-out on the same cells
(`0.974763` F1 at `0.0`, `0.999195` AUC, best swept `0.987179`). No seed spread is committed for the winner, so this
pass treats deltas larger than `0.02` AUC as resolvable and claims nothing smaller.

- **HELPFUL** - F1 at `0.0` `>= 0.382881` (control `+ 0.010`) **and** AUC `>= 0.954529`: the route was the problem,
  the raster's evidence reaches the decision when nothing else dilutes it.
  - **DECISIVE CEILING** (the strongest form): F1 at `0.0` `>= 0.70`, the campaign's standing per-cell gate, cleared
    by a learned read for the first time.
- **NEUTRAL** - F1 at `0.0` in `[0.362881, 0.382881)` **and** AUC `>= 0.949529`: the restriction neither helps nor
  hurts at the declared resolution; the raster-to-decision path is viable but not sufficient.
- **HARMFUL** - F1 at `0.0` `<= 0.362881` **or** AUC `< 0.949529`: reading the raster alone is not enough at this
  budget, so the raster-to-decision path is not the binding constraint and no further raster routing is open.
- **DEFECT** (not an arm reading) - the control misses `eddec1fd…`, or the arm's checkpoint equals the control's, or
  the arm's logits are bit-identical to the control's on the fixed population.

Also reported, as in every arm: the complete TP/FN/FP/TN table and precision/recall at the fixed `0.0` crossing, the
closure decomposition against the `0.443205` span, the per-cell / ranking / attribution gates, and the row-wise
false-positive profile for rows 10-19 and 0-9 separately, reconciled to the control's own `140`.

**Prediction, declared before the run:** the thirty statistics carry the ordering (the ceiling read-out proves it on
these cells), but a 30 -> 64 -> 192 projection trained for `64` steps at the `4x` objective is not the ceiling
read-out: F1 at `0.0` `0.45-0.75`, AUC `0.96-0.99`, precision at `0.0` `0.35-0.65`, late rows 10-19 `20-90`. The
predicted reading is **HELPFUL, possibly decisive**.

## Pre-committed reading of every branch

- **HELPFUL** - the latent route is the binding constraint on the raster's evidence; the next arm keeps this routing
  and replaces the weak part (the 30 -> 64 -> 192 projection and its 64-step budget) with the ceiling read-out's own
  committed read as the decision surface, then asks how much of the ceiling survives the joint objective.
- **NEUTRAL** - routing moves the read materially but not enough: the successor must change the decision surface
  (a per-cell logistic read on the same statistics, committed rather than learned) rather than the route again.
- **HARMFUL** - recorded as the campaign's answer that no routing of these thirty statistics, trained under this
  protocol, reaches the winner's read: the next step is structural elsewhere (supervision over the decision surface,
  or the registered decision surface itself), and no further raster routing or raster term is declared.
- **DEFECT** - the cause is found, recorded in `docs/BUG_LOG.md` with its prevention test, and any re-run happens
  under a fresh name only after the repair.

No branch is re-run for a better number.

## Measured

Recorded `docs/evidence/m47-pixel-sole-r1/scope.json` and `diagnosis.json`, one seed, one run, `64` steps, after the
predeclaration committed before it.

| read | F1 at `0.0` | AUC | best swept |
|---|---|---|---|
| control = winner `m30-level-calibrate-r5` | `0.372881` | `0.954529` | `0.689775` |
| **M47 `m47-pixel-sole`** | **`0.133750`** | **`0.860374`** | **`0.486957`** (at `0.60`) |
| delta | **`-0.239131`** | **`-0.094155`** | **`-0.202818`** |

Gates per-cell / ranking / attribution: false / false / true (registration admitted the anchored vector the same way),
`cleared` false, `short_stage` `closure`. Split against the `0.443205` span: closure `-0.057017`, placement
`-0.001341`, representation `+0.501563` (fractions `-0.128648` / `-0.003025` / `+1.131673`, sum asserted `1.0`).

**Verdict: `harmful`.** Both harmful falsifiers fired: F1 at `0.0` is far below `0.362881` and AUC is far below
`0.949529`. The pixel-sole read is recorded and is not re-run.

### Complete contingency table at the fixed `0.0` crossing

| at `0.0` | control | M47 | change |
|---|---|---|---|
| true positives | `209` | `214` | `+5` |
| false negatives | `5` | `0` | `-5` |
| false positives | `698` | `2,772` | `+2,074` |
| true negatives | `2,074` | `0` | `-2,074` |
| precision | `0.230430` | `0.071668` | **`-0.158762`** |
| recall | `0.976636` | `1.000000` | `+0.023364` |

**Every one of the `2,772` empty cells is positive at `0.0`, and every margin is positive** (empty range
`0.150923` to `2.264305`, median `0.238912`; occupied range `0.178125` to `2.238789`, median `0.978908`). The read
never produces a negative margin on this population.

### The late-row mass, reconciled and per row

Baseline is the **control's `140` false positives in rows 10-19**; per row, before -> now:

| row | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| before | 0 | 18 | 45 | 107 | 58 | 119 | 64 | 39 | 81 | 27 | 0 | 0 | 0 | 0 | 1 | 111 | 2 | 26 | 0 | 0 |
| now | 44 | 59 | 71 | 109 | 122 | 120 | 131 | 128 | 131 | 172 | 173 | 173 | 173 | 172 | 172 | 173 | 173 | 172 | 173 | 131 |

Rows 10-19 went **`140` -> `1,685`** (`+1,545`) and rows 0-9 went `400` -> `1,087`; bands `rows 0-4` `405`, `5-9`
`682`, `10-14` `863`, `15-19` `822`. The per-column profile (`280`/`446`/`430`/`435`/`389`/`310`/`266`/`118`/`98`/`0`)
is the same shape every arm has shown, so the mass is not row- or column-specific: it is the level of the whole read.

### Mechanism, measured

- **Not a large bias.** The raster projection's last layer ends with bias norm `0.0564` (mean `1.04e-4`) against
  weight norm `1.6086`; the winner's own projection is `0.0550` / `1.2222`. So the projection did not buy its
  outcome with an offset - it is slightly *further* from the source, and per-column concentration is comparable
  (`0.0123`-`0.4632` here against `0.0225`-`0.3989` in the winner).
- **The empties are levelled above zero, not ordered badly by chance.** Empty margins span `+0.151` to `+2.264`
  while occupied margins span `+0.178` to `+2.239`: the two distributions overlap almost entirely in value and the
  read's own level sits above the fixed crossing, so at `0.0` precision is `214/2,986`.
- **The ordering is genuinely worse, which no threshold shift can explain.** AUC is threshold-free: `0.860374`
  against the winner's `0.954529`. An offset can move every cell across the boundary and leave AUC untouched; this
  arm's AUC fell `0.094155`, so the thirty statistics *as this head reads them* order cells worse than the winner's
  read does. Even its best swept threshold reaches only `0.486957` F1, `0.202818` under the winner's `0.689775`.
- **The split says where**: representation `+0.501563` of the span with the short stage `closure` - the loss is in
  what the read is given, not in its level.

### What the arm settles about routing

The declaration's matrix had two arms: reproducing the ceiling read-out means the latent route was the problem;
collapsing to M46-like levels means the raster-to-decision path is not the binding constraint. The measurement is
the second, and more sharply than M46: reading **only** the thirty statistics, with no latent path at all, loses
`0.094155` AUC and `0.239131` F1 at `0.0`, and cannot exceed `0.486957` even at its best threshold, while the
committed ceiling read-out reaches `0.974763` F1 at `0.0` on the same cells from the same thirty numbers.

So **the route was not the problem and the raster's evidence is not what the head is using**. The gap between the
ceiling read-out (`0.974763`) and every arm this campaign has trained on those same statistics is a gap in the
*read-out*: what the ceiling read computes from the thirty numbers is not what a 30 -> 64 -> 192 projection trained
for `64` steps at the declared `4x` objective learns to compute from them.

## Campaign-level answer

**(a) Can any variant of this family raise fixed-crossing precision against the winner control without a structural
change?** No, on the measurement. Four successive answers now: the additive raster family is closed (M44 `-0.068`,
M45 `-0.189`, M46 `-0.232` F1 at `0.0`), and the routing restriction that removed every latent path is worse still
(`-0.239`, AUC `-0.094`). No raster statistic, no gating of one, and no routing of one has raised precision over
the winner's `0.230430`; the best any of them managed was M44's `0.180484`, a loss.

**(b) What is genuinely open?** One question, precisely: *is the ceiling read-out's quality a property of the
statistics or of the sampling geometry they are taken at?* The ceiling read-out was measured on the fixed lattice;
the model's read samples the same statistics at its own **predicted** lattice, per stage. If a committed read-out
fitted on the model's predicted-lattice statistics keeps ceiling quality, then the head's training is what fails and
the successor is a frozen-trunk, head-only read. If it collapses, the predicted lattice's sampling is what destroys
the evidence and the successor is registration, not the read.

### Ranked remaining successors

| rank | next arm | headroom | cost | falsifier that retires it |
|---|---|---|---|---|
| 1 | **committed read-out on the predicted-lattice statistics** - the decisive test: fit the campaign's committed per-cell read on the thirty statistics *sampled where the model believes the cells are*, and score it on the same `2,986` cells | decisive either way: it separates the sampling geometry from the read-out in one measurement | one script, no training run, minutes | F1 at `0.0` `< 0.70` - placement is binding and the successor is a registration arm; `>= 0.70` - the read-out's training is binding and the successor is rank 2 |
| 2 | **frozen-trunk, head-only training**: the occupancy head trained alone from the committed read-out's initialisation, the trunk frozen | high, if rank 1 says the read-out is binding | one `64`-step run, no new parameters | AUC `< 0.954529` or F1 at `0.0` `< 0.382881`: isolating the head from the joint objective does not rescue the read |
| 3 | **registration-first arm**: the decision's statistics sampled at the support-derived lattice instead of the predicted one | moderate, if rank 1 says placement is binding | one `64`-step run, no new parameters | AUC stays under `0.93`: the sampling geometry is not the cause either, and the campaign's read is at its structural limit |
| 4 | **seed-spread replicate of the winner** (measurement, not an improvement) | bounds what any future delta means | three `64`-step runs | spread in AUC `> 0.02`: every sub-`0.02` delta in the record is unresolvable and the campaign reports intervals from then on |
| R | **retired, must not be re-run**: every additive raster term (M44, M45, M46) and its variants; the pixel-sole routing (M47); M42 support crossing; M43 column step; M16 board crossing; M14/M31 levels; M9-M13 objective level terms; M15/M17 monotone and skyline; M34-M41 geometry | closed by measurement | - | not offered |

**The cheapest decisive test is rank 1**: it needs no training run, reuses the committed read-out and the committed
receipts, and its answer tells the campaign which of rank 2 or rank 3 is the real successor.

## Receipts

`runs/m47-control-winner-config{,-profile}`, `runs/m47-pixel-sole{,-profile,-registration}`,
`runs/m47-pixel-sole-r1-job.json`, `docs/evidence/m47-pixel-sole-r1/{scope,diagnosis}.json`.
