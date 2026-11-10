# Lattice-agnostic occupancy read-out — a gated channel that cannot see the predicted lattice

Status: predeclared (2026-11-10) and committed before any profile, study, sweep, probe or evidence
directory for this instrument exists. The change, its exact parameter cost, the matched start, the
envelope, the population, the four conditions, the regression falsifiers, the control-drift rule, the
predeclaration-ancestry rule and the non-circular next-suspect rule below are fixed; the measured
section is appended only after the canonical receipts exist.

## Where this comes from

The per-stage lattice-residual pass was the fifth geometry-side lever and it lost both scores
([M3_OCCUPANCY_LATTICE_RESIDUAL.md](M3_OCCUPANCY_LATTICE_RESIDUAL.md), `not-cleared`: attribution
**-0.0115**, placement **-19.2%**). Its declared non-circular rule took its first branch - placement
improvement `-0.192 < 0.05` - and named **a lattice-agnostic occupancy read-out**. `translation`,
`scale`, `geometry_capacity`, `placement_capacity` and `lattice_residual` remain excluded as outputs.

Why this is the right next question: the published occupancy head reads the decoder latent evaluated at
**the stage's own predicted lattice** (`occupancy(cells)`, plus the optional detail tap, which also
samples through that lattice). Five levers moved or re-supervised that lattice and none improved the
score; the residual pass moved it a lot and the head answered with an almost all-positive regime. One
possible reading - not yet measured - is that the read-out's dependence on the predicted lattice is
itself the limiter: if a decision channel that cannot see the lattice scores better, the lattice was the
problem; if it does not, the lattice is not what limits the score. This pass measures that.

## The single declared change

**A gated, lattice-agnostic occupancy channel that can replace the lattice-specific read-out.**

For each stage and each cell, the lever computes

```
pooled = weighted mean over the encoder's context tokens (image-level, no lattice sampling)
free   = row_embedding(cell) + column_embedding(cell) + pooled          # stage-invariant
gate   = tanh(occupancy_lattice_gate)                                  # one scalar, init 0
logits = (1 - gate) * occupancy(cells) + gate * free_head(free)
```

| arm | `occupancy_lattice_free` | meaning |
|---|---|---|
| lever | `1` | the gated channel above is present |
| baseline | `0` | the published read-out, byte-identical to today |

- **What it replaces:** the occupancy logits themselves. As `gate` moves from `0` toward `1` the
  published lattice-sampled latent path is suppressed and the decision comes entirely from a channel
  that never samples the image through the predicted lattice, so it is **identical at every decoder
  stage**. The gate can also go negative (to `-1`), which extrapolates the published channel instead;
  that is a declared, measured outcome, not a hidden one.
- **What it subsumes:** at `|gate| = 1` it subsumes the published lattice-specific path *and* the
  declared detail tap, because both read through the predicted lattice. It does **not** remove those
  modules from the network - it is a learned gate - so "the read-out became lattice-agnostic" is a
  **measured outcome** (the reported gate magnitude), never an assumption.
- **Declared parameter cost:** `Linear(64, 64)` = 4,160, `Linear(64, 1)` = 65, and the gate scalar = 1,
  so **4,226 new parameters in 5 new tensors** - `occupancy_lattice_free.{0,2}.{weight,bias}` and
  `occupancy_lattice_gate` - **all exactly zero at initialization**. Memory is under 20 KB above the
  published model, so the declared ceiling is unchanged.
- **Why step zero is the published function:** zero projections make `free_head(free) = 0` exactly
  and a zero gate makes `gate = 0`, so `logits = occupancy(cells)` bitwise.
- **What is not changed:** no new input, no change to the encoder, the box head, the geometry stage
  policy, the objective or the population. The lever is declared on the published box head, objective
  and stage policy.

Everything else is the campaign's matched 256-step configuration: checkpoint `runs/m3-percell-balance-r1`
(`9ae1fef9f649...`), 256 continuation steps, `occupancy_detail 1`, stencil `0`, `occupancy_context 0`,
`geometry_final_only 0`, `lattice_support 0`, `box_extent_weight 0.0`, `support_anchor_weight 0.0`,
`lattice_residual 0`, factor `4`, positive weight `4`, `seed 20261026`, learning rate `.001`, curriculum
`stream-v1`, capture `512`, capacity `64`, positional identity read-out, length endpoint, 1,152 MiB,
1,800 s.

## Matched start: proven, not assumed

This lever **adds tensors**, so the shared contract uses its parameter-addition branch (V-071): every
published tensor must be bitwise identical between the arms, and the added tensors must be **exactly the
declared five** and **exactly zero** - anything else added, missing or non-zero refuses the curation.
`step_zero_identical` then compares the step-0 probe records with the objective's own value stripped
(V-072). The added modules are constructed **last** in the model so the published tensors draw the same
seeded tensors in both arms. A launch refused by the guard is preserved under `runs/attempts/` with a
note and re-recorded under a fresh number.

## Baseline control: the curated reference is a hard check

The baseline must reproduce the control curated by the five earlier passes:

| quantity | curated control |
|---|---|
| per-cell F1 at `0.0` | 0.4146485528647372 (351 TP / 675 FP / 316 FN) |
| pooled AUC | 0.8255386800822172 |
| best swept F1 / threshold | 0.4237196765498652 at `-0.30` |
| mean placement error | 1.9812198625 cells |
| extent ratios u / v | 1.2256880473012681 / 1.18035288054444 |
| per-board error spread | 4.604344325 cells |

Tolerance `1e-6`. Any disagreement is a **drift defect**: the curator refuses before computing a lever
attribution.

## Envelope and declared work arithmetic

Per arm: **1,152 MiB** aggregate commit, **256 steps**, **1,800 s** runner wall, one heavy job at a
time. An arm that cannot finish inside the envelope is **NOT ADMISSIBLE AS DECLARED**; nothing is
relaxed.

| launch | work items | what is counted |
|---|---|---|
| each profile | 2 | 2 optimizer updates |
| each study | **256** | 256 optimizer updates |
| each sweep profile | 2 | 2 forwards |
| each measured sweep | **54** | 36 deciding + 18 development forwards |
| each placement probe | **36** | 36 deciding forwards, 640 MiB / 300 s |
| curation | **8** | eight receipts read |

## Scoring population

Every known cell of every nonempty deciding board - 28 boards, 200 cells each, 5,600 known cells, 667
occupied and 4,933 empty, twelve ancestry groups - at threshold `0.0`, plus the standing 81-point grid
on `[-6, 6]` step 0.15. The placement instrument measures all 200 cells of the same boards. The oracle
pairing's co-scoreable subset is **not** a judgement population.

## The four declared conditions

**The lever CLEARS iff all four hold**, unchanged from the four preceding geometry-side passes.

1. **Per-cell gate** - occupied-cell F1 at `0.0` on the full population `>= 0.70`.
2. **Ranking gate** - `sweep_verdict` is `informative` (AUC `>= 0.90` and some swept point `>= 0.70`).
3. **Attribution floor** - best swept F1 exceeds the matched baseline's by `>= 0.05`, with the
   whole-ancestry-cluster paired bootstrap (400 draws, seed `20261026`) reported beside it.
4. **Placement condition** - mean placement error over 200 cells and two axes must fall by `>= 25%`
   relative to the matched baseline (1.9812198625 -> at most 1.485914896875 cells). A read-out change
   that wrecks the box has not improved the system, whatever the occupancy score does.

Otherwise the pass is **`not-cleared`** with the failing conditions named mechanically.

## The declared gate diagnostic

The trained lever checkpoint's own gate is reported as `tanh(raw)` and `|tanh(raw)|`. This is a
**diagnostic, not a condition**: it says whether the read-out actually moved toward the lattice-free
channel, and it is the only evidence that can support - or deny - the description "lattice-agnostic".

## Declared regression falsifiers

- **Gate shut** - `|tanh(gate)| < 0.05`: the lattice-free channel was not used, so no occupancy outcome
  may be described as lattice-agnostic whatever the score does.
- **Placement regressed** - placement improvement `< 0`: a read-out change moved the box the wrong way.
- **Operating point only** - attribution `>= 0.05` while condition 1 fails: the gain exists only as a
  threshold shift, not as a per-cell improvement, and it may be reported but not credited to the
  channel.
- **Occupancy moved, placement not** - attribution `>= 0.05` with placement `< 5%`: the box did not
  follow, so neither "the read-out limits the score" nor "the lattice does" can be concluded.

## Declared next-suspect rule (non-circular)

First match wins. No branch can return `translation`, `scale`, `geometry_capacity`, `placement_capacity`
or `lattice_residual` - those five levers are inputs, never outputs.

| rule | condition | next suspect |
|---|---|---|
| dispersion | the lever's per-board error spread `>= 1.5` cells | per-board lattice refinement capacity |
| scaling | otherwise | occupancy read-out at a second scale |

The rule is defined only on the measured spread, so a lever that moves neither placement nor occupancy
leaves both remaining suspects to be decided by that one number.

## Predeclaration ancestry

The curator resolves the commit that first added this file with `git log --diff-filter=A`, requires it to
be an ancestor of `HEAD`, and refuses unless **every** receipt it reads was written after that commit's
committer timestamp. Receipts are untracked artifacts, so time is the proof available; the rule is
stated here, implemented in the curator and exercised by a test.

## Reconciliation the receipt must carry

From the five earlier evidence files (reference only, never verdict input): geometry **+0.0781** at
19.9% placement cost, origin **+0.0102** at 11.0%, extent **+0.0318** at 67.5%, capacity **-0.0241** at
28.2%, residual **-0.0115** at 19.2%. The receipt states where this lever sits on that single control.

## Receipts this pass must produce

1. `runs/m3-occlatfree-{lever,baseline}-r1-profile-r1` (2 steps, 1,152 MiB / 300 s / 2 work items).
2. `runs/m3-occlatfree-{lever,baseline}-r1` (256 steps, 1,152 MiB / 1,800 s / 256 work items), each
   admitted only by its own profile.
3. `...-sweep-r1-profile-r1` (2 forwards) and `...-sweep-r1` (54 forwards) per arm, 640 MiB.
4. `...-probe-r1` per arm (36 forwards, 640 MiB / 300 s).
5. Curation `runs/m3-occlatfree-curation-r1-job.json` into
   `docs/evidence/m3-occupancy-lattice-free-readout-r1/scope.json`, re-deriving every published number
   and asserting each probe reproduces its own sweep's per-cell F1 (1e-6) and TP/FP/FN counts.
6. The owned family `m3-occlatfree-*` declared in `record_stream_run.OWNED_RUN_PATTERN`, bound by a test.

Any launch that dies, leaves its declared arithmetic or is superseded is preserved under
`runs/attempts/` with a written note and re-recorded under a fresh number; none is evidence.

## Limits, stated before the measurement

- Synthetic, training-fit population; nine of twelve groups never appear in the run's own ledger. No
  real-screenshot, generalization, calibration or complete-state claim follows.
- One budget point (256 steps), one checkpoint, one gate form, one seed, committed lattice only.
- The channel is lattice-free by construction, but the *model* is not: the shared trunk still reads the
  image through its sampling and the gate can leave the published path in place. A null result therefore
  cannot prove that the read-out is not the bottleneck, only that this declared channel at this budget
  did not clear the conditions.
- Placement is measured against the committed target lattice, so it says nothing about deployment
  situations where that lattice is unavailable.

---

# Measured (2026-11-10)

Both arms ran the declared 256 continuation steps from the published checkpoint `9ae1fef9f649...` under one
captured catalogue (`93742d5fb23f...`), inside the declared 1,152 MiB / 1,800 s envelope: lever
**1,082,392,576 B / 1057.0 s**, baseline
**1,025,159,168 B / 1025.0 s**. Admission profiles
passed at 85.5 s / 84.2 s; the sweeps (54 forwards, 640 MiB) took 67.3 s / 67.4 s and the placement
probes (36 forwards, 640 MiB / 300 s) 46.5 s each, peaking at 561,872,896 B
and 559,980,544 B. Curation `runs/m3-occlatfree-curation-r1-job.json` passed
in 5.4 s (peak 355,921,920 B) and published
`docs/evidence/m3-occupancy-lattice-free-readout-r1/scope.json`; every receipt it read postdates the
predeclaration commit `7b3ca9a`.

**Matched start.** The parameter-addition branch passed: the shared initial tensors are bitwise
identical (`de614227fb6f...`), the two saved states differ by exactly the five declared tensors
(`occupancy_lattice_free.0.weight`, `.0.bias`, `.2.weight`, `.2.bias`, `occupancy_lattice_gate`), all
exactly zero at initialization, and the step-0 probes agree after the objective's own value is stripped.
The baseline's weights (`d149351cbdfb...`) and the lever's (`747714d38f28...`) differ only through the 256 steps.

**Verdict: `not-cleared`, failing all four declared conditions.** Per-cell F1@0 **0.4284**
(371 TP / 694 FP /
296 FN) against **0.4146** (351 /
675 / 316); AUC **0.8401** against
**0.8255** (+0.0146); best swept F1 **0.4384** at
`1.65` against **0.4237** at `-0.3`, so
attribution **+0.0147**, under the declared **+0.05** floor, with the
twelve-cluster paired bootstrap median **+0.0150** and interval **-0.0836 to
+0.1284**. Placement improved **22.3%** (1.9812 ->
1.5393 cells) against the declared **25%** floor.

**The gate opened only a crack.** The trained scalar settled at raw **0.0744**, so
`|tanh(gate)| = 0.0742`: the predeclared gate-shut falsifier (`< 0.05`) does not hold,
but the channel carried only ~7% of the occupancy logits. The declared
lattice-agnostic read-out was therefore *barely* the read-out that produced this arm's scores; the
result is a weak activation of the declared change, and the declared limit - that a null result here
cannot prove the read-out is not the bottleneck - applies with full force because the gate never moved
toward the free channel. The weight was the same at initialization (exactly zero), so this is what 256
steps at the published objective's own supervision bought.

**What the lever did buy: placement and dispersion, at a small occupancy gain.** Every geometry-side
lever in this campaign bought its occupancy delta by spending placement; this one moved placement the
other way, by **22.3%**, while per-cell F1 rose **+0.0138** and AUC
rose **+0.0146**. Signed offsets moved toward zero on both axes - column
1.3830 -> 0.4926, row
-2.1914 -> -1.0181 - the row extent ratio
tightened 1.1804 -> 1.1748 while the column ratio widened 1.2257 ->
1.2786, and the per-board error spread fell
4.6043 -> **3.3544** cells with the shifted
share rising 0.5375 -> 0.5859 and the mapped share 0.6507 ->
0.6577. Smaller, more uniform board error is the first movement in this direction any
lever in the campaign has produced, and it is still short of the declared 25% floor.

**Why the occupancy score stalled.** The free channel reads the encoder's image-level context and fixed
cell coordinates, so it carries no per-stage lattice information at all; with the gate near zero the
lattice-sampled path still dominates, and the arm reproduces the baseline's regime - the head is
per-cell hesitant (371 of 667 occupied cells found, 296 missed) rather than the near-all-positive regime
the residual lever fell into. AUC 0.8401 is the highest any lever in the campaign reached, so
the ranking signal improved while the decision boundary did not move far enough to clear the gates. The
declared read-out change is therefore *not* shown to be the bottleneck this campaign can close: it was
barely activated, its small gain sits inside a bootstrap interval spanning zero, and neither the
per-cell nor the ranking condition was reached.

**Reconciliation.** Six levers now sit on the same curated control:

| lever | attribution | placement cost | verdict |
|---|---|---|---|
| geometry stage | +0.0781 | 19.9% | not-cleared |
| box origin | +0.0102 | 11.0% | not-cleared |
| box extent | +0.0318 | 67.5% | not-cleared |
| geometry capacity | -0.0241 | 28.2% | not-cleared |
| per-stage residual | -0.0115 | 19.2% | not-cleared |
| lattice-free read-out | +0.0147 | 22.3% gain | not-cleared |

Three supervision-side changes bought occupancy with placement, two capacity-side levers lost both, and
the read-out change bought placement without enough occupancy. No lever has cleared the four conditions,
and the fit gate remains FALSE.

**Ranked next.** The declared non-circular rule takes its **first** branch on the measured spread -
3.3544 `>= 1.5` cells - and names **per-board lattice refinement
capacity**. **Ranked next: predeclare a per-board lattice refinement capacity lever**, read with the
same four conditions; `scaling` (a read-out at a second scale) stays behind it, and `translation`,
`scale`, `geometry_capacity`, `placement_capacity` and `lattice_residual` cannot be returned by the rule.
The rule's condition is met on the lever's own dispersion, not on its improvement, so this name is not a
second attempt at the read-out by another route.

**Defects.** Two authoring defects in this pass's own files are logged as V-077 (the rule-insertion anchor
matched two functions with the same trailing lines, refused by the patch's uniqueness guard before it
applied anything) and V-078 (a new test compared the declared full-model parameter constant against a
tiny configuration, caught by the test itself). No run was refused, killed or superseded in this pass, so
`runs/attempts/` gained nothing.

**Unchanged.** No gate widened or retuned, no ceiling, population, label or prior-evidence change, no
frozen manifest touched. Limits are as declared: synthetic training-fit, one budget point, one
checkpoint, one gate form, one seed, committed lattice only, and the model is still lattice-sampled
underneath a barely-open gate.
