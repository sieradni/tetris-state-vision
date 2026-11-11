# Is the deciding-population error bias or trajectory variance?

Status: **predeclaration.** Written before any receipt of this pass exists. No training, no weight change,
no architecture change, no new parameter and no gate claim is made by this document.

## The question this answers

Every occupancy lever of the campaign has been read as a pairwise contrast against a control, and the
schedule pass closed the last non-structural suspect: per-cell F1 at the standing `0.0` point moved
`+0.005041` against a `+0.05` floor while the best swept F1 rose `0.4307 -> 0.5173` and pooled AUC
`0.8457 -> 0.8864`. Twelve levers now agree on the shape *ordering moves, standing-point separation
does not*, and the trunk-width fork is priced but is the user's decision
([M3_TRUNK_WIDTH_PRICING.md](M3_TRUNK_WIDTH_PRICING.md)).

Before a new architecture identity is created, one decomposition is missing and is nearly free: is the
residual error **bias** (a representational ceiling the frozen architecture cannot pass) or **trajectory
variance** (the same network reaching different local solutions, so averaging and more/better
optimisation would buy separation without any architecture change)? The two answers support opposite
decisions, and both are compatible with every number recorded so far.

Two committed arms of the *identical* published architecture already exist and differ only in the
learning-rate schedule: `runs/m3-occsched-constant-r1` (AUC `0.8457`, F1@0 `0.4307`, best F1 `0.4307` at
`0.0`) and `runs/m3-occsched-cosine-r1` (AUC `0.8864`, F1@0 `0.4357`, best F1 `0.5173` at `1.95`). They
share one initial weight file, one seed, one catalogue and one data order, so their disagreement is a
measured *trajectory* difference. Averaging their occupancy logits is therefore a direct read of how much
of the residual is variance rather than bias.

## Declared measurement

- **Arms:** `m3-occsched-constant-r1` and `m3-occsched-cosine-r1`, each reloaded from its committed
  checkpoint with its recorded weights verified against `summary.json` before use.
- **Ensemble (one only, no tuning):** the unweighted arithmetic mean of the two arms' raw occupancy
  logits, per known cell. No temperature, no weighting, no threshold shifting beyond the standing grid.
- **Population:** exactly the standing deciding population - the 12 canonical training ancestry groups,
  36 cases, 28 nonempty boards, 667 occupied cells, 200 known cells per case - rebuilt through the same
  frozen replay the sweeps use, with the saved probe rasters and annotations re-verified.
- **Read:** the predeclared 81-point grid `[-6, 6]` step `0.15`, strict `logit > threshold`, pooled AUC
  (Mann-Whitney, ties half), swept precision/recall/F1, exact nonempty boards at the best point and at
  `0.0`. Every member arm is swept in the identical execution, so the ensemble is compared to its own
  arms and not to a number copied from another receipt.
- **Population match:** the two arms' saved probe recipes must be identical row for row; a difference
  refuses the pass rather than interpolating.

## Read rule, declared before the receipt

Let `best` be the ensemble's best swept F1, `auc` its pooled AUC, and `member_best` the larger of the two
arms' best swept F1 in this same execution. Thresholds are the standing `0.70` F1 and `0.90` AUC.

1. `ensemble_clears` iff `auc >= 0.90` **and** `best >= 0.70`. The ensemble would meet the occupancy bars
   on the deciding population.
2. `variance_limited` iff `auc >= 0.90` **and** `best - member_best >= 0.05`. Ordering is sufficient and
   averaging buys at least the campaign's own attribution floor: trajectory variance, not representation,
   is a live lever.
3. `bias_limited` otherwise. Averaging the two trajectories does not buy the attribution floor, so the
   residual separation is a representational ceiling shared by both solutions, and the width fork is the
   only remaining suspect.

The rule has no fourth branch. `ensemble_clears` is a claim about these 28 synthetic boards only; it is
not the fit gate, not a release claim and not a licence to skip the fork.

## Falsifiers

- Ensemble `best` below both member arms' best: averaging is destructive, nothing follows about variance;
  recorded as `bias_limited` with the regression named, never smoothed.
- Ensembling raising `best` while AUC stays under `0.90`: ordering still binds, so the branch is
  `bias_limited` and the F1 gain is recorded as an operating-point effect.
- Any population, recipe, raster, weight or catalogue mismatch: the pass refuses.

## Envelope and identity

Read-only, both checkpoints resident, 640 MiB and 300 s, **80 forwards** (36 cases x 2 arms plus a
2-case-per-arm admission profile), one bounded job with a receipt, plus the profile job that must pass
first. The two arms share `catalogue_sha256` (source `9ae1fef9f649...`), so one frozen archive serves both
replays; the read instrument is `scripts/occupancy_sweep.py` (the same module both sweeps used) and the
model source is the frozen archive, verified before any forward. Weights must be bitwise unchanged after
the pass.

## Limits, declared now

Synthetic training population, one seed, one checkpoint pair, committed lattice, no development-population
claim, no real-image claim and no complete-state claim. The pair shares its initialisation and data order,
so any variance measured here is a **lower bound** on independent-seed variance and an ensemble of these
two is weaker than an ensemble of independently initialised runs. Nothing about the fit gate changes
either way.

## Result (2026-11-11): `bias_limited` - combining two trajectories does not buy the floor

Read on the declared population - **12 groups, 36 training cases, 28 nonempty
boards, 667 occupied cells** - through the standing 81-point grid, 108 forwards, one bounded job
(peak **635,166,720 B** of 640 MiB). The members reproduce their own sweeps to the digit (constant AUC
**0.8457**, F1@0 **0.4307**; cosine **0.8864**, **0.4357**), so this pass joins the
receipts instead of restating them.

| read | constant | cosine | ensemble (mean logit) |
|---|---:|---:|---:|
| pooled AUC | 0.8457 | 0.8864 | **0.8872** |
| per-cell F1 at `0.0` | 0.4307 | 0.4357 | **0.4617** |
| best swept F1 (threshold) | 0.4307 (0.0) | 0.5173 (1.95) | 0.5074 (1.05) |
| exact nonempty boards at best | 0 | 1 | 0 |

**Rule branch 3: `bias_limited`.** The ensemble AUC **0.8872** is short of the 0.90 bar, and its best swept
F1 **0.5074** is **-0.0099** against the better member's own 0.5173 - averaging is
*slightly destructive at the optimum*, the declared regression falsifier, and it is named rather than smoothed.
Standing-point F1 moved +0.0259 (0.4307 -> 0.4617), well under the +0.05 floor.

The decomposition is unambiguous. At `0.0` the two trajectories disagree on **169** of 667
occupied cells (**462** both right, **21** control-only,
**148** lever-only, **36** both wrong), so there is plenty of trajectory
variance to average - and averaging still does not order better at the optimum. The residual is therefore a
representational ceiling **shared** by both solutions, not sampling variance that an ensemble, a second seed or
more optimisation would remove. Descriptive only, not read: the ensemble's development AUC is
0.9672 with best F1 0.7716.

Consequence for the fork: the cheap alternative to a wider trunk - average more trajectories and keep the
architecture - is measured out **at this pair**, and the trunk-width option remains the only ranked suspect. This
does not make the gate reachable and does not predict that a wider trunk passes it; it removes one way of avoiding
the question.

Limits: synthetic training population, one seed, one checkpoint pair that **shares its initialisation and data
order**, so the variance measured here is a lower bound on independent-seed variance and this ensemble is stronger
than one built from independently initialised runs. No development-population verdict, no real-image claim, no
complete-state claim.

## Instrument defect this pass exposed, and the guard (V-087)

The first execution ran on the **recorded narrow population** - 3 groups, 9 cases, 7 nonempty boards - because the
script defaulted to the run's own group count while this declaration names the canonical twelve. Nothing is
published from it: it is preserved with its receipts as `docs/evidence/m3-deciding-headroom-narrow-r1/`. A second
declared-population run was then made with the pre-guard script and is preserved the same way as
`docs/evidence/m3-deciding-headroom-preguard-r1/`, so the read above is bound to the guarded instrument:
`--training-groups` is now a **required** input that cannot default to a narrower population, and the predeclared
group count is asserted against the admission profile and recorded in the scope. The predeclared population is additionally asserted against the admission
profile and recorded in the scope.
