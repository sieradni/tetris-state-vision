# M42: the support as a decision rule, with no sampled point moved

Status: **predeclared before any M42 scoring run.**

## What is left to test

M41 settled the sampling side: form `15` holds M35's features, head width, decision rule and every other
recorded setting fixed and samples the support lattice instead of the predicted one, and the isolation costs
`-0.126522` AUC and `-0.230048` best swept (`0.945596` / `0.570048` -> `0.819074` / `0.340000` at an identical
`8,025,482` parameters). Geometry, on the sampling side, has a measured negative role at both feature sets.

The one untested role is the **decision** side. Nothing in the campaign has ever let the raster's support
choose a board's crossing level while every sampled point stayed where the network put it. This arm does that
with exactly one varied fact against the winner.

## The mechanism, exactly

`--occupancy-support-crossing 1`, applied inside the head after the read's own level mechanisms:

1. the predicted lattice the read already samples gives the board box the network believes in, and
   `pixel_support_lattice` gives the box the raster's valid pixels occupy - both deterministic, both already
   used elsewhere in the campaign;
2. their vertical mismatch is expressed in the read's **own rows**:
   `offset = (predicted_top - support_top) / predicted_row_height`, clamped to the board's row count;
3. the board's own row profile is `profile[r] = mean logit over the board's columns at row r`, and its current
   crossing height is `height = #rows with profile > 0`;
4. the declared level is `profile` interpolated at `height + offset`, and one constant is subtracted from the
   board's logits - so the crossing of `0.0` sits where the support puts the top of the board.

Properties: **one constant per board**, so within-board ordering is untouched by construction; **no
parameter**, so the parameter count cannot change; **no sample moved**, so `sample_points` is bit-identical to
the winner's; nothing fitted, no population statistic, no threshold grid, no label. It reads the raster mask
and the read's own logits only.

## The isolation, asserted in code

Against the winner's recorded configuration (`occupancy_pixel_query` `3`, `occupancy_read_standardise` `1`,
`dimension` `192`, `blocks` `6`, form `3`, `8,023,562` parameters, every lever `0`):

- **one differing config field**: `occupancy_support_crossing` `1` against `0`;
- **identical parameter count** (`8,023,562`), asserted from both checkpoint manifests;
- **identical sampled points**, asserted from the forward pass;
- **a distinct checkpoint hash**, asserted against every other arm's manifest.

So this predeclared contrast cannot be explained by feature count, head width or the monotone decision rule:
those are the winner's own settings, unchanged. (The winner was chosen as the partner rather than M36 because
M36 differs from the winner in *two* recorded facts - its `60`-input feature set and its one-crossing rule -
and the question here is whether the support has any decision-side value at the configuration that is actually
the operating candidate.)

## Protocol

One profile, one `64`-step study, one `36`-board probe under the new `m42-support-crossing` owned family
(`1,152` MiB study, `640` MiB probe), scored by `scripts/m37_stencil_read.py --inputs 30 --features 30
--instrument m42-support-crossing-r1 --predeclaration docs/M42_SUPPORT_CROSSING.md` on the same 2,986 cells
and both fixed label vectors, with the winner re-scored in the same pass. No prior arm is re-scored and nothing
under `docs/evidence` for a prior arm is edited.

## Gate thresholds, unchanged

per-cell `0.70` on both fixed label vectors; ranking (AUC `>= 0.90` and best swept `>= 0.70`); attribution
over the unchanged control `>= 0.05`.

## The bar, fixed before scoring

New operating point only if, against the winner's re-scored numbers:

- F1 at `0.0` `>= 0.372881`, and
- best swept F1 `>= 0.689775`.

## Prediction and discriminating outcomes, committed in advance

**Prediction: useful but insufficient, most likely neutral.** The winner already calibrates each board's own
level (its standardised read is the read), so a coarse placement-derived offset is a weak second opinion on a
quantity the read already decides; I expect `|dAUC| <= 0.010` and `F1 at 0.0 <= 0.40`, with the bar missed.
Having written that, the numbers decide, and these branches are fixed now:

- **CLEARS:** `F1@0 >= 0.372881` **and** best swept `>= 0.689775`. Then decision-side geometry carries real
  value, and the M41-dated closure must be read as scoped to the *sampled* placement rather than to geometry
  as such.
- **NOT CLEARS - inert (a defect, not a result):** AUC **and** F1 at `0.0` identical to the winner's to the
  last digit. No decision rule can leave both unchanged unless it never acted, so this would be a wiring
  failure to repair and re-run, not a verdict.
- **NOT CLEARS - neutral:** `|dAUC| <= 0.005` **and** `|dF1@0| <= 0.010`. Decision-side geometry carries no
  usable level signal at this depth and budget.
- **NOT CLEARS - harmful:** `dAUC < -0.005` **or** `dF1@0 < -0.010`. The support's placement actively
  misplaces the board's level.
- **NOT CLEARS - useful but insufficient:** at least one of `dAUC > +0.005` or `dF1@0 > +0.010`, with at
  least one bar condition missed. Decision-side geometry has measurable value that is not enough to promote.

Whichever branch fires, the report carries the gate flags, the deltas against the winner, the split
closure / placement / representation with the sum asserted, and the diagnosis in the same terms as the last
four passes (false-positive count, median margin, weakest true positive, row bands, textured-empty behaviour),
so the verdict is attributable rather than asserted. Capacity is not available as an explanation and will not
be offered; the arm adds no parameter.

## Receipts

`runs/m42-support-crossing{,-profile,-registration}` and
`docs/evidence/m42-support-crossing-r1/{scope,diagnosis}.json`. Refused attempts keep their receipts; only
completed runs are quoted.
