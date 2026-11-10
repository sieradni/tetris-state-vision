# The capacity branch of the fit-budget rule: wider than 64 is not expressible

Status: **admissibility record, written before any launch and committed with no measurement receipt in
existence.** The fit-budget record ranked *"a wider shared trunk than capacity 64"*. This pass was to run it.
It cannot be run, and the reason is structural rather than budgetary: the quantity the rule points at is not a
width, and the model refuses any value above 64 by construction. No arm, sweep or probe was launched, so this
document carries no stage timings and claims no occupancy number.

## The suspect, as ranked

`scripts/placement_error.py::fit_budget_decomposition` returns `capacity_limit -> "a wider shared trunk than
capacity 64"` when the activation engaged and the gate stayed closed, which is exactly the settled fit-budget
result (activation 0.5481, gate 0 of 28). Taken literally it asks for a model with a trunk wider than the
settled capacity.

## Why it cannot be instantiated

1. **`component_capacity` is a slot count, not a width.** Building the same declared configuration at capacity
   32 and at capacity 64 gives **identical** parameter and tensor censuses - 8,018,505 parameters
   over 271 tensors in both cases - while switching the M3 heads on at all moves the M2 model
   from 7,694,010 parameters / 242 tensors to that same figure. Capacity
   therefore gates whether the component decoder exists, not how wide anything is; values 32 and 64 are
   parameter-identical, so "wider than 64" has no smaller sibling to extrapolate from either.
2. **64 is a hard construction bound.** `ModelConfig.__post_init__` refuses anything above it
   (`component capacity must be0 (M2) or1..64 (M3)`), and `scripts/train_m2_stream.py` refuses it earlier still
   with the reason attached: *joint stream component capacity must be10..64 to cover the renderer's10 slots*.
   The bound exists because a board holds ten tetromino slots, so 64 is already four times the population any
   board can contain; there is no occupancy label above it to widen towards.
3. **The shared trunk has no knob.** The encoder widths are fixed at `(48, 96, 192, 256)` with attention dimension
   `192` in `ModelConfig`, and the training driver exposes no flag for either. A wider trunk is
   therefore not a configuration change but a different architecture: every tensor in the trunk changes shape,
   so the campaign's matched-start contract - bitwise-identical state dictionaries, or a declared set of *added*
   tensors - cannot be satisfied by copying the published checkpoint into it. The published checkpoint cannot
   be continued into a wider model at all.

## What a real trunk-width lever would have to declare

It is a new architecture identity, not a continuation, and it needs a decision the campaign cannot make for
itself: a declared width knob (or a second `ModelConfig` identity), a matched-start policy for shape-changed
tensors instead of added ones, a fresh baseline trained at that identity rather than the settled control, a
measured per-step cost before any step budget is chosen (the 1,800 s wall already binds), and an explicit
statement that the resulting numbers are not comparable to the capacity-64 control through the existing
single-control reconciliation table.

## Verdict

**Blocked on a user decision.** The `capacity_limit` branch is not a measurable suspect under the frozen
architecture: capacity 64 is a slot bound, capacity 32 and 64 are parameter-identical, and the trunk width has
no expression. The rule now reports this rather than naming an impossible arm, and the campaign's next step is
whichever of these the user authorizes:

- **A declared architecture-width lever**, new identity and fresh baseline, with the per-step cost measured
  before the step budget is fixed; or
- **standing limit**, accepting that under the frozen architecture the fit gate is FALSE and the seven spent
  levers plus the budget lever have all been closed without opening it.

Until one is chosen, no further occupancy lever is ranked: every suspect the rule can name under this
architecture has now been spent or shown inexpressible. Limits unchanged: synthetic training fit, one seed,
committed lattice, no real-image, generalization or complete-state claim.
